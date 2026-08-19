"""
6번: 삽입 불량 판단 - 정상/덜박힘/틀어짐 3분류
- 틀어짐(각도): detection_core.detect_screw_heads_by_color가 이미 판정 (여기서 재사용)
- 틀어짐(중심좌표): 나사머리 중심 vs 매칭되는 stud_hole 중심 거리. 임계값은 통계 추정이 아니라
  기하학적 제약(스터드홀 반지름 - 나사머리 반지름) - 이 값을 넘으면 나사가 물리적으로 구멍
  안에 있을 수 없으므로 원리적으로 타당한 임계값.
- 덜박힘: 나사머리 depth를 "바로 주변 국소 패널 표면 depth(고리 영역에 평면 피팅)"와 비교.
  전체 패널 평균을 안 쓰는 이유: 3번 과제에서 확인된 리브/타일 구조 때문에 위치마다 표면
  높이가 원래 들쭉날쭉해서, 전역 평균 기준이면 구조 자체 굴곡을 불량으로 오판할 수 있음.
  주의: 지금 가진 시편은 전부 정상 삽입(추정)이라 "진짜 덜박힘" 양성 샘플로 절대 임계값을
  검증하진 못함 - 정상 판정된 나사들의 protrusion_mm을 실행할 때마다 파일에 누적해 통계적
  이상치(평균+3표준편차)를 잠정 기준으로 쓰고, 표본이 부족하면 덜박힘 판정 자체를 보류한다.
- 검출/mm환산/평면보정 로직은 scripts/detection_core.py로 분리(5/7번과 공용)
"""

import os
import sys
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_preprocessing import build_pipeline, build_filters, capture_averaged_depth  # noqa: E402
import detection_core as dc  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEARCH_RADIUS_PX = 45   # 나사 주변 이 범위 안에서 평면 피팅용 샘플 수집
EXCLUDE_RADIUS_PX = 15  # 나사 바로 인접부(경계 흐림 영향권)는 탐색에서 제외
MIN_PLANE_SAMPLES = 30

# 덜박힘 임계값: 진짜 양성 샘플이 없어 절대치를 못 정함 - 정상 나사들의 protrusion_mm 분포를
# 실행마다 파일에 누적하고, 표본이 쌓이면 평균+3표준편차를 잠정 기준으로 사용.
BASELINE_PATH = os.path.join(RESULTS_DIR, "6_protrusion_baseline.json")
MIN_BASELINE_N = 5
BASELINE_STD_MULT = 3

# 틀어짐(중심좌표) 임계값: 통계 추정이 아니라 기하학적 제약 - 스터드홀이 나사머리보다 커서
# 생기는 "허용 편심 반경" = (stud_hole_mm - screw_head_mm) / 2. 이보다 중심이 어긋나면
# 나사가 물리적으로 구멍 벽에 닿아있다는 뜻이라, 캘리브레이션 없이도 타당한 임계값이 됨.
CENTER_OFFSET_THRESHOLD_MM = (
    dc.GROUND_TRUTH_MM["stud_hole"] - dc.GROUND_TRUTH_MM["screw_head"]
) / 2

# 실측 확인 결과: 제대로 삽입된 나사는 자기 구멍을 거의 다 가려서 그 구멍이 빈 stud_hole로
# 따로 검출되지 않음 -> match_nearest_stud_hole이 몇 칸 떨어진 "다른" 빈 구멍에 억지로
# 매칭되어 26~33mm짜리 가짜 오차를 만드는 걸 실캡처로 확인함(정상 나사가 전부 틀어짐으로
# 오판). 나사가 진짜 자기 구멍 안에 있다면 매칭 오차는 물리적으로 CENTER_OFFSET_THRESHOLD_MM을
# 크게 못 넘으므로, 그보다 훨씬 먼 매칭은 "다른 구멍에 잘못 매칭됨"으로 보고 판정을 보류한다.
MAX_PLAUSIBLE_MATCH_MM = CENTER_OFFSET_THRESHOLD_MM + 3


def local_panel_depth_mm(mask_bool, depth_mm, center_xy,
                          search_radius=SEARCH_RADIUS_PX, exclude_radius=EXCLUDE_RADIUS_PX,
                          debug_label=None):
    """나사 주변 고리 영역에 평면(z = a*x + b*y + c)을 피팅해서, 그 평면이 나사 정중앙 위치에서
    예측하는 depth를 국소 패널 표면 기준으로 사용.

    단순 평균/최소분산 패치 방식은 둘 다 이 시편이 카메라에 대해 기울어져 있어서(3번 과제에서도
    확인된 문제) 나사에서 조금만 떨어져도 경사 때문에 depth가 달라지는 걸 "튀어나옴"으로 오판했음
    (실측으로 확인 - 두 방식이 똑같이 8mm를 내놓아서 원인이 아니라 탐지법 공통의 결함임을 알아챔).
    평면 피팅은 이 경사 자체를 모델링해서 나사 위치의 "경사만 반영한 기대값"과 실측값을 비교하므로
    경사와 진짜 돌출을 분리해낼 수 있음."""
    h, w = depth_mm.shape
    cx, cy = center_xy

    y_lo, y_hi = max(0, int(cy - search_radius)), min(h, int(cy + search_radius))
    x_lo, x_hi = max(0, int(cx - search_radius)), min(w, int(cx + search_radius))
    ys, xs = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    patch = depth_mm[y_lo:y_hi, x_lo:x_hi]

    dist = np.hypot(xs - cx, ys - cy)
    valid = (patch > 0) & (dist >= exclude_radius) & (dist <= search_radius)
    if valid.sum() < MIN_PLANE_SAMPLES:
        if debug_label:
            print(f"  [{debug_label}] 제외: 평면 피팅 샘플 부족 ({valid.sum()}/{MIN_PLANE_SAMPLES}) "
                  f"- 나사 주변 depth 홀이 큰 것으로 추정")
        return None

    X, Y, Z = xs[valid].astype(np.float64), ys[valid].astype(np.float64), patch[valid].astype(np.float64)
    A = np.column_stack([X, Y, np.ones_like(X)])
    coeffs, *_ = np.linalg.lstsq(A, Z, rcond=None)
    a, b, c = coeffs
    return float(a * cx + b * cy + c)


def match_nearest_stud_hole(screw_center_px, stud_holes):
    """나사머리 중심에 가장 가까운 stud_hole을 짝짓기. (dist_px가 None이면 매칭 대상 없음)"""
    if not stud_holes:
        return None, None
    best = min(stud_holes, key=lambda s: np.hypot(
        s["center_px"][0] - screw_center_px[0], s["center_px"][1] - screw_center_px[1]))
    dist_px = float(np.hypot(best["center_px"][0] - screw_center_px[0],
                              best["center_px"][1] - screw_center_px[1]))
    return best, dist_px


def load_baseline():
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_baseline(values):
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(values, f, ensure_ascii=False, indent=2)


def classify_insertion(screw_dets, stud_holes, depth_mm, fx, baseline):
    """screw_dets(색상검출) + stud_holes(YOLO검출) + baseline(정상 protrusion 표본) ->
    (results, baseline) - 5/6/7번 공용 삽입판정 코어. baseline은 정상 판정분이 append된
    새 리스트로 반환(파일 저장은 호출자 책임 - 실시간 루프는 매 사이클 저장하지 않아도 되므로)."""
    protrusion_threshold = None
    if len(baseline) >= MIN_BASELINE_N:
        protrusion_threshold = float(np.mean(baseline) + BASELINE_STD_MULT * np.std(baseline))

    results = []
    for idx, det in enumerate(screw_dets):
        label = f"screw#{idx}"
        mask = det["mask"]
        inst = dc.build_instance(mask, "screw_head", 1.0, depth_mm, fx, debug_label=label,
                                  aspect_ratio=det["aspect_ratio"], tilt_status=det["status"])
        if inst is None:
            continue

        local_depth = local_panel_depth_mm(mask, depth_mm, inst["center_px"], debug_label=label)
        if local_depth is None:
            continue
        protrusion_mm = local_depth - inst["depth_mm"]  # 양수 = 나사가 패널보다 카메라 쪽으로 튀어나옴

        inst["local_panel_depth_mm"] = round(local_depth, 1)
        inst["protrusion_mm"] = round(protrusion_mm, 2)

        matched_hole, dist_px = match_nearest_stud_hole(inst["center_px"], stud_holes)
        center_offset_mm = None
        match_rejected = False
        if dist_px is not None:
            candidate_offset_mm = dist_px * inst["depth_mm"] / fx
            if candidate_offset_mm <= MAX_PLAUSIBLE_MATCH_MM:
                center_offset_mm = candidate_offset_mm
                inst["center_offset_mm"] = round(center_offset_mm, 2)
            else:
                # 가장 가까운 구멍도 너무 멀다 - 자기 구멍은 나사에 가려 안 보이는 것으로
                # 추정, 판정 근거 없음(오탐 방지 위해 틀어짐 판정 안 함)
                match_rejected = True
                inst["center_offset_mm"] = None

        statuses = []
        reasons = []
        if inst["tilt_status"] == "틀어짐":
            statuses.append("틀어짐")
            reasons.append("각도")
        if center_offset_mm is not None and center_offset_mm > CENTER_OFFSET_THRESHOLD_MM:
            if "틀어짐" not in statuses:
                statuses.append("틀어짐")
            reasons.append("중심")
        if protrusion_threshold is not None and protrusion_mm > protrusion_threshold:
            statuses.append("덜박힘")
        inst["final_status"] = "/".join(statuses) if statuses else "정상"
        inst["tilt_reasons"] = reasons
        inst["match_rejected"] = match_rejected

        # 정상 판정(틀어짐 없음)인 나사만 덜박힘 기준선 표본으로 누적 - 틀어짐 나사는
        # foreshortening 등으로 protrusion 계산 자체가 왜곡될 수 있어 기준선 오염 방지
        if inst["tilt_status"] == "정상":
            baseline.append(inst["protrusion_mm"])

        results.append(inst)
    return results, baseline, protrusion_threshold


def main():
    pipeline, align, depth_scale = build_pipeline()
    filters = build_filters()

    try:
        intr = dc.get_color_intrinsics(pipeline)
        fx = intr["fx"]
        depth_mm, color_img = capture_averaged_depth(pipeline, align, filters, depth_scale, n_frames=30)
    finally:
        pipeline.stop()

    print(f"캡처 완료. depth range={depth_mm[depth_mm>0].min()}~{depth_mm.max()}mm")

    screw_dets = dc.detect_screw_heads_by_color(color_img)
    print(f"screw_head 후보: {len(screw_dets)}개")
    stud_holes = dc.detect_stud_holes(color_img, depth_mm, intr)
    print(f"stud_hole 검출: {len(stud_holes)}개")

    baseline = load_baseline()
    results, baseline, protrusion_threshold = classify_insertion(screw_dets, stud_holes, depth_mm, fx, baseline)

    for idx, inst in enumerate(results):
        label = f"screw#{idx}"
        if inst["center_offset_mm"] is not None:
            offset_str = f"{inst['center_offset_mm']}mm"
        elif inst["match_rejected"]:
            offset_str = "판정보류(가장 가까운 구멍도 너무 멀어 신뢰불가)"
        else:
            offset_str = "매칭없음(빈 구멍 미검출)"
        print(f"  [{label}] 중심={inst['center_px']} 틀어짐={inst['tilt_status']}(ar={inst['aspect_ratio']}) "
              f"중심오차={offset_str} protrusion={inst['protrusion_mm']}mm -> {inst['final_status']}")

    save_baseline(baseline)

    json_path = os.path.join(RESULTS_DIR, "6_insertion_check_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {json_path}")

    if protrusion_threshold is not None:
        print(f"\n덜박힘 임계값(잠정, 정상 표본 n={len(baseline)}): {protrusion_threshold:.2f}mm "
              f"(평균+{BASELINE_STD_MULT}표준편차)")
    else:
        print(f"\n덜박힘 임계값 미확정 - 정상 표본 {len(baseline)}/{MIN_BASELINE_N}개 누적됨 "
              f"(계속 실행해 표본을 쌓으면 자동으로 확정됨)")
    print(f"틀어짐(중심) 임계값(기하 제약): {CENTER_OFFSET_THRESHOLD_MM:.2f}mm "
          f"= (stud_hole {dc.GROUND_TRUTH_MM['stud_hole']}mm - "
          f"screw_head {dc.GROUND_TRUTH_MM['screw_head']}mm) / 2")

    # 시각화 - 최종 결과물: stud_hole(빈 구멍, 노랑) + screw_head(삽입 상태별 색) 한 장에 표시
    vis = color_img.copy()
    for hole in stud_holes:
        hx, hy = hole["center_px"]
        hr_px = int(hole["diameter_px"] / 2)
        cv2.circle(vis, (int(hx), int(hy)), hr_px, (0, 255, 255), 2)  # 노랑 = 빈 stud_hole
        cv2.putText(vis, f"hole {hole['diameter_mm']}mm", (int(hx) - 35, int(hy) + hr_px + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
    for inst in results:
        cx, cy = inst["center_px"]
        r_px = int(inst["diameter_px"] / 2)
        color = (0, 0, 255) if inst["final_status"] != "정상" else (0, 255, 0)
        cv2.circle(vis, (int(cx), int(cy)), r_px, color, 2)
        label = f"{inst['final_status']} prot={inst['protrusion_mm']}mm"
        cv2.putText(vis, label, (int(cx) - 50, int(cy) - r_px - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    png_path = os.path.join(RESULTS_DIR, "6_insertion_check_result.png")
    cv2.imwrite(png_path, vis)
    print(f"시각화 저장: {png_path}")


if __name__ == "__main__":
    main()
