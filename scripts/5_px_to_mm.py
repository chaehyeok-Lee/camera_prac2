"""
5번: 세그멘테이션 마스크 + depth로 나사머리/스터드홀 실제 지름(mm) 계산
- 3번 depth 파이프라인(depth_preprocessing.py) + 4번 YOLO-seg 모델(runs/screw_seg/weights/best.pt) 결합
- 공식: 실제지름(mm) = 픽셀지름 x depth(mm) / fx  (핀홀 카메라 모델)
- 캘리퍼 실측값(screw_head=6mm, stud_hole=10mm)과 비교해 정확도 검증
"""

import os
import sys
import json
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_preprocessing import build_pipeline, build_filters, capture_averaged_depth  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_ROOT, "runs", "screw_seg", "weights", "best.pt")
RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

GROUND_TRUTH_MM = {"screw_head": 6.0, "stud_hole": 10.0}  # 캘리퍼 실측값
N_FRAMES = 30
# stud_hole만 YOLO confidence threshold 적용 대상 (screw_head는 색상 기반 검출이라 conf 개념 없음).
# 0.35는 실측 confidence 분포 확인 결과 과했음(0.26~0.36 구간에 진짜 구멍이 더 있고
# 노이즈는 0.10 밑으로 뚝 떨어짐 - 실제 경계는 0.28 근처). 0.35->0.28로 낮춤.
CONF_THRESHOLD_BY_CLASS = {"stud_hole": 0.28}
# 시편은 카메라에서 약 250mm(confirm.md 측정거리) 거리 - 이 범위를 벗어나면 배경(벽/모니터/케이블)으로 간주해 제외.
# color 기반 screw_head 검출기가 폼 패널 틈새로 보이는 먼 배경(흰 벽)을 나사로 오탐하는 게
# held-out 테스트에서 확인돼 추가함 (버텀업 검증: 실측 캡처로 범위 재조정 필요할 수 있음).
SPECIMEN_DEPTH_RANGE_MM = (180, 400)


def get_color_fx(pipeline):
    profile = pipeline.get_active_profile()
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    return color_stream.get_intrinsics().fx


def mask_diameter_px(mask_bool):
    """마스크 픽셀 수 -> 등가원 지름(면적 기준, sqrt(4*area/pi))."""
    area = mask_bool.sum()
    if area == 0:
        return None
    return float(np.sqrt(4 * area / np.pi))


def mask_center(mask_bool):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def blob_aspect_ratio(mask_bool):
    """블롭의 2차 모멘트(공분산 행렬) 고유값 비율 -> 회전에 무관한 장축/단축 비율.
    축 정렬 bounding box의 fill_ratio는 블롭이 회전(기울어짐)해 있으면 실제로 원인데도
    낮게 나와 오탐 처리되는 문제가 있어서, 회전-불변인 모멘트 기반으로 대체."""
    m = cv2.moments(mask_bool.astype(np.uint8), binaryImage=True)
    area = m["m00"]
    if area == 0:
        return None
    mu20 = m["mu20"] / area
    mu02 = m["mu02"] / area
    mu11 = m["mu11"] / area
    cov = np.array([[mu20, mu11], [mu11, mu02]])
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 1e-6, None)
    return float(np.sqrt(eigvals[1]) / np.sqrt(eigvals[0]))  # >=1, 1에 가까울수록 원형


def detect_screw_heads_by_color(color_img, min_area_px=100, max_area_px=4000,
                                 min_solidity=0.75, tilt_aspect_ratio=1.3, max_aspect_ratio=2.5):
    """금속 나사머리는 은색(밝음, 무채색) vs 무광 검은 배경 - 명도 대비가 커서
    학습 없이 밝기 임계값(Otsu, 이미지마다 자동 적응)만으로 검출.
    YOLO screw_head가 학습 데이터 부족으로 불안정한 것의 대안.

    나사가 기울어져 삽입되면(6번 '틀어짐' 케이스) 카메라 시점에서 원이 아니라 타원으로 보임
    -> 이걸 노이즈로 버리지 않고 장단축 비율(aspect_ratio)로 정상/틀어짐을 분류해서 같이 반환.
    solidity(블롭 면적/블롭 컨벡스헐 면적)로 케이블 하이라이트 같은 불규칙한 노이즈만 배제
    (타원은 solidity가 높게 유지되므로 축정렬 bbox fill_ratio보다 회전에 안전)."""
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    _, bright_mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)

    results = []
    for i in range(1, n_labels):  # 0 = 배경
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area_px or area > max_area_px:
            continue
        blob = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(contours[0]))
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < min_solidity:  # 케이블 반사 등 불규칙한 모양 배제 (회전에 안전)
            continue

        aspect_ratio = blob_aspect_ratio(labels == i)
        if aspect_ratio is None:
            continue
        if aspect_ratio > max_aspect_ratio:
            # 실제 나사가 기울어져도 이 정도로 안 늘어남 - 케이블/반사 등 노이즈로 판단해 배제
            continue
        status = "틀어짐" if aspect_ratio >= tilt_aspect_ratio else "정상"
        results.append({"mask": labels == i, "aspect_ratio": round(aspect_ratio, 2), "status": status})
    return results


def deduplicate_instances(instances, dist_ratio=0.6):
    """같은 클래스의 두 인스턴스 중심이 (평균 지름 x dist_ratio)보다 가까우면 같은 구멍/나사로 보고
    confidence 높은 쪽만 남긴다. YOLO 자체 NMS(IoU 기반)로 못 거른 살짝 어긋난 중복 박스 대비용."""
    kept = []
    for inst in sorted(instances, key=lambda x: -x["confidence"]):
        cx, cy = inst["center_px"]
        is_dup = False
        for k in kept:
            if k["class"] != inst["class"]:
                continue
            kcx, kcy = k["center_px"]
            dist = np.hypot(cx - kcx, cy - kcy)
            avg_diam = (k["diameter_px"] + inst["diameter_px"]) / 2
            if dist < avg_diam * dist_ratio:
                is_dup = True
                break
        if not is_dup:
            kept.append(inst)
    n_removed = len(instances) - len(kept)
    if n_removed:
        print(f"중복 제거: {n_removed}개 (같은 구멍/나사로 판단해 confidence 낮은 쪽 제외)")
    return kept


def build_instance(mask_bool, cls_name, conf, depth_mm, fx, debug_label=None, **extra):
    """마스크(bool) + depth + fx -> mm 환산된 인스턴스 dict. YOLO 마스크/색상 검출 마스크 공용.
    extra: aspect_ratio/status(정상/틀어짐) 등 검출기별 부가 정보를 그대로 실어 보냄.
    debug_label: 지정하면 제외될 때마다 사유를 콘솔에 출력 (원인불명 상태로 조용히 버려지는 것 방지)."""
    center = mask_center(mask_bool)
    diam_px = mask_diameter_px(mask_bool)
    if center is None or diam_px is None:
        if debug_label:
            print(f"  [{debug_label}] 제외: 마스크가 비어있음")
        return None

    cx, cy = center
    depth_vals = depth_mm[mask_bool]
    depth_vals = depth_vals[depth_vals > 0]
    depth_source = "mask"
    if depth_vals.size == 0:
        # 금속 나사머리는 반사가 강해 IR 구조광이 제대로 안 돌아와 depth가 통째로
        # 무효(0)일 수 있음 - 마스크를 조금씩 팽창시켜 주변 유효 depth라도 확보.
        # (이 경우 나사 자체가 아니라 인접부 depth라 정확도는 떨어짐 - depth_source로 표시)
        kernel = np.ones((3, 3), np.uint8)
        dilated = mask_bool.astype(np.uint8)
        for _ in range(5):
            dilated = cv2.dilate(dilated, kernel, iterations=1)
            depth_vals = depth_mm[dilated.astype(bool)]
            depth_vals = depth_vals[depth_vals > 0]
            if depth_vals.size > 0:
                depth_source = "dilated_mask"
                break
    if depth_vals.size == 0:
        if debug_label:
            print(f"  [{debug_label}] 제외: 유효 depth 없음 (반사로 인한 홀 - 팽창해도 복구 안됨)")
        return None
    depth_at_instance = float(np.median(depth_vals))
    if not (SPECIMEN_DEPTH_RANGE_MM[0] <= depth_at_instance <= SPECIMEN_DEPTH_RANGE_MM[1]):
        if debug_label:
            print(f"  [{debug_label}] 제외: depth={depth_at_instance:.0f}mm가 예상범위 "
                  f"{SPECIMEN_DEPTH_RANGE_MM} 밖 (배경 오탐 추정)")
        return None  # 시편 거리 범위 밖 - 배경(벽/모니터 등) 오탐으로 판단해 제외

    diam_mm = diam_px * depth_at_instance / fx
    gt = GROUND_TRUTH_MM.get(cls_name)
    error_mm = (diam_mm - gt) if gt is not None else None
    error_pct = (error_mm / gt * 100) if gt else None

    inst = {
        "class": cls_name,
        "confidence": round(conf, 3),
        "center_px": [round(cx, 1), round(cy, 1)],
        "diameter_px": round(diam_px, 2),
        "depth_mm": round(depth_at_instance, 1),
        "depth_source": depth_source,
        "diameter_mm": round(diam_mm, 3),
        "ground_truth_mm": gt,
        "error_mm": round(error_mm, 3) if error_mm is not None else None,
        "error_pct": round(error_pct, 1) if error_pct is not None else None,
    }
    inst.update(extra)
    return inst


_MODEL_CACHE = {}


def detect_stud_holes(color_img, depth_mm, fx, debug_label="stud_hole"):
    """YOLO-seg로 stud_hole 검출 -> build_instance 리스트. 5번 main()/6번 공용 (중복 제거용 추출)."""
    if "model" not in _MODEL_CACHE:
        _MODEL_CACHE["model"] = YOLO(MODEL_PATH)
    model = _MODEL_CACHE["model"]
    h_img, w_img = depth_mm.shape
    results = model.predict(color_img, conf=CONF_THRESHOLD_BY_CLASS["stud_hole"], iou=0.5, verbose=False)
    r = results[0]
    instances = []
    if r.masks is not None:
        for i, cls_idx in enumerate(r.boxes.cls.tolist()):
            cls_name = model.names[int(cls_idx)]
            if cls_name != "stud_hole":
                continue
            conf = r.boxes.conf[i].item()
            mask = r.masks.data[i].cpu().numpy()
            mask_resized = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST) > 0.5
            inst = build_instance(mask_resized, cls_name, conf, depth_mm, fx, debug_label=debug_label)
            if inst:
                instances.append(inst)
    return instances


def main():
    pipeline, align, depth_scale = build_pipeline()
    filters = build_filters()

    try:
        fx = get_color_fx(pipeline)
        depth_mm, color_img = capture_averaged_depth(pipeline, align, filters, depth_scale, n_frames=N_FRAMES)
    finally:
        pipeline.stop()

    print(f"캡처 완료. fx={fx:.2f}px, depth range={depth_mm[depth_mm>0].min()}~{depth_mm.max()}mm")

    instances = []

    # stud_hole: YOLO 세그멘테이션 (기존 방식 유지 - 배경과 구분이 어려워 학습 기반이 필요)
    instances.extend(detect_stud_holes(color_img, depth_mm, fx))

    # screw_head: 색상(명도) 기반 고전 CV - 은색 나사 vs 무광 검은 배경 대비가 커서 학습 불필요.
    # 타원으로 보이는(기울어진) 나사는 6번 '틀어짐' 케이스로 별도 표시 (버리지 않음).
    screw_dets = detect_screw_heads_by_color(color_img)
    for det in screw_dets:
        inst = build_instance(det["mask"], "screw_head", 1.0, depth_mm, fx,  # confidence 개념 없어 1.0 고정
                               debug_label="screw_head",
                               aspect_ratio=det["aspect_ratio"], insertion_status=det["status"])
        if inst:
            instances.append(inst)
    n_tilted = sum(1 for d in screw_dets if d["status"] == "틀어짐")
    print(f"screw_head(색상 기반) 후보: {len(screw_dets)}개 (틀어짐 {n_tilted}개)")

    instances = deduplicate_instances(instances)

    # 결과 저장
    json_path = os.path.join(RESULTS_DIR, "5_px_to_mm_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"fx": fx, "instances": instances}, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {json_path}")

    # 시각화 (screw_head 중 틀어짐은 빨강으로 구분)
    vis = color_img.copy()
    for inst in instances:
        cx, cy = inst["center_px"]
        label = f"{inst['class']} {inst['diameter_mm']}mm"
        if inst["error_pct"] is not None:
            label += f" ({inst['error_pct']:+.1f}%)"
        if inst["class"] == "screw_head":
            is_tilted = inst.get("insertion_status") == "틀어짐"
            color = (0, 0, 255) if is_tilted else (0, 255, 0)  # 빨강=틀어짐, 초록=정상
            if "aspect_ratio" in inst:
                label += f" [{inst['insertion_status']} ar={inst['aspect_ratio']}]"
        else:
            color = (255, 200, 0)
        cv2.circle(vis, (int(cx), int(cy)), int(inst["diameter_px"] / 2), color, 2)
        cv2.putText(vis, label, (int(cx) - 40, int(cy) - int(inst["diameter_px"] / 2) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    png_path = os.path.join(RESULTS_DIR, "5_px_to_mm_result.png")
    cv2.imwrite(png_path, vis)
    print(f"시각화 저장: {png_path}")

    # 클래스별 요약
    for cls_name, gt in GROUND_TRUTH_MM.items():
        errs = [inst["error_pct"] for inst in instances if inst["class"] == cls_name]
        if errs:
            print(f"{cls_name}: n={len(errs)}, 평균오차={np.mean(errs):+.1f}%, "
                  f"표준편차={np.std(errs):.1f}%p")
        else:
            print(f"{cls_name}: 검출 없음")
    tilted = [inst for inst in instances if inst.get("insertion_status") == "틀어짐"]
    if tilted:
        print(f"틀어짐 감지: {len(tilted)}개 - {[inst['center_px'] for inst in tilted]}")


if __name__ == "__main__":
    main()
