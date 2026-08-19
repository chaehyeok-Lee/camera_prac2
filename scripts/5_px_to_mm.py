"""
5번: 세그멘테이션 마스크 + depth로 나사머리/스터드홀 실제 지름(mm) 계산
- 3번 depth 파이프라인(depth_preprocessing.py) + 4번 YOLO-seg 모델(runs/screw_seg/weights/best.pt) 결합
- 공식: 실제지름(mm) = 픽셀지름 x depth(mm) / fx  (핀홀 카메라 모델)
- 캘리퍼 실측값(screw_head=6mm, stud_hole=10mm)과 비교해 정확도 검증
- 검출/mm환산/평면보정 로직은 scripts/detection_core.py로 분리(6/7번과 공용)
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

N_FRAMES = 30


def main():
    pipeline, align, depth_scale = build_pipeline()
    filters = build_filters()

    try:
        intr = dc.get_color_intrinsics(pipeline)
        fx = intr["fx"]
        depth_mm, color_img = capture_averaged_depth(pipeline, align, filters, depth_scale, n_frames=N_FRAMES)
    finally:
        pipeline.stop()

    print(f"캡처 완료. fx={fx:.2f}px, depth range={depth_mm[depth_mm>0].min()}~{depth_mm.max()}mm")

    instances = []

    # stud_hole: YOLO 세그멘테이션 + 평면-호모그래피 보정 원 피팅
    instances.extend(dc.detect_stud_holes(color_img, depth_mm, intr))

    # screw_head: 색상(명도) 기반 고전 CV - 은색 나사 vs 무광 검은 배경 대비가 커서 학습 불필요.
    # 타원으로 보이는(기울어진) 나사는 6번 '틀어짐' 케이스로 별도 표시 (버리지 않음).
    screw_dets = dc.detect_screw_heads_by_color(color_img)
    for det in screw_dets:
        inst = dc.build_instance(det["mask"], "screw_head", 1.0, depth_mm, fx,  # confidence 개념 없어 1.0 고정
                                  debug_label="screw_head",
                                  aspect_ratio=det["aspect_ratio"], insertion_status=det["status"])
        if inst:
            instances.append(inst)
    n_tilted = sum(1 for d in screw_dets if d["status"] == "틀어짐")
    print(f"screw_head(색상 기반) 후보: {len(screw_dets)}개 (틀어짐 {n_tilted}개)")

    instances = dc.deduplicate_instances(instances)

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
    for cls_name, gt in dc.GROUND_TRUTH_MM.items():
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
