"""
4-2: stud_hole 탐지 정확도 개선 실험
- 배경: 6번 과제 검증 중 stud_hole 재현율이 낮게 확인됨(육안 16개 중 10개, 62%)
- 같은 캡처 프레임(color_img, depth_mm) 하나에 여러 탐지 방식을 동일 조건으로 적용해 비교
  (캡처마다 손떨림/조명 등 조건이 달라지는 걸 막기 위해 프레임을 한 번만 캡처)
- 나사(screw_head) 로직은 건드리지 않음 - stud_hole 탐지만 대상
- EXPECTED_HOLE_COUNT=16은 4번 과제 라벨링 때 확인된 시편 전체 구멍 수(4_segmentation.md 참고)
"""

import os
import sys
import json
import numpy as np
import cv2
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_preprocessing import build_pipeline, build_filters, capture_averaged_depth  # noqa: E402
from importlib import import_module  # noqa: E402

_px_to_mm = import_module("5_px_to_mm")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EXPECTED_HOLE_COUNT = 16


def extract_instances(r, model, depth_mm, fx, h_img, w_img, min_circularity=None):
    """YOLO 결과(r)에서 stud_hole 인스턴스 추출. min_circularity 지정 시 원형도 후처리 필터 적용."""
    instances = []
    if r.masks is None:
        return instances
    for i, cls_idx in enumerate(r.boxes.cls.tolist()):
        cls_name = model.names[int(cls_idx)]
        if cls_name != "stud_hole":
            continue
        conf_val = r.boxes.conf[i].item()
        mask = r.masks.data[i].cpu().numpy()
        mask_resized = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST) > 0.5

        circularity = None
        if min_circularity is not None:
            contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
            if circularity < min_circularity:
                continue

        inst = _px_to_mm.build_instance(mask_resized, cls_name, conf_val, depth_mm, fx)
        if inst:
            if circularity is not None:
                inst["circularity"] = round(circularity, 2)
            instances.append(inst)
    return instances


def variant_baseline(color_img, depth_mm, fx, model):
    """기존 방식: conf=0.28(CONF_THRESHOLD_BY_CLASS), 후처리 없음."""
    h_img, w_img = depth_mm.shape
    conf = _px_to_mm.CONF_THRESHOLD_BY_CLASS["stud_hole"]
    r = model.predict(color_img, conf=conf, iou=0.5, verbose=False)[0]
    return extract_instances(r, model, depth_mm, fx, h_img, w_img)


def variant_low_conf_circularity(color_img, depth_mm, fx, model, conf=0.15, min_circularity=0.7):
    """conf를 낮춰 재현율을 올리고, 원형도 필터로 저confidence 구간에서 늘어난 오탐을 제거."""
    h_img, w_img = depth_mm.shape
    r = model.predict(color_img, conf=conf, iou=0.5, verbose=False)[0]
    return extract_instances(r, model, depth_mm, fx, h_img, w_img, min_circularity=min_circularity)


def variant_clahe(color_img, depth_mm, fx, model, conf=None):
    """무광 검정 위 무광 검정이라 원본 대비가 낮음 - CLAHE로 국소 대비를 올린 뒤 탐지."""
    conf = conf if conf is not None else _px_to_mm.CONF_THRESHOLD_BY_CLASS["stud_hole"]
    lab = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    enhanced = cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    h_img, w_img = depth_mm.shape
    r = model.predict(enhanced, conf=conf, iou=0.5, verbose=False)[0]
    return extract_instances(r, model, depth_mm, fx, h_img, w_img), enhanced


def variant_depth_local_max(depth_mm, fx, bg_kernel=41, min_residual_mm=3.0,
                             min_area_px=80, max_area_px=1500, min_circularity=0.6):
    """구멍은 패널보다 depth가 더 큼(카메라에서 더 멈) - 이걸 이용해 RGB 없이 depth만으로 탐지.
    큰 커널 median으로 '국소 배경(리브 굴곡 포함)' 추정 후 원본과의 차이(residual)에서
    양의 방향으로 두드러지는 국소 영역을 구멍 후보로 본다. RGB 대비 문제와 완전히 독립된 신호라
    YOLO/CLAHE가 놓치는 저대비 구멍을 보완할 수 있을지 확인하는 실험."""
    depth_f = depth_mm.astype(np.float32)
    valid = depth_f > 0

    depth_u8_scale = 255.0 / max(depth_f[valid].max(), 1)
    depth_u8 = np.clip(depth_f * depth_u8_scale, 0, 255).astype(np.uint8)
    background_u8 = cv2.medianBlur(depth_u8, bg_kernel)
    background_mm = background_u8.astype(np.float32) / depth_u8_scale

    residual = depth_f - background_mm
    residual[~valid] = 0

    hole_mask = (residual > min_residual_mm).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hole_mask, connectivity=8)
    instances = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area_px or area > max_area_px:
            continue
        blob = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = contours[0]
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
        if circularity < min_circularity:
            continue
        inst = _px_to_mm.build_instance(labels == i, "stud_hole", 1.0, depth_mm, fx)
        if inst:
            inst["circularity"] = round(circularity, 2)
            instances.append(inst)
    return instances


def draw_and_save(color_img, instances, name):
    vis = color_img.copy()
    for inst in instances:
        cx, cy = inst["center_px"]
        r_px = int(inst["diameter_px"] / 2)
        cv2.circle(vis, (int(cx), int(cy)), r_px, (0, 255, 255), 2)
        cv2.putText(vis, f"{inst['diameter_mm']}mm", (int(cx) - 30, int(cy) + r_px + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
    path = os.path.join(RESULTS_DIR, f"4-2_hole_{name}.png")
    cv2.imwrite(path, vis)
    return path


def main():
    pipeline, align, depth_scale = build_pipeline()
    filters = build_filters()
    try:
        fx = _px_to_mm.get_color_fx(pipeline)
        depth_mm, color_img = capture_averaged_depth(pipeline, align, filters, depth_scale, n_frames=30)
    finally:
        pipeline.stop()
    print(f"캡처 완료. depth range={depth_mm[depth_mm>0].min()}~{depth_mm.max()}mm "
          f"(이 프레임 한 장으로 아래 모든 실험을 동일 조건에서 비교)")

    model = YOLO(_px_to_mm.MODEL_PATH)

    summary = {}

    base = variant_baseline(color_img, depth_mm, fx, model)
    summary["baseline(conf=0.28)"] = base
    p = draw_and_save(color_img, base, "baseline")
    print(f"[baseline] {len(base)}개 검출 -> {p}")

    low_conf = variant_low_conf_circularity(color_img, depth_mm, fx, model)
    summary["low_conf+circularity(conf=0.15,circ>=0.7)"] = low_conf
    p = draw_and_save(color_img, low_conf, "low_conf_circularity")
    print(f"[low_conf+circularity] {len(low_conf)}개 검출 -> {p}")

    clahe_result, enhanced_img = variant_clahe(color_img, depth_mm, fx, model)
    summary["clahe(conf=0.28)"] = clahe_result
    p = draw_and_save(color_img, clahe_result, "clahe")
    cv2.imwrite(os.path.join(RESULTS_DIR, "4-2_hole_clahe_input.png"), enhanced_img)
    print(f"[clahe] {len(clahe_result)}개 검출 -> {p}")

    depth_result = variant_depth_local_max(depth_mm, fx)
    summary["depth_local_max"] = depth_result
    p = draw_and_save(color_img, depth_result, "depth_local_max")
    print(f"[depth_local_max] {len(depth_result)}개 검출 -> {p}")

    print(f"\n기대 구멍 수(육안, 시편 전체): {EXPECTED_HOLE_COUNT}개")
    print("변형별 재현율(추정, 오탐 포함 가능성 있음 - 이미지로 육안 확인 필요):")
    for name, insts in summary.items():
        recall_est = len(insts) / EXPECTED_HOLE_COUNT * 100
        print(f"  {name}: {len(insts)}개 ({recall_est:.0f}%)")

    json_path = os.path.join(RESULTS_DIR, "4-2_hole_detection_experiments.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in summary.items()}, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {json_path}")


if __name__ == "__main__":
    main()
