"""
7번 실시간 검출의 MOTION_DIFF_THRESHOLD_MM을 감으로 잡지 않고 실측으로 산출하는 도구.
- 4_auto_label.py의 max-gap 분류(밝기값 정렬 후 가장 큰 간격을 경계로 2그룹 분류) 방식을 재사용:
  "움직이는 중" diff 값들과 "정지" diff 값들을 각각 모아 합친 뒤, 두 그룹 사이 최대 간격을
  경계로 삼아 임계값을 산출한다.
- 카메라 필요 - 콘솔 안내에 따라 시편을 움직였다 멈췄다 하면 됨.

사용법: python scripts/calibrate_motion_threshold.py
결과: 권장 MOTION_DIFF_THRESHOLD_MM 값을 출력 - scripts/7_realtime_detect.py 상단 상수에 반영.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_preprocessing import build_pipeline, build_filters  # noqa: E402
from importlib import import_module  # noqa: E402

rtd = import_module("7_realtime_detect")  # 파일명이 숫자로 시작해 import 문으로 바로 못 씀

PHASE_FRAMES = 90  # 약 3초 @ 30fps


def collect_diffs(pipeline, align, filters, depth_scale, n_frames, prompt):
    print(f"\n{prompt}")
    print("3초 후 측정 시작...")
    time.sleep(3)
    diffs = []
    prev = None
    for i in range(n_frames):
        depth_mm, _ = rtd.grab_filtered_depth_mm(pipeline, align, filters, depth_scale)
        if depth_mm is None:
            continue
        ry, rx = rtd.roi_slice(*depth_mm.shape)
        roi = depth_mm[ry, rx]
        if prev is not None:
            valid = (roi > 0) & (prev > 0)
            if valid.sum():
                diffs.append(float(np.mean(np.abs(roi[valid] - prev[valid]))))
        prev = roi
        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{n_frames} 프레임...")
    print(f"수집 완료: {len(diffs)}개 diff 값 (평균 {np.mean(diffs):.2f}mm)")
    return diffs


def max_gap_threshold(values):
    """4_auto_label.classify_by_max_gap과 같은 방식 - 정렬 후 최대 간격 지점을 경계로."""
    sorted_vals = np.sort(values)
    gaps = np.diff(sorted_vals)
    split_at = int(np.argmax(gaps))
    return float((sorted_vals[split_at] + sorted_vals[split_at + 1]) / 2), split_at, gaps[split_at]


def main():
    pipeline, align, depth_scale = build_pipeline()
    filters = build_filters()

    try:
        moving_diffs = collect_diffs(
            pipeline, align, filters, depth_scale, PHASE_FRAMES,
            "[1단계] 시편을 카메라 앞에서 계속 움직여주세요 (좌우/앞뒤로 천천히).")
        still_diffs = collect_diffs(
            pipeline, align, filters, depth_scale, PHASE_FRAMES,
            "[2단계] 이제 시편을 실제 측정할 때처럼 가만히 내려놓아주세요.")
    finally:
        pipeline.stop()

    print(f"\n움직임 중 diff: min={min(moving_diffs):.2f} max={max(moving_diffs):.2f} "
          f"mean={np.mean(moving_diffs):.2f}mm")
    print(f"정지 중 diff:   min={min(still_diffs):.2f} max={max(still_diffs):.2f} "
          f"mean={np.mean(still_diffs):.2f}mm")

    combined = moving_diffs + still_diffs
    threshold, split_at, gap = max_gap_threshold(combined)

    overlap = max(still_diffs) >= min(moving_diffs)
    print(f"\n=== 권장 MOTION_DIFF_THRESHOLD_MM = {threshold:.2f} ===")
    print(f"(정렬된 {len(combined)}개 값 중 최대 간격 {gap:.2f}mm 지점의 중앙값)")
    if overlap:
        print("경고: '정지' 최대값이 '움직임' 최소값보다 큽니다 - 두 분포가 겹칩니다.")
        print("       조명/손떨림 등으로 재측정하거나, SETTLE_FRAMES를 늘려 안정성을 보완하세요.")
    print("\nscripts/7_realtime_detect.py 상단의 MOTION_DIFF_THRESHOLD_MM 값을 위 권장값으로 바꿔주세요.")


if __name__ == "__main__":
    main()
