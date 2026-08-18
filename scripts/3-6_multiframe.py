"""
실험 6: 다중 프레임 합성(median) N 스윕
- 3_depth_preprocessing.md 체크리스트 6번 항목
- 사전조건: RealSense 카메라 연결, 시편을 250mm 거리에 정지 고정
- 파이프라인 순서 그대로(threshold->decimation->spatial(0.75/20)->temporal(0.1/20, 시퀀스 유지) 재현
- 캘리퍼 ground truth가 없어, 가장 많은 프레임(60장) median 합성을 근사 ground truth로 사용해
  N장짜리 median이 그것과 얼마나 일치하는지(±1mm 이내 픽셀 비율)로 대체 측정 (보완 필요에 명시)
"""

import os
import json
import time
import numpy as np
import cv2
import pyrealsense2 as rs
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Malgun Gothic",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.unicode_minus": False,
    "mathtext.fontset": "dejavusans",
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8,
    "axes.edgecolor": "#c3c2b7",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": False,
})
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

WIDTH, HEIGHT = 1280, 720
CLIP_MIN_M, CLIP_MAX_M = 0.20, 0.32
DECIMATION_MAGNITUDE = 2
SPATIAL_ALPHA, SPATIAL_DELTA = 0.75, 20
TEMPORAL_ALPHA, TEMPORAL_DELTA = 0.1, 20

MAX_N = 60          # 이 중 최댓값을 근사 ground truth로 사용
N_SWEEP = [5, 10, 15, 20, 30, 50]
WARMUP = 60          # 실험4에서 확인된 드리프트 문제로 넉넉히
FPS = 30

TOL_MM = 1.0
FLAT_ROI_HALF = 12
FLAT_SEARCH_STEP = 8


def build_pipeline():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    return pipeline, align, depth_scale


def capture_pipeline_sequence(pipeline, align, depth_scale, n_frames=MAX_N):
    """threshold->decimation->spatial->temporal까지 실제 파이프라인 순서로 적용한 시퀀스(순서 보존)."""
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, DECIMATION_MAGNITUDE)
    threshold = rs.threshold_filter()
    threshold.set_option(rs.option.min_distance, CLIP_MIN_M)
    threshold.set_option(rs.option.max_distance, CLIP_MAX_M)

    to_disp = rs.disparity_transform(True)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_smooth_alpha, SPATIAL_ALPHA)
    spatial.set_option(rs.option.filter_smooth_delta, SPATIAL_DELTA)
    spatial.set_option(rs.option.holes_fill, 0)
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, TEMPORAL_ALPHA)
    temporal.set_option(rs.option.filter_smooth_delta, TEMPORAL_DELTA)
    to_depth = rs.disparity_transform(False)

    # temporal filter의 IIR 상태가 콜드스타트라 초반 프레임이 과도기일 수 있음(N=10 이상치 원인으로 추정).
    # 기록 시작 전에 별도로 몇 프레임을 흘려보내 미리 수렴시킨다(priming). 이 프레임들은 frames_mm에 넣지 않음.
    TEMPORAL_PRIME_FRAMES = 15
    for _ in range(TEMPORAL_PRIME_FRAMES):
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        if not depth_frame:
            continue
        f = threshold.process(depth_frame)
        f = decimation.process(f) if DECIMATION_MAGNITUDE > 1 else f
        f = to_disp.process(f)
        f = spatial.process(f)
        f = temporal.process(f)  # 상태만 갱신, 결과는 버림

    frames_mm = []
    per_frame_time_ms = []
    for _ in range(n_frames):
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        if not depth_frame:
            continue
        t0 = time.perf_counter()
        f = threshold.process(depth_frame)
        f = decimation.process(f) if DECIMATION_MAGNITUDE > 1 else f
        f = to_disp.process(f)
        f = spatial.process(f)
        f = temporal.process(f)
        f = to_depth.process(f)
        mm = np.asanyarray(f.get_data()).astype(np.float32) * depth_scale * 1000.0
        t1 = time.perf_counter()
        per_frame_time_ms.append((t1 - t0) * 1000)
        frames_mm.append(mm)
    return frames_mm, per_frame_time_ms


def find_flat_roi(depth_mm, roi_half=FLAT_ROI_HALF, step=FLAT_SEARCH_STEP):
    h, w = depth_mm.shape
    best_std, best_center = np.inf, (h // 2, w // 2)
    for cy in range(roi_half, h - roi_half, step):
        for cx in range(roi_half, w - roi_half, step):
            patch = depth_mm[cy - roi_half:cy + roi_half, cx - roi_half:cx + roi_half]
            valid = patch[patch > 0]
            if valid.size < patch.size * 0.9:
                continue
            std = float(valid.std())
            if std < best_std:
                best_std, best_center = std, (cy, cx)
    return best_center, best_std


def median_composite(stack):
    return np.median(np.stack(stack, axis=0), axis=0)


def main():
    pipeline, align, depth_scale = build_pipeline()
    try:
        for _ in range(WARMUP):
            pipeline.wait_for_frames()
        frames_mm, per_frame_time_ms = capture_pipeline_sequence(pipeline, align, depth_scale)
    finally:
        pipeline.stop()

    if not frames_mm:
        print("실패: 캡처된 프레임 없음")
        return

    print(f"캡처 완료: {len(frames_mm)}프레임, 프레임당 처리시간 평균 {np.mean(per_frame_time_ms):.2f}ms")

    reference = median_composite(frames_mm)  # 근사 ground truth (60장 median)
    (roi_cy, roi_cx), _ = find_flat_roi(reference)
    valid_ref = reference > 0

    results = []
    for n in N_SWEEP:
        subset = frames_mm[:n]
        t0 = time.perf_counter()
        composite = median_composite(subset)
        t1 = time.perf_counter()
        median_time_ms = (t1 - t0) * 1000

        both_valid = valid_ref & (composite > 0)
        diff = np.abs(composite[both_valid] - reference[both_valid])
        within_tol = float((diff <= TOL_MM).mean()) * 100 if diff.size > 0 else None

        capture_time_s = n / FPS
        proc_time_s = (np.mean(per_frame_time_ms[:n]) * n + median_time_ms) / 1000.0
        total_time_s = capture_time_s + proc_time_s

        patch = composite[roi_cy - FLAT_ROI_HALF:roi_cy + FLAT_ROI_HALF, roi_cx - FLAT_ROI_HALF:roi_cx + FLAT_ROI_HALF]
        pv = patch[patch > 0]
        flat_std = float(pv.std()) if pv.size > 0 else None

        results.append({
            "N": n,
            "within_tol_pct": within_tol,
            "flat_noise_std_mm": flat_std,
            "capture_time_s": capture_time_s,
            "total_time_s": total_time_s,
        })
        print(f"N={n}  within_tol={within_tol:.2f}%  flat_std={flat_std:.4f}mm  total_time={total_time_s:.3f}s")

    json_path = os.path.join(RESULTS_DIR, "3-6_multiframe_result.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({
            "results": results,
            "note": "ground truth는 캘리퍼 미확보로 60장 median 합성을 근사 기준으로 사용 (절대 정확도 아님)",
        }, jf, ensure_ascii=False, indent=2)
    print(f"결과 저장: {json_path}")

    save_graph(results)


def save_graph(results):
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ns = [r["N"] for r in results]
    acc = [r["within_tol_pct"] for r in results]
    t = [r["total_time_s"] for r in results]

    l1, = ax1.plot(ns, acc, "o-", color=PALETTE[0], label="기준(60장) ±1mm 이내 비율 (%)")
    ax1.set_xlabel("N (합성 프레임 수)")
    ax1.set_ylabel("±1mm 이내 픽셀 비율 (%)")

    ax2 = ax1.twinx()
    ax2.grid(False)
    l2, = ax2.plot(ns, t, "s--", color=PALETTE[1], label="캡처+처리 소요시간 (s)")
    ax2.set_ylabel("소요시간 (s)")
    ax2.spines["top"].set_visible(False)

    ax1.legend(handles=[l1, l2], loc="lower right", frameon=False)
    fig.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "3-6_multiframe_result.png")
    fig.savefig(png_path, dpi=150)
    print(f"그래프 저장: {png_path}")


if __name__ == "__main__":
    main()
