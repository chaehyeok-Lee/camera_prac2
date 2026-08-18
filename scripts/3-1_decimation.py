"""
실험 1: decimation magnitude x 해상도 비교
- 3_depth_preprocessing.md 체크리스트 1번 항목
- 사전조건: RealSense 카메라 연결, 시편을 250mm 거리에 정지 고정
- 측정: 평탄면 노이즈(std), 나사머리 폭의 픽셀수(intrinsics 기반 추정), 프레임당 처리시간
"""

import os
import time
import json
import numpy as np
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

SCREW_DIAMETER_MM = 8.0   # 캘리퍼로 실측 후 이 값 수정할 것
CLIP_MIN_M, CLIP_MAX_M = 0.20, 0.32

# 프로젝트 루트(scripts/의 상위)에서 실행하든 scripts/ 안에서 실행하든 항상 루트 기준 results/에 저장
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RESOLUTIONS = [(640, 480), (848, 480), (1280, 720)]
MAGNITUDES = [1, 2, 3, 4]  # 1 = decimation 미적용(baseline)
N_FRAMES = 15
WARMUP = 15

FLAT_SEARCH_STEP = 8


def find_flat_roi(depth_mm, roi_half, step=FLAT_SEARCH_STEP):
    """valid 비율 높고 국소 std가 최소인 평탄 지점 자동 탐색 (시편 리브 구조 회피).
    실험3/4/6과 동일 방식 - 중앙 고정 ROI가 리브에 걸리는 문제(1번 보완 필요 항목) 해결."""
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


def run_one(width, height, magnitude):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, magnitude)
    threshold = rs.threshold_filter()
    threshold.set_option(rs.option.min_distance, CLIP_MIN_M)
    threshold.set_option(rs.option.max_distance, CLIP_MAX_M)

    for _ in range(WARMUP):
        pipeline.wait_for_frames()

    depth_frames_mm, times, last_fx = [], [], None

    for _ in range(N_FRAMES):
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        t0 = time.perf_counter()
        f = threshold.process(depth_frame)
        if magnitude > 1:
            f = decimation.process(f)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

        last_fx = f.get_profile().as_video_stream_profile().get_intrinsics().fx
        depth_m = np.asanyarray(f.get_data()).astype(np.float32) * depth_scale
        depth_frames_mm.append(depth_m * 1000)  # mm

    pipeline.stop()

    # 평탄 ROI를 매번 중앙 고정 대신 자동 탐색 (리브 구조 회피 - 1번 보완 필요 항목 해결)
    mean_frame = np.mean(np.stack(depth_frames_mm, axis=0), axis=0)
    h, w = mean_frame.shape
    roi_half = max(5, int(10 * (w / width)))
    (cy, cx), _ = find_flat_roi(mean_frame, roi_half)

    stds, roi_means = [], []
    for depth_mm in depth_frames_mm:
        roi = depth_mm[cy - roi_half:cy + roi_half, cx - roi_half:cx + roi_half]
        valid = roi[roi > 0]
        if valid.size > 0:
            stds.append(float(valid.std()))
            roi_means.append(float(valid.mean()))

    mean_depth_mm = float(np.mean(roi_means)) if roi_means else None
    mm_per_pixel = mean_depth_mm / last_fx if mean_depth_mm else None
    screw_pixel_count = SCREW_DIAMETER_MM / mm_per_pixel if mm_per_pixel else None

    return {
        "resolution": f"{width}x{height}",
        "magnitude": magnitude,
        "noise_std_mm": float(np.mean(stds)) if stds else None,
        "mean_depth_mm": mean_depth_mm,
        "roi_center": [cy, cx],
        "screw_pixel_count": screw_pixel_count,
        "proc_time_ms": float(np.mean(times)),
    }


def main():
    results = []
    for (w, h) in RESOLUTIONS:
        for mag in MAGNITUDES:
            print(f"측정 중: {w}x{h}, magnitude={mag} ...")
            try:
                r = run_one(w, h, mag)
                results.append(r)
                print(f"  noise_std={r['noise_std_mm']:.3f}mm, "
                      f"screw_px={r['screw_pixel_count']:.1f}, "
                      f"time={r['proc_time_ms']:.2f}ms")
            except Exception as e:
                print(f"  실패: {e}")

    json_path = os.path.join(RESULTS_DIR, "3-1_decimation_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {json_path}")

    # 조건(해상도 3종) x 지표(노이즈/픽셀수) 비교: 해상도별 서브플롯+dual-axis 대신
    # 지표별 그래프 2개로 나누고, 해상도는 색상으로 구분해 한 그래프 안에서 바로 비교되게 함
    # (chart-style 스킬 5번 참고)
    fig, (ax_noise, ax_pix) = plt.subplots(1, 2, figsize=(11, 5))

    for i, (w, h) in enumerate(RESOLUTIONS):
        subset = sorted(
            [r for r in results if r["resolution"] == f"{w}x{h}"],
            key=lambda r: r["magnitude"],
        )
        mags = [r["magnitude"] for r in subset]
        noise = [r["noise_std_mm"] for r in subset]
        pix = [r["screw_pixel_count"] for r in subset]
        color = PALETTE[i]
        label = f"{w}x{h}"

        ax_noise.plot(mags, noise, "o-", color=color, label=label)
        ax_pix.plot(mags, pix, "o-", color=color, label=label)

    ax_noise.set_xlabel("decimation magnitude")
    ax_noise.set_ylabel("노이즈 std (mm)")
    ax_noise.legend(frameon=False)

    ax_pix.set_xlabel("decimation magnitude")
    ax_pix.set_ylabel("나사머리 폭 (px)")
    ax_pix.legend(frameon=False)

    fig.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "3-1_decimation_result.png")
    fig.savefig(png_path, dpi=150)
    print(f"그래프 저장: {png_path}")


if __name__ == "__main__":
    main()
