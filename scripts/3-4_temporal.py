"""
실험 4: temporal filter alpha 스윕
- 3_depth_preprocessing.md 체크리스트 4번 항목
- 사전조건: RealSense 카메라 연결, 시편을 250mm 거리에 정지 고정 (1280x720 + decimation magnitude=2, spatial alpha=0.75/delta=20 적용 후)
- temporal filter는 프레임 "시퀀스"에 대한 IIR이라 매 설정마다 원본 순서를 그대로 재생해야 함
  -> 프레임 40장을 순서대로 1회 캡처해두고, alpha별로 동일 시퀀스를 새 필터 인스턴스에 그대로 흘려보냄
- 지표: steady-state 노이즈(뒤쪽 K프레임 픽셀별 std), 수렴까지 걸리는 프레임 수(lag)
"""

import os
import json
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
SPATIAL_ALPHA, SPATIAL_DELTA = 0.75, 20  # 실험3-2 결과 반영

N_FRAMES = 40
STEADY_K = 15          # 뒤쪽 K프레임을 steady-state로 간주해 std 계산
LAG_TOL_MM = 0.5        # steady 평균과 이 이내로 붙으면 "수렴"으로 판정
WARMUP = 60  # 20프레임으로는 드리프트가 안 가라앉는 게 실측으로 확인돼 늘림

ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DELTA = 20

FLAT_ROI_HALF = 12
FLAT_SEARCH_STEP = 8
HEATMAP_CROP_HALF = 40


def build_pipeline():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    return pipeline, align, depth_scale


def apply_spatial(rs_frame, alpha=SPATIAL_ALPHA, delta=SPATIAL_DELTA):
    to_disp = rs.disparity_transform(True)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_smooth_alpha, alpha)
    spatial.set_option(rs.option.filter_smooth_delta, delta)
    spatial.set_option(rs.option.holes_fill, 0)
    to_depth = rs.disparity_transform(False)
    f = to_disp.process(rs_frame)
    f = spatial.process(f)
    f = to_depth.process(f)
    return f


def capture_sequence(pipeline, align, depth_scale, n_frames=N_FRAMES, with_spatial=False):
    """
    decimation+threshold(+옵션으로 spatial)까지 적용한 rs.frame 시퀀스(순서 보존) + 컬러 1장 반환.
    temporal filter 단독 효과를 보려면 with_spatial=False로 원본 노이즈를 남겨둬야 함
    (spatial을 먼저 적용하면 평탄면 노이즈가 이미 거의 0으로 saturate돼 temporal의 alpha가 안 보임 -> 실제로 확인됨).
    """
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, DECIMATION_MAGNITUDE)
    threshold = rs.threshold_filter()
    threshold.set_option(rs.option.min_distance, CLIP_MIN_M)
    threshold.set_option(rs.option.max_distance, CLIP_MAX_M)

    frames_out, color_img, fx = [], None, None
    for _ in range(n_frames):
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        f = threshold.process(depth_frame)
        f = decimation.process(f) if DECIMATION_MAGNITUDE > 1 else f
        if with_spatial:
            f = apply_spatial(f)
        f.keep()  # 기본 프레임 풀은 보관 개수가 제한적 -> 나중에 재사용할 프레임은 명시적으로 keep 필요
        frames_out.append(f)
        if color_img is None:
            vsp = f.get_profile().as_video_stream_profile()
            dec_w, dec_h = vsp.width(), vsp.height()
            full_color = np.asanyarray(color_frame.get_data())
            color_img = cv2.resize(full_color, (dec_w, dec_h), interpolation=cv2.INTER_AREA)
            fx = vsp.get_intrinsics().fx
    return frames_out, color_img, fx


def frame_to_mm(rs_frame, depth_scale):
    return np.asanyarray(rs_frame.get_data()).astype(np.float32) * depth_scale * 1000.0


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


def run_temporal(frames, depth_scale, alpha, delta, roi_cy, roi_cx):
    """동일 시퀀스를 새 temporal_filter 인스턴스에 순서대로 흘려보냄."""
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, alpha)
    temporal.set_option(rs.option.filter_smooth_delta, delta)

    roi_means = []
    mm_stack = []
    for f in frames:
        out = temporal.process(f)
        mm = frame_to_mm(out, depth_scale)
        mm_stack.append(mm)
        patch = mm[roi_cy - FLAT_ROI_HALF:roi_cy + FLAT_ROI_HALF, roi_cx - FLAT_ROI_HALF:roi_cx + FLAT_ROI_HALF]
        valid = patch[patch > 0]
        roi_means.append(float(valid.mean()) if valid.size > 0 else None)

    steady_frames = mm_stack[-STEADY_K:]
    steady_stack = np.stack(steady_frames, axis=0)
    patch_stack = steady_stack[:, roi_cy - FLAT_ROI_HALF:roi_cy + FLAT_ROI_HALF, roi_cx - FLAT_ROI_HALF:roi_cx + FLAT_ROI_HALF]
    valid_mask = patch_stack > 0
    per_pixel_std = np.full(patch_stack.shape[1:], np.nan, dtype=np.float32)
    for yy in range(patch_stack.shape[1]):
        for xx in range(patch_stack.shape[2]):
            col = patch_stack[:, yy, xx]
            col = col[col > 0]
            if col.size >= STEADY_K * 0.7:
                per_pixel_std[yy, xx] = col.std()
    steady_std = float(np.nanmean(per_pixel_std)) if np.any(~np.isnan(per_pixel_std)) else None

    steady_vals = [v for v in roi_means[-5:] if v is not None]
    steady_mean = float(np.mean(steady_vals)) if steady_vals else None

    lag = None
    if steady_mean is not None:
        for i, v in enumerate(roi_means):
            if v is not None and abs(v - steady_mean) <= LAG_TOL_MM:
                if all(vv is not None and abs(vv - steady_mean) <= LAG_TOL_MM for vv in roi_means[i:]):
                    lag = i
                    break

    # 시각화용 heatmap 크롭(더 넓은 컨텍스트) 저장
    heat_stack = steady_stack[:, roi_cy - HEATMAP_CROP_HALF:roi_cy + HEATMAP_CROP_HALF,
                               roi_cx - HEATMAP_CROP_HALF:roi_cx + HEATMAP_CROP_HALF]
    heat_std = np.full(heat_stack.shape[1:], np.nan, dtype=np.float32)
    for yy in range(heat_stack.shape[1]):
        for xx in range(heat_stack.shape[2]):
            col = heat_stack[:, yy, xx]
            col = col[col > 0]
            if col.size >= STEADY_K * 0.5:
                heat_std[yy, xx] = col.std()

    return {
        "alpha": alpha, "delta": delta,
        "steady_std_mm": steady_std,
        "lag_frames": lag,
        "roi_means": roi_means,
    }, heat_std


def capture_with_retry(max_attempts=3):
    """이번 세션에서 장시간 반복 캡처 후 align.process가 간헐적으로 RuntimeError를 낸 적이 있어
    하드웨어 리셋 후 재시도하는 방어 로직을 둔다."""
    import time
    last_err = None
    for attempt in range(1, max_attempts + 1):
        pipeline, align, depth_scale = build_pipeline()
        try:
            for _ in range(WARMUP):
                pipeline.wait_for_frames()
            frames, color_img, fx = capture_sequence(pipeline, align, depth_scale)
            return frames, color_img, fx, depth_scale
        except RuntimeError as ex:
            last_err = ex
            print(f"캡처 실패(시도 {attempt}/{max_attempts}): {ex}")
        finally:
            pipeline.stop()
        if attempt < max_attempts:
            ctx = rs.context()
            for d in ctx.query_devices():
                d.hardware_reset()
            time.sleep(8)
    raise RuntimeError(f"{max_attempts}회 재시도 후에도 캡처 실패: {last_err}")


def main():
    frames, color_img, fx, depth_scale = capture_with_retry()

    if not frames:
        print("실패: 캡처된 프레임 없음")
        return

    mm_frames = [frame_to_mm(f, depth_scale) for f in frames]
    baseline_median = np.median(np.stack(mm_frames, axis=0), axis=0)
    (roi_cy, roi_cx), _ = find_flat_roi(baseline_median)
    print(f"고정 ROI: flat=({roi_cy},{roi_cx}), 시퀀스 길이={len(frames)}")

    results = []
    heatmaps = []
    for alpha in ALPHAS:
        r, heat = run_temporal(frames, depth_scale, alpha, DELTA, roi_cy, roi_cx)
        results.append(r)
        heatmaps.append((alpha, heat))
        print(f"alpha={alpha}  steady_std={r['steady_std_mm']}  lag={r['lag_frames']}frames")

    valid_results = [r for r in results if r["steady_std_mm"] is not None]
    best = min(valid_results, key=lambda r: r["steady_std_mm"])
    print(f"OPTIMAL(steady noise 최소): alpha={best['alpha']} "
          f"(std={best['steady_std_mm']:.3f}mm, lag={best['lag_frames']})")

    json_path = os.path.join(RESULTS_DIR, "3-4_temporal_result.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({"results": results, "optimal": best}, jf, ensure_ascii=False, indent=2)
    print(f"결과 저장: {json_path}")

    save_heatmaps(heatmaps)
    save_tradeoff_graph(results, best)


def save_heatmaps(heatmaps):
    n = len(heatmaps)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.2, 2.6))
    vmax = np.nanpercentile(np.concatenate([h.flatten() for _, h in heatmaps]), 95)
    for ax, (alpha, heat) in zip(axes, heatmaps):
        im = ax.imshow(heat, cmap="turbo", vmin=0, vmax=vmax)
        ax.set_xlabel(f"alpha={alpha}")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.7, label="std (mm)")
    png_path = os.path.join(RESULTS_DIR, "3-4_temporal_heatmaps.png")
    fig.savefig(png_path, dpi=150)
    print(f"heatmap 저장: {png_path}")


def save_tradeoff_graph(results, best):
    fig, ax1 = plt.subplots(figsize=(7, 5))
    alphas = [r["alpha"] for r in results]
    stds = [r["steady_std_mm"] for r in results]
    lags = [r["lag_frames"] for r in results]

    l1, = ax1.plot(alphas, stds, "o-", color=PALETTE[0], label="steady 노이즈 std (mm)")
    ax1.set_xlabel("alpha")
    ax1.set_ylabel("steady-state 노이즈 std (mm)")
    ax1.axvline(best["alpha"], color="#c3c2b7", linestyle="--", linewidth=1)

    ax2 = ax1.twinx()
    ax2.grid(False)
    l2, = ax2.plot(alphas, lags, "s--", color=PALETTE[1], label="수렴 프레임 수(lag)")
    ax2.set_ylabel("수렴까지 프레임 수 (lag, frame)")
    ax2.spines["top"].set_visible(False)

    ax1.legend(handles=[l1, l2], loc="best", frameon=False)
    fig.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "3-4_temporal_tradeoff.png")
    fig.savefig(png_path, dpi=150)
    print(f"그래프 저장: {png_path}")


if __name__ == "__main__":
    main()
