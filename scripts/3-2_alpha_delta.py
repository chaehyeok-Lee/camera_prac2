"""
실험 3-2: RealSense spatial 필터의 alpha x delta 최적값 탐색
- 3_depth_preprocessing.md 체크리스트 "3-2. alpha/delta 최적값" 항목
- 사전조건: RealSense 카메라 연결, 시편을 250mm 거리에 정지 고정 (1280x720 + decimation magnitude=2)
- delta는 평탄면(edge 없음)에서는 효과가 안 보인다는 게 3-3 실험에서 확인됨
  -> 이번엔 나사머리 "경계"에서 edge 전환폭(sharpness)을 같이 측정해 노이즈-경계보존 트레이드오프로 판단
- 원본 프레임 1회 캡처, 동일 ROI/나사 위치 고정 후 alpha x delta 그리드 전부 같은 데이터에 적용
"""

import os
import json
import numpy as np
import cv2
import pyrealsense2 as rs
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

WIDTH, HEIGHT = 1280, 720
CLIP_MIN_M, CLIP_MAX_M = 0.20, 0.32
DECIMATION_MAGNITUDE = 2
N_FRAMES = 15
WARMUP = 20

ALPHAS = [0.25, 0.5, 0.75, 1.0]
DELTAS = [5, 20, 40]

FLAT_ROI_HALF = 12
FLAT_SEARCH_STEP = 8
EDGE_SEARCH_HALF = 18  # 시편이 대각선 리브(홈) 구조라 넓은 라인은 옆 리브/골까지 걸침 -> 나사 주변 좁은 구간만


def build_pipeline():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    return pipeline, align, depth_scale


def capture_stack(pipeline, align, depth_scale, n_frames=N_FRAMES):
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


def find_screw_circle(color_img, search_x_frac=(0.0, 0.78)):
    h, w = color_img.shape[:2]
    x_lo, x_hi = int(w * search_x_frac[0]), int(w * search_x_frac[1])
    gray_full = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray_full[:, x_lo:x_hi], 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
        param1=80, param2=20, minRadius=6, maxRadius=40,
    )
    if circles is None:
        return None
    circles = np.round(circles[0]).astype(int)
    cy0, cx0 = h // 2, (x_hi - x_lo) // 2
    circles = sorted(circles, key=lambda c: (c[0] - cx0) ** 2 + (c[1] - cy0) ** 2)
    cx, cy, r = circles[0]
    return int(cy), int(cx + x_lo), int(r)


def find_best_row(depth_mm, center, search_half=EDGE_SEARCH_HALF, row_search=6):
    """나사 주변은 작은 구멍이라 특정 행이 통째로 invalid일 수 있음 -> 유효 비율 최고인 행을 주변에서 탐색."""
    cy, cx = center
    best_ratio, best_cy = -1, cy
    for dy in range(-row_search, row_search + 1):
        y = cy + dy
        if y < 0 or y >= depth_mm.shape[0]:
            continue
        line = depth_mm[y, max(0, cx - search_half):cx + search_half]
        ratio = (line > 0).mean()
        if ratio > best_ratio:
            best_ratio, best_cy = ratio, y
    return best_cy, best_ratio


def edge_transition_width_px(depth_mm, center, search_half=EDGE_SEARCH_HALF):
    """나사머리 경계의 depth 전환폭(px). gradient가 피크의 10% 이상인 연속 구간 길이 -> 작을수록 경계가 선명(잘 보존됨).
    center=(row, cx): row는 baseline에서 1회만 탐색해 고정한 행(모든 기법 공통)."""
    cy, cx = center
    line = depth_mm[cy, max(0, cx - search_half):cx + search_half].astype(np.float64)
    valid_mask = line > 0
    if valid_mask.sum() < len(line) * 0.7:
        return None
    line = np.interp(np.arange(len(line)), np.where(valid_mask)[0], line[valid_mask])
    grad = np.abs(np.gradient(line))
    peak = grad.max()
    if peak < 1e-6:
        return None
    thresh = 0.1 * peak
    peak_idx = int(np.argmax(grad))
    left = peak_idx
    while left > 0 and grad[left - 1] > thresh:
        left -= 1
    right = peak_idx
    while right < len(grad) - 1 and grad[right + 1] > thresh:
        right += 1
    return right - left + 1


def apply_realsense_spatial(rs_frame, alpha, delta):
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


def main():
    pipeline, align, depth_scale = build_pipeline()
    try:
        for _ in range(WARMUP):
            pipeline.wait_for_frames()
        frames, color_img, fx = capture_stack(pipeline, align, depth_scale)
    finally:
        pipeline.stop()

    if not frames:
        print("실패: 캡처된 프레임 없음")
        return

    mm_frames = [frame_to_mm(f, depth_scale) for f in frames]
    baseline_mean = np.median(np.stack(mm_frames, axis=0), axis=0)
    (roi_cy, roi_cx), _ = find_flat_roi(baseline_mean)
    mm_per_pixel = float(np.median(baseline_mean[baseline_mean > 0])) / fx if fx else None
    screw = find_screw_circle(color_img)
    screw_center = None
    if screw:
        best_row, best_ratio = find_best_row(baseline_mean, (screw[0], screw[1]))
        screw_center = (best_row, screw[1]) if best_ratio >= 0.7 else None
    print(f"고정 ROI: flat=({roi_cy},{roi_cx}), screw={screw_center}, mm/px={mm_per_pixel}")

    results = []
    for alpha in ALPHAS:
        for delta in DELTAS:
            filtered_stack = [
                frame_to_mm(apply_realsense_spatial(f, alpha, delta), depth_scale) for f in frames
            ]
            mean_mm = np.median(np.stack(filtered_stack, axis=0), axis=0)

            patch = mean_mm[roi_cy - FLAT_ROI_HALF:roi_cy + FLAT_ROI_HALF, roi_cx - FLAT_ROI_HALF:roi_cx + FLAT_ROI_HALF]
            valid = patch[patch > 0]
            flat_std = float(valid.std()) if valid.size > 0 else None

            edge_px = edge_transition_width_px(mean_mm, screw_center) if screw_center else None
            edge_mm = edge_px * mm_per_pixel if (edge_px is not None and mm_per_pixel) else None

            results.append({
                "alpha": alpha, "delta": delta,
                "flat_noise_std_mm": flat_std,
                "edge_transition_width_mm": edge_mm,
            })
            print(f"alpha={alpha} delta={delta}  std={flat_std}  edge_width={edge_mm}")

    # 정규화 합산 점수로 optimal 선정. 나사머리는 금속 반사 때문에 depth 자체가 홀(0)이 되는 경우가 많아
    # 엣지 전환폭을 못 잰 경우(edge_mm 전부 None)엔 평탄면 노이즈만으로 점수 산정 (트레이드오프 검증 불가, 결과에 명시)
    stds = [r["flat_noise_std_mm"] for r in results if r["flat_noise_std_mm"] is not None]
    edges = [r["edge_transition_width_mm"] for r in results if r["edge_transition_width_mm"] is not None]
    std_min, std_max = min(stds), max(stds)
    edge_available = len(edges) > 0
    edge_min, edge_max = (min(edges), max(edges)) if edge_available else (None, None)

    for r in results:
        if r["flat_noise_std_mm"] is None:
            r["score"] = None
            continue
        n_std = (r["flat_noise_std_mm"] - std_min) / (std_max - std_min) if std_max > std_min else 0
        if edge_available and r["edge_transition_width_mm"] is not None:
            n_edge = (r["edge_transition_width_mm"] - edge_min) / (edge_max - edge_min) if edge_max > edge_min else 0
            r["score"] = n_std + n_edge
        else:
            r["score"] = n_std

    scored = [r for r in results if r["score"] is not None]
    best = min(scored, key=lambda r: r["score"])
    print(f"OPTIMAL: alpha={best['alpha']} delta={best['delta']} "
          f"(std={best['flat_noise_std_mm']:.3f}mm, edge_available={edge_available})")

    json_path = os.path.join(RESULTS_DIR, "3-2_alpha_delta_result.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({
            "results": results,
            "optimal": best,
            "edge_measurement_available": edge_available,
            "note": None if edge_available else
                "나사머리 영역이 반사성 금속이라 depth 센서 홀(무효값)이 생겨 엣지 전환폭 측정 불가 -> 평탄면 노이즈만으로 optimal 산정",
        }, jf, ensure_ascii=False, indent=2)
    print(f"결과 저장: {json_path}")

    save_graph(results, best, edge_available)


def save_graph(results, best, edge_available):
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("viridis")
    alpha_list = sorted(set(r["alpha"] for r in results))

    if edge_available:
        for a in alpha_list:
            subset = [r for r in results if r["alpha"] == a and r["score"] is not None]
            subset = sorted(subset, key=lambda r: r["delta"])
            xs = [r["edge_transition_width_mm"] for r in subset]
            ys = [r["flat_noise_std_mm"] for r in subset]
            color = cmap(alpha_list.index(a) / max(1, len(alpha_list) - 1))
            ax.plot(xs, ys, "o-", color=color, label=f"alpha={a}")
            for r in subset:
                ax.annotate(f"d={r['delta']}", (r["edge_transition_width_mm"], r["flat_noise_std_mm"]),
                            fontsize=7, textcoords="offset points", xytext=(4, 4))
        ax.scatter([best["edge_transition_width_mm"]], [best["flat_noise_std_mm"]],
                   s=200, facecolors="none", edgecolors="red", linewidths=2, label="optimal")
        ax.set_xlabel("엣지 전환폭 (mm, 작을수록 경계 선명)")
        ax.set_ylabel("평탄면 노이즈 std (mm, 작을수록 좋음)")
        ax.set_title("RealSense spatial: alpha x delta 트레이드오프 (노이즈 vs 경계보존)")
        ax.legend(fontsize=8)
    else:
        # 나사머리 depth가 홀이라 엣지 지표 없음 -> 평탄면 노이즈만 alpha별 막대그래프로 표시 (delta는 전 구간 무효과)
        deltas_sorted = sorted(set(r["delta"] for r in results))
        rep_delta = deltas_sorted[0]
        subset = sorted([r for r in results if r["delta"] == rep_delta], key=lambda r: r["alpha"])
        xs = [str(r["alpha"]) for r in subset]
        ys = [r["flat_noise_std_mm"] for r in subset]
        colors = ["red" if r["alpha"] == best["alpha"] else "tab:blue" for r in subset]
        ax.bar(xs, ys, color=colors)
        ax.set_xlabel("alpha (delta는 5~40 전 구간에서 효과 없어 대표값 1개만 표시)")
        ax.set_ylabel("평탄면 노이즈 std (mm, 작을수록 좋음)")
        ax.set_title("RealSense spatial: alpha별 평탄면 노이즈\n(나사머리 depth 홀로 엣지 트레이드오프는 미측정 — 붉은 막대=optimal)")

    fig.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "3-2_alpha_delta_result.png")
    fig.savefig(png_path, dpi=140)
    print(f"그래프 저장: {png_path}")


if __name__ == "__main__":
    main()
