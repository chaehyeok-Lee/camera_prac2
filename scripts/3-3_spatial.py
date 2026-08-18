"""
실험 3: spatial filter 기법 비교 (baseline / RealSense spatial / bilateral(depth) /
        guided(RGB guide) / NLM / median / anisotropic diffusion)
- 3_depth_preprocessing.md 체크리스트 3번 항목
- 사전조건: RealSense 카메라 연결, 시편을 250mm 거리에 정지 고정 (실험1 결론: 1280x720 + decimation magnitude=2)
- 원본 프레임은 한 번만 캡처하고, 모든 기법을 동일한 원본 스택에 적용 (공정 비교를 위해 캡처/ROI 위치 고정)
- 평탄 ROI(노이즈 std)와 나사머리 위치는 baseline(무필터)에서 한 번만 탐색해 모든 기법에 동일하게 사용
"""

import os
import time
import json
import numpy as np
import cv2
import scipy.ndimage as ndi
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

WIDTH, HEIGHT = 1280, 720
CLIP_MIN_M, CLIP_MAX_M = 0.20, 0.32
DECIMATION_MAGNITUDE = 2
N_FRAMES = 15  # 캡처가 1회뿐이라 기존보다 여유있게
WARMUP = 20
SCREW_DIAMETER_MM = 8.0  # 캘리퍼로 실측 후 이 값 수정할 것 (실험1과 동일 placeholder)

FLAT_ROI_HALF = 12
FLAT_SEARCH_STEP = 8
CROP_HALF = 60          # 시각화용 컨텍스트 crop 반경 (실측 영역보다 크게 보여줌 → 사각형으로 실측 영역 표시)
EDGE_SEARCH_HALF = 40   # edge_diameter_mm의 라인 프로파일 탐색 반경


# ---------- 캡처 (전체 실험에서 단 1회) ----------

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
    """decimation+threshold만 적용한 rs.frame 스택 + 정렬된(decimation 해상도로 리사이즈된) 컬러 1장 반환."""
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, DECIMATION_MAGNITUDE)
    threshold = rs.threshold_filter()
    threshold.set_option(rs.option.min_distance, CLIP_MIN_M)
    threshold.set_option(rs.option.max_distance, CLIP_MAX_M)

    frames_out = []
    color_img = None
    fx = None
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


# ---------- 자동 ROI / 나사머리 탐색 (baseline에서 1회만 실행, 이후 전 기법 공통 고정) ----------

def find_flat_roi(depth_mm, roi_half=FLAT_ROI_HALF, step=FLAT_SEARCH_STEP):
    """valid 비율 높고 국소 std가 최소인 평탄 지점 자동 탐색."""
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
    """
    컬러 프레임에서 Hough circle로 나사머리 후보 탐색.
    배경(모니터/케이블)이 프레임 우측에 걸려 오검출되는 것을 막기 위해
    시편이 있는 좌측 영역(search_x_frac)으로 탐색 범위를 제한한다.
    중심에 가장 가까운 원을 채택.
    """
    h, w = color_img.shape[:2]
    x_lo, x_hi = int(w * search_x_frac[0]), int(w * search_x_frac[1])
    gray_full = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray_full[:, x_lo:x_hi], 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
        param1=80, param2=25, minRadius=6, maxRadius=40,
    )
    if circles is None:
        return None
    circles = np.round(circles[0]).astype(int)
    cy0, cx0 = h // 2, (x_hi - x_lo) // 2
    circles = sorted(circles, key=lambda c: (c[0] - cx0) ** 2 + (c[1] - cy0) ** 2)
    cx, cy, r = circles[0]
    return int(cy), int(cx + x_lo), int(r)


def edge_diameter_mm(depth_mm, center, mm_per_pixel, search_half=EDGE_SEARCH_HALF):
    """고정된 중심점 기준 수평 라인 프로파일에서 depth 단차 경계(중간값 crossing)로 지름 추정."""
    cy, cx = center
    line = depth_mm[cy, max(0, cx - search_half):cx + search_half]
    valid = line[line > 0]
    if valid.size < 10:
        return None
    surround = np.median(np.concatenate([valid[:5], valid[-5:]]))
    center_val = np.median(valid[valid.size // 2 - 3: valid.size // 2 + 3])
    mid = (surround + center_val) / 2.0
    above = np.where(line > 0, (line > mid) if center_val > surround else (line < mid), False)
    idx = np.where(above)[0]
    if idx.size < 2:
        return None
    pixel_width = idx.max() - idx.min()
    return pixel_width * mm_per_pixel


# ---------- 필터 구현 ----------

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


def guided_filter(guide_gray, src, radius, eps):
    """
    무광 검은색 시편 특성상 RGB 가이드의 픽셀 분산(var_g)이 0에 가까울 수 있음.
    a = cov_gs / (var_g + eps)에서 분모가 작으면 radius가 커질수록 계수가 불안정해져
    노이즈를 오히려 증폭시킬 수 있음 (radius 스윕 결과 해석 시 참고).
    """
    guide = guide_gray.astype(np.float32)
    src = src.astype(np.float32)
    mean_g = cv2.boxFilter(guide, -1, (radius, radius))
    mean_s = cv2.boxFilter(src, -1, (radius, radius))
    mean_gs = cv2.boxFilter(guide * src, -1, (radius, radius))
    cov_gs = mean_gs - mean_g * mean_s
    mean_gg = cv2.boxFilter(guide * guide, -1, (radius, radius))
    var_g = mean_gg - mean_g * mean_g
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))
    return mean_a * guide + mean_b


def anisotropic_diffusion(img, n_iter, kappa, gamma=0.15):
    out = img.astype(np.float32).copy()
    for _ in range(n_iter):
        n = np.roll(out, -1, axis=0) - out
        s = np.roll(out, 1, axis=0) - out
        e = np.roll(out, -1, axis=1) - out
        w = np.roll(out, 1, axis=1) - out
        cn = np.exp(-(n / kappa) ** 2)
        cs = np.exp(-(s / kappa) ** 2)
        ce = np.exp(-(e / kappa) ** 2)
        cw = np.exp(-(w / kappa) ** 2)
        out += gamma * (cn * n + cs * s + ce * e + cw * w)
    return out


def nlm_on_depth(depth_mm, h):
    """16bit/float depth를 8bit로 정규화 후 NLM 적용, 다시 mm 스케일로 복원."""
    valid = depth_mm[depth_mm > 0]
    if valid.size == 0:
        return depth_mm
    lo, hi = float(valid.min()), float(valid.max())
    if hi <= lo:
        return depth_mm
    norm = np.clip((depth_mm - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(norm, h=h, templateWindowSize=7, searchWindowSize=21)
    restored = denoised.astype(np.float32) / 255 * (hi - lo) + lo
    restored[depth_mm == 0] = 0
    return restored


# ---------- 메인 ----------

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

    gray_guide = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    mm_frames = [frame_to_mm(f, depth_scale) for f in frames]

    # baseline(무필터) 평균에서 평탄 ROI / 나사머리 위치를 1회만 탐색 -> 이후 전 기법 공통 고정
    baseline_mean = np.median(np.stack(mm_frames, axis=0), axis=0)
    (roi_cy, roi_cx), _ = find_flat_roi(baseline_mean)
    mm_per_pixel = float(np.median(baseline_mean[baseline_mean > 0])) / fx if fx else None
    screw = find_screw_circle(color_img)
    screw_center = (screw[0], screw[1]) if screw else None
    print(f"고정 ROI: flat=({roi_cy},{roi_cx}), screw={screw_center}, mm/px={mm_per_pixel}")

    configs = [("baseline", "no-filter", lambda f, mm: mm)]

    for delta in [5, 15, 25, 35]:
        configs.append((
            "realsense_spatial", f"delta={delta}",
            lambda f, mm, d=delta: frame_to_mm(apply_realsense_spatial(f, 0.5, d), depth_scale)
        ))

    for sigma in [5, 15, 25, 35]:
        configs.append((
            "bilateral_depth", f"sigma={sigma}",
            lambda f, mm, s=sigma: cv2.bilateralFilter(mm, d=5, sigmaColor=s, sigmaSpace=15)
        ))

    for radius in [4, 8, 16]:
        configs.append((
            "guided_rgb", f"radius={radius}",
            lambda f, mm, r=radius: guided_filter(gray_guide, mm, r, eps=25.0)
        ))

    for h in [5, 10, 20]:
        configs.append(("nlm", f"h={h}", lambda f, mm, hh=h: nlm_on_depth(mm, hh)))

    for k in [3, 5, 7]:
        configs.append(("median", f"ksize={k}", lambda f, mm, kk=k: ndi.median_filter(mm, size=kk)))

    for it in [5, 15, 30]:
        configs.append((
            "anisotropic", f"iter={it}",
            lambda f, mm, ii=it: anisotropic_diffusion(mm, ii, kappa=20)
        ))

    results = []
    thumbnails = []

    for name, label, fn in configs:
        print(f"측정 중: {name} {label} ...")
        filtered_stack = []
        times = []
        for f, mm in zip(frames, mm_frames):
            t0 = time.perf_counter()
            result_mm = fn(f, mm)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            filtered_stack.append(result_mm)

        mean_mm = np.median(np.stack(filtered_stack, axis=0), axis=0)

        patch = mean_mm[roi_cy - FLAT_ROI_HALF:roi_cy + FLAT_ROI_HALF, roi_cx - FLAT_ROI_HALF:roi_cx + FLAT_ROI_HALF]
        valid = patch[patch > 0]
        flat_std = float(valid.std()) if valid.size > 0 else None

        diameter_mm = edge_diameter_mm(mean_mm, screw_center, mm_per_pixel) if (screw_center and mm_per_pixel) else None

        result = {
            "technique": name,
            "param": label,
            "flat_noise_std_mm": flat_std,
            "flat_roi_center": [roi_cy, roi_cx],
            "screw_center": list(screw_center) if screw_center else None,
            "screw_diameter_est_mm": diameter_mm,
            "screw_diameter_error_mm": (diameter_mm - SCREW_DIAMETER_MM) if diameter_mm else None,
            "proc_time_ms": float(np.mean(times)),
        }
        results.append(result)
        thumbnails.append((f"{name}\n{label}", mean_mm, result))
        print(f"  std={flat_std:.3f}mm, diam={diameter_mm}, time={result['proc_time_ms']:.2f}ms")

    json_path = os.path.join(RESULTS_DIR, "3-3_spatial_result.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(results, jf, ensure_ascii=False, indent=2)
    print(f"결과 저장: {json_path}")

    save_contact_sheet(thumbnails)


def save_contact_sheet(thumbnails, crop_half=CROP_HALF, roi_half=FLAT_ROI_HALF, edge_half=EDGE_SEARCH_HALF):
    """기법별 1행: 평탄면 crop(실측 영역 사각형 표시) + 나사머리 crop(측정 라인 표시)를 나란히 배치."""
    n = len(thumbnails)
    fig, axes = plt.subplots(n, 1, figsize=(6, n * 2.6))
    axes = np.atleast_1d(axes)

    for ax, (label, mean_mm, result) in zip(axes, thumbnails):
        h, w = mean_mm.shape
        cy, cx = result["flat_roi_center"]
        y0, y1 = max(0, cy - crop_half), min(h, cy + crop_half)
        x0, x1 = max(0, cx - crop_half), min(w, cx + crop_half)
        flat_crop = mean_mm[y0:y1, x0:x1]

        screw = result.get("screw_center")
        if screw:
            scy, scx = screw
            sy0, sy1 = max(0, scy - crop_half), min(h, scy + crop_half)
            sx0, sx1 = max(0, scx - crop_half), min(w, scx + crop_half)
            screw_crop = mean_mm[sy0:sy1, sx0:sx1]
        else:
            screw_crop = np.zeros_like(flat_crop)

        pad_w = crop_half * 2
        pad_h = crop_half * 2
        flat_pad = np.zeros((pad_h, pad_w), dtype=np.float32)
        flat_pad[:flat_crop.shape[0], :flat_crop.shape[1]] = flat_crop
        screw_pad = np.zeros((pad_h, pad_w), dtype=np.float32)
        screw_pad[:screw_crop.shape[0], :screw_crop.shape[1]] = screw_crop

        divider = np.full((pad_h, 4), 320.0, dtype=np.float32)
        combined = np.hstack([flat_pad, divider, screw_pad])

        ax.imshow(combined, cmap="turbo", vmin=200, vmax=320)

        # 평탄면 crop 안에서 실제 측정(roi_half) 영역을 사각형으로 표시
        rel_cy, rel_cx = cy - y0, cx - x0
        rect = patches.Rectangle(
            (rel_cx - roi_half, rel_cy - roi_half), roi_half * 2, roi_half * 2,
            linewidth=1.5, edgecolor="white", facecolor="none",
        )
        ax.add_patch(rect)

        # 나사머리 crop 안에서 라인 프로파일로 지름을 측정한 가로선 표시
        if screw:
            line_y = scy - sy0
            line_x0 = pad_w + 4 + max(0, (scx - sx0) - edge_half)
            line_x1 = pad_w + 4 + (scx - sx0) + edge_half
            ax.plot([line_x0, line_x1], [line_y, line_y], color="white", linewidth=1)

        std = result["flat_noise_std_mm"]
        t = result["proc_time_ms"]
        diam = result["screw_diameter_est_mm"]
        diam_txt = f"{diam:.1f}mm" if diam else "N/A"
        ax.set_title(f"{label}   std={std:.3f}mm   t={t:.1f}ms   diam={diam_txt}", fontsize=9, loc="left")
        ax.axis("off")

    fig.suptitle("Spatial filter 비교 (좌: 평탄면 crop [흰 사각형=실측 24x24 ROI], 우: 나사머리 crop [흰 선=지름 측정 라인])")
    fig.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "3-3_spatial_contact_sheet.png")
    fig.savefig(png_path, dpi=130)
    print(f"콘택트시트 저장: {png_path}")


if __name__ == "__main__":
    main()
