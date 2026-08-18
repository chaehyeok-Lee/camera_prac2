"""
RealSense depth frame 전처리 파이프라인
- 라이브 스트리밍 캡처 + 필터링 + 다중 프레임 평균화
- 저장된 .bag 재생 처리도 동일 함수로 가능
- 저장된 raw npy/png(순수 배열)만 있는 경우를 위한 OpenCV 기반 대체 경로 포함
"""

import numpy as np
import cv2
import pyrealsense2 as rs

# 작업 거리 250mm 기준 clip 범위. 시편/배경 실측 후 조정할 것.
CLIP_MIN_M = 0.20
CLIP_MAX_M = 0.32


def build_pipeline(width=1280, height=720, fps=30, bag_path=None):
    # 실험3-1: 1280x720이 나사머리 픽셀수 최다(19.9px) + magnitude=1(decimation 미적용) 기준 노이즈도 최저
    """라이브 캡처 또는 .bag 재생용 pipeline 생성. align 객체도 함께 반환."""
    pipeline = rs.pipeline()
    config = rs.config()

    if bag_path:
        config.enable_device_from_file(bag_path, repeat_playback=False)
    else:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # depth 픽셀값 -> meter 변환 계수

    align = rs.align(rs.stream.color)
    return pipeline, align, depth_scale


def build_filters():
    """RealSense 권장 순서로 적용할 필터 세트. 250mm 근거리에 맞춘 파라미터.
    decimation은 실험3-1 결론(1280x720 기준 magnitude=1이 magnitude=2보다 노이즈 낮고
    픽셀수도 그대로 최다)에 따라 미적용."""
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_smooth_alpha, 0.75)  # 실험3-2: alpha 0.25/0.5/0.75/1.0 중 std 최소
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    spatial.set_option(rs.option.holes_fill, 0)

    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.1)  # 실험4: alpha 낮을수록 steady 노이즈 최소, lag=0(정지 부품이라 무관)
    temporal.set_option(rs.option.filter_smooth_delta, 20)

    hole_filling = rs.hole_filling_filter(2)  # 2 = nearest_from_around (전경 값으로 채움)

    threshold = rs.threshold_filter()
    threshold.set_option(rs.option.min_distance, CLIP_MIN_M)
    threshold.set_option(rs.option.max_distance, CLIP_MAX_M)

    return {
        "to_disparity": depth_to_disparity,
        "spatial": spatial,
        "temporal": temporal,
        "to_depth": disparity_to_depth,
        "hole_filling": hole_filling,
        "threshold": threshold,
    }


def apply_filters(depth_frame, filters):
    f = depth_frame
    f = filters["threshold"].process(f)
    f = filters["to_disparity"].process(f)
    f = filters["spatial"].process(f)
    f = filters["temporal"].process(f)
    f = filters["to_depth"].process(f)
    f = filters["hole_filling"].process(f)
    return f


def capture_averaged_depth(pipeline, align, filters, depth_scale, n_frames=30, warmup=10):
    """N프레임 캡처 -> 필터링 -> 픽셀별 median으로 합성. mm 단위 uint16 배열 반환."""
    # 자동노출/필터 워밍업
    for _ in range(warmup):
        pipeline.wait_for_frames()

    depth_stack = []
    color_img = None

    for _ in range(n_frames):
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        filtered = apply_filters(depth_frame, filters)
        depth_m = np.asanyarray(filtered.get_data()).astype(np.float32) * depth_scale
        depth_stack.append(depth_m)
        color_img = np.asanyarray(color_frame.get_data())

    depth_median_m = np.median(np.stack(depth_stack, axis=0), axis=0)
    depth_mm = (depth_median_m * 1000.0).astype(np.uint16)
    return depth_mm, color_img


def to_segmentation_input(depth_mm, min_mm=None, max_mm=None, colorize=True):
    """정밀 depth(mm) -> 0~255 정규화. 세그멘테이션 모델 입력/시각화용."""
    min_mm = min_mm if min_mm is not None else CLIP_MIN_M * 1000
    max_mm = max_mm if max_mm is not None else CLIP_MAX_M * 1000

    clipped = np.clip(depth_mm.astype(np.float32), min_mm, max_mm)
    normalized = ((clipped - min_mm) / (max_mm - min_mm) * 255).astype(np.uint8)

    if colorize:
        return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    return normalized


def process_raw_array(depth_mm, min_mm=None, max_mm=None):
    """
    rs.frame 없이 이미 저장된 순수 depth 배열(mm, uint16)만 있을 때의 대체 경로.
    RealSense 필터는 frame 메타데이터가 필요해 여기서는 못 쓰므로 OpenCV로 유사 효과.
    """
    min_mm = min_mm if min_mm is not None else CLIP_MIN_M * 1000
    max_mm = max_mm if max_mm is not None else CLIP_MAX_M * 1000

    depth = depth_mm.astype(np.float32)

    # 1) 범위 밖(배경/무효값) 마스킹
    invalid = (depth < min_mm) | (depth > max_mm) | (depth == 0)
    depth[invalid] = 0

    # 2) 홀(0값) 채우기: 인접 유효 픽셀로 inpaint
    mask = (depth == 0).astype(np.uint8)
    depth_8u_scale = 65535.0 / max(max_mm, 1)
    depth_for_inpaint = np.clip(depth * depth_8u_scale / 1000, 0, 65535).astype(np.uint16)
    # cv2.inpaint는 8U/32F만 지원 -> 32F로 처리
    depth_filled = cv2.inpaint(
        (depth / max_mm * 255).astype(np.uint8), mask, 3, cv2.INPAINT_NS
    ).astype(np.float32) / 255 * max_mm

    # 3) 엣지 보존 스무딩 (나사머리 경계 보존 목적)
    depth_smoothed = cv2.bilateralFilter(depth_filled.astype(np.float32), d=5, sigmaColor=15, sigmaSpace=15)

    return depth_smoothed.astype(np.uint16)


if __name__ == "__main__":
    # 사용 예시: 시편 1개, 250mm 거리 고정 후 30프레임 캡처
    pipeline, align, depth_scale = build_pipeline()
    filters = build_filters()

    try:
        depth_mm, color_img = capture_averaged_depth(pipeline, align, filters, depth_scale, n_frames=30)
    finally:
        pipeline.stop()

    # Track A: 정밀 depth 보존 (5번 px-to-mm 계산용)
    np.save("specimen1_depth_mm.npy", depth_mm)

    # Track B: 세그멘테이션 입력/시각화용
    seg_input = to_segmentation_input(depth_mm)
    cv2.imwrite("specimen1_depth_vis.png", seg_input)
    cv2.imwrite("specimen1_color.png", color_img)

    print("depth range (mm):", depth_mm[depth_mm > 0].min(), depth_mm.max())
