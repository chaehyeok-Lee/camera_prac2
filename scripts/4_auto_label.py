"""
4번 세그멘테이션 라벨 자동 생성 (Hough circle 검출 + 밝기 기반 클래스 분류)
- dataset/images/*.png 전부에 대해 원 후보를 찾고, 4_label_tool.py와 동일한 포맷(YOLO-seg 폴리곤)으로
  dataset/labels/*.txt를 미리 채워둠 -> 사용자는 4_label_tool.py로 열어서 틀린 것만 수정
- 밝기 분류: 이미지별로 검출된 원들의 내부 평균 밝기를 모아 가장 큰 간격(gap)을 기준으로 2그룹으로 나눔
  (절대 임계값이 아니라 이미지마다 적응적 - 캡처마다 노출/거리가 달라서)
- 기존 라벨 파일이 있으면 건드리지 않음 (이미 사람이 손댄 걸 덮어쓰지 않기 위함)
"""

import os
import glob
import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(_ROOT, "dataset", "images")
LABELS_DIR = os.path.join(_ROOT, "dataset", "labels")
os.makedirs(LABELS_DIR, exist_ok=True)

POLYGON_POINTS = 20
MIN_RADIUS, MAX_RADIUS = 8, 45
MIN_DIST = 25


def circle_to_polygon(cx, cy, r, w, h, n=POLYGON_POINTS):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = np.clip(cx + r * np.cos(angles), 0, w - 1) / w
    ys = np.clip(cy + r * np.sin(angles), 0, h - 1) / h
    return list(zip(xs, ys))


def detect_circles(gray):
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=MIN_DIST,
        param1=80, param2=22, minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS,
    )
    if circles is None:
        return []
    return [(float(x), float(y), float(r)) for x, y, r in np.round(circles[0]).astype(float)]


def mean_interior_intensity(gray, cx, cy, r):
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), max(1, int(r * 0.6)), 255, -1)  # 테두리 그림자 피하려고 반지름의 60%만
    vals = gray[mask > 0]
    return float(vals.mean()) if vals.size else 0.0


def classify_by_max_gap(intensities):
    """밝기값들을 정렬 후 가장 큰 간격을 경계로 2그룹(어두움=stud_hole, 밝음=screw_head) 분류."""
    if len(intensities) < 2:
        # 원이 하나뿐이면 밝기 판단 기준이 없음 - 임의로 중간값(128) 기준
        return [0 if v >= 128 else 1 for v in intensities]
    order = np.argsort(intensities)
    sorted_vals = np.array(intensities)[order]
    gaps = np.diff(sorted_vals)
    split_at = np.argmax(gaps)  # sorted_vals[split_at]와 [split_at+1] 사이가 가장 큰 간격
    threshold = (sorted_vals[split_at] + sorted_vals[split_at + 1]) / 2
    return [0 if v >= threshold else 1 for v in intensities]  # 0=screw_head(밝음), 1=stud_hole(어두움)


def process_image(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = detect_circles(gray)
    if not circles:
        return [], 0

    intensities = [mean_interior_intensity(gray, cx, cy, r) for cx, cy, r in circles]
    classes = classify_by_max_gap(intensities)

    labeled = [(cx, cy, r, cls) for (cx, cy, r), cls in zip(circles, classes)]
    return labeled, len(circles)


def save_labels(label_path, circles, w, h):
    with open(label_path, "w", encoding="utf-8") as f:
        for cx, cy, r, cls in circles:
            poly = circle_to_polygon(cx, cy, r, w, h)
            coord_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
            f.write(f"{cls} {coord_str}\n")


def main():
    image_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")) +
                          glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    if not image_paths:
        print(f"이미지가 없습니다: {IMAGES_DIR}")
        return

    n_new, n_skip, n_empty = 0, 0, 0
    for path in image_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        label_path = os.path.join(LABELS_DIR, f"{name}.txt")

        if os.path.exists(label_path):
            n_skip += 1
            continue

        img = cv2.imread(path)
        h, w = img.shape[:2]
        circles, n_detected = process_image(path)

        if not circles:
            n_empty += 1
            print(f"{name}: 검출 실패 (0개) - 수동 라벨링 필요")
            continue

        save_labels(label_path, circles, w, h)
        n_screw = sum(1 for c in circles if c[3] == 0)
        n_hole = sum(1 for c in circles if c[3] == 1)
        print(f"{name}: screw_head={n_screw} stud_hole={n_hole}")
        n_new += 1

    print(f"\n완료: 새로 생성 {n_new}장, 검출실패(수동필요) {n_empty}장, 기존라벨 있어 스킵 {n_skip}장")
    print("4_label_tool.py로 열어서 틀린 것만 수정하세요 (자동 검출이라 오탐/미탐 있을 수 있음)")


if __name__ == "__main__":
    main()
