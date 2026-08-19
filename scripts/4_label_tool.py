"""
4번 세그멘테이션용 원형 라벨링 도구 (2클래스: screw_head, stud_hole)
- 나사머리/스터드홀 전부 원형이라 클릭 2번(중심 -> 반지름)으로 인스턴스 하나 라벨링
- 저장 포맷: YOLOv8-seg 폴리곤 txt (원을 N각형으로 근사) - dataset/labels/{이미지명}.txt
- 재실행 시 기존 라벨 자동 로드해서 이어서 작업 가능

조작법:
  좌클릭 1번째 = 중심, 2번째 = 반지름 확정 (그 사이엔 미리보기 원이 마우스 따라다님)
  1 = screw_head 클래스로 전환, 2 = stud_hole 클래스로 전환 (현재 클래스는 화면 상단에 표시)
  z = 마지막 원 취소
  n / p = 다음 / 이전 이미지 (자동 저장됨)
  q = 종료 (자동 저장됨)
"""

import os
import glob
import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(_ROOT, "dataset", "images")
LABELS_DIR = os.path.join(_ROOT, "dataset", "labels")
os.makedirs(LABELS_DIR, exist_ok=True)

CLASS_NAMES = {0: "screw_head", 1: "stud_hole"}
CLASS_COLORS = {0: (0, 255, 0), 1: (255, 200, 0)}  # 초록=screw_head, 하늘색=stud_hole
POLYGON_POINTS = 20  # 원을 근사할 폴리곤 꼭짓점 수


def circle_to_polygon(cx, cy, r, w, h, n=POLYGON_POINTS):
    """중심/반지름(픽셀) -> YOLO-seg 정규화 폴리곤 좌표 리스트."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = np.clip(cx + r * np.cos(angles), 0, w - 1) / w
    ys = np.clip(cy + r * np.sin(angles), 0, h - 1) / h
    return list(zip(xs, ys))


def load_labels(label_path, w, h):
    """저장된 YOLO-seg txt -> [(cx, cy, r, cls), ...] 픽셀 단위로 근사 복원."""
    circles = []
    if not os.path.exists(label_path):
        return circles
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            xs = np.array(coords[0::2]) * w
            ys = np.array(coords[1::2]) * h
            cx, cy = xs.mean(), ys.mean()
            r = np.mean(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2))
            circles.append((cx, cy, r, cls))
    return circles


def save_labels(label_path, circles, w, h):
    with open(label_path, "w", encoding="utf-8") as f:
        for cx, cy, r, cls in circles:
            poly = circle_to_polygon(cx, cy, r, w, h)
            coord_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
            f.write(f"{cls} {coord_str}\n")


class Labeler:
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.idx = 0
        self.circles = []       # 현재 이미지의 (cx, cy, r, cls) 리스트
        self.pending_center = None
        self.mouse_pos = (0, 0)
        self.active_class = 0
        self.load_current()

    def label_path(self):
        name = os.path.splitext(os.path.basename(self.image_paths[self.idx]))[0]
        return os.path.join(LABELS_DIR, f"{name}.txt")

    def load_current(self):
        img = cv2.imread(self.image_paths[self.idx])
        self.h, self.w = img.shape[:2]
        self.img = img
        self.circles = load_labels(self.label_path(), self.w, self.h)
        self.pending_center = None

    def save_current(self):
        save_labels(self.label_path(), self.circles, self.w, self.h)

    def mouse_cb(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.pending_center is None:
                self.pending_center = (x, y)
            else:
                cx, cy = self.pending_center
                r = float(np.hypot(x - cx, y - cy))
                if r >= 2:
                    self.circles.append((cx, cy, r, self.active_class))
                self.pending_center = None

    def undo(self):
        if self.circles:
            self.circles.pop()

    def render(self):
        vis = self.img.copy()
        for cx, cy, r, cls in self.circles:
            color = CLASS_COLORS.get(cls, (200, 200, 200))
            cv2.circle(vis, (int(cx), int(cy)), int(r), color, 2)
            cv2.circle(vis, (int(cx), int(cy)), 2, color, -1)
        if self.pending_center is not None:
            cx, cy = self.pending_center
            mx, my = self.mouse_pos
            r = int(np.hypot(mx - cx, my - cy))
            cv2.circle(vis, (int(cx), int(cy)), r, (0, 165, 255), 1)
            cv2.circle(vis, (int(cx), int(cy)), 2, (0, 165, 255), -1)

        active_name = CLASS_NAMES[self.active_class]
        active_color = CLASS_COLORS[self.active_class]
        counts = {c: sum(1 for _, _, _, cc in self.circles if cc == c) for c in CLASS_NAMES}
        status1 = (f"[{self.idx + 1}/{len(self.image_paths)}] {os.path.basename(self.image_paths[self.idx])}  "
                   f"screw_head={counts[0]} stud_hole={counts[1]}")
        status2 = (f"현재 클래스: {active_name}  (1=screw_head, 2=stud_hole, "
                   f"click=center->radius, z=undo, n/p=next/prev, q=quit)")
        cv2.putText(vis, status1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(vis, status2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, active_color, 1, cv2.LINE_AA)
        return vis

    def goto(self, delta):
        self.save_current()
        self.idx = max(0, min(len(self.image_paths) - 1, self.idx + delta))
        self.load_current()


def main():
    image_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")) +
                          glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    if not image_paths:
        print(f"이미지가 없습니다: {IMAGES_DIR} (먼저 4_capture_dataset.py로 캡처하세요)")
        return

    labeler = Labeler(image_paths)
    cv2.namedWindow("label")
    cv2.setMouseCallback("label", labeler.mouse_cb)

    while True:
        cv2.imshow("label", labeler.render())
        key = cv2.waitKey(20) & 0xFF
        if key == ord('z'):
            labeler.undo()
        elif key == ord('1'):
            labeler.active_class = 0
        elif key == ord('2'):
            labeler.active_class = 1
        elif key == ord('n'):
            labeler.goto(1)
        elif key == ord('p'):
            labeler.goto(-1)
        elif key == ord('q'):
            labeler.save_current()
            break

    cv2.destroyAllWindows()
    n_labeled = sum(1 for p in image_paths
                     if os.path.exists(os.path.join(LABELS_DIR,
                         os.path.splitext(os.path.basename(p))[0] + ".txt")))
    print(f"라벨링 완료: {n_labeled}/{len(image_paths)}장")


if __name__ == "__main__":
    main()
