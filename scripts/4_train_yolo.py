"""
4번 세그멘테이션 - YOLOv8n-seg 전이학습
- 데이터 14장(screw_head/stud_hole 2클래스)뿐이라 nano 모델 + 강한 증강으로 오버피팅 방지
- 나사머리/스터드홀이 원형(회전 대칭)이라 degrees 증강을 크게 줘서 각도 다양성 보완
"""

import os
from ultralytics import YOLO

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(_ROOT, "dataset", "data.yaml")


def main():
    model = YOLO("yolov8n-seg.pt")  # COCO 사전학습 nano 모델에서 전이학습

    model.train(
        data=DATA_YAML,
        epochs=150,
        imgsz=640,
        batch=4,           # 이미지 14장뿐이라 작은 배치
        patience=30,        # 소량 데이터라 early stopping 여유 있게
        degrees=180,        # 원형 객체라 회전 증강 최대로 (회전해도 같은 모양)
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.02, hsv_s=0.5, hsv_v=0.4,  # 조명/노출 다양성 보완 (캡처마다 밝기 달랐음)
        translate=0.1,
        scale=0.3,
        mosaic=1.0,
        project=os.path.join(_ROOT, "runs"),
        name="screw_seg",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
