"""
4번 세그멘테이션 - YOLOv8n-seg 전이학습
- 데이터 14장(screw_head/stud_hole 2클래스)뿐이라 nano 모델 + 강한 증강으로 오버피팅 방지
- 나사머리/스터드홀이 원형(회전 대칭)이라 degrees 증강을 크게 줘서 각도 다양성 보완
- train/val을 dataset/train.txt(11장) / val.txt(3장)로 진짜 분리 (기존엔 train=val 동일 14장이라 검증 지표가 암기 확인에 불과했음)
- freeze=10: COCO 사전학습 backbone을 고정하고 head만 학습 - 11장짜리 초소량 데이터로 backbone까지 전부 풀면
  오버피팅/파괴적 망각 위험이 커서 head만 파인튜닝
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
        freeze=10,          # backbone 고정, head만 학습 (11장짜리 train set 오버피팅 방지)
        degrees=180,        # 원형 객체라 회전 증강 최대로 (회전해도 같은 모양)
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.02, hsv_s=0.5, hsv_v=0.4,  # 조명/노출 다양성 보완 (캡처마다 밝기 달랐음)
        translate=0.1,
        scale=0.3,
        mosaic=1.0,
        copy_paste=0.5,             # screw_head 인스턴스가 3개뿐 - 다른 이미지 인스턴스를 합성해 등장 위치/배경 맥락 다양화
        copy_paste_mode="mixup",    # 같은 이미지 내 플립이 아니라 배치 내 다른 이미지에서 붙여넣기
        project=os.path.join(_ROOT, "runs"),
        name="screw_seg",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
