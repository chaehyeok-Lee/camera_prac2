# 4. Segmentation으로 나사머리 영역학습 및 추론

## 목적
나사머리 영역을 픽셀 단위로 분할해 위치/형태를 정확히 추출한다.

## 전제
시편 3개뿐이라 데이터가 극소 — 직접 만든 원형 라벨링 스크립트(클릭 2번=중심+반지름)로 라벨링 부담을 줄이고, YOLOv8-seg 전이학습 + 회전 중심 증강으로 소량 데이터 대응.

## 진행 방식
1. 캡처: RGB 프레임을 시편별로 여러 각도/위치에서 수동 캡처
2. 라벨링: 원형 클릭 2번(중심→반지름)으로 폴리곤 근사, YOLO-seg 포맷 저장. 클래스 2종(screw_head/stud_hole) — 6번 과제에서 나사머리 중심 vs 스터드홀 중심 비교가 필요해 빈 구멍도 별도 클래스로 라벨링
3. 학습: YOLOv8-seg 전이학습 (COCO pretrained 시작)
4. 추론: 나사머리/스터드홀 마스크 생성, 5/6번 과제 입력으로 연결

## 상태
진행중 (3차 학습(copy_paste 증강) 완료 — held-out에서 놓치던 screw_head를 conf 0.21로 검출 성공, confusion matrix도 정상화. 정식 검증은 specimen3 재캡처 후 필요)

---

### 데이터 준비 &nbsp;&nbsp;`검토 완료` (14장 2클래스 라벨링 완료, 육안 검증 통과)
- [x] 캡처 도구: `scripts/4_capture_dataset.py` (SPACE=캡처, n=다음 시편, 3장마다 이동 리마인더)
- [x] 라벨링 도구: `scripts/4_label_tool.py` (원형 클릭 2번, YOLO-seg 폴리곤 20각형 근사 저장, 1/2키로 screw_head/stud_hole 클래스 전환)
- [x] save/load 라운드트립 단위 테스트 통과 (2클래스 포함, 카메라 없이 검증)
- [x] `dataset/data.yaml` 생성 (screw_head, stud_hole 2클래스)
- [x] 데이터셋을 실제 라벨 있는 14장으로 축소 확정 (specimen1_001~010, specimen2_001~004) — 나머지 30장/빈 라벨 삭제
- [x] screw_head 라벨링 14장 완료 (장당 3개 인스턴스)
- [x] stud_hole 라벨링 14장 완료 (장당 13개 인스턴스, 전체 16홀 전부 라벨링)

**결론**: 14장 전부 screw_head 3개 + stud_hole 13개(=16홀)로 완전히 일관 — 렌더링해서 육안 확인, 배경 오탐/미탐 없음. 1차 라벨링 완료, 학습 단계로 진행 가능.

**보완 필요**: 14장으로 확정되면서 각도 다양성도 줄어듦(specimen1/2 초반 그룹 위주) — 학습 성능 보고 필요시 specimen3 쪽 다양한 각도 이미지 다시 캡처해 추가 예정.

**참고**: Hough circle 기반 자동 라벨링(`scripts/4_auto_label.py`) 시도했으나 배경(모니터/케이블/손) 오탐 + 구멍 미탐지가 많아 사용자 요청으로 되돌림 — 수동 라벨링만 진행.

**수정**: `4_label_tool.py`에 경계 클리핑 버그 수정 — 원이 프레임 경계에 걸리면 저장 시 반달 모양으로 왜곡되던 문제를 빨간 원 표시+콘솔 경고로 예방 (현재 14장엔 해당 없음, specimen3 재캡처 대비).

---

### 학습 &nbsp;&nbsp;`검토중` (3차 학습 완료, held-out screw_head 검출 성공 — specimen3 재캡처 전엔 정식 검증 아님)
- [x] `scripts/4_train_yolo.py` — yolov8n-seg, epochs=150(조기종료 94), batch=4, degrees=180 등 원형 객체용 증강
- [x] `dataset/data.yaml` 경로 버그 수정 (`path: .`가 dataset/ 아닌 프로젝트 루트로 잘못 해석되던 문제, 절대경로로 교체)
- [x] val 지표 확인: Box mAP50 0.945, Mask mAP50 0.859 (단, train=val 동일 14장이라 낙관적 수치)
- [x] 마스크 크기 검증: 마스크 픽셀수가 박스 면적의 78~90% — 원이 박스에 내접하는 이론값(~78.5%)과 일치, 마스크 형태 정상
- [x] held-out 이미지(`captures/snapshot_color_hd.png`, 학습에 전혀 없던 사진)로 실제 일반화 테스트
- [x] (2차) 3인 리뷰로 발견된 문제 수정: `dataset/train.txt`(11장)/`val.txt`(3장)로 진짜 train/val 분리, `freeze=10`(backbone 고정)로 재학습
- [x] (2차) 재학습 결과: Box mAP50 0.728, Mask mAP50 0.547 (진짜 held-out 3장 기준 — 1차의 0.945/0.859는 암기 수치였음이 확인됨)
- [x] (2차) v1(구)-v2(신) 모델을 동일 held-out 이미지로 비교
- [x] (3차) `conf=0.01`로 재확인 — 놓친 screw_head 위치(697,565)에서 conf 0.14~0.20 신호가 실제로 존재함을 확인 (완전 미인식이 아니라 threshold 문제 일부 포함)
- [x] (3차) `4_train_yolo.py`에 `copy_paste=0.5`(mixup 모드) 증강 추가해 재학습 — screw_head 인스턴스 3개를 다른 이미지에 합성해 등장 맥락 다양화
- [x] (3차) 재학습 결과: Box mAP50 0.97, Mask mAP50 0.85 (val 3장 기준) — confusion matrix도 screw_head 0.89, stud_hole 0.97 정탐, background 오탐 거의 해소
- [x] (3차) held-out 재검증: conf 0.15 기준 screw_head가 (696,571)에서 conf 0.215로 정확히 검출됨 — v1/v2가 놓치던 그 나사

**결론**: stud_hole은 held-out에서도 13개 대부분 정확히 검출. 하지만 화면에 뚜렷이 보이는 screw_head 1개를 완전히 놓침(박스조차 안 그려짐) — screw_head 학습 인스턴스가 14장 내내 동일한 3개가 반복(specimen1/2는 저다양성 그룹)되어 일반화가 약한 것으로 판단.

**2차 결론**: train/val 실분리+freeze로 val 지표는 정직해졌고 screw_head 검출 개수는 1개→2개로 소폭 개선. 하지만 held-out 이미지에서 가장 뚜렷하게 보이는 스크류(화면 하단 금속 나사)는 v1/v2 둘 다 여전히 놓침 — 학습 설정 개선만으로는 근본 해결 안 됨.

**3차 결론**: `copy_paste` 증강으로 held-out에서 그 나사가 처음으로 검출됨(conf 0.215) — val 지표와 confusion matrix도 크게 개선. 다만 held-out 검증 이미지가 1장뿐이고 진짜 val set도 3장뿐이라 통계적으로 약함 — "고쳐졌다"보다는 "개선 신호를 확인했다" 정도로 봐야 함. `5_px_to_mm.py`엔 클래스별 confidence threshold(screw_head=0.15, stud_hole=0.28)를 적용해 이 개선이 실제 파이프라인에 반영되게 함.

**보완 필요**: specimen3(다양한 각도) 재캡처는 여전히 최우선 — copy_paste는 기존 3개 인스턴스의 "배치 다양성"만 늘린 것이라 진짜 형태/조명 다양성 문제는 남아있음. held-out도 3~5장 이상으로 늘려 이번 개선이 우연이 아닌지 검증 필요. 배경 negative 샘플도 여전히 없어 오탐 정량 검증 불가.
