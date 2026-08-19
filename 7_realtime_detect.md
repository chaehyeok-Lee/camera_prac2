# 7. 실시간 삽입 불량 검출

## 목적
카메라 고정 + 시편 이동 전제로, 정지 감지 시 자동으로 5/6번 판정 파이프라인을 실행하는 실시간 버전.

## 배경
정적(1샷) 파이프라인은 GitHub `v1-static` 태그로 백업 완료. 3인 리뷰로 설계 검증(CPU 추론 41ms 실측,
정지→결과 약 0.55초 추정) — 정확성은 실카메라 검증이 필요해 계획만으로 못 올림, 코드부터 완성.

## 진행 방식
1. `detection_core.py`로 검출/mm환산/평면보정 로직 통합 (5/6/7번 공용, 3중 복붙 방지)
2. 리팩터링 전후 출력 동일성 회귀 테스트 (`test_detection_core_regression.py`)
3. `7_realtime_detect.py` 상태머신(WAITING→SETTLING→MEASURING→RESULT_SHOWN) 구현
4. `calibrate_motion_threshold.py`로 모션 임계값을 감이 아니라 실측(max-gap)으로 산출

## 상태
코드 작성 완료, 회귀 테스트 통과. **실카메라 미검증** — 모션 임계값 튜닝과 MEASURING 실제 소요시간
확인이 남음.

---

### 공용 모듈 분리 &nbsp;&nbsp;`검토 완료` (회귀 테스트로 동일 출력 확인)
- [x] `detection_core.py` 생성 — `5_px_to_mm.py`의 검출/build_instance/평면-호모그래피 보정 로직 이전
- [x] `5_px_to_mm.py`, `6_insertion_check.py`가 `detection_core`를 import하도록 변경 (기존 `import_module("5_px_to_mm")` 방식 대신 직접 import)
- [x] `4-2_hole_detection_experiments.py`도 새 모듈 참조로 수정 (안 하면 깨지는 걸 발견해 반영)
- [x] `6_insertion_check.py`의 판정 로직을 `classify_insertion()` 함수로 추출 (7번이 그대로 재사용)
- [x] 리팩터링 전/후 출력 비교 회귀 테스트 작성 및 통과 (`test_detection_core_regression.py`, 합성 depth + 실제 held-out 컬러 이미지 사용)

**결론**: 5/6번의 실제 동작(main() 결과)은 리팩터링 전후로 완전히 동일함을 회귀 테스트로 확인. 카메라 없이도 검증 가능한 부분은 여기까지 끝냄.

---

### 실시간 상태머신 &nbsp;&nbsp;`검토중` (코드 작성 완료, 실카메라 미검증)
- [x] WAITING/SETTLING/MEASURING/RESULT_SHOWN 상태머신 구현 (`7_realtime_detect.py`)
- [x] depth 프레임간 ROI 평균 절대차로 모션 감지 (color diff 대신 - 조명 변화에 안 흔들림)
- [x] MEASURING에서 15프레임 버스트 2회 자기검증 → 불일치 시 기존 검증된 30프레임으로 재측정
- [x] YOLO CPU 추론 41ms/색상검출 7ms 실측 (카메라 없이 기존 학습 이미지로 벤치마크 가능했음)
- [ ] 실카메라로 모션 임계값(`MOTION_DIFF_THRESHOLD_MM`) 실측 캘리브레이션 (`calibrate_motion_threshold.py` 작성 완료, 실행은 카메라 필요)
- [ ] MEASURING 실제 소요시간 실측 (depth 캡처 I/O + 필터 오버헤드는 추정치, 순수 추론시간만 실측됨)
- [ ] SETTLE_FRAMES/DEBOUNCE_FRAMES 체감 튜닝

**결론**: 코드 경로는 완성됐고 회귀 테스트로 기반 로직(5/6번)이 안 깨졌음도 확인함. 다만 상태머신 자체(모션감지→정지→측정 전환)는 실카메라가 있어야만 검증 가능해서 여기까지가 코드만으로 할 수 있는 한계.

**설계상 제약**: RealSense 파이프라인은 단일 스레드에서만 안전해서, MEASURING 중엔 라이브 프리뷰가 갱신되지 않고(마지막 화면 유지) 측정 끝나면 결과 화면으로 전환됨 — "완전히 끊김없는" 실시간은 아님. 정지-트리거 설계상 자연스러운 트레이드오프로 판단해 그대로 둠.

**보완 필요**: `calibrate_motion_threshold.py` 실행해서 임계값 실측 → `7_realtime_detect.py` 상수 반영 → 여러 시편 위치로 반복 테스트해서 정확성 점수(계획 단계 85점)를 실측으로 끌어올리는 게 다음 단계.
