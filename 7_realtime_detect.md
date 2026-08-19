# 7. 실시간 삽입 불량 검출

## 목적
카메라 고정 + 시편 이동 전제로, 정지 감지 시 자동으로 5/6번 판정 파이프라인을 실행하는 실시간 버전.

## 배경
정적(1샷) 파이프라인은 GitHub `v1-static` 태그로 백업 완료. 3인 리뷰로 설계 검증(CPU 추론 41ms 실측).

## 진행 방식
1. `detection_core.py`로 검출/mm환산/평면보정 로직 통합 (5/6/7번 공용, 3중 복붙 방지)
2. 리팩터링 전후 출력 동일성 회귀 테스트 (`test_detection_core_regression.py`)
3. `7_realtime_detect.py` 상태머신(WAITING→SETTLING→MEASURING→RESULT_SHOWN) 구현
4. `calibrate_motion_threshold.py`로 모션 임계값을 감이 아니라 실측(max-gap)으로 산출

## 상태
실카메라로 MEASURING 경로 1차 검증(정적 6번과 동일 품질, 5.3초). 원격 제약으로 시편 이동 불가 —
모션 임계값 실측·WAITING↔SETTLING 전환은 미검증.

---

### 공용 모듈 분리 &nbsp;&nbsp;`검토 완료` (회귀 테스트로 동일 출력 확인)
- [x] `detection_core.py` 생성 — `5_px_to_mm.py`의 검출/build_instance/평면-호모그래피 보정 로직 이전
- [x] `5_px_to_mm.py`, `6_insertion_check.py`가 `detection_core`를 import하도록 변경 (기존 `import_module("5_px_to_mm")` 방식 대신 직접 import)
- [x] `4-2_hole_detection_experiments.py`도 새 모듈 참조로 수정 (안 하면 깨지는 걸 발견해 반영)
- [x] `6_insertion_check.py`의 판정 로직을 `classify_insertion()` 함수로 추출 (7번이 그대로 재사용)
- [x] 리팩터링 전/후 출력 비교 회귀 테스트 작성 및 통과 (`test_detection_core_regression.py`, 합성 depth + 실제 held-out 컬러 이미지 사용)

**결론**: 5/6번의 실제 동작(main() 결과)은 리팩터링 전후로 완전히 동일함을 회귀 테스트로 확인. 카메라 없이도 검증 가능한 부분은 여기까지 끝냄.

---

### 실시간 상태머신 &nbsp;&nbsp;`검토중` (MEASURING 실검증 완료, 모션감지는 원격 제약으로 미검증)
- [x] WAITING/SETTLING/MEASURING/RESULT_SHOWN 상태머신 구현 (`7_realtime_detect.py`)
- [x] depth 프레임간 ROI 평균 절대차로 모션 감지 (color diff 대신 - 조명 변화에 안 흔들림)
- [x] MEASURING에서 15프레임 버스트 2회 자기검증 → 불일치 시 기존 검증된 30프레임으로 재측정
- [x] YOLO CPU 추론 41ms/색상검출 7ms 실측 (카메라 없이 기존 학습 이미지로 벤치마크 가능했음)
- [x] `--once` 옵션 추가(첫 측정 후 자동 종료) + 결과 json/png 저장 - 원격 환경(GUI에 q 입력 불가)에서도 검증 가능하게 함
- [x] 실카메라 1차 실행: WAITING→SETTLING→MEASURING 정상 전이, 자기검증이 실제로 두 버스트 불일치(screw 4개/3개)를 잡아내 30프레임 폴백 정상 동작, 결과가 정적 6번과 동일 수준(나사 2정상/1틀어짐, 구멍 13개) - 5.3초 소요
- [x] MEASURING↔RESULT_SHOWN 전환 시 temporal filter 상태 흔들림으로 재측정 루프 도는 버그 발견 → `POST_MEASURE_COOLDOWN_FRAMES`(20프레임 유예) 추가로 수정(실측 검증은 아직 - 시편을 움직여봐야 확인됨)
- [ ] 원격이라 시편 이동 불가 - `calibrate_motion_threshold.py`의 "움직임" 단계가 실제로는 "정지"와 같아서 결과 폐기, 노이즈 상한(0.16mm)에 여유를 둔 잠정값(0.5mm) 사용 중
- [ ] WAITING↔SETTLING 전환, RESULT_SHOWN에서 모션 재감지 후 WAITING 복귀 - 둘 다 시편을 움직여야 검증 가능, 미검증
- [ ] SETTLE_FRAMES/DEBOUNCE_FRAMES 체감 튜닝

**결론**: MEASURING(검출→mm환산→틀어짐/덜박힘 판정) 경로는 실카메라로 1차 검증 완료 - 정적 파이프라인과 동등한 품질을 실시간 트리거로도 재현함을 확인. 다만 이 세션은 원격 제약으로 시편을 물리적으로 못 움직여서, 상태머신의 절반(모션 감지→정지 판정, 결과 표시→재트리거)은 여전히 미검증 상태.

**설계상 제약**: RealSense 파이프라인은 단일 스레드에서만 안전해서, MEASURING 중엔 라이브 프리뷰가 갱신되지 않고(마지막 화면 유지) 측정 끝나면 결과 화면으로 전환됨 — "완전히 끊김없는" 실시간은 아님. 정지-트리거 설계상 자연스러운 트레이드오프로 판단해 그대로 둠.

**보완 필요**: 시편을 물리적으로 움직일 수 있게 되면 (1) `calibrate_motion_threshold.py` 재실행해 진짜 임계값 산출, (2) 시편을 치웠다 다시 놓는 시나리오로 WAITING↔RESULT_SHOWN 전체 순환 검증, (3) `POST_MEASURE_COOLDOWN_FRAMES` 수정이 재측정 루프 버그를 실제로 잡는지 확인.
