"""
7번: 실시간 삽입 불량 검출 - 정지 감지 후 자동 캡처 방식
- 카메라는 고정, 시편은 사람이 옮겨가며 놓는다는 전제(사용자 확인 완료)
- depth 프레임간 변화량으로 "정지"를 감지해 그 순간에만 기존 검증된 30(->15)프레임 평균
  + YOLO + 색상검출 + mm/틀어짐/덜박힘 파이프라인(detection_core.py, 6_insertion_check.py)을
  그대로 실행 - 매 프레임 재추론하지 않음 (실사용 워크플로우에 맞고, 결과 신뢰도도 정적
  캡처 방식과 동일하게 유지됨)

상태머신: WAITING -> SETTLING -> MEASURING -> RESULT_SHOWN -> (모션 재감지) -> WAITING

주의(설계상 제약, 실카메라로 검증 전 반드시 읽을 것):
- RealSense pipeline은 스레드 하나에서만 wait_for_frames를 호출해야 안전함(librealsense가
  단일 파이프라인의 동시 다중 스레드 사용을 보장하지 않음). 그래서 MEASURING 구간은
  "라이브 프리뷰를 잠깐 멈추고(마지막 프레임 정지화면 위에 '측정 중' 표시) 측정이 끝나면
  재개"하는 방식으로 구현함 - 화면이 계속 갱신되는 진짜 논블로킹은 아님. 정지-트리거
  설계라 자연스러운 트레이드오프이고, 측정 자체가 0.5~2초 내로 끝나서 체감상 큰 문제는
  아닐 것으로 예상하지만 실측 확인 전까지는 추정치.
- MOTION_DIFF_THRESHOLD_MM / SETTLE_FRAMES / DEBOUNCE_FRAMES는 전부 시작값 - 실카메라
  앞에서 `scripts/calibrate_motion_threshold.py`(4_auto_label.py의 max-gap 분류 방식 재사용)로
  재조정 필요.
"""

import os
import sys
import time
import json
import argparse
from enum import Enum, auto
from functools import partial

import numpy as np
import cv2

print = partial(print, flush=True)  # 원격/백그라운드 실행 시 버퍼링으로 로그 유실 방지

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_preprocessing import build_pipeline, build_filters, apply_filters, capture_averaged_depth  # noqa: E402
import detection_core as dc  # noqa: E402
from importlib import import_module  # noqa: E402

insertion = import_module("6_insertion_check")  # 파일명이 숫자로 시작해 import 문으로 바로 못 씀

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- 시작값 (실카메라 튜닝 전) -----------------------------------------------
ROI_FRACTION = 0.7          # 화면 중앙 이 비율만 모션 판정에 사용 (배경 흔들림 무시)
# 실측(calibrate_motion_threshold.py, 원격이라 시편을 못 움직인 채 측정 - 두 단계가 사실상 다
# 정지 상태였음): 완전 정지 시 센서 노이즈 diff가 최대 0.16mm(평균 0.12mm)로 확인됨.
# max-gap 산출값(0.07mm)은 두 단계가 실제로는 구분 안 돼서 폐기 - 대신 노이즈 상한(0.16mm)에
# 3배 이상 여유를 둔 값을 잠정 사용. 진짜 움직임 vs 정지 비교는 시편을 물리적으로 움직일 수
# 있을 때 calibrate_motion_threshold.py 재실행해서 갱신할 것.
MOTION_DIFF_THRESHOLD_MM = 0.5
SETTLE_FRAMES = 15          # ~0.5초 @ 30fps 연속 정지 확인
DEBOUNCE_FRAMES = 15        # RESULT_SHOWN에서 재트리거되려면 이만큼 연속으로 모션 감지돼야 함
MEASURE_N_FRAMES_FAST = 15
MEASURE_N_FRAMES_FALLBACK = 30
STUD_HOLE_COUNT_TOLERANCE = 1  # 두 버스트 stud_hole 개수 차이가 이 이하면 "일치"로 간주
# MEASURING(capture_averaged_depth, 자체 프레임 루프)에서 RESULT_SHOWN(grab_filtered_depth_mm,
# 다른 호출 패턴)으로 돌아올 때 같은 temporal filter 객체의 내부 상태가 순간적으로 흔들려
# 첫 diff가 과대하게 나올 수 있음(실측으로 재측정 루프가 도는 게 확인돼 추가) - 복귀 후
# 이 프레임 수만큼은 diff를 구해도 debounce 카운터에 반영하지 않고 버림(필터 재안정화 유예).
POST_MEASURE_COOLDOWN_FRAMES = 20


class State(Enum):
    WAITING = auto()
    SETTLING = auto()
    MEASURING = auto()
    RESULT_SHOWN = auto()


def roi_slice(h, w, fraction=ROI_FRACTION):
    ry, rx = int(h * (1 - fraction) / 2), int(w * (1 - fraction) / 2)
    return slice(ry, h - ry), slice(rx, w - rx)


def grab_filtered_depth_mm(pipeline, align, filters, depth_scale):
    """WAITING/SETTLING용 - 한 프레임만 캡처+필터링(스트리밍 temporal filter는 매 호출 상태 누적).
    capture_averaged_depth(다프레임 평균)보다 가벼움 - 모션 감지엔 이 정도로 충분."""
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None
    filtered = apply_filters(depth_frame, filters)
    depth_m = np.asanyarray(filtered.get_data()).astype(np.float32) * depth_scale
    depth_mm = depth_m * 1000.0
    color_img = np.asanyarray(color_frame.get_data())
    return depth_mm, color_img


def run_measurement(pipeline, align, filters, depth_scale, intr, baseline):
    """정지 확정 시 실행되는 검증된 파이프라인. 15프레임 버스트 두 번을 비교해 자기검증
    (A의 제안: 노이즈로 인한 단발성 오검출을 잡아냄 - 방법 자체의 체계적 오류는 못 잡지만,
    실시간이라 이 정도 이중확인은 거의 공짜로 됨). 불일치 시 검증된 30프레임 설정으로 재시도."""
    def one_pass(n_frames):
        depth_mm, color_img = capture_averaged_depth(pipeline, align, filters, depth_scale, n_frames=n_frames)
        fx = intr["fx"]
        screw_dets = dc.detect_screw_heads_by_color(color_img)
        stud_holes = dc.detect_stud_holes(color_img, depth_mm, intr)
        return depth_mm, color_img, fx, screw_dets, stud_holes

    depth_mm, color_img, fx, screw_dets, stud_holes = one_pass(MEASURE_N_FRAMES_FAST)
    depth_mm2, color_img2, fx2, screw_dets2, stud_holes2 = one_pass(MEASURE_N_FRAMES_FAST)

    consistent = (
        abs(len(stud_holes) - len(stud_holes2)) <= STUD_HOLE_COUNT_TOLERANCE
        and len(screw_dets) == len(screw_dets2)
    )
    if not consistent:
        print(f"  두 버스트 불일치 (stud={len(stud_holes)}/{len(stud_holes2)}, "
              f"screw={len(screw_dets)}/{len(screw_dets2)}) - 30프레임으로 재측정")
        depth_mm, color_img, fx, screw_dets, stud_holes = one_pass(MEASURE_N_FRAMES_FALLBACK)

    results, baseline, threshold = insertion.classify_insertion(
        screw_dets, stud_holes, depth_mm, fx, baseline)
    return color_img, results, stud_holes, baseline, threshold, consistent


def render_result(color_img, results, stud_holes, status_text=None):
    vis = color_img.copy()
    for hole in stud_holes:
        hx, hy = hole["center_px"]
        hr_px = int(hole["diameter_px"] / 2)
        cv2.circle(vis, (int(hx), int(hy)), hr_px, (0, 255, 255), 2)
        cv2.putText(vis, f"hole {hole['diameter_mm']}mm", (int(hx) - 35, int(hy) + hr_px + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
    for inst in results:
        cx, cy = inst["center_px"]
        r_px = int(inst["diameter_px"] / 2)
        color = (0, 0, 255) if inst["final_status"] != "정상" else (0, 255, 0)
        cv2.circle(vis, (int(cx), int(cy)), r_px, color, 2)
        label = f"{inst['final_status']} prot={inst['protrusion_mm']}mm"
        cv2.putText(vis, label, (int(cx) - 50, int(cy) - r_px - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    if status_text:
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(vis, status_text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true",
                         help="첫 측정 완료(RESULT_SHOWN 진입) 즉시 결과 저장 후 종료 - 원격/헤드리스 검증용"
                              "(GUI 창에 q를 못 누르는 상황 대비)")
    args = parser.parse_args()

    pipeline, align, depth_scale = build_pipeline()
    filters = build_filters()
    intr = dc.get_color_intrinsics(pipeline)

    baseline = insertion.load_baseline()

    state = State.WAITING
    prev_depth_mm = None
    settle_counter = 0
    debounce_counter = 0
    cooldown_counter = 0
    result_vis = None

    print("7번 실시간 검출 시작 - q로 종료" + (" (--once: 첫 측정 후 자동 종료)" if args.once else ""))
    print(f"모션 임계값(시작값)={MOTION_DIFF_THRESHOLD_MM}mm, 정지확인={SETTLE_FRAMES}프레임")

    try:
        while True:
            if state in (State.WAITING, State.SETTLING):
                depth_mm, color_img = grab_filtered_depth_mm(pipeline, align, filters, depth_scale)
                if depth_mm is None:
                    continue

                ry, rx = roi_slice(*depth_mm.shape)
                roi = depth_mm[ry, rx]
                if prev_depth_mm is not None:
                    prev_roi = prev_depth_mm[ry, rx]
                    valid = (roi > 0) & (prev_roi > 0)
                    diff = float(np.mean(np.abs(roi[valid] - prev_roi[valid]))) if valid.sum() else 999.0
                else:
                    diff = 999.0
                prev_depth_mm = depth_mm

                if diff < MOTION_DIFF_THRESHOLD_MM:
                    settle_counter += 1
                    state = State.SETTLING
                else:
                    settle_counter = 0
                    state = State.WAITING

                status = f"[{state.name}] diff={diff:.1f}mm settle={settle_counter}/{SETTLE_FRAMES}"
                preview = color_img.copy()
                cv2.rectangle(preview, (0, 0), (preview.shape[1], 30), (0, 0, 0), -1)
                cv2.putText(preview, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow("realtime_detect", preview)

                if settle_counter >= SETTLE_FRAMES:
                    state = State.MEASURING
                    settle_counter = 0

            elif state == State.MEASURING:
                # 안전상 이유(파이프라인 단일 스레드 제약, 파일 docstring 참고)로 측정 중엔
                # 새 프레임을 안 당기고 측정이 끝난 뒤 결과 화면으로 바로 전환함.
                print("정지 확정 - 측정 시작")
                t0 = time.time()
                color_img, results, stud_holes, baseline, threshold, consistent = run_measurement(
                    pipeline, align, filters, depth_scale, intr, baseline)
                elapsed = time.time() - t0
                insertion.save_baseline(baseline)
                tag = "일치(빠른측정)" if consistent else "재측정(30프레임)"
                status_text = f"측정 완료 ({elapsed:.1f}s, {tag}) - 움직이면 다음 측정"
                result_vis = render_result(color_img, results, stud_holes, status_text)
                cv2.imshow("realtime_detect", result_vis)
                cv2.waitKey(1)
                print(f"  결과: screw={len(results)}개, stud_hole={len(stud_holes)}개, {elapsed:.2f}초")

                json_path = os.path.join(RESULTS_DIR, "7_realtime_result.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump({"screw_results": results, "stud_holes": stud_holes,
                               "consistent_fast_pass": consistent, "elapsed_sec": round(elapsed, 2)},
                              f, ensure_ascii=False, indent=2)
                png_path = os.path.join(RESULTS_DIR, "7_realtime_result.png")
                cv2.imwrite(png_path, result_vis)
                print(f"  결과 저장: {json_path}, {png_path}")

                if args.once:
                    print("--once 지정됨 - 종료")
                    break

                state = State.RESULT_SHOWN
                prev_depth_mm = None  # 재개 시 첫 프레임은 diff 기준 없음(999로 시작)
                cooldown_counter = POST_MEASURE_COOLDOWN_FRAMES  # MEASURING 직후 temporal filter 재안정화 유예

            elif state == State.RESULT_SHOWN:
                cv2.imshow("realtime_detect", result_vis)
                depth_mm, _ = grab_filtered_depth_mm(pipeline, align, filters, depth_scale)
                if depth_mm is not None and prev_depth_mm is not None:
                    ry, rx = roi_slice(*depth_mm.shape)
                    roi, prev_roi = depth_mm[ry, rx], prev_depth_mm[ry, rx]
                    valid = (roi > 0) & (prev_roi > 0)
                    diff = float(np.mean(np.abs(roi[valid] - prev_roi[valid]))) if valid.sum() else 0.0
                    if cooldown_counter > 0:
                        cooldown_counter -= 1  # 재안정화 유예 구간 - diff는 구하되 판정엔 반영 안 함
                    elif diff >= MOTION_DIFF_THRESHOLD_MM:
                        debounce_counter += 1
                    else:
                        debounce_counter = 0
                    if debounce_counter >= DEBOUNCE_FRAMES:
                        print("모션 재감지 - 대기 상태로 복귀")
                        state = State.WAITING
                        debounce_counter = 0
                prev_depth_mm = depth_mm

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
