"""
7번: 실시간 삽입 불량 검출 - 정지 감지 후 자동 캡처 방식 + 이동 중 라이브 미리보기
- 카메라는 고정, 시편은 사람이 옮겨가며 놓는다는 전제(사용자 확인 완료)
- 정식 측정(mm/틀어짐/덜박힘, baseline 누적)은 depth 프레임간 변화량으로 "정지"를 감지한
  순간에만 기존 검증된 30(->15)프레임 평균 + YOLO + 색상검출 파이프라인(detection_core.py,
  6_insertion_check.py)을 그대로 실행 - 정확도는 정적 캡처 방식과 동일하게 유지됨
- 시편이 움직이는 중(WAITING/SETTLING)에도 단일 프레임 기준 라이브 검출 오버레이를 보여줌
  (LIVE_DETECT_EVERY_N_FRAMES마다 갱신) - mm 값은 참고용, baseline엔 반영 안 됨

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
# WAITING/SETTLING(시편이 움직이는 중)에도 라이브 검출 오버레이를 보여주기 위한 설정.
# YOLO+평면보정 한 번에 ~60~120ms 걸려서 매 프레임 돌리면 모션감지 루프(diff 계산)가
# 느려져 SETTLE_FRAMES 타이밍이 틀어짐 - 그래서 모션 diff는 매 프레임 계산하되, 무거운
# 검출은 이 프레임 수마다 한 번만 갱신(그 사이엔 마지막 결과를 그대로 화면에 유지).
LIVE_DETECT_EVERY_N_FRAMES = 5


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
    def one_pass(n_frames, warmup):
        # capture_averaged_depth 기본 warmup=10은 정적 스크립트(pipeline.start() 직후 콜드스타트)
        # 기준 - 실시간은 WAITING/SETTLING 내내 스트리밍 중이라 노출/필터가 이미 안정 상태.
        # 이 사이클의 첫 버스트만 시편 교체 직후 노출 재적응을 위해 소량 워밍업을 두고,
        # 같은 사이클의 재측정 버스트는 워밍업 생략(3회 x 10프레임=1초 가까이 절약됨).
        t_capture0 = time.time()
        depth_mm, color_img = capture_averaged_depth(
            pipeline, align, filters, depth_scale, n_frames=n_frames, warmup=warmup)
        t_capture = time.time() - t_capture0
        fx = intr["fx"]
        t_det0 = time.time()
        screw_dets = dc.detect_screw_heads_by_color(color_img)
        stud_holes = dc.detect_stud_holes(color_img, depth_mm, intr)
        t_det = time.time() - t_det0
        print(f"    [측정] n_frames={n_frames} warmup={warmup} -> capture={t_capture:.2f}s detect={t_det:.2f}s")
        return depth_mm, color_img, fx, screw_dets, stud_holes

    depth_mm, color_img, fx, screw_dets, stud_holes = one_pass(MEASURE_N_FRAMES_FAST, warmup=3)
    depth_mm2, color_img2, fx2, screw_dets2, stud_holes2 = one_pass(MEASURE_N_FRAMES_FAST, warmup=0)

    consistent = (
        abs(len(stud_holes) - len(stud_holes2)) <= STUD_HOLE_COUNT_TOLERANCE
        and len(screw_dets) == len(screw_dets2)
    )
    if not consistent:
        print(f"  두 버스트 불일치 (stud={len(stud_holes)}/{len(stud_holes2)}, "
              f"screw={len(screw_dets)}/{len(screw_dets2)}) - 30프레임으로 재측정")
        depth_mm, color_img, fx, screw_dets, stud_holes = one_pass(MEASURE_N_FRAMES_FALLBACK, warmup=0)

    results, baseline, threshold = insertion.classify_insertion(
        screw_dets, stud_holes, depth_mm, fx, baseline)
    return color_img, results, stud_holes, baseline, threshold, consistent


def detect_live(color_img, depth_mm, intr):
    """WAITING/SETTLING(시편 이동 중)용 - 단일 프레임 기준 라이브 검출.
    MEASURING의 15/30프레임 평균보다 노이즈가 커서 mm 값은 참고용일 뿐 - 정식 측정치는
    정지 후 MEASURING에서만 신뢰. classify_insertion(baseline 누적/판정)은 호출하지 않음 -
    초당 여러 번 돌면 protrusion 표본이 순식간에 오염됨."""
    fx = intr["fx"]
    screw_dets = dc.detect_screw_heads_by_color(color_img)
    screw_instances = []
    for det in screw_dets:
        inst = dc.build_instance(det["mask"], "screw_head", 1.0, depth_mm, fx,
                                  aspect_ratio=det["aspect_ratio"], insertion_status=det["status"])
        if inst:
            screw_instances.append(inst)
    stud_holes = dc.detect_stud_holes(color_img, depth_mm, intr, debug_label=None)
    return screw_instances, stud_holes


def render_live_overlay(color_img, screw_instances, stud_holes, status):
    vis = color_img.copy()
    for hole in stud_holes:
        hx, hy = hole["center_px"]
        hr_px = int(hole["diameter_px"] / 2)
        cv2.circle(vis, (int(hx), int(hy)), hr_px, (0, 255, 255), 1)
    for inst in screw_instances:
        cx, cy = inst["center_px"]
        r_px = int(inst["diameter_px"] / 2)
        is_tilted = inst.get("insertion_status") == "틀어짐"
        color = (0, 0, 255) if is_tilted else (0, 255, 0)
        cv2.circle(vis, (int(cx), int(cy)), r_px, color, 2)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(vis, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(vis, "라이브 미리보기(참고용, 정밀 측정 아님)", (10, vis.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    return vis


def run_live_demo(pipeline, align, filters, depth_scale, intr, duration_sec):
    """--live-demo-sec 전용 - 정지판정 없이 라이브 검출 갱신 프레임을 그대로 디스크에 저장.
    '이동 중 라이브 검출'이 실제로 동작하는지 스크린샷으로 증명하는 용도(순간적으로만 뜨고
    사라지는 cv2 창 대신 결과물로 남김)."""
    demo_dir = os.path.join(RESULTS_DIR, "8_live_demo")
    os.makedirs(demo_dir, exist_ok=True)
    print(f"라이브 데모 {duration_sec}초 시작 - {demo_dir}에 갱신 프레임 저장")

    frame_idx = 0
    saved = 0
    t_start = time.time()
    while time.time() - t_start < duration_sec:
        depth_mm, color_img = grab_filtered_depth_mm(pipeline, align, filters, depth_scale)
        if depth_mm is None:
            continue
        frame_idx += 1
        if frame_idx % LIVE_DETECT_EVERY_N_FRAMES == 0:
            t_det0 = time.time()
            screw_instances, stud_holes = detect_live(color_img, depth_mm, intr)
            t_det = time.time() - t_det0
            status = f"[LIVE DEMO] t={time.time()-t_start:.1f}s detect={t_det*1000:.0f}ms"
            vis = render_live_overlay(color_img, screw_instances, stud_holes, status)
            cv2.imshow("realtime_detect", vis)
            cv2.waitKey(1)
            path = os.path.join(demo_dir, f"frame_{saved:03d}.png")
            cv2.imwrite(path, vis)
            saved += 1
            print(f"  저장: {path} (screw={len(screw_instances)}, hole={len(stud_holes)}, {t_det*1000:.0f}ms)")
    print(f"라이브 데모 종료 - {saved}장 저장, 총 {frame_idx}프레임 처리")


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
    parser.add_argument("--live-demo-sec", type=float, default=0,
                         help="지정하면 정지판정/MEASURING 없이 WAITING 라이브 오버레이만 이 초만큼 "
                              "돌리면서 매 갱신 프레임을 results/8_live_demo/에 저장 - 라이브 검출이 "
                              "실제로 동작하는지 스크린샷으로 증명하기 위한 디버그 모드")
    args = parser.parse_args()

    pipeline, align, depth_scale = build_pipeline()
    filters = build_filters()
    intr = dc.get_color_intrinsics(pipeline)

    # YOLO 모델 워밍업: 첫 실제 추론에 초기화 비용(~1.7초, 실측)이 붙는 걸 확인함 - 이걸
    # 최초 측정(사용자가 기다리는 순간) 대신 시작 시점(대기 상태 진입 전)으로 옮겨서 숨김.
    print("모델 워밍업 중...")
    t_warm = time.time()
    dc.get_model().predict(np.zeros((720, 1280, 3), dtype=np.uint8), verbose=False)
    print(f"  완료 ({time.time() - t_warm:.2f}s)")

    if args.live_demo_sec > 0:
        run_live_demo(pipeline, align, filters, depth_scale, intr, args.live_demo_sec)
        pipeline.stop()
        cv2.destroyAllWindows()
        return

    baseline = insertion.load_baseline()

    state = State.WAITING
    prev_depth_mm = None
    settle_counter = 0
    debounce_counter = 0
    cooldown_counter = 0
    result_vis = None
    live_frame_counter = 0
    live_screw_instances, live_stud_holes = [], []

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

                # 시편이 움직이는 중에도 라이브 검출 - 무거운 검출은 N프레임마다만 갱신
                # (매 프레임 돌리면 diff 계산 루프가 느려져 SETTLE_FRAMES 타이밍이 틀어짐).
                live_frame_counter += 1
                if live_frame_counter >= LIVE_DETECT_EVERY_N_FRAMES:
                    live_frame_counter = 0
                    live_screw_instances, live_stud_holes = detect_live(color_img, depth_mm, intr)
                preview = render_live_overlay(color_img, live_screw_instances, live_stud_holes, status)
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
                        live_screw_instances, live_stud_holes = [], []  # 이전 시편의 잔상 표시 방지
                        live_frame_counter = 0
                prev_depth_mm = depth_mm

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
