# -*- coding: utf-8 -*-
"""tread_v2 라벨링 fixture(scripts/_regression_fixtures/tread_v2_labeled_01)로 카메라 없이
검출/판정 정확도를 평가. 자동화 루프(오탐/오분류 없어질 때까지 파라미터 조정)의 채점 스크립트.

성공 기준(exit code 0):
- ground_truth.json의 나사 4개 전부, 가장 가까운 검출 결과의 final_status가 true_status와 일치
- known_false_positive_regions 근처(match_radius_px 이내)에 검출된 screw_head가 하나도 없음
- (참고용) stud_hole 개수/오탐도 같이 출력하지만 현재는 pass/fail에 반영 안 함(정답 미확보)

사용법: python scripts/eval_tread_v2_fixture.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detection_core as dc  # noqa: E402

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_regression_fixtures", "tread_v2_labeled_01")


def load_fixture():
    import cv2
    color_img = cv2.imread(os.path.join(FIXTURE_DIR, "color.png"))
    depth_mm = np.load(os.path.join(FIXTURE_DIR, "depth_mm.npy"))
    with open(os.path.join(FIXTURE_DIR, "intrinsics.json")) as f:
        intr = json.load(f)
    with open(os.path.join(FIXTURE_DIR, "ground_truth.json"), encoding="utf-8") as f:
        gt = json.load(f)
    return color_img, depth_mm, intr, gt


def nearest(center_px, candidates, radius_px):
    best, best_dist = None, radius_px
    for c in candidates:
        d = float(np.hypot(c["center_px"][0] - center_px[0], c["center_px"][1] - center_px[1]))
        if d < best_dist:
            best, best_dist = c, d
    return best, best_dist


def main():
    dc.set_profile("tread_v2")
    color_img, depth_mm, intr, gt = load_fixture()
    fx = intr["fx"]
    radius = gt.get("match_radius_px", 60)

    screw_dets = dc.detect_screw_heads_by_color(color_img)
    screw_instances = []
    for det in screw_dets:
        inst = dc.build_instance(det["mask"], "screw_head", 1.0, depth_mm, fx,
                                  aspect_ratio=det["aspect_ratio"], insertion_status=det["status"])
        if inst:
            screw_instances.append(inst)

    insertion = __import__("importlib").import_module("6_insertion_check")
    results, stud_holes, _baseline, _threshold = insertion.classify_insertion(
        screw_dets, dc.detect_stud_holes(color_img, depth_mm, intr), depth_mm, fx, [])

    ok = True
    print(f"검출된 screw_head 후보: {len(screw_dets)}개 / 최종 판정: {len(results)}개 / stud_hole: {len(stud_holes)}개\n")

    print("=== 나사 4개 정답 대조 ===")
    matched_ids = set()
    for gt_screw in gt["screws"]:
        match, dist = nearest(gt_screw["approx_center_px"], results, radius)
        true_status = gt_screw["true_status"]
        if match is None:
            print(f"  [FAIL] {gt_screw.get('note', gt_screw['approx_center_px'])}: "
                  f"정답={true_status} / 검출 안됨(반경 {radius}px 이내 없음)")
            ok = False
            continue
        matched_ids.add(id(match))
        got = match["final_status"]
        status = "OK" if got == true_status else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  [{status}] {gt_screw.get('note', gt_screw['approx_center_px'])}: "
              f"정답={true_status} / 검출={got} (거리={dist:.0f}px, center={match['center_px']})")

    print("\n=== 배경 오탐 구역 대조 ===")
    for fp in gt.get("known_false_positive_regions", []):
        match, dist = nearest(fp["approx_center_px"], results, radius)
        if match is not None:
            print(f"  [FAIL] {fp['reason']}: 여전히 검출됨 (center={match['center_px']}, 거리={dist:.0f}px)")
            ok = False
        else:
            print(f"  [OK] {fp['reason']}: 검출 안됨(정상)")

    extra = [r for r in results if id(r) not in matched_ids]
    if extra:
        print(f"\n=== 정답에 없는 추가 검출 {len(extra)}개 (미분류 오탐 후보) ===")
        for r in extra:
            print(f"  center={r['center_px']} final_status={r['final_status']} ar={r['aspect_ratio']}")

    print(f"\n{'=== 전체 통과 ===' if ok else '=== 실패 항목 있음 ==='}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
