"""
detection_core.py 회귀 테스트 - 리팩터링/수정이 검출 로직의 동작을 조용히 바꾸지 않았는지 확인.
카메라 없이 합성 depth(기울어진 평면 + 노이즈)와 실제 held-out 컬러 이미지로 돌려서
scripts/_regression_fixtures/pre_refactor_baseline.json과 바이트 단위로 비교.

기준선(baseline) 갱신이 필요할 때(의도적으로 로직을 바꿨을 때)만 --update-baseline로 재생성.
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detection_core as dc  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "_regression_fixtures", "pre_refactor_baseline.json")
TEST_IMAGE = os.path.join(_ROOT, "captures", "snapshot_color_hd.png")


def build_synthetic_scene():
    img = cv2.imread(TEST_IMAGE)
    if img is None:
        raise FileNotFoundError(f"테스트 이미지 없음: {TEST_IMAGE}")
    h, w = img.shape[:2]
    rng = np.random.default_rng(42)  # 고정 시드 - 매 실행 동일 입력 보장
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    depth_mm = (300.0 + (xs - w / 2) * 0.02 + (ys - h / 2) * 0.01
                + rng.normal(0, 0.5, size=(h, w))).astype(np.float64)
    depth_mm = np.clip(depth_mm, 1, None)
    intr = {"fx": 640.0, "fy": 640.0, "ppx": w / 2, "ppy": h / 2}
    return img, depth_mm, intr


def run_pipeline():
    img, depth_mm, intr = build_synthetic_scene()

    screw_dets = dc.detect_screw_heads_by_color(img)
    screw_summary = [{"aspect_ratio": d["aspect_ratio"], "status": d["status"],
                       "area": int(d["mask"].sum())} for d in screw_dets]

    instances = []
    for det in screw_dets:
        inst = dc.build_instance(det["mask"], "screw_head", 1.0, depth_mm, intr["fx"],
                                  aspect_ratio=det["aspect_ratio"], insertion_status=det["status"])
        if inst:
            instances.append(inst)

    stud_holes = dc.detect_stud_holes(img, depth_mm, intr)

    return {"screw_summary": screw_summary, "screw_instances": instances, "stud_holes": stud_holes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-baseline", action="store_true",
                         help="의도적으로 검출 로직을 바꿨을 때만 사용 - 기준선을 지금 결과로 덮어씀")
    args = parser.parse_args()

    result = run_pipeline()

    if args.update_baseline:
        os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
        with open(FIXTURE_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"기준선 갱신: {FIXTURE_PATH}")
        return

    if not os.path.exists(FIXTURE_PATH):
        print(f"기준선 없음: {FIXTURE_PATH} - 먼저 --update-baseline로 생성하세요")
        raise SystemExit(1)

    with open(FIXTURE_PATH, encoding="utf-8") as f:
        baseline = json.load(f)

    def canon(obj):
        return json.dumps(obj, sort_keys=True)

    checks = {
        "screw_summary": canon(baseline["screw_summary"]) == canon(result["screw_summary"]),
        "screw_instances": canon(baseline["screw_instances"]) == canon(result["screw_instances"]),
        "stud_holes": canon(baseline["stud_holes"]) == canon(result["stud_holes"]),
    }
    for name, ok in checks.items():
        print(f"{name}: {'일치' if ok else '!! 불일치 !!'}")

    if not all(checks.values()):
        print("\n=== 회귀 테스트 실패: detection_core.py 동작이 기준선과 달라짐 ===")
        print("의도한 변경이면 --update-baseline로 기준선을 갱신하세요.")
        raise SystemExit(1)
    print("\n=== 회귀 테스트 통과 ===")


if __name__ == "__main__":
    main()
