"""
5/6/7번 공용 검출·mm환산·삽입판정 로직 (원래 5_px_to_mm.py에 있던 걸 분리).
- 3번 depth 파이프라인 + 4번 YOLO-seg 모델(runs/screw_seg/weights/best.pt) 결합
- 공식: 실제지름(mm) = 픽셀지름 x depth(mm) / fx  (핀홀 카메라 모델)
- 캘리퍼 실측값(screw_head=6mm, stud_hole=10mm)과 비교해 정확도 검증

7번(실시간)까지 이 로직을 세 번째로 복붙하게 되면 유지보수가 어려워져서 여기로 통합함 -
5_px_to_mm.py/6_insertion_check.py/7_realtime_detect.py는 전부 이 모듈을 import해서 씀.
"""

import os
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_ROOT, "runs", "screw_seg", "weights", "best.pt")

# 시편 종류별 파라미터 프로파일 - 시편이 바뀌면 캘리퍼 실측값/카메라 거리/나사 픽셀 크기 등이
# 전부 달라짐(다른 트레드 시편으로 시도했을 때 stud_hole 지름이 6~16mm로 제각각 나오고 나사
# 1개가 완전히 미검출된 걸 확인 - GROUND_TRUTH_MM 6/10mm을 그 시편에 그대로 적용해서 생긴
# 문제). set_profile()로 전환하면 아래 build_instance/detect_* 전부가 자동으로 반영함
# (모듈 전역을 재할당하는 방식이라 함수 시그니처를 안 건드림).
SPECIMEN_PROFILES = {
    "foam_panel_v1": {  # 기존 시편(specimen1/2/3) - 4~6번 개발에 쓴 원본
        "ground_truth_mm": {"screw_head": 6.0, "stud_hole": 10.0},  # 캘리퍼 실측값
        "depth_range_mm": (180, 400),  # confirm.md 측정거리(250mm) 기준
        "screw_area_px": (100, 4000),
        "screw_min_solidity": 0.75,
        "screw_tilt_aspect_ratio": 1.5,
        "screw_max_aspect_ratio": 2.5,
    },
    "tread_v2": {  # 2026-08-20 시도한 다른 트레드 시편 - 구멍이 90도로 꺾인 게 아니라
        # 오목(concave)하게 들어가 있어서 기존 방식의 캘리퍼 측정이 애매함(사용자 확인) ->
        # 캘리퍼 값 확보 전까지 ground_truth는 None으로 둠. build_instance가 gt=None이면
        # error_mm/error_pct를 그냥 건너뛰므로(기존 코드 경로 그대로) 에러 없이 동작함 -
        # 대신 diameter_mm(실측 mm 추정치) 자체는 그대로 나오니 그 값과 실제 구멍을 비교해
        # 정성적으로 판단하거나, 이후 다른 방식(예: 개구부 림 지름만 캘리퍼로 재기)으로 채울 것.
        "ground_truth_mm": {"screw_head": None, "stud_hole": None},
        "depth_range_mm": (180, 400),  # 실측 depth 258~272mm로 기존 범위 안이라 일단 재사용
        "screw_area_px": (100, 4000),  # 나사 1개 미검출 원인일 수 있음 - 재조정 필요(미확정)
        "screw_min_solidity": 0.75,
        "screw_tilt_aspect_ratio": 1.5,
        "screw_max_aspect_ratio": 2.5,
    },
}

CURRENT_PROFILE_NAME = None
GROUND_TRUTH_MM = None
SPECIMEN_DEPTH_RANGE_MM = None


def set_profile(name):
    """활성 시편 프로파일 전환 - GROUND_TRUTH_MM/SPECIMEN_DEPTH_RANGE_MM 모듈 전역을 갱신함.
    build_instance 등은 이 이름들을 함수 바디에서 매 호출마다 조회하므로(파이썬 전역 조회는
    호출 시점에 일어남) 재할당만으로 이후의 모든 호출에 자동 반영됨."""
    global CURRENT_PROFILE_NAME, GROUND_TRUTH_MM, SPECIMEN_DEPTH_RANGE_MM
    if name not in SPECIMEN_PROFILES:
        raise ValueError(f"알 수 없는 시편 프로파일: {name!r} (선택 가능: {list(SPECIMEN_PROFILES)})")
    CURRENT_PROFILE_NAME = name
    p = SPECIMEN_PROFILES[name]
    GROUND_TRUTH_MM = p["ground_truth_mm"]
    SPECIMEN_DEPTH_RANGE_MM = p["depth_range_mm"]
    print(f"[detection_core] 시편 프로파일 적용: {name}")


def current_profile():
    return SPECIMEN_PROFILES[CURRENT_PROFILE_NAME]


set_profile("foam_panel_v1")  # 기본값 - 기존 스크립트들이 인자 없이 그대로 써도 동작 동일

# stud_hole만 YOLO confidence threshold 적용 대상 (screw_head는 색상 기반 검출이라 conf 개념 없음).
# 0.28은 4-2 실험 스크립트가 "기존 방식"과 비교하는 baseline 값으로만 남겨둠(운영 미사용) -
# 실제 검출은 STUD_HOLE_CONF(0.15)를 씀.
CONF_THRESHOLD_BY_CLASS = {"stud_hole": 0.28}
STUD_HOLE_CONF = 0.15  # 4-2 실험(scripts/4-2_hole_detection_experiments.py)에서 재현율이
# 뚜렷이 좋아짐(9->12개, 같은 프레임에서 오탐 없이 확인) - 정확도는 평면보정으로 별도 확보

PLANE_FIT_SAMPLE_STEP = 8   # 전체 프레임에서 이 간격으로만 샘플링해 평면 피팅(속도 위해 서브샘플)
PLANE_FIT_MIN_POINTS = 200
CIRCLE_FIT_MIN_POINTS = 8       # 원 피팅에 필요한 최소 윤곽점 수(부족하면 피팅이 불안정)
CIRCLE_FIT_MIN_COVERAGE = 0.55  # 원 둘레의 이 비율(각도 기준) 이상 보여야 피팅을 신뢰
# (합성 마스크로 검증: coverage=0.61은 중심오차 1.3px, coverage=0.43은 5.6px까지 벌어짐 확인)
CIRCLE_FIT_MAX_RESIDUAL_PX = 2.5  # 피팅된 원에서 점들이 이 이상 벗어나면 원이 아닌 노이즈로 판단

_MODEL_CACHE = {}


def get_model():
    """YOLO 모델을 프로세스당 한 번만 로드(캐시). 실시간 루프에서 매 사이클 재로드 방지."""
    if "model" not in _MODEL_CACHE:
        _MODEL_CACHE["model"] = YOLO(MODEL_PATH)
    return _MODEL_CACHE["model"]


def get_color_fx(pipeline):
    profile = pipeline.get_active_profile()
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    return color_stream.get_intrinsics().fx


def get_color_intrinsics(pipeline):
    """fx/fy/ppx(주점)/ppy를 dict로 반환. 실측 확인: 이 색상 스트림은 왜곡계수(coeffs)가
    전부 0 - 즉 렌즈 왜곡은 없고, 핀홀 모델 그대로 써도 됨(아래 평면 피팅/호모그래피 보정에서
    rs2_deproject 대신 간단한 벡터화된 핀홀 역투영식을 쓰는 근거)."""
    profile = pipeline.get_active_profile()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    return {"fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy}


def mask_diameter_px(mask_bool):
    """마스크 픽셀 수 -> 등가원 지름(면적 기준, sqrt(4*area/pi))."""
    area = mask_bool.sum()
    if area == 0:
        return None
    return float(np.sqrt(4 * area / np.pi))


def mask_center(mask_bool):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def blob_aspect_ratio(mask_bool):
    """블롭의 2차 모멘트(공분산 행렬) 고유값 비율 -> 회전에 무관한 장축/단축 비율.
    축 정렬 bounding box의 fill_ratio는 블롭이 회전(기울어짐)해 있으면 실제로 원인데도
    낮게 나와 오탐 처리되는 문제가 있어서, 회전-불변인 모멘트 기반으로 대체."""
    m = cv2.moments(mask_bool.astype(np.uint8), binaryImage=True)
    area = m["m00"]
    if area == 0:
        return None
    mu20 = m["mu20"] / area
    mu02 = m["mu02"] / area
    mu11 = m["mu11"] / area
    cov = np.array([[mu20, mu11], [mu11, mu02]])
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 1e-6, None)
    return float(np.sqrt(eigvals[1]) / np.sqrt(eigvals[0]))  # >=1, 1에 가까울수록 원형


def detect_screw_heads_by_color(color_img, min_area_px=None, max_area_px=None,
                                 min_solidity=None, tilt_aspect_ratio=None, max_aspect_ratio=None):
    """금속 나사머리는 은색(밝음, 무채색) vs 무광 검은 배경 - 명도 대비가 커서
    학습 없이 밝기 임계값(Otsu, 이미지마다 자동 적응)만으로 검출.

    인자를 명시하지 않으면(None) 활성 시편 프로파일(set_profile)의 값을 씀 - 시편마다
    나사 픽셀 크기/형태 판정 기준이 다를 수 있어서(2026-08-20 다른 시편에서 나사 1개
    완전 미검출 확인, 원인 후보 중 하나).
    YOLO screw_head가 학습 데이터 부족으로 불안정한 것의 대안.

    나사가 기울어져 삽입되면(6번 '틀어짐' 케이스) 카메라 시점에서 원이 아니라 타원으로 보임
    -> 이걸 노이즈로 버리지 않고 장단축 비율(aspect_ratio)로 정상/틀어짐을 분류해서 같이 반환.
    solidity(블롭 면적/블롭 컨벡스헐 면적)로 케이블 하이라이트 같은 불규칙한 노이즈만 배제
    (타원은 solidity가 높게 유지되므로 축정렬 bbox fill_ratio보다 회전에 안전)."""
    p = current_profile()
    if min_area_px is None:
        min_area_px = p["screw_area_px"][0]
    if max_area_px is None:
        max_area_px = p["screw_area_px"][1]
    if min_solidity is None:
        min_solidity = p["screw_min_solidity"]
    if tilt_aspect_ratio is None:
        tilt_aspect_ratio = p["screw_tilt_aspect_ratio"]
    if max_aspect_ratio is None:
        max_aspect_ratio = p["screw_max_aspect_ratio"]

    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    _, bright_mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)

    results = []
    for i in range(1, n_labels):  # 0 = 배경
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area_px or area > max_area_px:
            continue
        blob = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(contours[0]))
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < min_solidity:  # 케이블 반사 등 불규칙한 모양 배제 (회전에 안전)
            continue

        aspect_ratio = blob_aspect_ratio(labels == i)
        if aspect_ratio is None:
            continue
        if aspect_ratio > max_aspect_ratio:
            # 실제 나사가 기울어져도 이 정도로 안 늘어남 - 케이블/반사 등 노이즈로 판단해 배제
            continue
        status = "틀어짐" if aspect_ratio >= tilt_aspect_ratio else "정상"
        results.append({"mask": labels == i, "aspect_ratio": round(aspect_ratio, 2), "status": status})
    return results


def deduplicate_instances(instances, dist_ratio=0.6):
    """같은 클래스의 두 인스턴스 중심이 (평균 지름 x dist_ratio)보다 가까우면 같은 구멍/나사로 보고
    confidence 높은 쪽만 남긴다. YOLO 자체 NMS(IoU 기반)로 못 거른 살짝 어긋난 중복 박스 대비용."""
    kept = []
    for inst in sorted(instances, key=lambda x: -x["confidence"]):
        cx, cy = inst["center_px"]
        is_dup = False
        for k in kept:
            if k["class"] != inst["class"]:
                continue
            kcx, kcy = k["center_px"]
            dist = np.hypot(cx - kcx, cy - kcy)
            avg_diam = (k["diameter_px"] + inst["diameter_px"]) / 2
            if dist < avg_diam * dist_ratio:
                is_dup = True
                break
        if not is_dup:
            kept.append(inst)
    n_removed = len(instances) - len(kept)
    if n_removed:
        print(f"중복 제거: {n_removed}개 (같은 구멍/나사로 판단해 confidence 낮은 쪽 제외)")
    return kept


def build_instance(mask_bool, cls_name, conf, depth_mm, fx, debug_label=None, **extra):
    """마스크(bool) + depth + fx -> mm 환산된 인스턴스 dict. YOLO 마스크/색상 검출 마스크 공용.
    extra: aspect_ratio/status(정상/틀어짐) 등 검출기별 부가 정보를 그대로 실어 보냄.
    debug_label: 지정하면 제외될 때마다 사유를 콘솔에 출력 (원인불명 상태로 조용히 버려지는 것 방지)."""
    center = mask_center(mask_bool)
    diam_px = mask_diameter_px(mask_bool)
    if center is None or diam_px is None:
        if debug_label:
            print(f"  [{debug_label}] 제외: 마스크가 비어있음")
        return None

    cx, cy = center
    depth_vals = depth_mm[mask_bool]
    depth_vals = depth_vals[depth_vals > 0]
    depth_source = "mask"
    if depth_vals.size == 0:
        # 금속 나사머리는 반사가 강해 IR 구조광이 제대로 안 돌아와 depth가 통째로
        # 무효(0)일 수 있음 - 마스크를 조금씩 팽창시켜 주변 유효 depth라도 확보.
        # (이 경우 나사 자체가 아니라 인접부 depth라 정확도는 떨어짐 - depth_source로 표시)
        kernel = np.ones((3, 3), np.uint8)
        dilated = mask_bool.astype(np.uint8)
        for _ in range(5):
            dilated = cv2.dilate(dilated, kernel, iterations=1)
            depth_vals = depth_mm[dilated.astype(bool)]
            depth_vals = depth_vals[depth_vals > 0]
            if depth_vals.size > 0:
                depth_source = "dilated_mask"
                break
    if depth_vals.size == 0:
        if debug_label:
            print(f"  [{debug_label}] 제외: 유효 depth 없음 (반사로 인한 홀 - 팽창해도 복구 안됨)")
        return None
    depth_at_instance = float(np.median(depth_vals))
    if not (SPECIMEN_DEPTH_RANGE_MM[0] <= depth_at_instance <= SPECIMEN_DEPTH_RANGE_MM[1]):
        if debug_label:
            print(f"  [{debug_label}] 제외: depth={depth_at_instance:.0f}mm가 예상범위 "
                  f"{SPECIMEN_DEPTH_RANGE_MM} 밖 (배경 오탐 추정)")
        return None  # 시편 거리 범위 밖 - 배경(벽/모니터 등) 오탐으로 판단해 제외

    diam_mm = diam_px * depth_at_instance / fx
    gt = GROUND_TRUTH_MM.get(cls_name)
    error_mm = (diam_mm - gt) if gt is not None else None
    error_pct = (error_mm / gt * 100) if gt else None

    inst = {
        "class": cls_name,
        "confidence": round(conf, 3),
        "center_px": [round(cx, 1), round(cy, 1)],
        "diameter_px": round(diam_px, 2),
        "depth_mm": round(depth_at_instance, 1),
        "depth_source": depth_source,
        "diameter_mm": round(diam_mm, 3),
        "ground_truth_mm": gt,
        "error_mm": round(error_mm, 3) if error_mm is not None else None,
        "error_pct": round(error_pct, 1) if error_pct is not None else None,
    }
    inst.update(extra)
    return inst


def fit_panel_plane_3d(depth_mm, intr, sample_step=PLANE_FIT_SAMPLE_STEP):
    """depth 프레임 전체에서 패널의 3D 평면(카메라 좌표계, aX+bY+cZ=1)을 최소자승으로 피팅.
    왜곡계수가 0으로 확인됐으므로(get_color_intrinsics 참고) rs2_deproject 대신 벡터화된
    핀홀 역투영식(X=(u-ppx)Z/fx, Y=(v-ppy)Z/fy)을 직접 씀 - numpy로 한 번에 처리돼 빠름.
    반환: (단위법선벡터 n(3,), 평면까지 수직거리 d0[mm]). 유효 표본 부족하면 (None, None)."""
    h, w = depth_mm.shape
    us = np.arange(0, w, sample_step)
    vs = np.arange(0, h, sample_step)
    grid_u, grid_v = np.meshgrid(us, vs)
    Z = depth_mm[grid_v, grid_u].astype(np.float64)
    valid = Z > 0
    if valid.sum() < PLANE_FIT_MIN_POINTS:
        return None, None

    U, V, Z = grid_u[valid].astype(np.float64), grid_v[valid].astype(np.float64), Z[valid]
    X = (U - intr["ppx"]) * Z / intr["fx"]
    Y = (V - intr["ppy"]) * Z / intr["fy"]

    A = np.column_stack([X, Y, Z])
    b = np.ones_like(X)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    norm = np.linalg.norm(sol)
    if norm < 1e-12:
        return None, None
    n = sol / norm
    d0 = 1.0 / norm
    return n, float(d0)


def build_rectification_homography(intr, normal):
    """패널 법선(normal)이 광축(0,0,1)과 나란해지도록 카메라를 제자리에서 회전시키는 것과
    동등한 호모그래피 H = K R K^-1 을 구성. 광학중심은 그대로 두고 시선 방향만 돌리는
    것이므로 평행이동 항 없이 회전만으로 충분(표준 fronto-parallel 정류 공식).
    H는 원본 이미지 좌표 -> '정면에서 본 것처럼' 보정된 가상 이미지 좌표로 매핑."""
    fx, fy, ppx, ppy = intr["fx"], intr["fy"], intr["ppx"], intr["ppy"]
    K = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]], dtype=np.float64)
    target = np.array([0.0, 0.0, 1.0])
    n = normal / np.linalg.norm(normal)
    axis = np.cross(n, target)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        R = np.eye(3)
    else:
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(n, target), -1.0, 1.0))
        R, _ = cv2.Rodrigues(axis * angle)
    H = K @ R @ np.linalg.inv(K)
    return H


def _fit_circle_lstsq(x, y):
    """점들에 최소자승으로 원 피팅 (x^2+y^2=2ax+2by+c 선형화). (cx,cy,r) 또는 실패 시 None."""
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a_coef, b_coef, c_coef = sol
    cx, cy = a_coef / 2, b_coef / 2
    r_sq = c_coef + cx ** 2 + cy ** 2
    if not np.isfinite(r_sq) or r_sq <= 0:
        return None
    return cx, cy, np.sqrt(r_sq)


def rectify_and_fit_circle(mask_bool, H, fx, d0, border_margin=3,
                            min_points=CIRCLE_FIT_MIN_POINTS, min_coverage=CIRCLE_FIT_MIN_COVERAGE,
                            max_residual_px=CIRCLE_FIT_MAX_RESIDUAL_PX):
    """마스크 윤곽선을 호모그래피 H로 '정면에서 본' 좌표계로 변환한 뒤 그 좌표계에서 원을
    피팅 - 가장자리에 잘렸든 안 잘렸든, 이미지 중심에서 멀든 가깝든 항상 같은 방식(연속적)으로
    처리됨 - '잘렸으면 A, 안 잘렸으면 B'식 이진분기 없음.

    이미지 경계에 붙은 점(절단면)은 원의 일부가 아니므로 호모그래피 변환 전에 미리 제외.
    반환값: dict(center_px(원본좌표), diameter_px, diameter_mm, rect_residual_px, rect_coverage,
    rect_n_points) 또는 신뢰 불가 시 None. diameter_mm은 보정된 반지름과 평면의 수직거리 d0로
    계산(보정된 가상 카메라는 패널과 정면으로 마주보므로 단순 핀홀식 그대로 적용 가능)."""
    h, w = mask_bool.shape
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)

    on_border = (
        (cnt[:, 0] <= border_margin) | (cnt[:, 0] >= w - 1 - border_margin) |
        (cnt[:, 1] <= border_margin) | (cnt[:, 1] >= h - 1 - border_margin)
    )
    pts = cnt[~on_border]
    if len(pts) < min_points:
        return None

    pts_rect = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    x, y = pts_rect[:, 0], pts_rect[:, 1]

    fit = _fit_circle_lstsq(x, y)
    if fit is None:
        return None
    cx, cy, r = fit

    dist = np.hypot(x - cx, y - cy)
    residual = float(np.sqrt(np.mean((dist - r) ** 2)))
    if residual > max_residual_px:
        return None

    angles = np.sort(np.arctan2(y - cy, x - cx))
    gaps = np.diff(np.concatenate([angles, angles[:1] + 2 * np.pi]))
    coverage = 1 - gaps.max() / (2 * np.pi)
    if coverage < min_coverage:
        return None

    # 보정된 좌표계 중심을 다시 원본 이미지 좌표로 되돌림 (시각화/거리매칭은 원본 좌표 기준)
    Hinv = np.linalg.inv(H)
    center_orig = cv2.perspectiveTransform(np.array([[[cx, cy]]], dtype=np.float64), Hinv)[0, 0]

    diam_mm = 2 * r * d0 / fx
    return {
        "center_px": [round(float(center_orig[0]), 1), round(float(center_orig[1]), 1)],
        "diameter_px": round(float(2 * r), 2),
        "diameter_mm": round(float(diam_mm), 3),
        "rect_residual_px": round(residual, 2),
        "rect_coverage": round(float(coverage), 2),
        "rect_n_points": int(len(pts)),
    }


def detect_stud_holes(color_img, depth_mm, intr, debug_label="stud_hole"):
    """YOLO-seg로 stud_hole 검출 -> build_instance 리스트. 5/6/7번 공용.
    intr: get_color_intrinsics()가 반환하는 dict({fx,fy,ppx,ppy}).

    conf=0.15(원래 0.28보다 낮음)로 재현율을 올림 (4-2 실험: 같은 프레임 기준 9->12개, 오탐
    증가는 육안상 없었음).

    측정 정확도는 원형도(circularity) 필터가 아니라 평면-호모그래피 보정 + 원 피팅으로 확보:
    패널이 카메라에 대해 기울어져 있어 광축(이미지 중심)에서 멀어질수록 원이 타원으로 찌그러져
    보이는 문제(렌즈 왜곡 아님 - 실측 확인함, get_color_intrinsics 참고)를 depth로 패널의 3D
    평면을 매 프레임 새로 피팅해 보정한다. 이미지 가장자리에 살짝 잘린 구멍도 같은 파이프라인
    으로 처리됨(별도 분기 없음) - 잘림 정도가 심해 피팅이 불안정해지는 경우만
    rectify_and_fit_circle 내부에서 자동으로 거름."""
    model = get_model()
    h_img, w_img = depth_mm.shape
    fx = intr["fx"]

    normal, d0 = fit_panel_plane_3d(depth_mm, intr)
    if normal is None:
        if debug_label:
            print(f"  [{debug_label}] 경고: 패널 평면 피팅 실패(유효 depth 부족) - stud_hole 검출 건너뜀")
        return []
    H = build_rectification_homography(intr, normal)

    results = model.predict(color_img, conf=STUD_HOLE_CONF, iou=0.5, verbose=False)
    r = results[0]
    instances = []
    if r.masks is not None:
        for i, cls_idx in enumerate(r.boxes.cls.tolist()):
            cls_name = model.names[int(cls_idx)]
            if cls_name != "stud_hole":
                continue
            conf = r.boxes.conf[i].item()
            mask = r.masks.data[i].cpu().numpy()
            mask_resized = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST) > 0.5

            fitted = rectify_and_fit_circle(mask_resized, H, fx, d0)
            if fitted is None:
                if debug_label:
                    print(f"  [{debug_label}] 제외: 원 피팅 신뢰불가 (노이즈/과도한 잘림)")
                continue

            inst = build_instance(mask_resized, cls_name, conf, depth_mm, fx, debug_label=debug_label)
            if inst is None:
                continue
            # 마스크/원본 기반 값(중심/지름/mm)을 평면보정 기반 값으로 덮어씀 - depth 유효성
            # 검사, ground_truth 오차 계산 등 build_instance의 나머지 로직은 그대로 재사용.
            inst["center_px"] = fitted["center_px"]
            inst["diameter_px"] = fitted["diameter_px"]
            diam_mm = fitted["diameter_mm"]
            inst["diameter_mm"] = diam_mm
            gt = GROUND_TRUTH_MM.get(cls_name)
            if gt is not None:
                inst["error_mm"] = round(diam_mm - gt, 3)
                inst["error_pct"] = round((diam_mm - gt) / gt * 100, 1)
            inst["rect_residual_px"] = fitted["rect_residual_px"]
            inst["rect_coverage"] = fitted["rect_coverage"]
            instances.append(inst)
    return instances
