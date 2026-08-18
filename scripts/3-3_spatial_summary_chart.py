"""
실험3(spatial filter) 결과를 더 읽기 쉬운 요약 그래프로 재생성.
- 기존 콘택트시트(21개 작은 crop)는 한눈에 비교하기 어려워서,
  1) 노이즈 std 기준 랭킹 막대그래프 (가장 직관적)
  2) std vs 처리시간 트레이드오프 산점도 (Pareto)
  두 가지로 대체/보완.
- 카메라 불필요, 기존 3-3_spatial_result.json만 사용.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"  # 로그축 지수(10^-4 등)의 minus glyph가 Malgun Gothic에 없어서 깨짐 방지

TIME_FLOOR_MS = 0.05  # baseline(~0ms)이 log축을 왜곡하지 않도록 표시용 하한

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "results")

# dataviz 스킬 참조 팔레트(라이트 모드) 중 7개 슬롯 (yellow는 orange와 인접 시 구분 어려워 제외)
FAMILY_COLORS = {
    "baseline":          "#2a78d6",  # blue
    "realsense_spatial":  "#eb6834",  # orange
    "bilateral_depth":    "#1baf7a",  # aqua
    "guided_rgb":         "#e87ba4",  # magenta
    "nlm":                "#008300",  # green
    "median":             "#4a3aa7",  # violet
    "anisotropic":        "#e34948",  # red
}
FAMILY_MARKERS = {
    "baseline": "D", "realsense_spatial": "o", "bilateral_depth": "s",
    "guided_rgb": "^", "nlm": "P", "median": "X", "anisotropic": "v",
}
FAMILY_LABELS_KO = {
    "baseline": "무필터", "realsense_spatial": "RealSense spatial",
    "bilateral_depth": "bilateral(depth)", "guided_rgb": "guided(RGB)",
    "nlm": "NLM", "median": "median", "anisotropic": "anisotropic",
}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def load_results():
    path = os.path.join(RESULTS_DIR, "3-3_spatial_result.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ranked_bar_chart(results):
    # guided(RGB)는 다른 기법 대비 10~40배 큰 이상치라 같이 그리면 나머지 비교가 눌림 -> 분리, 각주 처리
    OUTLIER_FAMILY = "guided_rgb"
    main_rows = sorted(
        [r for r in results if r["technique"] != OUTLIER_FAMILY],
        key=lambda r: r["flat_noise_std_mm"],
    )
    outlier_rows = sorted(
        [r for r in results if r["technique"] == OUTLIER_FAMILY],
        key=lambda r: r["flat_noise_std_mm"],
    )

    labels = [f'{FAMILY_LABELS_KO[r["technique"]]} ({r["param"]})' for r in main_rows]
    stds = [r["flat_noise_std_mm"] for r in main_rows]
    times = [r["proc_time_ms"] for r in main_rows]
    colors = [FAMILY_COLORS[r["technique"]] for r in main_rows]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    y = np.arange(len(main_rows))
    ax.barh(y, stds, color=colors, height=0.65, zorder=3)

    baseline_std = next(r["flat_noise_std_mm"] for r in results if r["technique"] == "baseline")
    ax.axvline(baseline_std, color=TEXT_MUTED, linestyle="--", linewidth=1, zorder=2)
    ax.text(baseline_std, -0.9, " baseline(무필터) 기준선", color=TEXT_SECONDARY,
            fontsize=9, va="bottom")

    for yi, std, t in zip(y, stds, times):
        ax.text(std + 0.004, yi, f"{std:.3f}mm  ({t:.1f}ms)", va="center",
                fontsize=8.5, color=TEXT_PRIMARY)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=TEXT_PRIMARY)
    ax.invert_yaxis()  # 가장 좋은(std 낮은) 게 위로
    ax.set_xlabel("평탄면 노이즈 std (mm) - 낮을수록 좋음", color=TEXT_SECONDARY, fontsize=10)
    ax.set_title("실험 3: spatial filter 기법별 노이즈 랭킹 (낮을수록 좋음, 괄호는 처리시간)",
                 color=TEXT_PRIMARY, fontsize=13, pad=14)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.set_xlim(0, max(stds) * 1.4)

    outlier_txt = ", ".join(
        f'{FAMILY_LABELS_KO[r["technique"]]} {r["param"]}={r["flat_noise_std_mm"]:.2f}mm'
        for r in outlier_rows
    )
    fig.text(0.5, 0.005,
              f"* guided(RGB)는 radius 클수록 악화되는 이상치라 제외: {outlier_txt} "
              f"(무광 검은 시편 -> RGB 가이드 분산 0에 가까워 계수 불안정, md 결론 참고)",
              ha="center", fontsize=8, color=TEXT_MUTED)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for k, c in FAMILY_COLORS.items() if k != OUTLIER_FAMILY]
    ax.legend(handles, [v for k, v in FAMILY_LABELS_KO.items() if k != OUTLIER_FAMILY], loc="upper right",
              fontsize=8.5, frameon=False, title="기법", title_fontsize=9)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(RESULTS_DIR, "3-3_spatial_ranked_bar.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"저장: {out}")


def pareto_scatter(results):
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    seen = set()
    for r in results:
        fam = r["technique"]
        color = FAMILY_COLORS[fam]
        marker = FAMILY_MARKERS[fam]
        label = FAMILY_LABELS_KO[fam] if fam not in seen else None
        seen.add(fam)
        x = max(r["proc_time_ms"], TIME_FLOOR_MS)  # baseline(~0ms)이 log축을 왜곡하지 않도록 표시상 하한
        ax.scatter(x, r["flat_noise_std_mm"], s=90, color=color,
                   marker=marker, edgecolor="white", linewidth=0.8, zorder=3, label=label)

    # 대표 지점만 직접 라벨링 (전체 라벨링은 지저분해짐)
    best_std = min(results, key=lambda r: r["flat_noise_std_mm"])
    fastest_nonbaseline = min(
        (r for r in results if r["technique"] != "baseline"),
        key=lambda r: r["proc_time_ms"],
    )
    for r, note in [(best_std, "최저 노이즈"), (fastest_nonbaseline, "최속(비교군 중)")]:
        x = max(r["proc_time_ms"], TIME_FLOOR_MS)
        ax.annotate(
            f'{note}\n{FAMILY_LABELS_KO[r["technique"]]} {r["param"]}',
            (x, r["flat_noise_std_mm"]),
            textcoords="offset points", xytext=(10, 8), fontsize=8, color=TEXT_SECONDARY,
        )

    baseline = next(r for r in results if r["technique"] == "baseline")
    ax.annotate(
        f'무필터 (실제 {baseline["proc_time_ms"]:.4f}ms, 표시상 {TIME_FLOOR_MS}ms로 고정)',
        (TIME_FLOOR_MS, baseline["flat_noise_std_mm"]),
        textcoords="offset points", xytext=(10, -14), fontsize=8, color=TEXT_MUTED,
    )

    ax.set_xscale("log")
    ax.set_xlim(TIME_FLOOR_MS * 0.6, 300)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v:g}"))  # 지수(마이너스 글리프) 대신 일반 숫자로 표기
    ax.set_xlabel("처리시간 (ms/frame, log scale)", color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel("평탄면 노이즈 std (mm)", color=TEXT_SECONDARY, fontsize=10)
    ax.set_title("실험 3: 노이즈 vs 처리시간 트레이드오프 (왼쪽 아래일수록 좋음)",
                 color=TEXT_PRIMARY, fontsize=13, pad=14)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False, title="기법", title_fontsize=9)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "3-3_spatial_pareto_scatter.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"저장: {out}")


if __name__ == "__main__":
    results = load_results()
    ranked_bar_chart(results)
    pareto_scatter(results)
