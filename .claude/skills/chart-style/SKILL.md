---
name: chart-style
description: "TRIGGER before generating or re-rendering ANY matplotlib chart/graph/plot for this project (c:\\Users\\pc\\Desktop\\tire) — apply the seaborn-muted + IEEE-paper look and avoid known glyph/legibility bugs before saving the figure."
---

이 프로젝트의 모든 그래프(실험 결과 시각화 등)는 아래 스타일을 기본값으로 적용한다.
매번 "보기 편하게 해줘"라고 말하지 않아도 이 스킬이 자동으로 적용되어야 한다.

## 목표 스타일

**seaborn 'muted' 톤 + IEEE 논문 관례**를 섞은 스타일. seaborn 패키지가 설치되어
있으면 `sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)`를 써도
되지만, 이 프로젝트엔 seaborn이 없을 수 있으므로 **matplotlib rcParams만으로 동일한
룩을 재현**하는 걸 기본으로 한다 (의존성 추가 없이 항상 동작).

### 1. 기본 rcParams (모든 차트 스크립트 상단에 적용)

```python
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Malgun Gothic",       # 한글 라벨 필수
    "font.size": 12,                      # 기본보다 큰 폰트
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.unicode_minus": False,          # 마이너스 글리프 크래시 방지 (아래 3번 참고)
    "mathtext.fontset": "dejavusans",     # 로그축 지수 등 mathtext의 minus glyph 누락 방지
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.grid": True,
    "grid.color": "#e1e0d9",              # 연한 회색 격자선
    "grid.linewidth": 0.8,
    "axes.edgecolor": "#c3c2b7",
    "axes.spines.top": False,             # 위/오른쪽 테두리 제거 (IEEE 관례)
    "axes.spines.right": False,
    "figure.autolayout": False,           # tight_layout()을 명시 호출
})
```

### 2. 색상 팔레트 (categorical, muted 톤)

고정 순서로 사용 — 계열이 바뀌어도 같은 대상은 항상 같은 색을 유지한다 (색이 서열이
아니라 정체성을 나타내야 함):

```python
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# blue, orange, aqua, magenta, green, violet, red
```

카테고리가 4개를 넘으면(산점도처럼 모든 쌍이 동시에 보이는 경우) 색만으로 구분하지
말고 마커 모양(o, s, ^, D, P, X, v 등)도 같이 다르게 준다.

### 3. IEEE 논문 관례에서 가져오는 것

- **그래프 안에 title을 넣지 않는다.** IEEE 논문은 캡션이 그림 밖(아래)에 붙기 때문에,
  그림 자체엔 축 라벨과 범례만 있으면 됨. 이 프로젝트에선 md 파일의 `<summary>`나
  본문 설명이 캡션 역할을 하므로, `ax.set_title()`은 생략하고 md 쪽 텍스트로 설명한다.
  (기존 실험1/3 차트처럼 이미 title을 넣은 것도 있는데, 새로 만들 때는 생략 우선.)
- **축 라벨에 단위를 꼭 포함.** "노이즈" (X) → "노이즈 std (mm)" (O)
- **범례는 최소한으로, 데이터를 가리지 않는 위치에.** 테두리 박스(`frameon=True`)는
  쓰지 않는다(`frameon=False`) — 얇고 튀지 않게.
- **여백을 넉넉히.** `fig.tight_layout()`을 항상 호출하고, 막대그래프처럼 텍스트
  라벨이 데이터 옆에 붙는 경우 `ax.set_xlim`/`set_ylim`을 실제 데이터 범위보다
  30~40% 여유 있게 잡아서 라벨이 잘리지 않게 한다.
- **해상도**: `fig.savefig(path, dpi=150)` 이상 (인쇄/확대 대비 최소 150, 여유 있으면 300).

### 4. 이번 세션에서 실제로 겪은 버그 — 반드시 피할 것

- **em-dash(—, U+2014)나 유니코드 마이너스(−, U+2212)를 `print()`나 라벨 문자열에
  쓰지 않는다.** Windows 콘솔(cp949)에서 `UnicodeEncodeError`로 스크립트 자체가
  죽는다. 일반 하이픈(`-`)만 사용.
- **이상치(outlier) 하나가 전체 차트를 망가뜨리지 않게 한다.** 막대/산점도에서 다른
  값보다 5배 이상 큰 값이 하나라도 있으면, 축 범위가 늘어나면서 나머지 데이터가
  전부 눌려서 안 보이게 됨 → 이상치는 메인 차트에서 제외하고 각주(`fig.text`)로
  별도 표기, 또는 log scale + 값 라벨 병기.
- **log scale 쓸 때 값이 0에 가까우면 축이 왜곡된다.** 표시용 하한(floor)을 정해서
  `max(value, floor)`로 클리핑하고, 실제 값은 라벨/각주에 텍스트로 남긴다.
- **시각화 crop 크기와 실제 측정(ROI) 크기를 다르게 쓰지 않는다.** 다르면 "이 영역이
  평탄하다"고 주장하는 그림에 실제로는 경계가 보이는 식으로 오해를 유발한다. 크기를
  맞추거나, 측정 영역을 사각형으로 오버레이 표시한다.

### 5. 다중 조건 x 다중 지표 비교 레이아웃

여러 조건(예: 해상도 3종)을 여러 지표(예: 노이즈 std, 픽셀수)로 비교할 때 흔히 저지르는
실수: **조건 수만큼 서브플롯을 쪼개고, 서브플롯 안에서 지표끼리 dual-axis(twinx)로 겹치는
구성.** 이러면 같은 지표를 조건끼리 비교하려면(예: 해상도 3개의 노이즈 std끼리 비교) 서로
다른 서브플롯을 눈으로 오가며 값을 비교해야 해서 불편함.

**원칙: 반대로 뒤집는다.**
- 서브플롯(또는 별도 그래프)은 **지표 수**만큼 만든다 (노이즈 그래프 1개, 픽셀수 그래프 1개).
- 그 안에서 **조건은 색상**(고정 팔레트, 위 2번)으로 구분한 선/막대로 한 그래프에 겹쳐 그린다.
- 즉 "파란선끼리, 빨간선끼리" 한 그래프에서 바로 비교되게 — 조건별 subplot이 아니라
  지표별 subplot + 조건별 색상.
- dual-axis(twinx)는 **조건 비교가 아니라, 단위가 다른 지표 2개를 어쩔 수 없이 한 x축
  위에 같이 봐야 할 때**(예: x축이 alpha 하나뿐이고 y가 노이즈mm/lag프레임 두 개)만 쓴다.
  조건(카테고리)이 여러 개라면 절대 조건별 서브플롯+dual-axis 조합을 쓰지 않는다.

예시(이번 세션에서 실제로 고친 사례): 실험1(decimation)은 해상도 3종 x magnitude를
원래 "해상도별 서브플롯 3개, 각 서브플롯 안에 노이즈(빨강)/픽셀수(파랑) dual-axis"로
그렸었는데, 이러면 해상도 간 노이즈 비교가 안 됨. "노이즈 그래프 1개(해상도 3색 선) +
픽셀수 그래프 1개(해상도 3색 선)"로 바꿔서 해결.

## 체크리스트 (저장 전 확인)

- [ ] 폰트가 Malgun Gothic으로 한글이 깨지지 않는가
- [ ] 격자선이 연한 회색이고 데이터보다 튀지 않는가
- [ ] 색상이 고정 팔레트 순서를 따르고, 카테고리가 4개 넘으면 마커도 다른가
- [ ] 축 라벨에 단위가 있는가
- [ ] 이상치 때문에 나머지 데이터가 눌리지 않았는가
- [ ] em-dash/유니코드 마이너스가 없는가 (콘솔 크래시 방지)
- [ ] `fig.tight_layout()` 호출했고 라벨이 잘리지 않는가
- [ ] 실제로 이미지를 열어서 눈으로 확인했는가 (경고 메시지만 보고 넘기지 않기)
