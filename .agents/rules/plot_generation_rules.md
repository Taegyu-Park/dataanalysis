# 그래프 생성 및 시각화 규칙 (Plot Generation Rules)

이 규칙은 프로젝트 내 모든 데이터 시각화 스크립트 작성 시 준수해야 하는 **성능 최적화(파일 용량 절감)** 및 **디자인 편집성(Figma 텍스트 편집 보존)** 가이드라인입니다.

---

## 1. 산점도(Scatter Plot) 대량 포인트 래스터화 규칙

### 📌 규칙
* 대량(수백 개 이상)의 시계열이나 관측 데이터를 산점도로 플롯할 때는 **반드시 `rasterized=True`를 명시**합니다.

```python
# 올바른 예시
sc = ax.scatter(
    x_data,
    y_data,
    c=color_data,
    s=10,
    alpha=0.5,
    edgecolors="none",
    rasterized=True,  # 필수: 수천 개의 점을 고해상도 비트맵으로 래스터화
    zorder=3
)
```

### 💡 적용 이유
* **용량 폭증 방지**: SVG 파일은 점 하나당 독립된 XML 태그(`<use ...>` 등)를 생성하므로, 8,760시간 데이터의 경우 파일 크기가 **1.5 MB 이상으로 급증**합니다. `rasterized=True`를 적용하면 축과 글자는 벡터로 유지하면서 점들만 300 DPI 이미지로 구워내어 **용량을 80~90% 이상(150~200 KB) 절감**합니다.
* **렌더링 성능 향상**: Figma, 웹 브라우저, 논문 PDF 뷰어 등에서 버벅거림(렉) 없이 즉시 부드럽게 로딩됩니다.

---

## 2. SVG 텍스트의 Figma 편집 가능 객체 보존 규칙

### 📌 규칙
* SVG로 저장할 때 글자(타이틀, 축 라벨, 눈금 텍스트, 주석 등)가 외곽선(Path/곡선)으로 변환되지 않고, **Figma나 Adobe Illustrator에서 텍스트 도구로 더블클릭하여 수정할 수 있는 `<text>` 객체로 유지**되도록 반드시 다음 설정을 코드 상단에 포함합니다.

```python
import matplotlib as mpl

# 필수: 글자를 패스(곡선)로 변환하지 않고 SVG <text> 객체로 저장하여 Figma 편집 가능 보장
mpl.rcParams['svg.fonttype'] = 'none'
```

> **주의**: Matplotlib의 기본값은 `'path'`(글자를 곡선 도형으로 변환)이므로, 명시적으로 `'none'`으로 선언하지 않으면 Figma에서 글자를 수정할 수 없고 단순 깨진 도형으로 인식됩니다.

---

## 3. 표준 코드 작성 템플릿

새로운 플롯 스크립트 작성 시 다음 뼈대 코드를 기본으로 적용합니다:

```python
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import dartwork_mpl as dm

# 1. Figma 편집 가능 텍스트 설정 (반드시 style 적용 전/후 명시)
mpl.rcParams['svg.fonttype'] = 'none'

# 2. 스타일 적용
dm.style.use("scientific-kr")
mpl.rcParams['svg.fonttype'] = 'none'  # 스타일 프리셋이 재정의할 수 있으므로 재보장

# 3. 플롯 생성
fig, ax = plt.subplots(figsize=dm.figsize("16cm", "wide"))

# 4. 산점도 플롯 시 rasterized=True 필수 적용
sc = ax.scatter(
    x, y,
    c=c,
    rasterized=True,
    zorder=3
)

# 5. 저장 (PNG + SVG 동시 생성)
dm.save_formats(fig, "save_path", formats=("png", "svg"), transparent=True, dpi=300)
```

---

## 4. 검증 체크리스트

1. [ ] `mpl.rcParams['svg.fonttype'] = 'none'`이 설정되어 있는가?
2. [ ] `ax.scatter()` 사용 시 `rasterized=True` 파라미터가 포함되어 있는가?
3. [ ] 생성된 SVG 파일 용량이 수 MB 단위가 아닌 100~500 KB 이내로 유지되는가?
4. [ ] Figma로 SVG를 임포트했을 때 텍스트 레이어가 유지되어 글자 내용 및 폰트 변경이 가능한가?
