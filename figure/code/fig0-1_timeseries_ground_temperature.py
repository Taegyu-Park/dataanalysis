import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import dartwork_mpl as dm

# 1. Apply dartwork-mpl publication styling and preserve editable text for Figma
dm.style.use("scientific-kr")
mpl.rcParams['svg.fonttype'] = 'none'

# 2. Load dataset (first 8760 hourly data rows)
data_path = r'c:\Users\taegyu\Codes\dataanalysis\data\Studenthuset_GSHP_data_201604-201703_Data.csv'
df = pd.read_csv(data_path, skiprows=[0, 2], encoding='utf-8', low_memory=False)
df_hourly = df.iloc[:8760].copy()

# Parse datetime and temperature features
datetime_series = pd.to_datetime(df_hourly.iloc[:, 0], format="mixed")
# Column index 5: Loop fluid to the boreholes (°C)
t_to = pd.to_numeric(df_hourly.iloc[:, 5], errors='coerce')
# Column index 6: Loop fluid from the boreholes (°C)
t_from = pd.to_numeric(df_hourly.iloc[:, 6], errors='coerce')
# Mean loop temperature
t_loop_mean = (t_to + t_from) / 2.0

# Key statistics for annotation
t_from_mean = t_from.mean()
t_from_min = t_from.min()
t_from_max = t_from.max()
ugt_reference = 10.6  # Undisturbed Ground Temperature in Stockholm region

# 3. Create figure with dartwork-mpl physical width & aspect ratio
fig, ax = plt.subplots(figsize=dm.figsize("16cm", "wide"))

# 4. Reference line for undisturbed ground temperature (UGT ~10.6°C)
ax.axhline(
    ugt_reference,
    color="#059669",
    linestyle=":",
    linewidth=1.0,
    alpha=0.85,
    label=f"스톡홀름 지역 비교란 지중온도 (UGT ≈ {ugt_reference} °C)",
    zorder=2
)

# 5. Plot ground loop supply and return temperatures
# t_to: fluid going into boreholes
ax.plot(
    datetime_series,
    t_to,
    color="#93C5FD",  # light blue
    linewidth=dm.lw(0),
    alpha=0.5,
    label="보어홀 주입 수온 ($T_{to}$, 지중 유입)",
    zorder=3
)

# t_from: fluid leaving boreholes (representative ground source temperature)
ax.plot(
    datetime_series,
    t_from,
    color="#D97706",  # warm amber/orange
    linewidth=dm.lw(0.5),
    alpha=0.75,
    label="보어홀 유출 수온 ($T_{from}$, 지중 열원)",
    zorder=4
)

# 6. 7-day rolling mean of ground source temperature
rolling_t_from = t_from.rolling(window=24 * 7, min_periods=1, center=True).mean()
ax.plot(
    datetime_series,
    rolling_t_from,
    color="#B45309",  # deep amber trend line
    linewidth=dm.lw(1.5),
    label="$T_{from}$ 7일 이동 평균 추세선",
    zorder=5
)

# 7. Annotation with ground thermal grounding statistics
ax.text(
    0.02, 0.96,
    f"연평균 지중 유출 수온: {t_from_mean:.1f} °C (최저 {t_from_min:.1f} °C / 최고 {t_from_max:.1f} °C)\n"
    f"여름철 직냉방 배열로 최고 13.4 °C까지 상승 (자연 방열)\n"
    f"겨울철 난방 흡열로 최저 6.8 °C까지 하강 (안정적 열원 유지)",
    transform=ax.transAxes,
    fontsize=dm.fs(-2),
    color="#1E293B",
    ha="left",
    va="top",
    zorder=6,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFBEB", edgecolor="#FDE68A", alpha=0.92)
)

# 8. Labels and Titles
ax.set_title(
    "1개년 지중 루프 순환 수온 시계열 및 계절 거동 (보어홀 20개 루프)",
    fontsize=dm.fs(2),
    fontweight=dm.fw(1),
    pad=12
)
ax.set_xlabel("계측 월 (4월 ~ 이듬해 3월)", fontsize=dm.fs(0))
ax.set_ylabel("지중 순환 유체 온도 (°C)", fontsize=dm.fs(0))

# 9. Axis limits and formatting
ax.set_ylim(-2, 18)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m월"))
fig.autofmt_xdate(rotation=0, ha="center")

# Hairline grid
ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5, zorder=0)
ax.legend(loc="lower right", frameon=True, framealpha=0.88, fontsize=dm.fs(-1))

# 10. Transparent background setup
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# 11. Content-aware layout
dm.simple_layout(fig, margin="3mm")

# 12. Save as both PNG and SVG simultaneously
script_stem = os.path.splitext(os.path.basename(__file__))[0]
plot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plot"))
os.makedirs(plot_dir, exist_ok=True)
save_stem = os.path.join(plot_dir, script_stem)

dm.save_formats(fig, save_stem, formats=("png", "svg"), transparent=True, dpi=300)
print(f"Successfully saved {save_stem}.png and {save_stem}.svg")

# Copy to brain artifact directory for IDE display
brain_dir = r"C:\Users\taegyu\.gemini\antigravity-ide\brain\a77bc78a-a1a1-44c8-bd6b-2a905022a6c5"
if os.path.exists(brain_dir):
    shutil.copy(f"{save_stem}.png", os.path.join(brain_dir, f"{script_stem}.png"))
    shutil.copy(f"{save_stem}.svg", os.path.join(brain_dir, f"{script_stem}.svg"))
    print("Copied files to brain artifact directory")
