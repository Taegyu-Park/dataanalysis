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

# Parse datetime and ambient temperature
datetime_series = pd.to_datetime(df_hourly.iloc[:, 0], format="mixed")
# Column index 2: Ambient temperature (°C)
t_amb = pd.to_numeric(df_hourly.iloc[:, 2], errors='coerce')

# Basic statistics for annotation
t_mean = t_amb.mean()
t_min = t_amb.min()
t_max = t_amb.max()
sub_zero_hours = (t_amb < 0).sum()
sub_zero_pct = (t_amb < 0).mean() * 100

# 3. Create figure with dartwork-mpl physical width & aspect ratio
fig, ax = plt.subplots(figsize=dm.figsize("16cm", "wide"))

# 4. Reference freezing line (0°C)
ax.axhline(0, color="#94A3B8", linestyle=":", linewidth=0.9, alpha=0.8, zorder=1)

# 5. Plot Hourly Ambient Temperature
ax.plot(
    datetime_series,
    t_amb,
    color="#93C5FD",  # soft blue
    linewidth=dm.lw(0),
    alpha=0.6,
    label="시간별 외기 온도 (Hourly)",
    zorder=2
)

# 6. 7-day rolling mean to show seasonal progression
rolling_t = t_amb.rolling(window=24 * 7, min_periods=1, center=True).mean()
ax.plot(
    datetime_series,
    rolling_t,
    color="#2563EB",  # clean deep blue
    linewidth=dm.lw(1),
    label="7일 이동 평균 (7-day Moving Average)",
    zorder=3
)

# 7. Annotation with climate grounding statistics
ax.text(
    0.02, 0.96,
    f"연평균 기온: {t_mean:.1f} °C (최저 {t_min:.1f} °C / 최고 {t_max:.1f} °C)\n"
    f"영하(<0 °C) 시간: {sub_zero_hours:,}시간 (연중 {sub_zero_pct:.1f}%)\n"
    f"기후 특성: 전형적인 북유럽 습윤 대륙성 기후 (스톡홀름)",
    transform=ax.transAxes,
    fontsize=dm.fs(-2),
    color="#1E293B",
    ha="left",
    va="top",
    zorder=5,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.9)
)

# 8. Labels and Titles
ax.set_title(
    "1개년 외기 온도 시계열 및 계절 변동 (스톡홀름 Frescati 캠퍼스)",
    fontsize=dm.fs(2),
    fontweight=dm.fw(1),
    pad=12
)
ax.set_xlabel("계측 월 (4월 ~ 이듬해 3월)", fontsize=dm.fs(0))
ax.set_ylabel("외기 온도 (°C)", fontsize=dm.fs(0))

# 9. Axis limits and formatting
ax.set_ylim(-20, 35)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m월"))
fig.autofmt_xdate(rotation=0, ha="center")

# Hairline grid
ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5, zorder=0)
ax.legend(loc="upper right", frameon=True, framealpha=0.85, fontsize=dm.fs(-1))

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
