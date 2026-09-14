import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import dartwork_mpl as dm

# 1. Apply dartwork-mpl publication styling
dm.style.use("scientific-kr")

# 2. Load dataset (first 8760 hourly data rows)
data_path = r'c:\Users\taegyu\Codes\dataanalysis\data\Studenthuset_GSHP_data_201604-201703_Data.csv'
df = pd.read_csv(data_path, skiprows=[0, 2], encoding='utf-8', low_memory=False)
df_hourly = df.iloc[:8760].copy()

# Parse datetime and pump electricity
datetime_series = pd.to_datetime(df_hourly.iloc[:, 0], format="mixed")
# Column index 34: Est. source-side circ pump energy consumption (kWh/h)
p_pmp = pd.to_numeric(df_hourly.iloc[:, 34], errors='coerce')

# 3. Create figure with dartwork-mpl physical width & aspect ratio
fig, ax = plt.subplots(figsize=dm.figsize("16cm", "wide"))

# 4. Plot Baseline line chart
ax.plot(
    datetime_series,
    p_pmp,
    color="#2563EB",  # clean scientific blue
    linewidth=dm.lw(0),
    alpha=0.75,
    label="지중 순환 펌프 소비 전력 (Hourly)"
)

# 7-day rolling mean to show long-term trend
rolling_pmp = p_pmp.rolling(window=24*7, min_periods=1, center=True).mean()
ax.plot(
    datetime_series,
    rolling_pmp,
    color="#DC2626",
    linewidth=dm.lw(1),
    linestyle="--",
    label="7일 이동 평균 (7-day Moving Average)"
)

# 5. Labels and Titles
ax.set_title(
    "Baseline: 1개년 지중 순환 펌프 소비 전력 시계열 (8,760시간)",
    fontsize=dm.fs(2),
    fontweight=dm.fw(1),
    pad=12
)
ax.set_xlabel("계측 월 (4월 ~ 이듬해 3월)", fontsize=dm.fs(0))
ax.set_ylabel("소비 전력 (kWh/h = kW)", fontsize=dm.fs(0))

# 6. Axis limits and formatting
ax.set_ylim(0, 2.0)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m월"))
fig.autofmt_xdate(rotation=0, ha="center")

# Sub-1 hairline grid
ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
ax.legend(loc="upper right", frameon=True, framealpha=0.8, fontsize=dm.fs(-1))

# 7. Transparent background setup
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# 8. Content-aware layout with comfortable margin to prevent canvas clipping
dm.simple_layout(fig, margin="3mm")

# 9. Save as both PNG and SVG with identical stem name as the script
script_stem = os.path.splitext(os.path.basename(__file__))[0]
plot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plot"))
os.makedirs(plot_dir, exist_ok=True)
save_stem = os.path.join(plot_dir, script_stem)

# Use dm.save_formats to save PNG and SVG simultaneously
dm.save_formats(fig, save_stem, formats=("png", "svg"), transparent=True, dpi=300)
print(f"Successfully saved {save_stem}.png and {save_stem}.svg")

# Copy to brain artifact directory for IDE display
brain_dir = r"C:\Users\taegyu\.gemini\antigravity-ide\brain\a77bc78a-a1a1-44c8-bd6b-2a905022a6c5"
if os.path.exists(brain_dir):
    shutil.copy(f"{save_stem}.png", os.path.join(brain_dir, f"{script_stem}.png"))
    shutil.copy(f"{save_stem}.svg", os.path.join(brain_dir, f"{script_stem}.svg"))
    print("Copied files to brain artifact directory")
