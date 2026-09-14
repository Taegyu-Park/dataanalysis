import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import dartwork_mpl as dm

# 1. Apply dartwork-mpl publication styling
dm.style.use("scientific-kr")

# 2. Load dataset (first 8760 hourly rows)
data_path = r'c:\Users\taegyu\Codes\dataanalysis\data\Studenthuset_GSHP_data_201604-201703_Data.csv'
df = pd.read_csv(data_path, skiprows=[0, 2], encoding='utf-8', low_memory=False)
df_hourly = df.iloc[:8760].copy()

# Parse timestamps and variables
datetime_series = pd.to_datetime(df_hourly.iloc[:, 0], format="mixed")
p_pmp = pd.to_numeric(df_hourly.iloc[:, 34], errors='coerce')
v_flow = pd.to_numeric(df_hourly.iloc[:, 4], errors='coerce')
t_in = pd.to_numeric(df_hourly.iloc[:, 5], errors='coerce')
t_out = pd.to_numeric(df_hourly.iloc[:, 6], errors='coerce')
delta_t = (t_in - t_out).abs()
q_loop = pd.to_numeric(df_hourly.iloc[:, 7], errors='coerce').abs()
q_heat = pd.to_numeric(df_hourly.iloc[:, 17], errors='coerce').fillna(0)
q_cool = pd.to_numeric(df_hourly.iloc[:, 10], errors='coerce').fillna(0)
q_dhw = pd.to_numeric(df_hourly.iloc[:, 32], errors='coerce').fillna(0)
q_total = q_heat + q_cool + q_dhw

# Define Phantom Pumping Condition:
# Active pump (> 0.5 kW) while thermal work is negligible (delta_T < 0.5 K & q_loop < 5 kW)
is_phantom = (p_pmp > 0.5) & (delta_t < 0.5) & (q_loop < 5.0)

# Build DataFrame for 2D pivot
df_carpet = pd.DataFrame({
    'datetime': datetime_series,
    'date': datetime_series.dt.date,
    'hour': datetime_series.dt.hour,
    'is_phantom': is_phantom.astype(int),
    'p_pmp': p_pmp,
    'wasted_power': np.where(is_phantom, p_pmp, 0.0)
})

# Pivot: Rows = Hour (0~23), Columns = Unique Dates (365 days)
pivot_wasted = df_carpet.pivot(index='hour', columns='date', values='wasted_power')

# 3. Create Figure with dartwork-mpl physical width & aspect ratio
fig, ax = plt.subplots(figsize=dm.figsize("16cm", "standard"))

# Custom Colormap: 0 is light neutral gray/transparent, >0 rises to bold warning red
cmap_colors = ["#F3F4F6", "#FED7AA", "#F97316", "#DC2626", "#991B1B"]
custom_cmap = LinearSegmentedColormap.from_list("phantom_cmap", cmap_colors, N=256)

# 4. Render 2D Carpet Plot (imshow)
dates = list(pivot_wasted.columns)
extent = [0, len(dates), 24, 0]  # top=0h, bottom=24h for intuitive diurnal reading

im = ax.imshow(
    pivot_wasted.values,
    cmap=custom_cmap,
    aspect="auto",
    extent=extent,
    vmin=0,
    vmax=1.2,
    interpolation="nearest"
)

# 5. X-axis monthly formatting (Month ticks without year)
date_series_unique = pd.Series(dates)
month_starts = []
month_labels = []
prev_m = None
for i, d in enumerate(dates):
    if d.month != prev_m:
        month_starts.append(i)
        month_labels.append(f"{d.month:02d}월")
        prev_m = d.month

ax.set_xticks(month_starts)
ax.set_xticklabels(month_labels, fontsize=dm.fs(-1))
ax.set_xlim(0, len(dates))

# 6. Y-axis: Diurnal Hours (0 to 24)
ax.set_yticks([0, 4, 8, 12, 16, 20, 24])
ax.set_yticklabels(["00시", "04시", "08시", "12시", "16시", "20시", "24시"], fontsize=dm.fs(-1))
ax.set_ylim(24, 0)  # 0 at top, 24 at bottom

# 7. Labels and Titles
ax.set_title(
    "Attempt 1: 2D 시간-달력 카펫 플롯을 통한 '유령 펌핑' 시공간 패턴 발굴",
    fontsize=dm.fs(1),
    fontweight=dm.fw(1),
    pad=10
)
ax.set_xlabel("계측 월 (4월 ~ 이듬해 3월)", fontsize=dm.fs(0))
ax.set_ylabel("일일 시간대 (24시간 주기)", fontsize=dm.fs(0))

# 8. Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
cbar.set_label("유령 펌핑 낭비 전력 (kW)", fontsize=dm.fs(-1))
cbar.ax.tick_params(labelsize=dm.fs(-2))

# Sub-1 hairline grid for hour lines
for h in [4, 8, 12, 16, 20]:
    ax.axhline(h, color="#9CA3AF", linestyle=":", linewidth=0.5, alpha=0.5)

# 9. Transparent background setup
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# 10. Content-aware layout
dm.simple_layout(fig, margin="3mm")

# 11. Save as PNG and SVG simultaneously
script_stem = os.path.splitext(os.path.basename(__file__))[0]
plot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plot"))
os.makedirs(plot_dir, exist_ok=True)
save_stem = os.path.join(plot_dir, script_stem)

dm.save_formats(fig, save_stem, formats=("png", "svg"), transparent=True, dpi=300)
print(f"Successfully saved {save_stem}.png and {save_stem}.svg")

# Copy to brain artifact directory for display
brain_dir = r"C:\Users\taegyu\.gemini\antigravity-ide\brain\a77bc78a-a1a1-44c8-bd6b-2a905022a6c5"
if os.path.exists(brain_dir):
    shutil.copy(f"{save_stem}.png", os.path.join(brain_dir, f"{script_stem}.png"))
    shutil.copy(f"{save_stem}.svg", os.path.join(brain_dir, f"{script_stem}.svg"))
    print("Copied files to brain artifact directory")
