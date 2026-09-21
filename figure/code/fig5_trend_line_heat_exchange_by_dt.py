import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.collections import LineCollection
import dartwork_mpl as dm

# 1. Apply dartwork-mpl publication styling
dm.style.use("scientific-kr")
mpl.rcParams['svg.fonttype'] = 'none'  # Ensure text remains editable in Figma

# 2. Load dataset (first 8760 hourly data rows)
data_path = r'c:\Users\taegyu\Codes\dataanalysis\data\Studenthuset_GSHP_data_201604-201703_Data.csv'
df = pd.read_csv(data_path, skiprows=[0, 2], encoding='utf-8', low_memory=False)
df_hourly = df.iloc[:8760].copy()

# Parse datetime and physical features
datetime_series = pd.to_datetime(df_hourly.iloc[:, 0], format="mixed")
v_flow = pd.to_numeric(df_hourly.iloc[:, 4], errors='coerce')
t_in = pd.to_numeric(df_hourly.iloc[:, 5], errors='coerce')
t_out = pd.to_numeric(df_hourly.iloc[:, 6], errors='coerce')
delta_t = (t_in - t_out).abs()
# Column 7: "Calculated heat provided by loop" (kW) - proportional to V*dT,
# taken directly from the paper's own calorimetric calculation (see fig3-1).
q_loop = pd.to_numeric(df_hourly.iloc[:, 7], errors='coerce').abs()

# Grounding stats for the annotation box
flow_mean = v_flow.mean()
flow_std = v_flow.std()

# 3. 7-day rolling mean of both Q_loop (line position) and dT (line color),
#    then downsample to one point per day - a 7-day mean changes slowly, so
#    hourly resolution here is redundant and only bloats the vector file.
window = 24 * 7
rolling_q = q_loop.rolling(window=window, min_periods=1, center=True).mean()
rolling_dt = delta_t.rolling(window=window, min_periods=1, center=True).mean()

daily_time = datetime_series.iloc[::24].to_numpy()
daily_q = rolling_q.iloc[::24].to_numpy()
daily_dt = rolling_dt.iloc[::24].to_numpy()

# 4. Create figure with dartwork-mpl physical width & aspect ratio
fig, ax = plt.subplots(figsize=dm.figsize("16cm", "wide"))

# 5. 7-day rolling mean trend as a single solid line, colored by dT along its length
x_num = mdates.date2num(daily_time)
points = np.array([x_num, daily_q]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
# Color each segment by the average dT of its two endpoints
segment_color = (daily_dt[:-1] + daily_dt[1:]) / 2

norm = plt.Normalize(vmin=0, vmax=3)
lc = LineCollection(
    segments,
    cmap="viridis",
    norm=norm,
    zorder=3,
    capstyle="round",   # round caps at each segment end so adjacent
    joinstyle="round",  # per-day segments blend into one smooth stroke
    antialiaseds=True,
)
lc.set_array(segment_color)
lc.set_linewidth(dm.lw(1.5))
line = ax.add_collection(lc)

# 6. Annotation with grounding statistics
ax.text(
    0.015, 0.97,
    f"지중 순환 유량 $\\dot{{V}}$: 평균 {flow_mean:.2f} ± {flow_std:.2f} $\\mathrm{{m}}^3/\\mathrm{{h}}$\n"
    f"(연중 거의 일정 — 그래서 색은 유량 대신 온도차 ΔT로 표현)",
    transform=ax.transAxes,
    fontsize=dm.fs(-3),
    color="#374151",
    ha="left",
    va="top",
    zorder=5,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#F3F4F6", edgecolor="#D1D5DB", alpha=0.9)
)

# 7. Labels and Titles
ax.set_title(
    "Attempt 4: 지중 열교환량 7일 이동평균 추이 — 선 색은 보어홀 수온차",
    fontsize=dm.fs(2),
    fontweight=dm.fw(1),
    pad=12
)
ax.set_xlabel("계측 월 (4월 ~ 이듬해 3월)", fontsize=dm.fs(0))
ax.set_ylabel(r"지중 루프 열교환량 $Q_{loop}$ (kW, 7일 이동평균)", fontsize=dm.fs(0))

# 8. Axis limits and formatting
ax.set_xlim(daily_time.min(), daily_time.max())
ax.set_ylim(0, daily_q.max() * 1.1)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m월"))
fig.autofmt_xdate(rotation=0, ha="center")

ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)

# 9. Colorbar
cbar = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.03)
cbar.set_label(r"보어홀 입출구 수온차 $\Delta T$ (K, 7일 이동평균)", fontsize=dm.fs(-1))
cbar.ax.tick_params(labelsize=dm.fs(-2))

# 10. Transparent background setup
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# 11. Content-aware layout
dm.simple_layout(fig, margin="3mm")

# 12. Save as PNG and SVG simultaneously
script_stem = os.path.splitext(os.path.basename(__file__))[0]
plot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plot"))
os.makedirs(plot_dir, exist_ok=True)
save_stem = os.path.join(plot_dir, script_stem)

dm.save_formats(fig, save_stem, formats=("png", "svg"), transparent=True, dpi=300)
print(f"Successfully saved {save_stem}.png and {save_stem}.svg")
print(f"flow_mean={flow_mean:.4f}, flow_std={flow_std:.4f}")

# Copy to brain artifact directory for IDE display
brain_dir = r"C:\Users\taegyu\.gemini\antigravity-ide\brain\a77bc78a-a1a1-44c8-bd6b-2a905022a6c5"
if os.path.exists(brain_dir):
    shutil.copy(f"{save_stem}.png", os.path.join(brain_dir, f"{script_stem}.png"))
    shutil.copy(f"{save_stem}.svg", os.path.join(brain_dir, f"{script_stem}.svg"))
    print("Copied files to brain artifact directory")
