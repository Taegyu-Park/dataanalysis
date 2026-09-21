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
import matplotlib as mpl
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
q_threshold = 5.0
flow_threshold = 25.0
frac_low_q = (q_loop < q_threshold).mean()
frac_low_q_high_flow = ((q_loop < q_threshold) & (v_flow >= flow_threshold)).mean()

# 3. Create figure with dartwork-mpl physical width & aspect ratio
fig, ax = plt.subplots(figsize=dm.figsize("16cm", "wide"))

# 4. Shade the "heat-exchange collapse" band (Q < threshold) and label for legend
ax.axhspan(
    0.0, q_threshold,
    color="#FEE2E2",
    alpha=0.7,
    label=f"열교환 붕괴 구간 ($Q < {q_threshold:.0f}\\,\\mathrm{{kW}}$)",
    zorder=1
)

# 5. Scatter: time vs heat exchanged, colored by borehole in/out temperature difference
#    (flow rate stays almost constant at ~29 m3/h year-round, so it carries little
#    information as a color channel; dT is the variable that actually drives Q here)
sc = ax.scatter(
    datetime_series,
    q_loop,
    c=delta_t,
    cmap="viridis",
    vmin=0,
    vmax=4,
    s=8,
    alpha=0.5,
    edgecolors="none",
    rasterized=True,
    zorder=3
)

# 6. 7-day rolling mean trend line for seasonal context
rolling_q = q_loop.rolling(window=24 * 7, min_periods=1, center=True).mean()
ax.plot(
    datetime_series,
    rolling_q,
    color="#1F2937",
    linewidth=dm.lw(0.5),
    linestyle="--",
    alpha=0.8,
    label="7일 이동 평균",
    zorder=5
)

# 7. Annotation with grounding statistics
ax.text(
    0.015, 0.97,
    f"Q<{q_threshold:.0f}kW인 시간: 연중 {frac_low_q*100:.1f}%\n"
    f"→ 그 중 유량 {flow_threshold:.0f} m³/h 이상(정상 순환 중)인 비율: {frac_low_q_high_flow/frac_low_q*100:.1f}%\n"
    f"즉 펌프는 계속 도는데 열교환만 계절 따라 붕괴됨",
    transform=ax.transAxes,
    fontsize=dm.fs(-3),
    color="#374151",
    ha="left",
    va="top",
    zorder=5,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#F3F4F6", edgecolor="#D1D5DB", alpha=0.9)
)

# 8. Labels and Titles
ax.set_title(
    "Attempt 3: 시간에 따른 지중 열교환량 붕괴 시점 — 색은 보어홀 수온차",
    fontsize=dm.fs(2),
    fontweight=dm.fw(1),
    pad=12
)
ax.set_xlabel("계측 월 (4월 ~ 이듬해 3월)", fontsize=dm.fs(0))
ax.set_ylabel(r"지중 루프 열교환량 $Q_{loop}$ (kW)", fontsize=dm.fs(0))

# 9. Axis limits and formatting
ax.set_ylim(0, q_loop.quantile(0.999))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m월"))
fig.autofmt_xdate(rotation=0, ha="center")

ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
ax.legend(loc="upper right", frameon=True, framealpha=0.8, fontsize=dm.fs(-1))

# 10. Colorbar
cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.03, extend="max")
cbar.set_label(r"보어홀 입출구 수온차 $\Delta T$ (K)", fontsize=dm.fs(-1))
cbar.ax.tick_params(labelsize=dm.fs(-2))

# 11. Transparent background setup
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# 12. Content-aware layout
dm.simple_layout(fig, margin="3mm")

# 13. Save as PNG and SVG simultaneously
script_stem = os.path.splitext(os.path.basename(__file__))[0]
plot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plot"))
os.makedirs(plot_dir, exist_ok=True)
save_stem = os.path.join(plot_dir, script_stem)

dm.save_formats(fig, save_stem, formats=("png", "svg"), transparent=True, dpi=300)
print(f"Successfully saved {save_stem}.png and {save_stem}.svg")
print(f"frac_low_q={frac_low_q:.4f}, frac_low_q_high_flow={frac_low_q_high_flow:.4f}")

# Copy to brain artifact directory for IDE display
brain_dir = r"C:\Users\taegyu\.gemini\antigravity-ide\brain\a77bc78a-a1a1-44c8-bd6b-2a905022a6c5"
if os.path.exists(brain_dir):
    shutil.copy(f"{save_stem}.png", os.path.join(brain_dir, f"{script_stem}.png"))
    shutil.copy(f"{save_stem}.svg", os.path.join(brain_dir, f"{script_stem}.svg"))
    print("Copied files to brain artifact directory")
