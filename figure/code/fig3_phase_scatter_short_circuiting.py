import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dartwork_mpl as dm

# 1. Apply dartwork-mpl publication styling
dm.style.use("scientific-kr")

# 2. Load dataset (first 8760 hourly rows)
data_path = r'c:\Users\taegyu\Codes\dataanalysis\data\Studenthuset_GSHP_data_201604-201703_Data.csv'
df = pd.read_csv(data_path, skiprows=[0, 2], encoding='utf-8', low_memory=False)
df_hourly = df.iloc[:8760].copy()

# Parse physical features
v_flow = pd.to_numeric(df_hourly.iloc[:, 4], errors='coerce')
t_in = pd.to_numeric(df_hourly.iloc[:, 5], errors='coerce')
t_out = pd.to_numeric(df_hourly.iloc[:, 6], errors='coerce')
delta_t = (t_in - t_out).abs()
p_total = pd.to_numeric(df_hourly.iloc[:, 3], errors='coerce')
p_pmp = pd.to_numeric(df_hourly.iloc[:, 34], errors='coerce')
p_comp = (p_total - p_pmp).clip(lower=0)
q_loop = pd.to_numeric(df_hourly.iloc[:, 7], errors='coerce').abs()

# 3. Create figure with dartwork-mpl physical width & aspect ratio
fig, ax = plt.subplots(figsize=dm.figsize("16cm", "wide"))

# 4. Highlight Inefficient / Phantom Pumping Region (Shaded rectangle)
ax.axhspan(
    0.0, 0.5,
    xmin=15/35, xmax=1.0,
    color="#FEE2E2",
    alpha=0.7,
    zorder=1
)
ax.text(
    22.5, 0.22,
    r"유령 펌핑 & 열적 단락 영역" + "\n" + r"($\Delta T < 0.5\ \mathrm{K}$, 열교환 붕괴)",
    color="#DC2626",
    fontsize=dm.fs(-1),
    fontweight=dm.fw(1),
    ha="center",
    va="center",
    zorder=4,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF", edgecolor="#FCA5A5", alpha=0.85)
)

# 5. Theoretical Iso-thermal Power Curves (Q = 1.161 * V * ΔT)
# ΔT = Q / (1.161 * V)
v_curve = np.linspace(10, 34, 200)
for q_val, line_color in [(10, "#9CA3AF"), (25, "#6B7280"), (50, "#4B5563"), (100, "#1F2937")]:
    dt_curve = q_val / (1.161 * v_curve)
    valid_mask = dt_curve <= 5.5
    ax.plot(
        v_curve[valid_mask],
        dt_curve[valid_mask],
        linestyle="--",
        linewidth=0.8,
        color=line_color,
        alpha=0.6,
        zorder=2
    )
    # Label at the right end of curve
    idx = np.where(valid_mask)[0][-1]
    ax.text(
        v_curve[idx] - 1.2,
        dt_curve[idx] + 0.08,
        f"{q_val} kW",
        fontsize=dm.fs(-3),
        color=line_color,
        ha="center",
        zorder=3
    )

# 6. Scatter Plot of 8,760 operational hours
sc = ax.scatter(
    v_flow,
    delta_t,
    c=p_comp,
    cmap="viridis",
    s=12,
    alpha=0.45,
    edgecolors="none",
    zorder=3
)

# 7. Labels and Titles
ax.set_title(
    r"Attempt 2: 유량($\dot{V}$) - 수온차($\Delta T$) 물리 위상 산점도를 통한 열교환 붕괴 규명",
    fontsize=dm.fs(1),
    fontweight=dm.fw(1),
    pad=10
)
ax.set_xlabel(r"지중 순환 유량 ($\mathrm{m}^3/\mathrm{h}$)", fontsize=dm.fs(0))
ax.set_ylabel(r"보어홀 입출구 수온차 $\Delta T$ (K)", fontsize=dm.fs(0))

# Axis limits
ax.set_xlim(0, 35)
ax.set_ylim(0, 5.5)

# Sub-1 hairline grid
ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5, zorder=0)

# 8. Colorbar
cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.03)
cbar.set_label("히트펌프 압축기 소비 전력 (kW)", fontsize=dm.fs(-1))
cbar.ax.tick_params(labelsize=dm.fs(-2))

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

# Copy to brain artifact directory for IDE display
brain_dir = r"C:\Users\taegyu\.gemini\antigravity-ide\brain\a77bc78a-a1a1-44c8-bd6b-2a905022a6c5"
if os.path.exists(brain_dir):
    shutil.copy(f"{save_stem}.png", os.path.join(brain_dir, f"{script_stem}.png"))
    shutil.copy(f"{save_stem}.svg", os.path.join(brain_dir, f"{script_stem}.svg"))
    print("Copied files to brain artifact directory")
