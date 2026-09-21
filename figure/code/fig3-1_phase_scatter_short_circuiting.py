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
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'  # Ensure text remains editable in Figma

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
# Column 7: "Calculated heat provided by loop" (kW) - the paper's own
# calorimetric Q = mdot*cp*dT result, using the ACTUAL brine (ethanol/water
# mixture) properties instead of a textbook pure-water constant.
q_loop_paper = pd.to_numeric(df_hourly.iloc[:, 7], errors='coerce').abs()

# 3. Calibrate the iso-power constant k directly from the paper's own data
#    instead of assuming pure water (rho*cp/3600 = 1.161).
#    Q = k * V * dT  ->  fit k by least squares through the origin.
fit_mask = v_flow.notna() & delta_t.notna() & q_loop_paper.notna() & (v_flow > 0) & (delta_t > 0.05)
x_fit = (v_flow[fit_mask] * delta_t[fit_mask]).to_numpy()
y_fit = q_loop_paper[fit_mask].to_numpy()
k_empirical = float((x_fit * y_fit).sum() / (x_fit * x_fit).sum())
r2 = float(1 - np.var(y_fit - k_empirical * x_fit) / np.var(y_fit))
print(f"Calibrated constant k = {k_empirical:.4f} (pure-water textbook value: 1.161), R^2 = {r2:.5f}")

# 4. Create figure with dartwork-mpl physical width & aspect ratio
fig, ax = plt.subplots(figsize=dm.figsize("16cm", "wide"))

# 5. Highlight Inefficient / Phantom Pumping Region (Shaded rectangle)
ax.axhspan(
    0.0, 0.5,
    xmin=(15 - 10) / (35 - 10), xmax=1.0,
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

# 6. Iso-power curves using the paper-calibrated constant (Q = k_empirical * V * dT)
v_curve = np.linspace(10, 34, 200)
for q_val, line_color in [(10, "#9CA3AF"), (25, "#6B7280"), (50, "#4B5563"), (100, "#1F2937")]:
    dt_curve = q_val / (k_empirical * v_curve)
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

# 7. Scatter Plot of 8,760 operational hours
sc = ax.scatter(
    v_flow,
    delta_t,
    c=p_comp,
    cmap="viridis",
    s=12,
    alpha=0.45,
    edgecolors="none",
    rasterized=True,
    zorder=3
)

# 8. Annotation explaining the calibration (traceability of the fix)
ax.text(
    0.02, 0.98,
    f"등열량선 보정: $Q = {k_empirical:.3f}\\,\\dot{{V}}\\,\\Delta T$\n"
    f"(논문 col.7 'Calculated heat provided by loop'로 최소자승 보정, $R^2$={r2:.4f})\n"
    f"※ 순수 물 가정치 1.161 대비 약 {(1 - k_empirical/1.161)*100:.1f}% 낮음 — 브라인(에탄올/물) 물성 반영",
    transform=ax.transAxes,
    fontsize=dm.fs(-3),
    color="#374151",
    ha="left",
    va="top",
    zorder=5,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#F3F4F6", edgecolor="#D1D5DB", alpha=0.9)
)

# 9. Labels and Titles
ax.set_title(
    r"Attempt 2-1: 논문 실측 계수로 보정한 등열량선 — 유량($\dot{V}$)-수온차($\Delta T$) 위상 산점도",
    fontsize=dm.fs(1),
    fontweight=dm.fw(1),
    pad=10
)
ax.set_xlabel(r"지중 순환 유량 ($\mathrm{m}^3/\mathrm{h}$)", fontsize=dm.fs(0))
ax.set_ylabel(r"보어홀 입출구 수온차 $\Delta T$ (K)", fontsize=dm.fs(0))

# Axis limits (starts from 10 m3/h)
ax.set_xlim(10, 35)
ax.set_ylim(0, 5.5)

# Sub-1 hairline grid
ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5, zorder=0)

# 10. Colorbar
cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.03)
cbar.set_label("히트펌프 압축기 소비 전력 (kW)", fontsize=dm.fs(-1))
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

# Copy to brain artifact directory for IDE display
brain_dir = r"C:\Users\taegyu\.gemini\antigravity-ide\brain\a77bc78a-a1a1-44c8-bd6b-2a905022a6c5"
if os.path.exists(brain_dir):
    shutil.copy(f"{save_stem}.png", os.path.join(brain_dir, f"{script_stem}.png"))
    shutil.copy(f"{save_stem}.svg", os.path.join(brain_dir, f"{script_stem}.svg"))
    print("Copied files to brain artifact directory")
