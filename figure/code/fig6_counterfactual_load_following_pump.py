import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
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
v_flow = pd.to_numeric(df_hourly.iloc[:, 4], errors='coerce')          # m3/h
# Column 34: source-side pump power, ALREADY computed by the paper via the
# variable-speed pump's own affinity-law regression (Eq. 4 in the paper):
#   P_SSCP [kW] = 0.001973 * V_dot[L/s]^3
# The paper states this pump is controlled to a fixed MINIMUM flow floor of
# 8 L/s (= 28.8 m3/h) - which matches the "always ~29 m3/h" we keep seeing.
p_actual = pd.to_numeric(df_hourly.iloc[:, 34], errors='coerce')       # kW, actual (current fixed-floor policy)
q_loop = pd.to_numeric(df_hourly.iloc[:, 7], errors='coerce').abs()    # kW, paper's own calorimetric Q_loop
p_total = pd.to_numeric(df_hourly.iloc[:, 3], errors='coerce')
p_comp = (p_total - p_actual).clip(lower=0)

PUMP_CONST = 0.001973  # from paper Eq. 4, P[kW] = PUMP_CONST * V[L/s]^3
K_CAL = 1.0977         # calorimetric constant fit to col.7 in fig3-1 (Q = K_CAL * V[m3/h] * dT)

# 3. Data-grounded reference dT for the counterfactual controller: instead of
#    guessing a "design" dT, use the dT the borehole loop ITSELF demonstrates
#    it can sustain under genuine near-peak load (top 5% of compressor power
#    hours) with the CURRENT pump/control setup. This is not an arbitrary
#    assumption - it is what the system already proves it can do.
peak_mask = p_comp >= p_comp.quantile(0.95)
dt_ref = float(((df_hourly.iloc[:, 5].astype(float) - df_hourly.iloc[:, 6].astype(float)).abs()[peak_mask]).median())

# 4. Counterfactual: a load-following minimum-flow policy that scales flow to
#    hold dT at dt_ref for whatever heat is actually being exchanged that
#    hour, instead of a fixed 8 L/s floor. A small absolute floor (2 L/s) is
#    kept as a stand-in for pump reliability / freeze-protection minimums
#    that a real controller would still need - this is an assumption, stated
#    explicitly in the annotation below, not a measured value.
V_ABS_MIN_LS = 2.0
v_max_ls_observed = v_flow.max() / 3.6
v_needed_ls = (q_loop / (K_CAL * dt_ref)) / 3.6
v_cf_ls = v_needed_ls.clip(lower=V_ABS_MIN_LS, upper=v_max_ls_observed)
p_cf = PUMP_CONST * v_cf_ls ** 3

# 5. Cumulative energy over the year for both policies
cum_actual = p_actual.cumsum()
cum_cf = p_cf.cumsum()
total_actual = float(p_actual.sum())
total_cf = float(p_cf.sum())
savings_kwh = total_actual - total_cf
savings_pct = savings_kwh / total_actual * 100

# 6. Create figure with dartwork-mpl physical width & aspect ratio
fig, ax = plt.subplots(figsize=dm.figsize("16cm", "wide"))

# 7. Shade the gap between actual and counterfactual cumulative energy
ax.fill_between(
    datetime_series, cum_cf, cum_actual,
    color="#FCA5A5", alpha=0.35, zorder=1, label="절감 가능분"
)
ax.plot(
    datetime_series, cum_actual,
    color="#1F2937", linewidth=dm.lw(1), label="실제 (고정 최소유량 8 L/s)", zorder=3
)
ax.plot(
    datetime_series, cum_cf,
    color="#059669", linewidth=dm.lw(1), linestyle="--",
    label=f"반사실적 부하추종 제어 ($\\Delta T_{{ref}}$={dt_ref:.2f}K)", zorder=3
)

# 8. Annotation with grounding statistics and stated assumptions
ax.text(
    0.015, 0.97,
    f"$\\Delta T_{{ref}}$={dt_ref:.2f}K = 압축기 상위 5% 고부하 시간의 실측 중앙값 (가정 아님, 실측 근거)\n"
    f"가정: 유량 하한 {V_ABS_MIN_LS:.0f} L/s (신뢰성/동파방지용, 임의 가정)\n"
    f"연간 펌프 전력: 실제 {total_actual:,.0f} kWh → 반사실적 {total_cf:,.0f} kWh\n"
    f"절감 가능분: {savings_kwh:,.0f} kWh ({savings_pct:.0f}%)",
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
    "Attempt 6: 최소유량을 부하추종형으로 제어했다면? — 누적 펌프 소비전력 비교",
    fontsize=dm.fs(2),
    fontweight=dm.fw(1),
    pad=12
)
ax.set_xlabel("계측 월 (4월 ~ 이듬해 3월)", fontsize=dm.fs(0))
ax.set_ylabel("누적 펌프 소비전력 (kWh)", fontsize=dm.fs(0))

# 10. Axis limits and formatting
ax.set_xlim(datetime_series.iloc[0], datetime_series.iloc[-1])
ax.set_ylim(0, total_actual * 1.05)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m월"))
fig.autofmt_xdate(rotation=0, ha="center")

ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
ax.legend(loc="lower right", frameon=True, framealpha=0.85, fontsize=dm.fs(-1))

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
print(f"dt_ref={dt_ref:.4f}, total_actual={total_actual:.2f}, total_cf={total_cf:.2f}, "
      f"savings_kwh={savings_kwh:.2f}, savings_pct={savings_pct:.2f}")

# Copy to brain artifact directory for IDE display
brain_dir = r"C:\Users\taegyu\.gemini\antigravity-ide\brain\a77bc78a-a1a1-44c8-bd6b-2a905022a6c5"
if os.path.exists(brain_dir):
    shutil.copy(f"{save_stem}.png", os.path.join(brain_dir, f"{script_stem}.png"))
    shutil.copy(f"{save_stem}.svg", os.path.join(brain_dir, f"{script_stem}.svg"))
    print("Copied files to brain artifact directory")
