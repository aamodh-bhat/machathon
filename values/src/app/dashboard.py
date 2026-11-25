"""
FloodLedger Offline Pitch Dashboard (Streamlit).
Reads ONLY from cache/ and config.yaml. Zero live API or map-tile calls.
Run from project root: streamlit run app/dashboard.py
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import rasterio
import streamlit as st
import yaml

st.set_page_config(
    page_title="FloodLedger — Orbit to Economy",
    page_icon="🛰️",
    layout="wide",
)

# Large readable styling for judges viewing from a distance
st.markdown(
    """
    <style>
    .headline-box {
        background-color: #0e1117;
        border-left: 6px solid #00b4d8;
        padding: 1.2rem 1.5rem;
        border-radius: 6px;
        margin-bottom: 1.2rem;
    }
    .headline-text {
        font-size: 2.1rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        line-height: 1.3;
    }
    .sub-text {
        font-size: 1.1rem;
        color: #b0bec5;
        margin-top: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT_DIR = Path(__file__).resolve().parents[2]  # points to values/
CACHE_DIR = ROOT_DIR / "cache"
CONFIG_PATH = ROOT_DIR / "config.yaml"


@st.cache_data
def load_all_cached_data() -> Dict[str, Any]:
    """Load all precomputed JSON, CSV, and YAML files from disk."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(CACHE_DIR / "exposure.json", "r", encoding="utf-8") as f:
        exposure = json.load(f)
    with open(CACHE_DIR / "loss.json", "r", encoding="utf-8") as f:
        loss = json.load(f)
    with open(CACHE_DIR / "observation_gap.json", "r", encoding="utf-8") as f:
        gap = json.load(f)
    with open(CACHE_DIR / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    whatif_df = pd.read_csv(CACHE_DIR / "whatif.csv")
    return {
        "cfg": cfg,
        "exposure": exposure,
        "loss": loss,
        "gap": gap,
        "metrics": metrics,
        "whatif_df": whatif_df,
    }


@st.cache_data
def load_raster_for_display(path_str: str, max_dim: int = 800) -> np.ndarray:
    """Load band 1 of a GeoTIFF and downsample if needed for instant rendering."""
    with rasterio.open(path_str) as src:
        arr = src.read(1)
    h, w = arr.shape
    step = max(1, int(max(h, w) // max_dim))
    return arr[::step, ::step]


def render_sar_with_overlay(
    vv_db: np.ndarray,
    flood_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert VV backscatter (dB) to a crisp grayscale RGB image and optionally
    blend a bright cyan flood overlay (100% offline, no external map tiles).
    """
    # Normalize dB [-25, -2] to [0, 255]
    norm = np.clip((np.nan_to_num(vv_db, nan=-25.0) + 25.0) / 23.0, 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)

    if flood_mask is not None:
        m = flood_mask == 1
        # Blend bright cyan (0, 212, 255) over flooded pixels
        rgb[m, 0] = (rgb[m, 0] * 0.25).astype(np.uint8)
        rgb[m, 1] = np.clip(rgb[m, 1] * 0.25 + 160, 0, 255).astype(np.uint8)
        rgb[m, 2] = np.clip(rgb[m, 2] * 0.25 + 195, 0, 255).astype(np.uint8)

    return rgb


data = load_all_cached_data()
cfg = data["cfg"]
exposure = data["exposure"]
loss = data["loss"]
gap = data["gap"]
metrics = data["metrics"]

n_days = gap["first_usable_pass_days"]
total_loss_cr = loss["rf"]["total_loss_crore"]
draft_claim_cr = loss["rf"]["draft_claim_crore"]

# ==============================================================================
# 1. HEADLINE BANNER (Visible without scrolling)
# ==============================================================================
st.markdown(
    f"""
    <div class="headline-box">
        <p class="headline-text">
            It took <span style="color:#ff9f1c;">{n_days} days</span> to get a usable picture.
            The flood caused <span style="color:#00b4d8;">₹ {total_loss_cr:.2f} crore</span> of damage.
        </p>
        <p class="sub-text">
            Event: {cfg['event']['name']} (Onset: {cfg['event']['onset_date']}) •
            Draft Crop Insurance Claim: ₹ {draft_claim_cr:.2f} crore •
            100% Offline Cache Demo
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. BEFORE / AFTER MAP
# ==============================================================================
st.header("1. Satellite Radar Before & After")
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    scene_choice = st.radio("Scene", ["After Flood", "Before Flood"], horizontal=True)
with c2:
    show_overlay = st.toggle("Show Flood Overlay", value=True)
with c3:
    model_choice = st.radio("Flood Mask", ["Random Forest", "Threshold Baseline"], horizontal=True)

before_vv = load_raster_for_display(str(CACHE_DIR / "scenes" / "before_s1.tif"))
after_vv = load_raster_for_display(str(CACHE_DIR / "scenes" / "after_s1.tif"))
rf_mask = load_raster_for_display(str(CACHE_DIR / "flood_rf.tif"))
base_mask = load_raster_for_display(str(CACHE_DIR / "flood_baseline.tif"))

active_vv = after_vv if scene_choice == "After Flood" else before_vv
active_mask = None
if show_overlay and scene_choice == "After Flood":
    active_mask = rf_mask if model_choice == "Random Forest" else base_mask

img_rgb = render_sar_with_overlay(active_vv, active_mask)
st.image(
    img_rgb,
    caption=(
        f"{scene_choice} ({metrics['after_scene'][:10] if scene_choice == 'After Flood' else metrics['before_scene'][:10]}) "
        f"| Cyan = Newly Flooded Area ({model_choice})"
    ),
    use_container_width=True,
)

st.divider()

# ==============================================================================
# 3. LOSS TABLE & 4. MODEL PANEL
# ==============================================================================
col_loss, col_model = st.columns([1.15, 1.0])

with col_loss:
    st.header("2. Flood Exposure & Rupee Loss")
    exp_rf = exposure["rf"]
    loss_rf = loss["rf"]
    loss_df = pd.DataFrame(
        [
            {"Land Cover Class": "Cropland", "Flooded Area (ha)": exp_rf["cropland_ha"], "Estimated Loss (₹ crore)": loss_rf["crop_loss_crore"]},
            {"Land Cover Class": "Built-up", "Flooded Area (ha)": exp_rf["builtup_ha"], "Estimated Loss (₹ crore)": loss_rf["built_loss_crore"]},
            {"Land Cover Class": "Other", "Flooded Area (ha)": exp_rf["other_ha"], "Estimated Loss (₹ crore)": 0.0},
            {"Land Cover Class": "TOTAL", "Flooded Area (ha)": exp_rf["total_flooded_ha"], "Estimated Loss (₹ crore)": loss_rf["total_loss_crore"]},
        ]
    )
    st.dataframe(loss_df, use_container_width=True, hide_index=True)
    st.info(
        f"**Draft Crop Insurance Claim:** ₹ {loss_rf['draft_claim_crore']:.2f} crore  \n"
        f"*{loss['caveat']}*  \n"
        f"*{exposure['radar_limit_note']}*"
    )

with col_model:
    st.header("3. Model Validation (Held-Out Test)")
    m_df = pd.DataFrame(
        [
            {"Model": "Threshold Baseline", **metrics["baseline_test"]},
            {"Model": "Random Forest", **metrics["rf_test"]},
        ]
    )
    st.dataframe(m_df, use_container_width=True, hide_index=True)
    st.caption(
        f"Evaluated on **{metrics['test_split']}**. "
        f"Assam unlabelled consistency (Baseline vs RF IoU): **{metrics['assam_baseline_vs_rf_iou']}** | "
        f"Model size: **{metrics['rf_model_size_mb']} MB**.  \n"
        f"⚠️ *{metrics['domain_shift_note']}*"
    )

st.divider()

# ==============================================================================
# 5. COST-OF-DELAY CURVE (Live Adjustable Sliders)
# ==============================================================================
st.header("4. What Does Waiting Cost? (Cost-of-Delay Curve)")
dc_cfg = cfg["delay_cost"]

s1, s2, s3, s4 = st.columns(4)
with s1:
    borrow_rate = st.slider(
        "Annual Borrowing Rate (%)", 2.0, 18.0, float(dc_cfg["annual_borrowing_rate"]) * 100, 0.5
    ) / 100.0
with s2:
    relief_budget_cr = st.slider(
        "Relief Budget (₹ crore)", 1.0, 50.0, float(dc_cfg["relief_budget_inr"]) / 1e7, 1.0
    )
with s3:
    misalloc_rate = st.slider(
        "Relief Misallocation / Day (%)", 0.1, 5.0, float(dc_cfg["misallocation_rate_per_day"]) * 100, 0.1
    ) / 100.0
with s4:
    window_hrs = st.slider(
        "Insurance Reporting Window (hrs)", 24, 168, int(dc_cfg["intimation_window_hours"]), 12
    )

draft_claim_inr = float(loss["rf"]["draft_claim_inr"])
relief_budget_inr = relief_budget_cr * 1e7
max_misalloc_inr = relief_budget_inr * float(dc_cfg["max_misallocation_fraction"])
claim_int_per_day = draft_claim_inr * borrow_rate / 365.0

max_day = max(int(math.ceil(n_days)) + 5, 10)
days = list(range(0, max_day + 1))
ch1_cr = [(claim_int_per_day * d) / 1e7 for d in days]
ch2_cr = [min(relief_budget_inr * misalloc_rate * d, max_misalloc_inr) / 1e7 for d in days]

fig_delay = go.Figure()
fig_delay.add_trace(go.Scatter(x=days, y=ch1_cr, mode="lines+markers", name="Ch 1: Claim Payout Interest (₹ cr)"))
fig_delay.add_trace(go.Scatter(x=days, y=ch2_cr, mode="lines+markers", name="Ch 2: Relief Misallocation (₹ cr)"))
fig_delay.add_vline(
    x=n_days, line_dash="dash", line_color="#ff9f1c",
    annotation_text=f"Observed Gap ({n_days}d)", annotation_position="top right"
)
fig_delay.add_vline(
    x=window_hrs / 24.0, line_dash="dot", line_color="#ef476f",
    annotation_text=f"Ch 3: {window_hrs}h Window", annotation_position="top left"
)
fig_delay.update_layout(
    xaxis_title="Days After Flood Onset",
    yaxis_title="Cumulative Cost (₹ crore)",
    height=380,
    margin=dict(l=20, r=20, t=30, b=20),
)
st.plotly_chart(fig_delay, use_container_width=True)

exceeded = n_days > (window_hrs / 24.0)
claims_at_risk_cr = (draft_claim_inr * float(dc_cfg["share_claims_at_risk_if_late"]) / 1e7) if exceeded else 0.0
if exceeded:
    st.warning(
        f"**Channel 3 Alert:** Observed gap (**{n_days} days**) exceeds the **{window_hrs}-hour** reporting window. "
        f"Estimated claims at risk (assumed {dc_cfg['share_claims_at_risk_if_late']:.0%}): **₹ {claims_at_risk_cr:.2f} crore**."
    )
else:
    st.success(f"**Channel 3:** Observed gap ({n_days} days) is within the {window_hrs}-hour reporting window.")

st.divider()

# ==============================================================================
# 6. WHAT-IF SENSING & BREAKEVEN PANEL
# ==============================================================================
st.header("5. What Is Faster Sensing Worth? (Breakeven Analysis)")

events_per_year = st.slider(
    "Major Flood Events per Year", 1, 10, int(cfg["whatif"].get("events_per_year", 2)), 1
)

est_delay_cost_per_day_inr = claim_int_per_day + (relief_budget_inr * misalloc_rate)
st.write(
    f"Current estimated daily cost of delay (Ch 1 + Ch 2): "
    f"**₹ {est_delay_cost_per_day_inr:,.0f} / day** (₹ {est_delay_cost_per_day_inr / 1e7:.4f} crore / day)"
)

whatif_rows = []
sens_rows = []
for name, c in cfg["whatif"]["configs"].items():
    raw_gap = c["gap_days"]
    g_days = n_days if str(raw_gap).upper() == "FROM_M5" else float(raw_gap)
    ann_cost = float(c["annual_cost_inr"])
    saved = max(0.0, n_days - g_days)
    be = (ann_cost / (events_per_year * saved)) if (saved > 0 and ann_cost > 0) else (0.0 if saved > 0 else None)
    clears = (est_delay_cost_per_day_inr >= be) if be is not None else None

    whatif_rows.append(
        {
            "Configuration": name,
            "Gap (days)": round(g_days, 2),
            "Days Saved / Event": round(saved, 2),
            "Annual Cost (₹ cr)": round(ann_cost / 1e7, 2),
            "Breakeven (₹/day)": round(be, 0) if be is not None else None,
            "Clears Threshold?": "Yes" if clears is True else ("No" if clears is False else "Baseline / Verify"),
        }
    )
    if saved > 0 and ann_cost > 0:
        for ev_cnt in range(1, 11):
            sens_rows.append(
                {
                    "Configuration": name,
                    "Events / Year": ev_cnt,
                    "Breakeven Cost (₹ lakh / day)": (ann_cost / (ev_cnt * saved)) / 1e5,
                }
            )

w_col1, w_col2 = st.columns([1.1, 1.0])
with w_col1:
    st.dataframe(pd.DataFrame(whatif_rows), use_container_width=True, hide_index=True)
    haps_cfg = cfg["whatif"].get("haps", {"altitude_km": 20, "min_elevation_deg": 30})
    haps_rad = round(haps_cfg["altitude_km"] / math.tan(math.radians(haps_cfg["min_elevation_deg"])), 1)
    st.caption(
        f"HAPS geometry: at {haps_cfg['altitude_km']} km altitude & {haps_cfg['min_elevation_deg']}° min elevation, "
        f"useful ground radius ≈ **{haps_rad} km** (not the 500 km horizon)."
    )

with w_col2:
    if sens_rows:
        fig_sens = px.line(
            pd.DataFrame(sens_rows),
            x="Events / Year",
            y="Breakeven Cost (₹ lakh / day)",
            color="Configuration",
            markers=True,
            title="Breakeven Sensitivity vs. Events / Year",
        )
        fig_sens.add_hline(
            y=est_delay_cost_per_day_inr / 1e5,
            line_dash="dash",
            line_color="#00b4d8",
            annotation_text="Est. Delay Cost",
        )
        fig_sens.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_sens, use_container_width=True)
    else:
        st.info("Set non-zero `annual_cost_inr` in `config.yaml` for Commercial SAR / HAPS to plot breakeven curves.")

st.divider()

# ==============================================================================
# 7. REAL VS ASSUMED PANEL
# ==============================================================================
st.header("6. Real (Measured) vs. Assumed (Adjustable)")
r_col, a_col = st.columns(2)
with r_col:
    st.subheader("Real (Measured from Data)")
    st.markdown(
        """
        - Sentinel-1 radar scenes (VV/VH backscatter)
        - Flooded extent (Random Forest & Threshold Baseline)
        - ESA WorldCover 10m land cover (cropland / built-up)
        - Real STAC satellite pass timestamps (**N days**)
        - Held-out Sen1Floods11 model metrics
        """
    )
with a_col:
    st.subheader("Assumed (Stated & Adjustable)")
    st.markdown(
        """
        - Crop value per hectare & built-up value per m²
        - Damage ratios (flooded ≠ destroyed)
        - Farm borrowing rate & relief misallocation rate
        - Insurance reporting window & claims-at-risk share
        - Commercial SAR, HAPS, and drone annual costs
        """
    )

st.divider()

# ==============================================================================
# 8. CLOSING ARCHITECTURE
# ==============================================================================
st.header("7. Future Architecture: Orbit → Stratosphere → Ground")
st.markdown(
    """
    1. **Satellite (Sentinel-1 / Commercial SAR):** Wide-area monsoon detection—flags flooded districts through cloud cover.
    2. **HAPS (High-Altitude Platform Station):** Persistent watch over a ~35 km radius between orbital passes.
    3. **Drones:** Targeted parcel-level depth & damage verification for rapid insurance claim settlement.
    """
)