"""
FloodLedger — Pitch-Ready Offline Dashboard (Part A + Part B).
Reads ONLY from cache/ and config.yaml. Zero live API calls.
Run: streamlit run src/app/dashboard.py
"""

import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import rasterio
import streamlit as st
import yaml

st.set_page_config(
    page_title="FloodLedger — Orbit to Economy",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        margin: 0 auto;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    html, body, [class*="css"] {
        font-size: 17px;
    }
    .stMarkdown p, .stMarkdown li {
        font-size: 1.06rem !important;
        line-height: 1.6 !important;
    }
    .hero-banner {
        background: linear-gradient(90deg, #0b132b 0%, #1c2541 100%);
        border-left: 6px solid #00b4d8;
        padding: 1.4rem 1.8rem;
        border-radius: 10px;
        margin-bottom: 1.4rem;
    }
    .hero-title {
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        line-height: 1.3 !important;
        margin-bottom: 0.4rem !important;
    }
    .hero-sub {
        font-size: 1.02rem !important;
        color: #cbd5e1 !important;
    }
    div[data-testid="stMetric"] {
        background-color: #131926;
        border: 1px solid #263248;
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.98rem !important;
        color: #94a3b8 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.0rem !important;
        font-weight: 700 !important;
    }
    button[data-baseweb="tab"] p {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT_DIR / "cache"
CONFIG_PATH = ROOT_DIR / "config.yaml"


@st.cache_data
def load_cached_data() -> Dict[str, Any]:
    """Load precomputed JSON and YAML files from disk."""
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

    bench_path = CACHE_DIR / "onboard_benchmark.json"
    if bench_path.exists():
        with open(bench_path, "r", encoding="utf-8") as f:
            benchmark = json.load(f)
    else:
        benchmark = None

    return {
        "cfg": cfg,
        "exposure": exposure,
        "loss": loss,
        "gap": gap,
        "metrics": metrics,
        "benchmark": benchmark,
    }


@st.cache_data
def load_raster(path_str: str, max_dim: int = 800) -> np.ndarray:
    """Load band 1 of a GeoTIFF and downsample for instant rendering."""
    with rasterio.open(path_str) as src:
        arr = src.read(1)
    step = max(1, int(max(arr.shape) // max_dim))
    return arr[::step, ::step]


def render_sar_image(vv_db: np.ndarray, flood_mask: np.ndarray | None = None) -> np.ndarray:
    """Render VV radar backscatter as grayscale RGB with cyan flood overlay."""
    norm = np.clip((np.nan_to_num(vv_db, nan=-25.0) + 25.0) / 23.0, 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)

    if flood_mask is not None:
        m = flood_mask == 1
        rgb[m, 0] = (rgb[m, 0] * 0.2).astype(np.uint8)
        rgb[m, 1] = np.clip(rgb[m, 1] * 0.2 + 175, 0, 255).astype(np.uint8)
        rgb[m, 2] = np.clip(rgb[m, 2] * 0.2 + 210, 0, 255).astype(np.uint8)
    return rgb


data = load_cached_data()
cfg = data["cfg"]
exposure = data["exposure"]
loss = data["loss"]
gap = data["gap"]
metrics = data["metrics"]

n_days = float(gap["first_usable_pass_days"])
total_loss_cr = float(loss["rf"]["total_loss_crore"])
draft_claim_cr = float(loss["rf"]["draft_claim_crore"])
total_flood_ha = float(exposure["rf"]["total_flooded_ha"])

# ==============================================================================
# SIDEBAR: Live Sliders for Part A Assumptions
# ==============================================================================
with st.sidebar:
    st.header("Adjust Part A Assumptions")
    dc_cfg = cfg["delay_cost"]
    borrow_rate = st.slider("Borrowing Rate (%/yr)", 2.0, 18.0, float(dc_cfg["annual_borrowing_rate"]) * 100, 0.5) / 100.0
    relief_budget_cr = st.slider("Relief Budget (₹ cr)", 1.0, 50.0, float(dc_cfg["relief_budget_inr"]) / 1e7, 1.0)
    misalloc_rate = st.slider("Relief Waste (%/day)", 0.1, 5.0, float(dc_cfg["misallocation_rate_per_day"]) * 100, 0.1) / 100.0
    window_hrs = st.slider("Insurance Window (hrs)", 24, 168, int(dc_cfg["intimation_window_hours"]), 12)
    events_per_year = st.slider("Flood Events / Year", 1, 10, int(cfg["whatif"].get("events_per_year", 2)), 1)

# ==============================================================================
# HEADLINE BANNER & 4 KPI CARDS
# ==============================================================================
st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-title">
            It took <span style="color:#ff9f1c;">{n_days} days</span> to get a usable picture.
            The flood caused <span style="color:#00b4d8;">₹ {total_loss_cr:.2f} crore</span> of damage.
        </div>
        <div class="hero-sub">
            FloodLedger &nbsp;•&nbsp; {cfg['event']['name']} (Onset: {cfg['event']['onset_date']}) &nbsp;•&nbsp; Real Sentinel-1 Radar + AI
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Wait for Satellite Picture", f"{n_days} days")
k2.metric("Estimated Damage", f"₹ {total_loss_cr:.2f} cr")
k3.metric("Draft Crop Claim", f"₹ {draft_claim_cr:.2f} cr")
k4.metric("Newly Flooded Land", f"{total_flood_ha:,.0f} ha")

st.write("")

# ==============================================================================
# 4 TABS: PART A (1-3) + PART B (4)
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Flood Map & Damage",
    "2. Cost of Waiting & Speed Value",
    "3. AI Proof & Honesty",
    "4. Part B: Onboard AI & Aerospace",
])

# --- TAB 1: PART A — MAP & DAMAGE ---
with tab1:
    st.write("")
    map_col, info_col = st.columns([1.35, 1.0], gap="large")

    with map_col:
        c1, c2, c3 = st.columns([1.15, 1.15, 1.0])
        scene_choice = c1.radio("View", ["After Flood", "Before Flood"], horizontal=True)
        model_choice = c2.radio("Mask", ["Random Forest", "Baseline"], horizontal=True)
        show_overlay = c3.toggle("Flood Overlay", value=True)

        before_vv = load_raster(str(CACHE_DIR / "scenes" / "before_s1.tif"))
        after_vv = load_raster(str(CACHE_DIR / "scenes" / "after_s1.tif"))
        rf_mask = load_raster(str(CACHE_DIR / "flood_rf.tif"))
        base_mask = load_raster(str(CACHE_DIR / "flood_baseline.tif"))

        active_vv = after_vv if scene_choice == "After Flood" else before_vv
        active_mask = (rf_mask if model_choice == "Random Forest" else base_mask) if (show_overlay and scene_choice == "After Flood") else None

        scene_date = metrics["after_scene"][:10] if scene_choice == "After Flood" else metrics["before_scene"][:10]
        st.image(
            render_sar_image(active_vv, active_mask),
            caption=f"{scene_choice} ({scene_date}) • Cyan = Newly Flooded Area",
            use_container_width=True,
        )

    with info_col:
        st.subheader("Where the Damage Happened")
        exp_rf, loss_rf = exposure["rf"], loss["rf"]
        st.dataframe(
            pd.DataFrame([
                {"Land Type": "Cropland", "Flooded (ha)": exp_rf["cropland_ha"], "Loss (₹ cr)": loss_rf["crop_loss_crore"]},
                {"Land Type": "Built-up", "Flooded (ha)": exp_rf["builtup_ha"], "Loss (₹ cr)": loss_rf["built_loss_crore"]},
                {"Land Type": "Other", "Flooded (ha)": exp_rf["other_ha"], "Loss (₹ cr)": 0.0},
                {"Land Type": "TOTAL", "Flooded (ha)": exp_rf["total_flooded_ha"], "Loss (₹ cr)": loss_rf["total_loss_crore"]},
            ]),
            use_container_width=True,
            hide_index=True,
        )
        st.info(
            "**Two honest caveats:**\n"
            "- *Flooded ≠ destroyed:* damage ratios are < 1.\n"
            "- *Radar limit:* detects flooded farmland well, but urban buildings are harder to read."
        )

# --- TAB 2: PART A — COST OF WAITING & BREAKEVEN ---
with tab2:
    st.write("")
    draft_claim_inr = float(loss["rf"]["draft_claim_inr"])
    relief_budget_inr = relief_budget_cr * 1e7
    max_misalloc_inr = relief_budget_inr * float(dc_cfg["max_misallocation_fraction"])
    claim_int_per_day = draft_claim_inr * borrow_rate / 365.0
    daily_wait_cost_inr = claim_int_per_day + (relief_budget_inr * misalloc_rate)

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.subheader("What Every Day of Waiting Costs")
        st.metric("Estimated Daily Cost of Delay", f"₹ {daily_wait_cost_inr / 1e5:.2f} lakh / day")

        max_day = max(int(math.ceil(n_days)) + 4, 10)
        days = list(range(0, max_day + 1))
        total_wait_cr = [
            ((claim_int_per_day * d) + min(relief_budget_inr * misalloc_rate * d, max_misalloc_inr)) / 1e7
            for d in days
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=total_wait_cr, mode="lines+markers", name="Cumulative Cost (₹ cr)", line=dict(color="#00b4d8", width=3)))
        fig.add_vline(x=n_days, line_dash="dash", line_color="#ff9f1c", annotation_text=f"Observed ({n_days}d)")
        fig.add_vline(x=window_hrs / 24.0, line_dash="dot", line_color="#ef476f", annotation_text=f"{window_hrs}h Window")
        fig.update_layout(
            xaxis_title="Days After Flood Onset",
            yaxis_title="Cumulative Cost (₹ crore)",
            height=330,
            margin=dict(l=10, r=10, t=25, b=10),
            font=dict(size=14),
        )
        st.plotly_chart(fig, use_container_width=True)

        if n_days > (window_hrs / 24.0):
            at_risk_cr = draft_claim_inr * float(dc_cfg["share_claims_at_risk_if_late"]) / 1e7
            st.warning(f"Satellite wait (**{n_days}d**) missed the **{window_hrs}h** insurance window → **₹ {at_risk_cr:.2f} cr** in claims at risk.")
        else:
            st.success(f"Satellite wait (**{n_days}d**) was inside the **{window_hrs}h** insurance reporting window.")

    with col_right:
        st.subheader(f"Is Faster Tech Worth It? ({events_per_year} floods/yr)")
        st.caption("How much must 1 day of delay cost for each faster layer to pay for itself? *(Adjust assumptions in sidebar `>`)*")

        label_map = {
            "satellite_only": "Free Satellite Only (Baseline)",
            "commercial_sar": "+ Commercial SAR Satellite",
            "haps_sar": "+ HAPS Radar Aircraft",
            "drone_verify": "+ Drone Claim Verification",
        }
        rows = []
        for name, c in cfg["whatif"]["configs"].items():
            raw_gap = c["gap_days"]
            g_days = n_days if str(raw_gap).upper() == "FROM_M5" else float(raw_gap)
            ann_cost = float(c["annual_cost_inr"])
            saved = max(0.0, n_days - g_days)
            be = (ann_cost / (events_per_year * saved)) if (saved > 0 and ann_cost > 0) else None
            worth_it = "Yes" if (be is not None and daily_wait_cost_inr >= be) else ("No" if be is not None else "Baseline")

            rows.append({
                "Sensing Option": label_map.get(name, name),
                "Wait (days)": round(g_days, 1),
                "Breakeven": f"₹ {be / 1e5:.1f} L/day" if be is not None else "—",
                "Pays Off?": worth_it,
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --- TAB 3: PART A — AI PROOF & REAL VS ASSUMED ---
with tab3:
    st.write("")
    st.subheader("AI Accuracy on an Unseen Flood (Sen1Floods11 Bolivia Test)")
    st.dataframe(
        pd.DataFrame([
            {
                "Method": "Simple Threshold Baseline",
                "Accuracy": f"{metrics['baseline_test']['accuracy']*100:.1f}%",
                "Precision": f"{metrics['baseline_test']['precision']*100:.1f}%",
                "Recall": f"{metrics['baseline_test']['recall']*100:.1f}%",
                "F1 Score": f"{metrics['baseline_test']['f1']*100:.1f}%",
                "IoU": f"{metrics['baseline_test']['iou']*100:.1f}%",
            },
            {
                "Method": "Random Forest AI",
                "Accuracy": f"{metrics['rf_test']['accuracy']*100:.1f}%",
                "Precision": f"{metrics['rf_test']['precision']*100:.1f}%",
                "Recall": f"{metrics['rf_test']['recall']*100:.1f}%",
                "F1 Score": f"{metrics['rf_test']['f1']*100:.1f}%",
                "IoU": f"{metrics['rf_test']['iou']*100:.1f}%",
            },
        ]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Assam unlabelled consistency (Baseline vs RF IoU): **{metrics['assam_baseline_vs_rf_iou']}** • "
        f"Domain-shift note: RF was trained on global Sen1Floods11 events, not Assam."
    )

    st.divider()
    r_col, a_col = st.columns(2, gap="large")
    with r_col:
        st.subheader("Real (Measured from Data)")
        st.markdown(
            f"- Sentinel-1 radar images & flood extent\n"
            f"- Exact satellite wait time (**{n_days} days**)\n"
            f"- ESA WorldCover farmland & built-up areas\n"
            f"- Held-out AI accuracy metrics"
        )
    with a_col:
        st.subheader("Assumed (Adjustable in Sidebar)")
        st.markdown(
            "- Crop & building values per hectare / m²\n"
            "- Damage ratios (flooded ≠ destroyed)\n"
            "- Borrowing rate & relief waste rate\n"
            "- Commercial SAR, HAPS & drone costs"
        )

# --- TAB 4: PART B — ONBOARD AI & AEROSPACE ARCHITECTURE ---
with tab4:
    st.write("")
    st.subheader("1. Onboard-AI Edge & Downlink Benchmark")
    st.caption("Why process radar onboard? Satellites have tight downlink windows—sending structured flood alerts instead of raw imagery cuts bandwidth drastically.")

    if data.get("benchmark") and "onboard_rf" in data["benchmark"]:
        b = data["benchmark"]
        g_rf = b["ground_rf"]
        o_rf = b["onboard_rf"]
        pct_label = ">99.9%" if b["downlink_reduction_pct"] >= 99.95 else f"{b['downlink_reduction_pct']:.2f}%"

        b1, b2, b3 = st.columns(3)
        b1.metric(
            "Onboard Model Size",
            f"{o_rf['model_size_mb']} MB",
            f"{b['size_reduction_pct']:.1f}% smaller than Ground RF",
            delta_color="off",
        )
        b2.metric(
            "Downlink Saved",
            pct_label,
            f"{b['raw_scene_mb']} MB → {b['alert_payload_kb']} KB",
            delta_color="off",
        )
        b3.metric(
            "Onboard Inference (Proxy)",
            f"{o_rf['inference_sec']} s",
            f"{b['speedup_x']}x faster ({b['pixels_scored']:,} px)",
            delta_color="off",
        )

        st.subheader("Ground AI vs. Onboard-Edge AI (Unseen Bolivia Flood)")
        st.dataframe(
            pd.DataFrame([
                {
                    "AI Version": "Ground AI (Full Model)",
                    "Best For": "Deep analysis on ground servers",
                    "File Size": f"{g_rf['model_size_mb']} MB",
                    "Speed": f"{g_rf['inference_sec']} s",
                    "Accuracy": f"{g_rf['test_metrics']['accuracy']*100:.1f}%",
                    "Flood Score (F1)": f"{g_rf['test_metrics']['f1']*100:.1f}%",
                },
                {
                    "AI Version": "Onboard AI (Compressed)",
                    "Best For": "Fast triage on satellite edge chips",
                    "File Size": f"{o_rf['model_size_mb']} MB",
                    "Speed": f"{o_rf['inference_sec']} s",
                    "Accuracy": f"{o_rf['test_metrics']['accuracy']*100:.1f}%",
                    "Flood Score (F1)": f"{o_rf['test_metrics']['f1']*100:.1f}%",
                },
            ]),
            use_container_width=True,
            hide_index=True,
        )

        st.success(
            f"**Takeaway:** Compressing the AI shrank it by **{b['size_reduction_pct']:.1f}%** "
            f"({g_rf['model_size_mb']} MB → {o_rf['model_size_mb']} MB) and made it **{b['speedup_x']}× faster** "
            f"while keeping **{o_rf['test_metrics']['accuracy']*100:.1f}% accuracy**."
        )

        raw_sband_sec = round(b["raw_scene_mb"] / 0.25, 1)
        alert_sband_ms = round((b["alert_payload_kb"] / 256.0) * 1000.0, 2)
        st.caption(
            f"ℹ️ *{b['proxy_note']} At a 2 Mbps small-sat S-band link, downlinking the raw scene takes ~{raw_sband_sec} s "
            f"vs. ~{alert_sband_ms} ms for the structured flood alert.*"
        )
    else:
        st.info("Run `python -m src.m8_onboard_benchmark` to train and benchmark the Onboard-Edge AI.")

    st.divider()
    st.subheader("2. 3-Tier Aerospace Pipeline: Orbit → Stratosphere → Ground")

    st.dataframe(
        pd.DataFrame([
            {
                "Tier": "1. Orbit (SAR Satellite + Onboard AI)",
                "Primary Role": "Wide-area monsoon flood detection through cloud cover",
                "Coverage / Speed": "100+ km swath • Limited by orbital revisit gap",
                "Key Engineering Constraint": "Onboard power/radiation limits; fixed orbital pass times",
            },
            {
                "Tier": "2. Stratosphere (HAPS at ~20 km)",
                "Primary Role": "Persistent regional watch between satellite passes",
                "Coverage / Speed": "~35 km usable radius (~3,850 km²) at 30° min elevation",
                "Key Engineering Constraint": "High fixed annual cost; payload weight; not yet operational in India",
            },
            {
                "Tier": "3. Ground (Verification Drones)",
                "Primary Role": "Parcel-level depth & crop damage verification for claims",
                "Coverage / Speed": "Targeted high-value farms flagged by Orbit/HAPS",
                "Key Engineering Constraint": "Battery endurance, monsoon winds, and BVLOS airspace rules",
            },
        ]),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("3. Interactive HAPS Radar Geometry (Back-of-Envelope Physics)")
    st.caption("Geometric horizon at 20 km is ~500 km, but SAR cannot image usefully at 0° elevation. Adjust altitude and minimum usable elevation angle:")

    g1, g2, g3, g4 = st.columns([1.2, 1.2, 1.0, 1.0])
    haps_cfg = cfg["whatif"].get("haps", {"altitude_km": 20, "min_elevation_deg": 30})
    with g1:
        alt_km = st.slider("HAPS Altitude (km)", 15, 25, int(haps_cfg["altitude_km"]), 1)
    with g2:
        elev_deg = st.slider("Min Elevation Angle (°)", 15, 60, int(haps_cfg["min_elevation_deg"]), 5)

    radius_km = alt_km / math.tan(math.radians(elev_deg))
    area_km2 = math.pi * (radius_km ** 2)
    g3.metric("Usable Ground Radius", f"{radius_km:.1f} km")
    g4.metric("Usable Coverage Area", f"{area_km2:,.0f} km²")
