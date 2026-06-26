"""
streamlit_app.py  —  Phase 12: Interactive demo for the PS-07 pipeline.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PS-07 | Exoplanet Transit Detector",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

    /* Main background with subtle gradient */
    .stApp {
        background: radial-gradient(circle at top left, #0f172a, #020617);
        color: #e2e8f0;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #0b1121;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Glassmorphism Hero Section */
    .hero {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 3rem 3.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }

    .hero::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%; width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.05) 0%, transparent 60%);
        z-index: -1;
    }

    .hero h1 {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #e0f2fe 0%, #38bdf8 50%, #818cf8 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0 0 0.8rem 0;
        letter-spacing: -0.02em;
    }
    .hero p { color: #94a3b8; margin: 0; font-size: 1.1rem; font-weight: 500; letter-spacing: 0.01em; }

    /* Metric Cards with hover effects */
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem 1.2rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        height: 100%;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(56, 189, 248, 0.3);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.1);
    }

    .metric-card .label { font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
    .metric-card .value { font-size: 2rem; font-weight: 800; color: #f8fafc; margin: 0; }
    .metric-card .sub   { font-size: 0.85rem; color: #64748b; font-weight: 500; margin-top: 0.5rem; }

    /* Verdict Banners */
    .verdict-planet  { background: linear-gradient(135deg, rgba(6, 78, 59, 0.6), rgba(2, 44, 34, 0.8)); border:1px solid #059669; border-radius:16px; padding:1.8rem; box-shadow: 0 4px 20px rgba(5, 150, 105, 0.15); display: flex; align-items: center; justify-content: space-between; }
    .verdict-binary  { background: linear-gradient(135deg, rgba(120, 53, 15, 0.6), rgba(69, 26, 3, 0.8)); border:1px solid #d97706; border-radius:16px; padding:1.8rem; box-shadow: 0 4px 20px rgba(217, 119, 6, 0.15); display: flex; align-items: center; justify-content: space-between; }
    .verdict-noise   { background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8)); border:1px solid #475569; border-radius:16px; padding:1.8rem; display: flex; align-items: center; justify-content: space-between; }

    .verdict-title { font-size: 2.2rem; font-weight: 800; margin: 0; color: white; display: flex; align-items: center; gap: 0.8rem; }
    .verdict-subtitle { font-size: 1rem; color: rgba(255,255,255,0.8); margin: 0.3rem 0 0 0; font-weight: 500; }
    .verdict-conf { font-size: 2.5rem; font-weight: 800; color: white; opacity: 0.9; margin: 0; text-align: right; }
    .verdict-conf span { font-size: 1rem; font-weight: 500; opacity: 0.7; display: block; text-transform: uppercase; letter-spacing: 0.05em; }

    .vet-pass { color: #10b981; font-weight: 700; font-size: 1.1em; }
    .vet-fail { color: #ef4444; font-weight: 700; font-size: 1.1em; }
    .vet-na   { color: #94a3b8; }

    /* Customizing the Streamlit Button */
    .stButton>button {
        background: linear-gradient(135deg, #0ea5e9, #2563eb);
        color: white; border: none; border-radius: 10px;
        padding: 0.8rem 2rem; font-weight: 600; font-size: 1.05rem;
        width: 100%; transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.39);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5);
        background: linear-gradient(135deg, #38bdf8, #3b82f6);
        color: white;
    }
    .stButton>button:active { transform: translateY(0); }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px;
        transition: all 0.3s;
    }
    div[data-testid="stExpander"]:hover {
        border-color: rgba(255,255,255,0.2);
        background: rgba(30, 41, 59, 0.5);
    }
    
    /* Progress bars styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
    }

    hr { border-color: rgba(255,255,255,0.1); margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🪐 Exoplanet Transit Detector</h1>
  <p>Bharatiya Antariksh Hackathon 2026 &nbsp;<span style="color:#38bdf8">•</span>&nbsp; Problem Statement 07 &nbsp;<span style="color:#38bdf8">•</span>&nbsp; AI-Based Pipeline</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    target_id = st.text_input("Target ID", value="KIC 11904151",
                               help="KIC, TIC, or Kepler ID — e.g. KIC 11904151 (Kepler-10)")
    mission   = st.selectbox("Mission", ["Kepler", "K2", "TESS"])

    st.divider()
    st.markdown("### Stellar Parameters")
    star_r = st.number_input("Stellar Radius [R☉]", value=1.0, step=0.01)
    star_m = st.number_input("Stellar Mass [M☉]",   value=1.0, step=0.01)

    st.divider()
    st.markdown("### Analysis Options")
    skip_fap   = st.checkbox("Skip bootstrap FAP (faster)", value=True)
    n_fap      = st.slider("FAP bootstrap trials", 100, 1000, 200, 100, disabled=skip_fap)
    save_plots = st.checkbox("Save plots to disk", value=True)

    st.divider()
    run_btn = st.button("🚀 Run Pipeline", type="primary")

# ---------------------------------------------------------------------------
# Quick-reference known planets
# ---------------------------------------------------------------------------
KNOWN = {
    "KIC 11904151": {"period": 0.8375243, "depth_ppm": 152.0, "duration_h": 1.811,
                      "note": "Kepler-10b — confirmed hot rocky super-Earth"},
}

with st.expander("📚 Quick-reference: Known planet targets"):
    for tid, info in KNOWN.items():
        st.markdown(
            f"**{tid}** — {info['note']}  \n"
            f"Period: {info['period']} d | Depth: {info['depth_ppm']} ppm | Duration: {info['duration_h']} h"
        )

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
if run_btn:
    with st.spinner("Running pipeline — this may take a few minutes …"):
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from pipeline import run_pipeline
            result = run_pipeline(
                target_id        = target_id.strip(),
                mission          = mission,
                n_fap_trials     = n_fap,
                skip_fap         = skip_fap,
                star_radius_rsun = star_r,
                star_mass_msun   = star_m,
                save_plots       = save_plots,
                rng_seed         = 42,
            )
            st.session_state["result"]    = result
            st.session_state["target_id"] = target_id.strip()
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.exception(exc)
            st.stop()

# ---------------------------------------------------------------------------
# Display results (if available)
# ---------------------------------------------------------------------------
result: dict = st.session_state.get("result", {})

if result:
    st.markdown("---")
    st.markdown("## 📊 Results")

    # --- Classification verdict banner ---
    clf    = result.get("classification", "unknown")
    conf   = result.get("classification_confidence", 0.0) * 100
    snr    = result.get("snr", float("nan"))
    fap    = result.get("false_alarm_probability", float("nan"))

    verdict_class = (
        "verdict-planet" if "planet" in clf else
        "verdict-binary" if "binary" in clf or "false" in clf else
        "verdict-noise"
    )
    verdict_emoji = "✅" if "planet" in clf else "⚠️" if "binary" in clf or "false" in clf else "❌"

    st.markdown(f"""
    <div class="{verdict_class}" style="margin-bottom:2rem">
      <div>
        <div class="verdict-title">{verdict_emoji} {clf.replace("_", " ").title()}</div>
        <p class="verdict-subtitle">{result.get('vetting_verdict', 'Analysis complete')}</p>
      </div>
      <div class="verdict-conf">
        {conf:.1f}%
        <span>Confidence</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Key metrics grid ---
    col1, col2, col3, col4 = st.columns(4)
    def _mc(col, label, value, sub=""):
        col.markdown(f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    # Helper to format uncertainty sub-label cleanly (no ±nan)
    def _unc(val, fmt=".1e", unit=""):
        if val is None: return ""
        try:
            v = float(val)
        except (TypeError, ValueError):
            return ""
        if not np.isfinite(v) or v == 0.0:
            return ""
        return f"±{v:{fmt}} {unit}".strip()

    pu = result.get("period_uncertainty", None)
    du = result.get("depth_uncertainty_pct", None)
    dhu = result.get("duration_uncertainty_hours", None)

    _mc(col1, "Period",   f"{result['period_days']:.5f} d",
        _unc(pu, ".2e", "d") or "BLS period")
    _mc(col2, "Depth",    f"{result['depth_pct']:.4f} %",
        _unc(du, ".2e", "%") or "transit depth")
    _mc(col3, "Duration", f"{result['duration_hours']:.3f} h",
        _unc(dhu, ".3f", "h") or "transit duration")
    _mc(col4, "SNR",
        f"{snr:.1f}" if np.isfinite(snr) else "—",
        f"FAP={fap:.4f}" if np.isfinite(fap) else "FAP not computed")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Two-column layout: vetting + probabilities ---
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("### 🔍 Vetting Flags")
        vet = result.get("vetting", {})
        def _flag(name, passed, pass_msg, fail_msg):
            icon  = '<span class="vet-pass">✔</span>' if passed else '<span class="vet-fail">✘</span>'
            label = pass_msg if passed else fail_msg
            st.markdown(f"{icon} **{name}** — {label}", unsafe_allow_html=True)

        _flag("Odd-Even Test",
              vet.get("odd_even_consistent", True),
              "Depths consistent (planet-like)",
              "Depths differ (possible EB!)")
        _flag("Secondary Eclipse",
              not vet.get("secondary_eclipse_detected", False),
              "No secondary eclipse",
              "Secondary eclipse detected (possible EB!)")
        _flag("Centroid Shift",
              not vet.get("centroid_shift_detected", False),
              "No centroid shift",
              "Centroid shifted (possible blend!)")

        st.markdown(f"**Vetting score:** {result.get('vetting_score', '?')} / 5")

    with c_right:
        st.markdown("### 🤖 Class Probabilities")
        probs = result.get("class_probabilities", {})
        label_map = {"PC": "Planet Candidate", "AFP": "Eclipsing Binary / FP", "NTP": "Noise"}
        colors = {"PC": "#4fc3f7", "AFP": "#ef9a9a", "NTP": "#b0bec5"}
        for k, v in sorted(probs.items(), key=lambda x: -x[1]):
            bar_pct = int(v * 100)
            label   = label_map.get(k, k)
            color   = colors.get(k, "#78909c")
            st.markdown(f"**{label}** — {v*100:.1f}%")
            st.markdown(
                f'<div style="background:#1e2a3a;border-radius:6px;height:12px;margin-bottom:8px">'
                f'<div style="background:{color};width:{bar_pct}%;height:100%;border-radius:6px"></div>'
                f'</div>', unsafe_allow_html=True
            )

    # --- Known-value comparison ---
    if "known_value_comparison" in result:
        st.markdown("### 📐 Recovery vs. Published Values")
        kvc = result["known_value_comparison"]
        cols = st.columns(3)
        for col, (pub_k, rec_k, err_k, unit) in zip(cols, [
            ("published_period_d",   "recovered_period_d",   "period_error_pct",   "d"),
            ("published_depth_ppm",  "recovered_depth_ppm",  "depth_error_pct",    "ppm"),
            ("published_duration_h", "recovered_duration_h", "duration_error_pct", "h"),
        ]):
            label = pub_k.split("_")[1].title()
            err   = kvc[err_k]
            status = "🟢" if err < 5 else "🟡" if err < 15 else "🔴"
            col.metric(
                label=f"{status} {label}",
                value=f"{kvc[rec_k]} {unit}",
                delta=f"{err:.1f}% vs published",
            )

    # --- Plots ---
    PLOTS_DIR = Path(__file__).parent / "plots"
    tag = result.get("target_id", "target").replace(" ", "_")
    plot_files = {
        "Detrended Light Curve": PLOTS_DIR / f"{tag}_detrending.png",
        "BLS Periodogram":       PLOTS_DIR / f"{tag}_periodogram.png",
        "Phase-Folded Transit":  PLOTS_DIR / f"{tag}_phasefold.png",
        "Transit Model Fit":     PLOTS_DIR / f"{tag}_transit_model.png",
        "Vetting Summary":       PLOTS_DIR / f"{tag}_vetting.png",
        "FAP Distribution":      PLOTS_DIR / f"{tag}_fap_distribution.png",
    }

    available = {k: v for k, v in plot_files.items() if v.exists()}
    if available:
        st.markdown("### 📈 Plots")
        names = list(available.keys())
        tab_objects = st.tabs(names)
        for tab, name in zip(tab_objects, names):
            with tab:
                st.image(str(available[name]), use_column_width=True)

    # --- Raw JSON ---
    with st.expander("📄 Raw JSON result"):
        st.json(result)

    # --- Download button ---
    json_str = json.dumps(result, indent=2, default=str)
    st.download_button(
        "⬇️  Download result JSON",
        data=json_str,
        file_name=f"{tag}_result.json",
        mime="application/json",
    )

else:
    # Placeholder state
    st.markdown("""
    <div style="text-align:center; padding:5rem 2rem; background: rgba(30, 41, 59, 0.3); border-radius: 20px; border: 1px dashed rgba(255,255,255,0.1); margin-top: 2rem;">
      <div style="font-size:4.5rem; margin-bottom: 1rem; opacity: 0.8; animation: float 6s ease-in-out infinite;">🔭</div>
      <h3 style="color: white; font-weight: 700; margin-bottom: 0.5rem;">Ready to Analyze</h3>
      <p style="font-size:1.1rem; color: #94a3b8;">Enter a target ID in the sidebar and click <strong>Run Pipeline</strong> to begin.</p>
      <style>
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-15px); }
            100% { transform: translateY(0px); }
        }
      </style>
    </div>
    """, unsafe_allow_html=True)

    # Show pre-existing results if any
    RESULTS_DIR = Path(__file__).parent / "results"
    prev = sorted(RESULTS_DIR.glob("*.json"))
    if prev:
        st.markdown("### 🗂️ Previous results")
        for p in prev:
            with st.expander(p.stem):
                st.json(json.loads(p.read_text()))
