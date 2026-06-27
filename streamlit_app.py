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
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

if "history" not in st.session_state:
    st.session_state["history"] = []

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

    .stApp {
        background: radial-gradient(circle at top left, #0f172a, #020617);
        color: #e2e8f0;
    }

    section[data-testid="stSidebar"] {
        background: #0b1121;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

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

    div[data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px;
        transition: all 0.3s;
    }
    div[data-testid="stExpander"]:hover {
        border-color: rgba(255,255,255,0.2);
        background: rgba(30, 41, 59, 0.5);
    }

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
# Sidebar — global settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Global Settings")
    mission = st.selectbox("Mission", ["Kepler", "K2", "TESS"])

    with st.expander("⚙️ Advanced Settings"):
        st.markdown("### Stellar Parameters")
        star_r = st.number_input("Stellar Radius [R☉]", value=1.0, step=0.01)
        star_m = st.number_input("Stellar Mass [M☉]",   value=1.0, step=0.01)

        st.divider()
        st.markdown("### Analysis Options")
        skip_fap   = st.checkbox("Skip bootstrap FAP (faster)", value=True)
        n_fap      = st.slider("FAP bootstrap trials", 100, 1000, 200, 100, disabled=skip_fap)
        save_plots = st.checkbox("Save plots to disk", value=True)

    if st.session_state["history"]:
        st.divider()
        st.markdown("### 🕒 Recent Runs")
        for hist_item in reversed(st.session_state["history"]):
            if st.button(f"Load {hist_item['target_id']}", key=f"hist_{hist_item['target_id']}"):
                st.session_state["result"] = hist_item["result"]
                st.session_state["target_id"] = hist_item["target_id"]

# ---------------------------------------------------------------------------
# App Tabs
# ---------------------------------------------------------------------------
tab_single, tab_bulk = st.tabs(["🎯 Single Target Analysis", "📦 Bulk Processing"])

# ===========================================================================
# TAB 1: Single Target
# ===========================================================================
with tab_single:
    st.markdown("### Target Configuration")
    col1, col2 = st.columns([2, 1])
    with col1:
        target_options = ["KIC 11904151", "KIC 3733346", "Custom"]
        selected_target = st.selectbox("Target Selection", target_options)
        if selected_target == "Custom":
            target_id = st.text_input("Custom Target ID", value="KIC 11904151", help="KIC, TIC, or Kepler ID")
        else:
            target_id = selected_target
    with col2:
        st.write("")
        st.write("")
        run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

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

    # -----------------------------------------------------------------------
    # Run pipeline
    # -----------------------------------------------------------------------
    if run_btn:
        with st.status("Running pipeline...", expanded=True) as status:
            try:
                sys.path.insert(0, str(Path(__file__).parent))
                from pipeline import run_pipeline

                def progress_cb(msg: str) -> None:
                    status.update(label=msg)

                result = run_pipeline(
                    target_id        = target_id.strip(),
                    mission          = mission,
                    n_fap_trials     = n_fap,
                    skip_fap         = skip_fap,
                    star_radius_rsun = star_r,
                    star_mass_msun   = star_m,
                    save_plots       = save_plots,
                    rng_seed         = 42,
                    progress_cb      = progress_cb,
                    return_plot_data = True,
                )
                st.session_state["result"]    = result
                st.session_state["target_id"] = target_id.strip()

                if not any(h["target_id"] == target_id.strip() for h in st.session_state["history"]):
                    st.session_state["history"].append({"target_id": target_id.strip(), "result": result})

                status.update(label="Analysis Complete!", state="complete", expanded=False)
            except Exception as exc:
                status.update(label=f"Pipeline error: {exc}", state="error")
                st.error(f"Pipeline error: {exc}")
                st.exception(exc)
                st.stop()

    # -----------------------------------------------------------------------
    # Display results (if available)
    # -----------------------------------------------------------------------
    result: dict = st.session_state.get("result", {})

    if result:
        st.markdown("---")
        st.markdown("## 📊 Results")

        clf   = result.get("classification", "unknown")
        conf  = result.get("classification_confidence", 0.0) * 100
        snr   = result.get("snr", float("nan"))
        fap   = result.get("false_alarm_probability", float("nan"))

        verdict_class = (
            "verdict-planet" if "planet" in clf else
            "verdict-binary" if "binary" in clf or "false" in clf else
            "verdict-noise"
        )
        verdict_emoji = "✅" if "planet" in clf else "⚠️" if "binary" in clf or "false" in clf else "❌"

        _rp_earth  = result.get("planet_radius_earth", None)
        _n_tr      = result.get("n_transits_observed", "?")
        _redchi    = result.get("fit_redchi", None)
        _fit_ok    = result.get("fit_ok", False)
        _baseline  = result.get("baseline_days", "?")
        _wtime     = result.get("pipeline_wall_time_s", "?")
        _depth_ppm = result.get("depth_ppm", None)
        tag        = result.get("target_id", "target").replace(" ", "_")

        fit_badge_color = "#10b981" if _fit_ok else "#ef4444"
        fit_badge_text  = f"redχ²={_redchi:.2f}" if _redchi and np.isfinite(float(_redchi)) else "fit uncertain"
        rp_str = f"{_rp_earth} R⊕" if _rp_earth else "—"

        st.markdown(f"""
        <div class="{verdict_class}" style="margin-bottom:2rem">
          <div>
            <div class="verdict-title">{verdict_emoji} {clf.replace("_", " ").title()}</div>
            <p class="verdict-subtitle">{result.get('vetting_verdict', 'Analysis complete')}</p>
            <p style="margin:0.6rem 0 0 0;font-size:0.9rem;color:rgba(255,255,255,0.6)">
              SNR = <strong style="color:white">{snr:.1f}</strong>
              &nbsp;•&nbsp; Rp = <strong style="color:white">{rp_str}</strong>
              &nbsp;•&nbsp; {_n_tr} transits
              &nbsp;•&nbsp; <span style="color:{fit_badge_color};font-weight:700">{fit_badge_text}</span>
            </p>
          </div>
          <div class="verdict-conf">
            {conf:.1f}%
            <span>Confidence</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)

        def _mc(col, label, value, sub="", accent="#38bdf8"):
            col.markdown(f"""
            <div class="metric-card">
              <div class="label">{label}</div>
              <div class="value" style="font-size:1.6rem;color:{accent}">{value}</div>
              <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        def _unc(val, fmt=".1e", unit=""):
            if val is None:
                return ""
            try:
                v = float(val)
            except (TypeError, ValueError):
                return ""
            if not np.isfinite(v) or v == 0.0:
                return ""
            return f"±{v:{fmt}} {unit}".strip()

        pu       = result.get("period_uncertainty", None)
        du       = result.get("depth_uncertainty_pct", None)
        dhu      = result.get("duration_uncertainty_hours", None)
        fap_note = result.get("fap_note", "")

        _mc(col1, "Period",   f"{result['period_days']:.5f} d",
            _unc(pu, ".2e", "d") or "BLS period")
        _mc(col2, "Depth",
            f"{result['depth_pct']:.4f} %",
            (f"{_depth_ppm:.1f} ppm" if _depth_ppm else "") + (" " + _unc(du, ".2e", "%") if _unc(du, ".2e", "%") else ""))
        _mc(col3, "Duration", f"{result['duration_hours']:.3f} h",
            _unc(dhu, ".3f", "h") or "batman-fitted")
        _mc(col4, "SNR",
            f"{snr:.1f}" if np.isfinite(snr) else "—",
            f"FAP = {fap:.4f}" if np.isfinite(fap) else "FAP not computed")
        _mc(col5, "Planet Radius",
            rp_str,
            f"Rp/Rs = {result.get('rp_rs', '?')}",
            accent="#a78bfa")

        if fap_note:
            st.caption(f"ℹ️ {fap_note}")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:1.5rem">
          <div style="background:rgba(30,41,59,0.6);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:0.5rem 1rem;font-size:0.85rem;color:#94a3b8">
            🔭 <strong style="color:#e2e8f0">{_n_tr}</strong> transits observed
          </div>
          <div style="background:rgba(30,41,59,0.6);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:0.5rem 1rem;font-size:0.85rem;color:#94a3b8">
            📅 <strong style="color:#e2e8f0">{_baseline}</strong> day baseline
          </div>
          <div style="background:rgba(30,41,59,0.6);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:0.5rem 1rem;font-size:0.85rem;color:#94a3b8">
            ⏱️ Pipeline: <strong style="color:#e2e8f0">{_wtime}s</strong>
          </div>
          <div style="background:rgba(30,41,59,0.6);border:1px solid rgba({('6,78,59' if _fit_ok else '127,29,29')},0.6);border-radius:10px;padding:0.5rem 1rem;font-size:0.85rem">
            🔬 Fit: <strong style="color:{fit_badge_color}">{fit_badge_text}</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

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

            vs = result.get('vetting_score', '?')
            vs_color = "#10b981" if isinstance(vs, int) and vs >= 3 else "#f59e0b" if isinstance(vs, int) and vs >= 0 else "#ef4444"
            st.markdown(
                f"**Vetting score:** "
                f'<span style="color:{vs_color};font-size:1.2em;font-weight:800">{vs}</span>'
                f' <span style="color:#64748b">/ 5 &nbsp;(range −5 to +5)</span>',
                unsafe_allow_html=True
            )

        with c_right:
            st.markdown("### 🤖 Class Probabilities")
            probs = result.get("class_probabilities", {})
            label_map = {"PC": "Planet Candidate", "AFP": "Eclipsing Binary / FP", "NTP": "Noise"}
            colors = {"PC": "#4fc3f7", "AFP": "#ef9a9a", "NTP": "#b0bec5"}
            for k, v in sorted(probs.items(), key=lambda x: -x[1]):
                label = label_map.get(k, k)
                color = colors.get(k, "#78909c")
                bar_pct = int(v * 100)
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">'
                    f'<div style="width:140px;font-size:0.85rem;color:#cbd5e1;font-weight:600">{label}</div>'
                    f'<div style="flex:1;background:#1e2a3a;border-radius:8px;height:14px">'
                    f'<div style="background:{color};width:{bar_pct}%;height:100%;border-radius:8px;'
                    f'transition:width 0.6s ease"></div></div>'
                    f'<div style="width:46px;text-align:right;font-size:0.9rem;font-weight:700;color:{color}">{v*100:.1f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # --- Known-value comparison ---
        if "known_value_comparison" in result:
            st.markdown("### 📐 Recovery vs. Published Values")
            kvc  = result["known_value_comparison"]
            cols = st.columns(3)
            for col, (pub_k, rec_k, err_k, unit) in zip(cols, [
                ("published_period_d",   "recovered_period_d",   "period_error_pct",   "d"),
                ("published_depth_ppm",  "recovered_depth_ppm",  "depth_error_pct",    "ppm"),
                ("published_duration_h", "recovered_duration_h", "duration_error_pct", "h"),
            ]):
                label  = pub_k.split("_")[1].title()
                err    = kvc[err_k]
                status_icon = "🟢" if err < 5 else "🟡" if err < 15 else "🔴"
                col.metric(
                    label=f"{status_icon} {label}",
                    value=f"{kvc[rec_k]} {unit}",
                    delta=f"{err:.1f}% vs published",
                )

        # --- Planet Size Visualizer ---
        if _rp_earth:
            st.markdown("### 🪐 Planet Size Comparison")
            scale      = min(max(float(_rp_earth), 0.2), 15.0)
            earth_px   = 30
            planet_px  = int(earth_px * scale)
            st.markdown(f"""
            <div style="display:flex;align-items:flex-end;gap:40px;margin:2rem 0;padding:2rem;background:rgba(30,41,59,0.3);border-radius:16px;border:1px solid rgba(255,255,255,0.05);justify-content:center">
                <div style="text-align:center">
                    <div style="width:{earth_px}px;height:{earth_px}px;border-radius:50%;background:radial-gradient(circle at 30% 30%, #60a5fa, #2563eb);margin:0 auto 1rem auto;box-shadow:inset -5px -5px 10px rgba(0,0,0,0.5), 0 0 15px rgba(37,99,235,0.4)"></div>
                    <div style="color:#94a3b8;font-size:0.9rem;font-weight:600">Earth (1 R⊕)</div>
                </div>
                <div style="text-align:center">
                    <div style="width:{planet_px}px;height:{planet_px}px;border-radius:50%;background:radial-gradient(circle at 30% 30%, #fcd34d, #d97706);margin:0 auto 1rem auto;box-shadow:inset -10px -10px 20px rgba(0,0,0,0.5), 0 0 25px rgba(217,119,6,0.3)"></div>
                    <div style="color:#f8fafc;font-size:1.1rem;font-weight:700">Detected ({float(_rp_earth):.2f} R⊕)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # --- Plots ---
        PLOTS_DIR  = Path(__file__).parent / "plots"
        plot_files = {
            "Detrended Light Curve": PLOTS_DIR / f"{tag}_detrending.png",
            "BLS Periodogram":       PLOTS_DIR / f"{tag}_periodogram.png",
            "Phase-Folded Transit":  PLOTS_DIR / f"{tag}_phasefold.png",
            "Transit Model Fit":     PLOTS_DIR / f"{tag}_transit_model.png",
            "Vetting Summary":       PLOTS_DIR / f"{tag}_vetting.png",
            "FAP Distribution":      PLOTS_DIR / f"{tag}_fap_distribution.png",
        }

        available = {k: v for k, v in plot_files.items() if v.exists()}
        if available or "plot_data" in result:
            st.markdown("### 📈 Plots")

            names: list = []
            if "plot_data" in result:
                pd_data = result["plot_data"]
                if pd_data.get("time") and pd_data.get("detrended_flux"):
                    names.append("Interactive Detrended LC")
                if pd_data.get("periods") and pd_data.get("power_spectrum"):
                    names.append("Interactive Periodogram")
                if pd_data.get("phase"):
                    names.append("Interactive Phase-Fold")
            names.extend(list(available.keys()))

            tab_objects = st.tabs(names)
            for tab, name in zip(tab_objects, names):
                with tab:
                    if name == "Interactive Detrended LC":
                        pd_data = result["plot_data"]
                        n_pts   = len(pd_data["time"])
                        stride  = max(1, n_pts // 10000)
                        fig = go.Figure()
                        fig.add_trace(go.Scattergl(
                            x=pd_data["time"][::stride],
                            y=pd_data["detrended_flux"][::stride],
                            mode="markers",
                            marker=dict(size=3, color="#38bdf8", opacity=0.7),
                            name="Detrended Flux",
                        ))
                        fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            xaxis_title="Time [days]",
                            yaxis_title="Normalized Flux",
                            height=400,
                            margin=dict(l=40, r=40, t=40, b=40),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif name == "Interactive Periodogram":
                        pd_data = result["plot_data"]
                        fig = go.Figure()
                        fig.add_trace(go.Scattergl(
                            x=pd_data["periods"],
                            y=pd_data["power_spectrum"],
                            mode="lines",
                            line=dict(color="#818cf8", width=1.5),
                            name="BLS Power",
                        ))
                        if result.get("period_days"):
                            fig.add_vline(
                                x=result["period_days"],
                                line_width=2, line_dash="dash", line_color="#ef4444",
                                annotation_text="Best Period",
                            )
                        fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            xaxis_title="Period [days]",
                            yaxis_title="Power (SNR objective)",
                            xaxis_type="log",
                            height=400,
                            margin=dict(l=40, r=40, t=40, b=40),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif name == "Interactive Phase-Fold":
                        pd_data = result["plot_data"]
                        fig = go.Figure()

                        if pd_data.get("phase"):
                            n_pts  = len(pd_data["phase"])
                            stride = max(1, n_pts // 5000)
                            fig.add_trace(go.Scattergl(
                                x=pd_data["phase"][::stride],
                                y=pd_data["flux_folded"][::stride],
                                mode="markers",
                                marker=dict(size=3, color="rgba(148,163,184,0.3)"),
                                name="Unbinned Data (subsampled)" if stride > 1 else "Unbinned Data",
                            ))

                        if pd_data.get("phase_b"):
                            fig.add_trace(go.Scattergl(
                                x=pd_data["phase_b"], y=pd_data["flux_b"],
                                mode="markers",
                                marker=dict(size=6, color="#f8fafc", line=dict(width=1, color="#334155")),
                                name="Binned Data",
                            ))

                        if pd_data.get("model_phase"):
                            fig.add_trace(go.Scattergl(
                                x=pd_data["model_phase"], y=pd_data["model_flux"],
                                mode="lines", line=dict(color="#38bdf8", width=3),
                                name="Transit Model",
                            ))

                        fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            xaxis_title="Phase",
                            yaxis_title="Normalized Flux",
                            hovermode="x unified",
                            height=500,
                            margin=dict(l=40, r=40, t=40, b=40),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.image(str(available[name]), use_column_width=True)

        # --- Raw JSON ---
        disp_result = {k: v for k, v in result.items() if k != "plot_data"}
        with st.expander("📄 Raw JSON result"):
            st.json(disp_result)

        # --- Download button ---
        json_str = json.dumps(disp_result, indent=2, default=str)
        st.download_button(
            "⬇️  Download result JSON",
            data=json_str,
            file_name=f"{tag}_result.json",
            mime="application/json",
        )

    else:
        st.markdown("""
        <div style="text-align:center; padding:5rem 2rem; background: rgba(30, 41, 59, 0.3); border-radius: 20px; border: 1px dashed rgba(255,255,255,0.1); margin-top: 2rem;">
          <div style="font-size:4.5rem; margin-bottom: 1rem; opacity: 0.8; animation: float 6s ease-in-out infinite;">🔭</div>
          <h3 style="color: white; font-weight: 700; margin-bottom: 0.5rem;">Ready to Analyze</h3>
          <p style="font-size:1.1rem; color: #94a3b8;">Select a target above and click <strong>Run Pipeline</strong> to begin.</p>
          <style>
            @keyframes float {
                0% { transform: translateY(0px); }
                50% { transform: translateY(-15px); }
                100% { transform: translateY(0px); }
            }
          </style>
        </div>
        """, unsafe_allow_html=True)

        RESULTS_DIR = Path(__file__).parent / "results"
        if RESULTS_DIR.exists():
            prev = sorted(RESULTS_DIR.glob("*.json"))
            if prev:
                st.markdown("### 🗂️ Previous results")
                for p in prev:
                    with st.expander(p.stem):
                        st.json(json.loads(p.read_text()))

# ===========================================================================
# TAB 2: Bulk Processing
# ===========================================================================
with tab_bulk:
    st.markdown("### 📦 Bulk Processing Mode")
    st.markdown(
        "Upload a CSV containing a column named **`Target ID`** "
        "(and optionally `Mission`, `Stellar Radius`, `Stellar Mass`)."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df_targets = pd.read_csv(uploaded_file)
        st.dataframe(df_targets, use_container_width=True)

        if "Target ID" not in df_targets.columns:
            st.error("Error: CSV must contain a 'Target ID' column.")
        else:
            bulk_run_btn = st.button("🚀 Start Bulk Run", type="primary", key="bulk_run")

            if bulk_run_btn:
                sys.path.insert(0, str(Path(__file__).parent))
                from pipeline import run_pipeline

                progress_bar = st.progress(0)
                status_text  = st.empty()
                results_list: list = []
                total = len(df_targets)

                for step, (_, row) in enumerate(df_targets.iterrows(), start=1):
                    tid       = str(row["Target ID"]).strip()
                    t_mission = str(row["Mission"]) if "Mission" in df_targets.columns else mission
                    t_sr      = float(row["Stellar Radius"]) if "Stellar Radius" in df_targets.columns else star_r
                    t_sm      = float(row["Stellar Mass"])   if "Stellar Mass"   in df_targets.columns else star_m

                    status_text.markdown(f"**Processing ({step}/{total}):** `{tid}`")

                    try:
                        res = run_pipeline(
                            target_id        = tid,
                            mission          = t_mission,
                            n_fap_trials     = 200,
                            skip_fap         = True,
                            star_radius_rsun = t_sr,
                            star_mass_msun   = t_sm,
                            save_plots       = False,
                            return_plot_data = False,
                            rng_seed         = 42,
                        )
                        results_list.append({
                            "Target ID":       tid,
                            "Period [d]":      res.get("period_days"),
                            "Depth [ppm]":     res.get("depth_ppm"),
                            "Duration [h]":    res.get("duration_hours"),
                            "Radius [R_Earth]":res.get("planet_radius_earth"),
                            "SNR":             res.get("snr"),
                            "Vetting Score":   res.get("vetting_score"),
                            "Verdict":         res.get("vetting_verdict"),
                            "ML Class":        res.get("classification"),
                            "CNN Class":       res.get("cnn_classification"),
                        })
                    except Exception as e:
                        results_list.append({"Target ID": tid, "Verdict": f"ERROR: {e}"})

                    progress_bar.progress(step / total)

                status_text.markdown("✅ **Bulk Processing Complete!**")

                df_results = pd.DataFrame(results_list)
                st.dataframe(df_results, use_container_width=True)

                csv_data = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results CSV",
                    data=csv_data,
                    file_name="bulk_analysis_results.csv",
                    mime="text/csv",
                )
