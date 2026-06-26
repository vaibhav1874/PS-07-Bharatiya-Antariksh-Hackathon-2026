"""
pipeline.py  —  Phase 11: End-to-end orchestrator.

Runs all phases (2-10) on a single target and returns a structured JSON result.

Usage:
    python pipeline.py --target "KIC 11904151" --mission Kepler
    python pipeline.py --target "TIC 279741377"  --mission TESS
"""

from __future__ import annotations

import argparse
import json
import logging
import time as _time_mod
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PLOTS_DIR   = Path(__file__).parent / "plots"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Known reference values (period, depth_ppm, duration_hours) for recovery comparison
KNOWN_PARAMS = {
    "KIC 11904151": {"period": 0.8375243, "depth_ppm": 152.0, "duration_hours": 1.811},
}


def run_pipeline(
    target_id: str,
    mission: str = "Kepler",
    n_fap_trials: int = 200,
    skip_fap: bool = False,
    star_radius_rsun: float = 1.0,
    star_mass_msun: float = 1.0,
    bin_cadence_min: float = 30.0,
    save_plots: bool = True,
    rng_seed: int = 42,
) -> dict:
    """
    Run the full pipeline on one target and return a structured result dict.

    Parameters match the project spec output format (see PS07 spec §OUTPUT FORMAT).
    """
    t_pipeline_start = _time_mod.time()
    np.random.seed(rng_seed)

    # -------------------------------------------------------------------
    # Phase 2-3: Data acquisition + preprocessing
    # -------------------------------------------------------------------
    logger.info("[Phase 2-3] Downloading and preprocessing %s …", target_id)
    from data_loader import download_lightcurve, preprocess
    lc = download_lightcurve(target_id, mission=mission)
    time, flux, flux_err = preprocess(lc)

    cadence_min = float(np.median(np.diff(time)) * 24 * 60)
    baseline    = float(time[-1] - time[0])
    logger.info("  %d cadences | %.1f-day baseline | %.2f-min cadence",
                len(time), baseline, cadence_min)

    # -------------------------------------------------------------------
    # Phase 4: Detrending
    # -------------------------------------------------------------------
    logger.info("[Phase 4] Detrending …")
    from detrend import run_detrending
    _, detrended, window_pts = run_detrending(
        time=time, flux=flux,
        period_max_days=baseline / 3.0,
        cadence_minutes=cadence_min,
        method="savgol",
        target_id=target_id,
        save_plot=save_plots,
    )
    logger.info("  Window = %d pts | std = %.4e", window_pts, float(detrended.std()))

    # -------------------------------------------------------------------
    # Phase 5: Period search (BLS)
    # -------------------------------------------------------------------
    logger.info("[Phase 5] BLS period search …")
    from identify import (
        build_period_grid, build_duration_grid,
        run_bls, bin_lc_for_bls, phase_fold,
        plot_periodogram, plot_phase_fold,
    )

    time_bls, det_bls, err_bls = bin_lc_for_bls(
        time, detrended, flux_err, target_cadence_min=bin_cadence_min
    )
    period_grid   = build_period_grid(baseline)
    duration_grid = build_duration_grid()
    bls_result, best_signal = run_bls(
        time_bls, det_bls, err_bls, period_grid, duration_grid
    )

    # Phase-fold and save plots
    phase, flux_folded = phase_fold(time, detrended, best_signal["period"], best_signal["t0"])
    best_signal["phase"]       = phase
    best_signal["flux_folded"] = flux_folded

    tag = target_id.replace(" ", "_")
    if save_plots:
        plot_periodogram(bls_result, best_signal["period"], target_id,
                         save_path=PLOTS_DIR / f"{tag}_periodogram.png")
        plot_phase_fold(phase, flux_folded, best_signal, target_id,
                        save_path=PLOTS_DIR / f"{tag}_phasefold.png")

    logger.info(
        "  Best period: %.5f d | depth: %.1f ppm | power: %.3f",
        best_signal["period"], best_signal["depth"] * 1e6, best_signal["power"],
    )

    # -------------------------------------------------------------------
    # Phase 6: Characterization (batman + lmfit)
    # -------------------------------------------------------------------
    logger.info("[Phase 6] Transit model fit …")
    from characterize import run_characterization
    fit_params, _ = run_characterization(
        time=time, flux=detrended, flux_err=flux_err,
        best_signal=best_signal, target_id=target_id, save_plot=save_plots,
    )

    # -------------------------------------------------------------------
    # Phase 7: Vetting
    # -------------------------------------------------------------------
    logger.info("[Phase 7] Vetting …")
    from vet import run_vetting
    vet_tests, vet_summary = run_vetting(
        time=time, flux=detrended, flux_err=flux_err,
        best_signal=best_signal,
        target_id=target_id,
        star_radius_rsun=star_radius_rsun,
        star_mass_msun=star_mass_msun,
        save_plot=save_plots,
    )

    # -------------------------------------------------------------------
    # Phase 8: Statistical significance
    # -------------------------------------------------------------------
    logger.info("[Phase 8] Significance …")
    from significance import run_significance
    # Pass fitted depth (more accurate than BLS box estimate) for SNR calculation
    fit_depth_ppm = fit_params.get("depth_ppm_val", None)
    sig = run_significance(
        time=time, flux=detrended, best_signal=best_signal,
        n_fap_trials=n_fap_trials, rng_seed=rng_seed,
        target_id=target_id, save_plot=save_plots, skip_fap=skip_fap,
        fit_depth_ppm=fit_depth_ppm,
    )

    # -------------------------------------------------------------------
    # Phase 9-10: Feature engineering + classification
    # -------------------------------------------------------------------
    logger.info("[Phase 9-10] Classification …")
    from classify import classify_from_pipeline_outputs
    clf_result = classify_from_pipeline_outputs(
        best_signal=best_signal,
        fit_params=fit_params,
        snr_result=sig,
        vet_results=vet_tests,
        star_radius_rsun=star_radius_rsun,
    )

    # -------------------------------------------------------------------
    # Assemble final JSON-serialisable result (spec §OUTPUT FORMAT)
    # -------------------------------------------------------------------
    period_d   = float(best_signal["period"])
    _bls_depth_ppm = float(best_signal["depth"]) * 1e6
    _bls_dur_h     = float(best_signal["duration"]) * 24.0
    depth_ppm  = float(fit_params.get("depth_ppm_val", _bls_depth_ppm))
    duration_h = float(fit_params.get("duration_h_val", _bls_dur_h))
    if not np.isfinite(depth_ppm):  depth_ppm  = _bls_depth_ppm
    if not np.isfinite(duration_h): duration_h = _bls_dur_h

    vetting_flags = {
        "odd_even_consistent":        vet_tests.get("odd_even", {}).get("score", 0) == 1,
        "secondary_eclipse_detected": vet_tests.get("secondary", {}).get("score", 0) < 0,
        "centroid_shift_detected":    vet_tests.get("centroid", {}).get("score", 0) < 0,
    }

    _baseline = float(time[-1] - time[0])
    _n_periods = 9498
    period_grid_sigma = float(np.clip(
        period_d ** 2 / max(_baseline * _n_periods, 1e-6),
        1e-6, period_d / 100.0
    ))

    result: dict = {
        "target_id":                 target_id,
        "mission":                   mission,
        "period_days":               round(period_d, 6),
        "period_uncertainty":        round(period_grid_sigma, 7),
        "depth_pct":                 round(depth_ppm / 1e4, 4),   # ppm → %
        "depth_uncertainty_pct":     round(float(fit_params.get("depth_ppm_err", np.nan)) / 1e4, 6),
        "duration_hours":            round(duration_h, 4),
        "duration_uncertainty_hours": round(float(fit_params.get("duration_h_err", np.nan)), 4),
        "snr":                       round(float(sig["snr"]), 3),
        "false_alarm_probability":   round(float(sig.get("fap", np.nan)), 5),
        "vetting":                   vetting_flags,
        "vetting_score":             int(vet_summary.get("overall_score", 0)),
        "vetting_verdict":           str(vet_summary.get("disposition", "unknown")),
        "classification":            clf_result["classification"],
        "classification_confidence": round(clf_result["classification_confidence"], 4),
        "class_probabilities":       {k: round(v, 4) for k, v in clf_result["class_probabilities"].items()},
        "pipeline_wall_time_s":      round(_time_mod.time() - t_pipeline_start, 1),
    }

    # Known-value comparison if target is in our reference DB
    if target_id in KNOWN_PARAMS:
        known = KNOWN_PARAMS[target_id]
        err_period = abs(period_d - known["period"]) / known["period"] * 100
        err_depth  = abs(depth_ppm - known["depth_ppm"]) / known["depth_ppm"] * 100
        err_dur    = abs(duration_h - known["duration_hours"]) / known["duration_hours"] * 100
        result["known_value_comparison"] = {
            "published_period_d":    known["period"],
            "recovered_period_d":    round(period_d, 6),
            "period_error_pct":      round(err_period, 2),
            "published_depth_ppm":   known["depth_ppm"],
            "recovered_depth_ppm":   round(depth_ppm, 1),
            "depth_error_pct":       round(err_depth, 2),
            "published_duration_h":  known["duration_hours"],
            "recovered_duration_h":  round(duration_h, 4),
            "duration_error_pct":    round(err_dur, 2),
        }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PS-07 Exoplanet Pipeline — end-to-end runner")
    p.add_argument("--target",   default="KIC 11904151")
    p.add_argument("--mission",  default="Kepler", choices=["Kepler", "K2", "TESS"])
    p.add_argument("--n-fap-trials", type=int, default=1000,
                   help="Bootstrap FAP iterations (use 1000 for full quality)")
    p.add_argument("--skip-fap", action="store_true",
                   help="Skip bootstrap FAP for a faster run")
    p.add_argument("--star-radius", type=float, default=1.0)
    p.add_argument("--star-mass",   type=float, default=1.0)
    p.add_argument("--no-plots",    action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print("\n" + "=" * 70)
    print(f"  PS-07 Exoplanet Detection Pipeline  -  {args.target}")
    print("=" * 70 + "\n")

    result = run_pipeline(
        target_id        = args.target,
        mission          = args.mission,
        n_fap_trials     = args.n_fap_trials,
        skip_fap         = args.skip_fap,
        star_radius_rsun = args.star_radius,
        star_mass_msun   = args.star_mass,
        save_plots       = not args.no_plots,
    )

    json_str = json.dumps(result, indent=2, default=str)
    print("\n== PIPELINE RESULT =========================================")
    print(json_str)
    print("=" * 65 + "\n")

    tag = args.target.replace(" ", "_")
    out_path = RESULTS_DIR / f"{tag}_result.json"
    out_path.write_text(json_str)
    logger.info("Result saved -> %s", out_path)

    print(f"Plots saved to: plots/")
    print(f"Result JSON:    {out_path}\n")
