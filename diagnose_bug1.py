"""
diagnose_bug1.py  —  TEMPORARY diagnostic script, safe to delete after debugging.
Runs the full KIC 11904151 pipeline and prints:
  - batman availability
  - Part A4 data-volume numbers
  - Actual fit result (depth, uncertainty, rp, redchi)
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from batman_wrapper import BATMAN_AVAILABLE
from data_loader import download_lightcurve, preprocess
from detrend import run_detrending
from identify import (
    build_period_grid, build_duration_grid, run_bls,
    bin_lc_for_bls, KNOWN_PARAMS,
)
from characterize import init_transit_params, fit_transit_lmfit, bin_phase_folded

SEP = "=" * 65

print(SEP)
print("  DIAGNOSTIC: BUG 1 — DEPTH / NaN INVESTIGATION")
print("  batman C extension available:", BATMAN_AVAILABLE)
print(SEP)

# ---- Data loading ----
print("\n[1] Loading KIC 11904151 ...")
lc = download_lightcurve("KIC 11904151", mission="Kepler")
time, flux, flux_err = preprocess(lc)
cadence_min = float(np.median(np.diff(time)) * 24 * 60)
baseline = float(time[-1] - time[0])
print("    n_points  =", len(time))
print("    baseline  = %.1f days" % baseline)
print("    cadence   = %.3f min" % cadence_min)

# ---- Detrending ----
print("\n[2] Detrending (savgol, whole-array) ...")
_, detrended, wpts = run_detrending(
    time, flux,
    period_max_days=baseline / 3.0,
    cadence_minutes=cadence_min,
    method="savgol",
    target_id="KIC_11904151",
    save_plot=False,
)

# ---- BLS ----
print("\n[3] BLS period search ...")
time_bls, det_bls, err_bls = bin_lc_for_bls(time, detrended, flux_err)
period_grid = build_period_grid(baseline)
duration_grid = build_duration_grid()
bls_result, best = run_bls(time_bls, det_bls, err_bls, period_grid, duration_grid)
known = KNOWN_PARAMS.get("KIC 11904151", {})
print("    BLS period   = %.6f d   (known: %s d)" % (best["period"], known.get("period_days", "?")))
print("    BLS depth    = %.1f ppm  (known: %s ppm)" % (best["depth"]*1e6, known.get("depth_ppm", "?")))
print("    BLS duration = %.3f h   (known: %s h)" % (best["duration"]*24, known.get("duration_hours", "?")))

# ---- Part A4 Data Volume Check ----
print("\n[4] Part A4 data-volume check ...")
N_transits = baseline / best["period"]
pts_per_transit_raw = best["duration"] * 24 * 60 / cadence_min
total_in_transit = pts_per_transit_raw * N_transits
print("    baseline_days        = %.1f  (ref: ~700)" % baseline)
print("    N_transits_observed  = %.0f  (ref: ~836)" % N_transits)
print("    pts_per_transit(raw) = %.2f  (ref: ~2)" % pts_per_transit_raw)
print("    total_in_transit     = %.0f  (ref: ~1670)" % total_in_transit)
for label, val, ref, tol in [
    ("N_transits", N_transits, 836, 0.20),
    ("pts_per_transit", pts_per_transit_raw, 2.0, 0.50),
    ("total_in_transit", total_in_transit, 1670, 0.20),
]:
    pct_err = abs(val - ref) / ref
    status = "OK" if pct_err < tol else ("MISMATCH (>%d%%)" % int(tol*100))
    print("    %-24s %s (err=%.1f%%)" % (label, status, pct_err*100))

# ---- Transit Fit ----
print("\n[5] Running transit model fit ...")
phase_raw = ((time - best["t0"]) % best["period"]) / best["period"]
phase = np.where(phase_raw > 0.5, phase_raw - 1.0, phase_raw)
sort_idx = np.argsort(phase)
phase_s = phase[sort_idx]
flux_s = detrended[sort_idx]
fe_s = flux_err[sort_idx]

init_p = init_transit_params(best["period"], best["t0"], best["depth"], best["duration"])
print("    init rp (sqrt(BLS_depth)) = %.6f" % init_p.rp)
print("    init a_rs                 = %.2f" % init_p.a)
print("    init inc                  = %.1f" % init_p.inc)

transit_half = 0.15
in_window = np.abs(phase_s) < transit_half
print("    points in transit window  = %d" % in_window.sum())

fit_result, fit_params = fit_transit_lmfit(
    phase_s, flux_s, fe_s, init_p,
    use_binned=True, n_bins=300,
)

print("\n" + SEP)
print("  FIT RESULT")
print(SEP)
print("  depth_ppm  = %.2f +/- %s" % (fit_params["depth_ppm_val"], fit_params["depth_ppm_err"]))
print("  Rp/Rs      = %.6f +/- %s" % (fit_params["rp_val"], fit_params["rp_err"]))
print("  a/Rs       = %.3f +/- %s" % (fit_params["a_rs_val"], fit_params["a_rs_err"]))
print("  inc        = %.3f +/- %s" % (fit_params["inc_val"], fit_params["inc_err"]))
print("  duration_h = %.4f" % fit_params["duration_h_val"])
print("  redchi     = %.4f" % fit_params["redchi"])
print("  fit_ok     = %s" % fit_params["fit_ok"])
print("  batman_C   = %s" % fit_params["batman_available"])
print(SEP)

known_depth = known.get("depth_ppm", 152.0)
depth_recovered = fit_params["depth_ppm_val"]
depth_err = fit_params["depth_ppm_err"]
depth_error_pct = abs(depth_recovered - known_depth) / known_depth * 100
print("\n  Depth error vs known: %.1f%%" % depth_error_pct)
print("  Uncertainty finite:  %s" % (not np.isnan(depth_err) if not isinstance(depth_err, float) else not np.isnan(depth_err)))
print(SEP)
