"""
diagnose_bug1_fixed.py — Verifies Bug 1 fix on real KIC 11904151 data.
Run after characterize.py changes to confirm depth > 0 and uncertainty is finite.
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
from characterize import init_transit_params, fit_transit_lmfit, run_characterization

SEP = "=" * 65
print(SEP)
print("  BUG 1 FIX VERIFICATION — KIC 11904151 (Kepler-10b)")
print("  batman:", "C extension" if BATMAN_AVAILABLE else "pure-Python fallback")
print(SEP)

print("\n[1] Loading data ...")
lc = download_lightcurve("KIC 11904151", mission="Kepler")
time, flux, flux_err = preprocess(lc)
cadence_min = float(np.median(np.diff(time)) * 24 * 60)
baseline = float(time[-1] - time[0])
print("    n_points=%d  baseline=%.1fd  cadence=%.3fmin" % (len(time), baseline, cadence_min))

print("\n[2] Detrending (gap-segmented) ...")
_, detrended, wpts = run_detrending(
    time, flux,
    period_max_days=baseline / 5.0,   # Part B4: baseline/5
    cadence_minutes=cadence_min,
    method="savgol",
    target_id="KIC_11904151",
    save_plot=True,
)

print("\n[3] BLS ...")
time_bls, det_bls, err_bls = bin_lc_for_bls(time, detrended, flux_err)
period_grid = build_period_grid(baseline)   # now uses max_fraction=1/5
duration_grid = build_duration_grid()
bls_result, best = run_bls(time_bls, det_bls, err_bls, period_grid, duration_grid)
known = KNOWN_PARAMS.get("KIC 11904151", {})
print("    BLS period   = %.6f d  (known: %.7f d)" % (best["period"], known.get("period_days", 0)))
print("    BLS depth    = %.1f ppm (known: %.1f ppm)" % (best["depth"]*1e6, known.get("depth_ppm", 0)))
print("    BLS duration = %.3f h  (known: %.3f h)" % (best["duration"]*24, known.get("duration_hours", 0)))

print("\n[4] Part A4 Data Volume (with actual data) ...")
N_transits = baseline / best["period"]
pts_per_transit_raw = best["duration"] * 24 * 60 / cadence_min
total_in_transit = pts_per_transit_raw * N_transits
print("    baseline_days        = %.1f d  [ref: ~700 d]" % baseline)
print("    N_transits_observed  = %.0f  [ref: ~836]" % N_transits)
print("    pts_per_transit(raw) = %.1f  [ref: ~2 LC; actual SC ~58]" % pts_per_transit_raw)
print("    total_in_transit     = %.0f  [ref: ~1670 LC; actual SC ~30000]" % total_in_transit)
print("    NOTE: Ref assumed 29.4-min LC; code uses ~1-min SC — actual data far exceeds minimum.")
print("    NOTE: Ref baseline ~700 d; code downloads 20 quarters = %.1f d." % baseline)

print("\n[5] Transit model fit (with fixes) ...")
fit_params, comparison = run_characterization(
    time=time, flux=detrended, flux_err=flux_err,
    best_signal=best,
    target_id="KIC 11904151",
    save_plot=True,
)

print("\n" + SEP)
print("  FIT RESULT (after Bug 1 fix)")
print(SEP)
print("  depth_ppm    = %.2f +/- %s" % (fit_params["depth_ppm_val"], fit_params["depth_ppm_err"]))
print("  Rp/Rs        = %.6f +/- %s" % (fit_params["rp_val"], fit_params["rp_err"]))
print("  a/Rs         = %.3f +/- %s" % (fit_params["a_rs_val"], fit_params["a_rs_err"]))
print("  duration_h   = %.4f h" % fit_params["duration_h_val"])
print("  redchi       = %.4f" % fit_params["redchi"])
print("  fit_ok       = %s" % fit_params["fit_ok"])
print(SEP)

# Evaluate D6 criterion on real data
depth_rec = fit_params["depth_ppm_val"]
depth_err = fit_params["depth_ppm_err"]
known_depth = known.get("depth_ppm", 152.0)
depth_pct = abs(depth_rec - known_depth) / known_depth * 100
unc_finite = not np.isnan(depth_err) if isinstance(depth_err, float) else True

print("\n  Depth vs known: %.1f%% error" % depth_pct)
print("  Uncertainty finite: %s" % unc_finite)
if depth_rec > 10.0 and unc_finite:
    print("  STATUS: BUG 1 FIXED — depth is non-zero and uncertainty is finite")
else:
    print("  STATUS: STILL BROKEN — depth=%.2f or uncertainty=%s" % (depth_rec, depth_err))
print(SEP)
