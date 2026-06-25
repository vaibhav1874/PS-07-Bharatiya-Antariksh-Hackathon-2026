"""
test_all_part_d.py — Part D Test Suite (full checklist)
=========================================================
Runs all Part D test cases from PS07_Technical_Math_Physics_Reference.md.

Run with:  python test_all_part_d.py
or:        pytest test_all_part_d.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import warnings
warnings.filterwarnings("ignore")

PASS_STR = "PASS"
FAIL_STR = "FAIL"

results = []

def check(name, condition, detail=""):
    status = PASS_STR if condition else FAIL_STR
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")
    results.append((name, condition))
    return condition

print("\n" + "="*70)
print("  PS-07 Part D Test Suite")
print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# D1: Data loader
# ─────────────────────────────────────────────────────────────────────────────
print("\n[D1] Data loader — synthetic")
from data_loader import generate_synthetic_lc
time, flux, flux_err = generate_synthetic_lc(n_points=5000, period_days=3.0, depth=0.005,
                                              duration_days=0.1, noise_level=2e-4, seed=42)
check("D1a: time monotonically increasing",
      bool(np.all(np.diff(time) > 0)),
      f"time[0]={time[0]:.2f}, time[-1]={time[-1]:.2f}")
check("D1b: array lengths match",
      len(time) == len(flux) == len(flux_err) == 5000,
      f"len(time)={len(time)}, len(flux)={len(flux)}, len(flux_err)={len(flux_err)}")
check("D1c: flux is non-empty, not all-NaN",
      len(flux) > 0 and not np.all(np.isnan(flux)),
      f"n_non_nan={np.sum(~np.isnan(flux))}")


# ─────────────────────────────────────────────────────────────────────────────
# D2: Sigma-clip removes outliers without eating transit
# ─────────────────────────────────────────────────────────────────────────────
print("\n[D2] Sigma-clip: removes outliers, preserves transit")
# Requirements:
#   - outlier level = 10% (20× noise): must be removed
#   - transit depth = 3× noise: must NOT be removed by 5-sigma clip
# Noise level = 1e-4, transit depth = 3e-4 = 3σ (well within 5σ clip)
noise_d2 = 1e-4
depth_d2 = 5e-4   # 5σ transit depth — large relative to noise but < 5σ robust clip boundary
# With MAD-based sigma, the transit points are displaced by 5σ exactly at the clip boundary
# Use a transit depth of 3σ to be safely inside the clip window
depth_d2 = 3 * noise_d2   # 3σ: safely preserved by 5σ clip

rng_d2 = np.random.default_rng(7)
period_d2 = 5.0; t0_d2 = 1.0; dur_d2 = 0.15
n_d2 = 5000
t_d2 = np.arange(n_d2) * 30.0 / 1440.0  # 30-min cadence

ph_d2_raw = ((t_d2 - t0_d2) % period_d2) / period_d2
ph_d2 = np.where(ph_d2_raw > 0.5, ph_d2_raw - 1.0, ph_d2_raw)
in_transit_d2 = np.abs(ph_d2) < dur_d2 / (2.0 * period_d2)
n_in_transit_before = in_transit_d2.sum()

flux_d2 = 1.0 + rng_d2.normal(0, noise_d2, n_d2)
flux_d2[in_transit_d2] -= depth_d2  # inject transit

# Add 10 large outliers at OOT positions
oot_indices = np.where(~in_transit_d2)[0]
outlier_idx = oot_indices[:10]
flux_d2[outlier_idx] += 0.05  # 500σ = very large outlier

# Apply robust sigma-clip (Part B2: MAD-based, k=5, 3 passes)
mask_good = np.ones(len(flux_d2), dtype=bool)
for _ in range(3):
    mu = np.median(flux_d2[mask_good])
    mad = np.median(np.abs(flux_d2[mask_good] - mu))
    sigma_r = 1.4826 * mad
    new_mask = np.abs(flux_d2 - mu) <= 5.0 * sigma_r
    if np.sum(mask_good & ~new_mask) == 0:
        break
    mask_good = mask_good & new_mask

n_outliers_removed = (~mask_good)[outlier_idx].sum()
in_transit_after = (in_transit_d2 & mask_good).sum()
transit_survival_pct = in_transit_after / max(n_in_transit_before, 1) * 100

check("D2a: all 10 outliers removed",
      n_outliers_removed == 10,
      f"outliers removed={n_outliers_removed}/10")
check("D2b: >95% of in-transit points survive",
      transit_survival_pct >= 95.0,
      f"transit survival={transit_survival_pct:.1f}% ({in_transit_after}/{n_in_transit_before})")


# ─────────────────────────────────────────────────────────────────────────────
# D3: Detrending on LC with artificial gap shows no edge artifact
# ─────────────────────────────────────────────────────────────────────────────
print("\n[D3] Detrending: no edge artifact across gap")
from detrend import detrend_savgol, compute_window_length, normalize_flux

# Use 1-min cadence (short cadence like real Kepler) so window is reasonably large
cadence_d3 = 1.0 / 1440.0   # 1 min in days
n_seg_d3 = 8000
t_s1 = np.arange(n_seg_d3) * cadence_d3
t_s2 = np.arange(n_seg_d3) * cadence_d3 + t_s1[-1] + 2.0  # 2-day gap
t_gapped = np.concatenate([t_s1, t_s2])

rng_d3 = np.random.default_rng(99)
trend_true = 1.0 + 0.005 * np.sin(2 * np.pi * t_gapped / 5.0)  # 5-day sinusoidal trend
noise_d3 = rng_d3.normal(0, 1e-4, len(t_gapped))
flux_d3 = trend_true + noise_d3

# Inject transit at period=0.8375d (Kepler-10b-like), depth=200ppm, dur=0.075d
period_d3 = 0.8375; t0_d3 = 0.2; dur_d3 = 0.075; depth_d3 = 200e-6
ph_d3 = ((t_gapped - t0_d3) % period_d3) / period_d3
ph_d3 = np.where(ph_d3 > 0.5, ph_d3 - 1.0, ph_d3)
in_tr_d3 = np.abs(ph_d3) < dur_d3 / (2.0 * period_d3)
flux_d3[in_tr_d3] -= depth_d3

# Window: per reference §3.1, window ≥ 3×transit_duration = 3×0.075d = 225 pts at 1-min cadence
cadence_min_d3 = 1.0  # minutes
window_pts_d3 = compute_window_length(dur_d3, cadence_min_d3, factor=3.0)
print(f"         D3 window = {window_pts_d3} pts (cadence=1min, dur={dur_d3:.3f}d)")

trend_out, detrended_out = detrend_savgol(t_gapped, flux_d3, window_pts_d3, polyorder=2)
detrended_out = normalize_flux(detrended_out)

gap_idx_d3 = n_seg_d3
win_check_d3 = 50
bulk_std = float(np.std(detrended_out[100:n_seg_d3 - 100]))

before_region = detrended_out[max(0, gap_idx_d3 - win_check_d3):gap_idx_d3]
after_region  = detrended_out[gap_idx_d3:gap_idx_d3 + win_check_d3]
max_dev_before = float(np.max(np.abs(before_region - 1.0))) if len(before_region) else 0
max_dev_after  = float(np.max(np.abs(after_region - 1.0))) if len(after_region) else 0

check("D3a: no edge artifact before gap (deviation < 5×bulk_std)",
      max_dev_before < 5.0 * bulk_std,
      f"max_dev={max_dev_before:.6f}, 5*bulk_std={5*bulk_std:.6f}")
check("D3b: no edge artifact after gap (deviation < 5×bulk_std)",
      max_dev_after < 5.0 * bulk_std,
      f"max_dev={max_dev_after:.6f}, 5*bulk_std={5*bulk_std:.6f}")

# Check transit depth recovery (only points NOT near trend inflection)
interior_mask = ((t_gapped > t_s1[100]) & (t_gapped < t_s1[-100])) | \
                ((t_gapped > t_s2[100]) & (t_gapped < t_s2[-100]))
in_tr_interior = in_tr_d3 & interior_mask
if in_tr_interior.sum() > 0:
    recovered_depth_d3 = float(1.0 - np.median(detrended_out[in_tr_interior]))
else:
    recovered_depth_d3 = 0.0
depth_error_pct_d3 = abs(recovered_depth_d3 - depth_d3) / depth_d3 * 100
check("D3c: recovered transit depth within 20% of injected",
      depth_error_pct_d3 < 20.0,
      f"recovered={recovered_depth_d3*1e6:.0f}ppm, injected={depth_d3*1e6:.0f}ppm, "
      f"err={depth_error_pct_d3:.1f}%  (in_transit_pts={in_tr_interior.sum()})")


# ─────────────────────────────────────────────────────────────────────────────
# D4: BLS recovers injected period within 0.1%
# ─────────────────────────────────────────────────────────────────────────────
print("\n[D4] BLS period recovery (synthetic, within 0.1%)")
from identify import build_period_grid, build_duration_grid, run_bls, bin_lc_for_bls

SYNTH_PERIOD_D4 = 3.5   # Use the default period that generate_synthetic_lc is tuned for
# Build a long synthetic light curve at 30-min cadence with strong transit signal
n_d4 = 40000   # ~833 days at 30-min cadence = 240 transit events at 3.5-day period
t_d4, f_d4, fe_d4 = generate_synthetic_lc(
    n_points=n_d4, period_days=SYNTH_PERIOD_D4,
    depth=0.01, duration_days=0.20,   # strong transit, long duration for easy BLS recovery
    noise_level=5e-5, seed=7,         # very low noise for clean detection
)
baseline4 = t_d4[-1] - t_d4[0]
period_grid4 = build_period_grid(baseline4)
dur_grid4 = build_duration_grid()
_, best4 = run_bls(t_d4, f_d4, fe_d4, period_grid4, dur_grid4)
period_err_pct_d4 = abs(best4["period"] - SYNTH_PERIOD_D4) / SYNTH_PERIOD_D4 * 100
check("D4: BLS recovers period within 0.1%",
      period_err_pct_d4 < 0.1,
      f"recovered={best4['period']:.5f}d, injected={SYNTH_PERIOD_D4:.5f}d, "
      f"err={period_err_pct_d4:.4f}%  (power={best4['power']:.2f})")


# ─────────────────────────────────────────────────────────────────────────────
# D5: BLS on pure white noise — flat floor, no fake peak
# ─────────────────────────────────────────────────────────────────────────────
print("\n[D5] BLS on pure white noise: flat floor (no fake 5-sigma peak)")
rng5 = np.random.default_rng(123)
n5 = 3000
t5 = np.linspace(0, 60, n5)
f5 = 1.0 + rng5.normal(0, 1e-3, n5)
fe5 = np.full(n5, 1e-3)
baseline5 = t5[-1] - t5[0]
pg5 = build_period_grid(baseline5)
dg5 = build_duration_grid()
bls5_result, best5 = run_bls(t5, f5, fe5, pg5, dg5)
powers5 = np.array(bls5_result.power)
median_p5 = float(np.median(powers5))
mad_p5 = float(np.median(np.abs(powers5 - median_p5)))
sigma_p5 = 1.4826 * mad_p5
threshold_5sigma = median_p5 + 5 * sigma_p5
max_power5 = float(np.max(powers5))
check("D5: No BLS power point > 5-sigma above median on pure noise",
      max_power5 < threshold_5sigma,
      f"max_power={max_power5:.3f}, 5-sigma threshold={threshold_5sigma:.3f}, "
      f"median={median_p5:.3f}, sigma={sigma_p5:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# D6: Transit-model fit on known synthetic transit: depth within 1%, finite σ
# ─────────────────────────────────────────────────────────────────────────────
print("\n[D6] Transit-model fit: depth recovery <1%, finite uncertainty")
from characterize import init_transit_params, fit_transit_lmfit
from batman_wrapper import BATMAN_AVAILABLE, TransitParams, make_batman_model, eval_model

KNOWN_RP_D6 = 0.02
KNOWN_DEPTH_PPM_D6 = KNOWN_RP_D6**2 * 1e6  # = 400 ppm
KNOWN_PERIOD_D6 = 3.0

bp6 = TransitParams()
bp6.t0 = 0.0; bp6.per = KNOWN_PERIOD_D6; bp6.rp = KNOWN_RP_D6
bp6.a = 15.0; bp6.inc = 90.0; bp6.ecc = 0.0; bp6.w = 90.0
bp6.u = [0.4, 0.2]; bp6.limb_dark = "quadratic"

# Use 500 bins in [-0.15, +0.15] phase window for stable covariance
# Phase spacing: 0.30/500 = 6e-4; transit occupies ~0.12/3.0 = 0.04 phase = 67 bins
rng6 = np.random.default_rng(55)
# Generate many "unbinned" points so binning produces stable mean per bin
n_points_d6 = 3000   # 3000 pts in [-0.15, +0.15] → ~10 per bin for 300 bins
phase6 = np.linspace(-0.15, 0.15, n_points_d6)
t6 = phase6 * KNOWN_PERIOD_D6
m6 = make_batman_model(bp6, t6)
flux6_clean = eval_model(m6, bp6)
noise6 = 5e-5
flux6 = flux6_clean + rng6.normal(0, noise6, n_points_d6)
fe6 = np.full(n_points_d6, noise6)

dur6_days = (KNOWN_PERIOD_D6 / np.pi) * np.arcsin(
    np.sqrt((1 + KNOWN_RP_D6)**2) / bp6.a
)
init6 = init_transit_params(KNOWN_PERIOD_D6, 0.0, KNOWN_RP_D6**2, dur6_days)
fit_result6, fit_params6 = fit_transit_lmfit(
    phase6, flux6, fe6, init6,
    use_binned=True,
    n_bins=300,
)

depth_recovered_d6 = fit_params6["depth_ppm_val"]
depth_err_d6 = fit_params6["depth_ppm_err"]
depth_error_pct_d6 = abs(depth_recovered_d6 - KNOWN_DEPTH_PPM_D6) / KNOWN_DEPTH_PPM_D6 * 100
uncertainty_finite_d6 = (depth_err_d6 is not None and not np.isnan(depth_err_d6))

check("D6a: batman C extension available",
      BATMAN_AVAILABLE,
      f"BATMAN_AVAILABLE={BATMAN_AVAILABLE}")
check("D6b: recovered depth within 1% of known (400 ppm)",
      depth_error_pct_d6 < 1.0,
      f"recovered={depth_recovered_d6:.2f}ppm, known={KNOWN_DEPTH_PPM_D6:.2f}ppm, "
      f"err={depth_error_pct_d6:.3f}%")
check("D6c: depth uncertainty is finite (not nan)",
      uncertainty_finite_d6,
      f"depth_ppm_err={depth_err_d6}")
check("D6d: fit did not silently catch exception (fit_ok=True)",
      fit_params6["fit_ok"],
      f"fit_ok={fit_params6['fit_ok']}, redchi={fit_params6['redchi']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# D7: Odd-even test on confirmed planet — NOT significant (<3-sigma)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[D7] Odd-even test: confirmed planet (no significant asymmetry)")
from vet import test_odd_even

rng7 = np.random.default_rng(77)
period7 = 0.8375
n7 = 50000
t7 = np.linspace(0, 700, n7)
ph7 = ((t7 - 0.0) % period7) / period7
ph7 = np.where(ph7 > 0.5, ph7 - 1.0, ph7)
in_tr7 = np.abs(ph7) < 0.04 / (2.0 * period7)
flux7 = 1.0 + rng7.normal(0, 2e-4, n7)
flux7[in_tr7] -= 152e-6  # all transits same depth (planet-like)

oe7 = test_odd_even(t7, flux7, period7, 0.0, 0.04)
check("D7: odd-even |delta|/sigma < 3 for confirmed planet",
      oe7["depth_diff_sigma"] < 3.0,
      f"|delta|/sigma={oe7['depth_diff_sigma']:.2f} (threshold 3), score={oe7['score']}")


# ─────────────────────────────────────────────────────────────────────────────
# D8: Odd-even / secondary test on EB — at least one flag fires
# ─────────────────────────────────────────────────────────────────────────────
print("\n[D8] Vetting on EB: at least one flag fires")
from vet import test_secondary_eclipse

rng8 = np.random.default_rng(88)
period8 = 1.5
n8 = 30000
t8 = np.linspace(0, 500, n8)
ph8 = ((t8 - 0.0) % period8) / period8
ph8 = np.where(ph8 > 0.5, ph8 - 1.0, ph8)
transit_num8 = np.floor(t8 / period8).astype(int)
in_primary8 = np.abs(ph8) < 0.05 / (2.0 * period8)
flux8 = 1.0 + rng8.normal(0, 5e-4, n8)
# Alternating depths: odd 10000ppm, even 3000ppm — strong EB signature
flux8[in_primary8 & (transit_num8 % 2 == 0)] -= 10000e-6
flux8[in_primary8 & (transit_num8 % 2 == 1)] -= 3000e-6

oe8 = test_odd_even(t8, flux8, period8, 0.0, 0.05)

ph8s = np.sort(ph8)
flux8s = flux8[np.argsort(ph8)]
sec8 = test_secondary_eclipse(ph8s, flux8s, 0.05, period8, primary_depth_ppm=6500.0)

odd_even_fires8 = oe8["depth_diff_sigma"] >= 3.0
secondary_fires8 = sec8["score"] == -1
check("D8: at least one vetting flag fires for EB",
      odd_even_fires8 or secondary_fires8,
      f"odd-even |delta|/sigma={oe8['depth_diff_sigma']:.1f} (fires={odd_even_fires8}), "
      f"secondary score={sec8['score']} (fires={secondary_fires8})")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
n_pass = sum(1 for _, ok in results if ok)
n_total = len(results)
print(f"  TOTAL: {n_pass}/{n_total} tests PASSED")
if n_pass == n_total:
    print("  ALL PART D TESTS PASSED")
else:
    print("  FAILED TESTS:")
    for name, ok in results:
        if not ok:
            print(f"    - {name}")
print("="*70 + "\n")

sys.exit(0 if n_pass == n_total else 1)
