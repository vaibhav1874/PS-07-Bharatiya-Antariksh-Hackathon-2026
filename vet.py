"""
vet.py
======
Phase 7 of the exoplanet transit detection pipeline.

Responsibilities
----------------
Apply five independent vetting tests to discriminate genuine planet transits
from instrumental artefacts and astrophysical false positives:

1. **Odd-even depth test** — test whether alternate transits differ in depth.
   Eclipsing binaries often produce deeper even transits (secondary eclipse
   seen alternately).  Kepler-10b has truly identical odd and even transits.
   Reference: Bryson et al. 2013, PASP.

2. **Secondary eclipse test** — search for a secondary dip at phase = 0.5
   (half-period offset).  A confirmed planet should show none; a grazing
   eclipsing binary will show a comparable secondary eclipse.

3. **Centroid shift test (proxy)** — for instruments with pixel data, a
   transit caused by a background eclipsing binary shows a centroid shift
   toward the contaminating source.  Here we use a simplified flux-weighted
   proxy: we compute the running-mean centroid of the flux-weighted time
   series; a real transit should not shift the centre of light.  Full PRF
   fitting requires pixel-level data not available via lightkurve's default
   API.

4. **Duration vs. Period consistency** (Sessin-Latham relation) — physical
   transits obey T_dur ≈ (P/π) × (R_s/a) × sqrt((1+Rp/Rs)² - b²).
   If the measured duration exceeds the physical maximum for a central transit
   of a main-sequence star of that luminosity, the signal is likely spurious.

5. **Transit shape test** — a genuine planetary transit has a flat bottom
   (trapezoid, not a V-shape).  We fit both a trapezoidal model and a V-shape
   model and compute the Bayesian Information Criterion (BIC) difference.
   If BIC(trapezoid) < BIC(V-shape), the signal is planet-like.

Each test returns a score in {−1, 0, +1}:
  +1 = evidence FOR planet (passes vetting)
   0 = inconclusive
  −1 = evidence AGAINST planet (fails vetting)

The overall vetting verdict is based on the sum of scores.

Usage (standalone)
------------------
    python vet.py --target "KIC 11904151" --mission Kepler
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind

from data_loader import download_lightcurve, preprocess
from detrend import run_detrending
from identify import (
    build_period_grid, build_duration_grid, run_bls,
    bin_lc_for_bls, phase_fold, KNOWN_PARAMS,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Utility: transit mask
# ---------------------------------------------------------------------------

def get_in_transit_mask(
    time: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    width_factor: float = 1.5,
) -> np.ndarray:
    """
    Return a boolean mask identifying cadences within the transit window.

    Parameters
    ----------
    time : np.ndarray
    period : float   [days]
    t0 : float       [days]
    duration : float [days]
    width_factor : float
        Expand the mask by this factor to catch ingress/egress.

    Returns
    -------
    np.ndarray of bool
    """
    phase_raw = ((time - t0) % period) / period
    phase = np.where(phase_raw > 0.5, phase_raw - 1.0, phase_raw)
    half = (duration * width_factor) / (2.0 * period)
    return np.abs(phase) < half


# ---------------------------------------------------------------------------
# Test 1: Odd-Even depth test
# ---------------------------------------------------------------------------

def test_odd_even(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> Dict:
    """
    Test 1: Compare odd vs even transit depths.

    For an eclipsing binary, alternate transits have noticeably different
    depths (primary + secondary eclipse alternate with period P/2).
    A planetary transit has statistically identical odd/even depths.

    Method: Assign each in-transit cadence to transit number N (counting from
    t0), then separate odd and even N, compute median in-transit flux for each
    group, and compare via Welch t-test.

    Returns
    -------
    dict with keys: score, odd_depth_ppm, even_depth_ppm, depth_diff_ppm,
    depth_diff_sigma, p_value, verdict.
    """
    phase_raw = ((time - t0) % period) / period
    phase = np.where(phase_raw > 0.5, phase_raw - 1.0, phase_raw)
    transit_num = np.floor((time - t0) / period).astype(int)

    in_transit = np.abs(phase) < (duration / (2 * period))

    odd_flux = flux[(in_transit) & (transit_num % 2 == 0)]
    even_flux = flux[(in_transit) & (transit_num % 2 == 1)]

    if len(odd_flux) < 5 or len(even_flux) < 5:
        return {
            "score": 0, "verdict": "inconclusive (too few in-transit points)",
            "odd_depth_ppm": np.nan, "even_depth_ppm": np.nan,
            "depth_diff_ppm": np.nan, "p_value": np.nan,
        }

    odd_depth_ppm = (1.0 - np.median(odd_flux)) * 1e6
    even_depth_ppm = (1.0 - np.median(even_flux)) * 1e6
    depth_diff_ppm = abs(odd_depth_ppm - even_depth_ppm)

    # Part B6 exact formula:
    #   depth_odd_err  = MAD-based uncertainty on the per-transit median depth
    #   depth_even_err = same for even group
    #   delta          = depth_odd - depth_even
    #   sigma_delta    = sqrt(depth_odd_err^2 + depth_even_err^2)
    #   flag if |delta| / sigma_delta > 3   (3-sigma threshold)
    def _depth_err_ppm(in_flux_group):
        """Robust uncertainty on the median depth estimate [ppm]."""
        mad = np.median(np.abs(in_flux_group - np.median(in_flux_group)))
        sigma = 1.4826 * mad / np.sqrt(len(in_flux_group))  # uncertainty on median
        return sigma * 1e6  # convert to ppm

    depth_odd_err  = _depth_err_ppm(odd_flux)
    depth_even_err = _depth_err_ppm(even_flux)
    sigma_delta = np.sqrt(depth_odd_err**2 + depth_even_err**2)
    delta = odd_depth_ppm - even_depth_ppm
    depth_diff_sigma = abs(delta) / max(sigma_delta, 1e-3)   # |delta|/sigma_delta

    # Welch t-test: kept as cross-check (H0 = same distribution)
    _, p_value = ttest_ind(odd_flux, even_flux, equal_var=False)

    # Fractional depth asymmetry (legacy diagnostic, not the primary criterion)
    mean_depth_ppm = (odd_depth_ppm + even_depth_ppm) / 2.0
    asymmetry = depth_diff_ppm / max(mean_depth_ppm, 1.0)

    # Score: use Part B6 primary criterion |delta|/sigma_delta > 3
    if depth_diff_sigma < 3.0:
        score, verdict = +1, "PASS — odd/even depths consistent (planet-like, |Δ|/σ=%.2f<3)" % depth_diff_sigma
    elif depth_diff_sigma >= 3.0:
        score, verdict = -1, "FAIL — significant odd/even depth asymmetry (EB-like, |Δ|/σ=%.2f≥3)" % depth_diff_sigma
    else:
        score, verdict = 0, "INCONCLUSIVE — mild asymmetry (|Δ|/σ=%.2f)" % depth_diff_sigma

    logger.info(
        "Odd-even test (Part B6): odd=%.1f ppm, even=%.1f ppm, "
        "|delta|/sigma_delta=%.2f (threshold 3), p=%.4f → %s",
        odd_depth_ppm, even_depth_ppm, depth_diff_sigma, p_value, verdict,
    )
    return {
        "score": score, "verdict": verdict,
        "odd_depth_ppm": odd_depth_ppm, "even_depth_ppm": even_depth_ppm,
        "depth_diff_ppm": depth_diff_ppm, "asymmetry": asymmetry,
        "depth_diff_sigma": float(depth_diff_sigma),   # Part B6 primary metric
        "p_value": float(p_value),
    }


# ---------------------------------------------------------------------------
# Test 2: Secondary eclipse test
# ---------------------------------------------------------------------------

def test_secondary_eclipse(
    phase: np.ndarray,
    flux: np.ndarray,
    duration: float,
    period: float,
    primary_depth_ppm: float,
    secondary_phase: float = 0.5,
) -> Dict:
    """
    Test 2: Search for a secondary eclipse at phase ≈ 0.5 (anti-transit).

    A genuine planet around a solar-type star should show no secondary eclipse
    detectable at TESS/Kepler photometric precision (thermal/reflected emission
    is << 10 ppm for most hot Jupiters; undetectable for super-Earths).
    An eclipsing binary will show a clear secondary dip.

    Parameters
    ----------
    phase : np.ndarray
        Phase array in [−0.5, +0.5] (transit at 0).
    flux : np.ndarray
        Flux sorted by phase.
    duration : float
        Transit duration [days].
    period : float
        Orbital period [days].
    primary_depth_ppm : float
        Primary transit depth [ppm] from BLS.
    secondary_phase : float
        Expected phase of secondary (default 0.5 for circular orbit).

    Returns
    -------
    dict with score, secondary_depth_ppm, secondary_snr, verdict.
    """
    half = (duration / period) * 0.75   # search window = 1.5× duration in phase

    # Fold to put secondary at phase=0 by shifting phase
    phase_sec = phase - secondary_phase
    phase_sec = np.where(phase_sec < -0.5, phase_sec + 1.0, phase_sec)
    phase_sec = np.where(phase_sec > 0.5, phase_sec - 1.0, phase_sec)

    in_secondary = np.abs(phase_sec) < half
    out_of_transit = np.abs(phase) > (duration / period)   # out-of-transit baseline

    if in_secondary.sum() < 3 or out_of_transit.sum() < 10:
        return {
            "score": 0, "verdict": "inconclusive (insufficient data)",
            "secondary_depth_ppm": np.nan, "secondary_snr": np.nan,
        }

    sec_flux_median = np.median(flux[in_secondary])
    baseline_median = np.median(flux[out_of_transit])
    baseline_std = np.std(flux[out_of_transit]) / np.sqrt(in_secondary.sum())

    secondary_depth_ppm = (baseline_median - sec_flux_median) * 1e6
    secondary_snr = secondary_depth_ppm / max(baseline_std * 1e6, 1e-3)

    # Ratio of secondary to primary depth
    depth_ratio = secondary_depth_ppm / max(primary_depth_ppm, 1.0)

    if secondary_snr < 3.0 and depth_ratio < 0.2:
        score = +1
        verdict = (f"PASS — no significant secondary eclipse detected "
                   f"(sec/primary = {depth_ratio:.2f}, SNR={secondary_snr:.1f})")
    elif secondary_snr > 5.0 or depth_ratio > 0.5:
        score = -1
        verdict = (f"FAIL — secondary eclipse detected at phase={secondary_phase:.2f} "
                   f"(depth={secondary_depth_ppm:.1f} ppm, SNR={secondary_snr:.1f})")
    else:
        score = 0
        verdict = (f"INCONCLUSIVE — marginal secondary signal "
                   f"(sec={secondary_depth_ppm:.1f} ppm, SNR={secondary_snr:.1f})")

    logger.info(
        "Secondary eclipse test: depth=%.1f ppm, SNR=%.1f, ratio=%.2f → %s",
        secondary_depth_ppm, secondary_snr, depth_ratio, verdict,
    )
    return {
        "score": score, "verdict": verdict,
        "secondary_depth_ppm": secondary_depth_ppm,
        "secondary_snr": float(secondary_snr),
        "depth_ratio": float(depth_ratio),
    }


# ---------------------------------------------------------------------------
# Test 3: Centroid shift test (simplified flux-weighted proxy)
# ---------------------------------------------------------------------------

def test_centroid_shift(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> Dict:
    """
    Test 3: Centroid shift proxy test.

    A background eclipsing binary (BEB) mimicking a transit would shift
    the flux-weighted centroid of the photometric aperture during transit.
    Full PRF (Pixel Response Function) fitting requires pixel-level data
    not available via standard lightkurve light curve access.

    This simplified proxy computes the difference in the flux-weighted
    mean timestamp between in-transit and out-of-transit cadences.  A real
    transit produces a symmetric dip with no preferred time offset; a BEB
    at the edge of the aperture produces an asymmetric time-domain signature.

    NOTE: This is a PROXY — not a substitute for full pixel-level centroid
    analysis.  Results should be interpreted cautiously.

    Returns
    -------
    dict with score, centroid_offset_proxy, verdict.
    """
    in_tr = get_in_transit_mask(time, period, t0, duration)
    out_tr = ~in_tr

    if in_tr.sum() < 5 or out_tr.sum() < 10:
        return {
            "score": 0, "verdict": "inconclusive (insufficient data)",
            "centroid_offset_proxy": np.nan,
        }

    # Flux-weighted mean phase for in-transit vs out-of-transit
    phase_raw = ((time - t0) % period) / period
    phase = np.where(phase_raw > 0.5, phase_raw - 1.0, phase_raw)

    # Compute flux-weighted centroid phase in-transit
    in_flux = np.clip(1.0 - flux[in_tr], 0, None)  # transit depth (positive)
    centroid_in = np.average(phase[in_tr], weights=in_flux + 1e-10)

    # Out-of-transit baseline centroid (should be near phase=0)
    out_flux = np.abs(flux[out_tr] - np.median(flux[out_tr])) + 1e-10
    centroid_out = np.average(phase[out_tr], weights=out_flux)

    centroid_offset = abs(centroid_in - centroid_out)

    # Score: a small offset suggests the transit is centred on the target star
    # Threshold: > 0.05 in phase (= 5% of the orbital period) is suspicious
    if centroid_offset < 0.02:
        score = +1
        verdict = f"PASS — small centroid offset ({centroid_offset:.4f} phase units) [proxy]"
    elif centroid_offset > 0.05:
        score = -1
        verdict = (f"FAIL — large centroid offset ({centroid_offset:.4f} phase units) "
                   f"[proxy, warrants pixel-level PRF check]")
    else:
        score = 0
        verdict = (f"INCONCLUSIVE — moderate centroid offset ({centroid_offset:.4f}) [proxy]")

    logger.info(
        "Centroid proxy test: offset=%.4f phase → %s",
        centroid_offset, verdict,
    )
    return {
        "score": score, "verdict": verdict,
        "centroid_offset_proxy": float(centroid_offset),
        "note": "PROXY: full pixel-level PRF centroid analysis not performed.",
    }


# ---------------------------------------------------------------------------
# Test 4: Duration-Period consistency (Seager-Mallén-Ornelas 2003)
# ---------------------------------------------------------------------------

def test_duration_period_consistency(
    period_days: float,
    duration_days: float,
    star_radius_rsun: float = 1.065,   # Kepler-10: R_star ≈ 1.065 R_sun
    star_mass_msun: float = 0.895,     # Kepler-10: M_star ≈ 0.895 M_sun
) -> Dict:
    """
    Test 4: Check whether the transit duration is physically consistent with
    the period and stellar properties.

    Maximum transit duration for a central (b=0) transit at a given period:
    T_max = (P/π) × (R_s / a) = (P/π) × (R_s / a_AU)
    where a_AU is derived from Kepler's 3rd law: a³ = M_star × P²

    If the measured duration exceeds T_max by more than 50%, the signal is
    likely an artefact (e.g., stellar variability misidentified as a transit).

    Parameters
    ----------
    period_days : float
    duration_days : float
    star_radius_rsun : float
        Stellar radius in solar radii.  Default = Kepler-10 value.
    star_mass_msun : float
        Stellar mass in solar masses.  Default = Kepler-10 value.

    Returns
    -------
    dict with score, t_max_hours, t_measured_hours, ratio, verdict.
    """
    # Kepler's 3rd law: a [AU] = (M_star [Msun] * P [yr]²)^(1/3)
    period_yr = period_days / 365.25
    a_au = (star_mass_msun * period_yr ** 2) ** (1.0 / 3.0)

    # R_sun = 0.00465 AU
    r_star_au = star_radius_rsun * 0.00465

    # Maximum transit duration (central, circular orbit)
    # T_max = (2 * R_s * P) / (2 * pi * a)  — in same units as P
    t_max_days = (2.0 * r_star_au / (2.0 * np.pi * a_au)) * period_days * (2 * np.pi)
    # Simplification: T_max = P * R_s / (pi * a)
    t_max_days = period_days * r_star_au / (np.pi * a_au)

    ratio = duration_days / t_max_days

    if ratio < 1.5:
        score = +1
        verdict = (f"PASS - duration {duration_days*24:.2f} h <= 1.5xT_max "
                   f"({t_max_days*24:.2f} h) [ratio={ratio:.2f}]")
    elif ratio < 2.5:
        score = 0
        verdict = (f"INCONCLUSIVE - duration {duration_days*24:.2f} h somewhat exceeds "
                   f"T_max={t_max_days*24:.2f} h [ratio={ratio:.2f}]")
    else:
        score = -1
        verdict = (f"FAIL - duration {duration_days*24:.2f} h >> T_max={t_max_days*24:.2f} h "
                   f"[ratio={ratio:.2f}]; physically implausible transit")

    logger.info(
        "Duration-period test: T_meas=%.3f h, T_max=%.3f h, ratio=%.2f → %s",
        duration_days * 24, t_max_days * 24, ratio, verdict,
    )
    return {
        "score": score, "verdict": verdict,
        "t_max_hours": t_max_days * 24,
        "t_measured_hours": duration_days * 24,
        "ratio": float(ratio),
    }


# ---------------------------------------------------------------------------
# Test 5: Transit shape test (trapezoid vs V-shape BIC comparison)
# ---------------------------------------------------------------------------

def _trapezoid_model(phase: np.ndarray, depth: float, t_flat: float, t_ingress: float,
                     baseline: float) -> np.ndarray:
    """Flat-bottomed trapezoid transit model in phase space."""
    flux = np.ones_like(phase) * baseline
    t_flat = max(t_flat, 0.0)
    t_ingress = max(t_ingress, 1e-5)
    t_total = t_flat + 2.0 * t_ingress

    for i, ph in enumerate(phase):
        aph = abs(ph)
        if aph < t_flat / 2.0:
            flux[i] = baseline - depth
        elif aph < t_total / 2.0:
            frac = (aph - t_flat / 2.0) / t_ingress
            flux[i] = baseline - depth * (1.0 - frac)
    return flux


def _v_shape_model(phase: np.ndarray, depth: float, t_total: float,
                   baseline: float) -> np.ndarray:
    """V-shaped transit model (grazing or EB)."""
    flux = np.ones_like(phase) * baseline
    t_half = t_total / 2.0
    for i, ph in enumerate(phase):
        aph = abs(ph)
        if aph < t_half:
            frac = 1.0 - aph / t_half
            flux[i] = baseline - depth * frac
    return flux


def _bic(residuals: np.ndarray, n_params: int) -> float:
    """Bayesian Information Criterion for a fit."""
    n = len(residuals)
    rss = np.sum(residuals ** 2)
    sigma2 = rss / n
    if sigma2 <= 0:
        return np.inf
    log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma2) - rss / (2 * sigma2)
    return -2 * log_likelihood + n_params * np.log(n)


def test_transit_shape(
    phase: np.ndarray,
    flux: np.ndarray,
    duration: float,
    period: float,
    primary_depth_ppm: float,
) -> Dict:
    """
    Test 5: Fit a trapezoid vs V-shape model; compare BIC.

    A genuine planetary transit has a flat bottom (trapezoid wins).
    A grazing eclipsing binary produces a V-shape.

    Parameters
    ----------
    phase : np.ndarray
        Phase in [−0.5, +0.5] (transit centred at 0).
    flux : np.ndarray
        Flux sorted by phase.
    duration : float
        Transit duration [days].
    period : float
        Orbital period [days].
    primary_depth_ppm : float
        Transit depth in ppm from BLS (used as initial guess).

    Returns
    -------
    dict with score, bic_trapezoid, bic_vshape, delta_bic, verdict.
    """
    # Work in transit window only
    half = (duration / period) * 2.0   # ±2× duration in phase
    mask = np.abs(phase) < half
    if mask.sum() < 8:
        return {
            "score": 0, "verdict": "inconclusive (too few points in transit window)",
            "bic_trapezoid": np.nan, "bic_vshape": np.nan, "delta_bic": np.nan,
        }

    ph_w = phase[mask]
    fl_w = flux[mask]
    depth0 = max(primary_depth_ppm * 1e-6, 1e-6)
    t_dur_phase = duration / period
    baseline0 = np.median(fl_w)

    # --- Fit trapezoid ---
    try:
        p0_trap = [depth0, t_dur_phase * 0.4, t_dur_phase * 0.3, baseline0]
        bounds_trap = ([0, 0, 1e-6, 0.99], [0.1, 1.0, 1.0, 1.01])
        popt_trap, _ = curve_fit(
            _trapezoid_model, ph_w, fl_w,
            p0=p0_trap, bounds=bounds_trap, maxfev=5000,
        )
        resid_trap = fl_w - _trapezoid_model(ph_w, *popt_trap)
        bic_trap = _bic(resid_trap, n_params=4)
    except Exception as e:
        logger.warning("Trapezoid fit failed: %s", e)
        bic_trap = np.inf

    # --- Fit V-shape ---
    try:
        p0_v = [depth0, t_dur_phase, baseline0]
        bounds_v = ([0, 1e-6, 0.99], [0.1, 1.0, 1.01])
        popt_v, _ = curve_fit(
            _v_shape_model, ph_w, fl_w,
            p0=p0_v, bounds=bounds_v, maxfev=5000,
        )
        resid_v = fl_w - _v_shape_model(ph_w, *popt_v)
        bic_v = _bic(resid_v, n_params=3)
    except Exception as e:
        logger.warning("V-shape fit failed: %s", e)
        bic_v = np.inf

    delta_bic = bic_trap - bic_v   # negative = trapezoid is better

    # BIC rule-of-thumb: |ΔBIC| > 10 = strong evidence
    if np.isfinite(delta_bic):
        if delta_bic < -6:
            score = +1
            verdict = f"PASS - flat bottom (trapezoid) preferred (dBIC={delta_bic:.1f})"
        elif delta_bic > 6:
            score = -1
            verdict = f"FAIL - V-shape preferred (dBIC={delta_bic:.1f}); may be grazing EB"
        else:
            score = 0
            verdict = f"INCONCLUSIVE - models comparable (dBIC={delta_bic:.1f})"
    else:
        score = 0
        verdict = "INCONCLUSIVE - model fitting failed"

    logger.info(
        "Shape test: BIC_trap=%.2f, BIC_v=%.2f, dBIC=%.2f -> %s",
        bic_trap, bic_v, delta_bic if np.isfinite(delta_bic) else float('nan'), verdict,
    )
    return {
        "score": score, "verdict": verdict,
        "bic_trapezoid": bic_trap if np.isfinite(bic_trap) else None,
        "bic_vshape": bic_v if np.isfinite(bic_v) else None,
        "delta_bic": delta_bic if np.isfinite(delta_bic) else None,
    }


# ---------------------------------------------------------------------------
# Vetting summary and verdict
# ---------------------------------------------------------------------------

def vetting_verdict(test_results: Dict[str, Dict]) -> Dict:
    """
    Aggregate vetting test results into an overall disposition.

    Scoring
    -------
    Sum of individual scores:
    ≥ +2 : CANDIDATE (likely planet or planet candidate)
    0–1  : UNCLEAR (needs follow-up)
    ≤ -1 : FALSE POSITIVE (likely EB or artefact)

    Returns
    -------
    dict: overall_score, n_pass, n_fail, n_inconclusive, disposition.
    """
    scores = [v["score"] for v in test_results.values()]
    total = sum(scores)
    n_pass = scores.count(+1)
    n_fail = scores.count(-1)
    n_inc = scores.count(0)

    if total >= 2:
        disposition = "CANDIDATE"
    elif total <= -1:
        disposition = "FALSE POSITIVE"
    else:
        disposition = "UNCLEAR"

    return {
        "overall_score": total,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_inconclusive": n_inc,
        "disposition": disposition,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_vetting(
    phase: np.ndarray,
    flux: np.ndarray,
    period: float,
    duration: float,
    test_results: Dict,
    summary: Dict,
    target_id: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    2×2 panel plot showing odd/even phase fold and secondary eclipse window.
    """
    save_path = Path(save_path) if save_path else PLOTS_DIR / "vetting.png"

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"Transit Vetting — {target_id}\n"
        f"Disposition: {summary['disposition']}  "
        f"(score={summary['overall_score']}: "
        f"✓{summary['n_pass']} ✗{summary['n_fail']} ?{summary['n_inconclusive']})",
        fontsize=12, fontweight="bold",
    )

    # --- Panel 1: Full phase fold (primary transit) ---
    ax = axes[0, 0]
    ax.plot(phase, flux, ".", ms=1, color="#aabbdd", alpha=0.3)
    # Bin for clarity
    bins = np.linspace(-0.5, 0.5, 150)
    dig = np.digitize(phase, bins) - 1
    ph_b = [phase[dig == i].mean() for i in range(len(bins)-1) if (dig==i).sum() > 0]
    fl_b = [flux[dig == i].mean() for i in range(len(bins)-1) if (dig==i).sum() > 0]
    ax.plot(ph_b, fl_b, "o", ms=3, color="#3366bb")
    half_dur = duration / (2 * period)
    ax.axvspan(-half_dur, half_dur, color="red", alpha=0.1)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel("Phase"); ax.set_ylabel("Normalised Flux")
    ax.set_title("Full Phase Fold", fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Odd vs Even transits near transit ---
    ax = axes[0, 1]
    ax.set_xlim(-0.25, 0.25)
    oe = test_results.get("odd_even", {})
    ax.plot(phase, flux, ".", ms=1, color="#bbbbbb", alpha=0.2)
    ax.plot(ph_b, fl_b, "o", ms=3, color="#3366bb", alpha=0.7)
    ax.axvspan(-half_dur, half_dur, color="red", alpha=0.1)
    odd_d = oe.get("odd_depth_ppm", np.nan)
    even_d = oe.get("even_depth_ppm", np.nan)
    ax.set_title(
        f"Odd/Even Test: odd={odd_d:.1f} ppm, even={even_d:.1f} ppm\n"
        f"{oe.get('verdict', 'N/A')[:50]}",
        fontsize=8,
    )
    ax.set_xlabel("Phase"); ax.set_ylabel("Normalised Flux")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Secondary eclipse window (phase ≈ 0.5) ---
    ax = axes[1, 0]
    mask_sec = (np.abs(phase - 0.5) < 0.15) | (np.abs(phase + 0.5) < 0.15)
    phase_sec = phase - 0.5
    phase_sec = np.where(phase_sec < -0.5, phase_sec + 1.0, phase_sec)
    ax.plot(phase_sec, flux, ".", ms=1, color="#aabbdd", alpha=0.3)
    ax.axhline(np.median(flux), color="#888888", ls="--", lw=0.8)
    ax.set_xlim(-0.3, 0.3)
    sec = test_results.get("secondary", {})
    ax.set_title(
        f"Secondary Eclipse (phase=0.5)\n"
        f"{sec.get('verdict', 'N/A')[:55]}",
        fontsize=8,
    )
    ax.set_xlabel("Phase (shifted, 0=secondary)"); ax.set_ylabel("Normalised Flux")
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Summary scorecard ---
    ax = axes[1, 1]
    ax.axis("off")
    rows = []
    for test_name, res in test_results.items():
        score = res.get("score", 0)
        symbol = "✓" if score > 0 else ("✗" if score < 0 else "?")
        rows.append([test_name.replace("_", " ").title(), symbol,
                     res.get("verdict", "N/A")[:40]])
    rows.append(["OVERALL", summary["disposition"], f"score={summary['overall_score']}"])

    tbl = ax.table(
        cellText=rows,
        colLabels=["Test", "Result", "Verdict (truncated)"],
        loc="center", cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.6)
    # Colour the overall row
    for col_idx in range(3):
        cell = tbl[len(rows), col_idx]
        colour = {"CANDIDATE": "#ccffcc", "FALSE POSITIVE": "#ffcccc",
                  "UNCLEAR": "#ffffcc"}.get(summary["disposition"], "white")
        cell.set_facecolor(colour)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Vetting plot saved to: %s", save_path)


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

def run_vetting(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    best_signal: Dict,
    target_id: str = "target",
    star_radius_rsun: float = 1.065,
    star_mass_msun: float = 0.895,
    save_plot: bool = True,
) -> Tuple[Dict, Dict]:
    """
    Run all five vetting tests and return results + summary.

    Parameters
    ----------
    time, flux, flux_err : np.ndarray
        Detrended light curve arrays.
    best_signal : dict
        BLS best-signal dict (period, t0, duration, depth).
    target_id : str
    star_radius_rsun, star_mass_msun : float
        Stellar parameters for the duration-period test.
    save_plot : bool

    Returns
    -------
    test_results : dict
        Per-test result dicts.
    summary : dict
        Overall score and disposition.
    """
    period = best_signal["period"]
    t0 = best_signal["t0"]
    duration = best_signal["duration"]
    depth_ppm = best_signal["depth"] * 1e6

    # Phase-fold
    phase_raw = ((time - t0) % period) / period
    phase = np.where(phase_raw > 0.5, phase_raw - 1.0, phase_raw)
    sort_idx = np.argsort(phase)
    phase_s = phase[sort_idx]
    flux_s = flux[sort_idx]

    test_results = {}

    # Test 1
    test_results["odd_even"] = test_odd_even(time, flux, period, t0, duration)

    # Test 2
    test_results["secondary"] = test_secondary_eclipse(
        phase_s, flux_s, duration, period, depth_ppm
    )

    # Test 3
    test_results["centroid"] = test_centroid_shift(time, flux, period, t0, duration)

    # Test 4
    test_results["duration_period"] = test_duration_period_consistency(
        period, duration, star_radius_rsun, star_mass_msun
    )

    # Test 5
    test_results["shape"] = test_transit_shape(phase_s, flux_s, duration, period, depth_ppm)

    summary = vetting_verdict(test_results)

    # Print report
    print("\n" + "=" * 65)
    print(f"  VETTING REPORT - {target_id}")
    print("=" * 65)
    for name, res in test_results.items():
        score = res["score"]
        sym = "P" if score > 0 else ("F" if score < 0 else "?")
        print(f"  [{sym}] {name.replace('_', ' ').title():<28}: {res['verdict']}")
    print("-" * 65)
    print(f"  DISPOSITION: {summary['disposition']}  "
          f"(score={summary['overall_score']}: "
          f"P:{summary['n_pass']} F:{summary['n_fail']} ?:{summary['n_inconclusive']})")
    print("=" * 65 + "\n")

    if save_plot:
        tag = target_id.replace(" ", "_")
        plot_vetting(
            phase_s, flux_s, period, duration,
            test_results, summary, target_id,
            save_path=PLOTS_DIR / f"{tag}_vetting.png",
        )

    return test_results, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Transit vetting (5 tests) on a real light curve.")
    p.add_argument("--target", default="KIC 11904151")
    p.add_argument("--mission", default="Kepler", choices=["Kepler", "K2", "TESS"])
    p.add_argument("--star-radius", type=float, default=1.065,
                   help="Stellar radius [R_sun]")
    p.add_argument("--star-mass", type=float, default=0.895,
                   help="Stellar mass [M_sun]")
    p.add_argument("--bin-cadence", type=float, default=30.0)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    lc = download_lightcurve(args.target, mission=args.mission)
    time, flux, flux_err = preprocess(lc)
    cadence_min = float(np.median(np.diff(time)) * 24 * 60)
    baseline = float(time[-1] - time[0])

    _, detrended, _ = run_detrending(
        time=time, flux=flux,
        period_max_days=baseline / 3.0,
        cadence_minutes=cadence_min,
        method="savgol",
        target_id=args.target,
        save_plot=False,
    )

    time_bls, det_bls, err_bls = bin_lc_for_bls(
        time, detrended, flux_err, target_cadence_min=args.bin_cadence
    )
    period_grid = build_period_grid(baseline)
    duration_grid = build_duration_grid()
    _, best_signal = run_bls(time_bls, det_bls, err_bls, period_grid, duration_grid)

    test_results, summary = run_vetting(
        time=time,
        flux=detrended,
        flux_err=flux_err,
        best_signal=best_signal,
        target_id=args.target,
        star_radius_rsun=args.star_radius,
        star_mass_msun=args.star_mass,
        save_plot=True,
    )
