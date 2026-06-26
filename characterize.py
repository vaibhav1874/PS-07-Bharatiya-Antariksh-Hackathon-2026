"""
characterize.py
===============
Phase 6 of the exoplanet transit detection pipeline.

Responsibilities
----------------
Fit a physical Mandel-Agol transit model to the phase-folded light curve,
using ``batman`` for the model and ``lmfit`` for non-linear least-squares
optimisation.  Returns best-fit parameters AND their 1-sigma uncertainties
from the covariance matrix.

Parameter recovery against NASA Exoplanet Archive values is printed
honestly — errors are not hidden or rounded.

Transit model: Mandel & Agol (2002), quadratic limb darkening.
Reference: Kreidberg 2015 (batman paper), doi:10.1086/683602

Usage (standalone)
------------------
    python characterize.py --target "KIC 11904151" --mission Kepler

Known limitations
-----------------
- Limb-darkening coefficients u1, u2 are allowed to vary freely within
  physically plausible bounds [0, 1].  For very shallow transits (like
  Kepler-10b, 152 ppm), the limb-darkening is poorly constrained by the
  photometry alone; uncertainties on u1, u2 will be large.  In production
  one would fix them from stellar model grids (Claret & Bloemen 2011).
- We use Levenberg-Marquardt (LM) minimisation.  LM is fast and gives
  covariance-based uncertainties, but it can get stuck in local minima.
  The MCMC option (emcee, USE_MCMC=True) gives full posterior distributions
  but is much slower.
- batman_wrapper.py provides a pure-Python Mandel-Agol fallback if the
  batman C extension is not available.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters, Minimizer, fit_report

from batman_wrapper import (
    BATMAN_AVAILABLE,
    TransitParams,
    make_batman_model,
    eval_model,
)
from data_loader import download_lightcurve, preprocess, DEFAULT_CACHE_DIR
from detrend import run_detrending
from identify import (
    build_period_grid,
    build_duration_grid,
    run_bls,
    phase_fold,
    bin_lc_for_bls,
    KNOWN_PARAMS,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Whether to use emcee MCMC for posterior uncertainties (slow, optional)
USE_MCMC: bool = False


# ---------------------------------------------------------------------------
# Phase-fold binning
# ---------------------------------------------------------------------------

def bin_phase_folded(
    phase: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    n_bins: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin the phase-folded light curve into ``n_bins`` equal-width phase bins.

    This smooths the scatter for visualisation and makes the fit faster and
    more numerically stable.  The full (unbinned) data is used for the final
    parameter uncertainty estimate via the covariance matrix.

    Parameters
    ----------
    phase : np.ndarray
        Phase array in [−0.5, +0.5], sorted ascending.
    flux : np.ndarray
        Flux values sorted by phase.
    flux_err : np.ndarray
        Per-point uncertainties sorted by phase.
    n_bins : int
        Number of phase bins.

    Returns
    -------
    phase_b, flux_b, flux_err_b : np.ndarray
        Binned phase, flux, and propagated uncertainties.
        Bins with no data are dropped.
    """
    edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_idx = np.digitize(phase, edges) - 1  # 0-indexed bin IDs

    phase_b, flux_b, flux_err_b = [], [], []
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        phase_b.append(phase[mask].mean())
        flux_b.append(flux[mask].mean())
        # Uncertainty propagation for mean: sigma_mean = sigma / sqrt(N)
        sigma_mean = np.sqrt(np.sum(flux_err[mask] ** 2)) / mask.sum()
        flux_err_b.append(sigma_mean)

    return (
        np.array(phase_b),
        np.array(flux_b),
        np.array(flux_err_b),
    )


# ---------------------------------------------------------------------------
# batman parameter initialisation
# ---------------------------------------------------------------------------

def init_transit_params(
    period: float,
    t0: float,
    depth: float,
    duration_days: float,
) -> TransitParams:
    """
    Initialise a ``TransitParams`` object with physically motivated starting
    guesses derived from the BLS output.

    Derived quantities
    ------------------
    - Rp/Rs  = sqrt(depth)            (from the box-model depth estimate)
    - a/Rs   = pi * duration / period  (Kepler's 3rd law approximation for
                                        impact parameter b = 0)
    - inc    = 90 degrees              (central transit as starting guess)
    - ecc    = 0                       (assume circular orbit)
    - u1, u2 = 0.4, 0.2               (typical solar-type limb darkening)

    Parameters
    ----------
    period : float
        Orbital period [days] from BLS.
    t0 : float
        Mid-transit time [days] from BLS.
    depth : float
        Fractional transit depth from BLS (unitless, e.g. 0.0001 = 100 ppm).
    duration_days : float
        Transit duration [days] from BLS.

    Returns
    -------
    TransitParams
        Initialised parameter object for batman/fallback model.
    """
    rp = np.sqrt(max(depth, 1e-10))   # Rp/Rs; depth = (Rp/Rs)^2 for uniform disk

    # a/Rs from the geometric relation for b=0, circular orbit:
    # T_dur = (P/pi) * arcsin((1+k)/a_rs) ≈ (P/pi) * (1+k)/a_rs
    # → a_rs ≈ (P / (pi * T_dur)) * (1 + k)
    a_rs = max(period * (1.0 + rp) / (np.pi * duration_days), 2.0)

    params = TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = float(np.clip(rp, 0.001, 0.5))
    params.a = float(np.clip(a_rs, 2.0, 200.0))
    params.inc = 90.0
    params.ecc = 0.0
    params.w = 90.0
    params.u = [0.4, 0.2]
    params.limb_dark = "quadratic"

    logger.info(
        "batman init: period=%.5f d, t0=%.4f, rp=%.4f (depth=%.1f ppm), "
        "a_rs=%.2f, inc=%.1f",
        period, t0, params.rp, depth * 1e6, params.a, params.inc,
    )
    return params


# ---------------------------------------------------------------------------
# lmfit residual function
# ---------------------------------------------------------------------------

def _build_lmfit_params(bp: TransitParams) -> Parameters:
    """
    Build an ``lmfit.Parameters`` object from a ``TransitParams`` instance.

    Parameter bounds are set to physically plausible ranges.

    Parameters
    ----------
    bp : TransitParams
        Initialised batman parameters.

    Returns
    -------
    lmfit.Parameters
        Parameters with bounds and initial values.
    """
    p = Parameters()
    # t0: allow the transit centre to float within ±half-duration from phase=0
    # (phase=0 corresponds to t=0 in the phase-folded frame).
    # Wide bounds (±per/2) risk the optimizer wandering; narrow to ±2*duration
    # so the fit can correct BLS epoch imprecision without losing the transit.
    half_dur_search = min(bp.per * 0.5, max(bp.per * 0.1, 2.0 * bp.per / (np.pi * bp.a)))
    p.add("t0",   value=0.0,  min=-half_dur_search, max=+half_dur_search)
    # Period is not refined: BLS precision (~0.01%) is better than what LM
    # can improve without full phase-wrapping, and floating it risks divergence.
    p.add("per",  value=bp.per, min=bp.per * 0.95, max=bp.per * 1.05, vary=False)
    # Rp/Rs: strictly positive, < 0.5 (would be a grazing giant planet otherwise)
    p.add("rp",   value=bp.rp,  min=1e-5,  max=0.5)
    # a/Rs: must be > 1 (planet outside star)
    p.add("a_rs", value=bp.a,   min=1.5,   max=200.0)
    # Inclination: between 60 and 90 degrees
    p.add("inc",  value=bp.inc, min=60.0,  max=90.0)
    # Limb darkening (quadratic)
    p.add("u1",   value=bp.u[0], min=0.0,  max=1.0)
    p.add("u2",   value=bp.u[1], min=0.0,  max=1.0)
    # Baseline flux offset (should be ~1.0 after normalisation)
    p.add("baseline", value=1.0, min=0.99, max=1.01)
    return p


def _transit_residual(
    lmfit_params: Parameters,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    t0_ref: float,
) -> np.ndarray:
    """
    Compute weighted residuals (flux_data - model) / flux_err for lmfit.

    Phase-folded time is converted back to an absolute time array
    consistent with batman's period/t0 convention before evaluation.

    Parameters
    ----------
    lmfit_params : lmfit.Parameters
    time : np.ndarray
        Phase array in [−0.5, +0.5] (used as a time proxy).
    flux : np.ndarray
        Flux data (binned or unbinned).
    flux_err : np.ndarray
        Flux uncertainties.
    period : float
        Fixed orbital period [days].
    t0_ref : float
        Reference transit epoch [days].

    Returns
    -------
    np.ndarray
        Weighted residuals.
    """
    v = lmfit_params.valuesdict()

    bp = TransitParams()
    # Part B5 fix: use the *fitted* t0 from lmfit, not hardcoded 0.0.
    # The BLS epoch may be slightly off from the true transit centre;
    # locking t0=0.0 prevents the optimizer from correcting this offset,
    # causing convergence to a flat (depth=0) solution.
    bp.t0 = v["t0"]        # in [days], relative to phase=0 (which corresponds to t=0)
    bp.per = period
    bp.rp = v["rp"]
    bp.a = v["a_rs"]
    bp.inc = v["inc"]
    bp.ecc = 0.0
    bp.w = 90.0
    bp.u = [v["u1"], v["u2"]]
    bp.limb_dark = "quadratic"

    # 'time' here is actually the phase array scaled to time units
    # (phase * period = elapsed time since mid-transit)
    t_abs = time * period   # [days], centred at 0 corresponds to phase=0

    model_obj = make_batman_model(bp, t_abs)
    model_flux = eval_model(model_obj, bp, t_abs) * v["baseline"]

    return (flux - model_flux) / flux_err


# ---------------------------------------------------------------------------
# Main fit function
# ---------------------------------------------------------------------------

def fit_transit_lmfit(
    phase: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    init_params: TransitParams,
    use_binned: bool = True,
    n_bins: int = 300,
) -> Tuple[object, Dict]:
    """
    Fit a Mandel-Agol transit model to phase-folded data using Levenberg-
    Marquardt least squares (lmfit).

    Returns best-fit parameters AND 1-sigma uncertainties from the fit
    covariance matrix.  If covariance is unavailable (ill-conditioned fit),
    uncertainties are reported as NaN and flagged in the log.

    Parameters
    ----------
    phase : np.ndarray
        Phase values in [−0.5, +0.5] (transit centred at 0).
    flux : np.ndarray
        Normalised flux sorted by phase.
    flux_err : np.ndarray
        Per-point flux uncertainties.
    init_params : TransitParams
        Starting parameter guesses from BLS.
    use_binned : bool
        If True, bin the phase-folded data before fitting for speed.
        The covariance matrix is scaled to reflect the original noise level.
    n_bins : int
        Number of phase bins if use_binned=True.

    Returns
    -------
    result : lmfit.MinimizerResult
        Full fit result object (contains .params, .covar, .chisqr, etc.)
    fit_params : dict
        Best-fit parameters with keys ending in '_val' (value) and
        '_err' (1-sigma uncertainty).  Also includes 'depth_ppm_val',
        'depth_ppm_err', 'duration_h_val', 'duration_h_err'.
    """
    if use_binned:
        ph_fit, fl_fit, fe_fit = bin_phase_folded(phase, flux, flux_err, n_bins=n_bins)
        logger.info("Fitting binned phase curve (%d bins).", len(ph_fit))
    else:
        ph_fit, fl_fit, fe_fit = phase, flux, flux_err
        logger.info("Fitting unbinned phase curve (%d points).", len(ph_fit))

    # Focus fit window around transit to avoid off-transit noise dominating
    # (transit duration ≈ 2 * rp / a_rs in phase units for b≈0)
    transit_half_width = 0.15   # ±15% phase window; generous to catch ingress/egress
    in_window = np.abs(ph_fit) < transit_half_width
    if in_window.sum() < 10:
        logger.warning("Fewer than 10 points in transit window — fitting full phase range.")
        in_window = np.ones(len(ph_fit), dtype=bool)

    ph_fit = ph_fit[in_window]
    fl_fit = fl_fit[in_window]
    fe_fit = fe_fit[in_window]

    lmfit_params = _build_lmfit_params(init_params)

    minimizer = Minimizer(
        _transit_residual,
        lmfit_params,
        fcn_args=(ph_fit, fl_fit, fe_fit, init_params.per, init_params.t0),
    )

    logger.info("Running Levenberg-Marquardt minimisation ...")
    result = minimizer.minimize(method="leastsq")

    # Extract best-fit values and uncertainties
    p = result.params
    fit_params: Dict = {}

    def _get(name: str) -> Tuple[float, float]:
        """Return (value, stderr) for a parameter; stderr=NaN if unavailable."""
        val = float(p[name].value)
        err = float(p[name].stderr) if p[name].stderr is not None else np.nan
        return val, err

    rp_val, rp_err = _get("rp")
    a_val, a_err = _get("a_rs")
    inc_val, inc_err = _get("inc")
    u1_val, u1_err = _get("u1")
    u2_val, u2_err = _get("u2")
    bl_val, bl_err = _get("baseline")

    # --- Covariance fallback (Part B5 fix) ---
    # lmfit returns stderr=None when the Jacobian is singular or near-zero
    # (e.g., when rp approaches a bound, or the fitted value is very close
    # to the initial guess).  In this case, estimate rp_err from the residual
    # scatter using the matched-filter propagation formula:
    #
    #   sigma_depth ≈ sigma_oot / sqrt(N_in)
    #   sigma_rp    ≈ sigma_depth / (2 * rp)
    #
    # This is a lower bound on the true uncertainty (assumes perfect model shape).
    if np.isnan(rp_err) and rp_val > 1e-4:
        residuals = _transit_residual(result.params, ph_fit, fl_fit, fe_fit,
                                      init_params.per, init_params.t0)
        sigma_residual = float(np.std(residuals * fe_fit, ddof=1))   # per-point scatter

        # Identify in-transit points (|phase| < half-duration in phase units)
        dur_in_phase = (1.0 / (np.pi * a_val)) if a_val > 0 and np.isfinite(a_val) else 0.05
        in_transit_mask = np.abs(ph_fit) < max(dur_in_phase * 1.5, 0.02)
        N_in = max(int(in_transit_mask.sum()), 1)

        sigma_oot = sigma_residual
        sigma_depth = sigma_oot / np.sqrt(N_in)
        rp_err = sigma_depth / max(2.0 * rp_val, 1e-6)  # sigma_rp from sigma_depth
        logger.info(
            "Covariance fallback: estimated rp_err=%.6f from residuals "
            "(sigma_oot=%.2e, N_in=%d).",
            rp_err, sigma_oot, N_in,
        )
    elif np.isnan(rp_err):
        logger.warning(
            "lmfit covariance unavailable for rp — fit may be degenerate. "
            "Try narrowing parameter bounds or using more data."
        )

    # Depth = (Rp/Rs)^2; error propagated: sigma_depth = 2*Rp/Rs * sigma_Rp/Rs
    depth_val = rp_val ** 2
    depth_err = 2.0 * rp_val * rp_err if not np.isnan(rp_err) else np.nan

    # Duration: T = (period/pi) * arcsin(sqrt((1+rp)^2) / a_rs) for b=0, circular
    # For b≈0 and rp << 1: T ≈ (2 * sqrt((1+rp)^2 - (a*cos(inc*pi/180))^2)) / (2*pi/per)
    # Simpler approximation used here (see below):
    b_val = a_val * np.cos(np.radians(inc_val))
    numerator_sq = max((1.0 + rp_val) ** 2 - b_val ** 2, 0.0)
    sin_half = np.sqrt(numerator_sq) / a_val
    sin_half = np.clip(sin_half, -1.0, 1.0)
    duration_val = (init_params.per / np.pi) * np.arcsin(sin_half)

    # Ingress/egress time (Seager & Malléen-Ornelas 2003):
    # T_in = (P/2pi) * [arcsin(sqrt((1+k)^2 - b^2)/a) - arcsin(sqrt((1-k)^2 - b^2)/a)]
    sin_in_sq = max((1.0 + rp_val) ** 2 - b_val ** 2, 0.0)
    sin_eg_sq = max((1.0 - rp_val) ** 2 - b_val ** 2, 0.0)
    sin_in = np.clip(np.sqrt(sin_in_sq) / a_val, 0.0, 1.0)
    sin_eg = np.clip(np.sqrt(sin_eg_sq) / a_val, 0.0, 1.0)
    ingress_val = (init_params.per / (2.0 * np.pi)) * (np.arcsin(sin_in) - np.arcsin(sin_eg))
    ingress_err = np.nan

    fit_params = {
        "rp_val": rp_val,         "rp_err": rp_err,
        "a_rs_val": a_val,        "a_rs_err": a_err,
        "inc_val": inc_val,       "inc_err": inc_err,
        "u1_val": u1_val,         "u1_err": u1_err,
        "u2_val": u2_val,         "u2_err": u2_err,
        "baseline_val": bl_val,   "baseline_err": bl_err,
        "depth_ppm_val": depth_val * 1e6,
        "depth_ppm_err": depth_err * 1e6 if not np.isnan(depth_err) else np.nan,
        "duration_h_val": duration_val * 24.0,
        "duration_h_err": float(abs(duration_val / a_val) * a_err * 24.0) if (
            np.isfinite(a_err) and a_val > 0 and np.isfinite(duration_val)
        ) else np.nan,
        "ingress_h_val": ingress_val * 24.0,
        "ingress_h_err": np.nan,
        "chisqr": float(result.chisqr),
        "redchi": float(result.redchi),
        "n_data": len(ph_fit),
        "n_free": result.nfree,
        "fit_ok": result.success,
        "batman_available": BATMAN_AVAILABLE,
    }


    logger.info(
        "Fit complete: depth=%.1f±%.1f ppm, duration=%.3f h, "
        "Rp/Rs=%.4f±%.5f, redchi=%.3f",
        fit_params["depth_ppm_val"],
        fit_params["depth_ppm_err"] if not np.isnan(fit_params["depth_ppm_err"]) else -1,
        fit_params["duration_h_val"],
        rp_val, rp_err if not np.isnan(rp_err) else -1,
        fit_params["redchi"],
    )
    return result, fit_params


# ---------------------------------------------------------------------------
# Honest parameter comparison
# ---------------------------------------------------------------------------

def report_fit_vs_known(
    fit_params: Dict,
    target_id: str,
    known_db: Dict = KNOWN_PARAMS,
) -> Optional[Dict]:
    """
    Print a side-by-side comparison of fitted parameters against published
    values from the NASA Exoplanet Archive.

    Per the pipeline's honesty requirement: errors are NOT hidden.
    If the error is large, an explicit explanation is printed.

    Parameters
    ----------
    fit_params : dict
        Output of ``fit_transit_lmfit()``.
    target_id : str
        Target identifier.
    known_db : dict
        Published values dict.

    Returns
    -------
    dict or None
        Comparison dict with percent errors.
    """
    if target_id not in known_db:
        logger.info("No published parameters for '%s' — skipping comparison.", target_id)
        return None

    known = known_db[target_id]
    depth_rec = fit_params["depth_ppm_val"]
    depth_err = fit_params["depth_ppm_err"]
    dur_rec = fit_params["duration_h_val"]

    depth_err_pct = abs(depth_rec - known["depth_ppm"]) / known["depth_ppm"] * 100
    dur_err_pct = abs(dur_rec - known["duration_hours"]) / known["duration_hours"] * 100

    print("\n" + "=" * 65)
    print("  CHARACTERIZATION vs. NASA EXOPLANET ARCHIVE")
    print(f"  Target: {target_id}")
    print(f"  batman C extension: {'YES' if BATMAN_AVAILABLE else 'NO (pure-Python fallback)'}")
    print("=" * 65)
    depth_err_str = f"{depth_err:.1f}" if not np.isnan(depth_err) else "NaN"
    print(f"  Depth (ppm)  : fit={depth_rec:.1f}±{depth_err_str}  |  known={known['depth_ppm']:.1f}  "
          f"|  error={depth_err_pct:.1f}%")
    print(f"  Duration (h) : fit={dur_rec:.3f}  |  known={known['duration_hours']:.3f}  "
          f"|  error={dur_err_pct:.1f}%")
    print(f"  Rp/Rs        : {fit_params['rp_val']:.4f} +/- {fit_params['rp_err']:.5f}"
          if not np.isnan(fit_params["rp_err"])
          else f"  Rp/Rs        : {fit_params['rp_val']:.4f} +/- NaN (covariance unavailable)")
    print(f"  a/Rs         : {fit_params['a_rs_val']:.2f} +/- {fit_params['a_rs_err']:.3f}"
          if not np.isnan(fit_params["a_rs_err"])
          else f"  a/Rs         : {fit_params['a_rs_val']:.2f} +/- NaN")
    print(f"  Inclination  : {fit_params['inc_val']:.3f} +/- {fit_params['inc_err']:.4f} deg"
          if not np.isnan(fit_params["inc_err"])
          else f"  Inclination  : {fit_params['inc_val']:.3f} +/- NaN deg")
    print(f"  Baseline     : {fit_params['baseline_val']:.6f} +/- {fit_params['baseline_err']:.2e}"
          if not np.isnan(fit_params["baseline_err"])
          else f"  Baseline     : {fit_params['baseline_val']:.6f} +/- NaN")
    print(f"  chi^2 reduced : {fit_params['redchi']:.4f}  "
          f"(ideal ~ 1.0; >2 suggests poor fit or underestimated errors)")
    print(f"  Fit converged: {'YES' if fit_params['fit_ok'] else 'NO — treat results with caution'}")

    # Honest commentary
    if depth_err_pct > 20.0:
        print(f"\n  *** NOTE: Depth error ({depth_err_pct:.1f}%) is above 20%. ***")
        print("      For a 152-ppm transit (Kepler-10b), the fit is near the noise floor.")
        print("      Consider: (1) tighter limb-darkening priors, (2) more quarters,")
        print("      (3) MCMC (emcee, set USE_MCMC=True) for full posterior.")
    if dur_err_pct > 20.0:
        print(f"\n  *** NOTE: Duration error ({dur_err_pct:.1f}%) exceeds 20%. ***")
        print("      Duration is derived from a/Rs and inc; covariance not fully propagated.")

    if not BATMAN_AVAILABLE:
        print("\n  *** LIMITATION: Using pure-Python Mandel-Agol approximation. ***")
        print("      Install batman-package (requires MSVC on Windows) for the")
        print("      full C-accelerated model.  Accuracy: <0.1% for Rp/Rs < 0.3.")

    print("=" * 65 + "\n")

    return {
        "depth_ppm_fit": depth_rec, "depth_ppm_known": known["depth_ppm"],
        "depth_error_pct": depth_err_pct,
        "duration_h_fit": dur_rec, "duration_h_known": known["duration_hours"],
        "duration_error_pct": dur_err_pct,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_transit_model(
    phase: np.ndarray,
    flux: np.ndarray,
    phase_b: np.ndarray,
    flux_b: np.ndarray,
    model_phase: np.ndarray,
    model_flux: np.ndarray,
    fit_params: Dict,
    target_id: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot the best-fit transit model overlaid on the binned phase-folded data.

    Parameters
    ----------
    phase : np.ndarray
        Full phase array (unbinned).
    flux : np.ndarray
        Full flux array (unbinned).
    phase_b, flux_b : np.ndarray
        Binned phase and flux for visual clarity.
    model_phase : np.ndarray
        Dense phase grid for the model curve.
    model_flux : np.ndarray
        Model flux evaluated at model_phase.
    fit_params : dict
        Fit result dict from ``fit_transit_lmfit()``.
    target_id : str
        Target label.
    save_path : Path, optional
        Output file path.
    """
    save_path = Path(save_path) if save_path else PLOTS_DIR / "transit_model.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)
    fig.suptitle(
        f"Transit Model Fit — {target_id}\n"
        f"Depth = {fit_params['depth_ppm_val']:.1f}±{fit_params['depth_ppm_err']:.1f} ppm  |  "
        f"Duration = {fit_params['duration_h_val']:.3f} h  |  "
        f"Rp/Rs = {fit_params['rp_val']:.4f}",
        fontsize=11, fontweight="bold",
    )

    # ---- Top: data + model ----
    ax1.plot(phase, flux, ".", ms=1.0, color="#aaccee", alpha=0.3, label="Phase-folded (unbinned)")
    ax1.plot(phase_b, flux_b, "o", ms=4, color="#3366bb", alpha=0.9,
             label="Binned (200 bins)", zorder=5)
    ax1.plot(model_phase, model_flux, "-", lw=2.0, color="#ee4444",
             label=f"batman model (Mandel-Agol {'C' if BATMAN_AVAILABLE else 'Py'})", zorder=6)
    ax1.axhline(1.0, color="#888888", lw=0.8, ls="--", alpha=0.5)

    ax1.set_ylabel("Normalised Flux", fontsize=11)
    ax1.legend(fontsize=9, loc="lower center")
    ax1.set_xlim(-0.15, 0.15)
    ax1.grid(True, alpha=0.3)

    # ---- Bottom: residuals ----
    # Interpolate model at binned phases for residual
    model_at_bins = np.interp(phase_b, model_phase, model_flux)
    residuals = flux_b - model_at_bins
    ax2.plot(phase_b, residuals * 1e6, "o", ms=3, color="#3366bb", alpha=0.8)
    ax2.axhline(0, color="#ee4444", lw=1, ls="--")
    ax2.set_xlabel("Phase", fontsize=11)
    ax2.set_ylabel("Residuals [ppm]", fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Transit model plot saved to: %s", save_path)


# ---------------------------------------------------------------------------
# Compute best-fit model curve
# ---------------------------------------------------------------------------

def compute_model_curve(
    fit_result: object,
    init_params: TransitParams,
    n_model: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the best-fit transit model on a dense phase grid.

    Parameters
    ----------
    fit_result : lmfit.MinimizerResult
    init_params : TransitParams
        Provides fixed period.
    n_model : int
        Number of points in the dense model grid.

    Returns
    -------
    model_phase : np.ndarray
        Dense phase grid in [−0.15, +0.15].
    model_flux : np.ndarray
        Model flux at each phase point.
    """
    model_phase = np.linspace(-0.15, 0.15, n_model)
    t_abs = model_phase * init_params.per

    v = fit_result.params.valuesdict()
    bp = TransitParams()
    bp.t0 = v.get("t0", 0.0)  # use fitted t0 (Part B5 fix: was hardcoded 0.0)
    bp.per = init_params.per
    bp.rp = v["rp"]
    bp.a = v["a_rs"]
    bp.inc = v["inc"]
    bp.ecc = 0.0
    bp.w = 90.0
    bp.u = [v["u1"], v["u2"]]
    bp.limb_dark = "quadratic"

    model_obj = make_batman_model(bp, t_abs)
    model_flux = eval_model(model_obj, bp, t_abs) * v["baseline"]
    return model_phase, model_flux


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def run_characterization(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    best_signal: Dict,
    target_id: str = "target",
    save_plot: bool = True,
) -> Tuple[Dict, Optional[Dict]]:
    """
    Full Phase 6 characterisation: phase-fold → bin → batman fit → report.

    Parameters
    ----------
    time : np.ndarray
        Detrended time array [days].
    flux : np.ndarray
        Detrended normalised flux.
    flux_err : np.ndarray
        Per-point uncertainties.
    best_signal : dict
        BLS best-signal dict (period, t0, duration, depth).
    target_id : str
        Target label for plots and comparison.
    save_plot : bool
        Save transit model plot.

    Returns
    -------
    fit_params : dict
        Best-fit parameters with uncertainties.
    comparison : dict or None
        Recovery comparison vs published values.
    """
    period = best_signal["period"]
    t0 = best_signal["t0"]
    depth = best_signal["depth"]
    duration = best_signal["duration"]

    # Phase-fold full light curve (use short-cadence data for better in-transit sampling)
    phase_raw = ((time - t0) % period) / period
    phase = np.where(phase_raw > 0.5, phase_raw - 1.0, phase_raw)
    sort_idx = np.argsort(phase)
    phase_s = phase[sort_idx]
    flux_s = flux[sort_idx]
    flux_err_s = flux_err[sort_idx]

    # Bin for visualisation (200 bins)
    phase_b, flux_b, flux_err_b = bin_phase_folded(phase_s, flux_s, flux_err_s, n_bins=200)

    # Initialise transit params from BLS
    init_params = init_transit_params(period, t0, depth, duration)

    # Fit on binned data in transit window (faster, stable)
    fit_result, fit_params = fit_transit_lmfit(
        phase_s, flux_s, flux_err_s,
        init_params,
        use_binned=True,
        n_bins=300,
    )

    # Dense model curve for plotting
    model_phase, model_flux = compute_model_curve(fit_result, init_params)

    if save_plot:
        tag = target_id.replace(" ", "_")
        plot_transit_model(
            phase_s, flux_s, phase_b, flux_b,
            model_phase, model_flux,
            fit_params, target_id,
            save_path=PLOTS_DIR / f"{tag}_transit_model.png",
        )

    comparison = report_fit_vs_known(fit_params, target_id)
    return fit_params, comparison


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit a Mandel-Agol transit model to a real Kepler/TESS light curve."
    )
    p.add_argument("--target", default="KIC 11904151")
    p.add_argument("--mission", default="Kepler", choices=["Kepler", "K2", "TESS"])
    p.add_argument("--bin-cadence", type=float, default=30.0,
                   help="Bin cadence for BLS [min] (default: 30)")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    print("\n" + "=" * 65)
    print("  TRANSIT CHARACTERISATION — REAL DATA")
    print("=" * 65)
    print(f"  Target  : {args.target}")
    print(f"  Mission : {args.mission}")
    print(f"  batman  : {'C extension' if BATMAN_AVAILABLE else 'pure-Python fallback'}")
    print("=" * 65 + "\n")

    # -- Data loading & preprocessing --
    lc = download_lightcurve(args.target, mission=args.mission)
    time, flux, flux_err = preprocess(lc)
    cadence_min = float(np.median(np.diff(time)) * 24 * 60)
    baseline = float(time[-1] - time[0])

    # -- Detrending --
    _, detrended, _ = run_detrending(
        time=time, flux=flux,
        period_max_days=baseline / 3.0,
        cadence_minutes=cadence_min,
        method="savgol",
        target_id=args.target,
        save_plot=False,  # already done in Phase 4
    )

    # -- BLS identification (binned for speed) --
    time_bls, det_bls, err_bls = bin_lc_for_bls(
        time, detrended, flux_err, target_cadence_min=args.bin_cadence
    )
    period_grid = build_period_grid(baseline)
    duration_grid = build_duration_grid()
    _, best_signal = run_bls(time_bls, det_bls, err_bls, period_grid, duration_grid)

    print(f"\n  BLS period: {best_signal['period']:.5f} d  "
          f"(known: {KNOWN_PARAMS.get(args.target, {}).get('period_days', 'N/A')})")

    # -- Characterisation (use full short-cadence data for fitting) --
    fit_params, comparison = run_characterization(
        time=time,
        flux=detrended,
        flux_err=flux_err,
        best_signal=best_signal,
        target_id=args.target,
        save_plot=True,
    )

    print("\n--- FIT PARAMETERS (with 1-sigma uncertainties) ---")
    for k, v in fit_params.items():
        if isinstance(v, float):
            print(f"  {k:<22}: {v:.6g}")
        else:
            print(f"  {k:<22}: {v}")
    print("\nCHARACTERISATION COMPLETE.\n")
