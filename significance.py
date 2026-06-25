"""
significance.py
===============
Phase 8 of the exoplanet transit detection pipeline.

Responsibilities
----------------
1. **SNR** — transit signal-to-noise ratio using the box-averaging formula:
       SNR = depth / (sigma_oot / sqrt(N_in_transit))
   where sigma_oot is the per-point scatter of the out-of-transit baseline
   and N_in_transit is the number of cadences inside the transit window.
   (Box-least-squares matched-filter SNR; see Kovacs et al. 2002, A&A 391, 369.)

2. **False Alarm Probability (FAP)** via bootstrap phase-shuffling:
   Randomly shuffle the phase of the detrended light curve N_trials times,
   re-run BLS on each shuffle, record the maximum BLS power, and report what
   fraction of trials achieved a power >= the real detection.
   (Jenkins et al. 2002 approach adapted for bootstrapped non-parametric FAP.)

Usage (standalone)
------------------
    python significance.py --target "KIC 11904151" --mission Kepler
"""

from __future__ import annotations

import argparse
import logging
import time as _time_module
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import BoxLeastSquares

logger = logging.getLogger(__name__)

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: transit mask
# ---------------------------------------------------------------------------

def _transit_mask(
    time: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    width_factor: float = 1.5,
) -> np.ndarray:
    """
    Boolean mask — True for cadences within ``width_factor * duration / 2``
    of a transit centre.

    Parameters
    ----------
    time : np.ndarray
    period, t0, duration : float  [days]
    width_factor : float  expand window to catch ingress/egress edges

    Returns
    -------
    np.ndarray of bool
    """
    phase_raw = ((time - t0) % period) / period
    phase = np.where(phase_raw > 0.5, phase_raw - 1.0, phase_raw)
    half_width = (duration * width_factor) / (2.0 * period)
    return np.abs(phase) < half_width


# ---------------------------------------------------------------------------
# Phase 8.1–8.2 : SNR
# ---------------------------------------------------------------------------

def compute_snr(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    depth: float,
    width_factor: float = 1.5,
) -> Dict:
    """
    Compute the transit SNR using the matched-filter box formula.

    Formula (shown explicitly per the project spec):
        sigma_oot = std(flux[out-of-transit])
        N_in      = number of in-transit cadences
        SNR       = depth / (sigma_oot / sqrt(N_in))

    This is equivalent to the BLS detection efficiency formula described
    in Kovacs et al. 2002, A&A 391, 369, §2.

    Parameters
    ----------
    time : np.ndarray  [days]
    flux : np.ndarray  [relative flux, median ≈ 1.0]
    period : float     [days]
    t0 : float         [days]
    duration : float   [days]
    depth : float      [relative flux units, positive value = dip]
    width_factor : float
        Expand transit window by this factor when counting in-transit points.

    Returns
    -------
    dict with keys:
        snr               – signal-to-noise ratio (float)
        n_in_transit      – number of in-transit cadences
        n_out_transit     – number of out-of-transit cadences
        sigma_oot         – per-point scatter outside transit [flux units]
        depth_used        – depth value used in calculation
        noise_on_transit  – sigma_oot / sqrt(N_in)  (denominator of SNR)
    """
    in_mask = _transit_mask(time, period, t0, duration, width_factor)
    oot_flux = flux[~in_mask]

    if oot_flux.size < 10:
        logger.warning("Too few out-of-transit points (%d) for reliable SNR.", oot_flux.size)
        return {"snr": np.nan, "n_in_transit": int(in_mask.sum()),
                "n_out_transit": int(oot_flux.size), "sigma_oot": np.nan,
                "depth_used": depth, "noise_on_transit": np.nan}

    n_in = int(in_mask.sum())
    if n_in < 3:
        logger.warning("Only %d in-transit cadences — SNR unreliable.", n_in)

    # sigma_oot: per-point robust scatter out of transit.
    # Part B7: use robust estimator (1.4826 * MAD) rather than std, which
    # is inflated by any residual outliers in the out-of-transit baseline.
    oot_median = np.median(oot_flux)
    mad = np.median(np.abs(oot_flux - oot_median))
    sigma_oot = float(1.4826 * mad)   # consistent sigma for Gaussian noise

    # Fallback to std if MAD is zero (pathological, e.g., flat synthetic data)
    if sigma_oot == 0.0:
        sigma_oot = float(np.std(oot_flux, ddof=1))
        logger.debug("MAD=0 for OOT flux; falling back to std.")

    # noise on the transit box depth = sigma_oot / sqrt(N_in)
    noise_on_transit = sigma_oot / np.sqrt(max(n_in, 1))

    # SNR = depth / noise_on_transit  (Kovacs et al. 2002 / Part B7 formula)
    snr = float(depth / noise_on_transit) if noise_on_transit > 0 else np.nan

    result = {
        "snr": snr,
        "n_in_transit": n_in,
        "n_out_transit": int(oot_flux.size),
        "sigma_oot": sigma_oot,
        "depth_used": float(depth),
        "noise_on_transit": noise_on_transit,
    }

    logger.info(
        "SNR: depth=%.2e, sigma_oot=%.2e, N_in=%d → SNR=%.2f",
        depth, sigma_oot, n_in, snr,
    )
    return result


# ---------------------------------------------------------------------------
# Phase 8.3 : Bootstrap FAP
# ---------------------------------------------------------------------------

def _run_bls_single(
    time: np.ndarray,
    flux: np.ndarray,
    period_min: float = 0.5,
    period_max_frac: float = 0.33,
    n_periods: int = 5000,
    duration_grid: Optional[np.ndarray] = None,
) -> float:
    """
    Run BLS on the provided time/flux arrays and return the maximum power.

    Parameters
    ----------
    time : np.ndarray
    flux : np.ndarray
    period_min : float        minimum period to search [days]
    period_max_frac : float   max period = period_max_frac * baseline [days]
    n_periods : int           number of periods in the grid
    duration_grid : np.ndarray or None  transit duration grid [days]

    Returns
    -------
    float : maximum BLS power over the grid
    """
    baseline = float(time[-1] - time[0])
    period_max = period_max_frac * baseline

    if period_max <= period_min:
        return np.nan

    period_grid = np.linspace(period_min, period_max, n_periods)
    if duration_grid is None:
        duration_grid = np.linspace(0.01, 0.3, 50)

    bls = BoxLeastSquares(time, flux)
    try:
        result = bls.power(period_grid, duration_grid, method="fast", objective="snr")
        return float(np.nanmax(result.power))
    except Exception as exc:
        logger.debug("BLS trial failed: %s", exc)
        return np.nan


def compute_fap_bootstrap(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    real_power: float,
    n_trials: int = 1000,
    rng_seed: int = 42,
    period_min: float = 0.5,
    n_periods: int = 2000,
    duration_grid: Optional[np.ndarray] = None,
    save_plot: bool = True,
    target_id: str = "target",
) -> Dict:
    """
    Estimate False Alarm Probability (FAP) via out-of-transit phase shuffling.

    Method
    ------
    1. Identify out-of-transit cadences (using transit window ×1.5).
    2. For each of N_trials trials:
       a. Randomly shuffle the *flux values* of the full light curve
          (preserving time stamps) — this destroys any periodic signal.
       b. Run BLS on the shuffled series.
       c. Record the peak BLS power.
    3. FAP = fraction of trials with peak power >= real_power.

    Note: We shuffle the full flux rather than phase-shift so that the cadence
    pattern (gaps, etc.) is preserved, giving a realistic null distribution.

    Parameters
    ----------
    time : np.ndarray
    flux : np.ndarray
    period, t0, duration : float  [days]  — best-fit transit parameters
    real_power : float            — BLS peak power of the real detection
    n_trials : int                — number of bootstrap iterations (≥1000 recommended)
    rng_seed : int                — random seed for reproducibility
    period_min : float            — minimum BLS period [days]
    n_periods : int               — period grid size for each trial (reduce for speed)
    duration_grid : np.ndarray or None
    save_plot : bool
    target_id : str

    Returns
    -------
    dict with keys:
        fap           – false alarm probability [0, 1]
        n_trials      – number of bootstrap iterations completed
        n_exceeding   – number of trials that exceeded the real power
        real_power    – BLS power of real detection (passed in)
        trial_powers  – np.ndarray of peak powers from shuffled trials
        elapsed_s     – wall-clock time for the bootstrap
    """
    rng = np.random.default_rng(rng_seed)
    baseline = float(time[-1] - time[0])
    period_max_frac = 0.33

    trial_powers: list[float] = []
    t_start = _time_module.time()

    logger.info(
        "Bootstrap FAP: %d trials, real_power=%.4f …", n_trials, real_power
    )

    if n_trials < 1000:
        logger.warning(
            "n_trials=%d < 1000 — FAP estimate is itself unreliable. "
            "Per Part B7: N_bootstrap >> 1/FAP is required for a stable estimate. "
            "Increase n_trials to ≥1000 (5000–10000 for FAP claims < 0.001).",
            n_trials,
        )

    for i in range(n_trials):
        # Part B7: use circular shift (not naive permutation) to preserve
        # autocorrelation / red-noise structure.  A naive shuffle destroys
        # all correlated noise, giving an over-optimistic FAP for red noise.
        shift = rng.integers(1, len(flux))          # random shift amount
        shifted_flux = np.roll(flux, shift)          # circular roll
        p = _run_bls_single(
            time, shifted_flux,
            period_min=period_min,
            period_max_frac=period_max_frac,
            n_periods=n_periods,
            duration_grid=duration_grid,
        )
        trial_powers.append(p)

        if (i + 1) % 100 == 0:
            elapsed = _time_module.time() - t_start
            logger.info("  … %d / %d trials done (%.1f s)", i + 1, n_trials, elapsed)

    elapsed_s = float(_time_module.time() - t_start)
    trial_powers_arr = np.array(trial_powers, dtype=float)
    valid = trial_powers_arr[np.isfinite(trial_powers_arr)]

    if valid.size == 0:
        logger.error("All bootstrap trials returned NaN — FAP undefined.")
        return {"fap": np.nan, "n_trials": n_trials, "n_exceeding": np.nan,
                "real_power": real_power, "trial_powers": trial_powers_arr,
                "elapsed_s": elapsed_s}

    n_exceeding = int(np.sum(valid >= real_power))
    fap = n_exceeding / len(valid)

    logger.info(
        "FAP = %d / %d = %.4f  (elapsed %.1f s)",
        n_exceeding, len(valid), fap, elapsed_s,
    )

    if save_plot:
        _plot_fap_distribution(
            trial_powers_arr, real_power, fap, target_id
        )

    return {
        "fap": fap,
        "n_trials": n_trials,
        "n_exceeding": n_exceeding,
        "real_power": real_power,
        "trial_powers": trial_powers_arr,
        "elapsed_s": elapsed_s,
    }


# ---------------------------------------------------------------------------
# Plot: FAP null distribution
# ---------------------------------------------------------------------------

def _plot_fap_distribution(
    trial_powers: np.ndarray,
    real_power: float,
    fap: float,
    target_id: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot the bootstrap null distribution of BLS peak powers and mark the
    real detection power.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    valid = trial_powers[np.isfinite(trial_powers)]
    ax.hist(valid, bins=50, color="#4a90d9", edgecolor="white", linewidth=0.5,
            alpha=0.85, label=f"Shuffled trials (N={len(valid)})")
    ax.axvline(real_power, color="#e74c3c", lw=2.5, linestyle="--",
               label=f"Real detection power = {real_power:.3f}")

    ax.set_xlabel("Peak BLS Power (shuffled null)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"{target_id} — Bootstrap FAP null distribution\n"
        f"FAP = {fap:.4f}  ({int(np.sum(valid >= real_power))} / {len(valid)} trials exceed real power)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    fig.tight_layout()

    tag = target_id.replace(" ", "_")
    path = save_path or (PLOTS_DIR / f"{tag}_fap_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("FAP distribution plot saved → %s", path)


# ---------------------------------------------------------------------------
# High-level wrapper for pipeline integration
# ---------------------------------------------------------------------------

def run_significance(
    time: np.ndarray,
    flux: np.ndarray,
    best_signal: dict,
    n_fap_trials: int = 1000,
    rng_seed: int = 42,
    target_id: str = "target",
    save_plot: bool = True,
    skip_fap: bool = False,
) -> Dict:
    """
    Run Phase 8 (statistical significance) and return a combined result dict.

    Parameters
    ----------
    time : np.ndarray  [days]
    flux : np.ndarray  [relative flux]
    best_signal : dict  output from identify.run_bls() — must contain keys:
                        period, t0, duration, depth, power
    n_fap_trials : int  number of bootstrap FAP iterations
    rng_seed : int
    target_id : str
    save_plot : bool
    skip_fap : bool     if True, skip the expensive bootstrap (for quick runs)

    Returns
    -------
    dict with merged SNR and FAP results, plus keys:
        significance_summary : human-readable verdict string
    """
    period = float(best_signal["period"])
    t0 = float(best_signal["t0"])
    duration = float(best_signal["duration"])
    depth = float(best_signal["depth"])
    real_power = float(best_signal.get("power", np.nan))

    # --- SNR ---
    snr_result = compute_snr(time, flux, period, t0, duration, depth)

    # --- FAP ---
    if skip_fap or not np.isfinite(real_power):
        logger.warning(
            "Skipping bootstrap FAP (skip_fap=%s, real_power=%s).",
            skip_fap, real_power,
        )
        fap_result: Dict = {
            "fap": np.nan,
            "n_trials": 0,
            "n_exceeding": np.nan,
            "real_power": real_power,
            "trial_powers": np.array([]),
            "elapsed_s": 0.0,
        }
    else:
        fap_result = compute_fap_bootstrap(
            time, flux,
            period=period, t0=t0, duration=duration,
            real_power=real_power,
            n_trials=n_fap_trials,
            rng_seed=rng_seed,
            target_id=target_id,
            save_plot=save_plot,
        )

    # --- Assemble output ---
    out: Dict = {**snr_result, **fap_result}

    snr = snr_result["snr"]
    fap = fap_result["fap"]

    # Simple human-readable significance verdict
    if np.isfinite(snr) and snr >= 7.1:          # Kepler detection threshold
        snr_verdict = "STRONG (SNR ≥ 7.1)"
    elif np.isfinite(snr) and snr >= 5.0:
        snr_verdict = "MODERATE (5 ≤ SNR < 7.1)"
    elif np.isfinite(snr):
        snr_verdict = f"WEAK (SNR={snr:.2f} < 5)"
    else:
        snr_verdict = "UNDEFINED"

    if np.isfinite(fap):
        fap_verdict = f"FAP={fap:.4f}" + (" (significant)" if fap < 0.01 else " (marginal)" if fap < 0.05 else " (not significant)")
    else:
        fap_verdict = "FAP not computed"

    out["significance_summary"] = f"{snr_verdict} | {fap_verdict}"
    logger.info("Significance: %s", out["significance_summary"])

    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 8 — Statistical Significance")
    p.add_argument("--target", default="KIC 11904151", help="Target ID")
    p.add_argument("--mission", default="Kepler", choices=["Kepler", "TESS"])
    p.add_argument("--n-fap-trials", type=int, default=200,
                   help="Bootstrap iterations (use 1000 for publication-quality)")
    p.add_argument("--skip-fap", action="store_true",
                   help="Skip the bootstrap FAP (fast SNR-only mode)")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()

    # Import pipeline components
    from data_loader import download_lightcurve, preprocess
    from detrend import run_detrending
    from identify import (
        build_period_grid, build_duration_grid,
        run_bls, bin_lc_for_bls,
    )

    print("\n" + "=" * 65)
    print(f"  PHASE 8 — Statistical Significance: {args.target}")
    print("=" * 65 + "\n")

    # Phases 2-5 (data → period search)
    print("[Phase 2-3] Loading data …")
    lc = download_lightcurve(args.target, mission=args.mission)
    time_arr, flux_arr, flux_err_arr = preprocess(lc)
    cadence_min = float(np.median(np.diff(time_arr)) * 24 * 60)
    baseline = float(time_arr[-1] - time_arr[0])

    print("[Phase 4] Detrending …")
    _, detrended, _ = run_detrending(
        time=time_arr, flux=flux_arr,
        period_max_days=baseline / 3.0,
        cadence_minutes=cadence_min,
        method="savgol",
        target_id=args.target,
        save_plot=False,
    )

    print("[Phase 5] BLS period search …")
    time_bls, det_bls, err_bls = bin_lc_for_bls(time_arr, detrended, flux_err_arr)
    period_grid = build_period_grid(baseline)
    duration_grid = build_duration_grid()
    bls_result, best_signal = run_bls(time_bls, det_bls, err_bls, period_grid, duration_grid)

    print(f"  Best period: {best_signal['period']:.5f} d | power: {best_signal['power']:.3f}\n")

    # Phase 8
    print(f"[Phase 8] Significance (FAP trials={args.n_fap_trials}) …")
    sig = run_significance(
        time=time_arr,
        flux=detrended,
        best_signal=best_signal,
        n_fap_trials=args.n_fap_trials,
        target_id=args.target,
        save_plot=not args.no_plot,
        skip_fap=args.skip_fap,
    )

    print("\n── Phase 8 Results ──────────────────────────────────────")
    print(f"  SNR                : {sig['snr']:.3f}")
    print(f"  N in transit       : {sig['n_in_transit']}")
    print(f"  sigma_oot          : {sig['sigma_oot']:.4e}")
    print(f"  noise_on_transit   : {sig['noise_on_transit']:.4e}")
    print(f"  FAP                : {sig.get('fap', float('nan')):.4f}")
    print(f"  FAP trials done    : {sig.get('n_trials', 0)}")
    print(f"  Trials exceeding   : {sig.get('n_exceeding', 'N/A')}")
    print(f"\n  ► {sig['significance_summary']}")
    print("=" * 65 + "\n")
