"""
identify.py
===========
Phase 5 of the exoplanet transit detection pipeline.

Responsibilities
----------------
Period search via two complementary algorithms:

1. **Box Least Squares (BLS)** — ``astropy.timeseries.BoxLeastSquares``.
   Primary period-search method.  Searches a period × duration grid and
   returns the power spectrum (BLS periodogram).

2. **Transit Least Squares (TLS)** — ``transitleastsquares``.
   More sensitive cross-check; uses a physical transit model instead of
   a box, which gives better signal recovery for shallow transits.

Period-grid design
------------------
- Minimum period: 0.5 days (sub-day periods are rare and require very fast
  cadence to characterise; below 0.5 d, BLS aliases become a problem).
- Maximum period: ``baseline / 3`` — ensures at least 3 complete transit
  epochs are visible in the data, which is the minimum needed for a robust
  period detection.  This is the "Nyquist-aware" choice: with fewer than 3
  transits it is impossible to distinguish a periodic signal from a single
  or double event.

Usage (standalone)
------------------
    python identify.py --target "KIC 11904151" --mission Kepler
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
from astropy.timeseries import BoxLeastSquares
import astropy.units as u

from data_loader import download_lightcurve, preprocess, DEFAULT_CACHE_DIR
from detrend import run_detrending

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Known published values for honest comparison (from NASA Exoplanet Archive)
KNOWN_PARAMS: Dict[str, Dict[str, float]] = {
    "KIC 11904151": {   # Kepler-10 b
        "period_days": 0.83749070,
        "depth_ppm": 152.0,        # ~0.015% depth
        "duration_hours": 1.811,
    },
    "KIC 6922244": {    # Kepler-8 b
        "period_days": 3.52254,
        "depth_ppm": 9392.0,
        "duration_hours": 3.3,
    },
}


# ---------------------------------------------------------------------------
# Optional: bin short-cadence data before BLS
# ---------------------------------------------------------------------------

def bin_lc_for_bls(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    target_cadence_min: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin a light curve to a coarser cadence before running BLS.

    On short-cadence (1-min) Kepler data, 571 K points × 12 K periods × 50
    durations would take ~10-30 minutes on a single core.  Binning to 30-min
    cadence reduces the array to ~14 K points and the BLS runtime to < 1 min,
    while preserving signal: Kepler-10b's 1.8-h transit still has ~3-4 bins
    inside it — enough for BLS to detect the period reliably.

    **Gap-aware**: Bins are placed on an absolute time grid anchored at
    ``time[0]``.  This means that quarter-gaps in Kepler data (which can be
    ~90 days long) are naturally preserved — no point is ever binned with a
    point on the other side of a gap.  This avoids the artifact where simple
    array-reshape binning creates mixed-gap bins that look like long-period
    signals to BLS.

    Uncertainty propagation: binned flux_err = sqrt(sum(err²)) / N,
    consistent with i.i.d. Gaussian noise.

    Parameters
    ----------
    time : np.ndarray
        Time array [days].
    flux : np.ndarray
        Detrended normalised flux.
    flux_err : np.ndarray
        Per-point uncertainties.
    target_cadence_min : float
        Desired cadence of the output [minutes].  Default: 30 min.

    Returns
    -------
    time_b, flux_b, flux_err_b : np.ndarray
        Binned arrays.  If input cadence >= target_cadence_min, returns
        the original arrays unchanged.
    """
    current_cadence_min = float(np.median(np.diff(time))) * 24 * 60
    if current_cadence_min >= target_cadence_min * 0.9:
        logger.info(
            "bin_lc_for_bls: cadence=%.1f min already >= target=%.1f min, no binning.",
            current_cadence_min, target_cadence_min,
        )
        return time, flux, flux_err

    target_cadence_days = target_cadence_min / (24.0 * 60.0)

    # Place bins on an absolute time grid anchored at time[0].
    # searchsorted assigns each point to a bin by its absolute timestamp —
    # Kepler quarter-gaps fall between bins naturally; no cross-gap mixing.
    bin_edges = np.arange(time[0], time[-1] + target_cadence_days, target_cadence_days)
    bin_idx = np.searchsorted(bin_edges, time, side="right") - 1

    t_b, f_b, fe_b = [], [], []
    for i in np.unique(bin_idx):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        t_b.append(time[mask].mean())
        f_b.append(flux[mask].mean())
        sigma_mean = np.sqrt(np.sum(flux_err[mask] ** 2)) / mask.sum()
        fe_b.append(sigma_mean)

    t_b = np.array(t_b)
    f_b = np.array(f_b)
    fe_b = np.array(fe_b)

    logger.info(
        "bin_lc_for_bls: %.1f-min cadence → %.1f-min cadence "
        "(%d → %d points, gap-aware absolute-grid binning).",
        current_cadence_min, target_cadence_min,
        len(time), len(t_b),
    )
    return t_b, f_b, fe_b


# ---------------------------------------------------------------------------
# Period & duration grids
# ---------------------------------------------------------------------------

def build_period_grid(
    baseline_days: float,
    min_period: float = 0.5,
    max_fraction: float = 1.0 / 5.0,
    n_per_decade: int = 5000,
) -> np.ndarray:
    """
    Build a logarithmically-spaced period grid.

    Maximum period = ``baseline_days * max_fraction``.

    **Part B4 / Part A1 fix:** default changed from 1/3 to 1/5.
    Reference Part A1 states ``baseline/3`` is the bare *minimum* (risky —
    BLS has only 3 transit instances to constrain the box, enabling
    false high power from small-sample overfitting).  ``baseline/5`` is the
    safer choice and reduces the rising noise floor at long periods seen in
    the periodogram.  For KIC 11904151 (period=0.837 d), this does not
    affect detection: the true period is well within even the ``/10`` limit.

    Parameters
    ----------
    baseline_days : float
        Total observing baseline [days] = time[-1] - time[0].
    min_period : float
        Minimum period to search [days].
    max_fraction : float
        max_period = baseline × this fraction.
        ``1/5`` (default) = safer per Part B4/A1 reference recommendation.
        ``1/3`` = bare minimum (more susceptible to edge-effect false peaks).
    n_per_decade : int
        Number of grid points per decade (controls resolution).

    Returns
    -------
    np.ndarray
        1-D period grid [days].
    """
    max_period = baseline_days * max_fraction
    if max_period <= min_period:
        raise ValueError(
            f"Baseline ({baseline_days:.1f} d) is too short to search "
            f"periods > {min_period} d with 3-transit requirement."
        )
    n_periods = max(
        500,
        int(np.log10(max_period / min_period) * n_per_decade),
    )
    grid = np.geomspace(min_period, max_period, num=n_periods)
    logger.info(
        "Period grid: %.2f – %.2f days  (%d points, log-spaced).",
        grid[0], grid[-1], len(grid),
    )
    return grid


def build_duration_grid(
    min_dur: float = 0.01,
    max_dur: float = 0.3,
    n_points: int = 50,
) -> np.ndarray:
    """
    Build a linearly-spaced transit duration grid.

    Parameters
    ----------
    min_dur : float
        Minimum duration [days].  0.01 d = 14.4 min (typical short ingress).
    max_dur : float
        Maximum duration [days].  0.3 d = 7.2 h (long-period, large-star transits).
    n_points : int
        Number of grid points.

    Returns
    -------
    np.ndarray
        1-D duration grid [days].
    """
    grid = np.linspace(min_dur, max_dur, n_points)
    logger.info(
        "Duration grid: %.2f – %.2f days  (%d points).",
        grid[0], grid[-1], len(grid),
    )
    return grid


# ---------------------------------------------------------------------------
# BLS period search
# ---------------------------------------------------------------------------

def run_bls(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period_grid: np.ndarray,
    duration_grid: np.ndarray,
) -> Tuple[object, Dict]:
    """
    Run Box Least Squares on the detrended light curve.

    Parameters
    ----------
    time : np.ndarray
        Time array [days].
    flux : np.ndarray
        Detrended, normalised flux (median ≈ 1.0).
    flux_err : np.ndarray
        Per-point flux uncertainties.
    period_grid : np.ndarray
        Candidate periods to search [days].
    duration_grid : np.ndarray
        Candidate transit durations to search [days].

    Returns
    -------
    bls_result : astropy BLSResults object
        Full result including periodogram power array.
    best_signal : dict
        Keys: period, t0, duration, depth, power (all floats).
    """
    # Convert detrended flux so that transits appear as positive "events"
    # BLS works on (1 - flux) so that dips become peaks
    flux_for_bls = 1.0 - flux   # transits are now positive bumps

    bls = BoxLeastSquares(
        time * u.day,
        flux_for_bls,
        dy=flux_err,
    )

    logger.info("Running BLS: %d periods × %d durations ...",
                len(period_grid), len(duration_grid))

    bls_result = bls.power(
        period=period_grid * u.day,
        duration=duration_grid * u.day,
        method="fast",      # vectorised C implementation
        objective="snr",    # maximise transit SNR rather than likelihood
    )

    # Extract best signal
    best_idx = np.argmax(bls_result.power)
    best_period = float(bls_result.period[best_idx].to(u.day).value)
    best_t0 = float(bls_result.transit_time[best_idx].to(u.day).value)
    best_dur = float(bls_result.duration[best_idx].to(u.day).value)
    best_depth = float(bls_result.depth[best_idx])
    best_power = float(bls_result.power[best_idx])

    best_signal = {
        "period": best_period,
        "t0": best_t0,
        "duration": best_dur,
        "depth": best_depth,
        "power": best_power,
    }

    logger.info(
        "BLS best signal: period=%.5f d, t0=%.4f, duration=%.4f d, "
        "depth=%.4f, power=%.2f",
        best_period, best_t0, best_dur, best_depth, best_power,
    )
    return bls_result, best_signal


def run_tls(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
) -> Dict:
    """
    Run Transit Least Squares (TLS) as a cross-check on the BLS result.

    TLS uses a physical Mandel-Agol transit shape instead of a box, giving
    better sensitivity especially for shallow transits.  Results should be
    broadly consistent with BLS; large disagreement may indicate a
    non-transit signal.

    Reference: Hippke & Heller 2019 — https://doi.org/10.1051/0004-6361/201834672

    Parameters
    ----------
    time : np.ndarray
        Time [days].
    flux : np.ndarray
        Detrended normalised flux.
    flux_err : np.ndarray
        Per-point uncertainties.

    Returns
    -------
    dict
        TLS result with keys: period, t0, duration, depth, SDE (Signal
        Detection Efficiency, the TLS analogue of BLS power).
    """
    try:
        from transitleastsquares import transitleastsquares, cleaned_array
    except ImportError:
        logger.warning(
            "transitleastsquares not installed — skipping TLS cross-check."
        )
        return {"error": "transitleastsquares not available"}

    # TLS requires evenly-spaced time with no large gaps.
    # cleaned_array removes NaN/inf and re-orders.
    t_c, f_c, fe_c = cleaned_array(time, flux, flux_err)

    logger.info("Running TLS cross-check ...")
    model = transitleastsquares(t_c, f_c, fe_c)

    # TLS auto-detects cadence and sets period range internally.
    # We override to match our BLS grid limits.
    baseline = t_c[-1] - t_c[0]
    results = model.power(
        period_min=0.5,
        period_max=baseline / 5.0,   # Part B4/A1: safer cap (was /3 bare minimum)
        show_progress_bar=False,
        use_threads=1,  # deterministic
    )

    tls_result = {
        "period": float(results.period),
        "t0": float(results.T0),
        "duration": float(results.duration),
        "depth": float(1.0 - results.depth_mean_odd),   # transit depth
        "SDE": float(results.SDE),
        "FAP": float(results.FAP) if hasattr(results, "FAP") else None,
    }
    logger.info(
        "TLS result: period=%.5f d, SDE=%.2f",
        tls_result["period"], tls_result["SDE"],
    )
    return tls_result


# ---------------------------------------------------------------------------
# Phase-folding
# ---------------------------------------------------------------------------

def phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Phase-fold the light curve at a given period and epoch.

    The transit is centred at phase = 0.

    Parameters
    ----------
    time : np.ndarray
        Time [days].
    flux : np.ndarray
        Detrended, normalised flux.
    period : float
        Orbital period [days].
    t0 : float
        Transit mid-time epoch [days].

    Returns
    -------
    phase : np.ndarray
        Phase values in [−0.5, +0.5].
    flux_folded : np.ndarray
        Flux values ordered by phase.
    """
    phase_raw = ((time - t0) % period) / period   # [0, 1)
    # Centre on transit at phase = 0: shift values > 0.5 by −1
    phase = np.where(phase_raw > 0.5, phase_raw - 1.0, phase_raw)
    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_periodogram(
    bls_result: object,
    best_period: float,
    target_id: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot the BLS periodogram with the best period clearly marked.

    Parameters
    ----------
    bls_result : astropy BLSResults
        Full BLS output.
    best_period : float
        Best-fit period [days] to mark.
    target_id : str
        Target name for plot title.
    save_path : Path, optional
        If given, save the figure to this path.
    """
    periods = bls_result.period.to(u.day).value
    power = bls_result.power

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(periods, power, "-", lw=0.7, color="#5566ee", alpha=0.8,
            label="BLS power")
    ax.axvline(best_period, color="#ee4444", lw=2, ls="--",
               label=f"Best period: {best_period:.5f} d")

    # Mark harmonics 2x and 0.5x
    for k, ls, col in [(2, ":", "#ff9900"), (0.5, ":", "#ff9900")]:
        ax.axvline(best_period * k, color=col, lw=1, ls=ls, alpha=0.6,
                   label=f"{k}× period" if k == 2 else f"½ period")

    ax.set_xlabel("Period [days]", fontsize=11)
    ax.set_ylabel("BLS Power (SNR objective)", fontsize=11)
    ax.set_title(f"BLS Periodogram — {target_id}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Periodogram plot saved to: %s", save_path)
    plt.close(fig)


def plot_phase_fold(
    phase: np.ndarray,
    flux: np.ndarray,
    best_signal: Dict,
    target_id: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot the phase-folded light curve with the transit centred at phase = 0.

    Parameters
    ----------
    phase : np.ndarray
        Phase array in [−0.5, +0.5].
    flux : np.ndarray
        Flux values sorted by phase.
    best_signal : dict
        BLS best-signal dict (period, depth, duration, …).
    target_id : str
        Target name for title.
    save_path : Path, optional
        Save destination.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(phase, flux, ".", ms=1.5, color="#5599cc", alpha=0.5,
            label="Phase-folded flux")
    ax.axhline(1.0, color="#888888", lw=0.8, ls="--")

    # Mark transit duration
    half_dur = best_signal["duration"] / (2 * best_signal["period"])
    ax.axvspan(-half_dur, half_dur, color="#ee4444", alpha=0.12,
               label=f"Transit duration ({best_signal['duration'] * 24:.2f} h)")
    ax.axhline(1.0 - best_signal["depth"], color="#ee4444", lw=1, ls=":",
               label=f"Transit depth = {best_signal['depth'] * 1e6:.0f} ppm")

    ax.set_xlabel("Phase", fontsize=11)
    ax.set_ylabel("Normalised Flux", fontsize=11)
    ax.set_title(
        f"Phase-folded Light Curve — {target_id}\n"
        f"Period = {best_signal['period']:.5f} d",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.set_xlim(-0.5, 0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Phase-fold plot saved to: %s", save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Honest period-recovery comparison
# ---------------------------------------------------------------------------

def compare_to_known(
    recovered: Dict,
    target_id: str,
    known_db: Dict = KNOWN_PARAMS,
) -> Optional[Dict]:
    """
    Compare recovered BLS signal parameters against published NASA Exoplanet
    Archive values.  Prints percent error honestly — does NOT hide large errors.

    Parameters
    ----------
    recovered : dict
        BLS best_signal dict (period, depth, duration).
    target_id : str
        Target identifier; used as key in ``known_db``.
    known_db : dict
        Mapping from target_id → known parameter dict.

    Returns
    -------
    dict or None
        Comparison dict with percent errors, or None if target not in known_db.
    """
    if target_id not in known_db:
        logger.info("No published parameters available for '%s'.", target_id)
        return None

    known = known_db[target_id]
    rec_period = recovered["period"]
    rec_depth_ppm = recovered["depth"] * 1e6
    rec_duration_h = recovered["duration"] * 24.0

    period_err_pct = abs(rec_period - known["period_days"]) / known["period_days"] * 100
    depth_err_pct = abs(rec_depth_ppm - known["depth_ppm"]) / known["depth_ppm"] * 100
    dur_err_pct = abs(rec_duration_h - known["duration_hours"]) / known["duration_hours"] * 100

    comparison = {
        "period_recovered": rec_period,
        "period_known": known["period_days"],
        "period_error_pct": period_err_pct,
        "depth_ppm_recovered": rec_depth_ppm,
        "depth_ppm_known": known["depth_ppm"],
        "depth_error_pct": depth_err_pct,
        "duration_h_recovered": rec_duration_h,
        "duration_h_known": known["duration_hours"],
        "duration_error_pct": dur_err_pct,
    }

    print("\n" + "=" * 60)
    print("  PARAMETER RECOVERY vs. NASA EXOPLANET ARCHIVE")
    print(f"  Target: {target_id}")
    print("=" * 60)
    print(f"  Period   : recovered={rec_period:.5f} d  |  known={known['period_days']:.5f} d  "
          f"|  error={period_err_pct:.2f}%")
    print(f"  Depth    : recovered={rec_depth_ppm:.1f} ppm  |  known={known['depth_ppm']:.1f} ppm  "
          f"|  error={depth_err_pct:.2f}%")
    print(f"  Duration : recovered={rec_duration_h:.3f} h  |  known={known['duration_hours']:.3f} h  "
          f"|  error={dur_err_pct:.2f}%")

    # Honest commentary — per the prompt requirement, we do not suppress bad results
    if period_err_pct > 5.0:
        print(f"\n  *** WARNING: Period error ({period_err_pct:.1f}%) exceeds 5%. ***")
        print("      Possible causes: BLS picks an alias (2x or 0.5x true period),")
        print("      or the available sectors do not cover enough baseline.")
    if depth_err_pct > 30.0:
        print(f"\n  *** NOTE: Depth error ({depth_err_pct:.1f}%) is large. ***")
        print("      BLS depth is a box-model estimate; batman fit (Phase 6)")
        print("      will give a more accurate depth with proper uncertainty.")
    if dur_err_pct > 30.0:
        print(f"\n  *** NOTE: Duration error ({dur_err_pct:.1f}%) is large. ***")
        print("      BLS duration grid resolution may be coarser than actual duration.")
    print("=" * 60 + "\n")

    return comparison


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

def run_identification(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    baseline_days: float,
    target_id: str = "target",
    run_tls_crosscheck: bool = True,
    save_plots: bool = True,
) -> Tuple[Dict, Optional[Dict], Optional[Dict]]:
    """
    Full Phase 5 period search: build grids → BLS → (optional) TLS → fold → plots.

    Parameters
    ----------
    time, flux, flux_err : np.ndarray
        Detrended clean arrays.
    baseline_days : float
        Total observing baseline [days].
    target_id : str
        Target label for plots and comparison.
    run_tls_crosscheck : bool
        Whether to run TLS in addition to BLS.
    save_plots : bool
        Save periodogram and phase-fold plots.

    Returns
    -------
    best_signal : dict
        BLS best signal (period, t0, duration, depth, power).
    tls_result : dict or None
        TLS cross-check result (or None if skipped/failed).
    comparison : dict or None
        Recovery comparison vs published values (or None if unknown target).
    """
    period_grid = build_period_grid(baseline_days)
    duration_grid = build_duration_grid()

    bls_result, best_signal = run_bls(
        time, flux, flux_err, period_grid, duration_grid
    )

    tls_result = None
    if run_tls_crosscheck:
        tls_result = run_tls(time, flux, flux_err)

    phase, flux_folded = phase_fold(
        time, flux, best_signal["period"], best_signal["t0"]
    )
    best_signal["phase"] = phase
    best_signal["flux_folded"] = flux_folded

    if save_plots:
        tag = target_id.replace(" ", "_")
        plot_periodogram(
            bls_result, best_signal["period"], target_id,
            save_path=PLOTS_DIR / f"{tag}_periodogram.png",
        )
        plot_phase_fold(
            phase, flux_folded, best_signal, target_id,
            save_path=PLOTS_DIR / f"{tag}_phasefold.png",
        )

    comparison = compare_to_known(best_signal, target_id)

    return best_signal, tls_result, comparison


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Period search (BLS + TLS) on a real TESS/Kepler light curve."
    )
    p.add_argument("--target", default="KIC 11904151",
                   help="Target identifier")
    p.add_argument("--mission", default="Kepler", choices=["Kepler", "K2", "TESS"])
    p.add_argument("--no-tls", action="store_true",
                   help="Skip TLS cross-check (faster)")
    p.add_argument("--period-max-frac", type=float, default=1.0 / 3.0,
                   help="Max period = baseline × this fraction (default: 1/3)")
    p.add_argument("--bin-cadence", type=float, default=30.0,
                   help="Bin short-cadence data to this cadence [min] before BLS (default: 30)")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    print("\n" + "=" * 60)
    print("  PERIOD SEARCH TEST — REAL DATA")
    print("=" * 60)
    print(f"  Target  : {args.target}")
    print(f"  Mission : {args.mission}")
    print("=" * 60 + "\n")

    lc = download_lightcurve(args.target, mission=args.mission)
    time, flux, flux_err = preprocess(lc)

    cadence_min = float(np.median(np.diff(time)) * 24 * 60)
    baseline = float(time[-1] - time[0])

    _, detrended, _ = run_detrending(
        time=time,
        flux=flux,
        period_max_days=baseline * args.period_max_frac,
        cadence_minutes=cadence_min,
        method="savgol",
        target_id=args.target,
        save_plot=True,
    )

    # Bin to 30-min cadence for BLS speed (preserves transit signal for T > 1 h)
    time_bls, detrended_bls, flux_err_bls = bin_lc_for_bls(
        time, detrended, flux_err,
        target_cadence_min=args.bin_cadence,
    )

    best_signal, tls_result, comparison = run_identification(
        time=time_bls,
        flux=detrended_bls,
        flux_err=flux_err_bls,
        baseline_days=baseline,
        target_id=args.target,
        run_tls_crosscheck=not args.no_tls,
        save_plots=True,
    )

    print("\n--- BLS BEST SIGNAL ---")
    for k, v in best_signal.items():
        if k not in ("phase", "flux_folded"):
            print(f"  {k:<12}: {v}")

    if tls_result and "error" not in tls_result:
        print("\n--- TLS CROSS-CHECK ---")
        for k, v in tls_result.items():
            print(f"  {k:<12}: {v}")
        if abs(tls_result["period"] - best_signal["period"]) > 0.01 * best_signal["period"]:
            print("\n  *** NOTE: BLS and TLS periods disagree by > 1%. ***")
            print("      Inspect the periodogram for aliases.")

    print("\nPERIOD SEARCH COMPLETE.\n")
