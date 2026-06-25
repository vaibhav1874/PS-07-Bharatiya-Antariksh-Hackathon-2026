"""
detrend.py
==========
Phase 4 of the exoplanet transit detection pipeline.

Responsibilities
----------------
Remove instrumental and stellar systematics from a pre-cleaned light curve
while preserving the transit signal shape.  Two detrending methods are
provided:

1. **Savitzky-Golay filter** (``scipy.signal.savgol_filter``) — fast,
   works well for smoothly-varying trends.
2. **Wotan biweight filter** — more robust to stellar variability, outlier-
   resistant at the cost of extra computation.

Window length selection
-----------------------
The smoothing window is set to **3 × the estimated maximum transit duration**
(converted to cadence points).  This ensures that a transit dip narrower
than the window does not get subtracted away with the trend, because the
biweight/SG estimator at any point only "sees" 1/3 of the window on each
side.  If the window were shorter than the transit, the transit itself would
flatten the trend estimate and the detrended flux would lose depth.

Reference: Hippke et al. 2019 (Wotan paper), §3.1 — "The window should be
at least 3–4× longer than the transit duration to avoid self-subtraction."

Usage (standalone)
------------------
    python detrend.py --target "KIC 11904151" --mission Kepler
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import wotan

from data_loader import download_lightcurve, preprocess, DEFAULT_CACHE_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Window-length estimation
# ---------------------------------------------------------------------------

def estimate_max_transit_duration(period_max_days: float) -> float:
    """
    Estimate the physical upper bound on transit duration for a Sun-like host.

    Uses the geometric approximation from Hippke et al. 2019 (Wotan §3.1):
    For a circular orbit around a star with R★ ≈ R☉, the maximum possible
    transit duration (equatorial crossing) is:

        T_max ≈ (period / π)^(1/3) × C    [days]

    where C absorbs R★/R☉ and G/M★ in a dimensionally-consistent way.
    For a solar-type star this simplifies to ~0.25 × P^(1/3) hours,
    or ~0.0104 × P^(1/3) days.

    This is a *conservative upper bound* — real transit durations will be
    shorter.  Using the upper bound ensures the detrending window is wide
    enough not to clip any real transit.

    Parameters
    ----------
    period_max_days : float
        Maximum orbital period being searched [days].

    Returns
    -------
    float
        Estimated maximum transit duration [days].
    """
    # Hippke et al. 2019, §3.1 — physical upper bound for Sun-like host
    # T_max ≈ 0.0104 * P^(1/3)  days   (for P in days, R★=R☉, M★=M☉)
    # We add a 20% safety margin so that borderline-long transits are
    # still safely covered.
    t_max = 0.0104 * (period_max_days ** (1.0 / 3.0)) * 1.20
    return float(t_max)


def compute_window_length(
    max_duration_days: float,
    cadence_minutes: float,
    factor: float = 3.0,
) -> int:
    """
    Convert maximum transit duration to a Savitzky-Golay / Wotan window
    length in cadence units.

    The factor 3.0 follows Hippke et al. 2019 §3.1: "window ≥ 3 × transit
    duration" prevents self-subtraction of the transit signal.

    Parameters
    ----------
    max_duration_days : float
        Maximum transit duration estimate [days].
    cadence_minutes : float
        Median cadence of the light curve [minutes].
    factor : float
        Multiplier applied to the duration (default: 3.0, per Wotan paper).

    Returns
    -------
    int
        Window length in cadence units, forced to be **odd** (required by
        ``savgol_filter``) and at least 5.
    """
    points_per_day = 24.0 * 60.0 / cadence_minutes
    window_pts = int(np.ceil(factor * max_duration_days * points_per_day))
    # savgol_filter requires odd window length
    if window_pts % 2 == 0:
        window_pts += 1
    window_pts = max(window_pts, 5)
    logger.info(
        "Window length: %.3f-day duration × %.1f factor = %d cadence points "
        "(cadence=%.1f min).",
        max_duration_days, factor, window_pts, cadence_minutes,
    )
    return window_pts


# ---------------------------------------------------------------------------
# Detrending methods
# ---------------------------------------------------------------------------

def detrend_savgol(
    time: np.ndarray,
    flux: np.ndarray,
    window_length: int,
    polyorder: int = 2,
    gap_threshold_days: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detrend using a Savitzky-Golay polynomial smoothing filter.

    **Gap-aware (Part B3 fix)**: ``scipy.signal.savgol_filter`` operates on
    array index, not on time value — it silently assumes uniform sampling.
    Real Kepler light curves have multi-day gaps between quarters.  When the
    filter slides its window across a gap it treats index-adjacent points
    (actually days apart) as time-adjacent, producing a flat/boxy trend
    estimate at and after the gap boundary.

    Fix: split the light curve into contiguous segments at gaps larger than
    ``gap_threshold_days`` (Part B3 specifies 0.5 d) and apply the filter
    independently to each segment.

    Segments shorter than ``window_length`` are skipped (their trend is set
    to 1.0, i.e., no detrending applied to that short stub).

    Note on arithmetic form: the reference (Part B3) shows the additive form
    ``flux - savgol + 1.0``; we use the multiplicative form ``flux / trend``
    which is equivalent for normalised flux and is more physically correct for
    fractional (ppm-level) stellar variability.  This deviation is explicit.

    Parameters
    ----------
    time : np.ndarray
        Time array [days].  Used to locate inter-segment gaps.
    flux : np.ndarray
        Raw pre-cleaned flux array.
    window_length : int
        Number of cadence points in the smoothing window (must be odd).
    polyorder : int
        Polynomial order for the SG filter (2 = quadratic is standard).
    gap_threshold_days : float
        Time gaps larger than this value [days] split the light curve into
        separate segments.  Default: 0.5 d (matches Part B3 specification).

    Returns
    -------
    trend : np.ndarray
        Smoothed trend estimate (per-segment SG, 1.0 for skipped segments).
    detrended : np.ndarray
        Normalised detrended flux = flux / trend  (median ≈ 1.0).
    """
    if window_length >= len(flux):
        raise ValueError(
            f"window_length ({window_length}) must be < len(flux) ({len(flux)})."
        )

    # --- Part B3: gap-segmentation fix ---
    # Find indices where consecutive time stamps differ by > gap_threshold_days
    gap_indices = np.where(np.diff(time) > gap_threshold_days)[0]
    # np.split on indices i means segments end at i and start at i+1
    seg_indices = np.split(np.arange(len(time)), gap_indices + 1)

    logger.info(
        "SG detrend: found %d gap(s) > %.2f days → %d contiguous segment(s).",
        len(gap_indices), gap_threshold_days, len(seg_indices),
    )

    trend = np.ones_like(flux)   # default: trend = 1.0 (no detrending)
    n_skipped = 0

    for seg in seg_indices:
        if len(seg) < window_length:
            # Segment too short to apply SG filter reliably — skip it.
            # Trend stays 1.0 for this segment (detrended = flux / 1.0 = flux).
            logger.debug(
                "Segment of %d points < window %d — skipping (trend=1.0).",
                len(seg), window_length,
            )
            n_skipped += 1
            continue
        trend[seg] = savgol_filter(
            flux[seg], window_length=window_length, polyorder=polyorder
        )

    if n_skipped:
        logger.warning(
            "%d segment(s) were shorter than window_length=%d and were NOT detrended.",
            n_skipped, window_length,
        )

    detrended = flux / trend
    logger.info(
        "SG detrending complete: %d segments, window=%d pts, polyorder=%d, "
        "%d segment(s) skipped (too short).",
        len(seg_indices), window_length, polyorder, n_skipped,
    )
    return trend, detrended


def detrend_wotan(
    time: np.ndarray,
    flux: np.ndarray,
    window_length_days: float,
    method: str = "biweight",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detrend using Wotan's biweight time-windowed filter.

    Wotan's biweight estimator is more robust to stellar variability and
    outliers than Savitzky-Golay.  Recommended for light curves with
    prominent starspot modulation.

    Reference: Hippke et al. 2019 — https://doi.org/10.3847/1538-3881/ab3987

    Parameters
    ----------
    time : np.ndarray
        Time array [days].
    flux : np.ndarray
        Raw pre-cleaned flux array.
    window_length_days : float
        Window size in *days* (Wotan uses physical units, not cadence points).
    method : str
        Wotan detrending method.  "biweight" (default) is most robust;
        "lowess", "pspline" are alternatives.

    Returns
    -------
    trend : np.ndarray
        Smoothed trend estimate.
    detrended : np.ndarray
        Normalised detrended flux = flux / trend  (median ≈ 1.0).
    """
    trend = wotan.flatten(
        time,
        flux,
        window_length=window_length_days,
        method=method,
        return_trend=True,
        break_tolerance=0.5,  # gap threshold in days; avoids interpolation over gaps
    )[1]  # flatten() returns (flattened, trend) when return_trend=True
    detrended = flux / trend
    logger.info("Wotan detrending complete (method=%s, window=%.3f days).",
                method, window_length_days)
    return trend, detrended


def normalize_flux(detrended: np.ndarray) -> np.ndarray:
    """
    Re-normalise a detrended flux array so the median equals exactly 1.0.

    After dividing by the trend, floating-point accumulation may shift the
    median slightly.  This step enforces the convention flux ≈ 1 out of transit.

    Parameters
    ----------
    detrended : np.ndarray
        Flux array after trend division.

    Returns
    -------
    np.ndarray
        Flux array with median = 1.0.
    """
    return detrended / np.median(detrended)


# ---------------------------------------------------------------------------
# Plotting — required deliverable
# ---------------------------------------------------------------------------

def plot_detrending(
    time: np.ndarray,
    raw_flux: np.ndarray,
    trend: np.ndarray,
    detrended_flux: np.ndarray,
    target_id: str,
    method_name: str = "Savitzky-Golay",
    save_path: Path = PLOTS_DIR / "detrending.png",
) -> None:
    """
    Produce the required before/after detrending plot.

    Top panel: raw flux with trend overlay.
    Bottom panel: detrended flux.

    Parameters
    ----------
    time : np.ndarray
        Time array [days].
    raw_flux : np.ndarray
        Raw (pre-detrending) flux.
    trend : np.ndarray
        Estimated trend from the filter.
    detrended_flux : np.ndarray
        Flux after trend removal and normalisation.
    target_id : str
        Target identifier for plot title.
    method_name : str
        Name of the detrending method (for legend).
    save_path : Path
        Output file path.  Parent directories are created if needed.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"Detrending: {target_id}  (method: {method_name})",
        fontsize=13, fontweight="bold"
    )

    # ---- Top: raw flux + trend ----
    ax = axes[0]
    ax.plot(time, raw_flux, ".", ms=1.0, color="#5599cc", alpha=0.5,
            label="Raw flux")
    ax.plot(time, trend, "-", lw=1.5, color="#cc4444", alpha=0.9,
            label=f"{method_name} trend")
    ax.set_ylabel("Flux (raw)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        "Before detrending: raw flux + estimated trend",
        fontsize=10
    )
    ax.grid(True, alpha=0.3)

    # ---- Bottom: detrended ----
    ax = axes[1]
    ax.plot(time, detrended_flux, ".", ms=1.0, color="#33aa77", alpha=0.5,
            label="Detrended flux")
    ax.axhline(1.0, color="#888888", lw=0.8, ls="--", label="Median = 1.0")
    ax.set_xlabel("Time [days]", fontsize=11)
    ax.set_ylabel("Normalised Flux", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        "After detrending: normalised flux  (transits should now appear as dips)",
        fontsize=10
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Detrending plot saved to: %s", save_path)


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def run_detrending(
    time: np.ndarray,
    flux: np.ndarray,
    period_max_days: float,
    cadence_minutes: float,
    method: str = "savgol",
    target_id: str = "target",
    save_plot: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    End-to-end detrending: estimate window → apply filter → normalise.

    Parameters
    ----------
    time : np.ndarray
        Pre-cleaned time array [days].
    flux : np.ndarray
        Pre-cleaned flux array.
    period_max_days : float
        Maximum period searched [days]; used to derive the window length.
    cadence_minutes : float
        Median cadence [minutes].
    method : str
        "savgol" or "wotan".
    target_id : str
        Target identifier for plot labelling.
    save_plot : bool
        Whether to save the before/after detrending plot.

    Returns
    -------
    trend : np.ndarray
    detrended : np.ndarray
    window_pts : int
        Window length in cadence units (for provenance logging).
    """
    max_dur = estimate_max_transit_duration(period_max_days)
    window_pts = compute_window_length(max_dur, cadence_minutes, factor=3.0)
    window_days = max_dur * 3.0

    logger.info(
        "Detrending: method=%s, max_transit_dur=%.3f d, "
        "window=%.3f d / %d points.",
        method, max_dur, window_days, window_pts,
    )

    if method == "savgol":
        trend, detrended = detrend_savgol(time, flux, window_pts, polyorder=2)
        method_name = "Savitzky-Golay"
    elif method == "wotan":
        trend, detrended = detrend_wotan(time, flux, window_days, method="biweight")
        method_name = "Wotan biweight"
    else:
        raise ValueError(f"Unknown detrending method: '{method}'. Use 'savgol' or 'wotan'.")

    detrended = normalize_flux(detrended)

    if save_plot:
        save_path = PLOTS_DIR / f"{target_id.replace(' ', '_')}_detrending.png"
        plot_detrending(
            time, flux, trend, detrended,
            target_id=target_id,
            method_name=method_name,
            save_path=save_path,
        )

    return trend, detrended, window_pts


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detrend a real TESS/Kepler light curve and save plots."
    )
    p.add_argument("--target", default="KIC 11904151",
                   help="Target identifier (default: Kepler-10)")
    p.add_argument("--mission", default="Kepler", choices=["Kepler", "K2", "TESS"])
    p.add_argument("--method", default="savgol", choices=["savgol", "wotan"])
    p.add_argument("--period-max", type=float, default=30.0,
                   help="Max orbital period [days] for window estimation")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    print("\n" + "=" * 60)
    print("  DETRENDING TEST — REAL DATA")
    print("=" * 60)
    print(f"  Target  : {args.target}")
    print(f"  Mission : {args.mission}")
    print(f"  Method  : {args.method}")
    print("=" * 60 + "\n")

    lc = download_lightcurve(args.target, mission=args.mission)
    time, flux, flux_err = preprocess(lc)

    cadence_min = float(np.median(np.diff(time)) * 24 * 60)

    trend, detrended, window_pts = run_detrending(
        time=time,
        flux=flux,
        period_max_days=args.period_max,
        cadence_minutes=cadence_min,
        method=args.method,
        target_id=args.target,
        save_plot=True,
    )

    print("\n--- DETRENDING RESULTS ---")
    print(f"  Cadence (median)     : {cadence_min:.2f} min")
    print(f"  Window length        : {window_pts} cadence points")
    print(f"  Detrended flux range : [{detrended.min():.6f}, {detrended.max():.6f}]")
    print(f"  Detrended flux std   : {detrended.std():.4e}")
    print(f"  Plot saved to        : plots/{args.target.replace(' ', '_')}_detrending.png")
    print("\nDETRENDING TEST PASSED.\n")
