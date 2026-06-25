# PS-07 — Full Technical Reference: Math, Physics, Test Cases & Troubleshooting

> Yeh document teen kaam karta hai: (1) har pipeline-stage ka exact formula/physics
> samjhata hai, (2) batata hai kitna data chahiye aur output kaisa dikhna chahiye,
> (3) tumhare actual screenshots mein jo bugs dikhe (nan depth, boxy detrending,
> rising noise floor) — unka root-cause math se explain karta hai, with fix.

---

## PART A — DATA REQUIREMENTS: Kitne Din Ka Data Chahiye?

### A1. Baseline vs. Detectable Period

Ek period ko reliably detect karne ke liye, tumhe us period ke kam-se-kam
**3 occurrences** apne observation baseline mein dikhni chahiye (statistically
robust detection ke liye 5-10 better hai, kyunki odd-even test aur secondary-eclipse
check ke liye multiple transits chahiye).

```
N_transits_observed = floor(baseline_days / period_days)

Rule of thumb:
  N_transits_observed >= 3   ->  bare minimum (risky, noisy)
  N_transits_observed >= 5   ->  acceptable for vetting
  N_transits_observed >= 10  ->  robust (odd-even split + secondary eclipse reliable)
```

Isी se reverse mein **maximum searchable period** bhi nikalta hai:

```
P_max = baseline_days / 3      (bare minimum)
P_max = baseline_days / 5      (recommended, safer against edge effects)
```

**Kyun?** Agar tum period grid mein P > baseline/3 search karoge, BLS ko sirf
1-2 transits milenge fit karne ke liye — itni freedom ke saath "box" kahin bhi
fit ho sakta hai, jisse **false high power** dikhta hai jo real signal nahi hai
(yeh wahi cheez hai jo tumhare periodogram mein right-side power badhte hue dikhi —
Part B4 mein detail hai).

### A2. Cadence vs. Transit Duration — "Points-in-Transit" Rule

Har transit ke andar kam-se-kam kuch data points hone chahiye, warna fit ka
covariance matrix singular ho jaata hai (uncertainty = nan, jo tumne dekha).

```
points_per_transit (single epoch) = transit_duration_hours / (cadence_minutes / 60)
```

| Mission | Cadence | Example: 1-hour transit |
|---|---|---|
| TESS (short) | 2 min | 30 points/transit — great |
| TESS (FFI) | 10/30 min | 2-6 points/transit — risky alone |
| Kepler (long cadence) | 29.4 min | ~2 points/transit — **needs phase-folding many transits** |
| Kepler (short cadence) | 58.9 sec | ~60 points/transit — great |

**Critical rule:** agar single-transit points kam hain, tumhe **phase-fold karke
saare transits ko combine karna hi padega** before fitting — kabhi bhi ek single
raw transit pe fit mat karo agar cadence coarse hai.

```
total_in_transit_points = points_per_transit x N_transits_observed
Minimum needed for a stable lmfit covariance: ~15-20 total in-transit points
```

### A3. Recommendation Table

| Target period range | Minimum baseline | Best data source |
|---|---|---|
| 0.5 - 3 days (ultra-short) | 1 TESS sector (27d) or 1 Kepler quarter (90d) | Either |
| 3 - 15 days | 2-3 TESS sectors (stitched) or 1 Kepler quarter | Kepler preferred |
| 15 - 60 days | Multiple Kepler quarters (stitched) | Kepler multi-quarter |
| 60+ days | Full Kepler mission (~4 years) | Kepler only — TESS baseline too short unless target is in continuous viewing zone |

### A4. Worked Example — Applying This To KIC 11904151 (Your Actual Target)

From your screenshots: recovered period = **0.83753 days** (~20 hours), and the
detrending plot shows a baseline from ~100 to ~700+ days (multiple Kepler
quarters stitched, with visible gaps).

```
baseline ~= 700 days (rough, from your plot's x-axis)
N_transits_observed ~= 700 / 0.83753 ~= 836 transits   <- huge number, plenty of data
points_per_transit (Kepler LC, 29.4 min, duration 0.96h)
    = 0.96 * 60 / 29.4 ~= 1.96 ~= 2 points/transit (raw, single-epoch)
total_in_transit_points (phase-folded) ~= 2 x 836 ~= 1670 points
```

**Conclusion: data volume is NOT your problem.** 1670 in-transit points after
phase-folding is more than enough for a stable fit. This tells us the
`Depth = 0.0 +/- nan` bug in your screenshot is **not a data-quantity issue —
it's a code/units/import bug** (see Part B5, this is diagnosed in detail there).

---

## PART B — PHASE-BY-PHASE: Math, Physics, Algorithm, Test Case, Known Gaps

### B1. Data Acquisition — Which Flux Column?

Kepler/TESS FITS files mein do flux columns hote hain:
- **SAP_FLUX** (Simple Aperture Photometry) — raw, unprocessed
- **PDCSAP_FLUX** (Pre-search Data Conditioning SAP) — NASA pipeline ne already
  kuch instrumental systematics (thermal drift, pointing jitter) remove kiye hote
  hain, lekin **astrophysical signal (transits) preserve** rehta hai

**Use PDCSAP_FLUX as your starting point**, not SAP_FLUX — tumhara apna
detrending stage isके baad bhi zaroori hai (stellar variability ke liye), lekin
SAP_FLUX se shuruat karne pe tumhe instrumental systematics bhi khud hatani
padengi jo PDCSAP already handle karta hai.

**Test case:**
```
input: lightkurve search result for KIC 11904151
assert: result.PDCSAP_FLUX column exists and is not all-NaN
assert: len(time) == len(flux) == len(flux_err)
assert: np.all(np.diff(time) > 0)   # time strictly increasing
```

### B2. Preprocessing — Quality Flags & Sigma-Clipping

**Quality flags are bitmasks**, not a simple 0/non-zero check. Each bit encodes
a specific issue (cosmic ray hit, reaction-wheel desaturation event, attitude
tweak, etc.). `quality == 0` filtering is too crude — use lightkurve's built-in
presets instead:

```python
lc = lc.remove_nans()
lc = lc[lc.quality == 0]   # OK for a first pass, but prefer:
# lc = search_result.download(quality_bitmask='default')  # smarter bit-aware filter
```

**Sigma-clipping (iterative, robust):**

```
mu     = median(flux)
MAD    = median(|flux - mu|)
sigma  = 1.4826 x MAD            <- robust std estimator, resistant to outliers
mask   = |flux - mu| > k x sigma   (k = 5 typical)
repeat until no new points are clipped, or max_iter reached (e.g. 3 passes)
```

**Why median/MAD instead of mean/std?** A few extreme cosmic-ray outliers can
inflate a normal mean/std estimate, making the clip threshold too loose. Median
and MAD are robust to exactly this.

**Gap / Error this causes if done wrong:** if you sigma-clip BEFORE detrending
using a *non-robust* mean+std, and the transit itself is deep enough, the clip
can accidentally **remove real transit points**, shrinking your measured depth.
Always sigma-clip on a per-segment, robust (median/MAD) basis, and clip
conservatively (k=5, not k=3) at this raw stage.

**Test case:**
```
input: synthetic flux array with 10 known outliers + a 1%-deep injected transit
assert: all 10 outliers removed
assert: >95% of in-transit points survive the clip (transit not eaten)
```

### B3. Detrending — Savitzky-Golay Math + THE GAP BUG (your Image 3)

**Savitzky-Golay filter**: at each point, fit a polynomial (degree 2, typically)
to a sliding window of `window_length` neighboring points, and take the
polynomial's value at the center as the smoothed trend estimate. This is a
local least-squares fit repeated at every point.

**Window length formula:**
```
window_days   = 3 x max_expected_transit_duration_days
window_points = round(window_days / median_cadence_days)
window_points must be odd (savgol_filter requirement)
```
**Why 3x?** If the window is shorter than the transit, the filter "sees" the
dip as part of the local trend and partially fits it away — eating your own
signal. 3x duration is the standard safety margin (Hippke & Heller 2019, the
Wotan paper, recommend a similar multiple).

#### THE BUG IN YOUR SCREENSHOT (Image 3 — boxy/rectangular trend line)

`scipy.signal.savgol_filter` operates on **array index**, not on **time
value**. It silently assumes uniform sampling. Your light curve has **real
time gaps** (visible in the plot around day ~200, ~400-440 — these are gaps
between Kepler quarters, when the spacecraft downlinked data or re-pointed).

When the filter slides its window across a gap, it treats index-adjacent
points as if they were time-adjacent — but they might be **days or weeks
apart**. This produces exactly the artifact you saw: a flat, boxy, wrong trend
estimate right at and after the gap boundary, because the "local window" is
silently spanning a huge real time jump.

**Fix — split into contiguous segments before detrending:**
```python
gap_threshold_days = 0.5   # tune based on your cadence
gaps = np.where(np.diff(time) > gap_threshold_days)[0]
segments = np.split(np.arange(len(time)), gaps + 1)

detrended_flux = np.copy(flux)
for seg in segments:
    if len(seg) < window_points:
        continue   # segment too short to detrend reliably — flag/skip it
    detrended_flux[seg] = flux[seg] - savgol_filter(flux[seg], window_points, 2) + 1.0
```
Alternatively, use the `wotan` package's `flatten()` function, which has gap
handling built in — this is the safer, less error-prone option for a hackathon
timeline.

**Test case:**
```
input: synthetic light curve, two segments separated by a 10-day gap,
       known sinusoidal trend + 0.5%-deep transit injected in each segment
assert: detrended flux has no edge-artifact (no flat/boxy region) within
        3x window_length of the gap boundary
assert: recovered transit depth within 10% of injected depth on both segments
```

### B4. Identification (BLS) — Power Formula, Period Grid + THE RISING-NOISE-FLOOR BUG

**Conceptually**, for each trial (period, duration, phase), BLS computes:
```
depth(P, w) = mean(flux_out_of_box) - mean(flux_in_box)
```
and a normalized "power" (signal-to-noise of that boxcar fit) is computed
across the full period grid; astropy's `objective='snr'` mode (which is what
your plot's y-axis label "BLS Power (SNR objective)" confirms you're using)
reports exactly this signal-to-noise quantity. The period grid value with the
highest power is your best-period candidate.

**Period grid spacing** (so you don't waste compute or miss the true period
between grid points):
```
delta_P  ~=  P^2 / baseline_days        (derived from delta_f ~= 1/baseline, f = 1/P)
```
This means the grid must be **finer at short periods, coarser at long
periods** — most implementations (including astropy's `autoperiod()` helper)
handle this automatically; do not use a linearly-spaced period grid across a
wide range, it under-samples short periods.

**Maximum period** (see Part A1): `P_max = baseline_days / 3` to `/5`.

#### THE BUG IN YOUR SCREENSHOT (Image 2 — power rising toward long periods)

A healthy periodogram has a **flat noise floor** with sharp, isolated spikes
only at real signal periods. Your plot shows power **steadily climbing** from
~5 at short periods to ~25+ near 100 days. Three possible causes, in order of
likelihood given your setup:

1. **Residual systematics from incomplete detrending** (most likely — directly
   caused by the Phase 4 gap-bug above). A poorly detrended long-term trend
   looks, to BLS, like a very-long-period "transit." Fix: resolve B3 first,
   then re-run BLS.
2. **Search period range too close to the baseline.** As P approaches
   baseline/2 or baseline/3, BLS has very few transit instances to constrain
   the box, so spurious high power becomes more likely purely from
   small-sample overfitting. Fix: cap `P_max = baseline/5` instead of a larger
   fraction.
3. **Real correlated ("red") noise** — stellar granulation/rotation produces
   noise that is NOT flat in frequency (more power at low frequencies = long
   periods), which is real astrophysics, not a bug. After fixing #1 and #2, if
   the rise persists, this is the likely explanation, and the fix is a
   pre-whitening step (fit and remove a long-period stellar-rotation signal
   before the planet search) — a known advanced technique, flag it as a
   limitation if you don't have time to implement it.

**Test case:**
```
input: synthetic flux, flat (white) noise only, no injected signal, no trend
assert: BLS power across the full period grid has no point > 5-sigma above
        the grid's median power (i.e., no fake significant peak)
```

### B5. Characterization (Transit Model Fit) — THE NaN DEPTH BUG (your Image 1)

**Physics — Mandel-Agol transit model parameters:**

| Symbol | Meaning |
|---|---|
| Rp/Rs (k) | Planet-to-star radius ratio |
| a/Rs | Scaled semi-major axis |
| i | Orbital inclination |
| b | Impact parameter = (a/Rs) x cos(i) |
| u1, u2 | Quadratic limb-darkening coefficients |
| t0 | Mid-transit time (epoch) |
| P | Orbital period |

**Key formulas (Winn 2010 conventions):**
```
Depth (approx, ignoring limb darkening) = (Rp/Rs)^2

Total duration T14 ~= (P / pi) x arcsin[ (Rs/a) x sqrt((1+k)^2 - b^2) / sin(i) ]
                  ~= (P / pi) x (Rs/a) x sqrt(1 - b^2)          (for a >> Rs, small angle)

Ingress/egress duration ~= T14 x k / sqrt(1 - b^2)

Semi-major axis (Kepler's 3rd law): a^3 = G x M_star x P^2 / (4 x pi^2)
```

#### THE BUG IN YOUR SCREENSHOT (Image 1 — Depth = 0.0 +/- nan ppm, Rp/Rs = 0.0000)

Given your Part A4 calculation shows data volume is NOT the issue, the real
cause is one of these five — check them **in this exact order**:

1. **The "Cannot find module `batman`" error from your Problems panel is
   probably real, not cosmetic.** If `batman` genuinely fails to import inside
   your actual running environment (not just the IDE's type-checker), any
   `try/except` around the fit will silently catch the ImportError and return
   a placeholder `depth=0.0, error=nan` instead of crashing loudly. **Run `pip
   list` inside the exact `.venv` your script uses and confirm `batman-package`
   is installed there.** This is the single most likely cause.

2. **Bad initial guess for Rp/Rs.** If the optimizer starts at, e.g., k=0.001
   when the real signal needs k=0.02, and the gradient of chi-square with
   respect to k is extremely flat near zero, `lmfit`'s Levenberg-Marquardt
   optimizer can get stuck at the starting value, reporting "fit succeeded"
   with depth ~0. **Fix:** initialize `k_guess = sqrt(BLS_depth)`, not an
   arbitrary small constant.

3. **Units mismatch between period/duration and the model's time axis.** If
   your phase-folded x-axis is in days but `batman`'s `params.t0`/`params.per`
   were accidentally set in hours (or vice-versa), the model transit window
   ends up far outside the data's actual phase range — the model evaluates to
   a flat 1.0 across your whole data range, and the best fit naturally
   converges to "no transit" (depth=0). **Fix:** assert units explicitly with
   a comment at every conversion point; add a unit test (below).

4. **Too few in-transit points feeding the covariance matrix** (not your case
   per Part A4, but check this for any future target): if `lmfit`'s parameter
   covariance matrix is singular (rank-deficient) because there isn't enough
   independent information in-transit, depth's point-estimate may compute fine
   but its **uncertainty** comes out as `nan` — matching exactly what you saw.
   Cross-check against #1 and #3 first since your data volume is large.

5. **Limb-darkening coefficient array shape mismatch.** `batman` expects `u`
   as a list matching the chosen `limb_dark` law (e.g. `[u1, u2]` for
   `"quadratic"`). Passing a scalar or wrong-length list raises an internal
   error that, again, a broad `except:` can swallow.

**Test case:**
```
input: synthetic transit with known Rp/Rs=0.02 (depth=400ppm), known period,
       known t0, zero noise
assert: recovered depth within 1% of 400ppm
assert: recovered depth_uncertainty is a finite positive number, NOT nan
assert: fit does not silently catch any exception — if batman import fails,
        the test itself must fail loudly, not return a zero
```

### B6. Vetting — Odd-Even, Secondary Eclipse, Centroid

**Odd-even depth test:**
```
epoch_number(t) = floor((t - t0) / P)
odd_group  = transits where epoch_number is odd
even_group = transits where epoch_number is even

delta      = depth_odd - depth_even
sigma_delta = sqrt(depth_odd_err^2 + depth_even_err^2)
flag_binary_suspected = |delta| / sigma_delta > 3      (3-sigma threshold)
```
**Physics rationale:** if you've accidentally locked onto **half the true
period** (a common BLS alias for eclipsing binaries, where primary and
secondary eclipses look similar), odd and even "transits" are actually
alternating primary/secondary eclipses of different depths — this test catches
exactly that.

**Secondary eclipse search:** for a circular orbit, the secondary eclipse is
at orbital phase 0.5. For an eccentric orbit:
```
phase_secondary ~= 0.5 + (2/pi) x e x cos(omega)     (Winn 2010, approx.)
```
Measure the depth at that phase the same way as the primary; if depth is
significantly non-zero (e.g. >3-sigma), flag as a likely eclipsing binary
rather than a planet (planets essentially never produce a detectable optical
secondary eclipse at this sensitivity level).

**Centroid shift check:** compute the flux-weighted centroid per cadence from
the Target Pixel File (TPF):
```
centroid_x(t) = sum(flux_pixel_i(t) * x_i) / sum(flux_pixel_i(t))
centroid_y(t) = sum(flux_pixel_i(t) * y_i) / sum(flux_pixel_i(t))
```
Compare the average centroid in-transit vs. out-of-transit. A genuine planet
transiting the target star dims the WHOLE aperture uniformly -> centroid does
not move. A blended eclipsing binary on a neighboring pixel dims only part of
the aperture -> centroid measurably shifts toward/away from that neighbor.

**Known gap:** TPF/pixel-level data is not always available for FFI-only
targets without a separate `TESScut`/pixel-cutout step — if unavailable, mark
`centroid_shift_detected: "not_tested"` rather than silently skipping it.

**Test cases:**
```
1. Known confirmed planet: assert odd-even delta is NOT significant (<3-sigma),
   no secondary eclipse detected, no centroid shift.
2. Known eclipsing binary (e.g. an EB from the Kepler EB catalog): assert AT
   LEAST ONE of the three vetting flags fires.
```

### B7. Statistical Significance — SNR & Bootstrap FAP

```
SNR = depth / (sigma_per_point / sqrt(N_in_transit_total))
```
where `sigma_per_point` is the robust out-of-transit scatter (post-detrending)
and `N_in_transit_total` is the SUM of in-transit points across ALL observed
transits (Part A2's calculation) — not just one transit.

**Bootstrap False Alarm Probability:**
```
repeat N_bootstrap times:
    shift the time series by a random circular offset (preserves
    autocorrelation/red-noise structure better than a naive shuffle)
    re-run the BLS search
    record the maximum power found
FAP = (count of trials where max_power >= observed_power) / N_bootstrap
```
**How many bootstrap trials do you need?** To resolve a FAP value of size f
with any statistical meaning, you need `N_bootstrap >> 1/f`. E.g., to claim
FAP ~ 0.001 you need at minimum several thousand trials (1000 is the bare
floor; 5000-10000 gives a stable estimate) — fewer than that and your FAP
number is itself just noise.

**Test case:**
```
input: pure white noise, no signal
assert: FAP computed on the (non-existent) "best period" is high (> 0.1) —
        i.e. the test correctly tells you "this isn't significant"
```

### B8. Feature Engineering & Classification — Scaling, Imbalance, Leakage

**Scaling must happen AFTER train/test split, fit only on train:**
```
scaler.fit(X_train)               # fit ONLY on training data
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # transform, do NOT re-fit
```
Fitting the scaler on the full dataset before splitting leaks test-set
statistics into training — a classic, very common bug that inflates reported
accuracy without you realizing it.

**Class imbalance:** real transit catalogs have far more false positives/noise
than confirmed planets. Use `class_weight='balanced'` in
`RandomForestClassifier`, or oversample the minority class with
`imbalanced-learn`'s SMOTE — applied only to the training set, never to the
test set.

**Test case:**
```
assert: scaler's .fit() is called exactly once, only on X_train
assert: classification_report on held-out test shows recall on the minority
        (planet) class > 0, not just high accuracy driven by the majority class
```

### B9. Final Output / Inference

Every processed target should emit one structured result — re-stated here for
completeness (matches the earlier prompt's OUTPUT FORMAT):
```
{
  "target_id", "period_days" (+- uncertainty), "depth_pct" (+- uncertainty),
  "duration_hours" (+- uncertainty), "snr", "false_alarm_probability",
  "vetting": {odd_even_consistent, secondary_eclipse_detected,
              centroid_shift_detected (or "not_tested")},
  "classification", "classification_confidence",
  "known_value_comparison" (if testing against a confirmed planet)
}
```

---

## PART C — MASTER TROUBLESHOOTING TABLE

| Symptom | Root Cause | Fix |
|---|---|---|
| Depth = 0.0, uncertainty = nan | `batman` import silently failing inside a broad `except`, OR bad initial guess, OR unit mismatch (days vs hours) | Verify `pip list` inside the actual `.venv`; seed `k_guess = sqrt(BLS_depth)`; assert units explicitly |
| Boxy/rectangular trend line at specific x-positions | Savitzky-Golay filtering across real time gaps (treats index-adjacent as time-adjacent) | Split into contiguous segments at gaps > threshold, detrend each separately, or use `wotan.flatten()` |
| BLS power rises steadily toward long periods instead of flat floor | Residual systematics from incomplete detrending; search range too close to baseline limit; or real stellar red noise | Fix detrending first; cap `P_max = baseline/5`; pre-whiten stellar rotation if rise persists |
| Best period is exactly 2x or 0.5x of the "true" expected period | Classic BLS alias for eclipsing binaries (primary/secondary look similar) | Run the odd-even depth test — this is exactly what it's designed to catch |
| Recovered depth is significantly off from a known/published value | Limb-darkening coefficients not set or wrong law; binned vs. unbinned fit mismatch; insufficient in-transit points for that specific target | Use published u1/u2 for the star if known; verify points-in-transit via Part A2's formula |
| ML accuracy is suspiciously high (>98%) | Data leakage — scaler or feature selection fit on full dataset before split | Refit pipeline strictly after `train_test_split`; recheck with stratified k-fold |
| Centroid check always returns "not tested" | Target is FFI-only, no pixel-level TPF available without `TESScut` | Either implement `TESScut` cutout retrieval, or explicitly report this limitation — don't hide it |
| FAP number looks suspiciously clean/round | Too few bootstrap iterations for the claimed FAP precision | Increase `N_bootstrap` so that `N_bootstrap >> 1/FAP` |
| Light curve has visible large gaps in the time-axis plot | Normal — multi-quarter/sector data stitched together (downlink/repointing gaps) | Not a bug by itself, but every gap-sensitive stage (detrending, BLS grid, vetting) must explicitly handle it |

---

## PART D — Test Case Suite (Summary Checklist)

Run all of these before trusting any "final" result:

1. Data loader returns monotonic time, matching array lengths, non-empty PDCSAP_FLUX
2. Sigma-clip removes injected outliers without eating an injected shallow transit
3. Detrending on a light curve with an artificial gap shows no edge artifact
4. BLS recovers the injected period in pure-signal synthetic data within 0.1%
5. BLS on pure white noise (no signal) shows a flat noise floor, no fake peak
6. Transit-model fit on a known synthetic transit recovers depth within 1%, with a finite (non-nan) uncertainty
7. Odd-even test on a known confirmed planet shows no significant split
8. Odd-even and/or secondary-eclipse test on a known eclipsing binary fires at least one flag
9. FAP on pure noise is high (>0.1); FAP on strong injected signal is low (<0.001) given enough bootstrap trials
10. Classifier recall on the minority (planet) class is non-zero on a held-out, leakage-free test split

If any of these fail, do not trust the pipeline's output on real unknown data yet — fix the failing test first.
