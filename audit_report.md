# PS-07 Full Pipeline Audit Report
*Generated: 2026-06-26 — Complete scientific & code review of all main files*

---

## 🔴 CRITICAL BUGS (cause wrong output or crashes)

---

### BUG-01 — `pipeline.py` line 205: Wrong vetting score key
**Severity:** 🔴 Critical — vetting score always shows **0** on dashboard

```python
# CURRENT (WRONG):
"vetting_score": int(vet_summary.get("total_score", 0)),   # key doesn't exist!
"vetting_verdict": str(vet_summary.get("verdict", "unknown")),  # key doesn't exist!

# vet.py vetting_verdict() ACTUALLY returns:
# { "overall_score": ..., "n_pass": ..., "n_fail": ..., "n_inconclusive": ..., "disposition": ... }
```

**Fix needed in `pipeline.py` lines 205–206:**
```python
# CORRECT:
"vetting_score":   int(vet_summary.get("overall_score", 0)),   # ✅ actual key
"vetting_verdict": str(vet_summary.get("disposition", "unknown")),  # ✅ actual key
```

---

### BUG-02 — `pipeline.py` line 176: `depth_ppm` uses BLS depth, not fitted depth
**Severity:** 🔴 Critical — `depth_pct` output uses rough BLS box estimate instead of accurate batman fit

```python
# CURRENT (line 176):
depth_ppm = float(best_signal["depth"]) * 1e6   # BLS box estimate — can be 2-3× off

# SHOULD BE:
depth_ppm = float(fit_params.get("depth_ppm_val", best_signal["depth"] * 1e6))
```

**Impact:** `depth_pct`, `depth_uncertainty_pct`, and `known_value_comparison.recovered_depth_ppm` all use wrong depth value.

---

### BUG-03 — `pipeline.py` line 177: `duration_h` uses BLS duration, not fitted
**Severity:** 🔴 Critical — same issue as BUG-02 for duration

```python
# CURRENT (line 177):
duration_h = float(best_signal["duration"]) * 24.0   # BLS raw estimate

# SHOULD BE (use batman-fitted value, fall back to BLS if fit failed):
duration_h = float(fit_params.get("duration_h_val", best_signal["duration"] * 24.0))
```

---

### BUG-04 — `pipeline.py` line 201: `duration_uncertainty_hours` always NaN
**Severity:** 🔴 Critical — `characterize.py` sets `"duration_h_err": np.nan` always (marked as "complex, skip here")

```python
# characterize.py line 465:
"duration_h_err": np.nan,   # ← never computed!
```

**Fix:** Propagate uncertainty from `a_rs_err` and `inc_err`:
```python
# In characterize.py, after computing duration_val:
# Partial derivative approximation:
# dT/da ≈ -(T/a)   →   σ_T ≈ (T/a) * σ_a
if not np.isnan(a_err) and a_val > 0:
    duration_err = abs(duration_val / a_val) * a_err
else:
    duration_err = np.nan
```

---

### BUG-05 — `streamlit_app.py` line 303: Vetting score display wrong
**Severity:** 🔴 Critical — shows `? / 5` always because of BUG-01

```python
# Line 303: st.markdown(f"**Vetting score:** {result.get('vetting_score', '?')} / 5")
# Due to BUG-01, vetting_score is always 0 — max possible from 5 tests is 5
# Fix: also fix the key in pipeline.py (BUG-01) AND cap display:
st.markdown(f"**Vetting score:** {result.get('vetting_score', '?')} / 5")
# After BUG-01 fix this will work correctly (score range: -5 to +5)
```

---

## 🟠 MEDIUM BUGS (wrong values, not crashes)

---

### BUG-06 — `significance.py` line 266: Wrong `baseline_std` for secondary SNR
**Severity:** 🟠 Medium — secondary eclipse SNR is overestimated

```python
# CURRENT (line 266): divides by sqrt(N_in_secondary) — wrong for point scatter
baseline_std = np.std(flux[out_of_transit]) / np.sqrt(in_secondary.sum())

# This gives uncertainty on the MEAN, but secondary_depth uses a single-point median.
# CORRECT for per-point noise (used in SNR = depth/per-point-noise):
baseline_std = 1.4826 * np.median(np.abs(flux[out_of_transit] - np.median(flux[out_of_transit])))
# Then secondary_snr = secondary_depth_ppm / (baseline_std * 1e6 / sqrt(N_in_secondary))
```

---

### BUG-07 — `vet.py` line 419: Kepler's 3rd law — missing factor for duration T_max
**Severity:** 🟠 Medium — T_max slightly underestimated for eccentric orbits

```python
# CURRENT (line 419):
t_max_days = period_days * r_star_au / (np.pi * a_au)

# This is correct only for circular orbit b=0. 
# More accurate (includes impact parameter range):
# T_max = (P/π) * arcsin(R_s/a) * (for b=0)
# The arcsin(R_s/a) ≈ R_s/a for R_s << a (valid for a > 10 R_s)
# Current formula is fine for most cases, but should document the approximation.
# No code change needed, but add comment:
# NOTE: valid for circular orbit, b=0, R_s << a (a > 5 R_s)
```

---

### BUG-08 — `classify.py` line 167: Rp radius conversion factor wrong
**Severity:** 🟠 Medium — planet radius in Earth radii is slightly off

```python
# CURRENT (line 167):
prad_earth = rp_rs * 109.076 * star_radius_rsun

# 1 R_sun = 109.076 R_Earth — this is CORRECT. ✅
# However, star_radius_rsun defaults to 1.0 in pipeline.py call (line 165),
# NOT passed from the user's star_r sidebar input!

# pipeline.py line 165 — classify_from_pipeline_outputs is called WITHOUT star_radius_rsun:
clf_result = classify_from_pipeline_outputs(
    best_signal=best_signal,
    fit_params=fit_params,
    snr_result=sig,
    vet_results=vet_tests,
    # ← star_radius_rsun MISSING! defaults to 1.0, ignores user sidebar input
)
```

**Fix in `pipeline.py` line 165:**
```python
clf_result = classify_from_pipeline_outputs(
    best_signal=best_signal,
    fit_params=fit_params,
    snr_result=sig,
    vet_results=vet_tests,
    star_radius_rsun=star_radius_rsun,  # ✅ pass actual stellar radius
)
```

---

### BUG-09 — `pipeline.py` line 203: `false_alarm_probability` NaN when skip_fap=True
**Severity:** 🟠 Medium — dashboard shows "FAP not computed" even when run completes

```python
# FAP is NaN when skip_fap=True (default in sidebar).
# streamlit_app.py line 293 correctly handles this with "FAP not computed".
# But pipeline.py should document this clearly.
# IMPROVEMENT: Add a note in output JSON:
"false_alarm_probability": round(float(sig.get("fap", np.nan)), 5),
"fap_note": "NaN = FAP not computed (skip_fap=True); run with FAP enabled for significance test",
```

---

### BUG-10 — `characterize.py` line 360: Transit window too generous for short periods
**Severity:** 🟠 Medium — 15% phase window excludes important baseline for P < 0.9d

```python
# CURRENT (line 360):
transit_half_width = 0.15   # ±15% of phase

# For Kepler-10b (P=0.837d), transit duration ≈ 1.81h → phase_dur ≈ 1.81/(0.837*24) = 0.090
# So 0.15 is only 1.67× the transit duration — narrow enough, but for very short periods
# this may be too tight. Consider dynamic calculation:
transit_half_width = max(0.15, 3.0 * init_params.per / (np.pi * init_params.a * init_params.per))
```

---

## 🟡 ACCURACY IMPROVEMENTS (no crash, but better results)

---

### IMPROVE-01 — `significance.py`: SNR formula uses full light curve noise
**Current:** sigma_oot = MAD-based scatter of all out-of-transit flux  
**Better:** Use only flux adjacent to transits (within 2× duration) to avoid long-term trends

```python
# Add local baseline option to compute_snr():
local_mask = _transit_mask(time, period, t0, duration * 4.0) & ~in_mask
if local_mask.sum() > 20:
    oot_flux_local = flux[local_mask]
    # Use local noise for SNR — more accurate for trending data
```

---

### IMPROVE-02 — `identify.py` line 197: BLS `min_period` default too low for Kepler long-cadence
**Current:** `min_period=0.5` days — BLS misses some short periods in 30-min cadence data  
**Better:** `min_period = max(0.5, 3.0 * cadence_days)` — needs cadence awareness  
**Impact:** Reduces false period detections at periods near cadence

---

### IMPROVE-03 — `characterize.py`: Period vary=False in lmfit is correct but period uncertainty is approximated
**Current:** Period sigma estimated via Kovacs 2002 formula `σ_P ≈ P²/(baseline × SNR)`  
**Issue:** Formula is order-of-magnitude estimate only; actual BLS grid step is `P²/(baseline × n_periods)`  
**Better formula:**
```python
# More accurate BLS period uncertainty:
n_periods = 9498  # from build_period_grid
baseline = float(time[-1] - time[0])
period_grid_sigma = period_d**2 / (baseline * n_periods)
```

---

### IMPROVE-04 — `vet.py` line 483: BIC calculation approximation
**Current:** `sigma² = RSS/N` — assumes uniform noise  
**Issue:** Doesn't account for varying flux_err per point  
**Better:** Weighted BIC using per-point uncertainties

---

### IMPROVE-05 — `detrend.py`: SavGol window too long for very short periods
**Impact:** If window > period, SavGol will partially remove the transit signal  
**Check needed:** Ensure `window_pts < period / cadence_min` always

---

## 📊 DASHBOARD DISPLAY ISSUES

---

### DASH-01 — Vetting score shows 0 (BUG-01 in pipeline.py)
- Fix BUG-01 to get correct values (+1 to +5 for good planet candidates)

### DASH-02 — `depth_pct` shows BLS rough depth (BUG-02)
- After BUG-02 fix, depth will show batman-fitted value (more accurate)

### DASH-03 — `duration_uncertainty_hours` always shows NaN/blank (BUG-04)
- After BUG-04 fix, will show propagated uncertainty from a_rs fit

### DASH-04 — Period uncertainty formula
- Current: Kovacs formula (approximate) — can be off by 2-3×
- After IMPROVE-03: will use exact BLS grid step

### DASH-05 — `class_probabilities` correct but planet radius feature uses wrong stellar radius (BUG-08)
- Fix BUG-08 to pass actual star_radius_rsun to classifier

---

## ✅ WHAT IS CORRECT

| Calculation | Status | Notes |
|-------------|--------|-------|
| SNR formula: `depth/(σ_oot/√N_in)` | ✅ Correct | Kovacs 2002 — good |
| BLS period search | ✅ Correct | astropy BoxLeastSquares, log-spaced grid |
| Phase folding | ✅ Correct | Correct modulo arithmetic |
| Batman transit model (depth = Rp²/Rs²) | ✅ Correct | Mandel-Agol 2002 |
| lmfit residual function | ✅ Correct | Weighted chi-square |
| `rp = sqrt(depth)` init | ✅ Correct | Standard formula |
| `a_rs` from duration init | ✅ Correct | Seager formula |
| Impact parameter `b = a*cos(inc)` | ✅ Correct | Standard |
| Odd-even |Δ|/σ threshold = 3 | ✅ Correct | Bryson 2013 |
| Kepler's 3rd law (T_max) | ✅ Correct | `a³ = M·P²` in AU/yr/Msun |
| BIC formula | ✅ Correct | `-2·ln(L) + k·ln(N)` |
| MAD sigma = 1.4826·MAD | ✅ Correct | Gaussian noise estimator |
| sigma_oot for SNR uses MAD | ✅ Correct | Robust estimator |
| `depth_pct = depth_ppm/1e4` | ✅ Correct | ppm → % conversion |
| FAP bootstrap method | ✅ Correct | Phase-shuffling approach |
| Limb darkening correction | ✅ Correct | Quadratic model |

---

## 🔧 PRIORITY FIX LIST

Apply in this order:

| Priority | Bug | File | Line | Description |
|----------|-----|------|------|-------------|
| 1 | BUG-01 | pipeline.py | 205-206 | `total_score`→`overall_score`, `verdict`→`disposition` |
| 2 | BUG-02 | pipeline.py | 176 | Use fitted depth not BLS depth for output |
| 3 | BUG-03 | pipeline.py | 177 | Use fitted duration not BLS duration for output |
| 4 | BUG-08 | pipeline.py | 165 | Pass `star_radius_rsun` to classifier |
| 5 | BUG-06 | vet.py | 266 | Fix secondary eclipse std calculation |
| 6 | BUG-04 | characterize.py | 465 | Compute duration_err from a_rs_err |
| 7 | IMPROVE-03 | pipeline.py | 189 | Use exact BLS grid step for period_uncertainty |
