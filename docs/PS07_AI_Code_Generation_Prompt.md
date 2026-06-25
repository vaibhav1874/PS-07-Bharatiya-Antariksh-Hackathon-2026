# Master Prompt: PS-07 Exoplanet Detection — Real, Working PoC

> **Kaise use karo:** Is poore document ko copy karke kisi bhi capable AI coding assistant
> (Claude, ChatGPT, Claude Code, Cursor, etc.) ko de do. Agar tool chhota context window
> leta hai, to "PHASE-WISE BREAKDOWN" section ke hisaab se isse 5 alag messages mein split
> karke bhejo (ek phase complete hone ke baad next bhejo). Neeche "How To Use This Prompt"
> section mein step-by-step instructions hain.

---

## THE PROMPT (copy everything below this line)

```
You are a senior research engineer with combined expertise in observational
astrophysics (exoplanet transit photometry) and applied machine learning. You
write production-quality, well-documented, honestly-evaluated code — not
demo-ware. You never hide or omit a metric because it looks bad; you report
it and explain it.

## CONTEXT — THE PROBLEM

I am building a submission for Problem Statement 07 of the Bharatiya Antariksh
Hackathon 2026 (ISRO + Physical Research Laboratory, Ahmedabad):

"Develop an AI-based data analysis pipeline capable of automatically detecting
exoplanet transit signals from noisy astronomical light curve data."

Background: when a planet transits its host star, brightness drops by a tiny,
periodic amount. Real light curves (TESS/Kepler) are contaminated by:
- Instrumental noise and systematic drift
- Stellar variability (starspots, rotation)
- Blending: light from a foreground/background star falling in the same
  photometric aperture, which can fake or distort a transit signal
- Eclipsing binary stars, which produce dips that mimic planetary transits

The pipeline must perform five official sub-tasks:
1. DETRENDING — remove noise/systematics, preserve the real transit shape
2. IDENTIFICATION — find periodic dips that could be transits
3. CHARACTERIZATION — estimate depth, duration, orbital period (with
   uncertainty, not just a point value)
4. CLASSIFICATION — sort each candidate into: genuine transit / eclipsing
   binary / blended false positive / noise
5. STATISTICAL SIGNIFICANCE — report SNR and a False Alarm Probability (FAP)
   for every detection, not just a yes/no answer

## WHAT I NEED FROM YOU

A real, runnable, end-to-end Python pipeline — not a toy. Specifically:

### Hard requirements (do not skip any of these):

1. **Use real public data, not only synthetic data.** Use `lightkurve` to
   download an actual TESS or Kepler light curve for a CONFIRMED planet host
   (e.g. Kepler-10, Kepler-8, or any TIC ID I can verify on the NASA
   Exoplanet Archive) as the primary test case. Additionally test on at
   least ONE known eclipsing binary or known false positive from the Kepler
   TCE catalog, to prove the vetting logic actually rejects it. You may ALSO
   include a synthetic light-curve generator, but label it clearly as a
   controlled unit test, never as the main result.

2. **Report parameter recovery error honestly.** When you test against a
   known/confirmed planet, print and log the % error between your recovered
   period/depth/duration and the published values from the NASA Exoplanet
   Archive. If the error is large, say so in the output and in code
   comments — do not silently round numbers to make results look better.

3. **Detrending** (`scipy.signal.savgol_filter` or the `wotan` package):
   - Sigma-clip outliers (5-sigma) before smoothing
   - Window length = 3x the expected maximum transit duration — compute
     this from the period grid, don't hardcode a magic number
   - Justify the choice with a one-line comment citing Hippke et al. 2019
     (Wotan paper) logic

4. **Identification** (`astropy.timeseries.BoxLeastSquares` or the
   `transitleastsquares` package):
   - Search a period grid appropriate to the baseline length of the data
     (Nyquist-aware: don't search periods longer than ~1/3 of the baseline)
   - Return best period, t0, duration, depth, and the full periodogram
   - Plot the periodogram with the best period marked

5. **Characterization**:
   - Fit a real Mandel-Agol transit model using the `batman` package,
     optimized with `lmfit` (Levenberg-Marquardt) against the phase-folded
     data
   - Return depth, duration, ingress/egress time, baseline flux, AND their
     1-sigma uncertainties (from the fit covariance, not guessed)

6. **Vetting checks — this is the part most teams skip, do not skip it:**
   - **Odd-even depth test**: split transits into odd-numbered and
     even-numbered occurrences, fit each separately, compare depths. A
     significant mismatch flags a possible eclipsing binary (mistaken
     period, half the true period).
   - **Secondary eclipse search**: search for a shallower dip near orbital
     phase 0.5. If found at significant depth, flag as likely binary, not
     a planet.
   - **Centroid shift check (if feasible)**: using `lightkurve`'s target
     pixel file (TPF) access, check whether the photometric centroid shifts
     during transit. A shift indicates the signal comes from a blended
     neighboring star, not the target — this directly addresses the
     "stellar blending" contamination named in the official problem
     statement. If this is too complex to implement fully, implement at
     least a simplified version and clearly comment on its limitations.

7. **Statistical significance**:
   - SNR = depth / (per-point scatter / sqrt(N_points_in_transit)) — show
     the formula in a comment, do not just write "SNR = depth/noise"
   - False Alarm Probability via bootstrap: shuffle/resample the
     out-of-transit data N times (e.g. N=1000), re-run BLS, and report what
     fraction of trials produce a signal at least as strong as the real one

8. **Classification (ML)**:
   - Engineer features: depth, duration, period, odd-even depth difference,
     secondary-eclipse depth ratio, transit shape metric (flat-bottom
     fraction), SNR
   - Train a `RandomForestClassifier` (scikit-learn) on the Kepler Labelled
     Time Series / TCE dataset (Shallue & Vanderburg 2018 — tell me where
     you are sourcing this from, e.g. Kaggle "Kepler labelled time series
     data" or the NASA Exoplanet Archive TCE table)
   - Report REAL metrics on a held-out test split: precision, recall,
     F1, confusion matrix across the actual classes (PC / AFP / NTP). Do
     not fabricate numbers — run the actual training and paste the actual
     output.
   - If accuracy looks too good to be true (>98%), check for data leakage
     before reporting it, and mention this check in your response.

9. **Code quality bar**:
   - Modular: separate files/functions per stage (`detrend.py`,
     `identify.py`, `characterize.py`, `vet.py`, `significance.py`,
     `classify.py`, `pipeline.py` as orchestrator)
   - Type hints + docstrings on every function
   - A `requirements.txt` with pinned versions
   - A `README.md` explaining how to run it, what each module does, and
     KNOWN LIMITATIONS (be specific: e.g. "centroid check is simplified
     and may miss small blends," "CNN stage not yet implemented," etc.)
   - Deterministic: set random seeds
   - No silent `except: pass` — handle/log real errors

10. **Do not oversell.** Anywhere you write a comment, docstring, or print
    statement describing what the pipeline achieves, only claim what the
    code actually demonstrates in its test run. If a stage is a stub or
    simplified version, label it as such explicitly.

## TECH STACK TO USE

Use exactly these libraries unless you have a specific, stated reason to
substitute one — if you substitute, tell me why before proceeding.

| Category | Library | Used For |
|---|---|---|
| Data access | `lightkurve` | Search/download TESS & Kepler light curves by target ID |
| Data access | `astroquery` | Query MAST archive / NASA Exoplanet Archive directly |
| Data access | `astropy` | FITS handling, time/unit conversions, `BoxLeastSquares` |
| Signal processing | `numpy` | Array math throughout |
| Signal processing | `scipy.signal` | `savgol_filter` for detrending |
| Signal processing | `wotan` | Purpose-built astronomical detrending (biweight/spline) |
| Period search | `astropy.timeseries.BoxLeastSquares` | Primary BLS implementation |
| Period search | `transitleastsquares` (TLS) | More sensitive alternative/cross-check to BLS |
| Transit modeling | `batman-package` | Mandel-Agol physical transit light-curve model |
| Transit modeling | `lmfit` | Non-linear least-squares fitting + parameter uncertainties |
| Transit modeling | `emcee` (optional) | MCMC posterior/uncertainty estimation, if time allows |
| Statistics | `scipy.stats`, `numpy` | SNR calculation, bootstrap resampling for FAP |
| Machine learning | `scikit-learn` | `RandomForestClassifier`, `train_test_split`, metrics |
| Machine learning | `imbalanced-learn` | Handle class imbalance (planets are rare) |
| Machine learning | `xgboost` (optional) | Stronger baseline classifier, compare vs RandomForest |
| Deep learning (stretch) | `tensorflow` or `pytorch` | CNN, global+local view (AstroNet-style), only after RF baseline works |
| Visualization | `matplotlib` | All plots (light curves, periodograms, phase folds) |
| Demo | `streamlit` | Interactive dashboard: input target -> see pipeline run |
| Packaging/Dev | `pip` + `requirements.txt`, `pytest`, `git` | Reproducibility, basic tests, version control |

State the exact pinned version of each library you used in `requirements.txt`.

## STEP-BY-STEP PIPELINE (FOLLOW THIS EXACT ORDER, do not skip or reorder)

**Phase 1 — Setup**
1.1 Create a virtual environment; install everything from `requirements.txt`
1.2 Sanity-check: confirm `lightkurve` can search and return results for one
    known target ID before doing anything else

**Phase 2 — Data Acquisition**
2.1 Query the target by TIC/KIC ID using `lightkurve.search_lightcurve()`
2.2 Download and, if multiple sectors/quarters exist, stitch them together
2.3 Extract `time`, `flux`, `flux_err`, `quality` arrays from the result

**Phase 3 — Preprocessing / Cleaning**
3.1 Drop NaN/invalid rows
3.2 Filter out points where the quality flag is non-zero
3.3 Iteratively sigma-clip remaining outliers (5-sigma, 2-3 passes)

**Phase 4 — Detrending**
4.1 Estimate the maximum plausible transit duration from your period search
    grid (a rough physical upper bound is fine at this stage)
4.2 Set the smoothing window length = 3x that duration, converted to number
    of cadence points
4.3 Apply `savgol_filter` (polyorder=2) or `wotan`'s biweight filter using
    that window
4.4 Subtract the trend from the raw flux and re-normalize so the median is 1.0
4.5 Plot raw vs. detrended flux side-by-side — this plot is a required
    deliverable, not optional

**Phase 5 — Identification (Period Search)**
5.1 Build a period grid (minimum ~0.5 days; maximum ~1/3 of the total
    observing baseline, to keep at least 3 transits visible)
5.2 Build a duration grid (e.g. 0.01-0.3 days)
5.3 Run `BoxLeastSquares.power()` (or TLS) across both grids
5.4 Extract the best period, t0 (epoch), duration, depth, and BLS power
5.5 Plot the periodogram with the best period clearly marked
5.6 Phase-fold the light curve at the best period

**Phase 6 — Characterization**
6.1 Bin the phase-folded data for a clean view
6.2 Initialize a `batman` `TransitParams` object with sensible starting
    guesses from Phase 5's output
6.3 Fit the model to the data using `lmfit` (Levenberg-Marquardt)
6.4 Extract depth, duration, ingress/egress time, and baseline flux, AND
    their 1-sigma uncertainties from the fit's covariance matrix
6.5 Plot the best-fit model on top of the phase-folded data

**Phase 7 — Vetting (False-Positive Checks)**
7.1 Odd-even test: split transits into odd- and even-numbered epochs, fit
    each group's depth separately, flag if they differ by more than 3-sigma
7.2 Secondary eclipse search: look for a dip near orbital phase 0.5; measure
    its depth if present
7.3 Centroid shift check: pull the target pixel file (TPF) via `lightkurve`,
    compute the flux-weighted centroid in-transit vs. out-of-transit, flag
    a significant shift (this is your blending/contamination check)
7.4 Combine all three into a single `vetting` flags dictionary

**Phase 8 — Statistical Significance**
8.1 Compute the out-of-transit per-point scatter (sigma)
8.2 Compute SNR = depth / (sigma / sqrt(N_points_in_transit))
8.3 Bootstrap FAP: resample/shuffle the out-of-transit data ~1000 times,
    rerun the period search on each shuffle, and report the fraction of
    trials that produce a signal at least as strong as the real one

**Phase 9 — Feature Engineering**
9.1 Assemble one feature vector per candidate: depth, duration, period,
    odd-even depth difference, secondary-eclipse depth ratio, SNR, and a
    transit-shape metric (e.g. flat-bottom fraction)
9.2 Scale/normalize features before feeding them to a classifier

**Phase 10 — Classification**
10.1 Load the Kepler Labelled TCE dataset (state your exact source)
10.2 Stratified train/test split
10.3 Train a `RandomForestClassifier` baseline
10.4 Evaluate on the held-out test set: precision, recall, F1, confusion
     matrix — paste the real numbers from the real run
10.5 (Optional, only after 10.1-10.4 work) Train a CNN with global+local
     view inputs, AstroNet-style
10.6 Save the trained model to disk (`joblib` or `pickle`)

**Phase 11 — Inference / Output**
11.1 Run the full pipeline (Phases 2-9) on a new target
11.2 Apply the trained classifier from Phase 10 to its features
11.3 Assemble the final structured JSON result (see OUTPUT FORMAT below)
11.4 Print and save the result

**Phase 12 — Demo**
12.1 Build a minimal Streamlit app: user enters a target ID -> app runs the
     pipeline -> shows the plots from Phases 4-6 and the final result
12.2 Test the demo end-to-end on at least one real target before calling it done

## OUTPUT FORMAT

For each light curve processed, the pipeline should output a single
structured result (dict/JSON), e.g.:

{
  "target_id": "...",
  "period_days": ..., "period_uncertainty": ...,
  "depth_pct": ..., "depth_uncertainty": ...,
  "duration_hours": ..., "duration_uncertainty": ...,
  "snr": ...,
  "false_alarm_probability": ...,
  "vetting": {"odd_even_consistent": true/false,
              "secondary_eclipse_detected": true/false,
              "centroid_shift_detected": true/false or "not_tested"},
  "classification": "planet_candidate" | "eclipsing_binary" |
                     "blended_false_positive" | "noise",
  "classification_confidence": ...,
  "known_value_comparison": {...if testing against a confirmed planet...}
}

## WHAT TO DO IF YOU CANNOT FULLY IMPLEMENT SOMETHING

Tell me explicitly which part you simplified or skipped and why (e.g. "real
centroid analysis requires the full TPF which is large to download in this
environment, so I implemented a simplified flux-weighted proxy instead").
Do not silently omit a requirement.

## DELIVERABLES, IN ORDER

1. Project file structure (just the tree, no code yet) — confirm with me
2. `data_loader.py` — fetch real TESS/Kepler data via lightkurve + a
   synthetic generator for unit tests
3. `detrend.py` + a plotted before/after example on real data
4. `identify.py` (BLS/TLS) + periodogram plot + recovered period vs. the
   known published period for your confirmed-planet test case
5. `characterize.py` (transit model fit) + recovered parameters with
   uncertainty vs. published values, with % error explicitly printed
6. `vet.py` (odd-even, secondary eclipse, centroid) tested on BOTH the
   confirmed planet AND the known eclipsing binary/false positive, showing
   the vetting correctly distinguishes them
7. `significance.py` (SNR + bootstrap FAP)
8. `classify.py` + `train_classifier.py` with real training output
   (actual metrics, not placeholders)
9. `pipeline.py` tying it all together + one end-to-end run printed in full
10. `README.md` with setup, usage, and an honest "Known Limitations" section

Work through these one deliverable at a time. After each one, show me the
actual code AND the actual output/plot from running it before moving to
the next. If something fails to run, show me the real error and fix it —
do not paper over it.
```

---

## PHASE-WISE BREAKDOWN (agar chhote context wale tool use kar rahe ho)

Agar AI tool ek baar mein itna lamba prompt handle nahi kar pa raha, isse
5 messages mein todo (yeh "Message #" niche wale pipeline ke "Phase 1-12"
se alag hai — ek message mein multiple pipeline-phases cover ho sakte hain):

| Message # | Kya bhejna hai (pipeline Phase #) | Expected output |
|---|---|---|
| 1 | Setup + Data + Preprocessing + Detrending + Identification (Phase 1-5) | `data_loader.py`, `detrend.py`, `identify.py` + plots |
| 2 | Characterization (Phase 6) | `characterize.py` + recovered-vs-true comparison |
| 3 | Vetting (Phase 7) | `vet.py` tested on planet + binary |
| 4 | Significance + Classification (Phase 8-10) | `significance.py`, `classify.py`, `train_classifier.py` + real metrics |
| 5 | Inference + Demo (Phase 11-12) | `pipeline.py`, full run, `README.md`, Streamlit app |

Har phase ke baad **khud verify karo** (neeche dekho) before next phase bhejna.

---

## HOW TO USE THIS PROMPT — TUMHE KYA KARNA HAI

1. **Ek powerful AI/coding tool use karo jo code run kar sake** — Claude
   (with code execution / Claude Code), ChatGPT with Code Interpreter, ya
   koi bhi tool jo actually Python chala sake. Sirf "describe karne wala"
   AI use karoge to woh code likh dega but verify nahi karega — usme
   hallucinated function names/wrong APIs ka risk zyada hota hai.

2. **Phase-by-phase aage badho.** Pura prompt ek saath bhejna theek hai
   agar tool capable hai, lekin **har deliverable ke baad ruk kar verify
   karo** — agla phase tab bhejo jab pichla sahi se chal raha ho.

3. **Har step pe yeh 3 cheezein check karo:**
   - Code actually run hua, ya AI ne sirf likh diya bina chalaye? Output/plot
     maango, sirf code nahi.
   - Real data use hua ya phir bhi synthetic? Agar synthetic hai, explicitly
     puchho "ise real TESS data pe chalao."
   - Numbers honest hain ya suspiciously perfect? Agar accuracy 99%+ ho ya
     error 0% ho, puchho "isme data leakage check kiya?"

4. **Known planet ID maango verification ke liye** — jaise "Kepler-10b" ya
   koi confirmed target jiska published period/depth NASA Exoplanet Archive
   pe available hai, taaki tum khud compare kar sako.

5. **Jab vetting (odd-even/secondary-eclipse/centroid) ka step aaye**, AI ko
   explicitly ek known eclipsing binary TIC/KIC ID pe bhi test karne ko
   bolo — sirf yeh dekhne ke liye ki vetting use sahi se reject karta hai
   ya nahi. Yeh wahi gap hai jo official problem statement mein "blending"
   contamination ke naam se explicitly mentioned hai.

6. **Jo bhi limitation AI bataye, README mein likhwao** — judges ke saamne
   "yeh hamne simplify kiya kyunki X" bolna "hamne sab perfect bana diya"
   bolne se zyada credible lagta hai, especially scientist judges ke liye.

7. **Last step: poora pipeline ek single confirmed planet pe end-to-end
   chalwao** aur poora output (JSON result) apne paas save karo — yehi
   tumhara real PoC evidence hai jo PPT ke Slide 8 mein use hoga.

---

## RED FLAGS — Agar AI Yeh Kare To Usse Wapas Bhejo

- Sirf synthetic data pe result dikhaye aur "validated" bole
- Depth/period error number na bataye, sirf "successfully recovered" bole
- Vetting checks likhe but kabhi test na kare ek real false-positive pe
- ML accuracy/metrics bina actual training chalaye paste kare
- Koi function `fold()`, `detect_transit()` jaise generic naam se bina
  definition ke use kare (hallucinated/incomplete code ka sign)
- "Centroid check" likhe but TPF data download/use kiye bina
