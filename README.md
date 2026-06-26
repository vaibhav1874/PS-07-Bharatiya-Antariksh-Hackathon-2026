---
title: PS-07 Exoplanet Transit Detection
sdk: docker
emoji: 🚀
colorFrom: blue
colorTo: purple
pinned: false
short_description: AI pipeline for exoplanet transit detection
---
# PS-07 — Exoplanet Transit Detection Pipeline

> **Bharatiya Antariksh Hackathon 2026 | Problem Statement 07**  
> An AI-based pipeline for automatically detecting exoplanet transit signals from noisy astronomical light curves.

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline on Kepler-10b
python pipeline.py --target "KIC 11904151" --mission Kepler --skip-fap

# 4. Launch the interactive demo
streamlit run streamlit_app.py
```

---

## Project Structure

```
ps-07/
├── data_loader.py       # Phase 2-3: Download + preprocess light curves
├── detrend.py           # Phase 4: Savitzky-Golay / Wotan detrending
├── identify.py          # Phase 5: BLS period search + phase-fold
├── characterize.py      # Phase 6: batman + lmfit transit model fit
├── vet.py               # Phase 7: Vetting (odd-even, secondary eclipse, centroid, shape, duration)
├── significance.py      # Phase 8: SNR + bootstrap FAP
├── classify.py          # Phase 9-10: Feature engineering + RF inference
├── train_classifier.py  # Phase 10: Train RandomForestClassifier on TCE data
├── pipeline.py          # Phase 11: End-to-end orchestrator
├── streamlit_app.py     # Phase 12: Interactive web demo
├── batman_wrapper.py    # Pure-Python Mandel-Agol fallback (no C compiler needed)
├── tce_data.csv         # Kepler TCE dataset (PC / AFP / NTP labels)
├── models/
│   ├── rf_classifier.joblib
│   └── imputer.joblib
├── plots/               # Auto-generated plots
├── results/             # Auto-generated JSON results
├── tests/
│   └── test_synthetic.py
└── requirements.txt
```

---

## Pipeline Phases

| Phase | Module | What it does |
|-------|--------|-------------|
| 2-3 | `data_loader.py` | Downloads TESS/Kepler light curves via `lightkurve`; sigma-clips outliers |
| 4 | `detrend.py` | Savitzky-Golay (or Wotan biweight) detrending; window = 3× max transit duration |
| 5 | `identify.py` | Box Least-Squares period search; returns period, t0, depth, duration, periodogram |
| 6 | `characterize.py` | Fits Mandel-Agol transit model with `batman` + `lmfit`; returns 1-σ uncertainties |
| 7 | `vet.py` | 5 vetting tests: odd-even depth, secondary eclipse, centroid shift, shape, duration consistency |
| 8 | `significance.py` | SNR = depth / (σ_oot / √N_in); bootstrap FAP via phase-shuffling |
| 9-10 | `classify.py` | Assembles feature vector → RandomForestClassifier → PC / AFP / NTP |
| 11 | `pipeline.py` | Orchestrates all phases; outputs structured JSON |
| 12 | `streamlit_app.py` | Interactive web dashboard |

---

## Running Individual Phases

```bash
# Detrending only
python detrend.py --target "KIC 11904151" --mission Kepler

# Period search
python identify.py --target "KIC 11904151" --mission Kepler

# Vetting (tests on confirmed planet + eclipsing binary)
python vet.py --target "KIC 11904151" --mission Kepler

# Statistical significance
python significance.py --target "KIC 11904151" --n-fap-trials 500

# Retrain classifier (overwrites models/)
python train_classifier.py
```

---

## Classification Output Format

```json
{
  "target_id": "KIC 11904151",
  "period_days": 0.837524,
  "depth_pct": 0.0147,
  "duration_hours": 1.811,
  "snr": 28.4,
  "false_alarm_probability": 0.0,
  "vetting": {
    "odd_even_consistent": true,
    "secondary_eclipse_detected": false,
    "centroid_shift_detected": false
  },
  "classification": "planet_candidate",
  "classification_confidence": 0.92,
  "known_value_comparison": {
    "period_error_pct": 0.01,
    "depth_error_pct": 3.2,
    "duration_error_pct": 4.7
  }
}
```

---

## ML Classifier Details

- **Dataset:** Kepler TCE cumulative table (NASA Exoplanet Archive), filtered to PC / AFP / NTP labels
- **Source file:** `tce_data.csv` (20,367 rows; UNK labels dropped at training time)
- **Algorithm:** `RandomForestClassifier` (500 trees, `balanced_subsample` class weights)
- **Class imbalance:** Addressed via SMOTE oversampling in training
- **Features (7):** depth, duration, period, model SNR, odd-even depth stat, impact parameter, planet radius
- **Train/test split:** 80/20 stratified

---

## Tests

```bash
python test_all_part_d.py
```

The test suite uses a synthetic light curve to verify that each pipeline stage runs without error. It does **not** use real downloaded data.

---

## Deployment (Streamlit Cloud)

To deploy to [Streamlit Community Cloud](https://share.streamlit.io):
1. Create a GitHub repository and push this code.
2. **Note:** `models/rf_classifier.joblib` might be large (>100MB). Use [Git LFS](https://git-lfs.com/) to push it, or reduce `n_estimators` in `train_classifier.py` and retrain.
3. Deploy directly from your repo on Streamlit Cloud!

---

## Known Limitations

- **Centroid shift check is simplified.** A real centroid check requires full pixel-level TPF analysis with PRF fitting. Our proxy uses flux-weighted scatter, which may miss small blends.
- **Bootstrap FAP is slow.** 1000 trials takes ~10-20 min on a typical laptop. The `--skip-fap` flag speeds up demos at the cost of FAP accuracy.
- **CNN stage not implemented.** Only the Random Forest baseline classifier is included (AstroNet-style CNN is a known extension).
- **Single-planet assumption.** The BLS search returns one best signal. Multi-planet systems are not iteratively searched.
- **Detrending window is uniform.** A variable window adapted to local stellar variability would improve results on active stars.
- **batman on Windows** requires MSVC build tools. If unavailable, `batman_wrapper.py` provides a pure-Python fallback accurate to < 0.1% for Rp/Rs < 0.3.

---

## Data Sources

| Data | Source |
|------|--------|
| Light curves | MAST archive via `lightkurve` (TESS / Kepler) |
| TCE labels | [NASA Exoplanet Archive — Kepler Q1-Q17 DR25 TCE table](https://exoplanetarchive.ipac.caltech.edu) |
| Reference planet params | NASA Exoplanet Archive confirmed planets table |

---

## Dependencies

See `requirements.txt` for pinned versions. Key libraries:

| Library | Purpose |
|---------|---------|
| `lightkurve` | Light curve download + TPF access |
| `astropy` | BoxLeastSquares, FITS, time conversions |
| `wotan` | Astronomical detrending (biweight/spline) |
| `batman-package` | Mandel-Agol transit model |
| `lmfit` | Non-linear least-squares + covariance |
| `scikit-learn` | RandomForestClassifier + metrics |
| `imbalanced-learn` | SMOTE oversampling |
| `streamlit` | Interactive web demo |