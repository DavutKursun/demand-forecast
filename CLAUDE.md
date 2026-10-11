# demand-forecast — weekly sales forecasting with an "what to order" dashboard

This file is the build spec. Read it fully before writing code. Work milestone by
milestone; after each milestone stop, show how to run it, and wait for Davut's OK.

## 0. Status (updated 2026-10-11)

**Milestones 1–4 are done (38 tests pass incl. headless page renders; eight artifacts for all 500 series committed; three-page dashboard runs locally). Next: M5 — deploy to Streamlit Community Cloud or a Hugging Face Space (`sdk: streamlit`), add screenshots to the README "Screenshots" section and a GIF to `assets/demo.gif`, then the live-demo link.**

Implemented:
- `demand_forecast/data.py` — load daily, Monday-anchored weekly aggregation (full weeks only, gaps → 0), checks, synthetic generator
- `demand_forecast/features.py` — one feature function for training and inference; lags, rolling stats, same-week-last-year, calendar, holidays, trend, series mean; `build_training_frame(last_origin=…)` never includes a target after the origin
- `demand_forecast/baselines.py` — seasonal naive, 8-week moving average
- `demand_forecast/models.py` — LightGBM L1 point model + P10/P90 quantile models per horizon, feature importance
- `demand_forecast/backtest.py` — rolling-origin folds, WAPE/MAE/bias/coverage, per-fold table, worst items
- `demand_forecast/recommend.py` — order quantity from stock, lead time, service level, with a plain-language explanation
- `scripts/download_data.py`, `prepare.py` (`--synthetic` option), `train.py` (backtest + final models + artifacts + metrics.json), `eval.py`
- `app.py` — entry point with `st.navigation`; pages live in `dashboard/`: `common.py` (cached `load()` of the eight artifacts, `require_artifacts()` guard, palette, pure helpers), `charts.py` (pure Plotly builders), `forecast.py`, `quality.py`, `how.py`
- tests incl. an explicit leakage test and AppTest renders of all three pages (`tests/test_dashboard.py`); CI runs tests + a synthetic train with 2 folds

M3 result on the real data (Kaggle, 500 series, 260 weeks, 6 folds, `scripts/train.py`, 7.8 min on an Apple-silicon
laptop): LightGBM WAPE 4.4% vs seasonal naive 6.4% vs 8-week MA 11.9%, 6/6 folds won, P10–P90 coverage 76% (inside
70–90%, no adjustment applied). Acceptance criteria of §10 for the model are met. Findings, per-fold table, worst items
and feature analysis are in `docs/DESIGN.md`; the README results table is filled from `scripts/eval.py`.
Committed in `artifacts/`: weekly.parquet (all 500 series), forecasts.parquet, backtest_summary.csv, backtest_by_fold.csv,
backtest_by_series.csv (LightGBM WAPE/MAE per series), worst_items.csv, feature_importance.csv, metrics.json.
Not committed: data/raw/ and backtest_predictions.parquet (ignored).

Environment notes (macOS):
- LightGBM's wheel links the OpenMP runtime; install it with `brew install libomp`, otherwise `import lightgbm` fails with
  "Library not loaded: @rpath/libomp.dylib". Training uses all cores through OpenMP; system CPU time is high, which is normal.
- `python3` may resolve to 3.9; create the venv from `python3.11`. Use `.venv/bin/python` for every script.
- Kaggle CLI 2.x authenticates with `~/.kaggle/access_token` (or `kaggle auth login`). A 403 on download means the account
  has not accepted the competition rules; `kaggle competitions list -s demand` shows `userHasEntered`.

M5 starting point: `requirements.txt` pins streamlit 1.65.0 and plotly 7.1.0 (the versions the pages were built and
tested with). The app imports only pandas/pyarrow/plotly/streamlit plus the pure modules `demand_forecast.config` and
`demand_forecast.recommend`, so a slimmer requirements file without lightgbm/scikit-learn would speed up the cold start on
free hosting if needed. Deploy, take the three screenshots for the README, record `assets/demo.gif`, add the live link, and
finish DESIGN.md's next-steps list (§11).

## 1. Goal

A Streamlit dashboard for a retailer with many products and stores: for any
store/product it shows the sales history, a 4-week forecast with an uncertainty band,
and a concrete order recommendation ("order 140 units this week") given current stock
and lead time. Behind it: a properly backtested gradient-boosting model that beats
the naive baselines the business would otherwise use.

Portfolio purpose: prove classic ML fundamentals — feature engineering, time-series
cross-validation, honest baselines, uncertainty — and the ability to turn a model
into something a non-technical manager uses. No LLM is needed; this project shows
Davut is not only an API caller.

## 2. Non-negotiables

- Every model is compared against two baselines (seasonal naive = same week last
  year; 8-week moving average) with rolling-origin backtesting. The README reports
  WAPE for baselines and the model on the same folds. If the model does not beat the
  baselines, that is the result; fix the model, don't hide the table.
- No leakage: features for week t use only data available at the forecast origin;
  lags and rolling statistics are computed per series with proper shifting; the
  backtest trains only on data before each fold's origin.
- Forecasts come with P10/P90 intervals (quantile models), and the order
  recommendation uses them (safety stock from the upper quantile).
- The app loads precomputed forecasts and backtest results from `artifacts/`; it
  starts in seconds on free hardware. Training is a separate script that runs on a
  laptop or Kaggle CPU in minutes.

## 3. Data

Default: Kaggle **Store Item Demand Forecasting Challenge** (`train.csv`: date,
store, item, sales — daily, 10 stores × 50 items, 2013–2017). Download with the Kaggle
API in `scripts/download_data.py` (needs `KAGGLE_USERNAME`/`KAGGLE_KEY`); do **not**
commit the raw file (competition data terms). Aggregate to weekly totals per
store-item (500 series). The app ships only the derived weekly aggregates for a
subset (e.g., all 10 stores × 10 items) plus forecasts, in `artifacts/`.

Alternative if licensing is a concern or Davut wants messier real data: UCI
**Online Retail II** (CC BY 4.0): aggregate invoices to weekly units per product,
keep the top 100 products by volume, handle returns (negative quantities).
Implement the default first; the data loader is an interface so the alternative can
be added later.

## 4. Stack

Python 3.11, `pandas`, `numpy`, `lightgbm`, `scikit-learn`, `streamlit`, `plotly`,
`holidays` (country calendar features), `pyarrow`, `pytest`. Optional comparison in
`docs/DESIGN.md`: `statsforecast` (ETS/Theta) as a classical model baseline.

## 5. Modeling

- Target: weekly units per store-item. Horizon: 4 weeks. Use **direct** multi-horizon
  models: one LightGBM model per horizon h ∈ {1,2,3,4} (plus quantile models for
  P10 and P90 per horizon → 12 models; small data, trains in minutes).
- Features per series at origin t: lags 1–4, 8, 12, 26, 52; rolling mean/std over 4,
  8, 13, 52 weeks; same-week-last-year value; week-of-year (sin/cos), month, year;
  holiday count in target week; store and item as categorical; series-level mean
  (target encoding computed on training data only); recent trend (slope over 8 weeks).
- Baselines: seasonal naive, 8-week moving average. Also report a global
  "LightGBM without lags" ablation if time allows (shows which features matter).
- Backtest: rolling origin, 6 folds, origins every 4 weeks over the last 24 weeks of
  data; retrain per fold (fast). Metrics: WAPE (primary), MAE, bias (mean error),
  interval coverage of P10–P90 (target ≈ 80%). Report overall, per fold, per store,
  and worst 10 items.
- Final models: retrain on all data; forecast the next 4 weeks for every series; save
  `artifacts/forecasts.parquet`, `backtest_summary.csv`, `backtest_by_fold.csv`,
  `backtest_by_series.csv` (store, item, wape, mae, mean_sales, n_points; LightGBM only),
  `worst_items.csv`, `feature_importance.csv`, `metrics.json`.

## 6. Order recommendation

Inputs in the sidebar: current stock on hand, lead time (1–4 weeks), service level
(P50 / P90). Recommendation = max(0, demand over lead time at chosen quantile −
stock). Show the arithmetic in plain words under the number ("Expected demand over
the next 2 weeks is 210 (P90: 265). You have 100. Order 165 to cover P90."). Allow
CSV download of recommendations for all products of the selected store.

## 7. App pages (Streamlit)

1. **Forecast** — selectors (store, item), chart: history (last 52 weeks), forecast,
   P10–P90 band, last year's same weeks as a dotted line; forecast table; order box.
2. **Model quality** — backtest table (baselines vs model), WAPE by fold chart,
   coverage, worst items; feature importance chart; a plain-language paragraph on
   what this means for the business.
3. **How it works** — short explanation with a diagram; link to DESIGN.md.
Keep the UI simple and fast; cache data loading with `st.cache_data`.
The app reads exactly these eight artifacts and nothing else (guarded by `require_artifacts()` in
`dashboard/common.py`): weekly.parquet, forecasts.parquet, backtest_summary.csv, backtest_by_fold.csv,
backtest_by_series.csv, worst_items.csv, feature_importance.csv, metrics.json.

## 8. Repository layout

```
demand-forecast/
  app.py                     # Streamlit entry
  demand_forecast/
    data.py (load/aggregate), features.py, models.py, backtest.py, recommend.py, config.py
  scripts/
    download_data.py, prepare.py, train.py (backtest + final models + artifacts), eval.py (prints metrics table)
  artifacts/                 # committed: weekly aggregates subset, forecasts, backtest, metrics (small parquet/json)
  tests/                     # feature leakage tests, backtest split tests, recommendation math
  docs/DESIGN.md, assets/demo.gif, README.md, requirements.txt, LICENSE, .github/workflows/ci.yml
```

## 9. Milestones

**M1 — data.** Download, weekly aggregation, sanity plots (saved to `docs/`), a
`tests/test_data.py` with shape/continuity checks (no missing weeks per series).
**M2 — features + baselines + backtest.** Feature builder with explicit leakage tests
(assert no feature at origin t uses data after t); baselines; backtest harness;
metrics table for baselines only.
**M3 — LightGBM models.** Point + quantile models, backtest table with model vs
baselines, coverage, importances; final training and artifacts.
**M4 — app.** Three pages, order recommendation, CSV download, runs locally.
**M5 — ship.** Deploy (Streamlit Community Cloud or HF Space with `sdk: streamlit`),
GIF, README with the backtest table, DESIGN.md with findings (what helped, what
didn't), CI, LICENSE.

## 10. Acceptance criteria

- Model WAPE beats both baselines on the backtest overall and on ≥ 5 of 6 folds;
  P10–P90 coverage between 70% and 90%.
- Leakage tests pass; backtest reproducible with a fixed seed.
- App loads in < 5 s on free hardware; switching store/item is instant.
- README table shows: baseline WAPEs, model WAPE, relative improvement, coverage.

## 11. Out of scope

Promotions/price data (not in the dataset — mention that real clients have it and it
usually helps), hierarchical reconciliation, daily granularity, inventory costs. List
as next steps.
