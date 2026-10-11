# demand-forecast — design notes

## Pipeline

```
daily sales ─► weekly (Monday weeks, full weeks only, gaps = 0)
            ─► features at origin t for horizon h (lags, rolling stats, same week last year,
               calendar of t+h, holidays, 8-week trend, series mean, store/item)
            ─► LightGBM per horizon: L1 point model + P10/P90 quantile models
            ─► rolling-origin backtest vs seasonal-naive and 8-week moving average
            ─► final models on all data → 4-week forecasts → Streamlit dashboard + order recommendation
```

## Decisions

- **Direct multi-horizon** (one model per h) instead of recursive: no error compounding,
  and the same feature function serves training and inference (`features.py`), so they
  cannot drift apart.
- **L1 objective** for the point model: aligned with WAPE and robust to spikes.
- **Quantile models** for P10/P90 rather than a parametric interval: cheap and honest;
  the recommendation uses P90 as safety stock. Backtest reports the realised coverage.
- **Leakage tests, not promises.** `test_no_leakage_features_only_use_past` multiplies every
  sale after the origin by 10 and asserts that no feature changes; the backtest asserts
  that every training target is ≤ the fold origin.
- **Baselines first.** The backtest table always shows seasonal naive and the moving
  average; the README reports the model's improvement over the *better* of the two.
- **App reads artifacts only.** Training is a script (minutes on a laptop); the dashboard
  starts in seconds on free hosting.

## Findings (fill in Milestone 3)

_Backtest table on the real data, per-fold results, coverage, worst items, what helped
(which features) and what did not._

## Known limitations

- No price/promotion data in the public dataset (real clients usually have it; it helps).
- Weekly granularity only; hierarchical reconciliation (store totals) not done.
- Interval coverage on synthetic data was ~62% (narrow); to be checked and, if needed,
  widened (quantile alpha or conformal adjustment) on the real data.
