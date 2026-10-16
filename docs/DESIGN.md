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

## Findings (Milestone 3, real data)

Data: Kaggle Store Item Demand Forecasting, 10 stores × 50 items, daily 2013-01-01 → 2017-12-31,
aggregated to 260 Monday-anchored weeks per series (500 series, no gaps). Backtest: 6 origins every
4 weeks from 2017-07-10 to 2017-11-27, horizon 1–4 weeks, all 12 LightGBM models retrained per fold
on 413k rows. The full run (`scripts/train.py`) took 7.8 minutes on an Apple-silicon laptop, about
75 s per fold.

### Backtest

| Method | WAPE | MAE (units/week) | Bias (units/week) | P10–P90 coverage |
|---|---|---|---|---|
| **LightGBM** | **4.4%** | 18.6 | +0.7 | 76.1% |
| Seasonal naive (same week last year) | 6.4% | 26.8 | −17.1 | – |
| 8-week moving average | 11.9% | 50.1 | +39.4 | – |

LightGBM improves on the better baseline by 30.6% and wins every fold. The seasonal naive is
systematically low (bias −17) because sales grow year over year and last year's value does not know
that; the model corrects it through `year`, `lag_1` and the series mean and ends up nearly unbiased.

### Per fold

| Origin | LightGBM | Seasonal naive | 8-week MA |
|---|---|---|---|
| 2017-07-10 | 4.4% | 6.0% | 6.4% |
| 2017-08-07 | 4.2% | 6.1% | 11.6% |
| 2017-09-04 | 4.2% | 6.3% | 12.2% |
| 2017-10-02 | 4.4% | 6.2% | 7.8% |
| 2017-10-30 | 4.5% | 6.9% | 6.6% |
| 2017-11-27 | 5.2% | 7.1% | 33.7% |

LightGBM is stable at 4.2–5.2% across folds; the last fold (forecasting December) is the hardest for
every method. WAPE by horizon is almost flat (4.3%, 4.3%, 4.7%, 4.6% for 1–4 weeks ahead): four
weeks out is barely harder than one, because the predictive signal is annual, not recent.

### The moving-average collapse in autumn

This dataset's seasonality is a month-level step function. Weekly totals across all 500 series sit
on a plateau within each month and jump at the month boundary: about 233k/week in August, 217k in
September, 202k in October, 217k in November, then 155k in December, a 28% step down. From the
2017-11-27 origin the 8-week moving average has just averaged October and November (≈210k/week) and
forecasts December at that level, roughly 35% too high, hence its 33.7% WAPE on that fold. The same
mechanism gives it 11–12% in the August and September folds. Any method without annual memory does
this; the seasonal naive and LightGBM both know December from 2016. The business lesson: the
"average of recent weeks" rule looks fine in a flat stretch (6.4% in the July fold, close to the
seasonal naive) and fails exactly when the order decision matters most, at a season change. That is
why both baselines are always reported.

### Uncertainty

P10–P90 coverage is 76.1% against a nominal 80%, inside the 70–90% acceptance range, so no
adjustment was applied. It is consistent across horizons (74.7–77.3%) and folds (72.4–79.4%).
The misses are asymmetric: 13.5% of actuals fall below P10 and 10.4% above P90, so the upper
quantile is calibrated and the lower one sits slightly too high. The median band is 14–15% of the
actual value wide. If tighter calibration is ever needed, the cheapest fix is a conformal correction:
shift each quantile by the corresponding residual quantile measured on the backtest.

### Per store

LightGBM WAPE ranges from 4.0% (store 10) to 5.2% (store 7). Stores 5, 6 and 7 are the hardest for
every method; they are also the three lowest-volume stores (255–279 units per item per week versus
331–470 elsewhere), so the noise-to-signal ratio is simply higher there.

### Worst 10 items (LightGBM)

| Store | Item | WAPE | Mean weekly sales |
|---|---|---|---|
| 5 | 5 | 10.9% | 114 |
| 7 | 27 | 10.5% | 119 |
| 5 | 47 | 10.5% | 133 |
| 7 | 41 | 10.1% | 121 |
| 6 | 5 | 9.3% | 110 |
| 1 | 37 | 9.2% | 217 |
| 6 | 4 | 9.1% | 134 |
| 7 | 16 | 8.9% | 140 |
| 4 | 4 | 8.9% | 187 |
| 6 | 27 | 8.8% | 137 |

All ten are low-volume series (110–217 units a week; the median series sells 351) and seven of the
ten are in stores 5–7. Even the worst is at 11%, so there is no catastrophic tail. The dataset has
no intermittent, near-zero products (the smallest series averages 89 units a week); real catalogues
do, and that is where this kind of model needs extra care.

### What helped: features

Share of total LightGBM gain, summed over the 12 models:

| Feature | Share of gain |
|---|---|
| `same_week_last_year` | 46.2% |
| `lag_52` | 22.5% |
| `lag_1` | 10.9% |
| `series_mean` | 3.6% |
| `roll_mean_4` | 3.1% |
| `lag_2` | 2.7% |
| `woy_cos` | 1.8% |
| `year` | 1.6% |
| `woy_sin` | 1.5% |
| `lag_3` | 1.3% |
| 17 other features | 4.8% |

- **Annual seasonality is the model.** Features that look one year back or encode the time of year
  (`same_week_last_year`, `lag_52`, `roll_mean_52`, `roll_std_52`, week-of-year sin/cos, `month`)
  carry 74% of the gain. Recent-level features (lags 1–4, 4/8-week rolling mean and std, 8-week
  trend) carry 19%, mostly through `lag_1`: they set the current level, the annual features set the
  shape.
- **`series_mean` replaces the categorical ids.** Once the per-series mean is a feature, `item`
  (0.9%) and `store` (<0.1%) add almost nothing.
- **`year` carries the growth.** It is the feature that lets the model avoid the seasonal naive's
  under-forecast.
- **Holidays did not help** (<0.1% of gain): the US holiday count in the target week is nearly
  irrelevant here. The December drop is a monthly level shift, not a holiday effect.
- **Candidates to drop:** lags 8, 12 and 26 and all rolling-std features each contribute under 0.3%.
  Not removed yet; an ablation ("LightGBM without lags", as the spec suggests) is a next step.
- `horizon` has zero gain by construction: there is one model per horizon, so it is constant within
  each model.

## Next steps

1. **Conformal calibration of the interval.** Coverage is 76% against a nominal 80%, and the misses are
   one-sided: 13.5% of actuals fall below P10 versus 10.4% above P90. A split-conformal step would
   shift each quantile by the matching residual quantile measured on the backtest folds, per horizon,
   bringing coverage to the target without retraining. It also makes the P90-based safety stock
   defensible to an auditor: the band's hit rate is measured, not assumed.
2. **Price and promotion inputs.** The public dataset has neither, so promotions, markdowns and
   stock-outs are invisible to the model and to the backtest. Real clients have this data; adding
   planned promotions for the target week (and their lagged effect) is usually the single largest
   accuracy gain after seasonality, and it is also what turns the tool into a what-if planner.
3. **Cold start for new items.** The seasonal features need 52 weeks of history, so a new product
   has no forecast for its first year. A fallback that borrows the seasonal profile of a similar
   product (same category, store) scaled to the new item's first weeks, and hands over to the full
   model once history exists, would cover the gap.
4. **A classical third baseline: ETS/Theta via `statsforecast`.** The two rules of thumb are what a
   business would use; a tuned statistical model is what a forecasting team would use. Adding ETS and
   Theta per series to the same folds shows whether the gradient-boosting machinery earns its
   complexity or merely matches a 40-year-old method.
5. **Monthly retraining job.** `scripts/train.py` takes under 8 minutes, so a scheduled GitHub Action
   (or cron) can append the new month, retrain, rerun the 6-fold backtest and commit fresh artifacts,
   failing the job if WAPE regresses against the previous metrics.json or coverage leaves the 70–90%
   band. Drift is then caught on a schedule rather than noticed by a store manager.

Also worth doing when time allows: the "LightGBM without lags" ablation from the spec (the gain table
suggests lags 8, 12 and 26 and the rolling-std features could be dropped), and per-store P10–P90
coverage to see whether the slight under-coverage is concentrated in the low-volume stores.

## Known limitations

- No price/promotion data in the public dataset (real clients usually have it; it helps).
- Weekly granularity only; hierarchical reconciliation (store totals) not done.
- P10–P90 coverage on the real data is 76% against a nominal 80%, with the lower quantile
  slightly too high. Acceptable by the project's criteria; a conformal correction from backtest
  residuals is the next step if tighter calibration is needed.
