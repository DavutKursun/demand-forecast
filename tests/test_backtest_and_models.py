import numpy as np
import pandas as pd
import pytest

from demand_forecast.backtest import actuals_after, bias, coverage, fold_origins, run_backtest, summarize, wape
from demand_forecast.baselines import moving_average, seasonal_naive
from demand_forecast.features import build_at_origin, build_training_frame
from demand_forecast.models import fit
from demand_forecast.recommend import recommend


def test_metrics_math():
    y, yhat = np.array([10, 20, 30]), np.array([12, 18, 30])
    assert wape(y, yhat) == pytest.approx(4 / 60)
    assert bias(y, yhat) == pytest.approx(0.0)
    assert coverage([5, 10, 15], [4, 11, 10], [6, 12, 20]) == pytest.approx(2 / 3)
    assert np.isnan(wape([0, 0], [1, 1]))


def test_fold_origins_spacing(weekly):
    weeks = pd.DatetimeIndex(sorted(weekly["week"].unique()))
    o = fold_origins(weeks, folds=3, step=4)
    assert len(o) == 3 and (o[1] - o[0]).days == 28 and o[-1] == weeks[-5]


def test_baselines(weekly):
    weeks = sorted(weekly["week"].unique())
    origin = weeks[-6]
    sn = seasonal_naive(weekly, origin)
    ma = moving_average(weekly, origin)
    s = weekly[(weekly.store == 1) & (weekly.item == 1)].sort_values("week").reset_index(drop=True)
    oi = s.index[s.week == origin][0]
    r = sn[(sn.store == 1) & (sn.item == 1) & (sn.horizon == 1)].iloc[0]
    assert r["yhat"] == s.loc[oi + 1 - 52, "sales"]
    m = ma[(ma.store == 1) & (ma.item == 1)]
    assert m["yhat"].nunique() == 1 and m["yhat"].iloc[0] == pytest.approx(s.loc[oi - 7 : oi, "sales"].mean())


def test_actuals_after(weekly):
    weeks = sorted(weekly["week"].unique())
    a = actuals_after(weekly, weeks[-5])
    assert set(a["horizon"]) == {1, 2, 3, 4} and len(a) == 4 * 4


def test_fit_predict_shapes_and_quantile_order(weekly):
    weeks = sorted(weekly["week"].unique())
    origin = weeks[-5]
    train = build_training_frame(weekly, last_origin=origin)
    models = fit(train.sample(frac=0.3, random_state=1))  # small for speed
    p = models.predict(build_at_origin(weekly, origin))
    assert len(p) == 4 * 4 and (p["yhat"] >= 0).all()
    assert (p["q10"] <= p["yhat"]).all() and (p["q90"] >= p["yhat"]).all()
    assert not models.feature_importance().empty


def test_backtest_runs_and_beats_nothing_silently(weekly):
    bt = run_backtest(weekly, folds=2, step=4, verbose=False)
    assert set(bt.summary["method"]) == {"seasonal_naive", "moving_average_8", "lightgbm"}
    assert bt.by_fold.shape[0] == 2
    lg = bt.summary.set_index("method").loc["lightgbm"]
    assert 0 < lg["wape"] < 1 and 0 <= lg["coverage_80"] <= 1


def test_summarize_handles_missing_quantiles():
    p = pd.DataFrame({"method": ["x"] * 3, "y": [1, 2, 3], "yhat": [1, 2, 4]})
    s = summarize(p)
    assert "coverage_80" not in s.columns and s.loc[0, "wape"] == pytest.approx(1 / 6)


def test_recommend_math():
    fc = pd.DataFrame({"horizon": [1, 2, 3, 4], "yhat": [100, 110, 120, 130], "q90": [130, 135, 140, 150]})
    r = recommend(fc, stock_on_hand=100, lead_time_weeks=2, service_level="P90")
    assert r.expected_demand == 210 and r.p90_demand == 265 and r.order_qty == 165
    assert "Order 165" in r.explanation
    r2 = recommend(fc, stock_on_hand=300, lead_time_weeks=2, service_level="P50")
    assert r2.order_qty == 0 and "No order needed" in r2.explanation
    with pytest.raises(ValueError):
        recommend(fc, 0, lead_time_weeks=5)
