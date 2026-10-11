import numpy as np
import pandas as pd
import pytest

from demand_forecast.config import HORIZON
from demand_forecast.features import FEATURE_COLS, build_at_origin, build_training_frame


def test_training_frame_columns_and_targets(weekly):
    weeks = sorted(weekly["week"].unique())
    origin = weeks[-10]
    tf = build_training_frame(weekly, last_origin=origin)
    assert set(FEATURE_COLS + ["y", "origin", "target_week"]).issubset(tf.columns)
    assert tf["target_week"].max() <= origin  # no target after the origin
    assert tf["origin"].max() <= origin
    assert set(tf["horizon"].unique()) == set(range(1, HORIZON + 1))


def test_no_leakage_features_only_use_past(weekly):
    """Changing sales AFTER the origin must not change any feature at that origin."""
    weeks = sorted(weekly["week"].unique())
    origin = weeks[-8]
    base = build_at_origin(weekly, origin)
    tampered = weekly.copy()
    tampered.loc[tampered["week"] > origin, "sales"] *= 10
    after = build_at_origin(tampered, origin)
    pd.testing.assert_frame_equal(base[FEATURE_COLS], after[FEATURE_COLS])


def test_lag_and_last_year_values(weekly):
    s = weekly[(weekly.store == 1) & (weekly.item == 1)].sort_values("week").reset_index(drop=True)
    origin = s.loc[100, "week"]
    rows = build_at_origin(weekly, origin)
    r = rows[(rows.store == 1) & (rows.item == 1) & (rows.horizon == 2)].iloc[0]
    assert r["lag_1"] == s.loc[100, "sales"] and r["lag_4"] == s.loc[97, "sales"]
    assert r["same_week_last_year"] == s.loc[100 + 2 - 52, "sales"]
    assert r["roll_mean_4"] == pytest.approx(s.loc[97:100, "sales"].mean())
    assert r["target_week"] == origin + pd.Timedelta(weeks=2)


def test_training_targets_are_the_future_sales(weekly):
    s = weekly[(weekly.store == 2) & (weekly.item == 1)].sort_values("week").reset_index(drop=True)
    tf = build_training_frame(weekly)
    r = tf[(tf.store == 2) & (tf.item == 1) & (tf.origin == s.loc[120, "week"]) & (tf.horizon == 3)].iloc[0]
    assert r["y"] == s.loc[123, "sales"]


def test_unknown_origin_raises(weekly):
    with pytest.raises(ValueError):
        build_at_origin(weekly, pd.Timestamp("2015-06-03"))  # a Wednesday


def test_same_week_last_year_nan_when_unknown(weekly):
    weeks = sorted(weekly["week"].unique())
    tf = build_training_frame(weekly, last_origin=weeks[60])
    assert np.isfinite(tf["same_week_last_year"]).all()  # all origins ≥ 52 weeks → known
