import pandas as pd

from demand_forecast.data import check_weekly, fill_gaps, make_synthetic_daily, to_weekly


def test_weekly_shape_and_continuity(weekly):
    assert check_weekly(weekly) == []
    assert set(weekly.columns) == {"store", "item", "week", "sales"}
    assert weekly["week"].dt.weekday.unique().tolist() == [0]  # Mondays
    per = weekly.groupby(["store", "item"])["week"].nunique()
    assert per.nunique() == 1 and per.iloc[0] > 200


def test_partial_edge_weeks_dropped():
    d = make_synthetic_daily(1, 1, start="2016-01-06", end="2016-03-30")  # Wed → Wed
    w = to_weekly(d)
    assert w["week"].min() == pd.Timestamp("2016-01-11") and w["week"].max() == pd.Timestamp("2016-03-21")


def test_weekly_sum_matches_daily():
    d = make_synthetic_daily(1, 1, start="2016-01-04", end="2016-01-31")  # 4 full weeks
    w = to_weekly(d)
    assert len(w) == 4 and w["sales"].sum() == d["sales"].sum()


def test_fill_gaps_adds_missing_weeks():
    w = pd.DataFrame({"store": [1, 1], "item": [1, 1], "week": pd.to_datetime(["2016-01-04", "2016-01-18"]), "sales": [5.0, 7.0]})
    f = fill_gaps(w)
    assert len(f) == 3 and f.loc[1, "sales"] == 0.0
    assert check_weekly(f) == []
