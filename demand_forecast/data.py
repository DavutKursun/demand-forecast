"""Load daily sales and aggregate to weekly series.

Input: Kaggle "Store Item Demand Forecasting Challenge" train.csv (date, store, item,
sales). Output: a tidy weekly frame with columns [store, item, week, sales], one row per
series-week, no gaps (missing weeks filled with 0 — the dataset has none, the test asserts).

Weeks are Monday-anchored (week = Monday date). Partial first/last weeks are dropped so
every week is a full 7 days.

A synthetic generator with the same shape exists for tests and for anyone without the
Kaggle data; it is NOT used for the published results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_daily(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "store", "item", "sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train.csv is missing columns: {sorted(missing)}")
    return df[["date", "store", "item", "sales"]].sort_values(["store", "item", "date"]).reset_index(drop=True)


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Sum daily sales into Monday-anchored weeks; drop partial edge weeks."""
    d = daily.copy()
    d["week"] = d["date"] - pd.to_timedelta(d["date"].dt.weekday, unit="D")
    counts = d.groupby("week")["date"].nunique()
    full_weeks = counts[counts == 7].index
    d = d[d["week"].isin(full_weeks)]
    weekly = d.groupby(["store", "item", "week"], as_index=False)["sales"].sum()
    weekly["week"] = pd.to_datetime(weekly["week"])
    return fill_gaps(weekly)


def fill_gaps(weekly: pd.DataFrame) -> pd.DataFrame:
    """Ensure every series covers every week in [min, max] (missing → 0)."""
    weeks = pd.date_range(weekly["week"].min(), weekly["week"].max(), freq="7D")
    series = weekly[["store", "item"]].drop_duplicates()
    grid = series.merge(pd.DataFrame({"week": weeks}), how="cross")
    out = grid.merge(weekly, on=["store", "item", "week"], how="left")
    out["sales"] = out["sales"].fillna(0).astype(float)
    return out.sort_values(["store", "item", "week"]).reset_index(drop=True)


def check_weekly(weekly: pd.DataFrame) -> list[str]:
    """Return a list of problems (empty = fine)."""
    problems = []
    if weekly.duplicated(["store", "item", "week"]).any():
        problems.append("duplicate series-week rows")
    n_weeks = weekly["week"].nunique()
    per_series = weekly.groupby(["store", "item"])["week"].nunique()
    if (per_series != n_weeks).any():
        problems.append("some series do not cover every week")
    if (weekly["sales"] < 0).any():
        problems.append("negative sales")
    if weekly["week"].dt.weekday.nunique() != 1:
        problems.append("weeks are not anchored on one weekday")
    return problems


def make_synthetic_daily(n_stores: int = 2, n_items: int = 3, start: str = "2013-01-01", end: str = "2017-12-31", seed: int = 7) -> pd.DataFrame:
    """Daily sales with yearly seasonality, weekly pattern, trend and noise (tests/demo only)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    rows = []
    doy = dates.dayofyear.values
    dow = dates.weekday.values
    t = np.arange(len(dates)) / 365.0
    for s in range(1, n_stores + 1):
        for i in range(1, n_items + 1):
            base = 20 + 10 * s + 5 * i
            season = 1 + 0.35 * np.sin(2 * np.pi * (doy - 100) / 365.25)
            weekly = 1 + 0.15 * (dow >= 5)
            trend = 1 + 0.06 * t
            lam = base * season * weekly * trend
            sales = rng.poisson(lam)
            rows.append(pd.DataFrame({"date": dates, "store": s, "item": i, "sales": sales}))
    return pd.concat(rows, ignore_index=True)


def subset(weekly: pd.DataFrame, stores: list[int] | None = None, items: list[int] | None = None) -> pd.DataFrame:
    out = weekly
    if stores:
        out = out[out["store"].isin(stores)]
    if items:
        out = out[out["item"].isin(items)]
    return out.reset_index(drop=True)
