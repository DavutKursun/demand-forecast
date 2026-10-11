"""The two baselines every model must beat.

* seasonal_naive: forecast for week t+h = sales at t+h-52 (same week last year)
* moving_average: forecast for every horizon = mean of the last 8 known weeks
Both produce a frame [store, item, origin, horizon, target_week, yhat].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HORIZON


def _origins_index(weekly: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(weekly["week"].unique()))


def seasonal_naive(weekly: pd.DataFrame, origin: pd.Timestamp, horizons=tuple(range(1, HORIZON + 1))) -> pd.DataFrame:
    weeks = _origins_index(weekly)
    oi = int(np.searchsorted(weeks, origin))
    rows = []
    for (s, i), g in weekly.groupby(["store", "item"], sort=True):
        sales = g.sort_values("week")["sales"].to_numpy(dtype=float)
        for h in horizons:
            ly = oi + h - 52
            if ly < 0:
                continue
            rows.append({"store": s, "item": i, "origin": origin, "horizon": h, "target_week": origin + pd.Timedelta(weeks=h), "yhat": float(sales[ly])})
    return pd.DataFrame(rows)


def moving_average(weekly: pd.DataFrame, origin: pd.Timestamp, window: int = 8, horizons=tuple(range(1, HORIZON + 1))) -> pd.DataFrame:
    weeks = _origins_index(weekly)
    oi = int(np.searchsorted(weeks, origin))
    rows = []
    for (s, i), g in weekly.groupby(["store", "item"], sort=True):
        sales = g.sort_values("week")["sales"].to_numpy(dtype=float)
        if oi + 1 < window:
            continue
        m = float(sales[oi + 1 - window : oi + 1].mean())
        for h in horizons:
            rows.append({"store": s, "item": i, "origin": origin, "horizon": h, "target_week": origin + pd.Timedelta(weeks=h), "yhat": m})
    return pd.DataFrame(rows)
