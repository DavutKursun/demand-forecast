"""Feature engineering without leakage.

For a forecast origin week t (the last week whose sales are known) and horizon h, the
target is sales at t+h and every feature is computed from weeks ≤ t:

* lags: sales at t, t-1, ... (lag k = sales at t-k+1, so lag 1 is the latest known week)
* rolling mean/std over the last 4/8/13/52 known weeks
* same week last year *relative to the target week* (t+h-52) — known as long as h ≤ 52
* calendar of the target week: week-of-year (sin/cos), month, year, holiday count
* trend: slope of the last 8 known weeks
* series mean over the known history (a target encoding that only sees the past)
* store, item as categorical

``build_training_frame`` stacks (origin, horizon) pairs for every series; ``build_at_origin``
produces the feature rows for forecasting from one origin. Both use the same function so
training and inference cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COUNTRY, HORIZON, LAGS, ROLLING

FEATURE_COLS: list[str] = (
    [f"lag_{k}" for k in LAGS]
    + [f"roll_mean_{w}" for w in ROLLING]
    + [f"roll_std_{w}" for w in ROLLING]
    + ["same_week_last_year", "trend_8", "series_mean", "woy_sin", "woy_cos", "month", "year", "holidays", "horizon", "store", "item"]
)
CATEGORICAL = ["store", "item"]


def _holiday_counts(weeks: pd.DatetimeIndex, country: str = COUNTRY) -> dict:
    import holidays as hol

    years = sorted({w.year for w in weeks} | {w.year + 1 for w in weeks})
    cal = hol.country_holidays(country, years=years)
    out = {}
    for w in weeks:
        out[w] = sum(1 for d in pd.date_range(w, periods=7, freq="D") if d in cal)
    return out


def _series_features(sales: np.ndarray, origin_idx: int, h: int, weeks: pd.DatetimeIndex, hol: dict, store: int, item: int) -> dict | None:
    """Features for one series at origin index (inclusive) and horizon h. None if history is too short."""
    max_lag = max(max(LAGS), max(ROLLING))
    if origin_idx + 1 < max_lag:
        return None
    hist = sales[: origin_idx + 1]  # known weeks only
    target_idx = origin_idx + h
    f: dict = {}
    for k in LAGS:
        f[f"lag_{k}"] = float(hist[-k])
    for w in ROLLING:
        win = hist[-w:]
        f[f"roll_mean_{w}"] = float(win.mean())
        f[f"roll_std_{w}"] = float(win.std(ddof=0))
    ly = target_idx - 52
    f["same_week_last_year"] = float(sales[ly]) if 0 <= ly <= origin_idx else float("nan")
    x = np.arange(8)
    f["trend_8"] = float(np.polyfit(x, hist[-8:], 1)[0])
    f["series_mean"] = float(hist.mean())
    tw = weeks[origin_idx] + pd.Timedelta(weeks=h)
    woy = tw.isocalendar().week
    f["woy_sin"] = float(np.sin(2 * np.pi * woy / 52.18))
    f["woy_cos"] = float(np.cos(2 * np.pi * woy / 52.18))
    f["month"] = int(tw.month)
    f["year"] = int(tw.year)
    f["holidays"] = int(hol.get(tw, 0))
    f["horizon"] = h
    f["store"] = store
    f["item"] = item
    f["origin"] = weeks[origin_idx]
    f["target_week"] = tw
    return f


def build_training_frame(weekly: pd.DataFrame, last_origin: pd.Timestamp | None = None, horizons: tuple[int, ...] = tuple(range(1, HORIZON + 1))) -> pd.DataFrame:
    """Rows for every (series, origin, horizon) whose target week is ≤ the data end and origin ≤ last_origin.

    Only origins ≤ last_origin are used, and a target is included only if its week ≤ last_origin
    too — so a backtest fold trained with last_origin = T never sees any sales after T.
    """
    weeks = pd.DatetimeIndex(sorted(weekly["week"].unique()))
    all_weeks = pd.DatetimeIndex(list(weeks) + [weeks[-1] + pd.Timedelta(weeks=h) for h in range(1, HORIZON + 1)])
    hol = _holiday_counts(all_weeks)
    if last_origin is None:
        last_origin = weeks[-1]
    last_idx = int(np.searchsorted(weeks, last_origin))
    rows = []
    for (store, item), g in weekly.groupby(["store", "item"], sort=True):
        sales = g.sort_values("week")["sales"].to_numpy(dtype=float)
        for h in horizons:
            for origin_idx in range(0, last_idx + 1):
                target_idx = origin_idx + h
                if target_idx > last_idx:
                    break
                f = _series_features(sales, origin_idx, h, weeks, hol, int(store), int(item))
                if f is None:
                    continue
                f["y"] = float(sales[target_idx])
                rows.append(f)
    return pd.DataFrame(rows)


def build_at_origin(weekly: pd.DataFrame, origin: pd.Timestamp, horizons: tuple[int, ...] = tuple(range(1, HORIZON + 1))) -> pd.DataFrame:
    """Feature rows to forecast horizons 1..H from ``origin`` for every series (no targets)."""
    weeks = pd.DatetimeIndex(sorted(weekly["week"].unique()))
    all_weeks = pd.DatetimeIndex(list(weeks) + [weeks[-1] + pd.Timedelta(weeks=h) for h in range(1, HORIZON + 1)])
    hol = _holiday_counts(all_weeks)
    origin_idx = int(np.searchsorted(weeks, origin))
    if weeks[origin_idx] != origin:
        raise ValueError(f"origin {origin.date()} is not a week in the data")
    rows = []
    for (store, item), g in weekly.groupby(["store", "item"], sort=True):
        sales = g.sort_values("week")["sales"].to_numpy(dtype=float)
        for h in horizons:
            f = _series_features(sales, origin_idx, h, weeks, hol, int(store), int(item))
            if f is not None:
                rows.append(f)
    return pd.DataFrame(rows)
