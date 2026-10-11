"""Rolling-origin backtest and metrics.

Folds: the last ``folds`` origins spaced ``step`` weeks apart, the latest leaving exactly
HORIZON weeks of actuals after it. For each origin T: baselines use data ≤ T; the model is
trained on a frame built with last_origin = T (so every target ≤ T) and predicts T+1..T+H.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .baselines import moving_average, seasonal_naive
from .config import BACKTEST_FOLDS, BACKTEST_STEP, HORIZON
from .features import build_at_origin, build_training_frame
from .models import fit


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.abs(y).sum()
    return float(np.abs(y - yhat).sum() / denom) if denom else float("nan")


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def bias(y, yhat) -> float:
    """Mean (forecast − actual): positive = over-forecasting."""
    return float(np.mean(np.asarray(yhat, float) - np.asarray(y, float)))


def coverage(y, lo, hi) -> float:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def fold_origins(weeks: pd.DatetimeIndex, folds: int = BACKTEST_FOLDS, step: int = BACKTEST_STEP, horizon: int = HORIZON) -> list[pd.Timestamp]:
    last = len(weeks) - 1 - horizon
    idx = [last - k * step for k in range(folds)][::-1]
    if idx[0] < 60:
        raise ValueError("not enough history for the requested folds")
    return [weeks[i] for i in idx]


def actuals_after(weekly: pd.DataFrame, origin: pd.Timestamp, horizon: int = HORIZON) -> pd.DataFrame:
    rows = []
    for h in range(1, horizon + 1):
        tw = origin + pd.Timedelta(weeks=h)
        a = weekly[weekly["week"] == tw][["store", "item", "sales"]].rename(columns={"sales": "y"})
        a["horizon"] = h
        a["target_week"] = tw
        rows.append(a)
    return pd.concat(rows, ignore_index=True)


@dataclass
class BacktestResult:
    predictions: pd.DataFrame  # one row per (method, fold origin, series, horizon)
    summary: pd.DataFrame  # per method: WAPE, MAE, bias, coverage
    by_fold: pd.DataFrame  # per method × origin: WAPE


def run_backtest(weekly: pd.DataFrame, folds: int = BACKTEST_FOLDS, step: int = BACKTEST_STEP, fit_fn=fit, verbose: bool = True) -> BacktestResult:
    weeks = pd.DatetimeIndex(sorted(weekly["week"].unique()))
    origins = fold_origins(weeks, folds, step)
    preds = []
    for k, origin in enumerate(origins, 1):
        actual = actuals_after(weekly, origin)
        for name, fn in (("seasonal_naive", seasonal_naive), ("moving_average_8", moving_average)):
            p = fn(weekly, origin).merge(actual, on=["store", "item", "horizon", "target_week"])
            p["method"] = name
            preds.append(p)
        train = build_training_frame(weekly, last_origin=origin)
        assert train["target_week"].max() <= origin, "leakage: a training target is after the origin"
        models = fit_fn(train)
        rows = build_at_origin(weekly, origin)
        p = models.predict(rows).merge(actual, on=["store", "item", "horizon", "target_week"])
        p["method"] = "lightgbm"
        preds.append(p)
        if verbose:
            print(f"fold {k}/{len(origins)} origin {origin.date()}: lightgbm WAPE {wape(p['y'], p['yhat']):.3f}", flush=True)
    predictions = pd.concat(preds, ignore_index=True)
    return BacktestResult(predictions, summarize(predictions), by_fold(predictions))


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in predictions.groupby("method"):
        r = {"method": m, "wape": wape(g["y"], g["yhat"]), "mae": mae(g["y"], g["yhat"]), "bias": bias(g["y"], g["yhat"])}
        if "q10" in g and g["q10"].notna().any():
            r["coverage_80"] = coverage(g["y"], g["q10"], g["q90"])
        rows.append(r)
    return pd.DataFrame(rows).sort_values("wape").reset_index(drop=True)


def by_fold(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = [{"method": m, "origin": o, "wape": wape(g["y"], g["yhat"])} for (m, o), g in predictions.groupby(["method", "origin"])]
    return pd.DataFrame(rows).pivot(index="origin", columns="method", values="wape").reset_index()


def worst_items(predictions: pd.DataFrame, method: str = "lightgbm", n: int = 10) -> pd.DataFrame:
    g = predictions[predictions["method"] == method]
    rows = [{"store": s, "item": i, "wape": wape(x["y"], x["yhat"]), "mean_sales": x["y"].mean()} for (s, i), x in g.groupby(["store", "item"])]
    return pd.DataFrame(rows).sort_values("wape", ascending=False).head(n).reset_index(drop=True)
