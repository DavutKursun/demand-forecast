"""Shared pieces of the dashboard: cached artifact loading, the no-artifacts guard, the chart
palette, and small pure helpers (no Streamlit calls) that the pages and tests both use."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from demand_forecast.config import ARTIFACTS, HORIZON

REQUIRED = (
    "weekly.parquet",
    "forecasts.parquet",
    "backtest_summary.csv",
    "backtest_by_fold.csv",
    "backtest_by_series.csv",
    "worst_items.csv",
    "feature_importance.csv",
    "metrics.json",
)

# Light-surface palette: categorical slots 1–2 for the two data series, grey for the reference
# line, hairline chrome. Validated with the dataviz palette checker (adjacent pairs pass).
PALETTE = {
    "actual": "#2a78d6",
    "forecast": "#eb6834",
    "band": "rgba(235, 104, 52, 0.14)",
    "last_year": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "ink": "#52514e",
    "surface": "#ffffff",
}

METHOD_LABELS = {
    "lightgbm": "LightGBM (this project)",
    "seasonal_naive": "Seasonal naive (same week last year)",
    "moving_average_8": "8-week moving average",
}


@dataclass(frozen=True)
class Artifacts:
    weekly: pd.DataFrame
    forecasts: pd.DataFrame  # with a last_year column added (actual 52 weeks before the target week)
    summary: pd.DataFrame
    by_fold: pd.DataFrame
    by_series: pd.DataFrame  # LightGBM backtest metrics per series: store, item, wape, mae, mean_sales, n_points
    worst: pd.DataFrame
    importance: pd.DataFrame
    metrics: dict

    @property
    def origin(self) -> pd.Timestamp:
        return pd.Timestamp(self.forecasts["origin"].iloc[0])


def missing_artifacts(root: Path = ARTIFACTS) -> list[str]:
    return [f for f in REQUIRED if not (root / f).exists()]


def require_artifacts() -> None:
    """Stop the app with a clear message if any of the seven artifacts is missing."""
    missing = missing_artifacts()
    if missing:
        st.error(
            "Missing artifacts in `artifacts/`: " + ", ".join(f"`{m}`" for m in missing)
            + ". Run `python scripts/prepare.py` and `python scripts/train.py` first (or `prepare.py --synthetic`)."
        )
        st.stop()


@st.cache_data(show_spinner="Loading forecasts…")
def load() -> Artifacts:
    weekly = pd.read_parquet(ARTIFACTS / "weekly.parquet")
    fc = add_last_year(pd.read_parquet(ARTIFACTS / "forecasts.parquet"), weekly)
    return Artifacts(
        weekly=weekly,
        forecasts=fc,
        summary=pd.read_csv(ARTIFACTS / "backtest_summary.csv"),
        by_fold=pd.read_csv(ARTIFACTS / "backtest_by_fold.csv", parse_dates=["origin"]),
        by_series=pd.read_csv(ARTIFACTS / "backtest_by_series.csv"),
        worst=pd.read_csv(ARTIFACTS / "worst_items.csv"),
        importance=pd.read_csv(ARTIFACTS / "feature_importance.csv"),
        metrics=json.loads((ARTIFACTS / "metrics.json").read_text()),
    )


# ---- pure helpers -------------------------------------------------------------------------


def series_slice(df: pd.DataFrame, store: int, item: int) -> pd.DataFrame:
    return df[(df["store"] == store) & (df["item"] == item)]


def add_last_year(fc: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    """Attach ``last_year``: the actual sold 52 weeks before each target week (NaN if unknown)."""
    ly = weekly[["store", "item", "week", "sales"]].copy()
    ly["target_week"] = ly["week"] + pd.Timedelta(weeks=52)
    ly = ly.drop(columns="week").rename(columns={"sales": "last_year"})
    return fc.merge(ly, on=["store", "item", "target_week"], how="left")


def forecast_table(f: pd.DataFrame) -> pd.DataFrame:
    """The on-screen table for one series: week, P10, forecast, P90, last year — whole units."""
    f = f.sort_values("horizon")
    t = pd.DataFrame(
        {
            "week": f["target_week"].dt.strftime("%Y-%m-%d").to_numpy(),
            "P10": f["q10"].to_numpy(),
            "forecast": f["yhat"].to_numpy(),
            "P90": f["q90"].to_numpy(),
            "last year": f["last_year"].to_numpy(),
        }
    )
    for c in ("P10", "forecast", "P90", "last year"):
        t[c] = t[c].round().astype("Int64")
    return t


def forecasts_csv(fc: pd.DataFrame) -> bytes:
    """CSV export (one row per series × horizon) with plain column names."""
    out = fc.sort_values(["store", "item", "horizon"]).rename(
        columns={"target_week": "week", "yhat": "forecast", "q10": "p10", "q90": "p90"}
    )
    cols = ["store", "item", "origin", "horizon", "week", "forecast", "p10", "p90", "last_year"]
    out = out[cols].copy()
    for c in ("origin", "week"):
        out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d")
    out[["forecast", "p10", "p90"]] = out[["forecast", "p10", "p90"]].round(1)
    return out.to_csv(index=False).encode()


def series_wape(by_series: pd.DataFrame, store: int, item: int) -> float | None:
    """Backtest WAPE for one series from backtest_by_series.csv (None if the series is not in it)."""
    hit = by_series[(by_series["store"] == store) & (by_series["item"] == item)]
    return float(hit["wape"].iloc[0]) if len(hit) else None


def is_worst_item(worst: pd.DataFrame, store: int, item: int) -> bool:
    return bool(((worst["store"] == store) & (worst["item"] == item)).any())


__all__ = ["ARTIFACTS", "HORIZON", "PALETTE", "METHOD_LABELS", "Artifacts", "load", "require_artifacts",
           "missing_artifacts", "series_slice", "add_last_year", "forecast_table", "forecasts_csv", "series_wape", "is_worst_item"]
