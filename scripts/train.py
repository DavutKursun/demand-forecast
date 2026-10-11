#!/usr/bin/env python
"""Backtest baselines vs LightGBM, then train final models and write forecasts + metrics.

    python scripts/train.py                 # uses artifacts/weekly.parquet
    python scripts/train.py --folds 3       # quicker

Outputs (artifacts/): backtest_predictions.parquet (not committed), backtest_summary.csv, backtest_by_fold.csv,
backtest_by_series.csv, worst_items.csv, feature_importance.csv, forecasts.parquet, metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demand_forecast.backtest import by_series, run_backtest, worst_items  # noqa: E402
from demand_forecast.config import ARTIFACTS, BACKTEST_FOLDS, BACKTEST_STEP  # noqa: E402
from demand_forecast.features import build_at_origin, build_training_frame  # noqa: E402
from demand_forecast.models import fit  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weekly", type=Path, default=ARTIFACTS / "weekly.parquet")
    ap.add_argument("--folds", type=int, default=BACKTEST_FOLDS)
    ap.add_argument("--step", type=int, default=BACKTEST_STEP)
    ap.add_argument("--out", type=Path, default=ARTIFACTS)
    a = ap.parse_args(argv)
    weekly = pd.read_parquet(a.weekly)
    t0 = time.perf_counter()
    bt = run_backtest(weekly, folds=a.folds, step=a.step)
    print("\n" + bt.summary.to_string(index=False))
    print("\n" + bt.by_fold.to_string(index=False))
    a.out.mkdir(parents=True, exist_ok=True)
    bt.predictions.to_parquet(a.out / "backtest_predictions.parquet", index=False)
    bt.summary.to_csv(a.out / "backtest_summary.csv", index=False)
    bt.by_fold.to_csv(a.out / "backtest_by_fold.csv", index=False)
    worst_items(bt.predictions).to_csv(a.out / "worst_items.csv", index=False)
    by_series(bt.predictions).to_csv(a.out / "backtest_by_series.csv", index=False)

    # final models on everything, forecast the next 4 weeks
    origin = weekly["week"].max()
    train = build_training_frame(weekly)
    models = fit(train)
    fc = models.predict(build_at_origin(weekly, origin))
    fc.to_parquet(a.out / "forecasts.parquet", index=False)
    models.feature_importance().to_csv(a.out / "feature_importance.csv", index=False)

    s = bt.summary.set_index("method")
    lg, sn, ma = s.loc["lightgbm"], s.loc["seasonal_naive"], s.loc["moving_average_8"]
    folds_won = int(((bt.by_fold["lightgbm"] < bt.by_fold["seasonal_naive"]) & (bt.by_fold["lightgbm"] < bt.by_fold["moving_average_8"])).sum())
    metrics = {
        "n_series": int(weekly[["store", "item"]].drop_duplicates().shape[0]),
        "weeks": int(weekly["week"].nunique()),
        "last_week": str(origin.date()),
        "folds": a.folds,
        "wape_lightgbm": round(float(lg["wape"]), 4),
        "wape_seasonal_naive": round(float(sn["wape"]), 4),
        "wape_moving_average_8": round(float(ma["wape"]), 4),
        "improvement_vs_best_baseline": round(float(1 - lg["wape"] / min(sn["wape"], ma["wape"])), 4),
        "coverage_80": round(float(lg.get("coverage_80", float("nan"))), 4),
        "bias_lightgbm": round(float(lg["bias"]), 3),
        "folds_won": folds_won,
        "train_rows": int(len(train)),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (a.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\n" + json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
