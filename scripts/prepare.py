#!/usr/bin/env python
"""Aggregate daily sales to weekly and save artifacts/weekly.parquet (+ a sanity report).

    python scripts/prepare.py                      # Kaggle data in data/raw/train.csv
    python scripts/prepare.py --synthetic          # generated data (tests/demo only)
    python scripts/prepare.py --stores 1,2 --items 1,2,3,4,5,6,7,8,9,10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demand_forecast.config import ARTIFACTS, RAW_DIR  # noqa: E402
from demand_forecast.data import check_weekly, load_daily, make_synthetic_daily, subset, to_weekly  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--stores", default="", help="comma-separated store ids to keep")
    ap.add_argument("--items", default="", help="comma-separated item ids to keep")
    ap.add_argument("--out", type=Path, default=ARTIFACTS / "weekly.parquet")
    a = ap.parse_args(argv)
    if a.synthetic:
        daily = make_synthetic_daily(n_stores=3, n_items=5)
        source = "synthetic"
    else:
        path = RAW_DIR / "train.csv"
        if not path.exists():
            print(f"{path} not found; run scripts/download_data.py or use --synthetic", file=sys.stderr)
            return 1
        daily = load_daily(path)
        source = str(path)
    weekly = to_weekly(daily)
    stores = [int(x) for x in a.stores.split(",") if x]
    items = [int(x) for x in a.items.split(",") if x]
    weekly = subset(weekly, stores, items)
    problems = check_weekly(weekly)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(a.out, index=False)
    n_series = weekly[["store", "item"]].drop_duplicates().shape[0]
    print(f"source: {source}\nseries: {n_series}  weeks: {weekly['week'].nunique()}  "
          f"{weekly['week'].min().date()} → {weekly['week'].max().date()}\nmean weekly sales: {weekly['sales'].mean():.1f}\n"
          f"problems: {problems or 'none'}\nsaved {a.out}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
