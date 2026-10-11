"""Project-wide constants. Kept small on purpose: everything else is a function argument."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"  # train.csv from Kaggle (not committed)
ARTIFACTS = ROOT / "artifacts"  # weekly aggregates, forecasts, backtest results (committed, small)

HORIZON = 4  # weeks ahead
LAGS = (1, 2, 3, 4, 8, 12, 26, 52)
ROLLING = (4, 8, 13, 52)
BACKTEST_FOLDS = 6
BACKTEST_STEP = 4  # weeks between fold origins
QUANTILES = (0.1, 0.9)
SEED = 7
COUNTRY = "US"  # holiday calendar for the Kaggle dataset
