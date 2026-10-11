from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demand_forecast.data import make_synthetic_daily, to_weekly  # noqa: E402


@pytest.fixture(scope="session")
def weekly() -> pd.DataFrame:
    return to_weekly(make_synthetic_daily(n_stores=2, n_items=2, start="2014-01-01", end="2017-12-31"))
