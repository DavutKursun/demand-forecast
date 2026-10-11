#!/usr/bin/env python
"""Print the backtest results table from artifacts/ (what the README quotes)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demand_forecast.config import ARTIFACTS  # noqa: E402


def main() -> int:
    m = json.loads((ARTIFACTS / "metrics.json").read_text())
    rows = [
        ("Seasonal naive (same week last year)", m["wape_seasonal_naive"]),
        ("8-week moving average", m["wape_moving_average_8"]),
        ("LightGBM (this project)", m["wape_lightgbm"]),
    ]
    print(f"Backtest: {m['folds']} rolling-origin folds, {m['n_series']} series, horizon 4 weeks\n")
    print("| Method | WAPE |\n|---|---|")
    for name, w in rows:
        print(f"| {name} | {100 * w:.1f}% |")
    print(f"\nImprovement vs best baseline: {100 * m['improvement_vs_best_baseline']:.1f}%  ·  folds won: {m['folds_won']}/{m['folds']}  ·  "
          f"P10–P90 coverage: {100 * m['coverage_80']:.0f}%  ·  bias: {m['bias_lightgbm']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
