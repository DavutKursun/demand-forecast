"""Order recommendation from a forecast with uncertainty.

order = max(0, demand over the lead time at the chosen service level − stock on hand)
where demand over L weeks = sum of the horizon-1..L forecasts (P50 = yhat, P90 = q90).
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import pandas as pd


@dataclass
class Recommendation:
    lead_time_weeks: int
    service_level: str  # "P50" | "P90"
    expected_demand: float
    p90_demand: float
    stock_on_hand: float
    order_qty: int
    explanation: str


def recommend(forecast: pd.DataFrame, stock_on_hand: float, lead_time_weeks: int = 2, service_level: str = "P90") -> Recommendation:
    """``forecast`` holds one series' rows with columns horizon, yhat, q90 (horizons 1..H)."""
    f = forecast.sort_values("horizon")
    if lead_time_weeks < 1 or lead_time_weeks > f["horizon"].max():
        raise ValueError(f"lead time must be between 1 and {int(f['horizon'].max())} weeks")
    window = f[f["horizon"] <= lead_time_weeks]
    expected = float(window["yhat"].sum())
    p90 = float(window["q90"].sum())
    target = p90 if service_level == "P90" else expected
    order = max(0, math.ceil(target - stock_on_hand))
    weeks = "week" if lead_time_weeks == 1 else f"{lead_time_weeks} weeks"
    expl = (
        f"Expected demand over the next {weeks} is {expected:.0f} units (P90: {p90:.0f}). "
        f"You have {stock_on_hand:.0f} on hand. "
        + (f"Order {order} to cover the {service_level} level." if order else f"No order needed: stock already covers the {service_level} level.")
    )
    return Recommendation(lead_time_weeks, service_level, expected, p90, float(stock_on_hand), order, expl)
