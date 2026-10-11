"""demand-forecast — Streamlit dashboard (Milestone 4 fills this in; see CLAUDE.md §7).

Reads precomputed artifacts only: artifacts/weekly.parquet, forecasts.parquet,
backtest_summary.csv, backtest_by_fold.csv, worst_items.csv, feature_importance.csv, metrics.json.
Run: streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from demand_forecast.config import ARTIFACTS, HORIZON
from demand_forecast.recommend import recommend

st.set_page_config(page_title="demand-forecast", layout="wide")


@st.cache_data
def load():
    weekly = pd.read_parquet(ARTIFACTS / "weekly.parquet")
    fc = pd.read_parquet(ARTIFACTS / "forecasts.parquet")
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    return weekly, fc, metrics


if not (ARTIFACTS / "forecasts.parquet").exists():
    st.error("No artifacts found. Run `python scripts/prepare.py` and `python scripts/train.py` first.")
    st.stop()

weekly, fc, metrics = load()
st.title("Weekly demand forecast")
st.caption(f"{metrics['n_series']} series · data to {metrics['last_week']} · LightGBM WAPE {100*metrics['wape_lightgbm']:.1f}% "
           f"vs best baseline {100*min(metrics['wape_seasonal_naive'], metrics['wape_moving_average_8']):.1f}%")

col1, col2 = st.columns([3, 1])
with col2:
    store = st.selectbox("Store", sorted(weekly["store"].unique()))
    item = st.selectbox("Item", sorted(weekly["item"].unique()))
    stock = st.number_input("Stock on hand", min_value=0, value=0, step=1)
    lead = st.slider("Lead time (weeks)", 1, HORIZON, 2)
    level = st.radio("Service level", ["P90", "P50"], horizontal=True)

hist = weekly[(weekly.store == store) & (weekly.item == item)].sort_values("week").tail(52)
f = fc[(fc.store == store) & (fc.item == item)].sort_values("horizon")

with col1:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["week"], y=hist["sales"], name="Actual", mode="lines"))
    fig.add_trace(go.Scatter(x=f["target_week"], y=f["q90"], name="P90", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=f["target_week"], y=f["q10"], name="P10–P90", fill="tonexty", line=dict(width=0), fillcolor="rgba(31,119,180,0.2)"))
    fig.add_trace(go.Scatter(x=f["target_week"], y=f["yhat"], name="Forecast", mode="lines+markers"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    r = recommend(f, stock, lead, level)
    st.metric("Order this week", f"{r.order_qty} units")
    st.write(r.explanation)
    st.dataframe(f[["target_week", "q10", "yhat", "q90"]].rename(columns={"target_week": "week", "yhat": "forecast"}).round(0), hide_index=True)

st.divider()
st.subheader("Model quality (rolling-origin backtest)")
summary = pd.read_csv(ARTIFACTS / "backtest_summary.csv")
st.dataframe(summary.round(3), hide_index=True)
st.caption("WAPE = total absolute error / total sales. Lower is better. P10–P90 coverage should be close to 80%.")
