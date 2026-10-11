"""Page 1 — Forecast: one series' history, 4-week forecast with band, and the order recommendation."""

from __future__ import annotations

import random

import streamlit as st

from demand_forecast.recommend import recommend

from .charts import forecast_chart
from .common import HORIZON, forecast_table, forecasts_csv, is_worst_item, load, series_slice, series_wape


def _randomize(stores: list[int], items: list[int]) -> None:
    st.session_state["store"] = random.choice(stores)
    st.session_state["item"] = random.choice(items)


@st.cache_data(show_spinner=False)
def _all_series_csv(origin_key: str) -> bytes:  # origin_key only keys the cache
    return forecasts_csv(load().forecasts)


def forecast_page() -> None:
    a = load()
    weekly, fc, m, origin = a.weekly, a.forecasts, a.metrics, a.origin
    stores = sorted(int(s) for s in weekly["store"].unique())
    items = sorted(int(i) for i in weekly["item"].unique())

    with st.sidebar:
        st.subheader("Series")
        st.button("🎲 Random item", on_click=_randomize, args=(stores, items), width="stretch")
        store = st.selectbox("Store", stores, key="store")
        item = st.selectbox("Item", items, key="item")
        st.subheader("Order inputs")
        stock = st.number_input("Stock on hand (units)", min_value=0, value=0, step=10)
        lead = st.slider("Lead time (weeks)", min_value=1, max_value=HORIZON, value=2)
        level = st.radio("Service level", ["P90", "P50"], horizontal=True,
                         help="P90 covers demand in 9 weeks out of 10 (safety stock from the upper band). "
                              "P50 covers the expected demand only.")

    best_baseline = min(m["wape_seasonal_naive"], m["wape_moving_average_8"])
    st.title(f"Store {store} · Item {item}")
    st.caption(
        f"{m['n_series']} series · history to {m['last_week']} · forecast origin {origin:%Y-%m-%d} · "
        f"backtest WAPE {100 * m['wape_lightgbm']:.1f}% vs best baseline {100 * best_baseline:.1f}% "
        f"({m['folds']} folds, {m['folds_won']} won)"
    )

    hist = series_slice(weekly, store, item).sort_values("week")
    f = series_slice(fc, store, item).sort_values("horizon")
    st.plotly_chart(forecast_chart(hist, f, origin), width="stretch")

    rec = recommend(f, stock, lead, level)
    c1, c2 = st.columns([1, 3])
    c1.metric("Order this week", f"{rec.order_qty} units", help=f"{level} demand over {lead} week(s) minus stock on hand")
    c2.markdown("&nbsp;")  # align with the metric label
    c2.markdown(f"**{rec.explanation}**")

    st.subheader("Forecast")
    table = forecast_table(f)
    st.dataframe(table, hide_index=True, width="stretch")

    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ Download forecast CSV (this series)",
        data=forecasts_csv(f), file_name=f"forecast_store{store}_item{item}.csv", mime="text/csv",
        width="stretch",
    )
    d2.download_button(
        f"⬇️ Download forecast CSV (all {m['n_series']} series)",
        data=_all_series_csv(str(origin)), file_name=f"forecasts_{origin:%Y-%m-%d}.csv", mime="text/csv",
        width="stretch",
    )

    w = series_wape(a.by_series, store, item)
    if w is not None:
        note = f" One of the {len(a.worst)} hardest series in the backtest." if is_worst_item(a.worst, store, item) else ""
        st.caption(f"Backtest WAPE for this series: {100 * w:.1f}% (portfolio: {100 * m['wape_lightgbm']:.1f}%).{note}")
