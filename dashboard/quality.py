"""Page 2 — Model quality: backtest table and verdict, WAPE by fold, headline metrics, per-series
error distribution, hardest series, feature importance, and a plain-language summary.
Everything is read from the artifacts; nothing is recomputed here beyond formatting."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .charts import importance_chart, wape_by_fold_chart, wape_histogram
from .common import METHOD_LABELS, Artifacts, load

HIGHLIGHT = "background-color: rgba(42, 120, 214, 0.14); font-weight: 600"


def best_baseline(metrics: dict) -> tuple[str, float]:
    sn, ma = metrics["wape_seasonal_naive"], metrics["wape_moving_average_8"]
    return ("seasonal_naive", sn) if sn <= ma else ("moving_average_8", ma)


def verdict(metrics: dict) -> str:
    name, base = best_baseline(metrics)
    points = 100 * (base - metrics["wape_lightgbm"])
    rel = 100 * metrics["improvement_vs_best_baseline"]
    return (f"LightGBM beats the best baseline ({METHOD_LABELS[name].split(' (')[0].lower()}) by {points:.1f} WAPE points "
            f"({rel:.1f}% relative) across {metrics['folds']} folds.")


def summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    s = summary.copy()
    s["method"] = s["method"].map(METHOD_LABELS).fillna(s["method"])
    s["wape"] = 100 * s["wape"]
    s["coverage_80"] = (100 * s["coverage_80"]).map(lambda v: f"{v:.1f}" if pd.notna(v) else "\u2013")
    return s.rename(columns={"method": "Method", "wape": "WAPE (%)", "mae": "MAE (units/week)",
                             "bias": "Bias (units/week)", "coverage_80": "P10–P90 coverage (%)"})


def summary_styler(summary: pd.DataFrame):
    t = summary_table(summary)
    lgb = t["Method"].str.startswith("LightGBM")
    return (t.style
            .apply(lambda row: [HIGHLIGHT if lgb.loc[row.name] else ""] * len(row), axis=1)
            .format({"WAPE (%)": "{:.1f}", "MAE (units/week)": "{:.1f}", "Bias (units/week)": "{:+.1f}"}))


def business_paragraph(metrics: dict, by_fold: pd.DataFrame, by_series: pd.DataFrame, worst: pd.DataFrame) -> str:
    """8–10 plain sentences for a store manager. Only 'WAPE' is technical."""
    m = metrics
    lgb, sn, ma = 100 * m["wape_lightgbm"], 100 * m["wape_seasonal_naive"], 100 * m["wape_moving_average_8"]
    origins = pd.to_datetime(by_fold["origin"])
    first, last = origins.min(), origins.max()
    rel = 100 * m["improvement_vs_best_baseline"]
    cov = round(100 * m["coverage_80"])
    ma_worst = by_fold.loc[by_fold["moving_average_8"].idxmax()]
    typical = by_series["mean_sales"].median()
    worst_mean = worst["mean_sales"].mean()
    worst_ratio = worst["wape"].mean() / m["wape_lightgbm"]
    worst_max = 100 * by_series["wape"].max()
    words = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve"}
    return (
        f"We tested the forecast the way it would be used in practice. "
        f"{words.get(m['folds'], m['folds'])} times we rolled the calendar back to a Monday between {first:%B} and {last:%B %Y}, let the model see "
        f"only what was known on that day, and asked it to predict the next four weeks for all {m['n_series']} store-and-product "
        f"combinations; then we compared those predictions with what actually sold. "
        f"The error measure is WAPE: total units missed divided by total units sold, so {lgb:.1f}% means that for every 1,000 units "
        f"sold the forecast was off by about {round(10 * lgb)} units one way or the other. "
        f"We ran the same test on the two rules of thumb a store would use anyway: “same as this week last year” "
        f"({sn:.1f}%) and “the average of the last 8 weeks” ({ma:.1f}%). "
        f"The model was more accurate than both rules in {m['folds_won']} of the {m['folds']} test rounds, and {rel:.0f}% more "
        f"accurate than the better rule. "
        f"It also gives a low and a high estimate for every week; the real number landed between them {cov} times out of 100, "
        f"close to the 80 we aim for, so the high estimate is a sensible safety buffer when ordering. "
        f"The 8-week-average rule did badly in the round starting {pd.Timestamp(ma_worst['origin']):%-d %B %Y} "
        f"({100 * ma_worst['moving_average_8']:.0f}% error): sales stepped down sharply in the following weeks, and an average of "
        f"recent weeks cannot see that coming, while the model knows the pattern from the year before. "
        f"The model is weakest on slow sellers: the {len(worst)} hardest series sell about {worst_mean:.0f} units a week against "
        f"{typical:.0f} for a typical one, and their error is about {worst_ratio:.1f} times the average, but even the worst stays "
        f"at {worst_max:.0f}%. "
        f"It knows nothing about promotions or price changes, because that information is not in this data, so it will be caught "
        f"out by any sale that is not a repeat of last year."
    )


def quality_page() -> None:
    a: Artifacts = load()
    m = a.metrics
    st.title("Model quality")
    st.caption(
        f"Rolling-origin backtest: {m['folds']} folds, {m['n_series']} series, horizon 1–4 weeks, all models retrained "
        f"per fold. Lower WAPE is better. Every number on this page is precomputed by scripts/train.py."
    )
    st.markdown(f"**{verdict(m)}**")
    st.dataframe(summary_styler(a.summary), hide_index=True, width="stretch")
    st.caption("WAPE = total absolute error / total sales. Bias > 0 means over-forecasting. P10–P90 coverage should be close to 80%.")

    c1, c2, c3, c4 = st.columns(4)
    cov = 100 * m["coverage_80"]
    c1.metric("P10–P90 coverage", f"{cov:.0f}%", delta=f"{cov - 80:+.0f} pts vs 80% target", delta_color="off")
    c2.metric("Folds won", f"{m['folds_won']} / {m['folds']}")
    c3.metric("Series", f"{m['n_series']:,}")
    c4.metric("Last week of data", m["last_week"])

    st.subheader("WAPE by fold origin")
    st.plotly_chart(wape_by_fold_chart(a.by_fold), width="stretch")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Error per series")
        st.plotly_chart(wape_histogram(a.by_series), width="stretch")
        st.caption(f"LightGBM WAPE for each of the {len(a.by_series)} series over all folds; dashed line = median.")
    with right:
        st.subheader(f"Hardest {len(a.worst)} series")
        w = a.worst.copy()
        w["wape"] = 100 * w["wape"]
        w = w.rename(columns={"store": "Store", "item": "Item", "wape": "WAPE (%)", "mean_sales": "Mean weekly sales"})
        st.dataframe(
            w, hide_index=True, width="stretch",
            column_config={"WAPE (%)": st.column_config.NumberColumn(format="%.1f"),
                           "Mean weekly sales": st.column_config.NumberColumn(format="%.0f")},
        )
        st.caption(
            f"These are low-volume series: they sell about {a.worst['mean_sales'].mean():.0f} units a week against a median of "
            f"{a.by_series['mean_sales'].median():.0f}, so week-to-week noise is a larger share of sales."
        )

    st.subheader("What the model relies on")
    st.plotly_chart(importance_chart(a.importance), width="stretch")
    st.caption("Top 15 features by LightGBM gain, summed over the 12 models. Same week last year and the 52-week lag dominate: "
               "the data's annual pattern is the main signal.")

    st.subheader("What this means for the business")
    st.markdown(business_paragraph(m, a.by_fold, a.by_series, a.worst))
