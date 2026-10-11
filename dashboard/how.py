"""Page 3 — How it works: a plain-language explainer of data, features, model, backtest and the
order recommendation, with a figure for the direct multi-horizon idea and a 'what this is not' list."""

from __future__ import annotations

import streamlit as st

from .charts import direct_horizon_figure
from .common import load

DESIGN_URL = "https://github.com/DavutKursun/demand-forecast/blob/main/docs/DESIGN.md"

SECTIONS = {
    "Data": """
The model learns from five years of daily sales for {n_series} store-and-product combinations
(10 stores × 50 products), a public retail dataset from Kaggle. Daily sales are summed into
Monday-to-Sunday weeks, because ordering happens weekly and weekly totals are far less noisy than
daily ones. Only complete weeks are kept, and a week with no recorded sales counts as zero rather
than being dropped. The raw file is not shipped with the app; the weekly totals and every result on
these pages are precomputed and stored as small files that the dashboard reads in seconds.
""",
    "Features": """
For every product, store and forecast date, the model is given a compact description of the past:
sales in each of the last four weeks and 8, 12, 26 and 52 weeks ago; averages and spreads over the
last 4, 8, 13 and 52 weeks; what sold in the same week last year; the recent trend; the product's
typical weekly volume; the time of year; and the number of public holidays in the week being
forecast. Everything is computed strictly from data available on the forecast date, and an automated
test enforces this: it alters all later sales and checks that no input changes. The same code builds
the inputs for training and for live forecasts, so there is no gap between what the model learned
from and what it is given in use.
""",
    "Model": """
The forecaster is LightGBM, a gradient-boosted decision-tree model: a committee of many small
decision rules that together turn the inputs into a number, widely used in industry and in
forecasting competitions. There is one model for each distance ahead, so the one-week-ahead and
four-week-ahead forecasts are produced directly rather than by chaining guesses on guesses (figure
below). For each distance there are three models: one for the most likely value and two for a low
(P10) and a high (P90) estimate, so that about 8 weeks out of 10 the actual figure should fall
between them. All twelve models train in under eight minutes on a laptop; no GPU or cloud service
is needed.
""",
    "Backtest & recommendation": """
To check the model honestly we replay history: {folds} times we pretend it is a Monday in the
last months of the data, train only on what was known up to that day, forecast the next four
weeks for every series, and score the forecasts against what really sold. The same test is run
for two rules of thumb a business would otherwise use, "same week last year" and "average of the
last 8 weeks"; the model's error (WAPE {wape:.1f}%) beat both in {won} of {folds} rounds, and the
Model quality page shows every number. The order recommendation turns the forecast into an action:
demand over your lead time at the chosen service level (expected, or P90 for a safety margin),
minus what you have on hand, is what to order. The arithmetic is spelled out next to the number
so it can be checked, and the full forecast can be downloaded as CSV.
""",
}

NOT_LIST = """
#### What this is not

- **Not a causal model.** It extrapolates patterns in past sales; it does not explain why sales
  move or what a price change would do.
- **No price or promotion inputs.** The public dataset has none, so promotions, markdowns and
  stock-outs are invisible to it. Real clients usually have this data, and it helps.
- **No cold start for new products.** A series needs about a year of history for the seasonal
  inputs; brand-new items need a separate rule, such as borrowing a similar product's profile.
- **Weekly totals per store and product only.** No daily forecasts, no reconciliation to store or
  chain totals, no inventory-cost optimisation.
"""


def how_page() -> None:
    m = load().metrics
    st.title("How it works")
    st.caption("The method in plain language. Full design notes and findings are in DESIGN.md (link at the bottom).")
    fmt = dict(n_series=m["n_series"], folds=m["folds"], won=m["folds_won"], wape=100 * m["wape_lightgbm"])
    for title, body in SECTIONS.items():
        st.subheader(title)
        st.markdown(body.format(**fmt))
        if title == "Model":
            st.plotly_chart(direct_horizon_figure(), width="stretch")
            st.caption("Direct multi-horizon forecasting: from one origin, a separate model predicts each of the next "
                       "four weeks, and the P10–P90 band grows with the distance ahead. Illustration, not data.")
    st.markdown(NOT_LIST)
    st.markdown(f"Full design notes and findings: [docs/DESIGN.md on GitHub]({DESIGN_URL}).")
