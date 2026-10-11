"""demand-forecast — Streamlit dashboard entry point (CLAUDE.md §7).

Three pages: Forecast, Model quality, How it works. Reads the eight precomputed artifacts only
(dashboard/common.py REQUIRED, incl. backtest_by_series.csv); no training or recomputation at runtime.
Run: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from dashboard.common import require_artifacts
from dashboard.forecast import forecast_page
from dashboard.how import how_page
from dashboard.quality import quality_page

st.set_page_config(page_title="demand-forecast", page_icon="📦", layout="wide")
require_artifacts()

pg = st.navigation(
    [
        st.Page(forecast_page, title="Forecast", icon="📈", url_path="forecast", default=True),
        st.Page(quality_page, title="Model quality", icon="✅", url_path="quality"),
        st.Page(how_page, title="How it works", icon="📖", url_path="how"),
    ]
)
pg.run()
