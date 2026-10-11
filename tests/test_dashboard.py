"""Dashboard helpers: pure functions in dashboard.common / dashboard.charts, plus an app smoke test."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dashboard.charts import forecast_chart
from dashboard.common import REQUIRED, add_last_year, forecast_table, forecasts_csv, missing_artifacts, series_wape


def _toy():
    weeks = pd.date_range("2016-01-04", periods=110, freq="7D")  # Mondays
    weekly = pd.DataFrame({"store": 1, "item": 1, "week": weeks, "sales": range(110)}).astype({"sales": float})
    origin = weeks[-1]
    fc = pd.DataFrame({
        "store": 1, "item": 1, "origin": origin, "horizon": [1, 2, 3, 4],
        "target_week": [origin + pd.Timedelta(weeks=h) for h in (1, 2, 3, 4)],
        "yhat": [110.4, 111.6, 112.5, 113.49], "q10": [100.0, 101.0, 102.0, 103.0], "q90": [120.0, 121.0, 122.0, 123.0],
    })
    return weekly, fc, origin


def test_add_last_year_aligns_52_weeks_back():
    weekly, fc, origin = _toy()
    out = add_last_year(fc, weekly)
    # sales are 0,1,2,... by week index; target week index is 110..113, so 52 weeks back is 58..61
    assert out["last_year"].tolist() == [58.0, 59.0, 60.0, 61.0]
    # unknown history → NaN, never an error
    out2 = add_last_year(fc, weekly[weekly["week"] < origin - pd.Timedelta(weeks=60)])
    assert out2["last_year"].isna().all()


def test_forecast_table_rounds_to_whole_units():
    weekly, fc, _ = _toy()
    t = forecast_table(add_last_year(fc, weekly))
    assert list(t.columns) == ["week", "P10", "forecast", "P90", "last year"]
    assert t["forecast"].tolist() == [110, 112, 112, 113]  # .5 rounds half to even
    assert str(t["forecast"].dtype) == "Int64"
    assert t["week"].iloc[0] == "2018-02-12"


def test_forecasts_csv_has_one_row_per_horizon():
    weekly, fc, _ = _toy()
    csv = forecasts_csv(add_last_year(fc, weekly)).decode().strip().splitlines()
    assert csv[0] == "store,item,origin,horizon,week,forecast,p10,p90,last_year"
    assert len(csv) == 1 + 4


def test_series_wape_only_for_published_items():
    worst = pd.DataFrame({"store": [5], "item": [5], "wape": [0.109], "mean_sales": [113.5]})
    assert series_wape(worst, 5, 5) == pytest.approx(0.109)
    assert series_wape(worst, 1, 1) is None


def test_missing_artifacts_lists_all_when_empty(tmp_path):
    assert missing_artifacts(tmp_path) == list(REQUIRED)


def test_forecast_chart_has_expected_traces():
    weekly, fc, origin = _toy()
    fig = forecast_chart(weekly, add_last_year(fc, weekly), origin)
    names = [t.name for t in fig.data]
    assert "Actual" in names and "Forecast" in names and "Same weeks last year" in names and "P10–P90 band" in names
    actual = next(t for t in fig.data if t.name == "Actual")
    assert len(actual.x) == 52  # last 52 weeks only
    forecast = next(t for t in fig.data if t.name == "Forecast")
    assert len(forecast.x) == 5 and pd.Timestamp(forecast.x[0]) == origin  # joined to the last actual


@pytest.mark.skipif(missing_artifacts(), reason="needs the committed artifacts/")
def test_app_runs_without_exceptions():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Order this week" in m.label for m in at.metric)


@pytest.mark.skipif(missing_artifacts(), reason="needs the committed artifacts/")
@pytest.mark.parametrize("module, func, title", [("quality", "quality_page", "Model quality"), ("how", "how_page", "How it works")])
def test_other_pages_run_without_exceptions(module, func, title):
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    src = f"import sys; sys.path.insert(0, {str(root)!r})\nfrom dashboard.{module} import {func}\n{func}()\n"
    at = AppTest.from_string(src, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.value for t in at.title] == [title]


# ---- eighth artifact: backtest_by_series.csv ---------------------------------------------------


def _toy_predictions():
    rows = []
    for method, offset in (("lightgbm", 1.0), ("seasonal_naive", 5.0)):
        for (s, i), base in (((1, 1), 100.0), ((1, 2), 20.0)):
            for h in (1, 2):
                rows.append({"store": s, "item": i, "origin": pd.Timestamp("2017-11-27"), "horizon": h,
                             "target_week": pd.Timestamp("2017-11-27") + pd.Timedelta(weeks=h),
                             "y": base, "yhat": base + offset, "method": method, "q10": base - 10, "q90": base + 10})
    return pd.DataFrame(rows)


def test_by_series_metrics_and_worst_items_agree():
    from demand_forecast.backtest import by_series, worst_items

    bs = by_series(_toy_predictions())
    assert list(bs.columns) == ["store", "item", "wape", "mae", "mean_sales", "n_points"]
    assert len(bs) == 2 and bs["n_points"].tolist() == [2, 2]
    s11 = bs[(bs.store == 1) & (bs.item == 1)].iloc[0]
    assert s11["wape"] == pytest.approx(0.01) and s11["mae"] == pytest.approx(1.0) and s11["mean_sales"] == 100.0
    s12 = bs[(bs.store == 1) & (bs.item == 2)].iloc[0]
    assert s12["wape"] == pytest.approx(0.05)  # same absolute error, smaller volume → larger WAPE
    w = worst_items(_toy_predictions(), n=1)
    assert w.iloc[0][["store", "item"]].tolist() == [1, 2] and w.iloc[0]["wape"] == pytest.approx(0.05)


def test_series_wape_from_by_series():
    bs = pd.DataFrame({"store": [1, 1], "item": [1, 2], "wape": [0.01, 0.05], "mae": [1.0, 1.0], "mean_sales": [100.0, 20.0], "n_points": [2, 2]})
    assert series_wape(bs, 1, 2) == pytest.approx(0.05)
    assert series_wape(bs, 9, 9) is None


def test_required_includes_by_series():
    assert "backtest_by_series.csv" in REQUIRED and len(REQUIRED) == 8


# ---- page 2 helpers ------------------------------------------------------------------------------

METRICS = {"n_series": 500, "weeks": 260, "last_week": "2017-12-25", "folds": 6, "wape_lightgbm": 0.0444,
           "wape_seasonal_naive": 0.064, "wape_moving_average_8": 0.1195, "improvement_vs_best_baseline": 0.3056,
           "coverage_80": 0.7608, "bias_lightgbm": 0.718, "folds_won": 6, "train_rows": 413000, "seconds": 468.9}


def _toy_quality_frames():
    by_fold = pd.DataFrame({"origin": pd.to_datetime(["2017-10-30", "2017-11-27"]), "lightgbm": [0.045, 0.052],
                            "moving_average_8": [0.066, 0.337], "seasonal_naive": [0.069, 0.071]})
    by_series = pd.DataFrame({"store": [1, 1, 2], "item": [1, 2, 1], "wape": [0.03, 0.05, 0.10], "mae": [3.0, 2.0, 5.0],
                              "mean_sales": [400.0, 300.0, 100.0], "n_points": [24, 24, 24]})
    worst = pd.DataFrame({"store": [2], "item": [1], "wape": [0.10], "mean_sales": [100.0]})
    summary = pd.DataFrame({"method": ["lightgbm", "seasonal_naive", "moving_average_8"], "wape": [0.0444, 0.064, 0.1195],
                            "mae": [18.6, 26.8, 50.1], "bias": [0.7, -17.1, 39.4], "coverage_80": [0.7608, None, None]})
    importance = pd.DataFrame({"feature": [f"f{i}" for i in range(20)], "gain": [float(100 - 5 * i) for i in range(20)]})
    return by_fold, by_series, worst, summary, importance


def test_verdict_text():
    from dashboard.quality import verdict

    assert verdict(METRICS) == "LightGBM beats the best baseline (seasonal naive) by 2.0 WAPE points (30.6% relative) across 6 folds."


def test_summary_table_and_styler():
    from dashboard.quality import summary_styler, summary_table

    _, _, _, summary, _ = _toy_quality_frames()
    t = summary_table(summary)
    assert list(t.columns) == ["Method", "WAPE (%)", "MAE (units/week)", "Bias (units/week)", "P10–P90 coverage (%)"]
    assert t["WAPE (%)"].tolist() == pytest.approx([4.44, 6.4, 11.95])
    html = summary_styler(summary).to_html()
    assert "LightGBM (this project)" in html and "rgba(42, 120, 214, 0.14)" in html


def test_business_paragraph_is_plain_and_8_to_10_sentences():
    from dashboard.quality import business_paragraph

    by_fold, by_series, worst, _, _ = _toy_quality_frames()
    text = business_paragraph(METRICS, by_fold, by_series, worst)
    sentences = [s for s in text.replace("%.", "%|").split(". ") if s.strip()]
    assert 8 <= len(sentences) <= 10, len(sentences)
    assert "WAPE" in text and "4.4%" in text and "6 of the 6" in text and "27 November 2017" in text
    for jargon in ("quantile", "P10", "P90", "LightGBM", "gradient", "lag", "feature"):
        assert jargon not in text, jargon


def test_quality_charts_build():
    from dashboard.charts import importance_chart, wape_by_fold_chart, wape_histogram

    by_fold, by_series, _, _, importance = _toy_quality_frames()
    f1 = wape_by_fold_chart(by_fold)
    assert [t.name for t in f1.data] == ["LightGBM", "Seasonal naive", "8-week MA"]
    assert f1.data[0].y[1] == pytest.approx(5.2) and f1.layout.yaxis.title.text == "WAPE (%)"
    f2 = wape_histogram(by_series)
    assert len(f2.data) == 1 and len(f2.layout.shapes) == 1 and "median 5.0%" in f2.layout.annotations[0].text
    f3 = importance_chart(importance, top=15)
    assert len(f3.data[0].y) == 15 and f3.data[0].y[-1] == "f0"  # largest at the top
    assert sum(f3.data[0].x) < 100


@pytest.mark.skipif(missing_artifacts(), reason="needs the committed artifacts/")
def test_quality_page_renders_all_sections():
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    src = f"import sys; sys.path.insert(0, {str(root)!r})\nfrom dashboard.quality import quality_page\nquality_page()\n"
    at = AppTest.from_string(src, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [m.label for m in at.metric] == ["P10–P90 coverage", "Folds won", "Series", "Last week of data"]
    assert len(at.dataframe) == 2  # summary + worst items
    assert len(at.get("plotly_chart")) == 3
    assert any("beats the best baseline" in md.value for md in at.markdown)
    assert any("WAPE" in md.value and "rules of thumb" in md.value for md in at.markdown)


@pytest.mark.skipif(missing_artifacts(), reason="needs the committed artifacts/")
def test_forecast_caption_shows_series_and_portfolio_wape():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
    caps = [c.value for c in at.caption]
    assert any(c.startswith("Backtest WAPE for this series:") and "(portfolio: " in c for c in caps), caps


# ---- page 3 ---------------------------------------------------------------------------------------


def test_direct_horizon_figure_builds():
    from dashboard.charts import direct_horizon_figure

    fig = direct_horizon_figure()
    names = [t.name for t in fig.data if t.name]
    assert any("history" in n for n in names) and any("one model per week" in n for n in names) and any("P10–P90" in n for n in names)
    arrows = [a for a in fig.layout.annotations if a.showarrow]
    assert len(arrows) == 4  # one arrow per horizon


@pytest.mark.skipif(missing_artifacts(), reason="needs the committed artifacts/")
def test_how_page_renders_four_sections_and_one_figure():
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    src = f"import sys; sys.path.insert(0, {str(root)!r})\nfrom dashboard.how import how_page\nhow_page()\n"
    at = AppTest.from_string(src, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [s.value for s in at.subheader] == ["Data", "Features", "Model", "Backtest & recommendation"]
    assert len(at.get("plotly_chart")) == 1
    text = " ".join(md.value for md in at.markdown)
    assert "What this is not" in text and "github.com/DavutKursun/demand-forecast/blob/main/docs/DESIGN.md" in text
