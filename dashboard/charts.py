"""Plotly figures for the dashboard. Pure functions: DataFrames in, Figure out."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .common import PALETTE


def forecast_chart(hist: pd.DataFrame, f: pd.DataFrame, origin: pd.Timestamp) -> go.Figure:
    """History (last 52 weeks), forecast with P10–P90 band, last year's same weeks, origin marker.

    ``hist`` has columns week, sales; ``f`` has target_week, yhat, q10, q90, last_year (horizons 1..H).
    The forecast line and band start at the last actual so they join the history visually.
    """
    P = PALETTE
    hist = hist.sort_values("week").tail(52)
    f = f.sort_values("horizon")
    at_origin = hist.loc[hist["week"] == origin, "sales"]
    start = [float(at_origin.iloc[0])] if len(at_origin) else []
    x = ([origin] if start else []) + list(f["target_week"])
    yhat, q10, q90 = (start + list(f[c]) for c in ("yhat", "q10", "q90"))
    marker_sizes = ([0] if start else []) + [9] * len(f)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["week"], y=hist["sales"], name="Actual", mode="lines",
        line=dict(color=P["actual"], width=2), hovertemplate="%{y:.0f}<extra>Actual</extra>",
    ))
    fig.add_trace(go.Scatter(x=x, y=q90, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=x, y=q10, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=P["band"],
        name="P10–P90 band", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=yhat, name="Forecast", mode="lines+markers",
        line=dict(color=P["forecast"], width=2),
        marker=dict(size=marker_sizes, color=P["forecast"], line=dict(color=P["surface"], width=2)),
        customdata=np.column_stack([q10, q90]),
        hovertemplate="%{y:.0f}  (P10 %{customdata[0]:.0f} · P90 %{customdata[1]:.0f})<extra>Forecast</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=f["target_week"], y=f["last_year"], name="Same weeks last year", mode="lines+markers",
        line=dict(color=P["last_year"], width=2, dash="dot"),
        marker=dict(size=7, color=P["last_year"], line=dict(color=P["surface"], width=2)),
        hovertemplate="%{y:.0f}<extra>Last year</extra>",
    ))
    fig.add_shape(type="line", x0=origin, x1=origin, y0=0, y1=1, yref="paper",
                  line=dict(color=P["axis"], width=1, dash="dash"))
    fig.add_annotation(x=origin, y=1, yref="paper", text="forecast origin", showarrow=False,
                       yanchor="bottom", xanchor="right", xshift=-4, font=dict(size=11, color=P["ink"]))
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=48, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None, traceorder="normal"),
        xaxis=dict(title=None, showgrid=False, linecolor=P["axis"]),
        yaxis=dict(title="units / week", rangemode="tozero", showgrid=True, gridcolor=P["grid"],
                   zeroline=False, linecolor=P["axis"]),
    )
    return fig


# ---- Model quality page -------------------------------------------------------------------

METHOD_ORDER = ("lightgbm", "seasonal_naive", "moving_average_8")
METHOD_COLORS = {"lightgbm": "#2a78d6", "seasonal_naive": "#eb6834", "moving_average_8": "#1baf7a"}  # categorical slots 1–3
METHOD_SHORT = {"lightgbm": "LightGBM", "seasonal_naive": "Seasonal naive", "moving_average_8": "8-week MA"}


def wape_by_fold_chart(by_fold: pd.DataFrame) -> go.Figure:
    """One line per method, WAPE in % by fold origin. Legend plus a direct label at each line end."""
    P = PALETTE
    by_fold = by_fold.sort_values("origin")
    fig = go.Figure()
    for m in (m for m in METHOD_ORDER if m in by_fold.columns):
        y = 100 * by_fold[m]
        fig.add_trace(go.Scatter(
            x=by_fold["origin"], y=y, name=METHOD_SHORT[m], mode="lines+markers",
            line=dict(color=METHOD_COLORS[m], width=2),
            marker=dict(size=8, color=METHOD_COLORS[m], line=dict(color=P["surface"], width=2)),
            hovertemplate="%{y:.1f}%<extra>" + METHOD_SHORT[m] + "</extra>",
        ))
        fig.add_annotation(x=by_fold["origin"].iloc[-1], y=float(y.iloc[-1]), text=METHOD_SHORT[m], showarrow=False,
                           xanchor="left", xshift=10, font=dict(size=11, color=P["ink"]))
    fig.update_layout(
        height=360, margin=dict(l=10, r=110, t=40, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title="fold origin (forecasts cover the 4 weeks after it)", showgrid=False, linecolor=P["axis"]),
        yaxis=dict(title="WAPE (%)", rangemode="tozero", showgrid=True, gridcolor=P["grid"], zeroline=False, linecolor=P["axis"]),
    )
    return fig


def wape_histogram(by_series: pd.DataFrame, bin_size: float = 0.5) -> go.Figure:
    """Distribution of per-series WAPE (in %), median marked with a dashed line."""
    P = PALETTE
    w = 100 * by_series["wape"]
    med = float(w.median())
    fig = go.Figure(go.Histogram(
        x=w, xbins=dict(start=0, size=bin_size), marker=dict(color=P["actual"]), name="series",
        hovertemplate="WAPE %{x}%: %{y} series<extra></extra>",
    ))
    fig.add_shape(type="line", x0=med, x1=med, y0=0, y1=1, yref="paper", line=dict(color=P["ink"], width=1, dash="dash"))
    fig.add_annotation(x=med, y=1, yref="paper", text=f"median {med:.1f}%", showarrow=False, xanchor="left", xshift=6,
                       yanchor="top", font=dict(size=11, color=P["ink"]))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10), bargap=0.08, showlegend=False,
        xaxis=dict(title="WAPE per series (%)", showgrid=False, linecolor=P["axis"]),
        yaxis=dict(title="number of series", showgrid=True, gridcolor=P["grid"], zeroline=False, linecolor=P["axis"]),
    )
    return fig


def importance_chart(importance: pd.DataFrame, top: int = 15) -> go.Figure:
    """Top-``top`` features by LightGBM gain, shown as share of total gain (%), largest at the top."""
    P = PALETTE
    imp = importance.copy()
    imp["share"] = 100 * imp["gain"] / imp["gain"].sum()
    t = imp.nlargest(top, "gain").sort_values("gain")
    fig = go.Figure(go.Bar(
        x=t["share"], y=t["feature"], orientation="h", marker=dict(color=P["actual"]),
        text=[f"{v:.1f}%" for v in t["share"]], textposition="outside", textfont=dict(color=P["ink"], size=11),
        customdata=t["gain"], hovertemplate="%{y}: %{x:.1f}% of gain (%{customdata:,.0f})<extra></extra>",
    ))
    fig.update_layout(
        height=30 * len(t) + 60, margin=dict(l=10, r=40, t=20, b=10), bargap=0.3,
        xaxis=dict(title="share of total gain (%)", showgrid=True, gridcolor=P["grid"], zeroline=False, linecolor=P["axis"],
                   range=[0, float(t["share"].max()) * 1.18]),
        yaxis=dict(title=None, showgrid=False, linecolor=P["axis"]),
    )
    return fig


# ---- How it works page ----------------------------------------------------------------------


def direct_horizon_figure() -> go.Figure:
    """Illustration (not data): one origin, four separate models for weeks +1..+4, band widening."""
    P = PALETTE
    hist_x, hist_y = [-4, -3, -2, -1, 0], [100, 104, 98, 103, 101]
    fx, fy = [1, 2, 3, 4], [105, 108, 104, 110]
    half = [6, 9, 12, 15]
    band_x = [0] + fx
    lo = [hist_y[-1]] + [y - h for y, h in zip(fy, half)]
    hi = [hist_y[-1]] + [y + h for y, h in zip(fy, half)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=band_x, y=hi, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=band_x, y=lo, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=P["band"],
                             name="P10–P90 band (widens with distance)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist_x, y=hist_y, mode="lines+markers", name="history the models can see",
                             line=dict(color=P["actual"], width=2),
                             marker=dict(size=7, color=P["actual"], line=dict(color=P["surface"], width=2)), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fx, y=fy, mode="markers", name="forecast: one model per week ahead",
                             marker=dict(size=11, color=P["forecast"], line=dict(color=P["surface"], width=2)), hoverinfo="skip"))
    for x, y in zip(fx, fy):
        fig.add_annotation(x=x, y=y, ax=0, ay=hist_y[-1], axref="x", ayref="y", text="", showarrow=True,
                           arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor=P["forecast"], standoff=7, startstandoff=6)
        fig.add_annotation(x=x, y=y + half[fx.index(x)] + 2, text=f"model +{x}", showarrow=False, yanchor="bottom",
                           font=dict(size=11, color=P["ink"]))
    fig.add_shape(type="line", x0=0, x1=0, y0=0, y1=1, yref="paper", line=dict(color=P["axis"], width=1, dash="dash"))
    fig.add_annotation(x=0, y=1, yref="paper", text="forecast origin", showarrow=False, yanchor="bottom", xanchor="right",
                       xshift=-4, font=dict(size=11, color=P["ink"]))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=56, b=10), showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0, traceorder="normal"),
        xaxis=dict(title="weeks from the forecast origin", tickmode="array", tickvals=hist_x + fx,
                   ticktext=["−4", "−3", "−2", "−1", "0", "+1", "+2", "+3", "+4"], showgrid=False, linecolor=P["axis"],
                   range=[-4.4, 4.6]),
        yaxis=dict(title="units / week", showticklabels=False, showgrid=False, zeroline=False, linecolor=P["axis"],
                   range=[80, 132]),
    )
    return fig
