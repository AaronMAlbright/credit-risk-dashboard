"""
Plotly-based regime visualization charts.

All public functions return go.Figure objects ready for st.plotly_chart().
No production logic is modified here.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Persistent color palettes
# ---------------------------------------------------------------------------

DECISION_COLORS: dict[str, str] = {
    "Buy Stress":                   "#27ae60",
    "Watch Entry":                  "#2ecc71",
    "Hold / Do Not Chase":          "#f1c40f",
    "Neutral":                      "#95a5a6",
    "Wait":                         "#e67e22",
    "Avoid Chasing Risk":           "#e74c3c",
    "Divergence Warning":           "#9b59b6",
    "Credit Warning":               "#c0392b",
    "Stress / Stabilization Watch": "#8e44ad",
}

TRANSITION_COLORS: dict[str, str] = {
    "Stable / Neutral":              "#27ae60",
    "Complacency / Late Cycle":      "#f1c40f",
    "Early Deterioration":           "#e67e22",
    "Deteriorating / Rising Stress": "#e74c3c",
    "Panic / Stabilizing":           "#8e44ad",
}

SCORE_COLORS: dict[str, str] = {
    "Macro Risk":     "#e74c3c",
    "Credit Risk":    "#e67e22",
    "Complacency":    "#9b59b6",
    "Liquidity":      "#3498db",
    "Treasury":       "#1abc9c",
    "Mean Reversion": "#f39c12",
    "Composite":      "#2c3e50",
}

_DEFAULT_COLOR = "#bdc3c7"

_SCORE_COLS = {
    "Macro Risk":     "macro_risk_score_smooth",
    "Credit Risk":    "credit_market_risk_score_smooth",
    "Complacency":    "complacency_score_smooth",
    "Liquidity":      "liquidity_regime_score_smooth",
    "Treasury":       "treasury_stress_score_smooth",
    "Mean Reversion": "mean_reversion_score_smooth",
    "Composite":      "composite_risk_score_smooth",
}

_LAYOUT_BASE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=20, r=20, t=45, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_dates(df: pd.DataFrame) -> pd.Series:
    if "date" in df.columns:
        return pd.to_datetime(df["date"]).reset_index(drop=True)
    return pd.to_datetime(df.index).to_series().reset_index(drop=True)


def _regime_spans(dates: pd.Series, series: pd.Series) -> list[tuple]:
    """Return list of (start_date, end_date_exclusive, regime) from a regime series."""
    s = series.reset_index(drop=True)
    d = dates.reset_index(drop=True)
    if s.empty:
        return []

    spans = []
    current = s.iloc[0]
    start = 0
    for i in range(1, len(s)):
        if s.iloc[i] != current:
            # end_exclusive = next period start (or +1 day for last row)
            end_excl = d.iloc[i]
            spans.append((d.iloc[start], end_excl, current))
            current = s.iloc[i]
            start = i
    # Close last span with +1 day so it's visible
    spans.append((d.iloc[start], d.iloc[-1] + pd.Timedelta(days=1), current))
    return spans


def _apply_date_mask(df: pd.DataFrame, dates: pd.Series,
                     date_start, date_end) -> tuple[pd.DataFrame, pd.Series]:
    mask = pd.Series([True] * len(df), index=df.index)
    if date_start:
        mask &= dates >= pd.Timestamp(date_start)
    if date_end:
        mask &= dates <= pd.Timestamp(date_end)
    return df[mask].copy(), dates[mask.values]


# ---------------------------------------------------------------------------
# Public chart functions
# ---------------------------------------------------------------------------

def build_decision_timeline(df: pd.DataFrame,
                             regime_col: str = "final_decision") -> go.Figure:
    """
    Gantt-style horizontal timeline where each regime run is a colored bar.
    Uses px.timeline so multiple non-contiguous runs of the same regime
    share a single y-row and legend entry.

    Returns go.Figure.
    """
    if regime_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(title=f"Column '{regime_col}' not found", **_LAYOUT_BASE)
        return fig

    dates = _get_dates(df)
    series = df[regime_col].reset_index(drop=True)
    spans = _regime_spans(dates, series)
    if not spans:
        return go.Figure()

    color_map = DECISION_COLORS if regime_col == "final_decision" else TRANSITION_COLORS

    records = [
        {
            "Regime": regime,
            "Start":  start,
            "End":    end_excl,
            "Days":   max(1, (end_excl - start).days),
        }
        for start, end_excl, regime in spans
    ]
    gdf = pd.DataFrame(records)

    fig = px.timeline(
        gdf,
        x_start="Start",
        x_end="End",
        y="Regime",
        color="Regime",
        color_discrete_map={k: color_map.get(k, _DEFAULT_COLOR) for k in gdf["Regime"].unique()},
        hover_data={"Days": True, "Start": "|%Y-%m-%d", "End": "|%Y-%m-%d"},
        opacity=0.85,
    )

    label = regime_col.replace("_", " ").title()
    fig.update_layout(
        title=f"{label} Timeline",
        xaxis_title="Date",
        yaxis_title="",
        height=max(280, len(gdf["Regime"].unique()) * 38 + 80),
        **_LAYOUT_BASE,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def build_sp500_with_regime_overlay(df: pd.DataFrame,
                                    regime_col: str = "final_decision",
                                    date_start=None,
                                    date_end=None) -> go.Figure:
    """
    SP500 price line with shaded regime backgrounds and transition markers.

    Returns go.Figure.
    """
    dates = _get_dates(df)
    sub, sub_dates = _apply_date_mask(df, dates, date_start, date_end)
    sub_dates = sub_dates.reset_index(drop=True)

    color_map = DECISION_COLORS if regime_col == "final_decision" else TRANSITION_COLORS
    fig = go.Figure()

    # Background shading per regime run
    if regime_col in sub.columns:
        series = sub[regime_col].reset_index(drop=True)
        spans = _regime_spans(sub_dates, series)

        seen_legend: set[str] = set()
        for start, end_excl, regime in spans:
            color = color_map.get(regime, _DEFAULT_COLOR)
            show = regime not in seen_legend
            seen_legend.add(regime)
            fig.add_vrect(
                x0=start, x1=end_excl,
                fillcolor=color, opacity=0.13,
                layer="below", line_width=0,
                name=regime,
                legendgroup=f"regime_{regime}",
                showlegend=show,
            )

        # Transition markers (dashed vertical line at regime change boundary)
        prev_regime = None
        for start, _, regime in spans:
            if prev_regime is not None:
                fig.add_vline(
                    x=start.timestamp() * 1000,
                    line=dict(color="#888888", width=0.7, dash="dot"),
                    opacity=0.6,
                )
            prev_regime = regime

    # SP500 price trace
    if "sp500" in sub.columns:
        fig.add_trace(go.Scatter(
            x=sub_dates,
            y=sub["sp500"].values,
            mode="lines",
            name="SP500",
            line=dict(color="#1a1a2e", width=1.8),
            hovertemplate="SP500: %{y:,.0f}<br>%{x|%Y-%m-%d}<extra></extra>",
        ))

    fig.update_layout(
        title="SP500 with Regime Overlay",
        xaxis_title="Date",
        yaxis_title="SP500 Level",
        height=420,
        hovermode="x unified",
        **_LAYOUT_BASE,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    return fig


def build_score_history(df: pd.DataFrame,
                        date_start=None,
                        date_end=None) -> go.Figure:
    """
    Interactive multi-line chart of smoothed scores over time.

    Macro Risk, Credit Risk, and Composite are visible by default;
    the others are in the legend and can be toggled on.
    Threshold lines at 40 (Caution) and 70 (Stress) are included.

    Returns go.Figure.
    """
    dates = _get_dates(df)
    sub, sub_dates = _apply_date_mask(df, dates, date_start, date_end)
    sub_dates = sub_dates.reset_index(drop=True)

    fig = go.Figure()

    # Threshold reference lines
    for level, label, color in [(40, "Caution", "#f39c12"), (70, "Stress", "#e74c3c")]:
        fig.add_hline(
            y=level,
            line=dict(color=color, width=1, dash="dash"),
            opacity=0.45,
            annotation_text=label,
            annotation_position="right",
            annotation_font_size=11,
        )

    always_visible = {"Macro Risk", "Credit Risk", "Composite"}

    for name, col in _SCORE_COLS.items():
        if col not in sub.columns:
            continue
        visible = True if name in always_visible else "legendonly"
        fig.add_trace(go.Scatter(
            x=sub_dates,
            y=sub[col].values,
            name=name,
            mode="lines",
            line=dict(
                color=SCORE_COLORS.get(name, "#666666"),
                width=2.2 if name == "Composite" else 1.5,
                dash="dash" if name == "Composite" else "solid",
            ),
            visible=visible,
            hovertemplate=f"{name}: %{{y:.1f}}<br>%{{x|%Y-%m-%d}}<extra></extra>",
        ))

    fig.update_layout(
        title="Score History",
        xaxis_title="Date",
        yaxis_title="Score (0–100)",
        yaxis=dict(range=[0, 105]),
        height=420,
        hovermode="x unified",
        **_LAYOUT_BASE,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    return fig
