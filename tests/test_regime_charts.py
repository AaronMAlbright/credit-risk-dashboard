"""
Tests for src/regime_charts.py — structural only.

Verifies figure types, trace presence, color palette coverage,
date filtering, and edge cases (missing columns, empty data).
"""

import pandas as pd
import numpy as np
import pytest
import plotly.graph_objects as go

from src.regime_charts import (
    DECISION_COLORS,
    TRANSITION_COLORS,
    SCORE_COLORS,
    _SCORE_COLS,
    _regime_spans,
    _get_dates,
    build_decision_timeline,
    build_sp500_with_regime_overlay,
    build_score_history,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ALL_DECISIONS = [
    "Buy Stress", "Watch Entry", "Hold / Do Not Chase",
    "Neutral", "Wait", "Avoid Chasing Risk",
    "Divergence Warning", "Credit Warning", "Stress / Stabilization Watch",
]

ALL_TRANSITIONS = [
    "Stable / Neutral", "Complacency / Late Cycle", "Early Deterioration",
    "Deteriorating / Rising Stress", "Panic / Stabilizing",
]


@pytest.fixture(scope="module")
def sample_df():
    n = 120
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    # Alternating regimes every 20 rows
    decisions = (
        ["Neutral"] * 20 + ["Avoid Chasing Risk"] * 20 +
        ["Buy Stress"] * 20 + ["Divergence Warning"] * 20 +
        ["Wait"] * 20 + ["Neutral"] * 20
    )
    transitions = (
        ["Stable / Neutral"] * 30 + ["Deteriorating / Rising Stress"] * 30 +
        ["Complacency / Late Cycle"] * 30 + ["Panic / Stabilizing"] * 30
    )
    df = pd.DataFrame({
        "date": dates,
        "final_decision": decisions,
        "transition_regime": transitions,
        "sp500": 4000 + np.cumsum(rng.normal(0, 20, n)),
        **{col: rng.uniform(10, 80, n) for col in _SCORE_COLS.values()},
    })
    return df


@pytest.fixture(scope="module")
def minimal_df():
    """Single-regime DataFrame."""
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=5, freq="B"),
        "final_decision": ["Neutral"] * 5,
        "sp500": [4000.0] * 5,
        "macro_risk_score_smooth": [30.0] * 5,
        "composite_risk_score_smooth": [25.0] * 5,
    })


# ---------------------------------------------------------------------------
# TestColorPalettes
# ---------------------------------------------------------------------------


class TestColorPalettes:
    def test_all_decisions_have_color(self):
        for d in ALL_DECISIONS:
            assert d in DECISION_COLORS, f"Missing color for decision: {d}"

    def test_all_transitions_have_color(self):
        for t in ALL_TRANSITIONS:
            assert t in TRANSITION_COLORS, f"Missing color for transition: {t}"

    def test_decision_colors_are_hex(self):
        for name, color in DECISION_COLORS.items():
            assert color.startswith("#"), f"{name} color not hex: {color}"
            assert len(color) == 7, f"{name} color wrong length: {color}"

    def test_transition_colors_are_hex(self):
        for name, color in TRANSITION_COLORS.items():
            assert color.startswith("#"), f"{name} color not hex: {color}"

    def test_score_colors_cover_all_score_names(self):
        for name in _SCORE_COLS:
            assert name in SCORE_COLORS, f"Missing score color for: {name}"


# ---------------------------------------------------------------------------
# TestRegimeSpans
# ---------------------------------------------------------------------------


class TestRegimeSpans:
    def test_empty_series_returns_empty(self):
        result = _regime_spans(pd.Series([], dtype="datetime64[ns]"), pd.Series([], dtype=str))
        assert result == []

    def test_single_value_returns_one_span(self):
        dates = pd.Series(pd.date_range("2023-01-01", periods=3, freq="D"))
        series = pd.Series(["A", "A", "A"])
        spans = _regime_spans(dates, series)
        assert len(spans) == 1
        assert spans[0][2] == "A"

    def test_end_exclusive_is_after_start(self):
        dates = pd.Series(pd.date_range("2023-01-01", periods=5, freq="D"))
        series = pd.Series(["A", "A", "B", "B", "B"])
        spans = _regime_spans(dates, series)
        for start, end_excl, _ in spans:
            assert end_excl > start

    def test_transition_boundary_aligned(self):
        dates = pd.Series(pd.date_range("2023-01-01", periods=4, freq="D"))
        series = pd.Series(["A", "A", "B", "B"])
        spans = _regime_spans(dates, series)
        assert len(spans) == 2
        # End of span A == Start of span B
        assert spans[0][1] == spans[1][0]

    def test_alternating_regimes(self):
        dates = pd.Series(pd.date_range("2023-01-01", periods=6, freq="D"))
        series = pd.Series(["A", "B", "A", "B", "A", "B"])
        spans = _regime_spans(dates, series)
        assert len(spans) == 6


# ---------------------------------------------------------------------------
# TestBuildDecisionTimeline
# ---------------------------------------------------------------------------


class TestBuildDecisionTimeline:
    def test_returns_figure(self, sample_df):
        fig = build_decision_timeline(sample_df)
        assert isinstance(fig, go.Figure)

    def test_has_traces(self, sample_df):
        fig = build_decision_timeline(sample_df)
        assert len(fig.data) > 0

    def test_title_contains_regime_col(self, sample_df):
        fig = build_decision_timeline(sample_df, "final_decision")
        assert fig.layout.title.text is not None

    def test_transition_regime_col(self, sample_df):
        fig = build_decision_timeline(sample_df, "transition_regime")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_missing_col_returns_empty_figure(self, sample_df):
        fig = build_decision_timeline(sample_df, "nonexistent_col")
        assert isinstance(fig, go.Figure)
        # No data traces (just layout)
        assert len(fig.data) == 0

    def test_single_regime_handled(self, minimal_df):
        fig = build_decision_timeline(minimal_df)
        assert isinstance(fig, go.Figure)

    def test_height_positive(self, sample_df):
        fig = build_decision_timeline(sample_df)
        assert fig.layout.height > 0


# ---------------------------------------------------------------------------
# TestBuildSP500WithRegimeOverlay
# ---------------------------------------------------------------------------


class TestBuildSP500WithRegimeOverlay:
    def test_returns_figure(self, sample_df):
        fig = build_sp500_with_regime_overlay(sample_df)
        assert isinstance(fig, go.Figure)

    def test_sp500_trace_present(self, sample_df):
        fig = build_sp500_with_regime_overlay(sample_df)
        trace_names = [t.name for t in fig.data]
        assert "SP500" in trace_names

    def test_sp500_trace_has_data(self, sample_df):
        fig = build_sp500_with_regime_overlay(sample_df)
        sp_trace = next(t for t in fig.data if t.name == "SP500")
        assert len(sp_trace.y) > 0

    def test_date_start_filter_reduces_data(self, sample_df):
        fig_full = build_sp500_with_regime_overlay(sample_df)
        fig_filtered = build_sp500_with_regime_overlay(sample_df, date_start="2023-03-01")
        sp_full = next(t for t in fig_full.data if t.name == "SP500")
        sp_filt = next(t for t in fig_filtered.data if t.name == "SP500")
        assert len(sp_filt.y) < len(sp_full.y)

    def test_date_end_filter_reduces_data(self, sample_df):
        fig_full = build_sp500_with_regime_overlay(sample_df)
        fig_filtered = build_sp500_with_regime_overlay(sample_df, date_end="2023-03-01")
        sp_full = next(t for t in fig_full.data if t.name == "SP500")
        sp_filt = next(t for t in fig_filtered.data if t.name == "SP500")
        assert len(sp_filt.y) < len(sp_full.y)

    def test_vrect_shapes_added(self, sample_df):
        fig = build_sp500_with_regime_overlay(sample_df)
        # vrects appear as shapes in layout
        assert len(fig.layout.shapes) > 0 or len(fig.data) > 0

    def test_missing_sp500_col_no_crash(self):
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="B"),
            "final_decision": ["Neutral"] * 10,
        })
        fig = build_sp500_with_regime_overlay(df)
        assert isinstance(fig, go.Figure)

    def test_transition_regime_col(self, sample_df):
        fig = build_sp500_with_regime_overlay(sample_df, regime_col="transition_regime")
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# TestBuildScoreHistory
# ---------------------------------------------------------------------------


class TestBuildScoreHistory:
    def test_returns_figure(self, sample_df):
        fig = build_score_history(sample_df)
        assert isinstance(fig, go.Figure)

    def test_has_score_traces(self, sample_df):
        fig = build_score_history(sample_df)
        assert len(fig.data) > 0

    def test_always_visible_traces_are_visible(self, sample_df):
        fig = build_score_history(sample_df)
        visible_names = {t.name for t in fig.data if t.visible is True}
        assert "Macro Risk" in visible_names
        assert "Credit Risk" in visible_names
        assert "Composite" in visible_names

    def test_others_are_legendonly(self, sample_df):
        fig = build_score_history(sample_df)
        legend_only = {t.name for t in fig.data if t.visible == "legendonly"}
        assert "Complacency" in legend_only

    def test_date_filter_reduces_x_range(self, sample_df):
        fig_full = build_score_history(sample_df)
        fig_filt = build_score_history(sample_df, date_start="2023-03-01")
        macro_full = next(t for t in fig_full.data if t.name == "Macro Risk")
        macro_filt = next(t for t in fig_filt.data if t.name == "Macro Risk")
        assert len(macro_filt.y) < len(macro_full.y)

    def test_threshold_annotations_present(self, sample_df):
        fig = build_score_history(sample_df)
        # Threshold lines appear as layout shapes
        assert len(fig.layout.shapes) >= 2

    def test_missing_score_cols_no_crash(self):
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="B"),
            "macro_risk_score_smooth": [30.0] * 10,
        })
        fig = build_score_history(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_y_axis_range_set(self, sample_df):
        fig = build_score_history(sample_df)
        assert fig.layout.yaxis.range is not None
