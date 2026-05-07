"""
Tests for src/stress_episodes.py.

Covers episode stats computation, score paths, timeline chart,
covered/uncovered logic, and the orchestrator.
"""

import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

from src.stress_episodes import (
    STRESS_EPISODES,
    _HIGH_ALERT_THRESHOLD,
    compute_episode_stats,
    compute_score_paths,
    build_stress_timeline,
    run_stress_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(start="2025-01-01", periods=200, seed=42):
    """Synthetic DataFrame spanning ~10 months from start."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq="B")
    n = len(dates)
    composite = rng.uniform(20, 80, n)
    sp500 = 4000 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    return pd.DataFrame({
        "date":                        dates,
        "composite_risk_score_smooth": composite,
        "final_decision":              np.where(composite > 60, "Avoid Chasing Risk", "Neutral"),
        "sp500":                       sp500,
        "sp500_daily_return":          rng.normal(0.0003, 0.01, n),
    })


@pytest.fixture(scope="module")
def base_df():
    return _make_df(start="2025-01-01", periods=300)


@pytest.fixture(scope="module")
def covered_episode():
    """A single episode fully within base_df range."""
    return {
        "name": "TestEpisode",
        "label": "Test Stress Event",
        "severity": "Moderate",
        "start": "2025-02-01",
        "end":   "2025-03-15",
    }


@pytest.fixture(scope="module")
def uncovered_episode():
    """A single episode completely outside base_df range."""
    return {
        "name": "GFC",
        "label": "Global Financial Crisis",
        "severity": "Severe",
        "start": "2008-09-01",
        "end":   "2009-03-31",
    }


@pytest.fixture(scope="module")
def high_alert_df():
    """DataFrame with consistently high composite scores (always ≥ threshold)."""
    dates = pd.date_range("2025-03-01", periods=60, freq="B")
    return pd.DataFrame({
        "date":                        dates,
        "composite_risk_score_smooth": np.full(60, 75.0),
        "final_decision":              ["Avoid Chasing Risk"] * 60,
        "sp500":                       4500 - np.arange(60) * 10.0,
    })


@pytest.fixture(scope="module")
def two_episode_list(covered_episode, uncovered_episode):
    return [covered_episode, uncovered_episode]


# ---------------------------------------------------------------------------
# STRESS_EPISODES registry
# ---------------------------------------------------------------------------


class TestEpisodeRegistry:
    def test_is_list(self):
        assert isinstance(STRESS_EPISODES, list)

    def test_non_empty(self):
        assert len(STRESS_EPISODES) > 0

    def test_all_have_required_keys(self):
        required = {"name", "start", "end", "label", "severity"}
        for ep in STRESS_EPISODES:
            assert required.issubset(set(ep.keys())), f"Missing keys in {ep['name']}"

    def test_start_before_end(self):
        for ep in STRESS_EPISODES:
            assert ep["start"] < ep["end"], f"{ep['name']} has start >= end"

    def test_severity_values_valid(self):
        valid = {"Severe", "Moderate", "Mild"}
        for ep in STRESS_EPISODES:
            assert ep["severity"] in valid, f"Invalid severity in {ep['name']}"

    def test_names_unique(self):
        names = [ep["name"] for ep in STRESS_EPISODES]
        assert len(names) == len(set(names))

    def test_contains_recent_episodes(self):
        names = {ep["name"] for ep in STRESS_EPISODES}
        assert "Liberation Day" in names or any("2025" in ep["start"] for ep in STRESS_EPISODES)


# ---------------------------------------------------------------------------
# compute_episode_stats
# ---------------------------------------------------------------------------


class TestComputeEpisodeStats:
    def test_returns_dataframe(self, base_df, two_episode_list):
        result = compute_episode_stats(base_df, two_episode_list)
        assert isinstance(result, pd.DataFrame)

    def test_index_is_episode_names(self, base_df, two_episode_list, covered_episode, uncovered_episode):
        result = compute_episode_stats(base_df, two_episode_list)
        assert covered_episode["name"] in result.index
        assert uncovered_episode["name"] in result.index

    def test_covered_flag_correct(self, base_df, two_episode_list):
        result = compute_episode_stats(base_df, two_episode_list)
        assert result.loc["TestEpisode", "covered"] is True or result.loc["TestEpisode", "covered"] == True
        assert result.loc["GFC", "covered"] is False or result.loc["GFC", "covered"] == False

    def test_uncovered_has_nan_numerics(self, base_df, two_episode_list):
        result = compute_episode_stats(base_df, two_episode_list)
        assert np.isnan(result.loc["GFC", "mean_composite"])
        assert np.isnan(result.loc["GFC", "peak_composite"])
        assert np.isnan(result.loc["GFC", "sp500_drawdown"])

    def test_uncovered_has_zero_n_obs(self, base_df, two_episode_list):
        result = compute_episode_stats(base_df, two_episode_list)
        assert result.loc["GFC", "n_obs"] == 0

    def test_covered_n_obs_positive(self, base_df, two_episode_list):
        result = compute_episode_stats(base_df, two_episode_list)
        assert result.loc["TestEpisode", "n_obs"] > 0

    def test_covered_mean_composite_in_range(self, base_df, two_episode_list):
        result = compute_episode_stats(base_df, two_episode_list)
        mc = result.loc["TestEpisode", "mean_composite"]
        assert 0 <= mc <= 100

    def test_peak_ge_mean(self, base_df, two_episode_list):
        result = compute_episode_stats(base_df, two_episode_list)
        assert result.loc["TestEpisode", "peak_composite"] >= result.loc["TestEpisode", "mean_composite"] - 1e-6

    def test_high_alert_pct_between_0_and_1(self, base_df, two_episode_list):
        result = compute_episode_stats(base_df, two_episode_list)
        pct = result.loc["TestEpisode", "high_alert_pct"]
        assert 0.0 <= pct <= 1.0

    def test_high_alert_pct_is_1_when_all_high(self, high_alert_df):
        ep = [{"name": "HighAlert", "label": "High", "severity": "Severe",
               "start": "2025-03-01", "end": "2025-05-31"}]
        result = compute_episode_stats(high_alert_df, ep)
        assert result.loc["HighAlert", "high_alert_pct"] == pytest.approx(1.0)

    def test_primary_decision_is_string(self, base_df, two_episode_list):
        result = compute_episode_stats(base_df, two_episode_list)
        pd_val = result.loc["TestEpisode", "primary_decision"]
        assert isinstance(pd_val, str)

    def test_sp500_drawdown_nonpositive(self, base_df, two_episode_list):
        result = compute_episode_stats(base_df, two_episode_list)
        dd = result.loc["TestEpisode", "sp500_drawdown"]
        if not np.isnan(dd):
            assert dd <= 0

    def test_peak_date_is_string_or_none(self, base_df, two_episode_list):
        result = compute_episode_stats(base_df, two_episode_list)
        pd_val = result.loc["TestEpisode", "peak_date"]
        assert pd_val is None or isinstance(pd_val, str)

    def test_empty_df_returns_empty(self, two_episode_list):
        result = compute_episode_stats(pd.DataFrame(), two_episode_list)
        assert result.empty

    def test_no_episodes_returns_empty(self, base_df):
        result = compute_episode_stats(base_df, [])
        assert result.empty

    def test_label_and_severity_preserved(self, base_df, two_episode_list):
        result = compute_episode_stats(base_df, two_episode_list)
        assert result.loc["TestEpisode", "label"] == "Test Stress Event"
        assert result.loc["TestEpisode", "severity"] == "Moderate"

    def test_all_standard_columns_present(self, base_df, two_episode_list):
        result = compute_episode_stats(base_df, two_episode_list)
        for col in ["label", "severity", "start", "end", "covered", "n_obs",
                    "mean_composite", "peak_composite", "sp500_drawdown",
                    "primary_decision", "high_alert_pct", "lead_days"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_declining_sp500_gives_negative_drawdown(self):
        dates = pd.date_range("2025-06-01", periods=30, freq="B")
        sp = 5000.0 - np.arange(30) * 50.0  # steadily declining
        df = pd.DataFrame({
            "date": dates,
            "composite_risk_score_smooth": np.full(30, 55.0),
            "final_decision": ["Neutral"] * 30,
            "sp500": sp,
        })
        ep = [{"name": "Decline", "label": "Declining market", "severity": "Moderate",
               "start": "2025-06-01", "end": "2025-07-31"}]
        result = compute_episode_stats(df, ep)
        assert result.loc["Decline", "sp500_drawdown"] < 0


# ---------------------------------------------------------------------------
# compute_score_paths
# ---------------------------------------------------------------------------


class TestComputeScorePaths:
    def test_returns_dict(self, base_df, two_episode_list):
        result = compute_score_paths(base_df, two_episode_list)
        assert isinstance(result, dict)

    def test_covered_episode_in_result(self, base_df, two_episode_list):
        result = compute_score_paths(base_df, two_episode_list)
        assert "TestEpisode" in result

    def test_uncovered_episode_not_in_result(self, base_df, two_episode_list):
        result = compute_score_paths(base_df, two_episode_list)
        assert "GFC" not in result

    def test_path_is_dataframe(self, base_df, two_episode_list):
        result = compute_score_paths(base_df, two_episode_list)
        assert isinstance(result["TestEpisode"], pd.DataFrame)

    def test_path_has_date_col(self, base_df, two_episode_list):
        result = compute_score_paths(base_df, two_episode_list)
        assert "date" in result["TestEpisode"].columns

    def test_path_includes_padding(self, base_df, two_episode_list):
        # pad_days=30 → path should start before episode start
        result = compute_score_paths(base_df, two_episode_list, pad_days=30)
        path = result["TestEpisode"]
        ep_start = pd.Timestamp("2025-02-01")
        assert path["date"].min() < ep_start

    def test_empty_df_returns_empty_dict(self, two_episode_list):
        result = compute_score_paths(pd.DataFrame(), two_episode_list)
        assert result == {}


# ---------------------------------------------------------------------------
# build_stress_timeline
# ---------------------------------------------------------------------------


class TestBuildStressTimeline:
    def test_returns_figure(self, base_df, two_episode_list):
        fig = build_stress_timeline(base_df, two_episode_list)
        assert isinstance(fig, go.Figure)

    def test_has_at_least_one_trace(self, base_df, two_episode_list):
        fig = build_stress_timeline(base_df, two_episode_list)
        assert len(fig.data) >= 1

    def test_composite_trace_present(self, base_df, two_episode_list):
        fig = build_stress_timeline(base_df, two_episode_list)
        names = [t.name for t in fig.data if t.name]
        assert any("Composite" in n for n in names)

    def test_empty_df_returns_figure(self):
        fig = build_stress_timeline(pd.DataFrame(), [])
        assert isinstance(fig, go.Figure)

    def test_vrects_added_for_covered_episodes(self, base_df, two_episode_list):
        fig = build_stress_timeline(base_df, two_episode_list)
        # vrects are added via layout.shapes
        shapes = fig.layout.shapes or []
        assert len(shapes) >= 1

    def test_yaxis_range_includes_100(self, base_df, two_episode_list):
        fig = build_stress_timeline(base_df, two_episode_list)
        y_range = fig.layout.yaxis.range
        assert y_range is not None
        assert y_range[1] >= 100


# ---------------------------------------------------------------------------
# run_stress_analysis (orchestrator)
# ---------------------------------------------------------------------------


class TestRunStressAnalysis:
    def test_returns_dict(self, base_df, two_episode_list):
        result = run_stress_analysis(base_df, two_episode_list)
        assert isinstance(result, dict)

    def test_all_keys_present(self, base_df, two_episode_list):
        result = run_stress_analysis(base_df, two_episode_list)
        for key in ["episode_stats", "score_paths", "timeline_chart", "n_covered", "n_historical"]:
            assert key in result

    def test_n_covered_correct(self, base_df, two_episode_list):
        result = run_stress_analysis(base_df, two_episode_list)
        assert result["n_covered"] == 1  # only TestEpisode is in range

    def test_n_historical_correct(self, base_df, two_episode_list):
        result = run_stress_analysis(base_df, two_episode_list)
        assert result["n_historical"] == 1  # GFC is out of range

    def test_episode_stats_is_dataframe(self, base_df, two_episode_list):
        result = run_stress_analysis(base_df, two_episode_list)
        assert isinstance(result["episode_stats"], pd.DataFrame)

    def test_score_paths_is_dict(self, base_df, two_episode_list):
        result = run_stress_analysis(base_df, two_episode_list)
        assert isinstance(result["score_paths"], dict)

    def test_timeline_chart_is_figure(self, base_df, two_episode_list):
        result = run_stress_analysis(base_df, two_episode_list)
        assert isinstance(result["timeline_chart"], go.Figure)

    def test_empty_df_graceful(self):
        result = run_stress_analysis(pd.DataFrame(), STRESS_EPISODES)
        assert result["n_covered"] == 0
        assert result["episode_stats"].empty
