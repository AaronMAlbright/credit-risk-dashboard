"""
Tests for src/regime_attribution.py — structural only, no exact value assertions.
"""

import numpy as np
import pandas as pd
import pytest

from src.regime_attribution import (
    COMPOSITE_WEIGHTS,
    DELTA_LOOKBACK,
    DISPLAY_NAMES,
    ELEVATION_PERCENTILE,
    SCORE_COLS,
    compute_rolling_contributions,
    compute_shift_attribution,
    compute_top_drivers,
    compute_trigger_elevation,
    run_regime_attribution,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 100


@pytest.fixture(scope="module")
def simple_df():
    rng = np.random.default_rng(42)
    n = N
    # Scores range 0-100
    df = pd.DataFrame({
        "macro_risk_score_smooth":               rng.uniform(10, 80, n),
        "credit_market_risk_score_smooth":       rng.uniform(10, 70, n),
        "complacency_score_smooth":              rng.uniform(20, 90, n),
        "liquidity_regime_score_smooth":         rng.uniform(5, 60, n),
        "treasury_stress_score_smooth":          rng.uniform(5, 50, n),
        "fx_commodity_score_smooth":             rng.uniform(0, 40, n),
        "enhanced_funding_stress_score_smooth":  rng.uniform(0, 30, n),
        "mean_reversion_score_smooth":           rng.uniform(10, 80, n),
        "cross_asset_divergence_score_smooth":   rng.uniform(20, 90, n),
        "composite_risk_score_smooth":           rng.uniform(20, 70, n),
    })
    # Alternating decisions every 10 rows to generate transitions
    decisions = (["Neutral"] * 10 + ["Risk-On"] * 10 + ["Caution"] * 10) * 4
    df["final_decision"] = decisions[:n]
    return df


@pytest.fixture(scope="module")
def monotone_df():
    """All scores fixed at 50 — no transitions."""
    n = 30
    df = pd.DataFrame({col: [50.0] * n for col in SCORE_COLS.values()})
    df["composite_risk_score_smooth"] = 50.0
    df["final_decision"] = "Neutral"
    return df


@pytest.fixture(scope="module")
def analytics_df(analytics_result):
    df, _, _ = analytics_result
    return df


# ---------------------------------------------------------------------------
# TestComputeRollingContributions
# ---------------------------------------------------------------------------


class TestComputeRollingContributions:
    def test_returns_dataframe(self, simple_df):
        result = compute_rolling_contributions(simple_df)
        assert isinstance(result, pd.DataFrame)

    def test_same_length_as_input(self, simple_df):
        result = compute_rolling_contributions(simple_df)
        assert len(result) == len(simple_df)

    def test_expected_columns(self, simple_df):
        # Only columns present in both COMPOSITE_WEIGHTS and simple_df are expected
        result = compute_rolling_contributions(simple_df)
        for key in COMPOSITE_WEIGHTS:
            col = SCORE_COLS.get(key)
            if col and col in simple_df.columns:
                assert f"{key}_contribution" in result.columns

    def test_all_contributions_non_negative(self, simple_df):
        result = compute_rolling_contributions(simple_df)
        assert (result >= 0).all().all()

    def test_higher_score_higher_contribution(self):
        """Higher score → higher contribution (all signals are now additive)."""
        df_low  = pd.DataFrame({"complacency_score_smooth": [20.0]})
        df_high = pd.DataFrame({"complacency_score_smooth": [80.0]})
        lo = compute_rolling_contributions(df_low)["complacency_contribution"].iloc[0]
        hi = compute_rolling_contributions(df_high)["complacency_contribution"].iloc[0]
        assert hi > lo

    def test_contributions_bounded_by_weight_times_100(self, simple_df):
        result = compute_rolling_contributions(simple_df)
        for key, w in COMPOSITE_WEIGHTS.items():
            col = f"{key}_contribution"
            if col in result.columns:
                assert (result[col] <= w * 100 + 1e-9).all()

    def test_contributions_sum_within_composite_range(self, simple_df):
        result = compute_rolling_contributions(simple_df)
        total = result.sum(axis=1)
        # Sum of weighted contributions must be in [0, 100]
        assert (total >= 0).all()
        assert (total <= 100 + 1e-6).all()

    def test_missing_columns_handled(self):
        df = pd.DataFrame({"macro_risk_score_smooth": [40.0]})
        result = compute_rolling_contributions(df)
        # Only macro_risk should appear
        assert "macro_risk_contribution" in result.columns
        assert "credit_risk_contribution" not in result.columns


# ---------------------------------------------------------------------------
# TestComputeShiftAttribution
# ---------------------------------------------------------------------------


class TestComputeShiftAttribution:
    def test_returns_dataframe(self, simple_df):
        result = compute_shift_attribution(simple_df)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, simple_df):
        result = compute_shift_attribution(simple_df)
        for key in SCORE_COLS:
            if f"{key}_delta" not in result.columns:
                # Only complain if that score column existed in df
                assert f"score_cols_{key}" not in simple_df.columns
        assert "primary_driver" in result.columns
        assert "secondary_driver" in result.columns
        assert "from_regime" in result.columns
        assert "to_regime" in result.columns

    def test_row_count_matches_transitions(self, simple_df):
        series = simple_df["final_decision"]
        expected_transitions = ((series != series.shift(1)) & series.notna() & series.shift(1).notna()).sum()
        result = compute_shift_attribution(simple_df)
        assert len(result) == expected_transitions

    def test_primary_driver_is_display_name(self, simple_df):
        result = compute_shift_attribution(simple_df)
        valid_names = set(DISPLAY_NAMES.values())
        for val in result["primary_driver"].dropna():
            assert val in valid_names, f"Unexpected primary_driver: {val}"

    def test_direction_values(self, simple_df):
        result = compute_shift_attribution(simple_df)
        valid = {"Risk-On", "Risk-Off", None}
        assert set(result["direction"].unique()).issubset(valid | {np.nan})

    def test_no_transitions_on_constant_regime(self, monotone_df):
        result = compute_shift_attribution(monotone_df)
        assert result.empty

    def test_missing_regime_col_returns_empty(self, simple_df):
        result = compute_shift_attribution(simple_df, regime_col="nonexistent")
        assert result.empty

    def test_composite_delta_present(self, simple_df):
        result = compute_shift_attribution(simple_df)
        assert "composite_delta" in result.columns

    def test_delta_columns_are_numeric(self, simple_df):
        result = compute_shift_attribution(simple_df)
        for key in SCORE_COLS:
            col = f"{key}_delta"
            if col in result.columns:
                assert pd.api.types.is_numeric_dtype(result[col])


# ---------------------------------------------------------------------------
# TestComputeTriggerElevation
# ---------------------------------------------------------------------------


class TestComputeTriggerElevation:
    def test_returns_dataframe(self, simple_df):
        result = compute_trigger_elevation(simple_df)
        assert isinstance(result, pd.DataFrame)

    def test_indexed_by_decision(self, simple_df):
        result = compute_trigger_elevation(simple_df)
        assert result.index.name == "decision"

    def test_n_obs_sums_to_total_rows(self, simple_df):
        result = compute_trigger_elevation(simple_df)
        assert result["n_obs"].sum() == len(simple_df)

    def test_elevated_columns_are_bool(self, simple_df):
        result = compute_trigger_elevation(simple_df)
        for key in SCORE_COLS:
            col = f"{key}_elevated"
            if col in result.columns:
                assert result[col].dtype == bool or result[col].apply(lambda x: isinstance(x, bool)).all()

    def test_mean_columns_within_score_range(self, simple_df):
        result = compute_trigger_elevation(simple_df)
        for key in SCORE_COLS:
            col = f"{key}_mean"
            if col in result.columns:
                assert (result[col] >= 0).all()
                assert (result[col] <= 100).all()

    def test_sorted_descending_by_n_obs(self, simple_df):
        result = compute_trigger_elevation(simple_df)
        n_obs = result["n_obs"].values
        assert list(n_obs) == sorted(n_obs, reverse=True)

    def test_missing_regime_col_returns_empty(self, simple_df):
        result = compute_trigger_elevation(simple_df, regime_col="nonexistent")
        assert result.empty

    def test_high_stress_regime_has_elevated_macro(self):
        """
        A regime that only fires at high macro scores should show macro elevated.

        90 neutral rows at score=20, 30 high-stress rows at score=90.
        Global 75th pct ≈ 37.5 (well below 90), so high-stress is elevated.
        """
        n_neutral, n_stress = 90, 30
        df = pd.DataFrame({
            "macro_risk_score_smooth": [20.0] * n_neutral + [90.0] * n_stress,
            "credit_market_risk_score_smooth":     [30.0] * (n_neutral + n_stress),
            "complacency_score_smooth":            [30.0] * (n_neutral + n_stress),
            "liquidity_regime_score_smooth":       [30.0] * (n_neutral + n_stress),
            "treasury_stress_score_smooth":        [30.0] * (n_neutral + n_stress),
            "mean_reversion_score_smooth":         [50.0] * (n_neutral + n_stress),
            "cross_asset_divergence_score_smooth": [30.0] * (n_neutral + n_stress),
            "final_decision": ["Neutral"] * n_neutral + ["High Stress"] * n_stress,
        })
        result = compute_trigger_elevation(df)
        assert result.loc["High Stress", "macro_risk_elevated"]
        assert not result.loc["Neutral", "macro_risk_elevated"]


# ---------------------------------------------------------------------------
# TestComputeTopDrivers
# ---------------------------------------------------------------------------


class TestComputeTopDrivers:
    def test_returns_list(self, simple_df):
        result = compute_top_drivers(simple_df.iloc[-1], simple_df)
        assert isinstance(result, list)

    def test_each_entry_is_dict(self, simple_df):
        result = compute_top_drivers(simple_df.iloc[-1], simple_df)
        assert all(isinstance(d, dict) for d in result)

    def test_required_keys(self, simple_df):
        result = compute_top_drivers(simple_df.iloc[-1], simple_df)
        required = {"name", "key", "level", "threshold_75", "excess", "elevated"}
        for entry in result:
            assert required.issubset(entry.keys())

    def test_sorted_descending_by_excess(self, simple_df):
        result = compute_top_drivers(simple_df.iloc[-1], simple_df)
        excesses = [d["excess"] for d in result]
        assert excesses == sorted(excesses, reverse=True)

    def test_elevated_consistent_with_excess(self, simple_df):
        result = compute_top_drivers(simple_df.iloc[-1], simple_df)
        for d in result:
            assert d["elevated"] == (d["excess"] > 0)

    def test_length_matches_available_scores(self, simple_df):
        result = compute_top_drivers(simple_df.iloc[-1], simple_df)
        # All 7 SCORE_COLS present in simple_df
        assert len(result) == len(SCORE_COLS)


# ---------------------------------------------------------------------------
# TestRunRegimeAttribution
# ---------------------------------------------------------------------------


class TestRunRegimeAttribution:
    def test_returns_dict(self, simple_df):
        result = run_regime_attribution(simple_df)
        assert isinstance(result, dict)

    def test_expected_keys(self, simple_df):
        result = run_regime_attribution(simple_df)
        for key in ["rolling_contributions", "shift_attribution",
                    "trigger_elevation", "top_drivers_current"]:
            assert key in result, f"Missing key: {key}"

    def test_rolling_contributions_length(self, simple_df):
        result = run_regime_attribution(simple_df)
        assert len(result["rolling_contributions"]) == len(simple_df)

    def test_shift_attribution_is_dataframe(self, simple_df):
        result = run_regime_attribution(simple_df)
        assert isinstance(result["shift_attribution"], pd.DataFrame)

    def test_trigger_elevation_is_dataframe(self, simple_df):
        result = run_regime_attribution(simple_df)
        assert isinstance(result["trigger_elevation"], pd.DataFrame)

    def test_top_drivers_is_list(self, simple_df):
        result = run_regime_attribution(simple_df)
        assert isinstance(result["top_drivers_current"], list)

    def test_integration_with_real_data(self, analytics_df):
        result = run_regime_attribution(analytics_df)
        assert not result["rolling_contributions"].empty
        assert not result["shift_attribution"].empty
        assert not result["trigger_elevation"].empty
        assert len(result["top_drivers_current"]) > 0
