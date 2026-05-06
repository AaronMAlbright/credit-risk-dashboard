"""
Tests for src/bootstrap.py.

All structural — no exact value assertions on random outputs.
CIs are verified to bracket the observed mean and be correctly ordered.
"""

import numpy as np
import pandas as pd
import pytest

from src.bootstrap import (
    DEFAULT_CI,
    MIN_OBS_REGIME,
    MIN_OBS_TRANSITION,
    _ci_bounds,
    bootstrap_hit_rates,
    bootstrap_regime_forward_returns,
    bootstrap_transition_probs,
    bootstrap_window_metrics,
    run_bootstrap_analysis,
)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_BOOT = 200  # fast for tests


@pytest.fixture(scope="module")
def windows_df():
    rng = np.random.default_rng(0)
    n = 12
    return pd.DataFrame({
        "strategy_sharpe": rng.normal(0.8, 0.3, n),
        "sp500_sharpe": rng.normal(0.5, 0.3, n),
        "strategy_total_return": rng.normal(0.10, 0.05, n),
        "sp500_total_return": rng.normal(0.07, 0.05, n),
        "strategy_max_drawdown": -rng.uniform(0.05, 0.20, n),
        "sp500_max_drawdown": -rng.uniform(0.08, 0.25, n),
        "strategy_hit_rate": rng.uniform(0.5, 0.8, n),
    })


@pytest.fixture(scope="module")
def scored_df():
    rng = np.random.default_rng(1)
    n = 200
    regimes = np.random.choice(
        ["Neutral", "Buy Stress", "Avoid Chasing Risk", "Credit Warning"],
        size=n, p=[0.45, 0.25, 0.20, 0.10],
    )
    df = pd.DataFrame({
        "final_decision": regimes,
        "transition_regime": np.random.choice(
            ["Stable / Neutral", "Deteriorating / Rising Stress", "Panic / Stabilizing"],
            size=n,
        ),
        "sp500_forward_30d_return": rng.normal(0.01, 0.05, n),
    })
    return df


@pytest.fixture(scope="module")
def sparse_windows_df():
    return pd.DataFrame({
        "strategy_sharpe": [0.5],
        "sp500_sharpe": [0.3],
    })


# ---------------------------------------------------------------------------
# Test _ci_bounds
# ---------------------------------------------------------------------------


class TestCiBounds:
    def test_95_ci(self):
        lo, hi = _ci_bounds(0.95)
        assert abs(lo - 2.5) < 1e-9
        assert abs(hi - 97.5) < 1e-9

    def test_90_ci(self):
        lo, hi = _ci_bounds(0.90)
        assert abs(lo - 5.0) < 1e-9
        assert abs(hi - 95.0) < 1e-9

    def test_lo_plus_hi_equals_100(self):
        for ci in [0.80, 0.90, 0.95, 0.99]:
            lo, hi = _ci_bounds(ci)
            assert abs(lo + hi - 100) < 1e-9


# ---------------------------------------------------------------------------
# TestBootstrapWindowMetrics
# ---------------------------------------------------------------------------


class TestBootstrapWindowMetrics:
    def test_returns_dataframe(self, windows_df):
        result = bootstrap_window_metrics(windows_df, n_boot=N_BOOT)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, windows_df):
        result = bootstrap_window_metrics(windows_df, n_boot=N_BOOT)
        assert {"mean", "ci_lower", "ci_upper", "n_windows", "flagged"}.issubset(result.columns)

    def test_ci_lower_le_mean_le_upper(self, windows_df):
        result = bootstrap_window_metrics(windows_df, n_boot=N_BOOT)
        valid = result.dropna(subset=["ci_lower", "ci_upper"])
        assert (valid["ci_lower"] <= valid["mean"]).all()
        assert (valid["mean"] <= valid["ci_upper"]).all()

    def test_n_windows_matches_input(self, windows_df):
        result = bootstrap_window_metrics(windows_df, n_boot=N_BOOT)
        assert (result["n_windows"] == len(windows_df)).all()

    def test_not_flagged_when_enough_windows(self, windows_df):
        result = bootstrap_window_metrics(windows_df, n_boot=N_BOOT)
        assert not result["flagged"].any()

    def test_flagged_when_single_window(self, sparse_windows_df):
        result = bootstrap_window_metrics(sparse_windows_df, n_boot=N_BOOT)
        assert result["flagged"].all()

    def test_ci_is_nan_for_single_window(self, sparse_windows_df):
        result = bootstrap_window_metrics(sparse_windows_df, n_boot=N_BOOT)
        assert result["ci_lower"].isna().all()
        assert result["ci_upper"].isna().all()

    def test_empty_input_returns_empty(self):
        result = bootstrap_window_metrics(pd.DataFrame(), n_boot=N_BOOT)
        assert result.empty

    def test_none_input_returns_empty(self):
        result = bootstrap_window_metrics(None, n_boot=N_BOOT)
        assert result.empty

    def test_index_is_metric_name(self, windows_df):
        result = bootstrap_window_metrics(windows_df, n_boot=N_BOOT)
        assert result.index.name == "metric"
        assert "strategy_sharpe" in result.index


# ---------------------------------------------------------------------------
# TestBootstrapRegimeForwardReturns
# ---------------------------------------------------------------------------


class TestBootstrapRegimeForwardReturns:
    def test_returns_dataframe(self, scored_df):
        result = bootstrap_regime_forward_returns(scored_df, "final_decision", n_boot=N_BOOT)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, scored_df):
        result = bootstrap_regime_forward_returns(scored_df, "final_decision", n_boot=N_BOOT)
        assert {"mean", "ci_lower", "ci_upper", "n_obs", "flagged"}.issubset(result.columns)

    def test_indexed_by_regime(self, scored_df):
        result = bootstrap_regime_forward_returns(scored_df, "final_decision", n_boot=N_BOOT)
        assert result.index.name == "regime"
        assert set(result.index).issubset(scored_df["final_decision"].unique())

    def test_ci_ordering(self, scored_df):
        result = bootstrap_regime_forward_returns(
            scored_df, "final_decision", n_boot=N_BOOT, min_obs=5
        )
        valid = result.dropna(subset=["ci_lower", "ci_upper"])
        assert (valid["ci_lower"] <= valid["mean"]).all()
        assert (valid["mean"] <= valid["ci_upper"]).all()

    def test_low_obs_regimes_flagged(self, scored_df):
        # Credit Warning has ~10% of 200 rows ≈ 20 obs < MIN_OBS_REGIME(30)
        result = bootstrap_regime_forward_returns(
            scored_df, "final_decision", n_boot=N_BOOT, min_obs=MIN_OBS_REGIME
        )
        assert result.loc["Credit Warning", "flagged"]

    def test_high_obs_regimes_not_flagged(self, scored_df):
        # Neutral has ~45% of 200 rows ≈ 90 obs
        result = bootstrap_regime_forward_returns(
            scored_df, "final_decision", n_boot=N_BOOT, min_obs=MIN_OBS_REGIME
        )
        assert not result.loc["Neutral", "flagged"]

    def test_missing_return_col_returns_empty(self, scored_df):
        result = bootstrap_regime_forward_returns(scored_df, "final_decision",
                                                   return_col="nonexistent", n_boot=N_BOOT)
        assert result.empty

    def test_missing_regime_col_returns_empty(self, scored_df):
        result = bootstrap_regime_forward_returns(scored_df, "nonexistent", n_boot=N_BOOT)
        assert result.empty


# ---------------------------------------------------------------------------
# TestBootstrapHitRates
# ---------------------------------------------------------------------------


class TestBootstrapHitRates:
    def test_returns_dataframe(self, scored_df):
        result = bootstrap_hit_rates(scored_df, "final_decision", n_boot=N_BOOT)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, scored_df):
        result = bootstrap_hit_rates(scored_df, "final_decision", n_boot=N_BOOT)
        assert {"hit_rate", "ci_lower", "ci_upper", "n_obs", "flagged"}.issubset(result.columns)

    def test_hit_rates_between_zero_and_one(self, scored_df):
        result = bootstrap_hit_rates(scored_df, "final_decision", n_boot=N_BOOT)
        valid = result["hit_rate"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_ci_bounds_between_zero_and_one(self, scored_df):
        result = bootstrap_hit_rates(
            scored_df, "final_decision", n_boot=N_BOOT, min_obs=5
        )
        valid = result.dropna(subset=["ci_lower", "ci_upper"])
        assert (valid["ci_lower"] >= 0).all()
        assert (valid["ci_upper"] <= 1).all()

    def test_ci_ordering(self, scored_df):
        result = bootstrap_hit_rates(
            scored_df, "final_decision", n_boot=N_BOOT, min_obs=5
        )
        valid = result.dropna(subset=["ci_lower", "ci_upper"])
        assert (valid["ci_lower"] <= valid["hit_rate"]).all()
        assert (valid["hit_rate"] <= valid["ci_upper"]).all()

    def test_all_positive_returns_hit_rate_one(self):
        df = pd.DataFrame({
            "final_decision": ["A"] * 50,
            "sp500_forward_30d_return": [0.01] * 50,
        })
        result = bootstrap_hit_rates(df, "final_decision", n_boot=N_BOOT, min_obs=10)
        assert result.loc["A", "hit_rate"] == pytest.approx(1.0)

    def test_missing_cols_returns_empty(self, scored_df):
        result = bootstrap_hit_rates(scored_df, "nonexistent", n_boot=N_BOOT)
        assert result.empty


# ---------------------------------------------------------------------------
# TestBootstrapTransitionProbs
# ---------------------------------------------------------------------------


class TestBootstrapTransitionProbs:
    def test_returns_dataframe(self, scored_df):
        result = bootstrap_transition_probs(scored_df["final_decision"], n_boot=N_BOOT)
        assert isinstance(result, pd.DataFrame)

    def test_multiindex_names(self, scored_df):
        result = bootstrap_transition_probs(scored_df["final_decision"], n_boot=N_BOOT)
        assert result.index.names == ["from_regime", "to_regime"]

    def test_expected_columns(self, scored_df):
        result = bootstrap_transition_probs(scored_df["final_decision"], n_boot=N_BOOT)
        assert {"mean_prob", "ci_lower", "ci_upper", "n_obs", "flagged"}.issubset(result.columns)

    def test_probs_between_zero_and_one(self, scored_df):
        result = bootstrap_transition_probs(scored_df["final_decision"], n_boot=N_BOOT)
        assert (result["mean_prob"] >= 0).all()
        assert (result["mean_prob"] <= 1).all()

    def test_ci_bounds_between_zero_and_one(self, scored_df):
        result = bootstrap_transition_probs(
            scored_df["final_decision"], n_boot=N_BOOT, min_obs=2
        )
        valid = result.dropna(subset=["ci_lower", "ci_upper"])
        assert (valid["ci_lower"] >= 0).all()
        assert (valid["ci_upper"] <= 1).all()

    def test_ci_ordering(self, scored_df):
        result = bootstrap_transition_probs(
            scored_df["final_decision"], n_boot=N_BOOT, min_obs=2
        )
        valid = result.dropna(subset=["ci_lower", "ci_upper"])
        assert (valid["ci_lower"] <= valid["mean_prob"]).all()
        assert (valid["mean_prob"] <= valid["ci_upper"]).all()

    def test_low_transition_count_flagged(self):
        # Force a very rare transition
        s = pd.Series(["A"] * 50 + ["B"] * 3 + ["A"] * 50)
        result = bootstrap_transition_probs(s, n_boot=N_BOOT, min_obs=MIN_OBS_TRANSITION)
        # B→A has only 3 outgoing transitions → flagged
        assert result.loc[("B", "A"), "flagged"]

    def test_high_count_transitions_not_flagged(self):
        s = pd.Series(["A", "B"] * 50)
        result = bootstrap_transition_probs(s, n_boot=N_BOOT, min_obs=MIN_OBS_TRANSITION)
        assert not result.loc[("A", "B"), "flagged"]

    def test_empty_series_returns_empty(self):
        result = bootstrap_transition_probs(pd.Series([], dtype=str), n_boot=N_BOOT)
        assert result.empty

    def test_single_element_series_returns_empty(self):
        result = bootstrap_transition_probs(pd.Series(["A"]), n_boot=N_BOOT)
        assert result.empty


# ---------------------------------------------------------------------------
# TestRunBootstrapAnalysis
# ---------------------------------------------------------------------------


class TestRunBootstrapAnalysis:
    def test_returns_dict(self, scored_df, windows_df):
        result = run_bootstrap_analysis(scored_df, windows_df, n_boot=N_BOOT)
        assert isinstance(result, dict)

    def test_expected_keys_present(self, scored_df, windows_df):
        result = run_bootstrap_analysis(scored_df, windows_df, n_boot=N_BOOT)
        for key in [
            "window_metrics",
            "regime_returns",
            "regime_hit_rates",
            "transition_probs_final_decision",
            "transition_probs_transition_regime",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_all_values_are_dataframes(self, scored_df, windows_df):
        result = run_bootstrap_analysis(scored_df, windows_df, n_boot=N_BOOT)
        for key, val in result.items():
            assert isinstance(val, pd.DataFrame), f"{key} is not a DataFrame"

    def test_no_windows_gives_empty_window_metrics(self, scored_df):
        result = run_bootstrap_analysis(scored_df, windows_df=None, n_boot=N_BOOT)
        assert result["window_metrics"].empty

    def test_regime_returns_has_correct_regimes(self, scored_df, windows_df):
        result = run_bootstrap_analysis(scored_df, windows_df, n_boot=N_BOOT)
        expected = set(scored_df["final_decision"].unique())
        assert set(result["regime_returns"].index) == expected

    def test_deterministic_with_seed(self, scored_df, windows_df):
        r1 = run_bootstrap_analysis(scored_df, windows_df, n_boot=N_BOOT)
        r2 = run_bootstrap_analysis(scored_df, windows_df, n_boot=N_BOOT)
        pd.testing.assert_frame_equal(r1["regime_returns"], r2["regime_returns"])

    def test_rng_state_restored_after_call(self, scored_df):
        np.random.seed(123)
        v_before = np.random.random()
        np.random.seed(123)
        run_bootstrap_analysis(scored_df, n_boot=10)
        v_after = np.random.random()
        assert abs(v_before - v_after) < 1e-12
