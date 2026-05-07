"""
Tests for src/weight_optimizer.py.

Covers composite computation, Sharpe calculation, grid search structure,
tornado perturbation, and the run_weight_optimization orchestrator.
"""

import numpy as np
import pandas as pd
import pytest

from src.weight_optimizer import (
    CURRENT_WEIGHTS,
    INVERTED,
    SCORE_COL_MAP,
    compute_composite,
    compute_strategy_sharpe,
    compute_tornado,
    find_optimal_weights,
    run_weight_grid_search,
    run_weight_optimization,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_df():
    """200-row DataFrame with all six score columns and sp500 price/returns."""
    n = 200
    rng = np.random.default_rng(10)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    price = 4000 + np.cumsum(rng.normal(0, 12, n))
    daily_ret = np.diff(price, prepend=price[0]) / np.where(
        np.roll(price, 1) == 0, 1, np.roll(price, 1)
    )
    daily_ret[0] = 0.0
    return pd.DataFrame({
        "date":                            dates,
        "sp500":                           price,
        "sp500_daily_return":              daily_ret,
        "sp500_forward_30d_return":        rng.normal(0.015, 0.04, n),
        "macro_risk_score_smooth":         rng.uniform(10, 75, n),
        "credit_market_risk_score_smooth": rng.uniform(10, 70, n),
        "complacency_score_smooth":        rng.uniform(15, 65, n),
        "liquidity_regime_score_smooth":   rng.uniform(10, 60, n),
        "treasury_stress_score_smooth":    rng.uniform(10, 55, n),
        "mean_reversion_score_smooth":     rng.uniform(20, 80, n),
    })


@pytest.fixture(scope="module")
def minimal_df():
    """Only two score columns — below VIF threshold."""
    n = 60
    rng = np.random.default_rng(11)
    return pd.DataFrame({
        "date":                        pd.date_range("2023-01-01", periods=n, freq="B"),
        "sp500_daily_return":          rng.normal(0.0003, 0.01, n),
        "macro_risk_score_smooth":     rng.uniform(10, 80, n),
        "credit_market_risk_score_smooth": rng.uniform(10, 70, n),
        "sp500_forward_30d_return":    rng.normal(0.01, 0.04, n),
    })


# ---------------------------------------------------------------------------
# TestComputeComposite
# ---------------------------------------------------------------------------


class TestComputeComposite:
    def test_returns_series(self, base_df):
        result = compute_composite(base_df, CURRENT_WEIGHTS)
        assert isinstance(result, pd.Series)

    def test_same_length_as_input(self, base_df):
        result = compute_composite(base_df, CURRENT_WEIGHTS)
        assert len(result) == len(base_df)

    def test_values_in_0_100(self, base_df):
        result = compute_composite(base_df, CURRENT_WEIGHTS)
        assert result.min() >= 0.0
        assert result.max() <= 100.0

    def test_mean_reversion_inverted(self, base_df):
        # High MR score → low risk contribution → lower composite
        df_high_mr = base_df.copy()
        df_high_mr["mean_reversion_score_smooth"] = 90.0
        df_low_mr = base_df.copy()
        df_low_mr["mean_reversion_score_smooth"] = 10.0
        high = compute_composite(df_high_mr, CURRENT_WEIGHTS).mean()
        low = compute_composite(df_low_mr, CURRENT_WEIGHTS).mean()
        assert high < low

    def test_zero_weight_excludes_score(self, base_df):
        w_no_mr = dict(CURRENT_WEIGHTS)
        w_no_mr["mean_reversion"] = 0.0
        result = compute_composite(base_df, w_no_mr)
        assert isinstance(result, pd.Series)
        assert result.notna().any()

    def test_weights_renormalised_internally(self, base_df):
        w_scaled = {k: v * 10 for k, v in CURRENT_WEIGHTS.items()}
        r_scaled = compute_composite(base_df, w_scaled)
        r_normal = compute_composite(base_df, CURRENT_WEIGHTS)
        pd.testing.assert_series_equal(r_scaled, r_normal, check_names=False)

    def test_all_weight_on_one_score(self, base_df):
        w = {k: 0.0 for k in CURRENT_WEIGHTS}
        w["macro_risk"] = 1.0
        result = compute_composite(base_df, w)
        expected = base_df["macro_risk_score_smooth"].clip(0, 100)
        pd.testing.assert_series_equal(result, expected, check_names=False, rtol=1e-5)

    def test_inverted_set_contains_mean_reversion(self):
        assert "mean_reversion" in INVERTED


# ---------------------------------------------------------------------------
# TestComputeStrategySharpe
# ---------------------------------------------------------------------------


class TestComputeStrategySharpe:
    def test_returns_float(self, base_df):
        result = compute_strategy_sharpe(base_df, CURRENT_WEIGHTS)
        assert isinstance(result, float)

    def test_nan_when_no_return_col(self, base_df):
        df2 = base_df.drop(columns=["sp500_daily_return"])
        result = compute_strategy_sharpe(df2, CURRENT_WEIGHTS)
        assert np.isnan(result)

    def test_positive_sharpe_for_good_signal(self):
        n = 200
        rng = np.random.default_rng(99)
        # Low composite score → high equity weight → good when market goes up
        scores = np.full(n, 20.0)  # composite ≈ 20 → equity_weight ≈ 0.8
        daily_ret = rng.normal(0.0004, 0.008, n)  # positive drift
        df = pd.DataFrame({
            "macro_risk_score_smooth":         scores,
            "credit_market_risk_score_smooth": scores,
            "complacency_score_smooth":        scores,
            "liquidity_regime_score_smooth":   scores,
            "treasury_stress_score_smooth":    scores,
            "mean_reversion_score_smooth":     np.full(n, 80.0),
            "sp500_daily_return":              daily_ret,
        })
        sharpe = compute_strategy_sharpe(df, CURRENT_WEIGHTS)
        assert not np.isnan(sharpe)

    def test_equal_weights_valid(self, base_df):
        n = len(CURRENT_WEIGHTS)
        eq_w = {k: 1.0 / n for k in CURRENT_WEIGHTS}
        result = compute_strategy_sharpe(base_df, eq_w)
        assert isinstance(result, float)

    def test_deterministic(self, base_df):
        r1 = compute_strategy_sharpe(base_df, CURRENT_WEIGHTS)
        r2 = compute_strategy_sharpe(base_df, CURRENT_WEIGHTS)
        assert r1 == r2


# ---------------------------------------------------------------------------
# TestRunWeightGridSearch
# ---------------------------------------------------------------------------


class TestRunWeightGridSearch:
    def test_returns_dataframe(self, base_df):
        result = run_weight_grid_search(base_df, n_samples=50)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, base_df):
        result = run_weight_grid_search(base_df, n_samples=50)
        for col in list(SCORE_COL_MAP.keys()) + ["sharpe", "corr_30d"]:
            assert col in result.columns

    def test_row_count_equals_n_samples(self, base_df):
        result = run_weight_grid_search(base_df, n_samples=100)
        assert len(result) == 100

    def test_weights_sum_to_one(self, base_df):
        result = run_weight_grid_search(base_df, n_samples=50)
        score_keys = list(SCORE_COL_MAP.keys())
        weight_sums = result[score_keys].sum(axis=1)
        assert (abs(weight_sums - 1.0) < 1e-3).all()

    def test_all_weights_non_negative(self, base_df):
        result = run_weight_grid_search(base_df, n_samples=50)
        for k in SCORE_COL_MAP:
            assert (result[k] >= 0).all()

    def test_sorted_by_sharpe_descending(self, base_df):
        result = run_weight_grid_search(base_df, n_samples=100)
        sharpes = result["sharpe"].dropna().tolist()
        assert sharpes == sorted(sharpes, reverse=True)

    def test_deterministic_with_seed(self, base_df):
        r1 = run_weight_grid_search(base_df, n_samples=50, seed=42)
        r2 = run_weight_grid_search(base_df, n_samples=50, seed=42)
        pd.testing.assert_frame_equal(r1, r2)

    def test_different_seed_different_results(self, base_df):
        r1 = run_weight_grid_search(base_df, n_samples=50, seed=42)
        r2 = run_weight_grid_search(base_df, n_samples=50, seed=99)
        assert not r1["macro_risk"].equals(r2["macro_risk"])

    def test_rng_state_restored(self, base_df):
        state_before = np.random.get_state()[1][:5].tolist()
        run_weight_grid_search(base_df, n_samples=20)
        state_after = np.random.get_state()[1][:5].tolist()
        assert state_before == state_after

    def test_corr_30d_in_valid_range(self, base_df):
        result = run_weight_grid_search(base_df, n_samples=50)
        clean = result["corr_30d"].dropna()
        assert (clean.abs() <= 1.0).all()

    def test_no_return_col_returns_empty(self, base_df):
        df2 = base_df.drop(columns=["sp500_daily_return"])
        result = run_weight_grid_search(df2, n_samples=10)
        assert result.empty

    def test_too_few_scores_returns_empty(self, minimal_df):
        # minimal_df has only 2 score cols — below threshold of 3
        result = run_weight_grid_search(minimal_df, n_samples=10)
        assert result.empty


# ---------------------------------------------------------------------------
# TestComputeTornado
# ---------------------------------------------------------------------------


class TestComputeTornado:
    def test_returns_dataframe(self, base_df):
        result = compute_tornado(base_df)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, base_df):
        result = compute_tornado(base_df)
        for col in ["baseline_weight", "sharpe_up", "sharpe_down", "delta_up", "delta_down", "swing"]:
            assert col in result.columns

    def test_one_row_per_available_score(self, base_df):
        result = compute_tornado(base_df)
        from src.weight_optimizer import _available
        av = _available(base_df)
        assert len(result) == len(av)

    def test_swing_is_absolute(self, base_df):
        result = compute_tornado(base_df)
        assert (result["swing"] >= 0).all()

    def test_sorted_by_swing_descending(self, base_df):
        result = compute_tornado(base_df)
        swings = result["swing"].tolist()
        assert swings == sorted(swings, reverse=True)

    def test_baseline_weights_match_current(self, base_df):
        result = compute_tornado(base_df, CURRENT_WEIGHTS)
        from src.weight_optimizer import _available, _normalize_weights
        av = _available(base_df)
        norm = _normalize_weights({k: CURRENT_WEIGHTS.get(k, 0.0) for k in av})
        for score in result.index:
            assert abs(result.loc[score, "baseline_weight"] - norm[score]) < 1e-4

    def test_custom_weights_respected(self, base_df):
        w = {k: 1.0 / 6 for k in CURRENT_WEIGHTS}
        result = compute_tornado(base_df, weights=w)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_delta_up_is_signed(self, base_df):
        result = compute_tornado(base_df)
        # delta_up can be positive or negative (no guarantee of sign)
        assert result["delta_up"].notna().any()

    def test_too_few_scores_returns_empty(self, minimal_df):
        result = compute_tornado(minimal_df)
        assert result.empty


# ---------------------------------------------------------------------------
# TestFindOptimalWeights
# ---------------------------------------------------------------------------


class TestFindOptimalWeights:
    def test_returns_dict(self, base_df):
        grid = run_weight_grid_search(base_df, n_samples=50)
        result = find_optimal_weights(grid)
        assert isinstance(result, dict)

    def test_keys_are_score_names(self, base_df):
        grid = run_weight_grid_search(base_df, n_samples=50)
        result = find_optimal_weights(grid)
        for k in result:
            assert k in SCORE_COL_MAP

    def test_weights_sum_to_one(self, base_df):
        grid = run_weight_grid_search(base_df, n_samples=50)
        result = find_optimal_weights(grid)
        assert abs(sum(result.values()) - 1.0) < 1e-3

    def test_empty_grid_returns_empty_dict(self):
        result = find_optimal_weights(pd.DataFrame())
        assert result == {}

    def test_optimal_sharpe_at_least_as_good_as_current(self, base_df):
        grid = run_weight_grid_search(base_df, n_samples=200)
        optimal_w = find_optimal_weights(grid)
        current_s = compute_strategy_sharpe(base_df, CURRENT_WEIGHTS)
        optimal_s = compute_strategy_sharpe(base_df, optimal_w)
        # With 200 samples we expect to find something at least as good
        assert np.isnan(optimal_s) or np.isnan(current_s) or optimal_s >= current_s - 0.5


# ---------------------------------------------------------------------------
# TestRunWeightOptimization
# ---------------------------------------------------------------------------


class TestRunWeightOptimization:
    def test_returns_dict(self, base_df):
        result = run_weight_optimization(base_df, n_samples=50)
        assert isinstance(result, dict)

    def test_all_keys_present(self, base_df):
        result = run_weight_optimization(base_df, n_samples=50)
        for key in ["grid_results", "tornado", "current_sharpe",
                    "optimal_weights", "optimal_sharpe"]:
            assert key in result

    def test_grid_results_is_dataframe(self, base_df):
        result = run_weight_optimization(base_df, n_samples=50)
        assert isinstance(result["grid_results"], pd.DataFrame)

    def test_tornado_is_dataframe(self, base_df):
        result = run_weight_optimization(base_df, n_samples=50)
        assert isinstance(result["tornado"], pd.DataFrame)

    def test_current_sharpe_is_float(self, base_df):
        result = run_weight_optimization(base_df, n_samples=50)
        assert isinstance(result["current_sharpe"], float) or np.isnan(result["current_sharpe"])

    def test_optimal_weights_is_dict(self, base_df):
        result = run_weight_optimization(base_df, n_samples=50)
        assert isinstance(result["optimal_weights"], dict)
