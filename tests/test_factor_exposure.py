"""
Tests for src/factor_exposure.py.

Covers OLS regression, rolling exposure, regime beta, return decomposition,
and the orchestrator — including edge cases and analytical properties.
"""

import numpy as np
import pandas as pd
import pytest

from src.factor_exposure import (
    _WINDOWS,
    _ols,
    compute_factor_regression,
    compute_regime_beta,
    compute_return_decomposition,
    compute_rolling_factor_exposure,
    run_factor_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(n=300, beta_true=0.6, alpha_true=0.0003, seed=42):
    """Synthetic DataFrame with known true beta and alpha."""
    rng = np.random.default_rng(seed)
    sp500 = rng.normal(0.0004, 0.010, n)
    noise = rng.normal(0.0, 0.005, n)
    strat = alpha_true + beta_true * sp500 + noise
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    regimes = (["Neutral"] * 120 + ["Avoid Chasing Risk"] * 80
               + ["Buy Stress"] * 60 + ["Wait"] * 40)
    return pd.DataFrame({
        "date":                   dates,
        "sp500_daily_return":     sp500,
        "strategy_daily_return":  strat,
        "final_decision":         regimes[:n],
    })


@pytest.fixture(scope="module")
def base_df():
    return _make_df()


@pytest.fixture(scope="module")
def known_beta_df():
    """Perfect beta=0.5, zero alpha, zero noise for exact tests."""
    rng = np.random.default_rng(7)
    sp500 = rng.normal(0.0004, 0.010, 200)
    strat = 0.5 * sp500
    dates = pd.date_range("2024-01-02", periods=200, freq="B")
    return pd.DataFrame({
        "date":                   dates,
        "sp500_daily_return":     sp500,
        "strategy_daily_return":  strat,
        "final_decision":         ["Neutral"] * 200,
    })


@pytest.fixture(scope="module")
def high_beta_df():
    """Beta ~1 strategy — near buy-and-hold."""
    rng = np.random.default_rng(8)
    sp500 = rng.normal(0.0004, 0.010, 200)
    strat = sp500 + rng.normal(0, 0.001, 200)
    dates = pd.date_range("2024-01-02", periods=200, freq="B")
    return pd.DataFrame({
        "date": dates,
        "sp500_daily_return":    sp500,
        "strategy_daily_return": strat,
        "final_decision":        ["Watch Entry"] * 200,
    })


@pytest.fixture(scope="module")
def low_beta_df():
    """Beta ~0.1 strategy — near cash."""
    rng = np.random.default_rng(9)
    sp500 = rng.normal(0.0004, 0.010, 200)
    strat = 0.1 * sp500 + rng.normal(0, 0.001, 200)
    dates = pd.date_range("2024-01-02", periods=200, freq="B")
    return pd.DataFrame({
        "date": dates,
        "sp500_daily_return":    sp500,
        "strategy_daily_return": strat,
        "final_decision":        ["Wait"] * 200,
    })


# ---------------------------------------------------------------------------
# Test _ols helper
# ---------------------------------------------------------------------------


class TestOlsHelper:
    def test_recovers_known_beta(self, known_beta_df):
        s = known_beta_df["strategy_daily_return"].values
        b = known_beta_df["sp500_daily_return"].values
        _, beta, _, _ = _ols(s, b)
        assert abs(beta - 0.5) < 0.01

    def test_recovers_known_alpha(self, known_beta_df):
        s = known_beta_df["strategy_daily_return"].values
        b = known_beta_df["sp500_daily_return"].values
        alpha, _, _, _ = _ols(s, b)
        assert abs(alpha) < 1e-5

    def test_r2_near_one_for_perfect_fit(self, known_beta_df):
        s = known_beta_df["strategy_daily_return"].values
        b = known_beta_df["sp500_daily_return"].values
        _, _, r2, _ = _ols(s, b)
        assert r2 > 0.999

    def test_nan_for_too_few_obs(self):
        rng = np.random.default_rng(1)
        y = rng.normal(0, 0.01, 5)
        x = rng.normal(0, 0.01, 5)
        a, be, r2, rs = _ols(y, x)
        assert all(np.isnan(v) for v in [a, be, r2, rs])

    def test_nan_for_zero_variance_x(self):
        y = np.random.default_rng(2).normal(0, 0.01, 50)
        x = np.full(50, 0.001)   # no variance
        a, be, r2, rs = _ols(y, x)
        assert np.isnan(be)

    def test_nan_values_in_input_skipped(self):
        rng = np.random.default_rng(3)
        y = rng.normal(0, 0.01, 100)
        x = rng.normal(0, 0.01, 100)
        y[::5] = np.nan
        a, be, r2, rs = _ols(y, x)
        assert not np.isnan(be)

    def test_residual_std_nonneg(self, base_df):
        s = base_df["strategy_daily_return"].values
        b = base_df["sp500_daily_return"].values
        _, _, _, rs = _ols(s, b)
        assert rs >= 0


# ---------------------------------------------------------------------------
# TestComputeFactorRegression
# ---------------------------------------------------------------------------


class TestComputeFactorRegression:
    def test_returns_dict(self, base_df):
        result = compute_factor_regression(base_df)
        assert isinstance(result, dict)

    def test_all_fields_present(self, base_df):
        result = compute_factor_regression(base_df)
        for f in ["beta", "alpha_daily", "ann_alpha", "r2",
                  "residual_vol", "info_ratio", "tracking_error", "n_obs"]:
            assert f in result, f"Missing field: {f}"

    def test_beta_in_valid_range(self, base_df):
        result = compute_factor_regression(base_df)
        b = result["beta"]
        assert -1.0 <= b <= 2.0

    def test_known_beta_recovered(self, known_beta_df):
        result = compute_factor_regression(known_beta_df)
        assert abs(result["beta"] - 0.5) < 0.02

    def test_r2_between_0_and_1(self, base_df):
        result = compute_factor_regression(base_df)
        assert 0.0 <= result["r2"] <= 1.0

    def test_r2_near_1_for_perfect_beta(self, known_beta_df):
        result = compute_factor_regression(known_beta_df)
        assert result["r2"] > 0.99

    def test_residual_vol_positive(self, base_df):
        result = compute_factor_regression(base_df)
        assert result["residual_vol"] > 0

    def test_tracking_error_positive(self, base_df):
        result = compute_factor_regression(base_df)
        assert result["tracking_error"] > 0

    def test_high_beta_strategy_has_beta_near_1(self, high_beta_df):
        result = compute_factor_regression(high_beta_df)
        assert result["beta"] > 0.9

    def test_low_beta_strategy_has_beta_near_0(self, low_beta_df):
        result = compute_factor_regression(low_beta_df)
        assert result["beta"] < 0.2

    def test_ann_alpha_equals_alpha_daily_times_252(self, base_df):
        result = compute_factor_regression(base_df)
        assert abs(result["ann_alpha"] - result["alpha_daily"] * 252) < 1e-4

    def test_missing_strategy_col_returns_empty(self, base_df):
        df2 = base_df.drop(columns=["strategy_daily_return"])
        result = compute_factor_regression(df2)
        assert result == {}

    def test_empty_df_returns_empty(self):
        result = compute_factor_regression(pd.DataFrame())
        assert result == {}

    def test_info_ratio_is_ann_alpha_over_resid_vol(self, base_df):
        result = compute_factor_regression(base_df)
        aa, rv, ir = result["ann_alpha"], result["residual_vol"], result["info_ratio"]
        if not any(np.isnan(v) for v in [aa, rv, ir]) and rv > 1e-10:
            assert abs(ir - aa / rv) < 1e-3


# ---------------------------------------------------------------------------
# TestComputeRollingFactorExposure
# ---------------------------------------------------------------------------


class TestComputeRollingFactorExposure:
    def test_returns_dict_keyed_by_window(self, base_df):
        result = compute_rolling_factor_exposure(base_df)
        for w in _WINDOWS:
            assert w in result

    def test_each_value_is_dataframe(self, base_df):
        result = compute_rolling_factor_exposure(base_df)
        for w in _WINDOWS:
            assert isinstance(result[w], pd.DataFrame)

    def test_expected_columns(self, base_df):
        result = compute_rolling_factor_exposure(base_df)
        for w in _WINDOWS:
            for col in ["beta", "ann_alpha", "r2"]:
                assert col in result[w].columns

    def test_length_matches_input(self, base_df):
        result = compute_rolling_factor_exposure(base_df)
        for w in _WINDOWS:
            assert len(result[w]) == len(base_df)

    def test_first_rows_nan(self, base_df):
        w = _WINDOWS[0]
        result = compute_rolling_factor_exposure(base_df, windows=[w])
        assert result[w]["beta"].iloc[:w - 1].isna().all()

    def test_later_rows_not_all_nan(self, base_df):
        w = _WINDOWS[0]
        result = compute_rolling_factor_exposure(base_df, windows=[w])
        assert result[w]["beta"].iloc[w:].notna().any()

    def test_rolling_beta_near_known_value(self, known_beta_df):
        w = _WINDOWS[0]
        result = compute_rolling_factor_exposure(known_beta_df, windows=[w])
        betas = result[w]["beta"].dropna()
        assert (betas - 0.5).abs().mean() < 0.05

    def test_r2_in_01_range(self, base_df):
        result = compute_rolling_factor_exposure(base_df)
        for w in _WINDOWS:
            r2 = result[w]["r2"].dropna()
            assert ((r2 >= 0) & (r2 <= 1)).all()

    def test_index_is_datetime(self, base_df):
        result = compute_rolling_factor_exposure(base_df)
        assert pd.api.types.is_datetime64_any_dtype(result[_WINDOWS[0]].index)

    def test_empty_df_returns_empty_frames(self):
        result = compute_rolling_factor_exposure(pd.DataFrame())
        for w in _WINDOWS:
            assert result[w].empty

    def test_custom_window(self, base_df):
        result = compute_rolling_factor_exposure(base_df, windows=[30])
        assert 30 in result
        assert result[30]["beta"].iloc[:29].isna().all()


# ---------------------------------------------------------------------------
# TestComputeRegimeBeta
# ---------------------------------------------------------------------------


class TestComputeRegimeBeta:
    def test_returns_dataframe(self, base_df):
        result = compute_regime_beta(base_df)
        assert isinstance(result, pd.DataFrame)

    def test_all_regimes_in_index(self, base_df):
        result = compute_regime_beta(base_df)
        for regime in base_df["final_decision"].unique():
            assert regime in result.index

    def test_expected_columns(self, base_df):
        result = compute_regime_beta(base_df)
        for col in ["n_obs", "beta", "ann_alpha", "r2", "residual_vol"]:
            assert col in result.columns

    def test_sorted_by_beta_ascending(self, base_df):
        result = compute_regime_beta(base_df)
        betas = result["beta"].dropna().tolist()
        assert betas == sorted(betas)

    def test_n_obs_sums_to_total(self, base_df):
        result = compute_regime_beta(base_df)
        assert result["n_obs"].sum() == len(base_df)

    def test_beta_in_range(self, base_df):
        result = compute_regime_beta(base_df)
        betas = result["beta"].dropna()
        assert (betas >= -1.0).all() and (betas <= 2.0).all()

    def test_missing_decision_col_returns_empty(self, base_df):
        df2 = base_df.drop(columns=["final_decision"])
        result = compute_regime_beta(df2)
        assert result.empty

    def test_empty_df_returns_empty(self):
        result = compute_regime_beta(pd.DataFrame())
        assert result.empty

    def test_high_beta_regime_has_higher_beta(self, base_df):
        # "Buy Stress" fixture should have higher beta than "Wait"
        result = compute_regime_beta(base_df)
        if "Buy Stress" in result.index and "Wait" in result.index:
            b_buy   = result.loc["Buy Stress", "beta"]
            b_wait  = result.loc["Wait", "beta"]
            if not (np.isnan(b_buy) or np.isnan(b_wait)):
                assert b_buy >= b_wait


# ---------------------------------------------------------------------------
# TestComputeReturnDecomposition
# ---------------------------------------------------------------------------


class TestComputeReturnDecomposition:
    def test_returns_dataframe(self, base_df):
        result = compute_return_decomposition(base_df)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, base_df):
        result = compute_return_decomposition(base_df)
        for col in ["strategy_return", "sp500_return",
                    "beta_contribution", "alpha_contribution",
                    "cum_strategy", "cum_sp500", "cum_beta", "cum_alpha"]:
            assert col in result.columns

    def test_length_matches_input(self, base_df):
        result = compute_return_decomposition(base_df)
        assert len(result) == len(base_df)

    def test_beta_plus_alpha_equals_strategy(self, base_df):
        result = compute_return_decomposition(base_df)
        diff = (result["beta_contribution"] + result["alpha_contribution"]
                - result["strategy_return"]).abs()
        assert (diff < 1e-9).all()

    def test_cum_series_start_near_1(self, base_df):
        result = compute_return_decomposition(base_df)
        for col in ["cum_strategy", "cum_sp500", "cum_beta", "cum_alpha"]:
            assert abs(result[col].iloc[0] - 1.0) < 0.05

    def test_cum_series_all_positive(self, base_df):
        result = compute_return_decomposition(base_df)
        for col in ["cum_strategy", "cum_sp500", "cum_beta"]:
            assert (result[col] > 0).all()

    def test_perfect_beta_zero_alpha_contribution(self, known_beta_df):
        result = compute_return_decomposition(known_beta_df)
        # alpha contribution should be near zero for perfect-fit fixture
        alpha_std = result["alpha_contribution"].std()
        beta_std  = result["beta_contribution"].std()
        assert alpha_std < beta_std * 0.05

    def test_missing_sp500_col_returns_empty(self, base_df):
        df2 = base_df.drop(columns=["sp500_daily_return"])
        result = compute_return_decomposition(df2)
        assert result.empty

    def test_empty_df_returns_empty(self):
        result = compute_return_decomposition(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# TestRunFactorAnalysis
# ---------------------------------------------------------------------------


class TestRunFactorAnalysis:
    def test_returns_dict(self, base_df):
        result = run_factor_analysis(base_df)
        assert isinstance(result, dict)

    def test_all_keys_present(self, base_df):
        result = run_factor_analysis(base_df)
        for key in ["regression", "rolling", "regime_beta",
                    "decomposition", "windows"]:
            assert key in result

    def test_regression_is_dict(self, base_df):
        result = run_factor_analysis(base_df)
        assert isinstance(result["regression"], dict)

    def test_rolling_has_expected_windows(self, base_df):
        result = run_factor_analysis(base_df)
        for w in _WINDOWS:
            assert w in result["rolling"]

    def test_regime_beta_is_dataframe(self, base_df):
        result = run_factor_analysis(base_df)
        assert isinstance(result["regime_beta"], pd.DataFrame)

    def test_decomposition_is_dataframe(self, base_df):
        result = run_factor_analysis(base_df)
        assert isinstance(result["decomposition"], pd.DataFrame)

    def test_windows_list_correct(self, base_df):
        result = run_factor_analysis(base_df)
        assert result["windows"] == _WINDOWS

    def test_empty_df_graceful(self):
        result = run_factor_analysis(pd.DataFrame())
        assert result["regression"] == {}
        assert result["regime_beta"].empty
        assert result["decomposition"].empty
