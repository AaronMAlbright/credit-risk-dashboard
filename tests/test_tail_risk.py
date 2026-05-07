"""
Tests for src/tail_risk.py.

Covers point estimators, per-regime tail stats, rolling CVaR,
drawdown profiles, strategy vs benchmark, and the orchestrator.
"""

import numpy as np
import pandas as pd
import pytest

from src.tail_risk import (
    DEFAULT_ALPHA,
    _MIN_OBS_BOOT,
    _MIN_OBS_CVAR,
    compute_drawdown_by_regime,
    compute_regime_tail_stats,
    compute_rolling_cvar,
    compute_strategy_vs_sp500,
    compute_var_cvar,
    run_tail_risk_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_df():
    """300-row DataFrame with daily returns, drawdowns, and two regimes."""
    n = 300
    rng = np.random.default_rng(20)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    sp500_ret = rng.normal(0.0003, 0.010, n)
    # Strategy: hedged in second half (lower equity weight)
    strat_ret = np.concatenate([
        rng.normal(0.0004, 0.010, 150),
        rng.normal(0.0002, 0.006, 150),   # lower vol in second regime
    ])
    decisions = ["Neutral"] * 150 + ["Avoid Chasing Risk"] * 150
    return pd.DataFrame({
        "date":                    dates,
        "sp500_daily_return":      sp500_ret,
        "strategy_daily_return":   strat_ret,
        "final_decision":          decisions,
        "sp500_future_drawdown_30d": rng.uniform(-0.15, 0.0, n),
        "sp500_future_drawdown_60d": rng.uniform(-0.20, 0.0, n),
    })


@pytest.fixture(scope="module")
def fat_tail_returns():
    """Returns with pronounced fat left tail."""
    rng = np.random.default_rng(21)
    normal = rng.normal(0.001, 0.008, 200)
    shocks = rng.uniform(-0.08, -0.04, 10)
    return np.concatenate([normal, shocks])


@pytest.fixture(scope="module")
def tiny_df():
    """5-row DataFrame — below _MIN_OBS_CVAR."""
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=5, freq="B"),
        "sp500_daily_return": [0.01, -0.02, 0.005, -0.015, 0.003],
        "final_decision": ["Neutral"] * 5,
    })


# ---------------------------------------------------------------------------
# TestComputeVarCvar
# ---------------------------------------------------------------------------


class TestComputeVarCvar:
    def test_returns_two_floats(self, fat_tail_returns):
        var, cvar = compute_var_cvar(fat_tail_returns)
        assert isinstance(var, float)
        assert isinstance(cvar, float)

    def test_cvar_le_var(self, fat_tail_returns):
        var, cvar = compute_var_cvar(fat_tail_returns)
        assert cvar <= var

    def test_both_negative_for_loss_heavy_returns(self, fat_tail_returns):
        # Fat tail fixture has many losses — VaR and CVaR should be negative
        var, cvar = compute_var_cvar(fat_tail_returns)
        assert var < 0
        assert cvar < 0

    def test_nan_for_too_few_obs(self):
        var, cvar = compute_var_cvar(np.array([0.01, -0.01]))
        assert np.isnan(var)
        assert np.isnan(cvar)

    def test_nan_propagation_with_nans_in_input(self, fat_tail_returns):
        with_nan = np.concatenate([fat_tail_returns, [np.nan, np.nan]])
        var, cvar = compute_var_cvar(with_nan)
        assert not np.isnan(var)

    def test_alpha_boundary_5pct(self, fat_tail_returns):
        var_5, cvar_5 = compute_var_cvar(fat_tail_returns, alpha=0.05)
        var_1, cvar_1 = compute_var_cvar(fat_tail_returns, alpha=0.01)
        # 1% VaR should be worse (more negative) than 5% VaR
        assert var_1 <= var_5
        assert cvar_1 <= cvar_5

    def test_cvar_below_or_equal_var_always(self):
        rng = np.random.default_rng(99)
        for _ in range(10):
            r = rng.normal(0, 0.01, 100)
            var, cvar = compute_var_cvar(r)
            assert cvar <= var + 1e-9

    def test_positive_returns_positive_var(self):
        r = np.full(50, 0.01)  # all positive, no left tail
        var, cvar = compute_var_cvar(r)
        assert var > 0


# ---------------------------------------------------------------------------
# TestComputeRegimeTailStats
# ---------------------------------------------------------------------------


class TestComputeRegimeTailStats:
    def test_returns_dataframe(self, base_df):
        result = compute_regime_tail_stats(base_df)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, base_df):
        result = compute_regime_tail_stats(base_df)
        for col in ["n_obs", "mean_return", "var_95", "cvar_95", "max_loss"]:
            assert col in result.columns

    def test_both_regimes_present(self, base_df):
        result = compute_regime_tail_stats(base_df)
        assert "Neutral" in result.index
        assert "Avoid Chasing Risk" in result.index

    def test_cvar_le_var_per_regime(self, base_df):
        result = compute_regime_tail_stats(base_df)
        clean = result.dropna(subset=["var_95", "cvar_95"])
        assert (clean["cvar_95"] <= clean["var_95"] + 1e-9).all()

    def test_n_obs_correct(self, base_df):
        result = compute_regime_tail_stats(base_df)
        assert result.loc["Neutral", "n_obs"] == 150
        assert result.loc["Avoid Chasing Risk", "n_obs"] == 150

    def test_sorted_by_cvar_ascending(self, base_df):
        result = compute_regime_tail_stats(base_df)
        cvars = result["cvar_95"].dropna().tolist()
        assert cvars == sorted(cvars)

    def test_ci_ordering(self, base_df):
        result = compute_regime_tail_stats(base_df)
        clean = result.dropna(subset=["ci_lower", "ci_upper", "cvar_95"])
        assert (clean["ci_lower"] <= clean["cvar_95"] + 1e-9).all()
        assert (clean["cvar_95"] <= clean["ci_upper"] + 1e-9).all()

    def test_max_loss_is_worst_daily_return(self, base_df):
        result = compute_regime_tail_stats(base_df)
        for regime, grp in base_df.groupby("final_decision"):
            worst = grp["sp500_daily_return"].min()
            assert abs(result.loc[regime, "max_loss"] - worst) < 1e-4  # rounded to 5dp

    def test_missing_regime_col_returns_empty(self, base_df):
        result = compute_regime_tail_stats(base_df, regime_col="nonexistent")
        assert result.empty

    def test_missing_return_col_returns_empty(self, base_df):
        result = compute_regime_tail_stats(base_df, return_col="nonexistent")
        assert result.empty

    def test_sparse_regime_has_nan_ci(self, tiny_df):
        result = compute_regime_tail_stats(tiny_df, n_boot=50)
        if not result.empty:
            clean = result.dropna(subset=["ci_lower"])
            assert len(clean) == 0  # all CIs NaN (n < _MIN_OBS_BOOT)

    def test_deterministic(self, base_df):
        r1 = compute_regime_tail_stats(base_df, n_boot=100)
        r2 = compute_regime_tail_stats(base_df, n_boot=100)
        pd.testing.assert_frame_equal(r1, r2)

    def test_rng_state_restored(self, base_df):
        before = np.random.get_state()[1][:5].tolist()
        compute_regime_tail_stats(base_df, n_boot=50)
        after = np.random.get_state()[1][:5].tolist()
        assert before == after


# ---------------------------------------------------------------------------
# TestComputeRollingCvar
# ---------------------------------------------------------------------------


class TestComputeRollingCvar:
    def test_returns_series(self, base_df):
        result = compute_rolling_cvar(base_df)
        assert isinstance(result, pd.Series)

    def test_same_length_as_input(self, base_df):
        result = compute_rolling_cvar(base_df)
        assert len(result) == len(base_df)

    def test_first_rows_are_nan(self, base_df):
        window = 60
        result = compute_rolling_cvar(base_df, window=window)
        assert result.iloc[:window - 1].isna().all()

    def test_later_rows_are_not_all_nan(self, base_df):
        result = compute_rolling_cvar(base_df, window=60)
        assert result.iloc[60:].notna().any()

    def test_values_are_negative_or_nan(self, base_df):
        result = compute_rolling_cvar(base_df)
        non_nan = result.dropna()
        assert (non_nan <= 0).all()

    def test_index_is_datetime_when_date_col_present(self, base_df):
        result = compute_rolling_cvar(base_df)
        assert pd.api.types.is_datetime64_any_dtype(result.index)

    def test_missing_return_col_returns_empty(self, base_df):
        result = compute_rolling_cvar(base_df, return_col="nonexistent")
        assert result.empty

    def test_empty_df_returns_empty(self):
        result = compute_rolling_cvar(pd.DataFrame())
        assert result.empty

    def test_tighter_alpha_gives_worse_cvar(self, base_df):
        cvar_5 = compute_rolling_cvar(base_df, alpha=0.05).dropna()
        cvar_1 = compute_rolling_cvar(base_df, alpha=0.01).dropna()
        # 1% CVaR should be ≤ 5% CVaR (more extreme tail)
        assert (cvar_1 <= cvar_5 + 1e-9).all()


# ---------------------------------------------------------------------------
# TestComputeDrawdownByRegime
# ---------------------------------------------------------------------------


class TestComputeDrawdownByRegime:
    def test_returns_dataframe(self, base_df):
        result = compute_drawdown_by_regime(base_df)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, base_df):
        result = compute_drawdown_by_regime(base_df)
        assert "n_obs" in result.columns
        assert any("dd" in c for c in result.columns)

    def test_both_regimes_present(self, base_df):
        result = compute_drawdown_by_regime(base_df)
        assert "Neutral" in result.index
        assert "Avoid Chasing Risk" in result.index

    def test_drawdowns_non_positive(self, base_df):
        result = compute_drawdown_by_regime(base_df)
        dd_cols = [c for c in result.columns if "dd" in c]
        for col in dd_cols:
            assert (result[col].dropna() <= 0).all()

    def test_worst_le_mean_drawdown(self, base_df):
        result = compute_drawdown_by_regime(base_df)
        if "worst_dd_30d" in result.columns and "mean_dd_30d" in result.columns:
            clean = result.dropna(subset=["worst_dd_30d", "mean_dd_30d"])
            assert (clean["worst_dd_30d"] <= clean["mean_dd_30d"] + 1e-9).all()

    def test_missing_regime_col_returns_empty(self, base_df):
        result = compute_drawdown_by_regime(base_df, regime_col="nonexistent")
        assert result.empty

    def test_no_drawdown_cols_returns_empty(self):
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=20, freq="B"),
            "final_decision": ["Neutral"] * 20,
            "sp500_daily_return": np.random.default_rng(5).normal(0, 0.01, 20),
        })
        result = compute_drawdown_by_regime(df)
        assert result.empty


# ---------------------------------------------------------------------------
# TestComputeStrategyVsSP500
# ---------------------------------------------------------------------------


class TestComputeStrategyVsSP500:
    def test_returns_dict(self, base_df):
        result = compute_strategy_vs_sp500(base_df)
        assert isinstance(result, dict)

    def test_both_keys_present(self, base_df):
        result = compute_strategy_vs_sp500(base_df)
        assert "strategy" in result
        assert "sp500" in result

    def test_each_entry_has_expected_fields(self, base_df):
        result = compute_strategy_vs_sp500(base_df)
        for label in ["strategy", "sp500"]:
            for field in ["n_obs", "var", "cvar", "max_loss", "ann_vol"]:
                assert field in result[label]

    def test_cvar_le_var_for_both(self, base_df):
        result = compute_strategy_vs_sp500(base_df)
        for label in ["strategy", "sp500"]:
            v, c = result[label]["var"], result[label]["cvar"]
            if not (np.isnan(v) or np.isnan(c)):
                assert c <= v + 1e-9

    def test_strategy_better_tail_than_sp500_when_hedged(self, base_df):
        # Strategy has lower vol in "Avoid Chasing Risk" regime → better CVaR
        result = compute_strategy_vs_sp500(base_df)
        s_cvar = result["strategy"]["cvar"]
        b_cvar = result["sp500"]["cvar"]
        if not (np.isnan(s_cvar) or np.isnan(b_cvar)):
            assert s_cvar >= b_cvar  # strategy CVaR less negative = better tail

    def test_missing_strategy_col_only_sp500(self):
        rng = np.random.default_rng(30)
        df = pd.DataFrame({
            "sp500_daily_return": rng.normal(0, 0.01, 100),
        })
        result = compute_strategy_vs_sp500(df)
        assert "sp500" in result
        assert "strategy" not in result

    def test_ann_vol_positive(self, base_df):
        result = compute_strategy_vs_sp500(base_df)
        for label in ["strategy", "sp500"]:
            if label in result:
                vol = result[label]["ann_vol"]
                assert np.isnan(vol) or vol > 0

    def test_deterministic(self, base_df):
        r1 = compute_strategy_vs_sp500(base_df, n_boot=100)
        r2 = compute_strategy_vs_sp500(base_df, n_boot=100)
        assert r1["sp500"]["cvar"] == r2["sp500"]["cvar"]


# ---------------------------------------------------------------------------
# TestRunTailRiskAnalysis
# ---------------------------------------------------------------------------


class TestRunTailRiskAnalysis:
    def test_returns_dict(self, base_df):
        result = run_tail_risk_analysis(base_df)
        assert isinstance(result, dict)

    def test_all_keys_present(self, base_df):
        result = run_tail_risk_analysis(base_df)
        for key in ["regime_tail_stats", "strategy_tail_stats", "rolling_cvar_sp500",
                    "rolling_cvar_strat", "drawdown_by_regime", "vs_benchmark", "alpha"]:
            assert key in result

    def test_alpha_stored_correctly(self, base_df):
        result = run_tail_risk_analysis(base_df, alpha=0.01)
        assert result["alpha"] == 0.01

    def test_regime_tail_stats_is_dataframe(self, base_df):
        result = run_tail_risk_analysis(base_df)
        assert isinstance(result["regime_tail_stats"], pd.DataFrame)

    def test_rolling_cvar_sp500_is_series(self, base_df):
        result = run_tail_risk_analysis(base_df)
        assert isinstance(result["rolling_cvar_sp500"], pd.Series)

    def test_drawdown_by_regime_is_dataframe(self, base_df):
        result = run_tail_risk_analysis(base_df)
        assert isinstance(result["drawdown_by_regime"], pd.DataFrame)

    def test_vs_benchmark_is_dict(self, base_df):
        result = run_tail_risk_analysis(base_df)
        assert isinstance(result["vs_benchmark"], dict)

    def test_empty_df_graceful(self):
        result = run_tail_risk_analysis(pd.DataFrame())
        assert result["regime_tail_stats"].empty
        assert result["rolling_cvar_sp500"].empty
