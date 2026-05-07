"""
Tests for src/performance_scorecard.py.

Covers full-period stats, rolling metrics, regime performance,
monthly calendar, and the orchestrator.
"""

import numpy as np
import pandas as pd
import pytest

from src.performance_scorecard import (
    _WINDOWS,
    compute_full_period_stats,
    compute_monthly_calendar,
    compute_regime_performance,
    compute_rolling_metrics,
    run_performance_scorecard,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(n=300, seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    sp500 = rng.normal(0.0004, 0.010, n)
    strat = rng.normal(0.0005, 0.007, n)   # slightly better Sharpe
    regimes = (["Neutral"] * 100 + ["Avoid Chasing Risk"] * 100
               + ["Buy Stress"] * 50 + ["Hold / Do Not Chase"] * 50)
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
def short_df():
    """10-row DataFrame — below _MIN_OBS for most metrics."""
    rng = np.random.default_rng(99)
    dates = pd.date_range("2024-06-01", periods=10, freq="B")
    return pd.DataFrame({
        "date":                   dates,
        "sp500_daily_return":     rng.normal(0, 0.01, 10),
        "strategy_daily_return":  rng.normal(0, 0.008, 10),
        "final_decision":         ["Neutral"] * 10,
    })


@pytest.fixture(scope="module")
def monotone_df():
    """All positive returns — hit_rate=1, no drawdown."""
    dates = pd.date_range("2024-01-02", periods=100, freq="B")
    return pd.DataFrame({
        "date":                   dates,
        "sp500_daily_return":     np.full(100, 0.002),
        "strategy_daily_return":  np.full(100, 0.003),
        "final_decision":         ["Neutral"] * 100,
    })


@pytest.fixture(scope="module")
def declining_df():
    """All negative returns — max drawdown = total loss."""
    dates = pd.date_range("2024-01-02", periods=100, freq="B")
    return pd.DataFrame({
        "date":                   dates,
        "sp500_daily_return":     np.full(100, -0.005),
        "strategy_daily_return":  np.full(100, -0.003),
        "final_decision":         ["Avoid Chasing Risk"] * 100,
    })


# ---------------------------------------------------------------------------
# TestComputeFullPeriodStats
# ---------------------------------------------------------------------------


class TestComputeFullPeriodStats:
    def test_returns_dict(self, base_df):
        result = compute_full_period_stats(base_df)
        assert isinstance(result, dict)

    def test_both_labels_present(self, base_df):
        result = compute_full_period_stats(base_df)
        assert "strategy" in result
        assert "sp500" in result

    def test_all_fields_present(self, base_df):
        result = compute_full_period_stats(base_df)
        for label in ["strategy", "sp500"]:
            for field in ["n_obs", "total_return", "ann_return", "ann_vol",
                          "sharpe", "max_drawdown", "calmar", "hit_rate",
                          "best_day", "worst_day"]:
                assert field in result[label], f"Missing {field} in {label}"

    def test_n_obs_correct(self, base_df):
        result = compute_full_period_stats(base_df)
        assert result["strategy"]["n_obs"] == len(base_df)

    def test_ann_vol_positive(self, base_df):
        result = compute_full_period_stats(base_df)
        for label in ["strategy", "sp500"]:
            assert result[label]["ann_vol"] > 0

    def test_hit_rate_in_range(self, base_df):
        result = compute_full_period_stats(base_df)
        for label in ["strategy", "sp500"]:
            hr = result[label]["hit_rate"]
            assert 0.0 <= hr <= 1.0

    def test_hit_rate_1_for_all_positive(self, monotone_df):
        result = compute_full_period_stats(monotone_df)
        assert result["strategy"]["hit_rate"] == pytest.approx(1.0)

    def test_max_drawdown_nonpositive(self, base_df):
        result = compute_full_period_stats(base_df)
        for label in ["strategy", "sp500"]:
            mdd = result[label]["max_drawdown"]
            if not np.isnan(mdd):
                assert mdd <= 0

    def test_max_drawdown_zero_for_monotone_gains(self, monotone_df):
        result = compute_full_period_stats(monotone_df)
        assert result["strategy"]["max_drawdown"] == pytest.approx(0.0, abs=1e-6)

    def test_calmar_positive_when_return_positive(self, monotone_df):
        result = compute_full_period_stats(monotone_df)
        calmar = result["strategy"]["calmar"]
        # monotone gains → no drawdown → calmar is nan (division by near-zero)
        assert np.isnan(calmar) or calmar > 0

    def test_best_day_ge_worst_day(self, base_df):
        result = compute_full_period_stats(base_df)
        for label in ["strategy", "sp500"]:
            assert result[label]["best_day"] >= result[label]["worst_day"]

    def test_worst_day_negative_in_declining(self, declining_df):
        result = compute_full_period_stats(declining_df)
        assert result["strategy"]["worst_day"] < 0

    def test_strategy_missing_returns_no_strategy_key(self):
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=50, freq="B"),
                           "sp500_daily_return": np.random.default_rng(1).normal(0, 0.01, 50)})
        result = compute_full_period_stats(df)
        assert "sp500" in result
        assert "strategy" not in result

    def test_empty_df_returns_empty_dict(self):
        result = compute_full_period_stats(pd.DataFrame())
        assert result == {}

    def test_better_sharpe_for_lower_vol_series(self, base_df):
        # Strategy fixture has lower vol → should have better Sharpe
        result = compute_full_period_stats(base_df)
        s_sh = result["strategy"]["sharpe"]
        b_sh = result["sp500"]["sharpe"]
        if not (np.isnan(s_sh) or np.isnan(b_sh)):
            assert s_sh >= b_sh - 0.5   # directional check with slack


# ---------------------------------------------------------------------------
# TestComputeRollingMetrics
# ---------------------------------------------------------------------------


class TestComputeRollingMetrics:
    def test_returns_dict_keyed_by_window(self, base_df):
        result = compute_rolling_metrics(base_df)
        for w in _WINDOWS:
            assert w in result

    def test_each_entry_is_dataframe(self, base_df):
        result = compute_rolling_metrics(base_df)
        for w in _WINDOWS:
            assert isinstance(result[w], pd.DataFrame)

    def test_correct_columns_present(self, base_df):
        result = compute_rolling_metrics(base_df)
        for w in _WINDOWS:
            for col in ["strategy_sharpe", "sp500_sharpe", "strategy_max_dd", "sp500_max_dd"]:
                assert col in result[w].columns

    def test_length_matches_input(self, base_df):
        result = compute_rolling_metrics(base_df)
        for w in _WINDOWS:
            assert len(result[w]) == len(base_df)

    def test_first_rows_are_nan(self, base_df):
        w = _WINDOWS[0]
        result = compute_rolling_metrics(base_df, windows=[w])
        assert result[w]["strategy_sharpe"].iloc[:w - 1].isna().all()

    def test_later_rows_not_all_nan(self, base_df):
        w = _WINDOWS[0]
        result = compute_rolling_metrics(base_df, windows=[w])
        assert result[w]["strategy_sharpe"].iloc[w:].notna().any()

    def test_max_dd_nonpositive(self, base_df):
        result = compute_rolling_metrics(base_df)
        for w in _WINDOWS:
            dd = result[w]["strategy_max_dd"].dropna()
            assert (dd <= 0).all()

    def test_max_dd_zero_for_monotone(self, monotone_df):
        w = _WINDOWS[0]
        result = compute_rolling_metrics(monotone_df, windows=[w])
        dd = result[w]["strategy_max_dd"].dropna()
        assert (dd.abs() < 1e-6).all()

    def test_index_is_datetime_when_date_col_present(self, base_df):
        result = compute_rolling_metrics(base_df)
        assert pd.api.types.is_datetime64_any_dtype(result[_WINDOWS[0]].index)

    def test_empty_df_returns_empty_frames(self):
        result = compute_rolling_metrics(pd.DataFrame())
        for w in _WINDOWS:
            assert result[w].empty

    def test_custom_window_respected(self, base_df):
        result = compute_rolling_metrics(base_df, windows=[30])
        assert 30 in result
        assert result[30]["strategy_sharpe"].iloc[:29].isna().all()


# ---------------------------------------------------------------------------
# TestComputeRegimePerformance
# ---------------------------------------------------------------------------


class TestComputeRegimePerformance:
    def test_returns_dataframe(self, base_df):
        result = compute_regime_performance(base_df)
        assert isinstance(result, pd.DataFrame)

    def test_all_regimes_present(self, base_df):
        result = compute_regime_performance(base_df)
        for regime in base_df["final_decision"].unique():
            assert regime in result.index

    def test_expected_columns(self, base_df):
        result = compute_regime_performance(base_df)
        for col in ["n_obs", "strat_mean", "strat_hit_rate", "strat_sharpe",
                    "sp500_mean", "sp500_sharpe"]:
            assert col in result.columns

    def test_n_obs_sums_to_total_rows(self, base_df):
        result = compute_regime_performance(base_df)
        assert result["n_obs"].sum() == len(base_df)

    def test_hit_rate_in_valid_range(self, base_df):
        result = compute_regime_performance(base_df)
        hr = result["strat_hit_rate"].dropna()
        assert (hr >= 0).all() and (hr <= 1).all()

    def test_small_regime_has_nan_sharpe(self, short_df):
        result = compute_regime_performance(short_df)
        if not result.empty:
            assert result["strat_sharpe"].isna().all()

    def test_sorted_by_strat_sharpe_descending(self, base_df):
        result = compute_regime_performance(base_df)
        sharpes = result["strat_sharpe"].dropna().tolist()
        assert sharpes == sorted(sharpes, reverse=True)

    def test_no_decision_col_returns_empty(self, base_df):
        df2 = base_df.drop(columns=["final_decision"])
        result = compute_regime_performance(df2)
        assert result.empty

    def test_empty_df_returns_empty(self):
        result = compute_regime_performance(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# TestComputeMonthlyCalendar
# ---------------------------------------------------------------------------


class TestComputeMonthlyCalendar:
    def test_returns_dict(self, base_df):
        result = compute_monthly_calendar(base_df)
        assert isinstance(result, dict)

    def test_both_labels_present(self, base_df):
        result = compute_monthly_calendar(base_df)
        assert "strategy" in result
        assert "sp500" in result

    def test_each_entry_is_dataframe(self, base_df):
        result = compute_monthly_calendar(base_df)
        for label in ["strategy", "sp500"]:
            assert isinstance(result[label], pd.DataFrame)

    def test_columns_are_months(self, base_df):
        result = compute_monthly_calendar(base_df)
        cols = result["strategy"].columns.tolist()
        assert all(1 <= c <= 12 for c in cols)

    def test_rows_are_years(self, base_df):
        result = compute_monthly_calendar(base_df)
        years = result["strategy"].index.tolist()
        assert all(isinstance(y, (int, np.integer)) for y in years)

    def test_monthly_returns_reasonable_magnitude(self, base_df):
        result = compute_monthly_calendar(base_df)
        vals = result["strategy"].values.flatten()
        vals = vals[~np.isnan(vals)]
        assert (np.abs(vals) < 0.5).all()   # no monthly return > 50%

    def test_no_date_col_returns_empty(self, base_df):
        df2 = base_df.drop(columns=["date"])
        result = compute_monthly_calendar(df2)
        assert result == {}

    def test_empty_df_returns_empty(self):
        result = compute_monthly_calendar(pd.DataFrame())
        assert result == {}

    def test_single_month_has_one_cell(self):
        dates = pd.date_range("2025-03-03", periods=20, freq="B")
        df = pd.DataFrame({
            "date":                   dates,
            "strategy_daily_return":  np.full(20, 0.001),
            "sp500_daily_return":     np.full(20, 0.0005),
        })
        result = compute_monthly_calendar(df)
        assert result["strategy"].shape == (1, 1)


# ---------------------------------------------------------------------------
# TestRunPerformanceScorecard
# ---------------------------------------------------------------------------


class TestRunPerformanceScorecard:
    def test_returns_dict(self, base_df):
        result = run_performance_scorecard(base_df)
        assert isinstance(result, dict)

    def test_all_keys_present(self, base_df):
        result = run_performance_scorecard(base_df)
        for key in ["full_period", "rolling", "regime_perf", "monthly_cal", "windows"]:
            assert key in result

    def test_full_period_is_dict(self, base_df):
        result = run_performance_scorecard(base_df)
        assert isinstance(result["full_period"], dict)

    def test_rolling_has_expected_windows(self, base_df):
        result = run_performance_scorecard(base_df)
        for w in _WINDOWS:
            assert w in result["rolling"]

    def test_regime_perf_is_dataframe(self, base_df):
        result = run_performance_scorecard(base_df)
        assert isinstance(result["regime_perf"], pd.DataFrame)

    def test_monthly_cal_is_dict(self, base_df):
        result = run_performance_scorecard(base_df)
        assert isinstance(result["monthly_cal"], dict)

    def test_windows_list_matches_module_default(self, base_df):
        result = run_performance_scorecard(base_df)
        assert result["windows"] == _WINDOWS

    def test_empty_df_graceful(self):
        result = run_performance_scorecard(pd.DataFrame())
        assert result["full_period"] == {}
        assert result["regime_perf"].empty
        assert result["monthly_cal"] == {}
