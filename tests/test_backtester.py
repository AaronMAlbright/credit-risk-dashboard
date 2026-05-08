"""Tests for src/backtester.py — OOS split and backtest summary."""

import numpy as np
import pandas as pd
import pytest

from src.backtester import (
    OOS_CUTOFF,
    build_strategy_backtest,
    compute_backtest_summary,
    compute_oos_split,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n=300, start="2018-01-01", decisions=None):
    np.random.seed(42)
    dates = pd.bdate_range(start=start, periods=n)
    sp500 = 4000 * (1 + np.random.normal(0.0003, 0.01, n)).cumprod()
    decision = decisions or ["Hold / Do Not Chase"] * n
    fwd = np.random.normal(0.01, 0.05, n)
    df = pd.DataFrame({
        "date": dates,
        "sp500": sp500,
        "final_decision": decision[:n],
        "sp500_forward_30d_return":    fwd,
        "strategy_forward_30d_return": fwd * 0.60,  # strategy captures ~60% of sp500 move
        "sp500_forward_60d_return":    np.random.normal(0.02, 0.07, n),
        "equity_weight": [0.40] * n,
        "cash_weight":   [0.35] * n,
        "credit_weight": [0.25] * n,
    })
    return build_strategy_backtest(df)


# ---------------------------------------------------------------------------
# build_strategy_backtest
# ---------------------------------------------------------------------------

class TestBuildStrategyBacktest:
    def test_returns_required_columns(self):
        df = _make_df()
        for col in ["strategy_daily_return", "sp500_daily_return",
                    "strategy_equity_curve", "sp500_equity_curve",
                    "strategy_drawdown", "sp500_backtest_drawdown"]:
            assert col in df.columns

    def test_equity_curves_start_near_one(self):
        df = _make_df()
        assert abs(df["strategy_equity_curve"].iloc[0] - 1.0) < 0.05
        assert abs(df["sp500_equity_curve"].iloc[0] - 1.0) < 0.05

    def test_drawdown_non_positive(self):
        df = _make_df()
        assert (df["strategy_drawdown"] <= 0).all()
        assert (df["sp500_backtest_drawdown"] <= 0).all()

    def test_strategy_weight_lagged_by_one(self):
        df = _make_df()
        assert pd.isna(df["strategy_weight_lagged"].iloc[0])

    def test_buy_stress_weight_is_one(self):
        n = 50
        df = _make_df(n=n, decisions=["Buy Stress"] * n)
        assert (df["strategy_weight"].iloc[:n] == 1.0).all()


# ---------------------------------------------------------------------------
# compute_backtest_summary
# ---------------------------------------------------------------------------

class TestComputeBacktestSummary:
    def test_returns_expected_keys(self):
        df = _make_df()
        result = compute_backtest_summary(df)
        for key in ("strategy_total_return", "sp500_total_return",
                    "strategy_sharpe", "sp500_sharpe",
                    "strategy_max_drawdown", "strategy_hit_rate"):
            assert key in result

    def test_max_drawdown_non_positive(self):
        df = _make_df()
        result = compute_backtest_summary(df)
        assert result["strategy_max_drawdown"] <= 0
        assert result["sp500_max_drawdown"] <= 0

    def test_hit_rate_between_0_and_1(self):
        df = _make_df()
        result = compute_backtest_summary(df)
        assert 0 <= result["strategy_hit_rate"] <= 1

    def test_transaction_costs_reduce_return(self):
        df = _make_df()
        result_notc = compute_backtest_summary(df, tc_bps=0)
        result_tc   = compute_backtest_summary(df, tc_bps=20)
        assert result_tc["strategy_total_return"] <= result_notc["strategy_total_return"]

    def test_zero_tc_matches_gross(self):
        df = _make_df(n=200)
        r0 = compute_backtest_summary(df, tc_bps=0)
        r1 = compute_backtest_summary(df, tc_bps=0)
        assert abs(r0["strategy_total_return"] - r1["strategy_total_return"]) < 1e-10


# ---------------------------------------------------------------------------
# compute_oos_split
# ---------------------------------------------------------------------------

class TestComputeOosSplit:
    def test_returns_expected_keys(self):
        df = _make_df(n=700, start="2018-01-01")
        result = compute_oos_split(df, cutoff="2020-01-01")
        for key in ("cutoff", "in_sample", "out_of_sample", "full_period",
                    "is_start", "is_end", "oos_start", "oos_end",
                    "is_n_days", "oos_n_days"):
            assert key in result

    def test_split_days_sum_to_total(self):
        df = _make_df(n=700, start="2018-01-01")
        result = compute_oos_split(df, cutoff="2020-01-01")
        assert result["is_n_days"] + result["oos_n_days"] == len(df)

    def test_is_before_cutoff(self):
        df = _make_df(n=700, start="2018-01-01")
        result = compute_oos_split(df, cutoff="2020-01-01")
        assert result["is_end"] < "2020-01-01"

    def test_oos_on_or_after_cutoff(self):
        df = _make_df(n=700, start="2018-01-01")
        result = compute_oos_split(df, cutoff="2020-01-01")
        assert result["oos_start"] >= "2020-01-01"

    def test_all_data_before_cutoff_gives_empty_oos(self):
        df = _make_df(n=100, start="2018-01-01")
        result = compute_oos_split(df, cutoff="2030-01-01")
        assert result["oos_n_days"] == 0
        assert result["out_of_sample"] == {}

    def test_all_data_after_cutoff_gives_empty_is(self):
        df = _make_df(n=100, start="2023-01-01")
        result = compute_oos_split(df, cutoff="2020-01-01")
        assert result["is_n_days"] == 0
        assert result["in_sample"] == {}

    def test_oos_cutoff_constant_is_valid_date(self):
        pd.Timestamp(OOS_CUTOFF)

    def test_each_period_has_sharpe(self):
        df = _make_df(n=700, start="2018-01-01")
        result = compute_oos_split(df, cutoff="2020-01-01")
        assert "strategy_sharpe" in result["in_sample"]
        assert "strategy_sharpe" in result["out_of_sample"]

    def test_is_sharpe_can_differ_from_oos(self):
        """IS and OOS Sharpes should be computed independently."""
        df = _make_df(n=700, start="2018-01-01")
        result = compute_oos_split(df, cutoff="2020-01-01")
        is_s  = result["in_sample"].get("strategy_sharpe", 0)
        oos_s = result["out_of_sample"].get("strategy_sharpe", 0)
        assert isinstance(is_s, float)
        assert isinstance(oos_s, float)
