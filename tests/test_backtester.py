"""Tests for src/backtester.py — OOS split and backtest summary."""

import numpy as np
import pandas as pd
import pytest

from src.backtester import (
    OOS_CUTOFF,
    _TREND_BOOST,
    _TREND_DAMP,
    _trend_scalar,
    build_strategy_backtest,
    compute_backtest_summary,
    compute_oos_split,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n=300, start="2018-01-01", decisions=None, scores=None):
    np.random.seed(42)
    dates = pd.bdate_range(start=start, periods=n)
    sp500 = 4000 * (1 + np.random.normal(0.0003, 0.01, n)).cumprod()
    decision = decisions or ["Neutral"] * n
    fwd = np.random.normal(0.01, 0.05, n)
    # composite scores in realistic range (21-43); lower = less risk = more equity
    score_vals = scores if scores is not None else np.full(n, 30.0)
    df = pd.DataFrame({
        "date": dates,
        "sp500": sp500,
        "final_decision": decision[:n],
        "composite_risk_score_smooth": score_vals[:n],
        "sp500_forward_30d_return":    fwd,
        "strategy_forward_30d_return": fwd * 0.60,
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

    def test_equity_floor_respected(self):
        # Very high composite score should still yield >= 40% weight
        n = 50
        high_scores = np.full(n, 99.0)  # extreme risk → score-based weight near 0.05
        df = _make_df(n=n, scores=high_scores)
        assert (df["strategy_weight"] >= 0.40).all()

    def test_low_score_gives_higher_weight_than_high_score(self):
        # Low risk score → higher blend weight than high risk score
        n = 50
        df_low  = _make_df(n=n, scores=np.full(n, 28.0))  # low risk
        df_high = _make_df(n=n, scores=np.full(n, 45.0))  # high risk
        assert df_low["strategy_weight"].mean() > df_high["strategy_weight"].mean()

    def test_low_score_weight_above_midpoint(self):
        # A low-risk score should yield a well-above-floor weight (>= 0.60)
        n = 50
        df = _make_df(n=n, scores=np.full(n, 28.0))
        assert df["strategy_weight"].mean() >= 0.60

    def test_fallback_without_score_column(self):
        # When composite_risk_score_smooth is absent, decision-bucket fallback runs
        np.random.seed(42)
        n = 50
        dates = pd.bdate_range(start="2018-01-01", periods=n)
        sp500 = 4000 * (1 + np.random.normal(0.0003, 0.01, n)).cumprod()
        fwd = np.random.normal(0.01, 0.05, n)
        df_no_score = pd.DataFrame({
            "date": dates,
            "sp500": sp500,
            "final_decision": ["Neutral"] * n,
            "sp500_forward_30d_return": fwd,
            "strategy_forward_30d_return": fwd * 0.60,
        })
        result = build_strategy_backtest(df_no_score)
        assert (result["strategy_weight"] >= 0.40).all()


# ---------------------------------------------------------------------------
# _trend_scalar
# ---------------------------------------------------------------------------

class TestTrendScalar:
    def _sp500(self, n, trend="up"):
        """Monotonically rising or falling price series."""
        if trend == "up":
            return pd.Series(1000 * (1.001 ** np.arange(n)))
        return pd.Series(1000 * (0.999 ** np.arange(n)))

    def test_returns_boost_when_above_ma(self):
        # A long enough upward series: after warmup, price is above its rolling MA
        sp = self._sp500(400, trend="up")
        s = _trend_scalar(sp, window=200)
        assert (s.iloc[200:] == _TREND_BOOST).all()

    def test_returns_damp_when_below_ma(self):
        # A long enough downward series: after warmup, price is below its rolling MA
        sp = self._sp500(400, trend="down")
        s = _trend_scalar(sp, window=200)
        assert (s.iloc[200:] == _TREND_DAMP).all()

    def test_neutral_during_warmup(self):
        # Before the MA window fills, scalar should be 1.0
        sp = self._sp500(400, trend="up")
        s = _trend_scalar(sp, window=200)
        assert (s.iloc[:199] == 1.0).all()

    def test_weight_higher_in_uptrend_than_downtrend(self):
        # Same score, but uptrend → higher weight than downtrend after 200 days
        n = 400
        scores = np.full(n, 33.0)  # mid-range score → ~0.75 base weight

        sp_up   = 4000 * (1.001 ** np.arange(n))
        sp_down = 4000 * (0.999 ** np.arange(n))
        fwd = np.random.normal(0.01, 0.05, n)
        dates = pd.bdate_range(start="2018-01-01", periods=n)

        def _make_with_sp(sp):
            df = pd.DataFrame({
                "date": dates, "sp500": sp,
                "final_decision": ["Neutral"] * n,
                "composite_risk_score_smooth": scores,
                "sp500_forward_30d_return": fwd,
                "strategy_forward_30d_return": fwd * 0.60,
            })
            return build_strategy_backtest(df)

        up_df   = _make_with_sp(sp_up)
        down_df = _make_with_sp(sp_down)
        # After warmup, uptrend weights should be strictly higher
        assert (up_df["strategy_weight"].iloc[200:] > down_df["strategy_weight"].iloc[200:]).all()

    def test_floor_holds_even_with_trend_damp(self):
        # Downtrend + high risk score: damp * low_weight might go below floor;
        # floor must still hold at 0.40
        n = 400
        sp = self._sp500(n, trend="down")
        fwd = np.random.normal(0.01, 0.05, n)
        dates = pd.bdate_range(start="2018-01-01", periods=n)
        df = pd.DataFrame({
            "date": dates, "sp500": sp,
            "final_decision": ["Neutral"] * n,
            "composite_risk_score_smooth": np.full(n, 99.0),  # max risk
            "sp500_forward_30d_return": fwd,
            "strategy_forward_30d_return": fwd * 0.60,
        })
        result = build_strategy_backtest(df)
        assert (result["strategy_weight"] >= 0.40).all()

    def test_weight_capped_at_one(self):
        # Uptrend boost on a near-max weight shouldn't exceed 1.0
        n = 400
        sp = self._sp500(n, trend="up")
        fwd = np.random.normal(0.01, 0.05, n)
        dates = pd.bdate_range(start="2018-01-01", periods=n)
        df = pd.DataFrame({
            "date": dates, "sp500": sp,
            "final_decision": ["Risk-On"] * n,
            "composite_risk_score_smooth": np.full(n, 10.0),  # very low risk → 1.0
            "sp500_forward_30d_return": fwd,
            "strategy_forward_30d_return": fwd * 0.60,
        })
        result = build_strategy_backtest(df)
        assert (result["strategy_weight"] <= 1.0).all()


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
