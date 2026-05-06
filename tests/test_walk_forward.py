"""
Tests for src/walk_forward.py.
Uses the same session-scoped fixtures as test_pipeline.py — the analytics
DataFrame is computed once and reused across all walk-forward tests.
"""
import pytest
import numpy as np
import pandas as pd

from src.walk_forward import (
    generate_windows,
    evaluate_window,
    compute_regime_breakdown,
    run_walk_forward,
    assert_no_leakage,
    _window_metrics,
)


# ---------------------------------------------------------------------------
# Shared fixture: analytics df (reuses session fixture from conftest.py)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def wf_df(analytics_result):
    df, _, _ = analytics_result
    return df


# ---------------------------------------------------------------------------
# assert_no_leakage
# ---------------------------------------------------------------------------
class TestAssertNoLeakage:
    def test_clean_df_passes(self, wf_df):
        assert_no_leakage(wf_df)  # should not raise

    def test_contaminated_df_raises(self, wf_df):
        bad = wf_df.copy()
        bad["macro_risk_score_smooth"] = bad["sp500_forward_30d_return"]
        with pytest.raises(AssertionError, match="Leakage"):
            assert_no_leakage(bad)


# ---------------------------------------------------------------------------
# generate_windows
# ---------------------------------------------------------------------------
class TestGenerateWindows:
    def test_returns_list(self, wf_df):
        windows = generate_windows(wf_df, train_days=126, test_days=42, step_days=42)
        assert isinstance(windows, list)

    def test_at_least_one_window(self, wf_df):
        windows = generate_windows(wf_df, train_days=126, test_days=42, step_days=42)
        assert len(windows) >= 1

    def test_test_start_after_train_end(self, wf_df):
        for w in generate_windows(wf_df, train_days=126, test_days=42, step_days=42):
            assert w["test_start"] > w["train_end"], (
                f"Window {w['window_id']}: test_start {w['test_start']} "
                f"not after train_end {w['train_end']}"
            )

    def test_window_ids_are_sequential(self, wf_df):
        windows = generate_windows(wf_df, train_days=126, test_days=42, step_days=42)
        ids = [w["window_id"] for w in windows]
        assert ids == list(range(len(windows)))

    def test_test_windows_non_overlapping_when_step_equals_test(self, wf_df):
        windows = generate_windows(wf_df, train_days=126, test_days=42, step_days=42)
        for i in range(1, len(windows)):
            assert windows[i]["test_start"] > windows[i - 1]["test_end"], (
                f"Windows {i-1} and {i} overlap"
            )

    def test_no_windows_when_dataset_too_small(self, wf_df):
        # Requesting more train days than the dataset has
        windows = generate_windows(wf_df, train_days=len(wf_df) + 100, test_days=42, step_days=42)
        assert windows == []

    def test_each_window_has_required_keys(self, wf_df):
        required = {"window_id", "train_start", "train_end", "test_start", "test_end", "_test_slice"}
        for w in generate_windows(wf_df, train_days=126, test_days=42, step_days=42):
            assert required.issubset(w.keys())


# ---------------------------------------------------------------------------
# _window_metrics (private helper, tested directly for coverage)
# ---------------------------------------------------------------------------
class TestWindowMetrics:
    def _pos_returns(self):
        return pd.Series([0.01, 0.005, 0.02, 0.008, 0.015] * 10)

    def _neg_returns(self):
        return pd.Series([-0.01, -0.005, -0.02, -0.008, -0.015] * 10)

    def test_returns_expected_keys(self):
        result = _window_metrics(self._pos_returns(), "strategy")
        expected = {
            "strategy_total_return", "strategy_sharpe", "strategy_hit_rate",
            "strategy_max_drawdown", "strategy_volatility",
        }
        assert expected == result.keys()

    def test_positive_returns_positive_sharpe(self):
        result = _window_metrics(self._pos_returns(), "strategy")
        assert result["strategy_sharpe"] > 0

    def test_negative_returns_negative_sharpe(self):
        result = _window_metrics(self._neg_returns(), "strategy")
        assert result["strategy_sharpe"] < 0

    def test_hit_rate_between_zero_and_one(self):
        result = _window_metrics(self._pos_returns(), "strategy")
        assert 0.0 <= result["strategy_hit_rate"] <= 1.0

    def test_max_drawdown_non_positive(self):
        result = _window_metrics(self._pos_returns(), "strategy")
        assert result["strategy_max_drawdown"] <= 0

    def test_volatility_positive(self):
        result = _window_metrics(self._pos_returns(), "strategy")
        assert result["strategy_volatility"] > 0

    def test_all_positive_returns_hit_rate_is_one(self):
        result = _window_metrics(pd.Series([0.01] * 20), "strategy")
        assert result["strategy_hit_rate"] == 1.0

    def test_empty_series_returns_nans(self):
        result = _window_metrics(pd.Series([], dtype=float), "strategy")
        assert all(np.isnan(v) for v in result.values())

    def test_single_row_returns_nans(self):
        result = _window_metrics(pd.Series([0.01]), "strategy")
        assert all(np.isnan(v) for v in result.values())

    def test_prefix_respected(self):
        result = _window_metrics(self._pos_returns(), "sp500")
        assert all(k.startswith("sp500_") for k in result.keys())


# ---------------------------------------------------------------------------
# evaluate_window
# ---------------------------------------------------------------------------
class TestEvaluateWindow:
    def test_returns_dict(self, wf_df):
        windows = generate_windows(wf_df, train_days=126, test_days=42, step_days=42)
        w = windows[0]
        df_test = wf_df.iloc[w["_test_slice"]]
        result = evaluate_window(df_test, w["window_id"], w["train_end"])
        assert isinstance(result, dict)

    def test_expected_keys_present(self, wf_df):
        windows = generate_windows(wf_df, train_days=126, test_days=42, step_days=42)
        w = windows[0]
        df_test = wf_df.iloc[w["_test_slice"]]
        result = evaluate_window(df_test, w["window_id"], w["train_end"])

        expected_keys = {
            "window_id", "train_end", "test_start", "test_end", "n_obs",
            "dominant_regime",
            "strategy_sharpe", "strategy_hit_rate", "strategy_max_drawdown",
            "strategy_volatility", "strategy_total_return",
            "sp500_sharpe", "sp500_hit_rate", "sp500_max_drawdown",
            "sp500_volatility", "sp500_total_return",
        }
        assert expected_keys.issubset(result.keys())

    def test_n_obs_matches_test_window_size(self, wf_df):
        windows = generate_windows(wf_df, train_days=126, test_days=42, step_days=42)
        w = windows[0]
        df_test = wf_df.iloc[w["_test_slice"]]
        result = evaluate_window(df_test, w["window_id"], w["train_end"])
        assert result["n_obs"] == len(df_test)

    def test_window_id_preserved(self, wf_df):
        windows = generate_windows(wf_df, train_days=126, test_days=42, step_days=42)
        w = windows[2]
        df_test = wf_df.iloc[w["_test_slice"]]
        result = evaluate_window(df_test, w["window_id"], w["train_end"])
        assert result["window_id"] == 2


# ---------------------------------------------------------------------------
# compute_regime_breakdown
# ---------------------------------------------------------------------------
class TestComputeRegimeBreakdown:
    def test_returns_list(self, wf_df):
        windows = generate_windows(wf_df, train_days=126, test_days=42, step_days=42)
        w = windows[0]
        df_test = wf_df.iloc[w["_test_slice"]]
        result = compute_regime_breakdown(df_test, w["window_id"], w["train_end"])
        assert isinstance(result, list)

    def test_each_row_is_dict(self, wf_df):
        windows = generate_windows(wf_df, train_days=126, test_days=42, step_days=42)
        w = windows[0]
        df_test = wf_df.iloc[w["_test_slice"]]
        rows = compute_regime_breakdown(df_test, w["window_id"], w["train_end"])
        assert all(isinstance(r, dict) for r in rows)

    def test_expected_keys_in_each_row(self, wf_df):
        windows = generate_windows(wf_df, train_days=126, test_days=42, step_days=42)
        w = windows[0]
        df_test = wf_df.iloc[w["_test_slice"]]
        rows = compute_regime_breakdown(df_test, w["window_id"], w["train_end"])
        expected = {
            "window_id", "train_end", "test_start", "test_end",
            "regime", "n_obs", "n_obs_with_fwd_return",
            "avg_forward_30d_return", "hit_rate", "worst_5pct", "avg_equity_weight",
        }
        for r in rows:
            assert expected.issubset(r.keys())

    def test_obs_count_sums_to_window_size(self, wf_df):
        windows = generate_windows(wf_df, train_days=126, test_days=42, step_days=42)
        w = windows[0]
        df_test = wf_df.iloc[w["_test_slice"]]
        rows = compute_regime_breakdown(df_test, w["window_id"], w["train_end"])
        assert sum(r["n_obs"] for r in rows) == len(df_test)

    def test_hit_rate_between_zero_and_one(self, wf_df):
        windows = generate_windows(wf_df, train_days=126, test_days=42, step_days=42)
        w = windows[0]
        df_test = wf_df.iloc[w["_test_slice"]]
        rows = compute_regime_breakdown(df_test, w["window_id"], w["train_end"])
        for r in rows:
            if r["hit_rate"] is not None and not np.isnan(r["hit_rate"]):
                assert 0.0 <= r["hit_rate"] <= 1.0

    def test_avg_equity_weight_between_zero_and_one(self, wf_df):
        windows = generate_windows(wf_df, train_days=126, test_days=42, step_days=42)
        w = windows[0]
        df_test = wf_df.iloc[w["_test_slice"]]
        rows = compute_regime_breakdown(df_test, w["window_id"], w["train_end"])
        for r in rows:
            assert 0.0 <= r["avg_equity_weight"] <= 1.0


# ---------------------------------------------------------------------------
# run_walk_forward (integration)
# ---------------------------------------------------------------------------
class TestRunWalkForward:
    def test_returns_two_dataframes(self, wf_df):
        result = run_walk_forward(wf_df, train_days=126, test_days=42, step_days=42)
        assert len(result) == 2
        assert all(isinstance(r, pd.DataFrame) for r in result)

    def test_windows_df_has_expected_columns(self, wf_df):
        windows_df, _ = run_walk_forward(wf_df, train_days=126, test_days=42, step_days=42)
        expected = {
            "window_id", "test_start", "test_end", "n_obs", "dominant_regime",
            "strategy_sharpe", "strategy_hit_rate", "strategy_max_drawdown",
            "strategy_volatility", "strategy_total_return",
            "sp500_sharpe", "sp500_hit_rate", "sp500_max_drawdown",
        }
        assert expected.issubset(windows_df.columns)

    def test_regimes_df_has_expected_columns(self, wf_df):
        _, regimes_df = run_walk_forward(wf_df, train_days=126, test_days=42, step_days=42)
        expected = {
            "window_id", "test_start", "test_end", "regime",
            "n_obs", "avg_forward_30d_return", "hit_rate", "avg_equity_weight",
        }
        assert expected.issubset(regimes_df.columns)

    def test_window_count_is_positive(self, wf_df):
        windows_df, _ = run_walk_forward(wf_df, train_days=126, test_days=42, step_days=42)
        assert len(windows_df) > 0

    def test_no_test_window_overlap(self, wf_df):
        windows_df, _ = run_walk_forward(wf_df, train_days=126, test_days=42, step_days=42)
        starts = pd.to_datetime(windows_df["test_start"])
        ends   = pd.to_datetime(windows_df["test_end"])
        for i in range(1, len(windows_df)):
            assert starts.iloc[i] > ends.iloc[i - 1], (
                f"Windows {i-1} and {i} have overlapping test periods"
            )

    def test_strategy_sharpe_is_finite_or_nan(self, wf_df):
        windows_df, _ = run_walk_forward(wf_df, train_days=126, test_days=42, step_days=42)
        valid = windows_df["strategy_sharpe"].dropna()
        assert np.isfinite(valid).all()

    def test_regime_obs_sum_equals_total_test_obs(self, wf_df):
        windows_df, regimes_df = run_walk_forward(wf_df, train_days=126, test_days=42, step_days=42)
        for wid in windows_df["window_id"]:
            total_from_windows = windows_df.loc[windows_df["window_id"] == wid, "n_obs"].iloc[0]
            total_from_regimes = regimes_df.loc[regimes_df["window_id"] == wid, "n_obs"].sum()
            assert total_from_windows == total_from_regimes, (
                f"Window {wid}: n_obs mismatch ({total_from_windows} vs {total_from_regimes})"
            )

    def test_raises_on_missing_required_column(self, wf_df):
        bad_df = wf_df.drop(columns=["strategy_daily_return"])
        with pytest.raises(ValueError, match="missing required columns"):
            run_walk_forward(bad_df)

    def test_raises_when_too_few_rows(self, wf_df):
        tiny = wf_df.head(50)
        with pytest.raises(ValueError, match="No windows generated"):
            run_walk_forward(tiny, train_days=126, test_days=42, step_days=42)

    def test_smaller_step_produces_more_windows(self, wf_df):
        w_large_step, _ = run_walk_forward(wf_df, train_days=126, test_days=42, step_days=42)
        w_small_step, _ = run_walk_forward(wf_df, train_days=126, test_days=42, step_days=21)
        assert len(w_small_step) >= len(w_large_step)
