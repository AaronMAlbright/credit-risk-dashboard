"""
Integration tests for run_scoring_pipeline() and run_analytics().
All tests use session-scoped fixtures from conftest.py so the pipeline
runs once per pytest invocation, not once per test.
"""
import pytest
import pandas as pd

# ---------------------------------------------------------------------------
# Column manifests
# ---------------------------------------------------------------------------
_SCORE_COLUMNS = [
    "macro_risk_score_smooth",
    "credit_market_risk_score_smooth",
    "liquidity_score_smooth",
    "liquidity_regime_score_smooth",
    "treasury_stress_score_smooth",
    "cross_asset_divergence_score_smooth",
    "market_internals_score_smooth",
    "risk_appetite_score_smooth",
    "complacency_score_smooth",
    "mean_reversion_score_smooth",
    "composite_risk_score_smooth",
]

_LABEL_COLUMNS = [
    "macro_risk_label",
    "credit_market_risk_label",
    "liquidity_label",
    "liquidity_regime_label",
    "liquidity_signal",
    "treasury_stress_label",
    "cross_asset_divergence_label",
    "market_internals_label",
    "risk_appetite_label",
    "complacency_label",
    "mean_reversion_label",
    "composite_risk_label",
    "transition_regime",
    "transition_signal",
    "final_decision",
    "final_environment",
    "final_action",
]

_BACKTEST_COLUMNS = [
    "strategy_equity_curve",
    "sp500_equity_curve",
    "strategy_daily_return",
    "sp500_daily_return",
    "strategy_drawdown",
    "sp500_backtest_drawdown",
    "strategy_weight",
    "strategy_weight_lagged",
]

_BACKTEST_SUMMARY_KEYS = {
    "avg_strategy_30d_return",
    "avg_sp500_30d_return",
    "strategy_hit_rate",
    "sp500_hit_rate",
    "strategy_worst_5pct",
    "sp500_worst_5pct",
    "strategy_total_return",
    "sp500_total_return",
    "strategy_volatility",
    "sp500_volatility",
    "strategy_max_drawdown",
    "sp500_max_drawdown",
}

_HEALTH_CHECK_KEYS = {
    "portfolio_weight_check",
    "missing_values",
    "final_decision_sample_check",
    "score_bounds_check",
}

_VALID_CONFIDENCE_VALUES = {"High", "Medium", "Low"}


# ---------------------------------------------------------------------------
# Goal 1 + 2: Pipeline stages run, required columns are present
# ---------------------------------------------------------------------------
class TestScoringPipelineColumns:

    def test_score_columns_exist(self, pipeline_df):
        missing = [c for c in _SCORE_COLUMNS if c not in pipeline_df.columns]
        assert not missing, f"Missing score columns: {missing}"

    def test_label_columns_exist(self, pipeline_df):
        missing = [c for c in _LABEL_COLUMNS if c not in pipeline_df.columns]
        assert not missing, f"Missing label columns: {missing}"

    def test_composite_risk_label_present(self, pipeline_df):
        assert "composite_risk_label" in pipeline_df.columns

    def test_final_decision_obj_present(self, pipeline_df):
        assert "final_decision_obj" in pipeline_df.columns


# ---------------------------------------------------------------------------
# Goal 3: Score columns stay within [0, 100]
# ---------------------------------------------------------------------------
class TestScoreBounds:

    @pytest.mark.parametrize("col", _SCORE_COLUMNS)
    def test_score_within_bounds(self, pipeline_df, col):
        series = pipeline_df[col].dropna()
        assert len(series) > 0, f"{col} is entirely NaN"
        assert (series >= 0).all(), f"{col} has values below 0: min={series.min()}"
        assert (series <= 100).all(), f"{col} has values above 100: max={series.max()}"

    def test_composite_risk_score_smooth_non_trivial(self, pipeline_df):
        series = pipeline_df["composite_risk_score_smooth"].dropna()
        assert series.nunique() > 1, "composite_risk_score_smooth is constant"


# ---------------------------------------------------------------------------
# Goal 4: Portfolio weights — tested at integration level against analytics_result
# ---------------------------------------------------------------------------
class TestPortfolioWeightsIntegration:

    def test_weights_sum_to_one_all_rows(self, analytics_result):
        df, _, _ = analytics_result
        weight_sum = df["equity_weight"] + df["credit_weight"] + df["cash_weight"]
        max_deviation = (weight_sum - 1.0).abs().max()
        assert max_deviation < 1e-4, f"Max weight deviation from 1.0: {max_deviation}"

    def test_equity_weights_non_negative(self, analytics_result):
        df, _, _ = analytics_result
        assert (df["equity_weight"] >= 0).all()

    def test_credit_weights_non_negative(self, analytics_result):
        df, _, _ = analytics_result
        assert (df["credit_weight"] >= 0).all()

    def test_cash_weights_non_negative(self, analytics_result):
        df, _, _ = analytics_result
        assert (df["cash_weight"] >= 0).all()

    def test_duration_bias_column_present(self, analytics_result):
        df, _, _ = analytics_result
        assert "duration_bias" in df.columns


# ---------------------------------------------------------------------------
# Goal 5: Backtest columns exist
# ---------------------------------------------------------------------------
class TestBacktestColumns:

    def test_backtest_columns_exist(self, analytics_result):
        df, _, _ = analytics_result
        missing = [c for c in _BACKTEST_COLUMNS if c not in df.columns]
        assert not missing, f"Missing backtest columns: {missing}"

    def test_backtest_summary_keys(self, analytics_result):
        _, _, backtest_summary = analytics_result
        missing = _BACKTEST_SUMMARY_KEYS - backtest_summary.keys()
        assert not missing, f"Missing backtest summary keys: {missing}"

    def test_strategy_equity_curve_starts_near_one(self, analytics_result):
        df, _, _ = analytics_result
        first_valid = df["strategy_equity_curve"].dropna().iloc[0]
        assert abs(first_valid - 1.0) < 0.05

    def test_sp500_equity_curve_starts_near_one(self, analytics_result):
        df, _, _ = analytics_result
        first_valid = df["sp500_equity_curve"].dropna().iloc[0]
        assert abs(first_valid - 1.0) < 0.05

    def test_strategy_drawdown_non_positive(self, analytics_result):
        df, _, _ = analytics_result
        assert (df["strategy_drawdown"].dropna() <= 0).all()

    def test_sp500_backtest_drawdown_non_positive(self, analytics_result):
        df, _, _ = analytics_result
        assert (df["sp500_backtest_drawdown"].dropna() <= 0).all()


# ---------------------------------------------------------------------------
# Goal 6: final_decision has no missing values
# ---------------------------------------------------------------------------
class TestFinalDecision:

    def test_final_decision_no_nulls(self, pipeline_df):
        null_count = pipeline_df["final_decision"].isna().sum()
        assert null_count == 0, f"final_decision has {null_count} null rows"

    def test_final_decision_is_string(self, pipeline_df):
        # pandas 3.0 uses StringDtype instead of object for string columns
        assert pd.api.types.is_string_dtype(pipeline_df["final_decision"])

    def test_final_decision_values_are_known(self, pipeline_df):
        known = {
            "Buy Stress", "Watch Entry", "Stress / Stabilization Watch",
            "Credit Warning", "Divergence Warning", "Avoid Chasing Risk",
            "Hold / Do Not Chase", "Wait", "Neutral",
        }
        actual = set(pipeline_df["final_decision"].unique())
        unknown = actual - known
        assert not unknown, f"Unexpected final_decision values: {unknown}"

    def test_final_environment_no_nulls(self, pipeline_df):
        assert pipeline_df["final_environment"].isna().sum() == 0

    def test_final_action_no_nulls(self, pipeline_df):
        assert pipeline_df["final_action"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Health check and model confidence (bonus: verifies analytics outputs)
# ---------------------------------------------------------------------------
class TestHealthCheckAndConfidence:

    def test_health_check_keys_present(self, analytics_result):
        _, health_check, _ = analytics_result
        missing = _HEALTH_CHECK_KEYS - health_check.keys()
        assert not missing, f"Missing health check keys: {missing}"

    def test_portfolio_weight_check_passes(self, analytics_result):
        _, health_check, _ = analytics_result
        assert health_check["portfolio_weight_check"] == "PASS"

    def test_model_confidence_valid_values(self, analytics_result):
        df, _, _ = analytics_result
        actual = set(df["model_confidence"].unique())
        unknown = actual - _VALID_CONFIDENCE_VALUES
        assert not unknown, f"Unexpected confidence values: {unknown}"

    def test_model_confidence_no_nulls(self, analytics_result):
        df, _, _ = analytics_result
        assert df["model_confidence"].isna().sum() == 0
