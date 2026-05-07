"""Tests for src/model_health_check.py."""

import pandas as pd
import pytest
from src.model_health_check import (
    check_missing_values,
    check_portfolio_weights,
    check_sample_sizes,
    check_score_bounds,
    run_model_health_check,
)


def _make_df(n=50, **overrides):
    base = {
        "macro_risk_score_smooth":           [40.0] * n,
        "credit_market_risk_score_smooth":   [35.0] * n,
        "liquidity_regime_score_smooth":     [30.0] * n,
        "cross_asset_divergence_score_smooth": [25.0] * n,
        "market_internals_score_smooth":     [45.0] * n,
        "risk_appetite_score_smooth":        [50.0] * n,
        "complacency_score_smooth":          [20.0] * n,
        "mean_reversion_score_smooth":       [15.0] * n,
        "treasury_stress_score_smooth":      [10.0] * n,
        "final_decision":                    ["Hold / Do Not Chase"] * n,
        "equity_weight":                     [0.40] * n,
        "credit_weight":                     [0.25] * n,
        "cash_weight":                       [0.35] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# check_portfolio_weights
# ---------------------------------------------------------------------------

class TestCheckPortfolioWeights:
    def test_pass_when_sum_is_one(self):
        row = {"equity_weight": 0.40, "credit_weight": 0.25, "cash_weight": 0.35}
        assert check_portfolio_weights(row) == "PASS"

    def test_fail_when_sum_not_one(self):
        row = {"equity_weight": 0.40, "credit_weight": 0.30, "cash_weight": 0.35}
        result = check_portfolio_weights(row)
        assert result.startswith("FAIL")

    def test_tolerance_respected(self):
        row = {"equity_weight": 0.40, "credit_weight": 0.25, "cash_weight": 0.3509}
        assert check_portfolio_weights(row, tolerance=0.01) == "PASS"


# ---------------------------------------------------------------------------
# check_missing_values
# ---------------------------------------------------------------------------

class TestCheckMissingValues:
    def test_all_present_returns_zeros(self):
        df = _make_df()
        cols = ["macro_risk_score_smooth", "final_decision"]
        result = check_missing_values(df, cols)
        assert all(result[c] == 0 for c in cols)

    def test_missing_column_flagged(self):
        df = _make_df()
        result = check_missing_values(df, ["nonexistent_col"])
        assert result["nonexistent_col"] == "MISSING COLUMN"

    def test_nan_counted(self):
        df = _make_df()
        df.loc[0, "macro_risk_score_smooth"] = None
        result = check_missing_values(df, ["macro_risk_score_smooth"])
        assert result["macro_risk_score_smooth"] == 1


# ---------------------------------------------------------------------------
# check_sample_sizes
# ---------------------------------------------------------------------------

class TestCheckSampleSizes:
    def test_pass_when_all_regimes_have_enough_obs(self):
        df = _make_df(n=50)
        assert check_sample_sizes(df, "final_decision") == "PASS"

    def test_fail_when_regime_too_small(self):
        df = _make_df(n=50)
        df.loc[0, "final_decision"] = "Rare Regime"
        result = check_sample_sizes(df, "final_decision", min_obs=30)
        assert "Rare Regime" in result

    def test_custom_min_obs(self):
        df = _make_df(n=10)
        result = check_sample_sizes(df, "final_decision", min_obs=5)
        assert result == "PASS"


# ---------------------------------------------------------------------------
# check_score_bounds
# ---------------------------------------------------------------------------

class TestCheckScoreBounds:
    def test_pass_when_all_in_bounds(self):
        df = _make_df()
        cols = ["macro_risk_score_smooth", "credit_market_risk_score_smooth"]
        assert check_score_bounds(df, cols) == "PASS"

    def test_detects_below_zero(self):
        df = _make_df()
        df.loc[0, "macro_risk_score_smooth"] = -5.0
        result = check_score_bounds(df, ["macro_risk_score_smooth"])
        assert "macro_risk_score_smooth" in result
        assert result["macro_risk_score_smooth"]["below_zero"] == 1

    def test_detects_above_100(self):
        df = _make_df()
        df.loc[0, "macro_risk_score_smooth"] = 105.0
        result = check_score_bounds(df, ["macro_risk_score_smooth"])
        assert result["macro_risk_score_smooth"]["above_100"] == 1

    def test_missing_column_flagged(self):
        df = _make_df()
        result = check_score_bounds(df, ["nonexistent_score"])
        assert result["nonexistent_score"] == "MISSING COLUMN"


# ---------------------------------------------------------------------------
# run_model_health_check
# ---------------------------------------------------------------------------

class TestRunModelHealthCheck:
    def test_returns_expected_keys(self):
        df = _make_df()
        result = run_model_health_check(df)
        for key in ("portfolio_weight_check", "missing_values",
                    "final_decision_sample_check", "score_bounds_check"):
            assert key in result

    def test_clean_df_passes_all_checks(self):
        df = _make_df(n=50)
        result = run_model_health_check(df)
        assert result["portfolio_weight_check"] == "PASS"
        assert result["score_bounds_check"] == "PASS"
        assert result["final_decision_sample_check"] == "PASS"
