"""
Tests for src/validation_guard.py — structural and behavioral.

Covers confidence classification, per-regime/transition/window audits,
correlation audit, warning generation, and the full run_validation_audit
orchestrator.
"""

import numpy as np
import pandas as pd
import pytest

from src.validation_guard import (
    CONFIDENCE_EXPLORATORY,
    CONFIDENCE_INDICATIVE,
    CONFIDENCE_ROBUST,
    MIN_OBS_REGIME,
    MIN_TRANSITIONS,
    MIN_WINDOWS,
    MIN_INFORMATIVE_CORR,
    classify_confidence,
    audit_regime_stats,
    audit_transition_matrix,
    audit_walk_forward,
    audit_correlations,
    generate_warnings,
    run_validation_audit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def robust_df():
    """DataFrame with two regimes, each having >50 obs and forward returns."""
    n = 200
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    decisions = ["Neutral"] * 100 + ["Risk-On"] * 100
    returns = np.concatenate([
        rng.normal(0.02, 0.05, 100),
        rng.normal(0.05, 0.04, 100),
    ])
    scores = rng.uniform(10, 80, n)
    return pd.DataFrame({
        "date": dates,
        "final_decision": decisions,
        "sp500_forward_30d_return": returns,
        "macro_risk_score_smooth": scores,
        "composite_risk_score_smooth": scores + rng.normal(0, 3, n),
    })


@pytest.fixture(scope="module")
def sparse_df():
    """DataFrame with one under-threshold regime (< 20 obs)."""
    n = 10
    rng = np.random.default_rng(2)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "date": dates,
        "final_decision": ["Risk-Off"] * n,
        "sp500_forward_30d_return": rng.normal(0.01, 0.05, n),
        "macro_risk_score_smooth": rng.uniform(30, 70, n),
        "composite_risk_score_smooth": rng.uniform(30, 70, n),
    })


@pytest.fixture(scope="module")
def windows_robust():
    """15 walk-forward windows with consistent positive Sharpe."""
    rng = np.random.default_rng(3)
    n = 15
    strategy_sharpe = rng.normal(1.2, 0.2, n)
    sp500_sharpe = rng.normal(0.8, 0.2, n)
    return pd.DataFrame({
        "strategy_sharpe": strategy_sharpe,
        "sp500_sharpe": sp500_sharpe,
    })


@pytest.fixture(scope="module")
def windows_sparse():
    """4 walk-forward windows (below robust threshold)."""
    return pd.DataFrame({
        "strategy_sharpe": [0.5, -0.2, 0.8, 0.3],
        "sp500_sharpe":    [0.4,  0.3, 0.5, 0.4],
    })


@pytest.fixture(scope="module")
def counts_df_robust():
    """Transition count matrix with robust counts (> 20 outgoing per row)."""
    regimes = ["Neutral", "Risk-On", "Caution"]
    data = {
        "Neutral": [10, 15, 5],
        "Risk-On": [12,  8, 8],
        "Caution": [5,   3, 20],
    }
    return pd.DataFrame(data, index=regimes)


@pytest.fixture(scope="module")
def counts_df_sparse():
    """Transition count matrix where one row has < 5 outgoing transitions."""
    regimes = ["Neutral", "Risk-Off"]
    data = {
        "Neutral":  [20, 2],
        "Risk-Off": [1, 1],
    }
    return pd.DataFrame(data, index=regimes)


# ---------------------------------------------------------------------------
# TestClassifyConfidence
# ---------------------------------------------------------------------------


class TestClassifyConfidence:
    def test_robust_regime(self):
        assert classify_confidence(MIN_OBS_REGIME["robust"], "regime") == CONFIDENCE_ROBUST

    def test_indicative_regime_low(self):
        assert classify_confidence(MIN_OBS_REGIME["indicative"], "regime") == CONFIDENCE_INDICATIVE

    def test_indicative_regime_high(self):
        n = MIN_OBS_REGIME["robust"] - 1
        assert classify_confidence(n, "regime") == CONFIDENCE_INDICATIVE

    def test_exploratory_regime(self):
        assert classify_confidence(MIN_OBS_REGIME["indicative"] - 1, "regime") == CONFIDENCE_EXPLORATORY

    def test_robust_transition(self):
        assert classify_confidence(MIN_TRANSITIONS["robust"], "transition") == CONFIDENCE_ROBUST

    def test_indicative_transition(self):
        assert classify_confidence(MIN_TRANSITIONS["indicative"], "transition") == CONFIDENCE_INDICATIVE

    def test_exploratory_transition(self):
        assert classify_confidence(MIN_TRANSITIONS["indicative"] - 1, "transition") == CONFIDENCE_EXPLORATORY

    def test_robust_window(self):
        assert classify_confidence(MIN_WINDOWS["robust"], "window") == CONFIDENCE_ROBUST

    def test_indicative_window(self):
        assert classify_confidence(MIN_WINDOWS["indicative"], "window") == CONFIDENCE_INDICATIVE

    def test_exploratory_window(self):
        assert classify_confidence(MIN_WINDOWS["indicative"] - 1, "window") == CONFIDENCE_EXPLORATORY

    def test_unknown_kind_defaults_to_regime_thresholds(self):
        result = classify_confidence(MIN_OBS_REGIME["robust"], "foobar")
        assert result == CONFIDENCE_ROBUST

    def test_zero_obs_is_exploratory(self):
        assert classify_confidence(0, "regime") == CONFIDENCE_EXPLORATORY

    def test_large_n_is_robust(self):
        assert classify_confidence(10_000, "regime") == CONFIDENCE_ROBUST


# ---------------------------------------------------------------------------
# TestAuditRegimeStats
# ---------------------------------------------------------------------------


class TestAuditRegimeStats:
    def test_returns_dataframe(self, robust_df):
        result = audit_regime_stats(robust_df)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns_present(self, robust_df):
        result = audit_regime_stats(robust_df)
        for col in ["n_obs", "mean_return", "hit_rate", "confidence", "suppressed"]:
            assert col in result.columns

    def test_both_regimes_present(self, robust_df):
        result = audit_regime_stats(robust_df)
        assert "Neutral" in result.index
        assert "Risk-On" in result.index

    def test_robust_regimes_not_suppressed(self, robust_df):
        result = audit_regime_stats(robust_df)
        assert not result.loc["Neutral", "suppressed"]
        assert not result.loc["Risk-On", "suppressed"]

    def test_sparse_regime_exploratory(self, sparse_df):
        result = audit_regime_stats(sparse_df)
        assert result.iloc[0]["confidence"] == CONFIDENCE_EXPLORATORY

    def test_sparse_regime_suppressed(self, sparse_df):
        result = audit_regime_stats(sparse_df)
        assert result.iloc[0]["suppressed"]

    def test_n_obs_matches_data(self, robust_df):
        result = audit_regime_stats(robust_df)
        assert result.loc["Neutral", "n_obs"] == 100
        assert result.loc["Risk-On", "n_obs"] == 100

    def test_mean_return_is_float(self, robust_df):
        result = audit_regime_stats(robust_df)
        assert isinstance(result.loc["Neutral", "mean_return"], float)

    def test_hit_rate_in_unit_interval(self, robust_df):
        result = audit_regime_stats(robust_df)
        for r in result["hit_rate"].dropna():
            assert 0.0 <= r <= 1.0

    def test_missing_regime_col_returns_empty(self, robust_df):
        result = audit_regime_stats(robust_df, regime_col="nonexistent")
        assert result.empty

    def test_missing_return_col_no_crash(self, robust_df):
        df2 = robust_df.drop(columns=["sp500_forward_30d_return"])
        result = audit_regime_stats(df2)
        assert isinstance(result, pd.DataFrame)
        assert result["mean_return"].isna().all()

    def test_worst_5pct_nan_when_few_obs(self):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=3, freq="B"),
            "final_decision": ["Risk-Off"] * 3,
            "sp500_forward_30d_return": [0.01, 0.02, -0.01],
        })
        result = audit_regime_stats(df)
        assert np.isnan(result.iloc[0]["worst_5pct"])

    def test_sorted_by_n_obs_descending(self, robust_df):
        result = audit_regime_stats(robust_df)
        assert result["n_obs"].is_monotonic_decreasing or len(result) == 1


# ---------------------------------------------------------------------------
# TestAuditTransitionMatrix
# ---------------------------------------------------------------------------


class TestAuditTransitionMatrix:
    def test_returns_dataframe(self, counts_df_robust):
        result = audit_transition_matrix(counts_df_robust)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, counts_df_robust):
        result = audit_transition_matrix(counts_df_robust)
        for col in ["n_outgoing", "confidence", "suppressed"]:
            assert col in result.columns

    def test_robust_rows_not_suppressed(self, counts_df_robust):
        result = audit_transition_matrix(counts_df_robust)
        for regime in counts_df_robust.index:
            n_out = int(counts_df_robust.loc[regime].sum())
            if n_out >= MIN_TRANSITIONS["robust"]:
                assert not result.loc[regime, "suppressed"]

    def test_sparse_row_suppressed(self, counts_df_sparse):
        result = audit_transition_matrix(counts_df_sparse)
        n_out_stress = int(counts_df_sparse.loc["Risk-Off"].sum())
        if n_out_stress < MIN_TRANSITIONS["indicative"]:
            assert result.loc["Risk-Off", "suppressed"]

    def test_none_returns_empty(self):
        result = audit_transition_matrix(None)
        assert result.empty

    def test_empty_df_returns_empty(self):
        result = audit_transition_matrix(pd.DataFrame())
        assert result.empty

    def test_n_outgoing_sums_match(self, counts_df_robust):
        result = audit_transition_matrix(counts_df_robust)
        for regime in counts_df_robust.index:
            expected = int(counts_df_robust.loc[regime].sum())
            assert result.loc[regime, "n_outgoing"] == expected


# ---------------------------------------------------------------------------
# TestAuditWalkForward
# ---------------------------------------------------------------------------


class TestAuditWalkForward:
    def test_robust_windows_confidence(self, windows_robust):
        result = audit_walk_forward(windows_robust)
        assert result["confidence"] == CONFIDENCE_ROBUST

    def test_sparse_windows_confidence(self, windows_sparse):
        result = audit_walk_forward(windows_sparse)
        assert result["confidence"] in [CONFIDENCE_INDICATIVE, CONFIDENCE_EXPLORATORY]

    def test_none_returns_exploratory(self):
        result = audit_walk_forward(None)
        assert result["confidence"] == CONFIDENCE_EXPLORATORY
        assert result["n_windows"] == 0

    def test_empty_df_returns_exploratory(self):
        result = audit_walk_forward(pd.DataFrame())
        assert result["confidence"] == CONFIDENCE_EXPLORATORY

    def test_n_windows_correct(self, windows_robust):
        result = audit_walk_forward(windows_robust)
        assert result["n_windows"] == len(windows_robust)

    def test_sparse_windows_has_warning(self, windows_sparse):
        result = audit_walk_forward(windows_sparse)
        assert len(result["warnings"]) > 0

    def test_robust_windows_no_count_warning(self, windows_robust):
        result = audit_walk_forward(windows_robust)
        count_warnings = [w for w in result["warnings"] if "window" in w.lower()]
        assert len(count_warnings) == 0

    def test_sharpe_cv_computed(self, windows_robust):
        result = audit_walk_forward(windows_robust)
        assert not np.isnan(result["sharpe_cv"])

    def test_sharpe_cv_nan_when_no_col(self):
        df = pd.DataFrame({"other_col": [1.0, 2.0, 3.0]})
        result = audit_walk_forward(df)
        assert np.isnan(result["sharpe_cv"])

    def test_high_sharpe_cv_triggers_warning(self):
        df = pd.DataFrame({
            "strategy_sharpe": [10.0, -10.0, 10.0, -10.0, 10.0, -10.0,
                                  10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0],
            "sp500_sharpe":    [0.5] * 13,
        })
        result = audit_walk_forward(df)
        cv_warnings = [w for w in result["warnings"] if "variability" in w.lower()]
        assert len(cv_warnings) > 0

    def test_poor_beat_rate_triggers_warning(self):
        df = pd.DataFrame({
            "strategy_sharpe": [-0.5] * 15,
            "sp500_sharpe":     [0.5] * 15,
        })
        result = audit_walk_forward(df)
        beat_warnings = [w for w in result["warnings"] if "beat" in w.lower()]
        assert len(beat_warnings) > 0

    def test_warnings_is_list(self, windows_robust):
        result = audit_walk_forward(windows_robust)
        assert isinstance(result["warnings"], list)


# ---------------------------------------------------------------------------
# TestAuditCorrelations
# ---------------------------------------------------------------------------


class TestAuditCorrelations:
    def test_returns_dataframe(self, robust_df):
        result = audit_correlations(robust_df)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, robust_df):
        result = audit_correlations(robust_df)
        for col in ["correlation", "n_obs", "informative", "confidence"]:
            assert col in result.columns

    def test_high_correlation_is_informative(self):
        n = 200
        x = np.linspace(0, 1, n)
        df = pd.DataFrame({
            "macro_risk_score_smooth": x,
            "sp500_forward_30d_return": x + np.random.default_rng(9).normal(0, 0.01, n),
        })
        result = audit_correlations(df)
        assert "Macro Risk" in result.index
        assert result.loc["Macro Risk", "informative"]

    def test_uninformative_correlation_flagged(self):
        n = 200
        rng = np.random.default_rng(10)
        df = pd.DataFrame({
            "macro_risk_score_smooth": rng.uniform(0, 100, n),
            "sp500_forward_30d_return": rng.uniform(-0.001, 0.001, n),
        })
        result = audit_correlations(df)
        if "Macro Risk" in result.index:
            corr = result.loc["Macro Risk", "correlation"]
            if abs(corr) < MIN_INFORMATIVE_CORR:
                assert not result.loc["Macro Risk", "informative"]

    def test_too_few_obs_exploratory(self):
        n = 10
        rng = np.random.default_rng(11)
        df = pd.DataFrame({
            "macro_risk_score_smooth": rng.uniform(0, 100, n),
            "sp500_forward_30d_return": rng.normal(0, 0.05, n),
        })
        result = audit_correlations(df)
        if "Macro Risk" in result.index:
            assert result.loc["Macro Risk", "confidence"] == CONFIDENCE_EXPLORATORY

    def test_missing_return_col_returns_empty(self, robust_df):
        df2 = robust_df.drop(columns=["sp500_forward_30d_return"])
        result = audit_correlations(df2)
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ---------------------------------------------------------------------------
# TestGenerateWarnings
# ---------------------------------------------------------------------------


class TestGenerateWarnings:
    def test_returns_list(self):
        audit = {"walk_forward": {"warnings": []}}
        result = generate_warnings(audit)
        assert isinstance(result, list)

    def test_exploratory_regime_produces_warning(self):
        regime_stats = pd.DataFrame({
            "n_obs": [5],
            "confidence": [CONFIDENCE_EXPLORATORY],
            "suppressed": [True],
        }, index=["Risk-Off"])
        audit = {
            "walk_forward": {"warnings": []},
            "regime_stats": regime_stats,
        }
        warnings = generate_warnings(audit)
        assert any("exploratory" in w.lower() for w in warnings)

    def test_indicative_regime_produces_warning(self):
        regime_stats = pd.DataFrame({
            "n_obs": [30],
            "confidence": [CONFIDENCE_INDICATIVE],
            "suppressed": [False],
        }, index=["Risk-On"])
        audit = {
            "walk_forward": {"warnings": []},
            "regime_stats": regime_stats,
        }
        warnings = generate_warnings(audit)
        assert any("indicative" in w.lower() for w in warnings)

    def test_exploratory_transition_produces_warning(self):
        trans_df = pd.DataFrame({
            "n_outgoing": [2],
            "confidence": [CONFIDENCE_EXPLORATORY],
            "suppressed": [True],
        }, index=["Caution"])
        audit = {
            "walk_forward": {"warnings": []},
            "transition_matrix": trans_df,
        }
        warnings = generate_warnings(audit)
        assert any("transition" in w.lower() for w in warnings)

    def test_wf_warnings_included(self):
        audit = {
            "walk_forward": {"warnings": ["Only 3 windows available."]},
        }
        warnings = generate_warnings(audit)
        assert "Only 3 windows available." in warnings

    def test_no_warnings_for_robust_audit(self, robust_df, windows_robust, counts_df_robust):
        from src.regime_transition import compute_transition_matrix
        _, count_df = compute_transition_matrix(robust_df["final_decision"])
        audit = run_validation_audit(
            robust_df, windows_df=windows_robust,
            transition_counts=count_df,
        )
        wf_warn = [w for w in audit["warnings"] if "window" in w.lower()]
        assert len(wf_warn) == 0


# ---------------------------------------------------------------------------
# TestRunValidationAudit
# ---------------------------------------------------------------------------


class TestRunValidationAudit:
    def test_returns_dict(self, robust_df):
        result = run_validation_audit(robust_df)
        assert isinstance(result, dict)

    def test_all_keys_present(self, robust_df):
        result = run_validation_audit(robust_df)
        for key in ["regime_stats", "transition_matrix", "walk_forward",
                    "correlations", "warnings", "summary"]:
            assert key in result

    def test_summary_has_overall_confidence(self, robust_df):
        result = run_validation_audit(robust_df)
        assert "overall_confidence" in result["summary"]

    def test_summary_counts_sum(self, robust_df):
        result = run_validation_audit(robust_df)
        s = result["summary"]
        total = s["n_robust"] + s["n_indicative"] + s["n_exploratory"]
        assert total >= 1

    def test_overall_confidence_valid_tier(self, robust_df):
        result = run_validation_audit(robust_df)
        assert result["summary"]["overall_confidence"] in [
            CONFIDENCE_ROBUST, CONFIDENCE_INDICATIVE, CONFIDENCE_EXPLORATORY
        ]

    def test_robust_data_gives_robust_or_indicative(self, robust_df, windows_robust):
        result = run_validation_audit(robust_df, windows_df=windows_robust)
        assert result["summary"]["overall_confidence"] in [
            CONFIDENCE_ROBUST, CONFIDENCE_INDICATIVE
        ]

    def test_sparse_data_gives_exploratory(self, sparse_df):
        result = run_validation_audit(sparse_df)
        assert result["summary"]["overall_confidence"] == CONFIDENCE_EXPLORATORY

    def test_transition_counts_used_when_provided(self, robust_df, counts_df_robust):
        result = run_validation_audit(robust_df, transition_counts=counts_df_robust)
        assert not result["transition_matrix"].empty

    def test_no_transition_counts_empty_matrix(self, robust_df):
        result = run_validation_audit(robust_df, transition_counts=None)
        assert result["transition_matrix"].empty

    def test_warnings_is_list(self, robust_df):
        result = run_validation_audit(robust_df)
        assert isinstance(result["warnings"], list)

    def test_custom_regime_col(self, robust_df):
        df2 = robust_df.copy()
        df2["alt_regime"] = df2["final_decision"]
        result = run_validation_audit(df2, regime_col="alt_regime")
        assert isinstance(result["regime_stats"], pd.DataFrame)
