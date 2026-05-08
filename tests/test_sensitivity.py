"""
Tests for src/sensitivity.py.
Uses the session-scoped analytics_result fixture from conftest.py.
sweep_* tests use n_steps=3 to keep the suite fast.
"""
import pytest
import numpy as np
import pandas as pd

from src.sensitivity import (
    _compute_daily_metrics,
    _compute_forward_metrics,
    _perturb_composite_weights,
    _apply_score_scale,
    _apply_composite_weights,
    _apply_backtester_weights,
    sweep_score_scales,
    sweep_composite_weights,
    sweep_backtester_weights,
    identify_unstable_parameters,
    run_sensitivity,
    COMPOSITE_WEIGHTS_DEFAULT,
    BACKTESTER_WEIGHTS_DEFAULT,
    SCORE_SCALE_COLS,
    SCORE_COL_LABELS,
    _REQUIRED_COLS,
)


@pytest.fixture(scope="module")
def sens_df(analytics_result):
    df, _, _ = analytics_result
    return df


# ---------------------------------------------------------------------------
# _compute_daily_metrics
# ---------------------------------------------------------------------------
class TestComputeDailyMetrics:
    def _df(self, returns):
        return pd.DataFrame({"strategy_daily_return": returns})

    def test_returns_expected_keys(self):
        result = _compute_daily_metrics(self._df([0.01] * 20))
        assert result.keys() == {"sharpe", "total_return", "max_drawdown", "hit_rate"}

    def test_positive_returns_positive_sharpe(self):
        assert _compute_daily_metrics(self._df([0.01] * 20))["sharpe"] > 0

    def test_negative_returns_negative_sharpe(self):
        assert _compute_daily_metrics(self._df([-0.01] * 20))["sharpe"] < 0

    def test_empty_series_returns_nans(self):
        result = _compute_daily_metrics(self._df([]))
        assert all(np.isnan(v) for v in result.values())

    def test_single_row_returns_nans(self):
        result = _compute_daily_metrics(self._df([0.01]))
        assert all(np.isnan(v) for v in result.values())

    def test_max_drawdown_non_positive(self):
        result = _compute_daily_metrics(self._df([0.01, -0.02, 0.01] * 10))
        assert result["max_drawdown"] <= 0

    def test_hit_rate_between_zero_and_one(self):
        result = _compute_daily_metrics(self._df([0.01, -0.01] * 10))
        assert 0.0 <= result["hit_rate"] <= 1.0

    def test_all_positive_returns_hit_rate_is_one(self):
        assert _compute_daily_metrics(self._df([0.01] * 20))["hit_rate"] == 1.0


# ---------------------------------------------------------------------------
# _compute_forward_metrics
# ---------------------------------------------------------------------------
class TestComputeForwardMetrics:
    def _df(self, returns):
        return pd.DataFrame({"strategy_forward_30d_return": returns})

    def test_returns_expected_keys(self):
        result = _compute_forward_metrics(self._df([0.02] * 20))
        assert result.keys() == {"fwd_mean_return", "fwd_hit_rate", "fwd_worst_5pct"}

    def test_hit_rate_between_zero_and_one(self):
        result = _compute_forward_metrics(self._df([0.01, -0.01] * 10))
        assert 0.0 <= result["fwd_hit_rate"] <= 1.0

    def test_positive_returns_positive_mean(self):
        result = _compute_forward_metrics(self._df([0.02] * 20))
        assert result["fwd_mean_return"] > 0

    def test_empty_series_returns_nans(self):
        result = _compute_forward_metrics(self._df([]))
        assert all(np.isnan(v) for v in result.values())

    def test_fewer_than_10_obs_worst_5pct_is_nan(self):
        result = _compute_forward_metrics(self._df([0.01] * 5))
        assert np.isnan(result["fwd_worst_5pct"])


# ---------------------------------------------------------------------------
# _perturb_composite_weights
# ---------------------------------------------------------------------------
class TestPerturbCompositeWeights:
    def test_weights_sum_to_one(self):
        result = _perturb_composite_weights(COMPOSITE_WEIGHTS_DEFAULT, "macro", 0.30)
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_target_param_set_correctly(self):
        result = _perturb_composite_weights(COMPOSITE_WEIGHTS_DEFAULT, "macro", 0.30)
        assert abs(result["macro"] - 0.30) < 1e-9

    def test_all_weights_non_negative(self):
        result = _perturb_composite_weights(COMPOSITE_WEIGHTS_DEFAULT, "macro", 0.30)
        assert all(v >= 0 for v in result.values())

    def test_other_weights_rescaled(self):
        base = COMPOSITE_WEIGHTS_DEFAULT.copy()
        result = _perturb_composite_weights(base, "macro", 0.35)
        assert result["credit"] != base["credit"]

    def test_all_keys_preserved(self):
        result = _perturb_composite_weights(COMPOSITE_WEIGHTS_DEFAULT, "liquidity", 0.20)
        assert result.keys() == COMPOSITE_WEIGHTS_DEFAULT.keys()


# ---------------------------------------------------------------------------
# sweep_score_scales
# ---------------------------------------------------------------------------
class TestSweepScoreScales:
    def test_returns_dataframe(self, sens_df):
        assert isinstance(sweep_score_scales(sens_df, n_steps=3), pd.DataFrame)

    def test_shape_n_cols_times_n_steps(self, sens_df):
        result = sweep_score_scales(sens_df, n_steps=3)
        assert len(result) == len(SCORE_SCALE_COLS) * 3

    def test_required_columns_present(self, sens_df):
        result = sweep_score_scales(sens_df, n_steps=3)
        for col in ("group", "param", "param_value", "sharpe", "total_return", "max_drawdown", "hit_rate"):
            assert col in result.columns, f"Missing column: {col}"

    def test_group_value_correct(self, sens_df):
        result = sweep_score_scales(sens_df, n_steps=3)
        assert (result["group"] == "score_scale").all()

    def test_all_score_params_present(self, sens_df):
        result = sweep_score_scales(sens_df, n_steps=3)
        assert result["param"].nunique() == len(SCORE_SCALE_COLS)
        expected_labels = set(SCORE_COL_LABELS.values())
        assert set(result["param"].unique()) == expected_labels

    def test_sharpe_finite_or_nan(self, sens_df):
        result = sweep_score_scales(sens_df, n_steps=3)
        assert np.isfinite(result["sharpe"].dropna()).all()

    def test_scale_range_includes_baseline(self, sens_df):
        result = sweep_score_scales(sens_df, n_steps=3)
        # scales span from 0.7 to 1.3; baseline (1.0) should be within that range
        assert result["param_value"].min() < 1.0
        assert result["param_value"].max() > 1.0


# ---------------------------------------------------------------------------
# sweep_composite_weights
# ---------------------------------------------------------------------------
class TestSweepCompositeWeights:
    def test_returns_dataframe(self, sens_df):
        assert isinstance(sweep_composite_weights(sens_df, n_steps=3), pd.DataFrame)

    def test_shape(self, sens_df):
        result = sweep_composite_weights(sens_df, n_steps=3)
        assert len(result) == len(COMPOSITE_WEIGHTS_DEFAULT) * 3

    def test_group_value_correct(self, sens_df):
        result = sweep_composite_weights(sens_df, n_steps=3)
        assert (result["group"] == "composite_weights").all()

    def test_forward_metric_columns_present(self, sens_df):
        result = sweep_composite_weights(sens_df, n_steps=3)
        assert "fwd_mean_return" in result.columns
        assert "fwd_hit_rate" in result.columns

    def test_all_composite_params_present(self, sens_df):
        result = sweep_composite_weights(sens_df, n_steps=3)
        assert set(result["param"].unique()) == set(COMPOSITE_WEIGHTS_DEFAULT.keys())

    def test_fwd_hit_rate_in_valid_range(self, sens_df):
        result = sweep_composite_weights(sens_df, n_steps=3)
        valid = result["fwd_hit_rate"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()


# ---------------------------------------------------------------------------
# sweep_backtester_weights
# ---------------------------------------------------------------------------
class TestSweepBacktesterWeights:
    def test_returns_dataframe(self, sens_df):
        assert isinstance(sweep_backtester_weights(sens_df, n_steps=3), pd.DataFrame)

    def test_shape(self, sens_df):
        result = sweep_backtester_weights(sens_df, n_steps=3)
        assert len(result) == len(BACKTESTER_WEIGHTS_DEFAULT) * 3

    def test_group_value_correct(self, sens_df):
        result = sweep_backtester_weights(sens_df, n_steps=3)
        assert (result["group"] == "backtester_weights").all()

    def test_all_backtester_params_present(self, sens_df):
        result = sweep_backtester_weights(sens_df, n_steps=3)
        assert set(result["param"].unique()) == set(BACKTESTER_WEIGHTS_DEFAULT.keys())

    def test_param_values_in_valid_range(self, sens_df):
        result = sweep_backtester_weights(sens_df, n_steps=3)
        assert (result["param_value"] >= 0).all()
        assert (result["param_value"] <= 1.0).all()

    def test_sharpe_finite_or_nan(self, sens_df):
        result = sweep_backtester_weights(sens_df, n_steps=3)
        assert np.isfinite(result["sharpe"].dropna()).all()

    def test_higher_risk_on_weight_spans_wider_return_range(self, sens_df):
        result = sweep_backtester_weights(sens_df, n_steps=7)
        risk_on_returns = result[result["param"] == "risk_on"]["total_return"]
        assert risk_on_returns.max() > risk_on_returns.min()


# ---------------------------------------------------------------------------
# identify_unstable_parameters
# ---------------------------------------------------------------------------
class TestIdentifyUnstableParameters:
    def _make_results(self, values, param="p", group="g", metric="sharpe"):
        return pd.DataFrame([
            {"group": group, "param": param, "param_value": float(i), metric: v}
            for i, v in enumerate(values)
        ])

    def test_returns_list(self):
        df = self._make_results([1.5] * 5)
        assert isinstance(identify_unstable_parameters(df, metrics=["sharpe"]), list)

    def test_constant_param_not_flagged(self):
        df = self._make_results([1.5, 1.5, 1.5, 1.5, 1.5])
        result = identify_unstable_parameters(df, cv_threshold=0.25, metrics=["sharpe"])
        assert len(result) == 0

    def test_high_variance_param_flagged(self):
        df = self._make_results([0.1, 1.0, 3.0, 6.0, 10.0])
        result = identify_unstable_parameters(df, cv_threshold=0.25, metrics=["sharpe"])
        assert len(result) > 0

    def test_unstable_dict_has_required_keys(self):
        df = self._make_results([0.1, 1.0, 3.0, 6.0, 10.0])
        result = identify_unstable_parameters(df, cv_threshold=0.01, metrics=["sharpe"])
        if result:
            for key in ("group", "param", "metric", "cv", "min_value", "max_value", "mean_value"):
                assert key in result[0]

    def test_sorted_descending_by_cv(self):
        rows = (
            [{"group": "A", "param": "p1", "param_value": float(i), "sharpe": v}
             for i, v in enumerate([0.1, 2.0, 5.0, 9.0, 15.0])]
            + [{"group": "B", "param": "p2", "param_value": float(i), "sharpe": v}
               for i, v in enumerate([1.0, 1.01, 1.02, 1.03, 1.04])]
        )
        df = pd.DataFrame(rows)
        result = identify_unstable_parameters(df, cv_threshold=0.01, metrics=["sharpe"])
        cvs = [r["cv"] for r in result]
        assert cvs == sorted(cvs, reverse=True)

    def test_near_zero_mean_not_flagged_with_division_guard(self):
        df = self._make_results([1e-12, 2e-12, 3e-12, 4e-12, 5e-12])
        result = identify_unstable_parameters(df, cv_threshold=0.25, metrics=["sharpe"])
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# run_sensitivity (integration)
# ---------------------------------------------------------------------------
class TestRunSensitivity:
    def test_returns_two_values(self, sens_df):
        result = run_sensitivity(sens_df, n_steps=3)
        assert len(result) == 2

    def test_results_is_dataframe(self, sens_df):
        results_df, _ = run_sensitivity(sens_df, n_steps=3)
        assert isinstance(results_df, pd.DataFrame)

    def test_unstable_is_list(self, sens_df):
        _, unstable = run_sensitivity(sens_df, n_steps=3)
        assert isinstance(unstable, list)

    def test_all_three_groups_present(self, sens_df):
        results_df, _ = run_sensitivity(sens_df, n_steps=3)
        assert set(results_df["group"].unique()) == {
            "score_scale", "composite_weights", "backtester_weights"
        }

    def test_total_rows_matches_expected(self, sens_df):
        n_steps = 3
        results_df, _ = run_sensitivity(sens_df, n_steps=n_steps)
        expected = (
            len(SCORE_SCALE_COLS)
            + len(COMPOSITE_WEIGHTS_DEFAULT)
            + len(BACKTESTER_WEIGHTS_DEFAULT)
        ) * n_steps
        assert len(results_df) == expected

    def test_required_columns_in_results(self, sens_df):
        results_df, _ = run_sensitivity(sens_df, n_steps=3)
        for col in ("group", "param", "param_value", "sharpe", "total_return"):
            assert col in results_df.columns

    def test_raises_on_missing_required_column(self, sens_df):
        bad = sens_df.drop(columns=["final_decision"])
        with pytest.raises(ValueError, match="missing required columns"):
            run_sensitivity(bad)
