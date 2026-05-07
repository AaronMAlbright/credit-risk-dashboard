"""
Tests for src/signal_decay.py.

Covers forward-return computation, regime decay statistics, score
correlations, and the run_signal_decay orchestrator.
"""

import numpy as np
import pandas as pd
import pytest

from src.signal_decay import (
    HORIZONS,
    compute_decay_by_regime,
    compute_forward_returns,
    compute_score_correlations,
    run_signal_decay,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_df():
    """200-row DataFrame with sp500 price, scores, and two regimes."""
    n = 200
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    price = 4000 + np.cumsum(rng.normal(0, 15, n))
    decisions = ["Neutral"] * 100 + ["Buy Stress"] * 100
    return pd.DataFrame({
        "date":                            dates,
        "sp500":                           price,
        "sp500_forward_30d_return":        rng.normal(0.02, 0.05, n),
        "sp500_forward_60d_return":        rng.normal(0.04, 0.07, n),
        "final_decision":                  decisions,
        "macro_risk_score_smooth":         rng.uniform(10, 80, n),
        "credit_market_risk_score_smooth": rng.uniform(10, 70, n),
        "composite_risk_score_smooth":     rng.uniform(20, 75, n),
    })


@pytest.fixture(scope="module")
def no_price_df():
    """DataFrame without the sp500 price column."""
    n = 60
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "date":             pd.date_range("2023-01-01", periods=n, freq="B"),
        "final_decision":   ["Neutral"] * n,
        "sp500_forward_30d_return": rng.normal(0.01, 0.03, n),
    })


@pytest.fixture(scope="module")
def tiny_df():
    """3-row DataFrame — below bootstrap threshold for some horizons."""
    return pd.DataFrame({
        "date":           pd.date_range("2023-01-01", periods=3, freq="B"),
        "sp500":          [4000.0, 4010.0, 4020.0],
        "final_decision": ["Neutral"] * 3,
    })


# ---------------------------------------------------------------------------
# TestComputeForwardReturns
# ---------------------------------------------------------------------------


class TestComputeForwardReturns:
    def test_returns_dataframe(self, base_df):
        result = compute_forward_returns(base_df)
        assert isinstance(result, pd.DataFrame)

    def test_adds_fwd_columns_for_all_horizons(self, base_df):
        result = compute_forward_returns(base_df, HORIZONS)
        for n in HORIZONS:
            assert f"sp500_fwd_{n}d" in result.columns

    def test_does_not_modify_original(self, base_df):
        orig_cols = set(base_df.columns)
        compute_forward_returns(base_df)
        assert set(base_df.columns) == orig_cols

    def test_reuses_existing_30d_col(self, base_df):
        result = compute_forward_returns(base_df, [30])
        # Should be identical to the precomputed column
        pd.testing.assert_series_equal(
            result["sp500_fwd_30d"].reset_index(drop=True),
            base_df["sp500_forward_30d_return"].reset_index(drop=True),
            check_names=False,
        )

    def test_reuses_existing_60d_col(self, base_df):
        result = compute_forward_returns(base_df, [60])
        pd.testing.assert_series_equal(
            result["sp500_fwd_60d"].reset_index(drop=True),
            base_df["sp500_forward_60d_return"].reset_index(drop=True),
            check_names=False,
        )

    def test_computed_7d_last_rows_are_nan(self, base_df):
        result = compute_forward_returns(base_df, [7])
        assert result["sp500_fwd_7d"].iloc[-7:].isna().all()

    def test_computed_7d_first_rows_are_not_nan(self, base_df):
        result = compute_forward_returns(base_df, [7])
        assert result["sp500_fwd_7d"].iloc[:10].notna().all()

    def test_no_price_col_skips_new_horizons(self, no_price_df):
        result = compute_forward_returns(no_price_df, [7])
        assert "sp500_fwd_7d" not in result.columns

    def test_no_price_col_still_reuses_30d(self, no_price_df):
        result = compute_forward_returns(no_price_df, [30])
        assert "sp500_fwd_30d" in result.columns

    def test_idempotent_when_col_already_present(self, base_df):
        result1 = compute_forward_returns(base_df, [7])
        result2 = compute_forward_returns(result1, [7])
        pd.testing.assert_series_equal(result1["sp500_fwd_7d"], result2["sp500_fwd_7d"])

    def test_7d_return_direction_matches_price(self, base_df):
        result = compute_forward_returns(base_df, [7])
        # Spot-check row 0: fwd_7d ≈ price[7]/price[0] - 1
        price = base_df["sp500"].values
        expected = price[7] / price[0] - 1
        assert abs(result["sp500_fwd_7d"].iloc[0] - expected) < 1e-9


# ---------------------------------------------------------------------------
# TestComputeDecayByRegime
# ---------------------------------------------------------------------------


class TestComputeDecayByRegime:
    def test_returns_dataframe(self, base_df):
        result = compute_decay_by_regime(base_df, n_boot=50)
        assert isinstance(result, pd.DataFrame)

    def test_index_is_multiindex(self, base_df):
        result = compute_decay_by_regime(base_df, n_boot=50)
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ["regime", "horizon"]

    def test_expected_columns(self, base_df):
        result = compute_decay_by_regime(base_df, n_boot=50)
        for col in ["mean_return", "hit_rate", "n_obs", "ci_lower", "ci_upper"]:
            assert col in result.columns

    def test_both_regimes_present(self, base_df):
        result = compute_decay_by_regime(base_df, n_boot=50)
        regimes = result.index.get_level_values("regime").unique()
        assert "Neutral" in regimes
        assert "Buy Stress" in regimes

    def test_all_horizons_present_per_regime(self, base_df):
        result = compute_decay_by_regime(base_df, horizons=[7, 30], n_boot=50)
        for regime in ["Neutral", "Buy Stress"]:
            for h in [7, 30]:
                assert (regime, h) in result.index

    def test_hit_rate_in_unit_interval(self, base_df):
        result = compute_decay_by_regime(base_df, n_boot=50)
        for hr in result["hit_rate"].dropna():
            assert 0.0 <= hr <= 1.0

    def test_ci_ordering(self, base_df):
        result = compute_decay_by_regime(base_df, n_boot=100)
        clean = result.dropna(subset=["ci_lower", "ci_upper"])
        assert (clean["ci_lower"] <= clean["mean_return"]).all()
        assert (clean["mean_return"] <= clean["ci_upper"]).all()

    def test_n_obs_matches_regime_size(self, base_df):
        result = compute_decay_by_regime(base_df, horizons=[30], n_boot=50)
        # Regime sizes: Neutral=100, Buy Stress=100; 30d col has no NaN in fixture
        n_neutral = result.loc[("Neutral", 30), "n_obs"]
        assert n_neutral == 100

    def test_deterministic_with_seed(self, base_df):
        r1 = compute_decay_by_regime(base_df, horizons=[30], n_boot=100)
        r2 = compute_decay_by_regime(base_df, horizons=[30], n_boot=100)
        pd.testing.assert_frame_equal(r1, r2)

    def test_rng_state_restored(self, base_df):
        state_before = np.random.get_state()[1][:5].tolist()
        compute_decay_by_regime(base_df, n_boot=50)
        state_after = np.random.get_state()[1][:5].tolist()
        assert state_before == state_after

    def test_empty_df_returns_empty(self):
        result = compute_decay_by_regime(pd.DataFrame())
        assert result.empty

    def test_missing_regime_col_returns_empty(self, base_df):
        result = compute_decay_by_regime(base_df, regime_col="nonexistent", n_boot=10)
        assert result.empty

    def test_tiny_df_ci_nan_when_below_threshold(self, tiny_df):
        result = compute_decay_by_regime(tiny_df, horizons=[7], n_boot=50)
        if not result.empty and ("Neutral", 7) in result.index:
            n = result.loc[("Neutral", 7), "n_obs"]
            if n < 5:
                assert np.isnan(result.loc[("Neutral", 7), "ci_lower"])

    def test_custom_horizons_respected(self, base_df):
        custom = [14, 90]
        result = compute_decay_by_regime(base_df, horizons=custom, n_boot=50)
        present_horizons = result.index.get_level_values("horizon").unique().tolist()
        for h in custom:
            assert h in present_horizons
        assert 30 not in present_horizons


# ---------------------------------------------------------------------------
# TestComputeScoreCorrelations
# ---------------------------------------------------------------------------


class TestComputeScoreCorrelations:
    def test_returns_dataframe(self, base_df):
        result = compute_score_correlations(base_df)
        assert isinstance(result, pd.DataFrame)

    def test_columns_are_horizons(self, base_df):
        result = compute_score_correlations(base_df, HORIZONS)
        for h in HORIZONS:
            assert h in result.columns

    def test_index_contains_present_signals(self, base_df):
        result = compute_score_correlations(base_df)
        assert "Macro Risk" in result.index
        assert "Composite" in result.index

    def test_correlations_in_valid_range(self, base_df):
        result = compute_score_correlations(base_df)
        for val in result.values.flatten():
            if not np.isnan(val):
                assert -1.0 <= val <= 1.0

    def test_missing_score_col_excluded(self, base_df):
        df2 = base_df.drop(columns=["credit_market_risk_score_smooth"])
        result = compute_score_correlations(df2)
        assert "Credit Risk" not in result.index

    def test_no_score_cols_returns_empty(self):
        df = pd.DataFrame({
            "date":   pd.date_range("2023-01-01", periods=30, freq="B"),
            "sp500":  range(4000, 4030),
            "final_decision": ["Neutral"] * 30,
        })
        result = compute_score_correlations(df, HORIZONS)
        assert result.empty

    def test_nan_when_too_few_obs(self, tiny_df):
        df_aug = tiny_df.copy()
        df_aug["macro_risk_score_smooth"] = [30.0, 40.0, 50.0]
        result = compute_score_correlations(df_aug, [7])
        if "Macro Risk" in result.index:
            assert np.isnan(result.loc["Macro Risk", 7])

    def test_high_correlation_detected(self):
        n = 100
        x = np.linspace(10, 80, n)
        rng = np.random.default_rng(5)
        df = pd.DataFrame({
            "date":                        pd.date_range("2022-01-01", periods=n, freq="B"),
            "sp500":                       4000 + np.cumsum(rng.normal(0, 5, n)),
            "macro_risk_score_smooth":     x,
            "sp500_forward_30d_return":    x / 100 + rng.normal(0, 0.002, n),
        })
        result = compute_score_correlations(df, [30])
        assert "Macro Risk" in result.index
        assert abs(result.loc["Macro Risk", 30]) > 0.8

    def test_shape_signals_x_horizons(self, base_df):
        result = compute_score_correlations(base_df, [7, 30, 90])
        assert result.shape[1] == 3


# ---------------------------------------------------------------------------
# TestRunSignalDecay
# ---------------------------------------------------------------------------


class TestRunSignalDecay:
    def test_returns_dict(self, base_df):
        result = run_signal_decay(base_df)
        assert isinstance(result, dict)

    def test_all_keys_present(self, base_df):
        result = run_signal_decay(base_df)
        for key in ["decay_by_regime", "score_correlations", "horizons"]:
            assert key in result

    def test_decay_by_regime_is_dataframe(self, base_df):
        result = run_signal_decay(base_df)
        assert isinstance(result["decay_by_regime"], pd.DataFrame)

    def test_score_correlations_is_dataframe(self, base_df):
        result = run_signal_decay(base_df)
        assert isinstance(result["score_correlations"], pd.DataFrame)

    def test_horizons_matches_default(self, base_df):
        result = run_signal_decay(base_df)
        assert result["horizons"] == HORIZONS

    def test_custom_horizons_propagate(self, base_df):
        custom = [14, 60]
        result = run_signal_decay(base_df, horizons=custom)
        assert result["horizons"] == custom
        present = result["decay_by_regime"].index.get_level_values("horizon").unique().tolist()
        for h in custom:
            assert h in present

    def test_empty_df_returns_empty_decay(self):
        result = run_signal_decay(pd.DataFrame())
        assert result["decay_by_regime"].empty
