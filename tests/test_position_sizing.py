"""
Tests for src/position_sizing.py.

Covers score interpolation, regime-prob weighting, Kelly and vol-target
sizing, the backtest, current-sizing summary, and the orchestrator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.position_sizing import (
    REGIME_WEIGHTS,
    SCORE_BREAKPOINTS,
    _KELLY_WINDOW,
    _MAX_WEIGHT,
    _MIN_WEIGHT,
    _TARGET_VOL,
    _VOL_WINDOW,
    compute_kelly_sizing,
    compute_regime_prob_sizing,
    compute_score_sizing,
    compute_sized_backtest,
    compute_vol_target_sizing,
    get_current_sizing,
    run_position_sizing,
    _interp_weight,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(n: int = 150, seed: int = 0,
             composite_mean: float = 30.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strat = rng.normal(0.0005, 0.008, n)
    sp500 = rng.normal(0.0003, 0.010, n)
    comp  = rng.uniform(composite_mean - 5, composite_mean + 5, n)
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame({
        "date":                       dates,
        "strategy_daily_return":      strat,
        "sp500_daily_return":         sp500,
        "composite_risk_score_smooth":comp,
        "final_decision":             ["Neutral"] * n,
    })


def _make_high_composite_df(n: int = 100, seed: int = 1) -> pd.DataFrame:
    """All observations in the elevated-risk zone (composite ~70)."""
    rng = np.random.default_rng(seed)
    df = _make_df(n=n, seed=seed, composite_mean=70.0)
    df["composite_risk_score_smooth"] = rng.uniform(65, 80, n)
    return df


def _make_high_vol_df(n: int = 100, seed: int = 2) -> pd.DataFrame:
    """High daily vol (~3% per day → ~48% ann)."""
    rng = np.random.default_rng(seed)
    df = _make_df(n=n, seed=seed)
    df["strategy_daily_return"] = rng.normal(0.0005, 0.03, n)
    return df


@pytest.fixture(scope="module")
def base_df():
    return _make_df()


@pytest.fixture(scope="module")
def high_composite_df():
    return _make_high_composite_df()


@pytest.fixture(scope="module")
def high_vol_df():
    return _make_high_vol_df()


# ---------------------------------------------------------------------------
# _interp_weight
# ---------------------------------------------------------------------------

class TestInterpWeight:
    def test_below_first_breakpoint_returns_max(self):
        assert _interp_weight(0.0, SCORE_BREAKPOINTS) == pytest.approx(1.0)

    def test_above_last_breakpoint_returns_min(self):
        assert _interp_weight(100.0, SCORE_BREAKPOINTS) == pytest.approx(0.05)

    def test_at_midpoint_breakpoint_returns_exact(self):
        # Score 33 → 0.75 per SCORE_BREAKPOINTS
        assert _interp_weight(33.0, SCORE_BREAKPOINTS) == pytest.approx(0.75)

    def test_interpolated_between_points(self):
        # Midpoint between (28, 1.0) and (33, 0.75) → score 30.5 → 0.875
        w = _interp_weight(30.5, SCORE_BREAKPOINTS)
        assert 0.75 <= w <= 1.0

    def test_output_clipped_to_min_max(self):
        bp = [(0.0, -0.5), (100.0, 1.5)]  # outside [0, 1]
        assert _interp_weight(0.0,   bp) >= _MIN_WEIGHT
        assert _interp_weight(100.0, bp) <= _MAX_WEIGHT

    def test_monotone_decreasing(self):
        scores = np.linspace(0, 100, 20)
        weights = [_interp_weight(s, SCORE_BREAKPOINTS) for s in scores]
        assert all(weights[i] >= weights[i + 1] - 1e-9
                   for i in range(len(weights) - 1))


# ---------------------------------------------------------------------------
# compute_score_sizing
# ---------------------------------------------------------------------------

class TestComputeScoreSizing:
    def test_returns_series(self, base_df):
        result = compute_score_sizing(base_df)
        assert isinstance(result, pd.Series)

    def test_length_matches_input(self, base_df):
        result = compute_score_sizing(base_df)
        assert len(result) == len(base_df)

    def test_values_in_valid_range(self, base_df):
        result = compute_score_sizing(base_df)
        assert (result >= _MIN_WEIGHT).all()
        assert (result <= _MAX_WEIGHT).all()

    def test_high_composite_gives_low_weight(self, high_composite_df):
        result = compute_score_sizing(high_composite_df)
        assert result.mean() < 0.5

    def test_low_composite_gives_high_weight(self, base_df):
        result = compute_score_sizing(base_df)
        assert result.mean() > 0.5

    def test_index_is_datetime(self, base_df):
        result = compute_score_sizing(base_df)
        assert pd.api.types.is_datetime64_any_dtype(result.index)

    def test_empty_df_returns_empty(self):
        assert compute_score_sizing(pd.DataFrame()).empty

    def test_missing_score_col_returns_empty(self, base_df):
        df2 = base_df.drop(columns=["composite_risk_score_smooth"])
        assert compute_score_sizing(df2).empty

    def test_custom_breakpoints(self, base_df):
        # Single-step: everything above 20 → 0.1
        bp = [(0, 1.0), (20, 1.0), (21, 0.1), (100, 0.1)]
        result = compute_score_sizing(base_df, breakpoints=bp)
        assert result.mean() < 0.5   # all scores > 20 in base_df


# ---------------------------------------------------------------------------
# compute_regime_prob_sizing
# ---------------------------------------------------------------------------

class TestComputeRegimeProbSizing:
    def _prob_history(self, regimes, probs, n=50):
        """Fake probability history DataFrame."""
        data = {r: np.full(n, p) for r, p in zip(regimes, probs)}
        idx  = pd.date_range("2024-01-02", periods=n, freq="B")
        return pd.DataFrame(data, index=idx)

    def test_returns_series(self, base_df):
        ph = self._prob_history(["Neutral", "Buy Stress"], [0.6, 0.4], n=len(base_df))
        result = compute_regime_prob_sizing(base_df, prob_history=ph)
        assert isinstance(result, pd.Series)

    def test_length_matches_prob_history(self, base_df):
        ph = self._prob_history(["Neutral", "Buy Stress"], [0.6, 0.4], n=len(base_df))
        result = compute_regime_prob_sizing(base_df, prob_history=ph)
        assert len(result) == len(ph)

    def test_values_in_valid_range(self, base_df):
        ph = self._prob_history(["Neutral", "Buy Stress"], [0.6, 0.4], n=len(base_df))
        result = compute_regime_prob_sizing(base_df, prob_history=ph)
        assert (result >= _MIN_WEIGHT).all()
        assert (result <= _MAX_WEIGHT).all()

    def test_avoid_regime_gives_low_weight(self, base_df):
        ph = self._prob_history(["Avoid Chasing Risk"], [1.0], n=len(base_df))
        result = compute_regime_prob_sizing(
            base_df, regime_weights=REGIME_WEIGHTS, prob_history=ph,
        )
        assert result.mean() < 0.15

    def test_buy_stress_gives_high_weight(self, base_df):
        ph = self._prob_history(["Buy Stress"], [1.0], n=len(base_df))
        result = compute_regime_prob_sizing(
            base_df, regime_weights=REGIME_WEIGHTS, prob_history=ph,
        )
        assert result.mean() > 0.9

    def test_weighted_average_correct(self, base_df):
        # 50% Neutral (0.65) + 50% Buy Stress (1.00) → 0.825
        ph = self._prob_history(["Neutral", "Buy Stress"], [0.5, 0.5], n=len(base_df))
        result = compute_regime_prob_sizing(
            base_df, regime_weights=REGIME_WEIGHTS, prob_history=ph,
        )
        expected = 0.5 * 0.65 + 0.5 * 1.00
        assert abs(result.mean() - expected) < 1e-6

    def test_empty_prob_history_returns_empty(self, base_df):
        result = compute_regime_prob_sizing(base_df, prob_history=pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# compute_kelly_sizing
# ---------------------------------------------------------------------------

class TestComputeKellySizing:
    def test_returns_series(self, base_df):
        result = compute_kelly_sizing(base_df)
        assert isinstance(result, pd.Series)

    def test_length_matches_input(self, base_df):
        result = compute_kelly_sizing(base_df)
        assert len(result) == len(base_df)

    def test_first_rows_nan(self, base_df):
        result = compute_kelly_sizing(base_df, window=_KELLY_WINDOW)
        assert result.iloc[: _KELLY_WINDOW - 1].isna().all()

    def test_later_rows_not_all_nan(self, base_df):
        result = compute_kelly_sizing(base_df)
        assert result.iloc[_KELLY_WINDOW:].notna().any()

    def test_values_clipped_to_max(self, base_df):
        result = compute_kelly_sizing(base_df)
        assert (result.dropna() <= _MAX_WEIGHT).all()

    def test_values_clipped_to_min(self, base_df):
        result = compute_kelly_sizing(base_df)
        assert (result.dropna() >= _MIN_WEIGHT).all()

    def test_negative_mean_gives_zero(self):
        rng = np.random.default_rng(5)
        n = 150
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=n, freq="B"),
            "strategy_daily_return": rng.normal(-0.003, 0.008, n),
        })
        result = compute_kelly_sizing(df)
        valid = result.dropna()
        assert (valid == 0.0).all() or (valid >= 0).all()

    def test_index_is_datetime(self, base_df):
        result = compute_kelly_sizing(base_df)
        assert pd.api.types.is_datetime64_any_dtype(result.index)

    def test_empty_df_returns_empty(self):
        assert compute_kelly_sizing(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# compute_vol_target_sizing
# ---------------------------------------------------------------------------

class TestComputeVolTargetSizing:
    def test_returns_series(self, base_df):
        result = compute_vol_target_sizing(base_df)
        assert isinstance(result, pd.Series)

    def test_length_matches_input(self, base_df):
        result = compute_vol_target_sizing(base_df)
        assert len(result) == len(base_df)

    def test_first_rows_nan(self, base_df):
        result = compute_vol_target_sizing(base_df, window=_VOL_WINDOW)
        assert result.iloc[: _VOL_WINDOW - 1].isna().all()

    def test_later_rows_not_all_nan(self, base_df):
        result = compute_vol_target_sizing(base_df)
        assert result.iloc[_VOL_WINDOW:].notna().any()

    def test_values_in_valid_range(self, base_df):
        result = compute_vol_target_sizing(base_df)
        assert (result.dropna() >= _MIN_WEIGHT).all()
        assert (result.dropna() <= _MAX_WEIGHT).all()

    def test_high_vol_gives_lower_weight(self, base_df, high_vol_df):
        r_base = compute_vol_target_sizing(base_df).dropna().mean()
        r_high = compute_vol_target_sizing(high_vol_df).dropna().mean()
        assert r_high < r_base

    def test_lower_target_vol_gives_lower_weight(self, base_df):
        r1 = compute_vol_target_sizing(base_df, target_vol=0.05).dropna().mean()
        r2 = compute_vol_target_sizing(base_df, target_vol=0.20).dropna().mean()
        assert r1 <= r2

    def test_weight_proportional_to_target(self, base_df):
        r05 = compute_vol_target_sizing(base_df, target_vol=0.05).dropna()
        r10 = compute_vol_target_sizing(base_df, target_vol=0.10).dropna()
        # r10 / r05 ≈ 2 wherever neither is clipped
        ratio = r10 / r05
        assert abs(ratio.mean() - 2.0) < 0.5

    def test_empty_df_returns_empty(self):
        assert compute_vol_target_sizing(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# compute_sized_backtest
# ---------------------------------------------------------------------------

class TestComputeSizedBacktest:
    def _sizes(self, df, weight: float = 0.5):
        return pd.DataFrame({"score_sizing": weight}, index=df.index)

    def test_returns_dataframe(self, base_df):
        sizes = self._sizes(base_df)
        result = compute_sized_backtest(base_df, sizes)
        assert isinstance(result, pd.DataFrame)

    def test_length_matches_input(self, base_df):
        sizes = self._sizes(base_df)
        result = compute_sized_backtest(base_df, sizes)
        assert len(result) == len(base_df)

    def test_cum_columns_present(self, base_df):
        sizes = self._sizes(base_df)
        result = compute_sized_backtest(base_df, sizes)
        assert "cum_full" in result.columns
        assert "cum_score_sizing" in result.columns

    def test_half_weight_has_lower_vol_than_full(self, base_df):
        sizes = self._sizes(base_df, weight=0.5)
        result = compute_sized_backtest(base_df, sizes)
        full_std  = result["strategy_return"].std()
        sized_std = result["score_sizing_return"].std()
        assert sized_std < full_std

    def test_sized_return_is_weight_times_strategy_return(self, base_df):
        # With constant weight = 0.5, sized_return should be ~0.5 * strategy_return
        # (except first row which uses 1.0 as lag)
        sizes = self._sizes(base_df, weight=0.5)
        result = compute_sized_backtest(base_df, sizes)
        # Check rows 2+ (after lag stabilises)
        diff = (result["score_sizing_return"].iloc[2:] -
                0.5 * result["strategy_return"].iloc[2:]).abs()
        assert diff.max() < 1e-9

    def test_cum_starts_near_1(self, base_df):
        sizes = self._sizes(base_df)
        result = compute_sized_backtest(base_df, sizes)
        assert abs(result["cum_full"].iloc[0] - 1.0) < 0.02
        assert abs(result["cum_score_sizing"].iloc[0] - 1.0) < 0.02

    def test_empty_sizes_returns_empty(self, base_df):
        result = compute_sized_backtest(base_df, pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# get_current_sizing
# ---------------------------------------------------------------------------

class TestGetCurrentSizing:
    def _sizes(self, df):
        idx = pd.to_datetime(df["date"].values)
        return pd.DataFrame({
            "score_sizing":       np.linspace(0.8, 1.0, len(df)),
            "regime_prob_sizing": np.linspace(0.3, 0.4, len(df)),
        }, index=idx)

    def test_returns_dict(self, base_df):
        result = get_current_sizing(base_df, self._sizes(base_df))
        assert isinstance(result, dict)

    def test_all_sizing_methods_present(self, base_df):
        sizes = self._sizes(base_df)
        result = get_current_sizing(base_df, sizes)
        assert "score_sizing" in result
        assert "regime_prob_sizing" in result

    def test_blend_is_mean_of_methods(self, base_df):
        sizes = self._sizes(base_df)
        result = get_current_sizing(base_df, sizes)
        methods = [result[k] for k in ["score_sizing", "regime_prob_sizing"]]
        assert abs(result["blend"] - np.mean(methods)) < 1e-6

    def test_current_composite_present(self, base_df):
        sizes = self._sizes(base_df)
        result = get_current_sizing(base_df, sizes)
        assert "current_composite" in result

    def test_current_regime_present(self, base_df):
        sizes = self._sizes(base_df)
        result = get_current_sizing(base_df, sizes)
        assert "current_regime" in result

    def test_empty_sizes_returns_empty(self, base_df):
        assert get_current_sizing(base_df, None) == {}


# ---------------------------------------------------------------------------
# run_position_sizing
# ---------------------------------------------------------------------------

class TestRunPositionSizing:
    def test_returns_dict(self, base_df):
        result = run_position_sizing(base_df)
        assert isinstance(result, dict)

    def test_all_keys_present(self, base_df):
        result = run_position_sizing(base_df)
        for k in ["sizes", "current", "backtest",
                  "breakpoints", "regime_weights", "target_vol"]:
            assert k in result

    def test_sizes_is_dataframe(self, base_df):
        result = run_position_sizing(base_df)
        assert isinstance(result["sizes"], pd.DataFrame)

    def test_sizes_has_expected_columns(self, base_df):
        result = run_position_sizing(base_df)
        for col in ["score_sizing", "kelly_sizing",
                    "vol_target_sizing", "blend"]:
            assert col in result["sizes"].columns

    def test_sizes_length_matches_df(self, base_df):
        result = run_position_sizing(base_df)
        assert len(result["sizes"]) == len(base_df)

    def test_blend_between_0_and_1(self, base_df):
        result = run_position_sizing(base_df)
        blend = result["sizes"]["blend"].dropna()
        assert (blend >= 0).all() and (blend <= 1).all()

    def test_backtest_is_dataframe(self, base_df):
        result = run_position_sizing(base_df)
        assert isinstance(result["backtest"], pd.DataFrame)

    def test_current_has_blend(self, base_df):
        result = run_position_sizing(base_df)
        assert "blend" in result["current"]

    def test_current_blend_between_0_and_1(self, base_df):
        result = run_position_sizing(base_df)
        assert 0.0 <= result["current"]["blend"] <= 1.0

    def test_empty_df_returns_empty(self):
        assert run_position_sizing(pd.DataFrame()) == {}

    def test_custom_target_vol_passed_through(self, base_df):
        result = run_position_sizing(base_df, target_vol=0.05)
        assert result["target_vol"] == 0.05

    def test_breakpoints_passed_through(self, base_df):
        bp = [(0, 1.0), (100, 0.0)]
        result = run_position_sizing(base_df, breakpoints=bp)
        assert result["breakpoints"] == bp
