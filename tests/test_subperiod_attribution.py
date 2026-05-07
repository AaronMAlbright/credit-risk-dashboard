"""
Tests for src/subperiod_attribution.py.

Covers period definition, per-period stats, the full table,
rolling time series, regime frequency pivot, and the orchestrator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.subperiod_attribution import (
    _MIN_OBS,
    _ROLL_WINDOW,
    _STRAT_COL,
    _SP500_COL,
    _LABEL_COL,
    _DATE_COL,
    compute_period_stats,
    compute_regime_frequency,
    compute_rolling_stats,
    compute_subperiod_table,
    define_periods,
    run_subperiod_attribution,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(years: list[int] = None, n_per_year: int = 120,
             seed: int = 0) -> pd.DataFrame:
    """Multi-year synthetic dataset with known regime labels."""
    if years is None:
        years = [2023, 2024, 2025]
    rng   = np.random.default_rng(seed)
    rows  = []
    for y in years:
        dates = pd.date_range(f"{y}-01-02", periods=n_per_year, freq="B")
        strat = rng.normal(0.0004, 0.010, n_per_year)
        sp500 = rng.normal(0.0003, 0.012, n_per_year)
        regimes = np.random.default_rng(seed + y).choice(
            ["Neutral", "Watch Entry", "Buy Stress"], size=n_per_year,
        )
        composite = rng.uniform(20, 60, n_per_year)
        for i in range(n_per_year):
            rows.append({
                _DATE_COL:   dates[i],
                _STRAT_COL:  strat[i],
                _SP500_COL:  sp500[i],
                _LABEL_COL:  regimes[i],
                "composite_risk_score_smooth": composite[i],
            })
    return pd.DataFrame(rows).sort_values(_DATE_COL).reset_index(drop=True)


def _make_known_df(n: int = 60, strat_mean: float = 0.001,
                   sp500_mean: float = 0.0, seed: int = 1) -> pd.DataFrame:
    """Dataset with known expected Sharpe/beta for analytical tests."""
    rng   = np.random.default_rng(seed)
    sp500 = rng.normal(sp500_mean, 0.010, n)
    strat = 0.5 * sp500 + strat_mean + rng.normal(0, 0.003, n)
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame({
        _DATE_COL:   dates,
        _STRAT_COL:  strat,
        _SP500_COL:  sp500,
        _LABEL_COL:  ["Neutral"] * n,
        "composite_risk_score_smooth": 30.0,
    })


@pytest.fixture(scope="module")
def multi_year_df():
    return _make_df()


@pytest.fixture(scope="module")
def known_df():
    return _make_known_df()


# ---------------------------------------------------------------------------
# define_periods
# ---------------------------------------------------------------------------

class TestDefinePeriods:
    def test_returns_list(self, multi_year_df):
        periods = define_periods(multi_year_df)
        assert isinstance(periods, list)

    def test_last_period_is_full(self, multi_year_df):
        periods = define_periods(multi_year_df)
        assert periods[-1]["is_full"] is True
        assert periods[-1]["label"] == "Full Period"

    def test_year_periods_not_full(self, multi_year_df):
        periods = define_periods(multi_year_df)
        for p in periods[:-1]:
            assert p["is_full"] is False

    def test_correct_number_of_periods(self, multi_year_df):
        # 3 years + 1 full period
        periods = define_periods(multi_year_df)
        assert len(periods) == 4

    def test_n_obs_correct(self, multi_year_df):
        periods = define_periods(multi_year_df)
        for p in periods[:-1]:
            assert p["n_obs"] == 120

    def test_full_period_n_obs(self, multi_year_df):
        periods = define_periods(multi_year_df)
        assert periods[-1]["n_obs"] == len(multi_year_df)

    def test_periods_ordered_chronologically(self, multi_year_df):
        periods = define_periods(multi_year_df)[:-1]  # exclude full
        starts = [p["start"] for p in periods]
        assert starts == sorted(starts)

    def test_partial_year_label(self):
        df = _make_df(years=[2024], n_per_year=60)
        # Force partial: start in February
        df = df[df[_DATE_COL] >= "2024-02-01"].reset_index(drop=True)
        periods = define_periods(df)
        year_periods = [p for p in periods if not p["is_full"]]
        assert any("partial" in p["label"] for p in year_periods)

    def test_short_year_excluded(self):
        # A year with fewer than _MIN_OBS rows should be excluded
        df = _make_df(years=[2023, 2024], n_per_year=120)
        # Keep only 5 rows for 2023
        keep_2024 = df[df[_DATE_COL].dt.year == 2024]
        keep_2023 = df[df[_DATE_COL].dt.year == 2023].head(_MIN_OBS - 1)
        df2 = pd.concat([keep_2023, keep_2024]).reset_index(drop=True)
        periods = define_periods(df2)
        year_labels = [p["label"] for p in periods if not p["is_full"]]
        assert not any("2023" in lbl for lbl in year_labels)

    def test_empty_df_returns_empty(self):
        assert define_periods(pd.DataFrame()) == []

    def test_each_period_has_required_keys(self, multi_year_df):
        periods = define_periods(multi_year_df)
        for p in periods:
            for k in ["label", "start", "end", "n_obs", "is_full"]:
                assert k in p


# ---------------------------------------------------------------------------
# compute_period_stats
# ---------------------------------------------------------------------------

class TestComputePeriodStats:
    def test_returns_dict(self, known_df):
        result = compute_period_stats(known_df)
        assert isinstance(result, dict)

    def test_all_expected_keys_present(self, known_df):
        result = compute_period_stats(known_df)
        for k in ["n_obs", "strat_sharpe", "strat_max_dd",
                  "strat_ann_ret", "strat_ann_vol", "strat_hit_rate",
                  "sp500_sharpe", "beta", "ann_alpha", "r2"]:
            assert k in result, f"Missing: {k}"

    def test_n_obs_correct(self, known_df):
        result = compute_period_stats(known_df)
        assert result["n_obs"] == len(known_df)

    def test_beta_near_known_value(self, known_df):
        result = compute_period_stats(known_df)
        assert abs(result["beta"] - 0.5) < 0.1

    def test_sharpe_positive_for_positive_mean(self, known_df):
        result = compute_period_stats(known_df)
        assert result["strat_sharpe"] > 0

    def test_max_dd_nonpositive(self, known_df):
        result = compute_period_stats(known_df)
        assert result["strat_max_dd"] <= 0.0

    def test_hit_rate_between_0_and_1(self, known_df):
        result = compute_period_stats(known_df)
        assert 0.0 <= result["strat_hit_rate"] <= 1.0

    def test_r2_between_0_and_1(self, known_df):
        result = compute_period_stats(known_df)
        assert 0.0 <= result["r2"] <= 1.0

    def test_excess_ret_equals_strat_minus_sp500(self, known_df):
        result = compute_period_stats(known_df)
        expected = result["strat_ann_ret"] - result["sp500_ann_ret"]
        assert abs(result["excess_ann_ret"] - expected) < 1e-9

    def test_ann_alpha_positive_for_positive_mean_strat(self, known_df):
        result = compute_period_stats(known_df)
        assert result["ann_alpha"] > 0

    def test_empty_df_returns_empty(self):
        assert compute_period_stats(pd.DataFrame()) == {}

    def test_too_few_rows_returns_empty(self):
        df = _make_known_df(n=_MIN_OBS - 1)
        assert compute_period_stats(df) == {}

    def test_declining_strategy_has_negative_total_ret(self):
        rng = np.random.default_rng(9)
        n = 60
        df = pd.DataFrame({
            _DATE_COL:  pd.date_range("2024-01-02", periods=n, freq="B"),
            _STRAT_COL: np.full(n, -0.002),
            _SP500_COL: np.full(n, 0.001),
            _LABEL_COL: ["Neutral"] * n,
        })
        result = compute_period_stats(df)
        assert result["strat_total_ret"] < 0

    def test_mean_composite_present_when_col_exists(self, known_df):
        result = compute_period_stats(known_df)
        assert "mean_composite" in result
        assert abs(result["mean_composite"] - 30.0) < 1e-6


# ---------------------------------------------------------------------------
# compute_subperiod_table
# ---------------------------------------------------------------------------

class TestComputeSubperiodTable:
    def test_returns_dataframe(self, multi_year_df):
        result = compute_subperiod_table(multi_year_df)
        assert isinstance(result, pd.DataFrame)

    def test_index_contains_full_period(self, multi_year_df):
        result = compute_subperiod_table(multi_year_df)
        assert "Full Period" in result.index

    def test_row_count(self, multi_year_df):
        # 3 years + 1 full period
        result = compute_subperiod_table(multi_year_df)
        assert len(result) == 4

    def test_sharpe_column_exists(self, multi_year_df):
        result = compute_subperiod_table(multi_year_df)
        assert "strat_sharpe" in result.columns

    def test_n_obs_sums_correctly(self, multi_year_df):
        result = compute_subperiod_table(multi_year_df)
        year_total = result.loc[result.index != "Full Period", "n_obs"].sum()
        assert year_total == len(multi_year_df)

    def test_full_period_n_obs_equals_df_length(self, multi_year_df):
        result = compute_subperiod_table(multi_year_df)
        assert result.loc["Full Period", "n_obs"] == len(multi_year_df)

    def test_empty_df_returns_empty(self):
        assert compute_subperiod_table(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# compute_rolling_stats
# ---------------------------------------------------------------------------

class TestComputeRollingStats:
    def test_returns_dataframe(self, multi_year_df):
        result = compute_rolling_stats(multi_year_df)
        assert isinstance(result, pd.DataFrame)

    def test_length_matches_input(self, multi_year_df):
        result = compute_rolling_stats(multi_year_df)
        assert len(result) == len(multi_year_df)

    def test_expected_columns(self, multi_year_df):
        result = compute_rolling_stats(multi_year_df)
        for col in ["strat_sharpe", "sp500_sharpe", "beta", "ann_alpha"]:
            assert col in result.columns

    def test_first_rows_are_nan(self, multi_year_df):
        result = compute_rolling_stats(multi_year_df, window=_ROLL_WINDOW)
        assert result["strat_sharpe"].iloc[: _ROLL_WINDOW - 1].isna().all()

    def test_later_rows_not_all_nan(self, multi_year_df):
        result = compute_rolling_stats(multi_year_df)
        assert result["strat_sharpe"].iloc[_ROLL_WINDOW:].notna().any()

    def test_index_is_datetime(self, multi_year_df):
        result = compute_rolling_stats(multi_year_df)
        assert pd.api.types.is_datetime64_any_dtype(result.index)

    def test_rolling_beta_near_known_value(self, known_df):
        result = compute_rolling_stats(known_df, window=_ROLL_WINDOW)
        betas = result["beta"].dropna()
        if len(betas) > 0:
            assert (betas - 0.5).abs().mean() < 0.15

    def test_empty_df_returns_empty(self):
        assert compute_rolling_stats(pd.DataFrame()).empty

    def test_custom_window(self, multi_year_df):
        result = compute_rolling_stats(multi_year_df, window=30)
        assert result["strat_sharpe"].iloc[:29].isna().all()
        assert result["strat_sharpe"].iloc[30:].notna().any()


# ---------------------------------------------------------------------------
# compute_regime_frequency
# ---------------------------------------------------------------------------

class TestComputeRegimeFrequency:
    def test_returns_dataframe(self, multi_year_df):
        result = compute_regime_frequency(multi_year_df)
        assert isinstance(result, pd.DataFrame)

    def test_rows_sum_to_one(self, multi_year_df):
        result = compute_regime_frequency(multi_year_df)
        for label, row in result.iterrows():
            assert abs(row.sum() - 1.0) < 1e-6, f"Row {label} sums to {row.sum()}"

    def test_all_period_labels_present(self, multi_year_df):
        periods = define_periods(multi_year_df)
        result  = compute_regime_frequency(multi_year_df, periods)
        for p in periods:
            assert p["label"] in result.index

    def test_values_between_0_and_1(self, multi_year_df):
        result = compute_regime_frequency(multi_year_df)
        assert (result >= 0).all().all()
        assert (result <= 1).all().all()

    def test_all_regimes_present_as_columns(self, multi_year_df):
        result = compute_regime_frequency(multi_year_df)
        for r in multi_year_df[_LABEL_COL].unique():
            assert r in result.columns

    def test_empty_df_returns_empty(self):
        assert compute_regime_frequency(pd.DataFrame()).empty

    def test_missing_label_col_returns_empty(self, multi_year_df):
        df2 = multi_year_df.drop(columns=[_LABEL_COL])
        assert compute_regime_frequency(df2).empty


# ---------------------------------------------------------------------------
# run_subperiod_attribution
# ---------------------------------------------------------------------------

class TestRunSubperiodAttribution:
    def test_returns_dict(self, multi_year_df):
        result = run_subperiod_attribution(multi_year_df)
        assert isinstance(result, dict)

    def test_all_keys_present(self, multi_year_df):
        result = run_subperiod_attribution(multi_year_df)
        for k in ["periods", "table", "rolling", "regime_freq", "roll_window"]:
            assert k in result

    def test_table_is_dataframe(self, multi_year_df):
        result = run_subperiod_attribution(multi_year_df)
        assert isinstance(result["table"], pd.DataFrame)

    def test_rolling_is_dataframe(self, multi_year_df):
        result = run_subperiod_attribution(multi_year_df)
        assert isinstance(result["rolling"], pd.DataFrame)

    def test_regime_freq_is_dataframe(self, multi_year_df):
        result = run_subperiod_attribution(multi_year_df)
        assert isinstance(result["regime_freq"], pd.DataFrame)

    def test_roll_window_matches_constant(self, multi_year_df):
        result = run_subperiod_attribution(multi_year_df)
        assert result["roll_window"] == _ROLL_WINDOW

    def test_empty_df_returns_empty_structures(self):
        result = run_subperiod_attribution(pd.DataFrame())
        assert result["periods"] == []
        assert result["table"].empty
        assert result["rolling"].empty
        assert result["regime_freq"].empty

    def test_table_has_full_period_row(self, multi_year_df):
        result = run_subperiod_attribution(multi_year_df)
        assert "Full Period" in result["table"].index

    def test_table_and_regime_freq_share_index(self, multi_year_df):
        result = run_subperiod_attribution(multi_year_df)
        for lbl in result["table"].index:
            assert lbl in result["regime_freq"].index
