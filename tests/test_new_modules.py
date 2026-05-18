"""
Tests for 18 recently-added analytical modules.

Each class covers one module with 3-4 tests:
  - happy-path (available=True, required columns present)
  - graceful failure (empty df or missing key columns)
  - current dict key presence
  - output shapes / column existence

Network-dependent modules (global_credit, em_credit, vix_term_structure,
options_skew) only assert isinstance(result["available"], bool) — no network
calls are made in non-skipped tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_df():
    """Realistic DataFrame with a DatetimeIndex spanning ~6 years (1500 trading days).

    Includes all columns that the new modules may consume.
    """
    n = 1500  # ~6 years of trading days
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "sp500":         4000 * np.exp(rng.normal(0.0003, 0.012, n).cumsum()),
            "vix":           np.clip(18 + rng.normal(0, 4, n), 8, 80),
            "hy_spread":     350 + rng.normal(0, 4, n).cumsum(),
            "ig_spread":     120 + rng.normal(0, 2, n).cumsum(),
            "hy_yield":      6.5 + rng.normal(0, 0.3, n).cumsum() * 0.05,
            "ig_yield":      4.0 + rng.normal(0, 0.2, n).cumsum() * 0.05,
            "yield_10y":     3.5 + rng.normal(0, 0.1, n).cumsum() * 0.03,
            "yield_2y":      3.0 + rng.normal(0, 0.1, n).cumsum() * 0.03,
            "yield_3m":      2.0 + rng.normal(0, 0.1, n).cumsum() * 0.02,
            "breakeven_10y": 2.2 + rng.normal(0, 0.05, n).cumsum() * 0.02,
            "dff":           2.1 + rng.normal(0, 0.05, n).cumsum() * 0.02,
            "unemployment":  4.0 + rng.normal(0, 0.1, n).cumsum() * 0.01,
            "dxy":           100 + rng.normal(0, 0.5, n).cumsum() * 0.05,
            "composite_risk_score_smooth": 50 + rng.normal(0, 8, n),
            "final_decision": rng.choice(
                ["Risk-On", "Neutral", "Caution", "Risk-Off"], n
            ),
            "sp500_forward_30d_return":  rng.normal(0.01, 0.04, n),
            "sp500_forward_60d_return":  rng.normal(0.02, 0.06, n),
            "sp500_forward_126d_return": rng.normal(0.04, 0.08, n),
            "sp500_forward_252d_return": rng.normal(0.08, 0.12, n),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# 1. VRP
# ---------------------------------------------------------------------------

class TestVRP:
    def test_available_with_required_columns(self, synthetic_df):
        from src.vrp import run_vrp_analysis
        result = run_vrp_analysis(synthetic_df)
        assert result["available"]

    def test_unavailable_with_empty_df(self):
        from src.vrp import run_vrp_analysis
        result = run_vrp_analysis(pd.DataFrame())
        assert result["available"] is False

    def test_unavailable_without_required_columns(self):
        from src.vrp import run_vrp_analysis
        df = pd.DataFrame({"yield_10y": [3.5, 3.6]}, index=pd.date_range("2023-01-01", periods=2, freq="B"))
        result = run_vrp_analysis(df)
        assert result["available"] is False

    def test_current_dict_keys(self, synthetic_df):
        from src.vrp import run_vrp_analysis
        result = run_vrp_analysis(synthetic_df)
        assert "current" in result
        current = result["current"]
        for key in ("vrp_21d", "vrp_regime", "vix", "realized_vol_21d"):
            assert key in current, f"Missing key in current: {key}"

    def test_output_shapes(self, synthetic_df):
        from src.vrp import run_vrp_analysis
        result = run_vrp_analysis(synthetic_df)
        assert "historical" in result
        hist = result["historical"]
        assert isinstance(hist, pd.DataFrame)
        assert len(hist) > 0
        assert "vrp_21d" in hist.columns


# ---------------------------------------------------------------------------
# 2. Credit Momentum
# ---------------------------------------------------------------------------

class TestCreditMomentum:
    def test_available_with_required_columns(self, synthetic_df):
        from src.credit_momentum import run_credit_momentum_analysis
        result = run_credit_momentum_analysis(synthetic_df)
        assert result["available"]

    def test_unavailable_with_empty_df(self):
        from src.credit_momentum import run_credit_momentum_analysis
        result = run_credit_momentum_analysis(pd.DataFrame())
        assert result["available"] is False

    def test_unavailable_without_spread_columns(self):
        from src.credit_momentum import run_credit_momentum_analysis
        df = pd.DataFrame({"vix": [18.0, 19.0]}, index=pd.date_range("2023-01-01", periods=2, freq="B"))
        result = run_credit_momentum_analysis(df)
        assert result["available"] is False

    def test_current_dict_keys(self, synthetic_df):
        from src.credit_momentum import run_credit_momentum_analysis
        result = run_credit_momentum_analysis(synthetic_df)
        current = result["current"]
        for key in ("hy_mom_63d", "hy_mom_regime"):
            assert key in current, f"Missing key: {key}"

    def test_trend_is_string(self, synthetic_df):
        from src.credit_momentum import run_credit_momentum_analysis
        result = run_credit_momentum_analysis(synthetic_df)
        assert "trend" in result
        assert result["trend"] in ("Improving", "Deteriorating", "Flat")


# ---------------------------------------------------------------------------
# 3. Funding Stress
# ---------------------------------------------------------------------------

class TestFundingStress:
    def test_available_with_required_columns(self, synthetic_df):
        from src.funding_stress import run_funding_stress_analysis
        result = run_funding_stress_analysis(synthetic_df)
        assert result["available"]

    def test_unavailable_with_empty_df(self):
        from src.funding_stress import run_funding_stress_analysis
        result = run_funding_stress_analysis(pd.DataFrame())
        assert result["available"] is False

    def test_unavailable_without_ff_columns(self):
        from src.funding_stress import run_funding_stress_analysis
        df = pd.DataFrame(
            {"hy_spread": [350.0, 360.0]},
            index=pd.date_range("2023-01-01", periods=2, freq="B"),
        )
        result = run_funding_stress_analysis(df)
        assert result["available"] is False

    def test_current_dict_has_funding_regime(self, synthetic_df):
        from src.funding_stress import run_funding_stress_analysis
        result = run_funding_stress_analysis(synthetic_df)
        assert "current" in result
        assert "funding_regime" in result["current"]

    def test_funding_regime_is_valid_string(self, synthetic_df):
        from src.funding_stress import run_funding_stress_analysis
        result = run_funding_stress_analysis(synthetic_df)
        valid = {"Benign", "Normal", "Elevated", "Stressed", "Crisis", "Unknown"}
        assert result["current"]["funding_regime"] in valid


# ---------------------------------------------------------------------------
# 4. Global Credit (live data — network-graceful)
# ---------------------------------------------------------------------------

class TestGlobalCredit:
    def test_returns_dict_with_available_key(self, synthetic_df):
        from src.global_credit import run_global_credit_analysis
        result = run_global_credit_analysis(synthetic_df)
        assert isinstance(result, dict)
        assert "available" in result

    def test_available_is_bool(self, synthetic_df):
        from src.global_credit import run_global_credit_analysis
        result = run_global_credit_analysis(synthetic_df)
        assert isinstance(result["available"], bool)

    def test_empty_df_returns_dict(self):
        from src.global_credit import run_global_credit_analysis
        result = run_global_credit_analysis(pd.DataFrame())
        assert isinstance(result, dict)
        assert "available" in result


# ---------------------------------------------------------------------------
# 5. Corporate Leverage
# ---------------------------------------------------------------------------

class TestCorporateLeverage:
    def test_always_available(self, synthetic_df):
        from src.corporate_leverage import run_corporate_leverage_analysis
        result = run_corporate_leverage_analysis(synthetic_df)
        # always True per module docstring (synthetic proxy fallback)
        assert result["available"]

    def test_empty_df_graceful(self):
        from src.corporate_leverage import run_corporate_leverage_analysis
        result = run_corporate_leverage_analysis(pd.DataFrame())
        assert isinstance(result, dict)
        assert "available" in result

    def test_cycle_phase_is_string(self, synthetic_df):
        from src.corporate_leverage import run_corporate_leverage_analysis
        result = run_corporate_leverage_analysis(synthetic_df)
        assert "cycle_phase" in result
        assert isinstance(result["cycle_phase"], str)

    def test_current_dict_keys(self, synthetic_df):
        from src.corporate_leverage import run_corporate_leverage_analysis
        result = run_corporate_leverage_analysis(synthetic_df)
        assert "current" in result
        for key in ("leverage_stress_signal", "synthetic_leverage_proxy", "using_synthetic"):
            assert key in result["current"], f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 6. Seasonality
# ---------------------------------------------------------------------------

class TestSeasonality:
    def test_available_with_5yr_data(self, synthetic_df):
        from src.seasonality import run_seasonality_analysis
        result = run_seasonality_analysis(synthetic_df)
        assert result["available"]

    def test_unavailable_with_short_data(self):
        from src.seasonality import run_seasonality_analysis
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        df = pd.DataFrame({"hy_spread": 350 + np.random.normal(0, 4, 100)}, index=dates)
        result = run_seasonality_analysis(df)
        assert result["available"] is False

    def test_unavailable_with_empty_df(self):
        from src.seasonality import run_seasonality_analysis
        result = run_seasonality_analysis(pd.DataFrame())
        assert result["available"] is False

    def test_available_result_has_expected_keys(self, synthetic_df):
        from src.seasonality import run_seasonality_analysis
        result = run_seasonality_analysis(synthetic_df)
        if result["available"]:
            assert "current" in result or "interpretation" in result or "hy_avg" in result


# ---------------------------------------------------------------------------
# 7. VIX Term Structure (live data — network-graceful)
# ---------------------------------------------------------------------------

class TestVIXTermStructure:
    def test_returns_dict_with_available_key(self, synthetic_df):
        from src.vix_term_structure import run_vix_term_analysis
        result = run_vix_term_analysis(synthetic_df)
        assert isinstance(result, dict)
        assert "available" in result

    def test_available_is_bool(self, synthetic_df):
        from src.vix_term_structure import run_vix_term_analysis
        result = run_vix_term_analysis(synthetic_df)
        assert isinstance(result["available"], bool)

    def test_empty_df_returns_dict(self):
        from src.vix_term_structure import run_vix_term_analysis
        result = run_vix_term_analysis(pd.DataFrame())
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 8. Options Skew (live data — network-graceful)
# ---------------------------------------------------------------------------

class TestOptionsSkew:
    def test_returns_dict_with_available_key(self, synthetic_df):
        from src.options_skew import run_skew_analysis
        result = run_skew_analysis(synthetic_df)
        assert isinstance(result, dict)
        assert "available" in result

    def test_available_is_bool(self, synthetic_df):
        from src.options_skew import run_skew_analysis
        result = run_skew_analysis(synthetic_df)
        assert isinstance(result["available"], bool)

    def test_empty_df_returns_dict(self):
        from src.options_skew import run_skew_analysis
        result = run_skew_analysis(pd.DataFrame())
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 9. Regime Return Table
# ---------------------------------------------------------------------------

class TestRegimeReturnTable:
    def test_available_with_required_columns(self, synthetic_df):
        from src.regime_return_table import run_regime_return_analysis
        result = run_regime_return_analysis(synthetic_df)
        assert result["available"]

    def test_unavailable_with_empty_df(self):
        from src.regime_return_table import run_regime_return_analysis
        result = run_regime_return_analysis(pd.DataFrame())
        assert result["available"] is False

    def test_unavailable_without_regime_column(self):
        from src.regime_return_table import run_regime_return_analysis
        df = pd.DataFrame(
            {"sp500": [4000.0, 4010.0]},
            index=pd.date_range("2023-01-01", periods=2, freq="B"),
        )
        result = run_regime_return_analysis(df)
        assert result["available"] is False

    def test_result_has_table_key(self, synthetic_df):
        from src.regime_return_table import run_regime_return_analysis
        result = run_regime_return_analysis(synthetic_df)
        # "table" is always returned (even when available=False)
        assert "table" in result


# ---------------------------------------------------------------------------
# 10. Default Cycle
# ---------------------------------------------------------------------------

class TestDefaultCycle:
    def test_available_with_required_columns(self, synthetic_df):
        from src.default_cycle import run_default_cycle_analysis
        result = run_default_cycle_analysis(synthetic_df)
        assert result["available"]

    def test_unavailable_with_empty_df(self):
        from src.default_cycle import run_default_cycle_analysis
        result = run_default_cycle_analysis(pd.DataFrame())
        assert result["available"] is False

    def test_unavailable_without_spread_columns(self):
        from src.default_cycle import run_default_cycle_analysis
        df = pd.DataFrame(
            {"vix": [18.0, 19.0]},
            index=pd.date_range("2023-01-01", periods=2, freq="B"),
        )
        result = run_default_cycle_analysis(df)
        assert result["available"] is False

    def test_current_dict_is_dict(self, synthetic_df):
        from src.default_cycle import run_default_cycle_analysis
        result = run_default_cycle_analysis(synthetic_df)
        if result["available"]:
            assert "current" in result
            assert isinstance(result["current"], dict)


# ---------------------------------------------------------------------------
# 11. Carry Breakeven
# ---------------------------------------------------------------------------

class TestCarryBreakeven:
    def test_available_with_required_columns(self, synthetic_df):
        from src.carry_breakeven import run_breakeven_analysis
        result = run_breakeven_analysis(synthetic_df)
        assert result["available"]

    def test_unavailable_with_empty_df(self):
        from src.carry_breakeven import run_breakeven_analysis
        result = run_breakeven_analysis(pd.DataFrame())
        assert result["available"] is False

    def test_current_has_breakeven_keys(self, synthetic_df):
        from src.carry_breakeven import run_breakeven_analysis
        result = run_breakeven_analysis(synthetic_df)
        if result["available"]:
            assert "current" in result
            current = result["current"]
            # At minimum one of the carry columns should be present
            any_carry = any(k.startswith("carry") or k.startswith("breakeven") for k in current)
            assert any_carry

    def test_historical_is_dataframe(self, synthetic_df):
        from src.carry_breakeven import run_breakeven_analysis
        result = run_breakeven_analysis(synthetic_df)
        if result["available"]:
            assert "historical_breakeven" in result
            assert isinstance(result["historical_breakeven"], pd.DataFrame)


# ---------------------------------------------------------------------------
# 12. Comparison Mode
# ---------------------------------------------------------------------------

class TestComparisonMode:
    def test_get_available_dates_returns_list(self, synthetic_df):
        from src.comparison_mode import get_available_dates
        dates = get_available_dates(synthetic_df)
        assert isinstance(dates, list)

    def test_get_available_dates_sorted(self, synthetic_df):
        from src.comparison_mode import get_available_dates
        dates = get_available_dates(synthetic_df)
        if len(dates) >= 2:
            assert dates == sorted(dates)

    def test_get_available_dates_empty_df(self):
        from src.comparison_mode import get_available_dates
        result = get_available_dates(pd.DataFrame())
        assert result == []

    def test_compare_dates_returns_dict(self, synthetic_df):
        from src.comparison_mode import get_available_dates, compare_dates
        dates = get_available_dates(synthetic_df)
        if len(dates) >= 2:
            result = compare_dates(synthetic_df, dates[0], dates[-1])
            assert isinstance(result, dict)
            assert "available" in result

    def test_compare_dates_invalid_dates_graceful(self, synthetic_df):
        from src.comparison_mode import compare_dates
        result = compare_dates(synthetic_df, "1900-01-01", "1901-01-01")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 13. Real Rates
# ---------------------------------------------------------------------------

class TestRealRates:
    def test_available_with_required_columns(self, synthetic_df):
        from src.real_rates import run_real_rates_analysis
        result = run_real_rates_analysis(synthetic_df)
        assert result["available"]

    def test_unavailable_with_empty_df(self):
        from src.real_rates import run_real_rates_analysis
        result = run_real_rates_analysis(pd.DataFrame())
        assert result["available"] is False

    def test_unavailable_without_yield_columns(self):
        from src.real_rates import run_real_rates_analysis
        df = pd.DataFrame(
            {"vix": [18.0, 19.0]},
            index=pd.date_range("2023-01-01", periods=2, freq="B"),
        )
        result = run_real_rates_analysis(df)
        assert result["available"] is False

    def test_current_has_real_rate_10y(self, synthetic_df):
        from src.real_rates import run_real_rates_analysis
        result = run_real_rates_analysis(synthetic_df)
        assert "current" in result
        assert "real_rate_10y" in result["current"]

    def test_historical_contains_real_rate_column(self, synthetic_df):
        from src.real_rates import run_real_rates_analysis
        result = run_real_rates_analysis(synthetic_df)
        if result["available"]:
            hist = result["historical"]
            assert isinstance(hist, pd.DataFrame)
            assert "real_rate_10y" in hist.columns


# ---------------------------------------------------------------------------
# 14. Correlation Heatmap
# ---------------------------------------------------------------------------

class TestCorrelationHeatmap:
    def test_available_with_required_columns(self, synthetic_df):
        from src.correlation_heatmap import run_correlation_heatmap_analysis
        result = run_correlation_heatmap_analysis(synthetic_df)
        assert result["available"]

    def test_unavailable_with_empty_df(self):
        from src.correlation_heatmap import run_correlation_heatmap_analysis
        result = run_correlation_heatmap_analysis(pd.DataFrame())
        assert result["available"] is False

    def test_matrix_current_is_dataframe(self, synthetic_df):
        from src.correlation_heatmap import run_correlation_heatmap_analysis
        result = run_correlation_heatmap_analysis(synthetic_df)
        if result["available"]:
            assert "matrix_current" in result
            assert isinstance(result["matrix_current"], pd.DataFrame)

    def test_matrix_values_in_valid_range(self, synthetic_df):
        from src.correlation_heatmap import run_correlation_heatmap_analysis
        result = run_correlation_heatmap_analysis(synthetic_df)
        if result["available"]:
            mat = result["matrix_current"].values
            valid = mat[~np.isnan(mat)]
            assert np.all(valid >= -1.0 - 1e-6)
            assert np.all(valid <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# 15. Spread Volatility
# ---------------------------------------------------------------------------

class TestSpreadVolatility:
    def test_available_with_required_columns(self, synthetic_df):
        from src.spread_volatility import run_spread_volatility_analysis
        result = run_spread_volatility_analysis(synthetic_df)
        assert result["available"]

    def test_unavailable_with_empty_df(self):
        from src.spread_volatility import run_spread_volatility_analysis
        result = run_spread_volatility_analysis(pd.DataFrame())
        assert result["available"] is False

    def test_current_has_hy_vol_regime(self, synthetic_df):
        from src.spread_volatility import run_spread_volatility_analysis
        result = run_spread_volatility_analysis(synthetic_df)
        assert "current" in result
        assert "hy_vol_regime" in result["current"]

    def test_hy_vol_regime_is_valid(self, synthetic_df):
        from src.spread_volatility import run_spread_volatility_analysis
        result = run_spread_volatility_analysis(synthetic_df)
        valid = {"Low", "Normal", "Elevated", "Stress", "Unknown"}
        assert result["current"]["hy_vol_regime"] in valid


# ---------------------------------------------------------------------------
# 16. Fallen Angel
# ---------------------------------------------------------------------------

class TestFallenAngel:
    def test_available_with_required_columns(self, synthetic_df):
        from src.fallen_angel import run_fallen_angel_analysis
        result = run_fallen_angel_analysis(synthetic_df)
        assert result["available"]

    def test_unavailable_with_empty_df(self):
        from src.fallen_angel import run_fallen_angel_analysis
        result = run_fallen_angel_analysis(pd.DataFrame())
        assert result["available"] is False

    def test_unavailable_without_spread_columns(self):
        from src.fallen_angel import run_fallen_angel_analysis
        df = pd.DataFrame(
            {"vix": [18.0, 19.0]},
            index=pd.date_range("2023-01-01", periods=2, freq="B"),
        )
        result = run_fallen_angel_analysis(df)
        assert result["available"] is False

    def test_current_has_hy_ig_ratio(self, synthetic_df):
        from src.fallen_angel import run_fallen_angel_analysis
        result = run_fallen_angel_analysis(synthetic_df)
        assert "current" in result
        assert "hy_ig_ratio" in result["current"]

    def test_hy_ig_ratio_is_positive(self, synthetic_df):
        from src.fallen_angel import run_fallen_angel_analysis
        result = run_fallen_angel_analysis(synthetic_df)
        ratio = result["current"]["hy_ig_ratio"]
        if ratio is not None and not (isinstance(ratio, float) and np.isnan(ratio)):
            assert ratio > 0


# ---------------------------------------------------------------------------
# 17. EM Credit (live data — network-graceful)
# ---------------------------------------------------------------------------

class TestEMCredit:
    def test_returns_dict_with_available_key(self, synthetic_df):
        from src.em_credit import run_em_credit_analysis
        result = run_em_credit_analysis(synthetic_df)
        assert isinstance(result, dict)
        assert "available" in result

    def test_available_is_bool(self, synthetic_df):
        from src.em_credit import run_em_credit_analysis
        result = run_em_credit_analysis(synthetic_df)
        assert isinstance(result["available"], bool)

    def test_empty_df_returns_dict(self):
        from src.em_credit import run_em_credit_analysis
        result = run_em_credit_analysis(pd.DataFrame())
        assert isinstance(result, dict)
        assert "available" in result


# ---------------------------------------------------------------------------
# 18. Macro Nowcast
# ---------------------------------------------------------------------------

class TestMacroNowcast:
    def test_available_with_required_columns(self, synthetic_df):
        from src.macro_nowcast import run_macro_nowcast
        result = run_macro_nowcast(synthetic_df)
        assert result["available"]

    def test_unavailable_with_empty_df(self):
        from src.macro_nowcast import run_macro_nowcast
        result = run_macro_nowcast(pd.DataFrame())
        # Empty df has no indicators → available=False (need ≥2 indicators)
        assert result["available"] is False

    def test_unavailable_with_single_indicator(self):
        from src.macro_nowcast import run_macro_nowcast
        df = pd.DataFrame(
            {"vix": [18.0, 19.0]},
            index=pd.date_range("2023-01-01", periods=2, freq="B"),
        )
        result = run_macro_nowcast(df)
        # vix is not in NOWCAST_INDICATORS → 0 indicators → not available
        assert result["available"] is False

    def test_current_has_nowcast_score(self, synthetic_df):
        from src.macro_nowcast import run_macro_nowcast
        result = run_macro_nowcast(synthetic_df)
        assert "current" in result
        assert "nowcast_score" in result["current"]

    def test_nowcast_score_in_valid_range(self, synthetic_df):
        from src.macro_nowcast import run_macro_nowcast
        result = run_macro_nowcast(synthetic_df)
        score = result["current"]["nowcast_score"]
        if not (isinstance(score, float) and np.isnan(score)):
            assert 0.0 <= score <= 100.0

    def test_historical_is_dataframe(self, synthetic_df):
        from src.macro_nowcast import run_macro_nowcast
        result = run_macro_nowcast(synthetic_df)
        if result["available"]:
            assert "historical" in result
            assert isinstance(result["historical"], pd.DataFrame)
            assert "nowcast_score" in result["historical"].columns
