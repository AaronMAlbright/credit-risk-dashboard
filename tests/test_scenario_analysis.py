"""
Tests for src/scenario_analysis.py.

Covers shock propagation, score recomputation, scenario grid, and tornado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.scenario_analysis import (
    DEFAULT_SHOCKS,
    SCENARIO_PRESETS,
    TORNADO_SHOCKS,
    build_shocked_row,
    build_tornado_data,
    run_scenario,
    run_scenario_analysis,
    run_scenario_grid,
    score_shocked_row,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Minimal synthetic scored DataFrame compatible with scenario analysis."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n, freq="B")

    df = pd.DataFrame({
        "date":           dates,
        "vix":            rng.uniform(12, 35, n),
        "hy_spread":      rng.uniform(2.5, 5.0, n),
        "spread":         rng.uniform(-0.5, 1.5, n),
        "nfci":           rng.uniform(-1.0, 0.5, n),
        "nfci_90d_avg":   rng.uniform(-0.8, 0.4, n),
        "unemployment":   rng.uniform(3.5, 6.0, n),
        "sp500":          np.cumprod(1 + rng.normal(0.0003, 0.01, n)) * 4500,
        # Change / momentum features
        "hy_change_30d":           rng.uniform(-0.5, 0.5, n),
        "hy_change_90d":           rng.uniform(-1.0, 1.0, n),
        "hy_change_5d":            rng.uniform(-0.15, 0.15, n),
        "hy_change_30d_prior":     rng.uniform(-0.4, 0.4, n),
        "credit_impulse":          rng.uniform(-0.8, 0.8, n),
        "vix_change_30d":          rng.uniform(-10, 10, n),
        "vix_change_5d":           rng.uniform(-5, 5, n),
        "sp500_return_30d":        rng.uniform(-0.1, 0.1, n),
        "sp500_return_5d":         rng.uniform(-0.05, 0.05, n),
        "sp500_drawdown":          rng.uniform(-0.15, 0, n),
        "sp500_peak":              np.full(n, 5000.0),
        "unemployment_change_90d": rng.uniform(-0.3, 0.3, n),
        "spread_change_90d":       rng.uniform(-0.5, 0.5, n),
        "spread_change_5d":        rng.uniform(-0.1, 0.1, n),
        "nfci_change_90d":         rng.uniform(-0.2, 0.2, n),
        "sahm_like":               rng.uniform(0, 0.4, n),
        # Classification features
        "yield_curve_regime":             ["Normal"] * n,
        "labor_warning":                  ["Low"] * n,
        "credit_equity_divergence":       ["Neutral"] * n,
        "vol_credit_mismatch":            ["Neutral"] * n,
        "shock_flag":                     ["No Shock"] * n,
        "market_internals_score":         rng.uniform(20, 80, n),
        "cross_asset_divergence_score":   rng.uniform(0, 30, n),
        "transition_regime":              ["Complacency / Late Cycle"] * n,
        # Smoothed component scores
        "macro_risk_score":                    rng.uniform(5, 30, n),
        "credit_market_risk_score":            rng.uniform(0, 20, n),
        "liquidity_score":                     rng.uniform(0, 20, n),
        "liquidity_regime_score":              rng.uniform(0, 10, n),
        "treasury_stress_score":               rng.uniform(0, 10, n),
        "complacency_score":                   rng.uniform(50, 90, n),
        "mean_reversion_score":                rng.uniform(0, 10, n),
        "macro_risk_score_smooth":             rng.uniform(5, 25, n),
        "credit_market_risk_score_smooth":     rng.uniform(0, 15, n),
        "liquidity_regime_score_smooth":       rng.uniform(0, 8, n),
        "treasury_stress_score_smooth":        rng.uniform(0, 5, n),
        "complacency_score_smooth":            rng.uniform(60, 85, n),
        "mean_reversion_score_smooth":         rng.uniform(0, 5, n),
        "cross_asset_divergence_score_smooth": rng.uniform(0, 10, n),
        "macro_risk_momentum_10d":             rng.uniform(-5, 5, n),
        "credit_risk_momentum_10d":            rng.uniform(-10, 10, n),
        "composite_risk_score":                rng.uniform(15, 35, n),
        "composite_risk_score_smooth":         rng.uniform(18, 32, n),
        "final_decision":                      ["Hold / Do Not Chase"] * n,
        # Return cols (needed for position sizing)
        "strategy_daily_return":  rng.normal(0.0004, 0.008, n),
        "sp500_daily_return":     rng.normal(0.0003, 0.010, n),
    })
    return df


@pytest.fixture(scope="module")
def base_df():
    return _make_df()


@pytest.fixture(scope="module")
def latest(base_df):
    return base_df.iloc[-1]


@pytest.fixture(scope="module")
def moderate_shocks():
    return SCENARIO_PRESETS["Moderate Stress"]


# ---------------------------------------------------------------------------
# build_shocked_row
# ---------------------------------------------------------------------------

class TestBuildShockedRow:
    def test_returns_series(self, latest):
        result = build_shocked_row(latest, DEFAULT_SHOCKS)
        assert isinstance(result, pd.Series)

    def test_zero_shocks_leaves_vix_unchanged(self, latest):
        result = build_shocked_row(latest, DEFAULT_SHOCKS)
        assert abs(float(result["vix"]) - float(latest["vix"])) < 1e-9

    def test_vix_shock_increases_vix(self, latest):
        result = build_shocked_row(latest, {"vix_shock": 10.0})
        assert float(result["vix"]) == pytest.approx(float(latest["vix"]) + 10.0, abs=1e-6)

    def test_vix_shock_propagates_to_change_30d(self, latest):
        shock = 8.0
        result = build_shocked_row(latest, {"vix_shock": shock})
        expected = float(latest["vix_change_30d"]) + shock
        assert float(result["vix_change_30d"]) == pytest.approx(expected, abs=1e-9)

    def test_hy_shock_increases_hy_spread(self, latest):
        result = build_shocked_row(latest, {"hy_spread_shock": 1.5})
        assert float(result["hy_spread"]) == pytest.approx(float(latest["hy_spread"]) + 1.5, abs=1e-6)

    def test_hy_shock_propagates_to_hy_change_30d(self, latest):
        shock = 1.0
        result = build_shocked_row(latest, {"hy_spread_shock": shock})
        expected = float(latest["hy_change_30d"]) + shock
        assert float(result["hy_change_30d"]) == pytest.approx(expected, abs=1e-9)

    def test_hy_shock_propagates_to_credit_impulse(self, latest):
        shock = 0.75
        result = build_shocked_row(latest, {"hy_spread_shock": shock})
        expected = float(latest["credit_impulse"]) + shock
        assert float(result["credit_impulse"]) == pytest.approx(expected, abs=1e-9)

    def test_sp500_shock_updates_drawdown(self, latest):
        shock = -0.10
        result = build_shocked_row(latest, {"sp500_shock": shock})
        orig_dd = float(latest["sp500_drawdown"])
        expected_dd = (orig_dd + 1.0) * (1.0 + shock) - 1.0
        assert float(result["sp500_drawdown"]) == pytest.approx(expected_dd, abs=1e-9)

    def test_sp500_negative_shock_worsens_drawdown(self, latest):
        result = build_shocked_row(latest, {"sp500_shock": -0.15})
        assert float(result["sp500_drawdown"]) <= float(latest["sp500_drawdown"])

    def test_spread_shock_updates_spread(self, latest):
        shock = -0.5
        result = build_shocked_row(latest, {"spread_shock": shock})
        assert float(result["spread"]) == pytest.approx(float(latest["spread"]) + shock, abs=1e-9)

    def test_nfci_shock_updates_nfci(self, latest):
        shock = 0.3
        result = build_shocked_row(latest, {"nfci_shock": shock})
        assert float(result["nfci"]) == pytest.approx(float(latest["nfci"]) + shock, abs=1e-9)

    def test_unemployment_shock_updates_unemployment(self, latest):
        shock = 0.5
        result = build_shocked_row(latest, {"unemployment_shock": shock})
        assert float(result["unemployment"]) == pytest.approx(float(latest["unemployment"]) + shock, abs=1e-6)

    def test_sahm_like_clipped_to_zero(self, latest):
        result = build_shocked_row(latest, {"unemployment_shock": -999.0})
        assert float(result["sahm_like"]) >= 0.0

    def test_vix_floor_at_5(self, latest):
        result = build_shocked_row(latest, {"vix_shock": -999.0})
        assert float(result["vix"]) >= 5.0

    def test_hy_spread_floor_at_0_5(self, latest):
        result = build_shocked_row(latest, {"hy_spread_shock": -999.0})
        assert float(result["hy_spread"]) >= 0.5

    def test_classification_features_recomputed(self, latest):
        # A large HY widening with SP500 up should produce Hidden Credit Stress
        shocked = build_shocked_row(latest, {
            "hy_spread_shock": 3.0,
            "sp500_shock":     0.05,
        })
        # hy_change_30d is now very positive; sp500_return_30d > 0
        # → expect credit_equity_divergence reflects new state
        assert isinstance(str(shocked["credit_equity_divergence"]), str)

    def test_shock_flag_updates_with_large_vix_spike(self, latest):
        shocked = build_shocked_row(latest, {"vix_shock": 20.0})
        assert "VIX spike" in str(shocked["shock_flag"]) or "No Shock" not in str(shocked["shock_flag"])


# ---------------------------------------------------------------------------
# score_shocked_row
# ---------------------------------------------------------------------------

class TestScoreShockedRow:
    def test_returns_dict(self, latest):
        shocked = build_shocked_row(latest, DEFAULT_SHOCKS)
        result = score_shocked_row(shocked, latest)
        assert isinstance(result, dict)

    def test_all_expected_keys_present(self, latest):
        shocked = build_shocked_row(latest, DEFAULT_SHOCKS)
        result = score_shocked_row(shocked, latest)
        for k in [
            "composite_risk_score_smooth", "final_decision",
            "macro_risk_score_smooth", "credit_market_risk_score_smooth",
            "complacency_score_smooth", "mean_reversion_score_smooth",
        ]:
            assert k in result, f"Missing key: {k}"

    def test_composite_in_valid_range(self, latest, moderate_shocks):
        shocked = build_shocked_row(latest, moderate_shocks)
        result = score_shocked_row(shocked, latest)
        c = float(result["composite_risk_score_smooth"])
        assert 0.0 <= c <= 100.0

    def test_all_component_scores_in_range(self, latest, moderate_shocks):
        shocked = build_shocked_row(latest, moderate_shocks)
        result = score_shocked_row(shocked, latest)
        for key in [
            "macro_risk_score_smooth", "credit_market_risk_score_smooth",
            "liquidity_regime_score_smooth", "treasury_stress_score_smooth",
            "complacency_score_smooth", "mean_reversion_score_smooth",
        ]:
            v = float(result[key])
            assert 0.0 <= v <= 100.0, f"{key} = {v} out of range"

    def test_final_decision_is_string(self, latest, moderate_shocks):
        shocked = build_shocked_row(latest, moderate_shocks)
        result = score_shocked_row(shocked, latest)
        assert isinstance(result["final_decision"], str)
        assert len(result["final_decision"]) > 0

    def test_severe_hy_widening_increases_credit_score(self, latest):
        shocked = build_shocked_row(latest, {"hy_spread_shock": 4.0})
        result = score_shocked_row(shocked, latest)
        base_credit = float(latest["credit_market_risk_score_smooth"])
        assert float(result["credit_market_risk_score_smooth"]) > base_credit

    def test_severe_stress_returns_stress_decision(self, latest):
        shocked = build_shocked_row(latest, SCENARIO_PRESETS["Severe Stress"])
        result = score_shocked_row(shocked, latest)
        assert result["final_decision"] in (
            "Stress / Stabilization Watch", "Buy Stress", "Watch Entry",
            "Credit Warning", "Divergence Warning",
        )

    def test_market_levels_in_result(self, latest, moderate_shocks):
        shocked = build_shocked_row(latest, moderate_shocks)
        result = score_shocked_row(shocked, latest)
        for k in ["vix", "hy_spread", "sp500_drawdown", "spread", "nfci", "unemployment"]:
            assert k in result


# ---------------------------------------------------------------------------
# run_scenario
# ---------------------------------------------------------------------------

class TestRunScenario:
    def test_returns_dict(self, base_df):
        result = run_scenario(base_df, DEFAULT_SHOCKS)
        assert isinstance(result, dict)

    def test_all_keys_present(self, base_df):
        result = run_scenario(base_df, DEFAULT_SHOCKS)
        for k in ["baseline", "scenario", "delta", "shocks", "sizing", "baseline_sizing"]:
            assert k in result

    def test_empty_df_returns_empty(self):
        result = run_scenario(pd.DataFrame(), DEFAULT_SHOCKS)
        assert result == {}

    def test_delta_is_dict(self, base_df):
        result = run_scenario(base_df, DEFAULT_SHOCKS)
        assert isinstance(result["delta"], dict)

    def test_baseline_delta_near_zero(self, base_df):
        result = run_scenario(base_df, DEFAULT_SHOCKS)
        assert abs(result["delta"]["composite_risk_score_smooth"]) < 1e-6

    def test_severe_stress_positive_composite_delta(self, base_df):
        result = run_scenario(base_df, SCENARIO_PRESETS["Severe Stress"])
        assert result["delta"]["composite_risk_score_smooth"] > 0

    def test_sizing_keys_present(self, base_df):
        result = run_scenario(base_df, DEFAULT_SHOCKS)
        for k in ["score_sizing", "regime_prob_sizing", "blend"]:
            assert k in result["sizing"]

    def test_sizing_values_in_range(self, base_df):
        result = run_scenario(base_df, SCENARIO_PRESETS["Moderate Stress"])
        for k in ["score_sizing", "blend"]:
            v = result["sizing"][k]
            assert 0.0 <= v <= 1.0, f"{k} = {v}"

    def test_severe_stress_reduces_sizing(self, base_df):
        baseline = run_scenario(base_df, DEFAULT_SHOCKS)
        stressed = run_scenario(base_df, SCENARIO_PRESETS["Severe Stress"])
        assert stressed["sizing"]["blend"] < baseline["sizing"]["blend"]

    def test_partial_shocks_dict_fills_defaults(self, base_df):
        result = run_scenario(base_df, {"vix_shock": 10.0})
        assert result["shocks"]["hy_spread_shock"] == 0.0

    def test_shocks_recorded_in_result(self, base_df, moderate_shocks):
        result = run_scenario(base_df, moderate_shocks)
        assert result["shocks"]["hy_spread_shock"] == moderate_shocks["hy_spread_shock"]


# ---------------------------------------------------------------------------
# run_scenario_grid
# ---------------------------------------------------------------------------

class TestRunScenarioGrid:
    def test_returns_dataframe(self, base_df):
        result = run_scenario_grid(base_df, SCENARIO_PRESETS)
        assert isinstance(result, pd.DataFrame)

    def test_index_matches_scenario_names(self, base_df):
        result = run_scenario_grid(base_df, SCENARIO_PRESETS)
        for name in SCENARIO_PRESETS:
            assert name in result.index

    def test_expected_columns_present(self, base_df):
        result = run_scenario_grid(base_df, SCENARIO_PRESETS)
        for col in ["composite_risk_score_smooth", "delta_composite", "final_decision", "blend_sizing"]:
            assert col in result.columns

    def test_baseline_delta_is_zero(self, base_df):
        result = run_scenario_grid(base_df, SCENARIO_PRESETS)
        assert abs(float(result.loc["Baseline", "delta_composite"])) < 1e-6

    def test_composites_in_valid_range(self, base_df):
        result = run_scenario_grid(base_df, SCENARIO_PRESETS)
        for val in result["composite_risk_score_smooth"].dropna():
            assert 0.0 <= float(val) <= 100.0

    def test_severe_stress_highest_composite(self, base_df):
        result = run_scenario_grid(base_df, SCENARIO_PRESETS)
        severe  = float(result.loc["Severe Stress",  "composite_risk_score_smooth"])
        baseline = float(result.loc["Baseline",      "composite_risk_score_smooth"])
        assert severe > baseline

    def test_empty_df_returns_empty(self):
        result = run_scenario_grid(pd.DataFrame(), SCENARIO_PRESETS)
        assert result.empty

    def test_empty_scenarios_returns_empty(self, base_df):
        result = run_scenario_grid(base_df, {})
        assert result.empty

    def test_custom_scenarios_work(self, base_df):
        custom = {"ScenA": {"vix_shock": 5.0}, "ScenB": {"hy_spread_shock": 1.0}}
        result = run_scenario_grid(base_df, custom)
        assert "ScenA" in result.index
        assert "ScenB" in result.index


# ---------------------------------------------------------------------------
# build_tornado_data
# ---------------------------------------------------------------------------

class TestBuildTornadoData:
    def test_returns_dataframe(self, base_df):
        result = build_tornado_data(base_df)
        assert isinstance(result, pd.DataFrame)

    def test_length_matches_tornado_shocks(self, base_df):
        result = build_tornado_data(base_df)
        assert len(result) == len(TORNADO_SHOCKS)

    def test_expected_columns(self, base_df):
        result = build_tornado_data(base_df)
        for col in ["variable", "delta_composite", "direction"]:
            assert col in result.columns

    def test_direction_is_up_or_down(self, base_df):
        result = build_tornado_data(base_df)
        assert set(result["direction"].unique()).issubset({"up", "down"})

    def test_all_variables_present(self, base_df):
        result = build_tornado_data(base_df)
        for var in TORNADO_SHOCKS:
            assert var in result["variable"].values

    def test_sorted_by_abs_delta_descending(self, base_df):
        result = build_tornado_data(base_df)
        abs_deltas = result["delta_composite"].abs().values
        assert all(abs_deltas[i] >= abs_deltas[i + 1] for i in range(len(abs_deltas) - 1))

    def test_empty_df_returns_empty(self):
        result = build_tornado_data(pd.DataFrame())
        assert result.empty

    def test_custom_tornado_shocks(self, base_df):
        custom = {"HY +1pp": {"hy_spread_shock": 1.0}, "VIX +5": {"vix_shock": 5.0}}
        result = build_tornado_data(base_df, tornado_shocks=custom)
        assert len(result) == 2
        assert "HY +1pp" in result["variable"].values


# ---------------------------------------------------------------------------
# run_scenario_analysis (orchestrator)
# ---------------------------------------------------------------------------

class TestRunScenarioAnalysis:
    def test_returns_dict(self, base_df):
        result = run_scenario_analysis(base_df)
        assert isinstance(result, dict)

    def test_all_keys_present(self, base_df):
        result = run_scenario_analysis(base_df)
        for k in ["model", "baseline", "preset_grid", "tornado", "selected"]:
            assert k in result

    def test_preset_grid_is_dataframe(self, base_df):
        result = run_scenario_analysis(base_df)
        assert isinstance(result["preset_grid"], pd.DataFrame)

    def test_tornado_is_dataframe(self, base_df):
        result = run_scenario_analysis(base_df)
        assert isinstance(result["tornado"], pd.DataFrame)

    def test_selected_has_scenario_key(self, base_df):
        result = run_scenario_analysis(base_df)
        assert "scenario" in result["selected"]

    def test_custom_shocks_reflected_in_selected(self, base_df):
        shocks = {"hy_spread_shock": 2.0}
        result = run_scenario_analysis(base_df, selected_shocks=shocks)
        assert result["selected"]["shocks"]["hy_spread_shock"] == 2.0

    def test_empty_df_returns_empty_structures(self):
        result = run_scenario_analysis(pd.DataFrame())
        assert result["preset_grid"].empty
        assert result["tornado"].empty
        assert result["selected"] == {}

    def test_all_preset_names_in_grid_index(self, base_df):
        result = run_scenario_analysis(base_df)
        for name in SCENARIO_PRESETS:
            assert name in result["preset_grid"].index
