"""
Unit tests for pure scoring and classification functions.
No fixtures, no I/O, no FRED API — each test constructs its own minimal inputs.
"""
import pytest
import pandas as pd

from src.risk_engine import (
    classify_yield_curve_regime,
    classify_credit_regime,
    classify_labor_warning,
    compute_macro_risk_score,
    compute_credit_market_risk_score,
    compute_liquidity_score,
    compute_risk_appetite_score,
    compute_complacency_score,
    compute_mean_reversion_score,
)
from src.treasury_engine import compute_treasury_stress_score
from src.liquidity_engine import compute_liquidity_regime_score
from src.market_internals_engine import (
    compute_cross_asset_divergence_score,
    compute_market_internals_score,
)
from src.portfolio_engine import generate_portfolio_weights
from src.crisis_similarity import compute_crisis_similarity


# ---------------------------------------------------------------------------
# Yield curve regime classifier
# ---------------------------------------------------------------------------
class TestClassifyYieldCurveRegime:
    def test_steep(self):
        assert classify_yield_curve_regime(1.5) == "Steep (Expansion)"

    def test_normal(self):
        assert classify_yield_curve_regime(0.5) == "Normal"

    def test_flat(self):
        assert classify_yield_curve_regime(-0.25) == "Flat / Inversion Risk"

    def test_inverted(self):
        assert classify_yield_curve_regime(-0.75) == "Inverted (Recession Risk)"

    def test_boundary_at_one(self):
        # spread == 1 is NOT > 1, so it falls into "Normal"
        assert classify_yield_curve_regime(1.0) == "Normal"

    def test_boundary_at_zero(self):
        # spread == 0 is NOT > 0, so it falls into "Flat / Inversion Risk"
        assert classify_yield_curve_regime(0.0) == "Flat / Inversion Risk"


# ---------------------------------------------------------------------------
# Credit regime classifier
# ---------------------------------------------------------------------------
class TestClassifyCreditRegime:
    def test_credit_stress(self):
        assert classify_credit_regime(-0.5, 6.0) == "Credit Stress"

    def test_late_cycle(self):
        assert classify_credit_regime(-0.5, 4.0) == "Late Cycle"

    def test_expansion(self):
        assert classify_credit_regime(1.5, 3.5) == "Expansion"

    def test_neutral(self):
        assert classify_credit_regime(0.5, 4.5) == "Neutral"


# ---------------------------------------------------------------------------
# Labor warning classifier
# ---------------------------------------------------------------------------
class TestClassifyLaborWarning:
    def test_triggered(self):
        assert classify_labor_warning(0.6, 0.1) == "Triggered"

    def test_watch_sahm(self):
        assert classify_labor_warning(0.35, 0.0) == "Watch"

    def test_watch_change(self):
        assert classify_labor_warning(0.1, 0.3) == "Watch"

    def test_low(self):
        assert classify_labor_warning(0.1, 0.0) == "Low"


# ---------------------------------------------------------------------------
# Score bounds: every scoring function must return a float in [0, 100]
# ---------------------------------------------------------------------------
class TestScoreBounds:

    # --- macro ---
    def _macro(self, spread=0.5, unemployment=4.0, nfci=-0.2,
                unemployment_change=0.0, spread_change=0.0,
                nfci_change=0.0, sahm_like=0.0):
        return compute_macro_risk_score(
            spread, unemployment, nfci,
            unemployment_change, spread_change, nfci_change, sahm_like,
        )

    def test_macro_benign_in_bounds(self):
        score = self._macro(spread=2.0, unemployment=3.5)
        assert 0 <= score <= 100

    def test_macro_stressed_in_bounds(self):
        score = self._macro(
            spread=-1.0, unemployment=7.5, nfci=1.2,
            unemployment_change=1.0, nfci_change=0.5, sahm_like=0.8,
        )
        assert 0 <= score <= 100

    def test_macro_neutral_in_bounds(self):
        assert 0 <= self._macro() <= 100

    # --- credit market ---
    def _credit(self, hy_spread=3.0, hy_change_30d=0.0, hy_change_90d=0.0,
                 credit_impulse=0.0, vix=15.0, vix_change_30d=0.0,
                 credit_equity_divergence="Neutral", vol_credit_mismatch="Neutral"):
        return compute_credit_market_risk_score(
            hy_spread, hy_change_30d, hy_change_90d,
            credit_impulse, vix, vix_change_30d,
            credit_equity_divergence, vol_credit_mismatch,
        )

    def test_credit_benign_in_bounds(self):
        assert 0 <= self._credit() <= 100

    def test_credit_stressed_in_bounds(self):
        score = self._credit(
            hy_spread=8.0, hy_change_30d=1.0, hy_change_90d=1.0,
            credit_impulse=0.5, vix=40.0, vix_change_30d=12.0,
            credit_equity_divergence="Hidden Credit Stress",
            vol_credit_mismatch="Credit Warning / Vol Complacency",
        )
        assert 0 <= score <= 100

    def test_credit_recovery_confirmation_in_bounds(self):
        score = self._credit(
            vix=28.0, hy_change_30d=-0.5,
            vol_credit_mismatch="Recovery Confirmation",
        )
        assert 0 <= score <= 100

    # --- liquidity ---
    def test_liquidity_score_in_bounds(self):
        score = compute_liquidity_score(
            nfci=0.3, nfci_change_90d=0.05,
            hy_change_30d=0.1, vix_change_30d=2.0,
        )
        assert 0 <= score <= 100

    def test_liquidity_score_stressed_in_bounds(self):
        score = compute_liquidity_score(
            nfci=0.8, nfci_change_90d=0.3,
            hy_change_30d=0.6, vix_change_30d=8.0,
        )
        assert 0 <= score <= 100

    # --- treasury stress ---
    def test_treasury_stress_score_in_bounds(self):
        score = compute_treasury_stress_score(
            real_yield_z=0.5, real_yield_change_90d=0.1,
            curve_steepening_velocity_90d=0.2,
        )
        assert 0 <= score <= 100

    def test_treasury_stress_score_max_in_bounds(self):
        score = compute_treasury_stress_score(
            real_yield_z=2.0, real_yield_change_90d=0.8,
            curve_steepening_velocity_90d=0.8,
        )
        assert 0 <= score <= 100

    # --- liquidity regime ---
    def test_liquidity_regime_score_in_bounds(self):
        score = compute_liquidity_regime_score(
            nfci=0.0, nfci_change_90d=0.0,
            hy_change_30d=0.0, vix_change_30d=0.0,
            treasury_stress_score=0.0,
        )
        assert 0 <= score <= 100

    def test_liquidity_regime_score_severe_in_bounds(self):
        score = compute_liquidity_regime_score(
            nfci=0.8, nfci_change_90d=0.3,
            hy_change_30d=0.6, vix_change_30d=8.0,
            treasury_stress_score=80.0,
        )
        assert 0 <= score <= 100

    # --- cross-asset divergence ---
    def test_cross_asset_divergence_in_bounds(self):
        score = compute_cross_asset_divergence_score(
            sp500_return_30d=0.04, sp500_drawdown=-0.005,
            hy_change_30d=0.2, vix=14.0, vix_change_30d=-3.0,
        )
        assert 0 <= score <= 100

    def test_cross_asset_no_divergence_in_bounds(self):
        score = compute_cross_asset_divergence_score(
            sp500_return_30d=-0.01, sp500_drawdown=-0.08,
            hy_change_30d=-0.1, vix=25.0, vix_change_30d=2.0,
        )
        assert 0 <= score <= 100

    # --- market internals ---
    def test_market_internals_strong_in_bounds(self):
        score = compute_market_internals_score(
            sp500_return_30d=0.06, sp500_return_5d=0.02,
            sp500_drawdown=-0.002, vix=13.0, vix_change_30d=-3.0,
        )
        assert 0 <= score <= 100

    def test_market_internals_weak_in_bounds(self):
        score = compute_market_internals_score(
            sp500_return_30d=-0.05, sp500_return_5d=-0.02,
            sp500_drawdown=-0.15, vix=30.0, vix_change_30d=6.0,
        )
        assert 0 <= score <= 100

    # --- mean reversion ---
    def test_mean_reversion_in_bounds(self):
        score = compute_mean_reversion_score(
            macro_risk_score=75.0, credit_market_risk_score=80.0,
            hy_spread=6.0, hy_change_30d=0.6,
            vix=38.0, vix_change_30d=9.0, sp500_drawdown=-0.18,
        )
        assert 0 <= score <= 100

    def test_mean_reversion_benign_in_bounds(self):
        score = compute_mean_reversion_score(
            macro_risk_score=15.0, credit_market_risk_score=10.0,
            hy_spread=2.8, hy_change_30d=-0.1,
            vix=13.0, vix_change_30d=-2.0, sp500_drawdown=-0.005,
        )
        assert 0 <= score <= 100


# ---------------------------------------------------------------------------
# Score monotonicity: inputs that should raise the score must do so
# ---------------------------------------------------------------------------
class TestScoreMonotonicity:

    def test_higher_unemployment_raises_macro_risk(self):
        low = compute_macro_risk_score(0.5, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0)
        high = compute_macro_risk_score(0.5, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert high > low

    def test_wider_hy_spread_raises_credit_risk(self):
        low = compute_credit_market_risk_score(
            2.8, 0.0, 0.0, 0.0, 15.0, 0.0, "Neutral", "Neutral"
        )
        high = compute_credit_market_risk_score(
            7.0, 0.0, 0.0, 0.0, 15.0, 0.0, "Neutral", "Neutral"
        )
        assert high > low

    def test_higher_vix_raises_credit_risk(self):
        low = compute_credit_market_risk_score(
            3.0, 0.0, 0.0, 0.0, 15.0, 0.0, "Neutral", "Neutral"
        )
        high = compute_credit_market_risk_score(
            3.0, 0.0, 0.0, 0.0, 40.0, 0.0, "Neutral", "Neutral"
        )
        assert high > low

    def test_higher_nfci_raises_liquidity_score(self):
        low = compute_liquidity_score(-0.6, 0.0, 0.0, 0.0)
        high = compute_liquidity_score(0.8, 0.0, 0.0, 0.0)
        assert high > low

    def test_higher_real_yield_z_raises_treasury_stress(self):
        low = compute_treasury_stress_score(0.5, 0.0, 0.0)
        high = compute_treasury_stress_score(2.0, 0.0, 0.0)
        assert high > low


# ---------------------------------------------------------------------------
# Portfolio weights
# ---------------------------------------------------------------------------
_ALL_DECISIONS = [
    "Risk-On",
    "Neutral",
    "Caution",
    "Risk-Off",
]


class TestPortfolioWeights:

    @pytest.mark.parametrize("decision", _ALL_DECISIONS)
    def test_weights_sum_to_one(self, decision):
        row = pd.Series({"final_decision": decision})
        w = generate_portfolio_weights(row)
        total = w["equity_weight"] + w["credit_weight"] + w["cash_weight"]
        assert abs(total - 1.0) < 1e-6, f"'{decision}' weights sum to {total}"

    @pytest.mark.parametrize("decision", _ALL_DECISIONS)
    def test_all_weights_non_negative(self, decision):
        row = pd.Series({"final_decision": decision})
        w = generate_portfolio_weights(row)
        assert w["equity_weight"] >= 0
        assert w["credit_weight"] >= 0
        assert w["cash_weight"] >= 0

    def test_unknown_decision_weight_sum(self):
        row = pd.Series({"final_decision": "Unrecognized Regime"})
        w = generate_portfolio_weights(row)
        total = w["equity_weight"] + w["credit_weight"] + w["cash_weight"]
        assert abs(total - 1.0) < 1e-6

    def test_duration_bias_key_present(self):
        row = pd.Series({"final_decision": "Neutral"})
        w = generate_portfolio_weights(row)
        assert "duration_bias" in w

    def test_risk_on_higher_equity_than_risk_off(self):
        risk_on = generate_portfolio_weights(pd.Series({"final_decision": "Risk-On"}))
        risk_off = generate_portfolio_weights(pd.Series({"final_decision": "Risk-Off"}))
        assert risk_on["equity_weight"] > risk_off["equity_weight"]

    def test_risk_off_higher_cash_than_risk_on(self):
        risk_on = generate_portfolio_weights(pd.Series({"final_decision": "Risk-On"}))
        risk_off = generate_portfolio_weights(pd.Series({"final_decision": "Risk-Off"}))
        assert risk_off["cash_weight"] > risk_on["cash_weight"]


# ---------------------------------------------------------------------------
# Crisis similarity
# ---------------------------------------------------------------------------
class TestCrisisSimilarity:

    def _row(self, macro=30, credit=30, complacency=50, mean_rev=20, treasury=20):
        return pd.Series({
            "macro_risk_score_smooth": macro,
            "credit_market_risk_score_smooth": credit,
            "complacency_score_smooth": complacency,
            "mean_reversion_score_smooth": mean_rev,
            "treasury_stress_score_smooth": treasury,
        })

    def test_returns_list_of_tuples(self):
        result = compute_crisis_similarity(self._row())
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_returns_all_five_analogs(self):
        assert len(compute_crisis_similarity(self._row())) == 5

    def test_sorted_descending_by_similarity(self):
        result = compute_crisis_similarity(self._row())
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_non_negative(self):
        result = compute_crisis_similarity(self._row())
        assert all(s >= 0 for _, s in result)

    def test_covid_profile_tops_covid_analog(self):
        # Near-COVID inputs: high macro/credit stress, low complacency, high mean-reversion
        result = compute_crisis_similarity(self._row(macro=88, credit=93, complacency=6, mean_rev=90, treasury=55))
        top_name, _ = result[0]
        assert top_name == "COVID Crash"

    def test_meltup_profile_tops_meltup_analog(self):
        # Near late-cycle melt-up: low macro/credit, very high complacency
        result = compute_crisis_similarity(self._row(macro=12, credit=8, complacency=88, mean_rev=2, treasury=18))
        top_name, _ = result[0]
        assert top_name == "Late-Cycle Melt-Up"
