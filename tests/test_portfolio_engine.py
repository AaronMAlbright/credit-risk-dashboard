"""Tests for src/portfolio_engine.py and src/crisis_similarity.py."""

import pytest
import numpy as np
from src.portfolio_engine import generate_portfolio_weights, normalize_weights
from src.crisis_similarity import CRISIS_ANALOGS, compute_crisis_similarity


# ---------------------------------------------------------------------------
# normalize_weights
# ---------------------------------------------------------------------------

class TestNormalizeWeights:
    def test_weights_sum_to_one(self):
        w = normalize_weights({"equity_weight": 0.5, "credit_weight": 0.3,
                               "cash_weight": 0.2, "duration_bias": "Neutral"})
        assert abs(w["equity_weight"] + w["credit_weight"] + w["cash_weight"] - 1.0) < 1e-6

    def test_zero_total_returns_all_cash(self):
        w = normalize_weights({"equity_weight": 0, "credit_weight": 0,
                               "cash_weight": 0, "duration_bias": "Neutral"})
        assert w["cash_weight"] == 1.0
        assert w["equity_weight"] == 0.0
        assert w["credit_weight"] == 0.0

    def test_duration_bias_preserved(self):
        w = normalize_weights({"equity_weight": 1, "credit_weight": 1,
                               "cash_weight": 1, "duration_bias": "Long Duration"})
        assert w["duration_bias"] == "Long Duration"


# ---------------------------------------------------------------------------
# generate_portfolio_weights
# ---------------------------------------------------------------------------

class TestGeneratePortfolioWeights:
    def _row(self, decision):
        return {"final_decision": decision}

    def test_returns_four_keys(self):
        w = generate_portfolio_weights(self._row("Neutral"))
        assert set(w) == {"equity_weight", "credit_weight", "cash_weight", "duration_bias"}

    def test_weights_always_sum_to_one(self):
        decisions = [
            "Buy Stress", "Watch Entry", "Stress / Stabilization Watch",
            "Credit Warning", "Divergence Warning", "Avoid Chasing Risk",
            "Hold / Do Not Chase", "Wait", "Neutral", "Unknown Decision",
        ]
        for d in decisions:
            w = generate_portfolio_weights(self._row(d))
            total = w["equity_weight"] + w["credit_weight"] + w["cash_weight"]
            assert abs(total - 1.0) < 1e-6, f"Failed for decision: {d}"

    def test_buy_stress_high_equity(self):
        w = generate_portfolio_weights(self._row("Buy Stress"))
        assert w["equity_weight"] > w["cash_weight"]
        assert w["duration_bias"] == "Long Duration"

    def test_stress_watch_high_cash(self):
        w = generate_portfolio_weights(self._row("Stress / Stabilization Watch"))
        assert w["cash_weight"] > w["equity_weight"]

    def test_unknown_decision_returns_safe_default(self):
        w = generate_portfolio_weights(self._row("Something New"))
        total = w["equity_weight"] + w["credit_weight"] + w["cash_weight"]
        assert abs(total - 1.0) < 1e-6

    def test_all_weights_non_negative(self):
        for d in ["Buy Stress", "Neutral", "Credit Warning", "Wait"]:
            w = generate_portfolio_weights(self._row(d))
            assert w["equity_weight"] >= 0
            assert w["credit_weight"] >= 0
            assert w["cash_weight"] >= 0


# ---------------------------------------------------------------------------
# compute_crisis_similarity
# ---------------------------------------------------------------------------

class TestComputeCrisisSimilarity:
    def _row(self, macro=50, credit=50, complacency=50,
             mean_rev=50, treasury=50):
        return {
            "macro_risk_score_smooth":          macro,
            "credit_market_risk_score_smooth":  credit,
            "complacency_score_smooth":         complacency,
            "mean_reversion_score_smooth":      mean_rev,
            "treasury_stress_score_smooth":     treasury,
        }

    def test_returns_list_of_tuples(self):
        result = compute_crisis_similarity(self._row())
        assert isinstance(result, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in result)

    def test_one_entry_per_analog(self):
        result = compute_crisis_similarity(self._row())
        assert len(result) == len(CRISIS_ANALOGS)

    def test_sorted_descending(self):
        result = compute_crisis_similarity(self._row())
        scores = [r[1] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_perfect_match_scores_100(self):
        analog = CRISIS_ANALOGS["COVID Crash"]
        row = self._row(
            macro=analog["macro_risk"],
            credit=analog["credit_risk"],
            complacency=analog["complacency"],
            mean_rev=analog["mean_reversion"],
            treasury=analog["treasury_stress"],
        )
        result = compute_crisis_similarity(row)
        top_name, top_score = result[0]
        assert top_name == "COVID Crash"
        assert abs(top_score - 100.0) < 0.1

    def test_similarity_non_negative(self):
        result = compute_crisis_similarity(self._row(0, 0, 0, 0, 0))
        assert all(r[1] >= 0 for r in result)

    def test_analog_names_are_known(self):
        result = compute_crisis_similarity(self._row())
        names = {r[0] for r in result}
        assert names == set(CRISIS_ANALOGS.keys())

    def test_extreme_values_dont_crash(self):
        result = compute_crisis_similarity(self._row(100, 100, 100, 100, 100))
        assert len(result) == len(CRISIS_ANALOGS)
