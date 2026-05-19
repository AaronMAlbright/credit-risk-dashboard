import pandas as pd

from src.credit_strategy_memo import generate_credit_strategy_memo
from src.credit_taxonomy import compute_channel_scores, latest_channel_snapshot
from src.spread_decomposition import (
    decompose_spreads,
    expected_loss_bps,
    latest_spread_snapshot,
)


def _df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    return pd.DataFrame(
        {
            "hy_spread": [3.5, 4.0, 5.0, 6.0],
            "final_decision": ["Risk-On", "Neutral", "Caution", "Risk-Off"],
            "composite_risk_score_smooth": [20.0, 35.0, 55.0, 80.0],
            "macro_risk_score_smooth": [15.0, 30.0, 50.0, 75.0],
            "liquidity_regime_score_smooth": [20.0, 35.0, 55.0, 70.0],
            "credit_market_risk_score_smooth": [25.0, 40.0, 65.0, 85.0],
            "cross_asset_divergence_score_smooth": [10.0, 25.0, 45.0, 60.0],
            "market_internals_score_smooth": [20.0, 30.0, 50.0, 65.0],
        },
        index=idx,
    )


def test_expected_loss_bps():
    assert expected_loss_bps(0.05, recovery_rate=0.40) == 300.0


def test_decompose_spreads_percent_units():
    result = decompose_spreads(_df())
    assert list(result.columns) == [
        "spread_oas_bps",
        "default_probability",
        "recovery_rate",
        "expected_loss_bps",
        "excess_spread_bps",
        "spread_compensation_ratio",
        "spread_valuation",
    ]
    assert result["spread_oas_bps"].iloc[-1] == 600.0
    assert result["expected_loss_bps"].iloc[-1] == 720.0
    assert result["spread_valuation"].iloc[-1] == "Very Rich"


def test_latest_spread_snapshot_available():
    snapshot = latest_spread_snapshot(_df())
    assert snapshot["available"] is True
    assert snapshot["spread_oas_bps"] == 600.0
    assert snapshot["default_probability"] == 0.12


def test_channel_scores_and_snapshot():
    scores = compute_channel_scores(_df())
    assert "institutional_credit_score" in scores.columns
    assert scores["credit_market_channel_score"].iloc[-1] == 85.0

    snapshot = latest_channel_snapshot(_df())
    assert snapshot["available"] is True
    assert snapshot["composite"] is not None
    assert len(snapshot["channels"]) == 6


def test_strategy_memo_contains_credit_language():
    memo = generate_credit_strategy_memo(_df())
    assert "Credit Strategy Memo" in memo
    assert "Regime: Risk-Off" in memo
    assert "Spread Compensation" in memo
    assert "Dominant Risk Channels" in memo
    assert "Protect capital" in memo

