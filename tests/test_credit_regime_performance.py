import pandas as pd

from src.credit_positioning import current_positioning, positioning_table
from src.credit_regime_performance import (
    add_forward_market_moves,
    confidence_flag,
    latest_regime_performance_note,
    summarize_by_regime,
)


def _df() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=90, freq="B")
    regimes = ["Risk-On"] * 30 + ["Neutral"] * 30 + ["Caution"] * 20 + ["Risk-Off"] * 10
    sp500 = pd.Series(range(1000, 1090), index=idx).astype(float)
    hy = pd.Series([3.0 + i * 0.01 for i in range(90)], index=idx)
    ig = pd.Series([1.2 + i * 0.004 for i in range(90)], index=idx)
    return pd.DataFrame(
        {
            "final_decision": regimes,
            "sp500": sp500,
            "hy_spread": hy,
            "ig_spread": ig,
            "composite_risk_score_smooth": 50.0,
        },
        index=idx,
    )


def test_confidence_flag_breakpoints():
    assert confidence_flag(10) == "Exploratory"
    assert confidence_flag(20) == "Indicative"
    assert confidence_flag(50) == "Reliable"


def test_add_forward_market_moves():
    result = add_forward_market_moves(_df(), horizons=(21,))
    assert "sp500_forward_21d_return" in result.columns
    assert "hy_spread_forward_21d_change" in result.columns
    assert result["hy_spread_forward_21d_change"].dropna().iloc[0] > 0


def test_summarize_by_regime():
    table = summarize_by_regime(_df(), horizons=(21,))
    assert not table.empty
    assert {"regime", "horizon_days", "n_obs", "confidence"}.issubset(table.columns)
    assert set(table["regime"]).issubset({"Risk-On", "Neutral", "Caution", "Risk-Off"})


def test_latest_regime_performance_note():
    note = latest_regime_performance_note(_df(), horizon_days=21)
    assert note["available"] is True
    assert note["regime"] == "Risk-Off"
    assert note["horizon_days"] == 21


def test_current_positioning_and_table():
    pos = current_positioning(_df())
    assert pos["available"] is True
    assert pos["regime"] == "Risk-Off"
    assert pos["credit_beta"] == "Defensive"

    table = positioning_table(_df())
    assert not table.empty
    assert {"dimension", "stance"}.issubset(table.columns)

