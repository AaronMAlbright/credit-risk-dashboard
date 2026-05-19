import pandas as pd

from src.credit_presentation import (
    build_credit_brief,
    credit_brief_markdown,
    credit_glossary_table,
    framework_assumptions_table,
)


def _df() -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=90, freq="B")
    return pd.DataFrame(
        {
            "final_decision": ["Neutral"] * 45 + ["Caution"] * 45,
            "composite_risk_score_smooth": [35.0 + i * 0.1 for i in range(90)],
            "hy_spread": [3.5 + i * 0.01 for i in range(90)],
            "ig_spread": [1.2 + i * 0.002 for i in range(90)],
            "macro_risk_score_smooth": [25.0 + i * 0.1 for i in range(90)],
            "credit_market_risk_score_smooth": [35.0 + i * 0.1 for i in range(90)],
            "liquidity_regime_score_smooth": [30.0 + i * 0.1 for i in range(90)],
            "treasury_stress_score_smooth": [25.0 + i * 0.1 for i in range(90)],
            "cross_asset_divergence_score_smooth": [20.0 + i * 0.1 for i in range(90)],
            "market_internals_score_smooth": [15.0 + i * 0.1 for i in range(90)],
            "sp500": [4000.0 + i for i in range(90)],
        },
        index=idx,
    )


def test_build_credit_brief():
    brief = build_credit_brief(_df())
    assert brief["available"] is True
    assert brief["regime"] == "Caution"
    assert brief["credit_beta"] == "Underweight"
    assert brief["evidence"]


def test_credit_brief_markdown():
    md = credit_brief_markdown(_df())
    assert "# Credit Brief" in md
    assert "## Stance" in md
    assert "## Evidence" in md
    assert "Credit beta" in md


def test_assumptions_and_glossary_tables():
    assumptions = framework_assumptions_table()
    glossary = credit_glossary_table()
    assert {"Assumption", "Value", "Use"}.issubset(assumptions.columns)
    assert {"Term", "Definition"}.issubset(glossary.columns)
    assert len(assumptions) >= 5
    assert len(glossary) >= 5

