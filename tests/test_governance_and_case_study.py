import pandas as pd

from src.composite_comparison import compare_composites, composite_comparison_summary
from src.credit_case_study import case_study_table, episode_case_study
from src.model_governance import (
    governance_status_table,
    known_limitations_table,
    required_institutional_data_table,
)


def _df() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=1100, freq="B")
    n = len(idx)
    return pd.DataFrame(
        {
            "date": idx,
            "final_decision": ["Neutral"] * 300 + ["Risk-Off"] * 200 + ["Caution"] * 300 + ["Risk-On"] * 300,
            "composite_risk_score_smooth": [30 + (i % 50) * 0.5 for i in range(n)],
            "institutional_credit_score": [32 + (i % 50) * 0.45 for i in range(n)],
            "macro_risk_score_smooth": [20 + (i % 50) * 0.3 for i in range(n)],
            "credit_market_risk_score_smooth": [25 + (i % 50) * 0.4 for i in range(n)],
            "liquidity_regime_score_smooth": [22 + (i % 50) * 0.2 for i in range(n)],
            "treasury_stress_score_smooth": [28 + (i % 50) * 0.2 for i in range(n)],
            "cross_asset_divergence_score_smooth": [18 + (i % 50) * 0.2 for i in range(n)],
            "market_internals_score_smooth": [15 + (i % 50) * 0.1 for i in range(n)],
            "hy_spread": [3.0 + (i % 100) * 0.01 for i in range(n)],
            "sp500": [3000 + i for i in range(n)],
        },
        index=idx,
    )


def test_governance_tables():
    assert not governance_status_table().empty
    assert not known_limitations_table().empty
    assert not required_institutional_data_table().empty


def test_composite_comparison():
    comp = compare_composites(_df())
    assert not comp.empty
    assert "composite_gap" in comp.columns

    summary = composite_comparison_summary(_df())
    assert summary["available"] is True
    assert "correlation" in summary


def test_case_study_table():
    table = case_study_table(_df())
    assert not table.empty
    assert {"episode", "window", "peak_score", "top_driver"}.issubset(table.columns)


def test_episode_case_study():
    result = episode_case_study(_df(), "2022 Rates Shock")
    assert result["available"] is True
    assert result["episode"] == "2022 Rates Shock"
    assert result["lesson"]

