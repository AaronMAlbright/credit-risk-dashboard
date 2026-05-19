import pandas as pd

from src.channel_attribution import channel_contribution_table, top_channel_drivers
from src.credit_relative_value import (
    compute_credit_relative_value,
    latest_relative_value_snapshot,
    relative_value_table,
)
from src.credit_strategy_memo import generate_credit_strategy_memo


def _df() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    return pd.DataFrame(
        {
            "hy_spread": [3.0 + i * 0.01 for i in range(120)],
            "ig_spread": [1.0 + i * 0.002 for i in range(120)],
            "bbb_spread": [1.5 + i * 0.004 for i in range(120)],
            "final_decision": ["Neutral"] * 60 + ["Caution"] * 60,
            "macro_risk_score_smooth": [20.0 + i * 0.1 for i in range(120)],
            "liquidity_regime_score_smooth": [25.0 + i * 0.1 for i in range(120)],
            "treasury_stress_score_smooth": [30.0 + i * 0.1 for i in range(120)],
            "credit_market_risk_score_smooth": [35.0 + i * 0.2 for i in range(120)],
            "cross_asset_divergence_score_smooth": [10.0 + i * 0.1 for i in range(120)],
            "market_internals_score_smooth": [15.0 + i * 0.1 for i in range(120)],
            "composite_risk_score_smooth": [40.0 + i * 0.1 for i in range(120)],
        },
        index=idx,
    )


def test_compute_credit_relative_value_columns():
    rv = compute_credit_relative_value(_df())
    assert "hy_spread_percentile" in rv.columns
    assert "ig_spread_percentile" in rv.columns
    assert "hy_ig_spread_ratio" in rv.columns
    assert "quality_tilt" in rv.columns
    assert rv["hy_spread_percentile"].dropna().iloc[-1] == 100.0


def test_latest_relative_value_snapshot_and_table():
    snap = latest_relative_value_snapshot(_df())
    assert snap["available"] is True
    assert snap["hy_spread_valuation"] in {"Cheap", "Crisis / Very Cheap"}
    assert snap["quality_tilt"]

    table = relative_value_table(_df())
    assert not table.empty
    assert {"metric", "level", "percentile", "valuation"}.issubset(table.columns)


def test_channel_contribution_table_and_drivers():
    table = channel_contribution_table(_df())
    assert not table.empty
    assert {"channel", "score", "contribution", "contribution_share"}.issubset(table.columns)
    assert abs(table["contribution_share"].sum() - 1.0) < 1e-9

    drivers = top_channel_drivers(_df(), n=2)
    assert len(drivers) == 2
    assert "score" in drivers[0]


def test_strategy_memo_includes_relative_value():
    memo = generate_credit_strategy_memo(_df())
    assert "Relative Value / Quality" in memo
    assert "Quality tilt" in memo

