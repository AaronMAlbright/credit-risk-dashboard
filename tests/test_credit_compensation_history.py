from datetime import datetime

import pandas as pd

from src.credit_compensation_history import (
    HISTORY_COLUMNS,
    append_scorecard_history,
    build_scorecard_history_row,
    load_scorecard_history,
)


def _df(**overrides):
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    data = {
        "hy_spread": [3.0, 3.2, 3.4, 6.0],
        "ig_spread_bps": [120, 125, 130, 150],
        "hy_yield": [7.0, 7.1, 7.2, 8.0],
        "ig_yield": [5.0, 5.1, 5.2, 5.4],
        "hy_spread_percentile": [30, 35, 40, 80],
        "final_decision": ["Neutral", "Neutral", "Neutral", "Neutral"],
        "composite_risk_score_smooth": [30, 32, 34, 35],
        "default_cycle_pct": [4.5, 4.7, 4.9, 5.1],
        "actual_chargeoff_pct": [1.0, 1.0, 1.0, 1.0],
        "actual_delinq_pct": [1.2, 1.2, 1.2, 1.2],
        "implied_vs_actual_gap": [3.5, 3.7, 3.9, 4.1],
        "sloos_change_90d": [-5.0, -4.0, -3.0, -2.0],
        "chargeoff_change_90d": [0.0, 0.0, 0.0, 0.0],
        "delinquency_change_90d": [0.0, 0.0, 0.0, 0.0],
    }
    data.update(overrides)
    return pd.DataFrame(data, index=idx)


def test_build_scorecard_history_row_shapes_snapshot():
    result = build_scorecard_history_row(
        _df(),
        recorded_at=datetime(2026, 6, 2, 12, 0, 0),
    )

    assert result["available"] is True
    row = result["row"]
    assert set(row) == set(HISTORY_COLUMNS)
    assert row["as_of"] == "2024-01-04"
    assert row["recorded_at"] == "2026-06-02T12:00:00"
    assert row["recommendation"] == "Add"
    assert row["hy_oas_bps"] == 600.0
    assert row["net_spread_beta"] is not None
    assert row["incremental_cdx_hy_protection_pct"] is not None


def test_append_scorecard_history_upserts_same_as_of(tmp_path):
    path = tmp_path / "scorecard_history.csv"
    first = append_scorecard_history(
        _df(),
        history_path=path,
        recorded_at=datetime(2026, 6, 2, 12, 0, 0),
    )
    second = append_scorecard_history(
        _df(hy_spread=[3.0, 3.2, 3.4, 6.5]),
        history_path=path,
        recorded_at=datetime(2026, 6, 2, 13, 0, 0),
    )

    history = pd.read_csv(path)
    assert first["row_count"] == 1
    assert second["row_count"] == 1
    assert len(history) == 1
    assert history.loc[0, "hy_oas_bps"] == 650.0


def test_load_scorecard_history_returns_empty_columns_for_missing_path(tmp_path):
    history = load_scorecard_history(tmp_path / "missing.csv")

    assert history.empty
    assert list(history.columns) == HISTORY_COLUMNS
