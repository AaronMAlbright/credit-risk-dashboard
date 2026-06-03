from datetime import datetime

import pandas as pd

from src.credit_compensation_history import (
    HISTORY_COLUMNS,
    append_scorecard_history,
    build_scorecard_trend_views,
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


def test_build_scorecard_trend_views_single_row_disables_charts():
    history = pd.DataFrame([
        {
            "as_of": "2026-06-02",
            "recommendation": "Upgrade Quality",
            "hy_oas_bps": 272.0,
            "expected_loss_bps": 240.0,
            "excess_spread_bps": 32.0,
            "spread_compensation_ratio": 1.13,
            "net_spread_beta": 0.288,
            "target_net_spread_beta": 0.300,
            "incremental_cdx_hy_protection_pct": 0.0,
            "constraint_breach_count": 0,
            "ig_weight_pct": 43.0,
            "bbb_weight_pct": 20.0,
            "bb_weight_pct": 20.0,
            "b_weight_pct": 0.0,
            "ccc_weight_pct": 0.0,
            "cash_weight_pct": 12.0,
            "hedge_weight_pct": 5.0,
        }
    ])

    views = build_scorecard_trend_views(history)

    assert views["available"] is True
    assert views["has_charts"] is False
    assert len(views["recommendation_timeline"]) == 1
    assert list(views["spread_trends"].columns) == ["hy_oas_bps", "expected_loss_bps", "excess_spread_bps"]


def test_build_scorecard_trend_views_multi_row_returns_chart_frames():
    history = pd.DataFrame([
        {
            "as_of": "2026-06-01",
            "recommendation": "Hold",
            "hy_oas_bps": 300.0,
            "expected_loss_bps": 240.0,
            "excess_spread_bps": 60.0,
            "spread_compensation_ratio": 1.25,
            "net_spread_beta": 0.25,
            "target_net_spread_beta": 0.30,
            "incremental_cdx_hy_protection_pct": 0.0,
            "constraint_breach_count": 0,
            "ig_weight_pct": 30.0,
            "bbb_weight_pct": 20.0,
            "bb_weight_pct": 20.0,
            "b_weight_pct": 15.0,
            "ccc_weight_pct": 0.0,
            "cash_weight_pct": 10.0,
            "hedge_weight_pct": 5.0,
        },
        {
            "as_of": "2026-06-02",
            "recommendation": "Upgrade Quality",
            "hy_oas_bps": 272.0,
            "expected_loss_bps": 240.0,
            "excess_spread_bps": 32.0,
            "spread_compensation_ratio": 1.13,
            "net_spread_beta": 0.288,
            "target_net_spread_beta": 0.30,
            "incremental_cdx_hy_protection_pct": 0.0,
            "constraint_breach_count": 0,
            "ig_weight_pct": 43.0,
            "bbb_weight_pct": 20.0,
            "bb_weight_pct": 20.0,
            "b_weight_pct": 0.0,
            "ccc_weight_pct": 0.0,
            "cash_weight_pct": 12.0,
            "hedge_weight_pct": 5.0,
        },
    ])

    views = build_scorecard_trend_views(history)

    assert views["has_charts"] is True
    assert len(views["spread_trends"]) == 2
    assert "net_spread_beta" in views["beta_trends"].columns
    assert "ig_weight_pct" in views["weight_trends"].columns
