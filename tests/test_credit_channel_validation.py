import pandas as pd

from src.credit_channel_validation import channel_validation_table, latest_channel_validation_snapshot
from src.credit_feature_proxies import add_credit_feature_proxies


def _df() -> pd.DataFrame:
    idx = pd.date_range("2023-01-02", periods=260, freq="B")
    base = pd.DataFrame(
        {
            "hy_spread": [3.0 + i * 0.01 for i in range(260)],
            "ig_spread": [0.9 + i * 0.002 for i in range(260)],
            "bbb_spread": [1.2 + i * 0.003 for i in range(260)],
            "vix": [14.0 + i * 0.02 for i in range(260)],
            "sp500_drawdown": [-0.01 - i * 0.0002 for i in range(260)],
            "unemployment": [4.0 + i * 0.001 for i in range(260)],
            "sloos_ci": [5.0 + i * 0.05 for i in range(260)],
            "hy_total_return_daily": [-0.0005] * 260,
            "ig_total_return_daily": [-0.0002] * 260,
            "loan_growth_90d": [-0.01] * 260,
            "credit_market_risk_score_smooth": [30.0] * 260,
            "liquidity_regime_score_smooth": [20.0] * 260,
            "sahm_like": [0.1] * 260,
            "sp500": [100.0 + i * 0.2 for i in range(260)],
        },
        index=idx,
    )
    return add_credit_feature_proxies(base)


def test_channel_validation_table_shape():
    table = channel_validation_table(_df())
    assert not table.empty
    assert set(table["horizon_days"]) == {21, 63, 126}
    assert {"avg_sp500_return", "avg_hy_spread_change", "hit_rate_hy_widening"}.issubset(table.columns)


def test_channel_validation_snapshot_available():
    snap = latest_channel_validation_snapshot(_df())
    assert snap["available"] is True
    assert "summary" in snap
    assert len(snap["table"]) > 0
