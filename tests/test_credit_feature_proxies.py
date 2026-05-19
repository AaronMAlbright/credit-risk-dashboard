import pandas as pd

from src.credit_feature_proxies import add_credit_feature_proxies
from src.credit_taxonomy import compute_channel_scores


def test_credit_feature_proxies_populate_fundamentals_and_technicals():
    idx = pd.date_range("2023-01-02", periods=180, freq="B")
    df = pd.DataFrame(
        {
            "hy_spread": [3.0 + i * 0.01 for i in range(180)],
            "ig_spread": [0.9 + i * 0.002 for i in range(180)],
            "bbb_spread": [1.2 + i * 0.003 for i in range(180)],
            "vix": [14.0 + i * 0.02 for i in range(180)],
            "sp500_drawdown": [-0.01 - i * 0.0002 for i in range(180)],
            "unemployment": [4.0 + i * 0.001 for i in range(180)],
            "sloos_ci": [5.0 + i * 0.05 for i in range(180)],
            "hy_total_return_daily": [-0.0005] * 180,
            "ig_total_return_daily": [-0.0002] * 180,
            "loan_growth_90d": [-0.01] * 180,
            "credit_market_risk_score_smooth": [30.0] * 180,
            "liquidity_regime_score_smooth": [20.0] * 180,
            "sahm_like": [0.1] * 180,
        },
        index=idx,
    )

    enriched = add_credit_feature_proxies(df)
    scores = compute_channel_scores(enriched)

    assert enriched["default_cycle_score"].notna().any()
    assert enriched["primary_market_score"].notna().any()
    assert scores["fundamentals_channel_score"].notna().any()
    assert scores["technicals_channel_score"].notna().any()
