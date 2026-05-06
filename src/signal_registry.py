SIGNAL_REGISTRY = {
    "macro_risk": {"weight": 1.0, "threshold": 40, "direction": "higher_is_riskier"},
    "credit_risk": {"weight": 1.2, "threshold": 45, "direction": "higher_is_riskier"},
    "liquidity_regime": {"weight": 1.0, "threshold": 40, "direction": "higher_is_riskier"},
    "cross_asset_divergence": {"weight": 0.5, "threshold": 50, "direction": "higher_is_riskier"},
    "complacency": {"weight": 1.4, "threshold": 70, "direction": "higher_is_riskier"},
    "mean_reversion": {"weight": 1.0, "threshold": 35, "direction": "higher_is_bullish"},
    "risk_appetite": {"weight": 1.0, "threshold": 70, "direction": "higher_is_bullish"},
    "treasury_stress": {"weight": 0.8, "threshold": 45, "direction": "higher_is_riskier"},
}