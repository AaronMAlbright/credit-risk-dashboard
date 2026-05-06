def simulate_shock(latest, vix_spike=10, hy_widening=0.75, sp500_drop=-0.07):
    shocked = latest.copy()

    shocked["vix"] = latest["vix"] + vix_spike
    shocked["hy_spread"] = latest["hy_spread"] + hy_widening
    shocked["hy_change_30d"] = latest["hy_change_30d"] + hy_widening
    shocked["sp500_drawdown"] = latest["sp500_drawdown"] + sp500_drop

    return shocked


def summarize_scenario(base_latest, shocked_latest):
    return {
        "base_vix": base_latest["vix"],
        "shock_vix": shocked_latest["vix"],
        "base_hy_spread": base_latest["hy_spread"],
        "shock_hy_spread": shocked_latest["hy_spread"],
        "base_sp500_drawdown": base_latest["sp500_drawdown"],
        "shock_sp500_drawdown": shocked_latest["sp500_drawdown"],
    }