from pathlib import Path
from datetime import datetime
import pandas as pd


HISTORY_DIR = Path("history")
HISTORY_PATH = HISTORY_DIR / "model_run_history.csv"


def log_model_run(latest):
    HISTORY_DIR.mkdir(exist_ok=True)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "macro_risk": latest["macro_risk_score_smooth"],
        "credit_risk": latest["credit_market_risk_score_smooth"],
        "liquidity_regime": latest["liquidity_regime_score_smooth"],
        "cross_asset_divergence": latest["cross_asset_divergence_score_smooth"],
        "complacency": latest["complacency_score_smooth"],
        "mean_reversion": latest["mean_reversion_score_smooth"],
        "risk_appetite": latest["risk_appetite_score_smooth"],
        "treasury_stress": latest["treasury_stress_score_smooth"],

        "final_decision": latest["final_decision"],
        "transition_regime": latest["transition_regime"],

        "equity_weight": latest["equity_weight"],
        "credit_weight": latest["credit_weight"],
        "cash_weight": latest["cash_weight"],
        "duration_bias": latest["duration_bias"],
    }

    new_row = pd.DataFrame([row])

    if HISTORY_PATH.exists():
        existing = pd.read_csv(HISTORY_PATH)
        output = pd.concat([existing, new_row], ignore_index=True)
    else:
        output = new_row

    output.to_csv(HISTORY_PATH, index=False)

    print("\n=== MODEL RUN LOGGED ===")
    print(f"Saved to: {HISTORY_PATH}")