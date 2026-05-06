import pandas as pd


# =====================
# Basic Regime Classifiers
# =====================
def classify_yield_curve_regime(spread):
    if spread > 1:
        return "Steep (Expansion)"
    elif spread > 0:
        return "Normal"
    elif spread > -0.5:
        return "Flat / Inversion Risk"
    return "Inverted (Recession Risk)"


def classify_credit_regime(spread, unemployment):
    if spread < 0 and unemployment > 5:
        return "Credit Stress"
    elif spread < 0:
        return "Late Cycle"
    elif spread > 1 and unemployment < 4:
        return "Expansion"
    return "Neutral"


def classify_labor_warning(sahm_like, unemployment_change):
    if pd.notna(sahm_like) and sahm_like >= 0.5:
        return "Triggered"

    if (
        (pd.notna(sahm_like) and sahm_like >= 0.3)
        or (pd.notna(unemployment_change) and unemployment_change > 0.2)
    ):
        return "Watch"

    return "Low"


# =====================
# Risk Scores
# =====================
def compute_macro_risk_score(
    spread,
    unemployment,
    nfci,
    unemployment_change,
    spread_change,
    nfci_change,
    sahm_like
):
    score = 0

    score += max(0, min(30, (0.75 - spread) * 20))

    if unemployment > 6:
        score += 30
    elif unemployment > 5:
        score += 15

    if pd.notna(unemployment_change) and unemployment_change > 0:
        score += min(25, unemployment_change * 120)

    if pd.notna(sahm_like) and sahm_like > 0.5:
        score += 30
    elif pd.notna(sahm_like) and sahm_like > 0.3:
        score += 15

    score += max(0, min(35, nfci * 55))

    if pd.notna(nfci_change) and nfci_change > 0.1:
        score += 20
    elif pd.notna(nfci_change) and nfci_change > 0:
        score += min(15, nfci_change * 80)

    if pd.notna(spread_change) and spread_change > 0.5 and spread < 0.75:
        score += 10

    return round(max(0, min(score, 100)), 1)


def compute_credit_market_risk_score(
    hy_spread,
    hy_change_30d,
    hy_change_90d,
    credit_impulse,
    vix,
    vix_change_30d,
    credit_equity_divergence,
    vol_credit_mismatch
):
    score = 0

    if hy_spread > 6:
        score += 35
    elif hy_spread > 4:
        score += 22
    elif hy_spread > 3.5:
        score += 12
    elif hy_spread < 3:
        score += 5

    if pd.notna(hy_change_30d) and hy_change_30d > 0.5:
        score += 30
    elif pd.notna(hy_change_30d) and hy_change_30d > 0.25:
        score += 20
    elif pd.notna(hy_change_30d) and hy_change_30d > 0:
        score += 10

    if pd.notna(hy_change_90d) and hy_change_90d > 0.75:
        score += 25
    elif pd.notna(hy_change_90d) and hy_change_90d > 0.35:
        score += 15

    if pd.notna(credit_impulse) and credit_impulse > 0.4:
        score += 25
    elif pd.notna(credit_impulse) and credit_impulse > 0.15:
        score += 15

    if vix > 35:
        score += 30
    elif vix > 25:
        score += 20
    elif vix > 20:
        score += 10

    if pd.notna(vix_change_30d) and vix_change_30d > 5:
        score += 20
    elif pd.notna(vix_change_30d) and vix_change_30d > 2:
        score += 10

    if credit_equity_divergence == "Hidden Credit Stress":
        score += 20
    elif credit_equity_divergence == "Risk-On Confirmation":
        score -= 5

    if vol_credit_mismatch == "Credit Warning / Vol Complacency":
        score += 20
    elif vol_credit_mismatch == "Recovery Confirmation":
        score -= 10

    return round(max(0, min(score, 100)), 1)


def compute_liquidity_score(nfci, nfci_change_90d, hy_change_30d, vix_change_30d):
    score = 0

    if nfci > 0.5:
        score += 35
    elif nfci > 0:
        score += 20
    elif nfci < -0.5:
        score -= 5

    if pd.notna(nfci_change_90d) and nfci_change_90d > 0.1:
        score += 25
    elif pd.notna(nfci_change_90d) and nfci_change_90d > 0:
        score += 10

    if pd.notna(hy_change_30d) and hy_change_30d > 0.25:
        score += 25
    elif pd.notna(hy_change_30d) and hy_change_30d > 0:
        score += 10

    if pd.notna(vix_change_30d) and vix_change_30d > 5:
        score += 20
    elif pd.notna(vix_change_30d) and vix_change_30d > 2:
        score += 10

    return round(max(0, min(score, 100)), 1)


def compute_risk_appetite_score(
    hy_change_30d,
    vix_change_30d,
    sp500_drawdown,
    macro_risk_momentum,
    credit_risk_momentum
):
    score = 0

    if pd.notna(hy_change_30d) and hy_change_30d < -0.3:
        score += 25
    elif pd.notna(hy_change_30d) and hy_change_30d < 0:
        score += 12

    if pd.notna(vix_change_30d) and vix_change_30d < -5:
        score += 25
    elif pd.notna(vix_change_30d) and vix_change_30d < 0:
        score += 10

    if sp500_drawdown > -0.02:
        score += 25
    elif sp500_drawdown > -0.05:
        score += 12

    if pd.notna(macro_risk_momentum) and macro_risk_momentum < -10:
        score += 15
    elif pd.notna(macro_risk_momentum) and macro_risk_momentum < 0:
        score += 8

    if pd.notna(credit_risk_momentum) and credit_risk_momentum < -10:
        score += 15
    elif pd.notna(credit_risk_momentum) and credit_risk_momentum < 0:
        score += 8

    return round(max(0, min(score, 100)), 1)


def compute_complacency_score(
    vix,
    hy_spread,
    sp500_drawdown,
    macro_risk_momentum,
    nfci,
    hy_change_30d,
    risk_appetite_score,
    market_internals_score
):
    score = 0

    if vix < 14:
        score += 20
    elif vix < 18:
        score += 15
    elif vix < 22:
        score += 8

    if hy_spread < 2.75:
        score += 25
    elif hy_spread < 3:
        score += 18
    elif hy_spread < 3.5:
        score += 10

    if sp500_drawdown > -0.02:
        score += 15
    elif sp500_drawdown > -0.05:
        score += 8

    if pd.notna(macro_risk_momentum) and macro_risk_momentum < -10:
        score += 15
    elif pd.notna(macro_risk_momentum) and macro_risk_momentum < -5:
        score += 8

    if nfci < -0.5:
        score += 12
    elif nfci < 0:
        score += 8

    if pd.notna(hy_change_30d) and hy_change_30d < -0.3:
        score += 12
    elif pd.notna(hy_change_30d) and hy_change_30d < 0:
        score += 5

    if risk_appetite_score > 80:
        score += 15
    elif risk_appetite_score > 60:
        score += 8

    if market_internals_score > 70:
        score += 10

    return round(max(0, min(score, 100)), 1)


def compute_mean_reversion_score(
    macro_risk_score,
    credit_market_risk_score,
    hy_spread,
    hy_change_30d,
    vix,
    vix_change_30d,
    sp500_drawdown
):
    score = 0
    combined_stress = max(macro_risk_score, credit_market_risk_score)

    if combined_stress > 70:
        score += 35
    elif combined_stress > 50:
        score += 25
    elif combined_stress > 40:
        score += 15

    if hy_spread > 5:
        score += 25
    elif hy_spread > 4:
        score += 15

    if pd.notna(hy_change_30d) and hy_change_30d > 0.5:
        score += 25
    elif pd.notna(hy_change_30d) and hy_change_30d > 0.2:
        score += 15

    if vix > 35:
        score += 25
    elif vix > 25:
        score += 15

    if pd.notna(vix_change_30d) and vix_change_30d > 5:
        score += 15

    if sp500_drawdown < -0.15:
        score += 25
    elif sp500_drawdown < -0.10:
        score += 15
    elif sp500_drawdown < -0.05:
        score += 10

    return round(max(0, min(score, 100)), 1)


# =====================
# Divergence / Shock Logic
# =====================
def classify_credit_equity_divergence(sp500_return_30d, hy_change_30d):
    if pd.isna(sp500_return_30d) or pd.isna(hy_change_30d):
        return "Unknown"

    if sp500_return_30d > 0.02 and hy_change_30d > 0.15:
        return "Hidden Credit Stress"

    if sp500_return_30d > 0 and hy_change_30d < 0:
        return "Risk-On Confirmation"

    if sp500_return_30d < -0.02 and hy_change_30d <= 0:
        return "Equity-Led Weakness"

    if sp500_return_30d < 0 and hy_change_30d > 0:
        return "Broad Risk-Off"

    return "Neutral"


def classify_vol_credit_mismatch(vix, vix_change_30d, hy_change_30d):
    if pd.isna(vix) or pd.isna(vix_change_30d) or pd.isna(hy_change_30d):
        return "Unknown"

    if vix < 18 and hy_change_30d > 0.15:
        return "Credit Warning / Vol Complacency"

    if vix > 25 and hy_change_30d < 0:
        return "Recovery Confirmation"

    if vix_change_30d < -5 and hy_change_30d > 0:
        return "Vol-Credit Divergence"

    return "Neutral"


def detect_shock(vix_change_5d, hy_change_5d, sp500_return_5d, spread_change_5d):
    shock_flags = []

    if pd.notna(vix_change_5d) and vix_change_5d > 5:
        shock_flags.append("VIX spike")

    if pd.notna(hy_change_5d) and hy_change_5d > 0.25:
        shock_flags.append("HY spread widening")

    if pd.notna(sp500_return_5d) and sp500_return_5d < -0.03:
        shock_flags.append("Equity selloff")

    if pd.notna(spread_change_5d) and abs(spread_change_5d) > 0.25:
        shock_flags.append("Yield curve shock")

    return "No Shock" if len(shock_flags) == 0 else " | ".join(shock_flags)


# =====================
# Labels / Signals
# =====================
def classify_score(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Elevated"
    elif score >= 20:
        return "Moderate"
    return "Low"


def classify_risk_score(score):
    if score >= 70:
        return "High Stress"
    elif score >= 40:
        return "Elevated"
    elif score >= 20:
        return "Moderate"
    return "Low"


def classify_complacency_score(score):
    if score >= 90:
        return "Extreme Complacency"
    elif score >= 70:
        return "Elevated Complacency"
    elif score >= 40:
        return "Moderate Complacency"
    return "Low Complacency"


def classify_mean_reversion_score(score):
    if score >= 70:
        return "Strong Panic-Buy Setup"
    elif score >= 40:
        return "Moderate Mean-Reversion Setup"
    elif score >= 20:
        return "Weak Setup"
    return "No Setup"


def generate_macro_signal(score):
    if score >= 70:
        return "DEFENSIVE"
    elif score >= 50:
        return "DE-RISKING"
    elif score >= 30:
        return "NEUTRAL"
    return "RISK ON"


def generate_mean_reversion_signal(score):
    if score >= 70:
        return "BUY STRESS"
    elif score >= 40:
        return "WATCH ENTRY"
    elif score >= 20:
        return "WAIT"
    return "NO TRADE"


def classify_transition_regime(macro_score, credit_score, momentum):
    combined_score = max(macro_score, credit_score)

    if pd.isna(momentum):
        return "Unknown"

    if combined_score >= 50 and momentum > 0:
        return "Deteriorating / Rising Stress"
    elif combined_score >= 50 and momentum < 0:
        return "Panic / Stabilizing"
    elif combined_score < 30 and momentum < 0:
        return "Complacency / Late Cycle"
    elif combined_score < 30 and momentum > 0:
        return "Early Deterioration"

    return "Stable / Neutral"


def generate_transition_signal(transition_regime):
    if transition_regime == "Deteriorating / Rising Stress":
        return "REDUCE RISK"
    elif transition_regime == "Panic / Stabilizing":
        return "WATCH FOR RECOVERY"
    elif transition_regime == "Complacency / Late Cycle":
        return "REDUCE RISK"
    elif transition_regime == "Early Deterioration":
        return "MONITOR"
    return "NEUTRAL"


# =====================
# Decision / Drivers
# =====================
def generate_final_decision(
    macro_score,
    credit_score,
    liquidity_regime_score,
    cross_asset_divergence_score,
    complacency_score,
    mean_reversion_score,
    transition_regime,
    shock_flag,
    credit_equity_divergence,
    vol_credit_mismatch,
    composite_score=0,
):
    max_stress = max(macro_score, credit_score, liquidity_regime_score)

    if shock_flag != "No Shock" and max_stress >= 40:
        return {
            "final_decision": "Stress / Stabilization Watch",
            "environment": "Active deterioration or shock",
            "action": "Reduce risk and monitor for stabilization",
            "buy_trigger": "Wait for VIX rollover, HY spread stabilization, and equity drawdown exhaustion",
            "risk_off_trigger": "Already triggered",
        }

    if mean_reversion_score >= 70:
        return {
            "final_decision": "Buy Stress",
            "environment": "Stress with mean-reversion opportunity",
            "action": "Consider phased risk entry",
            "buy_trigger": "VIX rolling over, HY spreads stabilizing, SP500 drawdown washed out",
            "risk_off_trigger": "New HY widening or VIX acceleration",
        }

    if mean_reversion_score >= 40:
        return {
            "final_decision": "Watch Entry",
            "environment": "Stress building but entry setup incomplete",
            "action": "Watch for entry, do not chase yet",
            "buy_trigger": "VIX > 25, SP500 drawdown > 5%, HY spread stabilization",
            "risk_off_trigger": "HY 5D widening > 0.25 and VIX +5",
        }

    if cross_asset_divergence_score >= 40:
        return {
            "final_decision": "Divergence Warning",
            "environment": "Cross-asset signals are diverging",
            "action": "Avoid adding risk until credit, vol, and equities confirm",
            "buy_trigger": "Divergence fades and credit stabilizes",
            "risk_off_trigger": "Divergence worsens with HY widening or VIX spike",
        }

    if credit_equity_divergence == "Hidden Credit Stress":
        return {
            "final_decision": "Credit Warning",
            "environment": "Equities firm but credit deteriorating",
            "action": "Avoid adding risk until credit confirms",
            "buy_trigger": "HY spreads stop widening and VIX remains contained",
            "risk_off_trigger": "HY widening continues while equities stay complacent",
        }

    if vol_credit_mismatch == "Credit Warning / Vol Complacency":
        return {
            "final_decision": "Credit Warning",
            "environment": "Low equity volatility but credit stress emerging",
            "action": "Reduce complacent risk exposure",
            "buy_trigger": "VIX reprices or credit stabilizes",
            "risk_off_trigger": "HY widening with VIX still suppressed",
        }

    if complacency_score >= 90:
        return {
            "final_decision": "Avoid Chasing Risk",
            "environment": "Stable but extremely crowded",
            "action": "Reduce new exposure / avoid chasing risk",
            "buy_trigger": "Wait for VIX spike, HY widening, or SP500 drawdown > 5%",
            "risk_off_trigger": "HY spread widening + VIX rising",
        }

    if complacency_score >= 70:
        return {
            "final_decision": "Hold / Do Not Chase",
            "environment": "Stable but crowded",
            "action": "Hold existing risk, avoid aggressive new risk",
            "buy_trigger": "Wait for better entry after volatility or credit repricing",
            "risk_off_trigger": "5D VIX spike or HY widening",
        }

    if transition_regime == "Complacency / Late Cycle":
        return {
            "final_decision": "Wait",
            "environment": "Low stress but late-cycle complacency",
            "action": "Do not chase; keep dry powder",
            "buy_trigger": "SP500 drawdown > 5% or VIX > 25",
            "risk_off_trigger": "Credit spreads widen while VIX rises",
        }

    if macro_score < 30 and credit_score < 30 and complacency_score < 40:
        if composite_score >= 50:
            return {
                "final_decision": "Wait",
                "environment": "Individual signals benign but composite risk elevated",
                "action": "Hold neutral — composite risk gauge signals caution",
                "buy_trigger": "Composite risk score falls below 50 and complacency resets",
                "risk_off_trigger": "Composite risk continues rising or shock flag triggers",
            }
        return {
            "final_decision": "Neutral",
            "environment": "Benign but not enough edge to force exposure",
            "action": "Maintain neutral risk posture",
            "buy_trigger": "Complacency reset or mean-reversion setup",
            "risk_off_trigger": "Complacency > 70 or shock flag",
        }

    if composite_score >= 50:
        return {
            "final_decision": "Wait",
            "environment": "No dominant signal but composite risk elevated",
            "action": "Neutral positioning with elevated background risk",
            "buy_trigger": "Composite risk score falls below 50",
            "risk_off_trigger": "Individual signal fires or composite risk continues rising",
        }

    return {
        "final_decision": "Neutral",
        "environment": "Mixed / neutral",
        "action": "Neutral positioning",
        "buy_trigger": "Wait for clearer mean-reversion or benign setup",
        "risk_off_trigger": "Shock flag or rising macro/credit stress",
    }


def get_signal_drivers(row):
    drivers = []

    if row["macro_risk_score_smooth"] >= 40:
        drivers.append("Macro risk elevated")

    if row["credit_market_risk_score_smooth"] >= 40:
        drivers.append("Credit market risk elevated")

    if row["liquidity_regime_score_smooth"] >= 40:
        drivers.append("Liquidity conditions tightening")

    if row["cross_asset_divergence_score_smooth"] >= 40:
        drivers.append("Cross-asset divergence elevated")

    if row["complacency_score_smooth"] >= 70:
        drivers.append("Crowded / complacent market setup")

    if row["mean_reversion_score_smooth"] >= 40:
        drivers.append("Mean-reversion setup developing")

    if row["hy_spread"] < 3:
        drivers.append("HY spreads tight")

    if row["hy_change_30d"] < -0.3:
        drivers.append("HY spreads tightening quickly")

    if row["hy_change_30d"] > 0.25:
        drivers.append("HY spreads widening")

    if row["vix"] < 18:
        drivers.append("VIX low")

    if row["vix_change_30d"] < -5:
        drivers.append("VIX falling sharply")

    if row["sp500_drawdown"] > -0.02:
        drivers.append("SP500 near highs")

    if row["treasury_stress_score_smooth"] > 40:
        drivers.append("Treasury / real-rate stress elevated")

    if row["credit_equity_divergence"] != "Neutral":
        drivers.append(f"Credit/equity signal: {row['credit_equity_divergence']}")

    if row["vol_credit_mismatch"] != "Neutral":
        drivers.append(f"Vol/credit signal: {row['vol_credit_mismatch']}")

    if row["shock_flag"] != "No Shock":
        drivers.append(f"Shock flag: {row['shock_flag']}")

    if len(drivers) == 0:
        drivers.append("No dominant risk driver")

    return drivers[:8]


def compute_model_confidence(row, regime_counts):
    score = 0
    reasons = []

    current_decision = row["final_decision"]
    obs = regime_counts.get(current_decision, 0)

    if obs >= 30:
        score += 30
        reasons.append("Adequate decision sample size")
    elif obs >= 10:
        score += 15
        reasons.append("Limited decision sample size")
    else:
        reasons.append("Weak decision sample size")

    if row["shock_flag"] == "No Shock":
        score += 20
        reasons.append("No active shock flag")
    else:
        score += 10
        reasons.append("Shock flag active")

    signal_alignment = 0

    if row["macro_risk_score_smooth"] < 30 and row["credit_market_risk_score_smooth"] < 30:
        signal_alignment += 1

    if row["complacency_score_smooth"] >= 70 and row["mean_reversion_score_smooth"] < 20:
        signal_alignment += 1

    if row["credit_equity_divergence"] in ["Risk-On Confirmation", "Neutral"]:
        signal_alignment += 1

    if row["liquidity_regime_score_smooth"] < 30:
        signal_alignment += 1

    if signal_alignment >= 3:
        score += 30
        reasons.append("Signals mostly aligned")
    elif signal_alignment >= 2:
        score += 20
        reasons.append("Signals partly aligned")
    else:
        reasons.append("Signals conflicting")

    if row["labor_warning"] == "Low":
        score += 20
        reasons.append("Labor warning low")
    elif row["labor_warning"] == "Watch":
        score += 10
        reasons.append("Labor warning watch")

    if score >= 75:
        label = "High"
    elif score >= 50:
        label = "Medium"
    else:
        label = "Low"

    return label, reasons


def calculate_trigger_distances(vix, sp500_drawdown, hy_change_5d):
    return {
        "vix_to_25": round(25 - vix, 2),
        "sp500_to_5pct_drawdown": round(max(0, 0.05 + sp500_drawdown), 4),
        "hy_5d_to_025_widening": (
            round(0.25 - hy_change_5d, 2)
            if pd.notna(hy_change_5d)
            else None
        ),
    }