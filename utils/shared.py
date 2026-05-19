"""
Shared utilities for all analytics pages.
Extracted from streamlit_app.py.
"""
import os
import pandas as pd
import streamlit as st
from pathlib import Path
import numpy as np

DATA_PATH = Path("data/scored_macro_credit_data.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df



_ANALYTICS_VIEWS = {
    "Credit Markets": [
        ("Overview", "ov_cm"),
        ("Defaults", 19), ("Fwd Sim", 20), ("CDX Proxy", 21),
        ("Default Cycle", 24), ("Spread Vol", 27), ("Fallen Angel", 28),
        ("Global Credit", 29), ("Corp Leverage", 30), ("Seasonality", 31),
        ("Quality Migration", 40), ("Loan Market", 42), ("Credit Basis", 48),
        ("Issuance", 60), ("Distressed", 61), ("CLO", 62),
        ("CDS PD", 73), ("EM Credit", 77), ("Carry Breakeven", 78),
        ("Credit Momentum", 87), ("Quality Curve", 89),
        ("Spread Percentile", 100), ("Credit Cycle", 102),
        ("HY Momentum", 105), ("Credit Regime", 110),
        ("Credit Risk Score", 117), ("HY Lead-Lag", 130),
        ("Spread Vol Regime", 136), ("Regime Spread Dist", 141),
        ("Credit Beta", 146), ("HY Multi-Horizon", 149),
        ("Spread Dispersion", 151), ("Carry Decomp", 156),
        ("Credit Cycle Clock", 157), ("Spread Velocity", 158),
        ("Risk Appetite", 163), ("Carry Efficiency", 164),
        ("HY Term Structure", 172), ("Credit Diffusion", 174),
        ("BBB Cliff Risk", 175), ("Banking Flows", 183), ("Default Prob", 185),
        ("Spread Decomp", 187), ("Corp Health", 188), ("Primary & Loan", 191),
    ],
    "Rates & Macro": [
        ("Overview", "ov_rm"),
        ("Regime Returns", 23), ("X-Asset Momentum", 38), ("Macro Surprise", 41),
        ("Inflation Regime", 45), ("Fed Liquidity", 53), ("G4 Divergence", 54),
        ("Swap Spreads", 57), ("XCcy Basis", 58), ("FCI", 63),
        ("Credit Impulse", 64), ("Term Premium", 68), ("Curve Butterfly", 69),
        ("SLOOS", 70), ("Corp Profits", 72),
        ("Recession Model", 74), ("Real Rates", 75),
        ("Fed Sentiment", 81), ("Taylor Rule", 84), ("Macro Nowcast", 85),
        ("Term Structure", 90), ("NFCI", 92), ("Sahm Rule", 93),
        ("Curve Velocity", 103), ("Curve Regime", 104), ("Real Yield Z", 106),
        ("Unemp Momentum", 109), ("NFCI Trend", 111), ("Labor Warning", 114),
        ("Treasury Score", 118), ("Spread Changes", 121),
        ("Breakeven Inflation", 122), ("10y Yield", 125),
        ("Macro-Credit Corr", 138), ("Credit Impulse Drill", 142),
        ("Sahm Episodes", 143), ("Real Yield Episodes", 145),
        ("Credit-Labor Div", 152), ("Liquidity-Credit", 161),
        ("Initial Claims", 176), ("Oil-Credit", 177), ("FX-Credit", 181), ("Fed Balance Sheet", 182),
        ("Yield Monitor", 189),
    ],
    "Risk Monitors": [
        ("Overview", "ov_risk"),
        ("Tail Risk", 6), ("Stress Test", 7), ("Contagion", 12),
        ("Vol Regime", 39), ("Deleveraging", 44), ("Sector Stress", 46),
        ("Put/Call", 47), ("Tail Dependency", 52), ("Portfolio Stress", 55),
        ("AT1/CoCo", 56), ("CRE Stress", 59), ("ETF Dislocation", 65),
        ("Sovereign Contagion", 66), ("Consumer Credit", 67), ("ETF Flows", 71),
        ("MOVE Index", 76), ("VIX Term Structure", 79),
        ("Options Skew", 80), ("DV01", 82), ("CVaR", 83),
        ("VRP", 86), ("Funding Stress", 88),
        ("Market Internals", 95), ("Vol-Credit Mismatch", 96), ("Drawdown", 97),
        ("EQ-Credit Divergence", 98), ("Shock Monitor", 99),
        ("Liquidity Score", 107), ("VIX Momentum", 112), ("FX/Commodity", 113),
        ("Enhanced Funding", 119), ("Equity Returns", 124),
        ("VIX Context", 127), ("Drawdown Anatomy", 133), ("Alert History", 139),
        ("Vol of Vol", 148), ("Corr Regime", 154),
        ("Tail Skew", 167), ("Vol Clustering", 170),
        ("STLFSI", 178), ("MOVE-VIX Ratio", 179), ("X-Asset Corr", 180), ("ANFCI Suite", 186),
        ("ETF Signals", 190),
    ],
    "Signal Lab": [
        ("Overview", "ov_sl"),
        ("Validation", 1), ("Attribution", 2), ("Timeline", 3),
        ("Signal Decay", 4), ("Orthogonality", 5), ("Factors", 9),
        ("Granger Causality", 18), ("EQ-Credit Corr", 22), ("Correlation Heatmap", 26),
        ("PCA", 35), ("Composite", 37), ("Signal Movement", 50),
        ("Data Diagnostics", 91), ("Score Decomposition", 94),
        ("Score Velocity", 101), ("X-Asset Divergence", 108),
        ("Complacency Score", 115), ("Macro Risk Score", 116),
        ("Mean Reversion", 120), ("Score Consensus", 123),
        ("Composite History", 126), ("Score Correlations", 128),
        ("Score Z-Scores", 129), ("Score Persistence", 132),
        ("Score Distributions", 135), ("Score Gradient", 137),
        ("Forward Returns", 140), ("Score Momentum", 144),
        ("Seasonality", 147), ("Score Ensemble", 150),
        ("Score Bootstrap CI", 153), ("Score Calendar", 159),
        ("Factor Corr Scan", 160), ("Score Lead-Lag", 162),
        ("Score Drawdown", 165), ("Factor Decomp", 169),
        ("Score Inflection", 173), ("Fwd Return Cal", 184),
    ],
    "Regime": [
        ("Overview", "ov_reg"),
        ("Performance", 8), ("Regime Validity", 10), ("Failure Analysis", 11),
        ("Analogs", 13), ("Persistence", 14), ("Compare Dates", 25),
        ("Traffic Light", 32), ("Shock Simulation", 33), ("Alert Backtest", 34),
        ("Regime Forecast", 36), ("Regime Age", 43), ("Drawdown", 49),
        ("Recession Analog", 131), ("Transition Matrix", 134),
        ("Regime Dwell", 155), ("Macro Scorecard", 166),
        ("Quant Analogs", 168), ("Stress Probability", 171),
    ],
    "Models": [
        ("Overview", "ov_mod"),
        ("Sensitivity", "m1"), ("Transitions", "m2"), ("Regimes", "m3"),
        ("Monte Carlo", "m4"), ("Sub-period", "m5"), ("Sizing", "m6"),
        ("Scenarios", "m7"), ("OOS Splits", "m8"), ("Thresholds", "m9"),
        ("Merton DD", 15), ("Efficient Frontier", 16), ("Kelly Sizing", 17),
        ("Risk Parity", 51), ("Model Confidence", 192),
    ],
}
_ANALYTICS_SECTIONS = list(_ANALYTICS_VIEWS.keys())
_CORE_SECTIONS = ["Dashboard", "Credit View", "Signal", "Charts", "Portfolio", "Backtest", "History", "Allocation", "Health"]

_ANALYTICS_SECTIONS = list(_ANALYTICS_VIEWS.keys())
_CORE_SECTIONS = ["Dashboard", "Credit View", "Signal", "Charts", "Portfolio", "Backtest", "History", "Allocation", "Health"]

# ── UI helpers & view metadata ────────────────────────────────────────────────
def _sig_badge(pct, invert=False):
    """Colored HTML dot for percentile signal badges."""
    if invert:
        c = "#22c55e" if pct > 66 else ("#f59e0b" if pct > 33 else "#ef4444")
    else:
        c = "#ef4444" if pct > 66 else ("#f59e0b" if pct > 33 else "#22c55e")
    return f'<span style="color:{c}">&#9679;</span>'

def _pct_clr(pct, invert=False):
    """Return hex color based on percentile rank (for colored text)."""
    if invert:
        return "#22c55e" if pct > 66 else ("#f59e0b" if pct > 33 else "#ef4444")
    return "#ef4444" if pct > 66 else ("#f59e0b" if pct > 33 else "#22c55e")

_VIEW_DESC = {
    # Signal Lab
    1:   "Walk-forward validation — does the signal hold on truly unseen data?",
    2:   "Factor attribution — which inputs contribute most to today's score?",
    4:   "Signal decay — how quickly does predictive power erode over time?",
    5:   "Factor orthogonality — are input signals redundant or independent?",
    9:   "Factor loadings — PCA-based decomposition of the score's factor structure",
    18:  "Granger causality — which macro variables statistically lead the credit score?",
    22:  "EQ-credit correlation — is equity market behavior aligned with credit signals?",
    26:  "Correlation heatmap — pairwise correlations across all input factors",
    35:  "PCA — how much variance do the principal components explain?",
    94:  "Score decomposition — how does each sub-score contribute to the composite?",
    101: "Score velocity — is the composite score accelerating or decelerating?",
    120: "Mean reversion — is the current score at an extreme likely to revert?",
    123: "Score consensus — do sub-scores agree or diverge?",
    128: "Score correlations — how correlated are the sub-scores with each other?",
    132: "Score persistence — how autocorrelated is the composite score?",
    140: "Forward returns — empirical return distribution for the current score level",
    144: "Score momentum — is the trend in the score itself a signal?",
    153: "Score bootstrap CI — statistical confidence interval for today's reading",
    159: "Score calendar heatmap — daily/monthly score patterns across the calendar year",
    160: "Factor correlation scan — are input factors becoming redundantly correlated?",
    162: "Score lead-lag map — which factors lead vs. lag the composite score?",
    165: "Score drawdown profile — depth and duration of historical score deteriorations",
    169: "Factor decomp — Shapley-value attribution: which factors drive today's score?",
    173: "Score inflection points — historical score turning points and their macro context",
    184: "Forward return calibration — average HY spread change after signals like today's",
    # Risk Monitors
    6:   "Tail risk — are fat-tail credit outcomes becoming more probable?",
    7:   "Stress test — what happens to the portfolio under historical shock scenarios?",
    39:  "Vol regime — which volatility regime (low/medium/high) are we in?",
    52:  "Tail dependency — how correlated are tails across risk factors?",
    65:  "ETF dislocation — are credit ETF prices diverging from NAV?",
    71:  "ETF flows — are fund inflows or outflows signaling directional stress?",
    76:  "MOVE index — bond market implied volatility vs. historical stress levels",
    79:  "VIX term structure — is equity vol inverted (near-term > long-term)?",
    80:  "Options skew — is the market paying up for downside protection?",
    83:  "CVaR — conditional value-at-risk: expected loss in the worst tail scenarios",
    86:  "VRP — volatility risk premium: is implied vol cheap or expensive vs. realized?",
    88:  "Funding stress — is short-term funding market under pressure (TED/OIS)?",
    96:  "Vol-credit mismatch — is equity vol diverging from credit spread levels?",
    97:  "Drawdown — current and historical portfolio drawdown anatomy",
    99:  "Shock monitor — dashboard of active tail-risk flags across all factors",
    107: "Liquidity score — composite signal of market and funding liquidity conditions",
    112: "VIX momentum — is fear momentum building or subsiding?",
    154: "Correlation regime — have cross-factor correlations shifted to a stress pattern?",
    167: "Tail skew monitor — is the credit return distribution skewed to the downside?",
    170: "Vol regime clustering — are high-vol days clustering? A regime-change signal.",
    178: "STLFSI — St. Louis Fed Financial Stress Index vs. historical stress thresholds",
    179: "MOVE-VIX ratio — is bond market fear or equity fear leading right now?",
    180: "Cross-asset correlation — have correlation shifts preceded historical stress events?",
    186: "ANFCI suite — is financial tightness above what economic conditions justify?",
    190: "ETF flow & dislocation — are ETF dislocations and flows signaling forced selling?",
    # Credit Markets
    19:  "Default cycle — where are we in the HY default rate cycle?",
    24:  "Credit cycle stage — which of the four credit cycle phases are we in?",
    27:  "Spread volatility — is HY spread variance itself elevated?",
    28:  "Fallen angel risk — how many BBB issuers are at risk of downgrade to HY?",
    48:  "Credit basis — what is the basis between CDS and cash bond spreads?",
    61:  "Distressed ratio — what fraction of HY bonds are trading at distressed levels?",
    62:  "CLO market — what do CLO spreads signal about structured credit conditions?",
    78:  "Carry breakeven — how much spread widening can you absorb before carry is lost?",
    87:  "Credit momentum — is HY spread trend momentum positive or negative?",
    100: "Spread percentile — how stretched are current spreads vs. full history?",
    102: "Credit cycle — 4-phase cycle position: expansion, slowdown, contraction, recovery",
    105: "HY momentum — HY spread momentum across multiple timeframes",
    110: "Credit regime — is HY in a tightening or widening spread regime?",
    117: "Credit risk score — composite credit stress signal: current reading and context",
    130: "HY lead-lag — which factors lead HY spread moves?",
    149: "HY multi-horizon — how does spread momentum compare across 1m, 3m, 6m horizons?",
    151: "Spread dispersion — is stress widening broadly or concentrated in pockets?",
    155: "Regime dwell time — how long do credit regimes typically persist once established?",
    156: "Carry decomp — how much of HY carry is roll-down vs. pure credit risk premium?",
    157: "Credit cycle clock — 4-phase cycle position and likely transition direction",
    158: "Spread velocity — how fast are spreads moving? Velocity as an early stress signal.",
    163: "Risk appetite index — composite signal of market risk-on vs. risk-off stance",
    164: "Carry efficiency — is credit carry compensating for realized volatility?",
    172: "HY term structure — is near-term HY spread stress elevated vs. the long end?",
    174: "Credit diffusion index — how broadly is stress spreading across the credit market?",
    175: "BBB cliff risk — fallen angel risk from BBB-rated corporates and HY index impact",
    183: "Banking flow monitor — deposit and loan trends as early credit crunch signals",
    185: "Default probability monitor — model-estimated default probability vs. historical norms",
    187: "Spread decomposition — how much of HY spread is expected loss vs. risk premium?",
    188: "Corporate health suite — leverage and profit cycle: fundamental credit quality drivers",
    191: "Primary & loan market — is the new issuance market open and functioning normally?",
    # Rates & Macro
    41:  "Macro surprise — are macro data prints surprising to the upside or downside?",
    45:  "Inflation regime — which inflation environment are we in?",
    53:  "Fed liquidity — how does Fed policy stance affect financial conditions?",
    63:  "Financial conditions index — how tight or loose are overall financial conditions?",
    64:  "Credit impulse — is the credit impulse accelerating or decelerating?",
    68:  "Term premium — what does the term premium signal about recession risk?",
    70:  "SLOOS — what are banks signaling about lending standards and loan demand?",
    72:  "Corporate profits — is the profit cycle supporting or undermining credit quality?",
    74:  "Recession model — statistical probability of recession in 12 months",
    75:  "Real rates — what do real (inflation-adjusted) yields imply for credit?",
    93:  "Sahm rule — has the real-time recession indicator triggered?",
    103: "Curve velocity — how fast is the yield curve shape changing?",
    104: "Curve regime — which yield curve regime (inverted/flat/steep) are we in?",
    106: "Real yield Z-score — how extreme are current real yields vs. history?",
    109: "Unemployment momentum — is unemployment accelerating toward credit-stress territory?",
    122: "Breakeven inflation — what are inflation expectations implying for real rates?",
    125: "10y yield — where is the 10y Treasury in its historical range and trend?",
    142: "Credit impulse drill — detailed breakdown of credit impulse components",
    143: "Sahm episodes — historical Sahm rule triggers and their credit market outcomes",
    145: "Real yield episodes — current real yields vs. past stress episodes",
    146: "Credit beta — how sensitive is credit to equity and rate moves right now?",
    152: "Credit-labor divergence — is the labor market diverging from credit conditions?",
    161: "Liquidity-credit nexus — is illiquidity premium or fundamentals driving spreads?",
    176: "Initial claims — what do jobless claims signal for credit conditions?",
    177: "Oil-credit nexus — how energy market stress feeds into HY and sector spreads",
    181: "FX-credit nexus — does USD strength or weakness amplify credit stress?",
    182: "Fed balance sheet — is QE or QT the dominant driver of current credit spreads?",
    189: "Absolute yield monitor — HY and IG yield levels, carry over cash, and real yield",
    # Regime
    8:   "Regime performance — how has the portfolio performed in each credit regime?",
    10:  "Regime validity — are the defined regimes statistically distinct?",
    11:  "Failure analysis — when and why does the regime signal fail?",
    13:  "Regime analogs — which historical periods most resemble the current regime?",
    25:  "Compare dates — compare current conditions to any selected historical date",
    32:  "Traffic light — simplified red/amber/green signal dashboard",
    36:  "Regime forecast — model probability of being in each regime next month",
    43:  "Regime age — how old is the current regime and what does that imply?",
    131: "Recession analog — how do current conditions compare to past recessions?",
    134: "Transition matrix — probability of moving between regimes in the next period",
    155: "Regime dwell time — historical regime duration distribution",
    166: "Macro scorecard — how many macro indicators are currently flashing risk?",
    168: "Quant analogs — which past periods are most quantitatively similar to today?",
    171: "Stress probability — forward probability of transitioning to a stress regime",
    # Models
    15:  "Merton model — distance-to-default derived from equity market data",
    16:  "Efficient frontier — optimal credit/duration allocation across regimes",
    17:  "Kelly sizing — Kelly-criterion position sizing based on signal strength",
    51:  "Risk parity — volatility-weighted allocation across credit and rate factors",
    192: "Model confidence — how much should we trust the model's current reading?",
    "m1": "Sensitivity analysis — how do outputs change under parameter perturbations?",
    "m2": "Regime transitions — statistical transition matrix between credit regimes",
    "m3": "Regime structure — regime clustering and separation analysis",
    "m4": "Monte Carlo simulation — forward spread distribution under model uncertainty",
    "m5": "Sub-period stability — does the model perform consistently across time periods?",
    "m6": "Position sizing — how should position size scale with signal strength?",
    "m7": "Scenario analysis — outputs under user-defined macro scenarios",
    "m8": "Out-of-sample splits — forward performance across expanding/rolling OOS windows",
    "m9": "Threshold robustness — how sensitive are regime calls to score threshold choices?",
}

_VIEW_INSIGHT = {
    184: (
        "**What it measures:** Pre-computed forward HY spread changes (21d, 63d, 126d) binned by the "
        "composite score's historical decile. This is the most direct measure of signal validity — it shows "
        "what actually happened after similar score readings in the past.\n\n"
        "**How to read it:** D1 = lowest 10% of score readings (least stressed); D10 = highest (most stressed). "
        "If the signal works, D9-D10 should show positive (widening) forward changes. "
        "The 63d horizon is the most stable — 21d has high noise. "
        "Wide confidence intervals at a given decile indicate the signal was regime-dependent.\n\n"
        "**Current reading:** Find today's decile from the header percentile. "
        "If today is D7+ and the mean 63d forward change at that decile is >+30 bps, "
        "historical conditions favor widening. If mean change is near zero, the signal lacks directional clarity."
    ),
    187: (
        "**What it measures:** The HY spread split into (1) **Expected Loss** — the spread compensating "
        "for statistically expected credit defaults, and (2) **Excess Spread** — the risk premium above expected loss.\n\n"
        "**How to read it:** Compensation ratio >1.0x means investors are paid above actuarial expected losses "
        "(spreads are 'cheap'). Ratio <0.9x means credit is 'rich'. "
        "Most historical compression cycles (2021, pre-GFC) were driven by excess spread collapsing while "
        "expected loss stayed stable — the stacked area chart reveals this.\n\n"
        "**Current reading:** Excess spread percentile >70th = strong cushion vs. fair value. "
        "<30th historically precedes subsequent underperformance as the premium mean-reverts. "
        "The valuation summary box at the bottom gives the quick verdict."
    ),
    165: (
        "**What it measures:** Historical drawdown profile of the composite credit risk score — "
        "how deep scores have fallen from peaks and how long it took to recover.\n\n"
        "**How to read it:** Compare today's level to the depth/duration of past episodes. "
        "A score at a new high watermark is a different signal than the same level mid-drawdown. "
        "The 'underwater' chart (time below prior peak) is most useful for regime dating — "
        "extended underwater periods typically coincide with credit stress cycles.\n\n"
        "**Current reading:** If the score is near a local peak and accelerating, check for overshoot risk. "
        "If it's recovering from a deep trough, check the slope: V-shaped recoveries vs. slow grinds "
        "have different implications for spread trajectory."
    ),
    157: (
        "**What it measures:** Position on a 4-phase credit cycle (Expansion → Slowdown → Contraction → "
        "Recovery) based on spread levels, momentum, volatility, and macro conditions.\n\n"
        "**How to read it:** The clock shows current position and historical trajectory. "
        "Expansion→Slowdown transitions are the most actionable signal — "
        "spread momentum typically turns negative 30-60 days before full widening. "
        "Recovery→Expansion transitions are historically the best entry point for credit risk.\n\n"
        "**Current reading:** Both position AND rotation speed matter. "
        "Fast clockwise movement = rapid deterioration. Stalling or counter-clockwise = regime uncertainty. "
        "Cross-check with Stress Probability (sub171) and Score Inflection Points (sub173) for confirmation."
    ),
    171: (
        "**What it measures:** Forward probability of transitioning to a 'stress' credit regime "
        "within 30 and 90 days, derived from a regime-switching model using spread momentum, "
        "volatility, and macro inputs.\n\n"
        "**How to read it:** Probability >40% at the 30d horizon has historically preceded significant "
        "credit widening. The model is not a market-implied probability — it's a statistical model. "
        "Compare against realized signal history in the chart.\n\n"
        "**Current reading:** High stress probability + elevated Spread Velocity (sub158) = "
        "transition may already be in progress. Low stress probability + stable spreads = stay the course. "
        "Confidence degrades when Model Confidence (sub192) is low — check it alongside this view."
    ),
    192: (
        "**What it measures:** A meta-signal for the model itself: how reliable is the composite "
        "reading right now? Confidence is low when: (1) input factors conflict, "
        "(2) the current environment has few historical analogs, or (3) regime uncertainty is high.\n\n"
        "**How to read it:** High confidence + high risk score = strong signal to reduce exposure. "
        "High confidence + low risk = safe to maintain. "
        "**Low confidence + any score = directional guidance only** — widen stop-losses, "
        "reduce sizing, and cross-check with ANFCI (sub186) and Stress Probability (sub171).\n\n"
        "**Current reading:** The score-by-confidence-regime table shows how HY spread has historically "
        "behaved in each combination. If confidence is in the bottom quartile, treat the model as "
        "one input among several rather than the primary signal."
    ),
    181: (
        "**What it measures:** EUR/USD and USD/JPY versus HY spread — testing whether FX is "
        "an independent predictor of credit stress or endogenous to the same risk-off shock.\n\n"
        "**How to read it:** EUR/USD falling (USD strengthening) historically leads HY widening by "
        "15-30 days. USD/JPY falling (yen strengthening) is a particularly strong risk-off signal "
        "as carry trades unwind. The rolling 90d correlation chart shows when the FX-credit "
        "linkage is active vs. dormant.\n\n"
        "**Current reading:** A correlation spike from near-zero to -0.5+ is a regime shift worth acting on. "
        "The scatter regression slope quantifies how many bps of HY widening historically accompany "
        "a 1% FX move — multiply by today's FX momentum for a rough spread impact estimate."
    ),
    186: (
        "**What it measures:** ANFCI (Adjusted NFCI) removes financial tightness explained by current "
        "economic activity, isolating the pure risk premium component. "
        "A high ANFCI means markets are tight beyond what fundamentals justify.\n\n"
        "**How to read it:** The NFCI - ANFCI gap (shaded area) is the 'excess tightness' — "
        "positive = markets pricing in more stress than the economy warrants. "
        "This gap historically mean-reverts and is a contrarian signal when extreme.\n\n"
        "**Current reading:** Banking stress + CLO stress both elevated simultaneously is the most "
        "dangerous combination — stress spreading from market finance into the banking system. "
        "Low excess spread percentile (thin cushion) + elevated ANFCI is the highest systemic risk flag."
    ),
    188: (
        "**What it measures:** Two fundamental credit quality drivers: "
        "(1) **Corporate leverage score** — how levered are corporate balance sheets vs. history, "
        "and (2) **Corporate profit cycle** — is the profit cycle in expansion or contraction?\n\n"
        "**How to read it:** The quadrant scatter is the key visual. "
        "Bottom-right (high leverage / low profits) is the worst combination — historically coincides "
        "with peak default cycles. Top-left (low leverage / high profits) is most supportive. "
        "Transitions FROM top-left TO bottom-right typically take 12-18 months and are visible in the scatter.\n\n"
        "**Current reading:** Check the quadrant label at the bottom and the *direction* of quarterly movement "
        "on the scatter. 'Mixed signals' often precedes a decisive shift — watch the trajectory, not just the level."
    ),
    185: (
        "**What it measures:** Three complementary default risk measures: "
        "(1) model-based default probability, (2) default cycle position score, "
        "(3) CDS-implied probability of default from swap market pricing.\n\n"
        "**How to read it:** Divergence between model default probability and CDS-implied PD is informative: "
        "CDS >> model = markets pricing more risk than fundamentals warrant (potential mean-reversion buy). "
        "Model >> CDS = market complacency despite deteriorating fundamentals (warning signal).\n\n"
        "**Current reading:** The 80th percentile threshold line marks the historical level above which "
        "defaults subsequently elevated. Current position relative to this threshold is the key binary signal. "
        "Compare to Corporate Health (sub188) for the fundamental driver context."
    ),
    163: (
        "**What it measures:** Composite risk appetite index constructed from cross-asset momentum, "
        "vol surface shape, credit positioning, and FX carry — all compressed into a single risk-on/risk-off reading.\n\n"
        "**How to read it:** The index is normalized 0-1 where 1 = maximum risk appetite. "
        "Readings >0.7 historically precede spread tightening; <0.3 precede widening. "
        "Transition speed matters as much as level — a rapid drop from 0.8 to 0.4 is more bearish "
        "than a slow grind from 0.6 to 0.4.\n\n"
        "**Current reading:** If risk appetite is high but the credit risk score is also high, "
        "something is inconsistent — one will likely revert. Credit scores tend to win over 2-3 month horizons. "
        "Use the rolling correlation chart to see whether the linkage to HY is currently active."
    ),
    175: (
        "**What it measures:** Risk that BBB-rated bonds get downgraded to HY (fallen angels), "
        "forcing index-constrained investors to sell and amplifying spread widening.\n\n"
        "**How to read it:** The BBB/IG spread ratio is the key signal — BBB widening much more than "
        "overall IG signals stress in the lowest investment-grade tier. "
        "Historical fallen angel waves (2002, 2015-16 energy, 2020) all showed sharp ratio spikes "
        "6-12 months before the bulk of downgrades.\n\n"
        "**Current reading:** BBB cliff risk + Credit Diffusion (sub174) both elevated = systemic "
        "downgrade cycle beginning. Either signal alone is less reliable. "
        "Cross-check Corporate Health (sub188) for leverage confirmation."
    ),
    172: (
        "**What it measures:** HY spread term structure — near-term vs. long-dated spread expectations, "
        "analogous to yield curve inversion but in credit spread space.\n\n"
        "**How to read it:** Inverted HY term structure (near > long) signals acute near-term stress "
        "that may not be priced as a permanent shift — historically associated with technical/liquidity events. "
        "Upward-sloping (long > near) signals markets expect stress to build over time — more fundamental.\n\n"
        "**Current reading:** An inversion in the 90th+ percentile has historically been a mean-reversion "
        "buy signal in HY — but only confirm with fundamentals (sub188) and default probability (sub185) "
        "before acting. Inversions driven by equity panic rather than credit fundamentals resolve faster."
    ),
    174: (
        "**What it measures:** The breadth of credit stress across sectors — are spreads widening broadly "
        "(contagion) or only in specific pockets (idiosyncratic)?\n\n"
        "**How to read it:** High diffusion index (>70) means stress is broad-based across sectors — "
        "historically associated with systematic credit risk and larger subsequent drawdowns. "
        "Low diffusion with overall elevated HY spread = sector-specific stress that may not escalate.\n\n"
        "**Current reading:** Watch for diffusion rising from low to high as an early signal that "
        "idiosyncratic stress is becoming systemic. The combination of rising diffusion + rising BBB cliff risk "
        "(sub175) is the most reliable precursor to a fallen angel cycle."
    ),
}




def _err_track(sub_id, err):
    """Record view errors to session state for health-check visibility."""
    if "_view_errors" not in st.session_state:
        st.session_state["_view_errors"] = {}
    st.session_state["_view_errors"][str(sub_id)] = str(err)
