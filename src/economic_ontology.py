"""
Economic ontology layer.

Maps each signal and sub-score to its economic interpretation.
Used as documentation and as a lookup table in the dashboard.

Public API
----------
  SIGNAL_ONTOLOGY   — list of signal metadata dicts
  get_ontology_df   — returns ontology as a DataFrame
"""

from __future__ import annotations
import pandas as pd

SIGNAL_ONTOLOGY: list[dict] = [
    {
        "signal":        "treasury_stress_score",
        "display":       "Treasury Stress",
        "in_composite":  True,
        "weight":        "20%",
        "category":      "Rates",
        "meaning":       "Real yield elevation, curve velocity (90d change), and MOVE bond implied vol",
        "mechanism":     "High real yields raise the equity discount rate; rapid rate moves create duration losses and force deleveraging; elevated bond vol signals funding uncertainty",
        "expected_lag":  "Leading — 21–63 trading days",
        "asset_impact":  "Risk-off: multiple compression, credit spread widening, duration selling",
        "sign_conv":     "High = more risk",
        "limitations":   "Persistently high real yields can coexist with strong equity in late-cycle; MOVE can spike without credit contagion (e.g. 2022 rate shock isolated to duration)",
    },
    {
        "signal":        "complacency_score",
        "display":       "Complacency",
        "in_composite":  True,
        "weight":        "20%",
        "category":      "Market Sentiment",
        "meaning":       "Suppressed VIX + negative VIX trend + compressed HY spreads relative to history — market pricing out tail risk",
        "mechanism":     "Low volatility enables risk-parity and levered strategies; negative VIX changes signal vol supply; complacency precedes vol regime shifts as carry strategies crowd",
        "expected_lag":  "Leading — 30–63 trading days before mean reversion",
        "asset_impact":  "Leading indicator of drawdown when sustained at extreme; high score = complacency, not safety",
        "sign_conv":     "High = more risk (high complacency = unpriced tail risk)",
        "limitations":   "Can remain elevated for 1–2 years (1997–1999, 2013–2015); timing of mean reversion is notoriously imprecise",
    },
    {
        "signal":        "credit_market_risk_score",
        "display":       "Credit Risk",
        "in_composite":  True,
        "weight":        "20%",
        "category":      "Credit",
        "meaning":       "HY OAS level + 30d/90d momentum + credit impulse + credit regime classification",
        "mechanism":     "HY spreads reflect aggregate default expectations and risk appetite; widening spreads signal reduced corporate credit availability and precede equity stress",
        "expected_lag":  "Concurrent — tight co-movement with equity drawdowns (5–21d lag)",
        "asset_impact":  "Concurrent risk indicator; spread widening leads equity corrections by 0–21 days",
        "sign_conv":     "High = more risk",
        "limitations":   "Concurrent rather than leading; spread moves reverse quickly (2020: widened and tightened within 60d); includes illiquidity premium that can disconnect from fundamentals",
    },
    {
        "signal":        "macro_risk_score",
        "display":       "Macro Risk",
        "in_composite":  True,
        "weight":        "15%",
        "category":      "Macro",
        "meaning":       "Unemployment trajectory (Sahm-like impulse) + NFCI tightening + labor market deterioration signals",
        "mechanism":     "Rising unemployment signals demand destruction and corporate earnings pressure; NFCI tightening reflects credit availability contraction; both reflect real economy damage",
        "expected_lag":  "Lagging — confirms recessions rather than predicting them (−30 to −90d vs market turning point)",
        "asset_impact":  "Regime duration signal — useful for confirming protracted stress, not for timing entry/exit",
        "sign_conv":     "High = more risk",
        "limitations":   "Unemployment is monthly and significantly lagged; NFCI is weekly but backward-looking; consistently the weakest predictive signal in multi-horizon analysis",
    },
    {
        "signal":        "liquidity_regime_score",
        "display":       "Liquidity",
        "in_composite":  True,
        "weight":        "10%",
        "category":      "Macro / Financial Conditions",
        "meaning":       "NFCI + ANFCI + STLFSI + Fed balance sheet change — aggregate financial conditions tightness",
        "mechanism":     "Tight financial conditions reduce credit availability, raise borrowing costs, and force risk asset de-grossing through leverage unwind",
        "expected_lag":  "Concurrent — co-moves with spread widening and equity drawdowns (10–30d lag)",
        "asset_impact":  "Risk indicator during tightening; becomes less useful as leading indicator at horizons > 126 days",
        "sign_conv":     "High = tighter / more risk",
        "limitations":   "FCI indices include equity prices, creating circularity; signal reverses direction at long horizons (>126d) — reduced weight from 0.15 to 0.10 after empirical review",
    },
    {
        "signal":        "enhanced_funding_stress_score",
        "display":       "Funding Stress",
        "in_composite":  True,
        "weight":        "10%",
        "category":      "Rates / Funding",
        "meaning":       "Rates term structure shape + funding spread proxies + Fed policy path acceleration",
        "mechanism":     "Funding stress in repo and money markets transmits to credit availability and margin calls; Fed path acceleration changes corporate financing costs and duration risk",
        "expected_lag":  "Medium-horizon leading — best predictive power at 42–189 trading days",
        "asset_impact":  "Useful for identifying pre-conditions for stress, not the onset; good 6–9 month horizon signal",
        "sign_conv":     "High = more stress",
        "limitations":   "Repo spread data is imputed; actual short-term funding markets are opaque; lead time is long enough that signal misses fast-onset crises",
    },
    {
        "signal":        "fx_commodity_score",
        "display":       "FX / Commodity",
        "in_composite":  True,
        "weight":        "5%",
        "category":      "Global",
        "meaning":       "USD strength + EURUSD/USDJPY trend + oil price decline + commodity stress composite",
        "mechanism":     "USD strength tightens global dollar funding conditions; oil declines signal demand contraction or supply shock; commodity stress reflects global growth slowdown",
        "expected_lag":  "Weak leading — consistent but small directional signal at all horizons",
        "asset_impact":  "Small incremental signal; 5% weight reflects limited marginal information beyond credit and treasury signals",
        "sign_conv":     "High = more stress",
        "limitations":   "USD can be risk-off haven (strengthens during flight-to-safety), creating directionality confusion; oil is also supply-driven (2014 oil collapse was supply, not demand)",
    },
    {
        "signal":        "banking_stress_score",
        "display":       "Banking Stress",
        "in_composite":  False,
        "weight":        "0% (excluded)",
        "category":      "Sector (Excluded)",
        "meaning":       "Bank deposit flows + loan growth + yield curve slope on banking profitability",
        "mechanism":     "Deposit outflows tighten bank funding; loan contraction signals credit demand collapse; inverted yield curve pressures net interest margin",
        "expected_lag":  "Lagging — confirms banking stress after the fact (−45 to −90d vs market peak)",
        "asset_impact":  "EXCLUDED from composite: persistently wrong-direction Spearman r at all forward horizons. Confirmed lagging indicator, not a predictor.",
        "sign_conv":     "High = more stress (but WRONG-SIGN forward correlation — excluded from composite)",
        "limitations":   "Deposit data is monthly and revised; loan growth strongly lagged; inverted curve has been wrong-sign signal repeatedly; removed after multi-horizon empirical analysis showed persistent wrong-direction correlation at all 6 test horizons (21d through 252d)",
    },
    {
        "signal":        "mean_reversion_score",
        "display":       "Mean Reversion",
        "in_composite":  False,
        "weight":        "0% (tracked only)",
        "category":      "Valuation / Supplemental",
        "meaning":       "Composite score distance from recent trend — regime overextension proxy",
        "mechanism":     "Extreme composite levels tend to revert; measures how far current stress level deviates from its own recent history",
        "expected_lag":  "Variable — reversal timing is imprecise",
        "asset_impact":  "Supplemental only; excluded from composite as redundant with Complacency signal",
        "sign_conv":     "High = overextended (could be either fear or complacency extremity)",
        "limitations":   "Excluded from composite due to redundancy with Complacency and lack of marginal predictive power; useful for identifying inflection points but not reliable enough to weight",
    },
    {
        "signal":        "cross_asset_divergence_score",
        "display":       "Cross-Asset Divergence",
        "in_composite":  False,
        "weight":        "0% (tracked only)",
        "category":      "Divergence / Supplemental",
        "meaning":       "SP500 vs HY credit vs VIX divergence — internal inconsistency across asset classes",
        "mechanism":     "When equity rises but credit spreads widen or VIX stays elevated, the cross-asset inconsistency signals that the rally lacks credit confirmation and may reverse",
        "expected_lag":  "Situational — divergences can persist for months before resolving",
        "asset_impact":  "Risk warning when elevated and credit-domain confirmed; resolution direction is uncertain",
        "sign_conv":     "High = more divergence / latent risk",
        "limitations":   "Divergences can persist indefinitely; resolution direction is ambiguous (credit can tighten to equity rather than equity falling); tracked as supplemental signal only",
    },
]


def get_ontology_df() -> pd.DataFrame:
    """Return SIGNAL_ONTOLOGY as a DataFrame indexed by signal name."""
    return pd.DataFrame(SIGNAL_ONTOLOGY).set_index("signal")
