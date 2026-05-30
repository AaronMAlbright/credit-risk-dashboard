# Macro Credit Risk Framework Methodology

## Objective

This project is a macro-credit regime framework. Its purpose is to translate public macro, rates, liquidity, credit-market, and cross-asset data into:

- a current credit risk regime,
- the dominant drivers of that regime,
- regime-conditioned forward-return evidence,
- and practical portfolio implications.

The intended user is a credit strategist, portfolio risk analyst, or allocator who wants a disciplined daily read on whether credit beta is being compensated.

## Institutional Framing

The model should be explained as a channel model, not a collection of indicators. Each signal belongs to one of six economic risk channels:

1. **Macro Cycle Risk**
   Measures whether growth and labor conditions are deteriorating.

2. **Rates And Liquidity Risk**
   Measures whether real rates, yield-curve dynamics, funding stress, and financial conditions are tightening.

3. **Credit Market Risk**
   Measures spread levels, spread momentum, credit impulse, and spread-volatility stress.

4. **Credit Fundamentals**
   Measures leverage, profit-cycle risk, default-cycle risk, fallen angel risk, and lending-standard pressure.

5. **Market Technicals**
   Measures flows, issuance, ETF dislocations, liquidity, and positioning stress.

6. **Cross-Asset Confirmation**
   Measures whether equities, volatility, banks, FX, commodities, and rates confirm or contradict credit.

The composite risk score should be defensible as a weighted blend of these channels, with each channel exposing its current score, percentile rank, recent change, and contribution to the total.

## Spread Decomposition

Credit spreads should be tied to expected loss and risk compensation:

```text
Expected Loss Spread = Default Probability x Loss Given Default
Loss Given Default = 1 - Recovery Rate
Excess Spread = OAS - Expected Loss Spread
```

For high yield, a simple starting point is:

- benign/default-cycle PD: 2-3%
- neutral PD: 4-5%
- stressed PD: 7-10%
- crisis PD: 12%+
- recovery rate: 35-45%

The model should classify spreads as attractive only when compensation remains strong after expected losses, not simply because nominal spreads are wide.

See `docs/credit_compensation_scorecard.md` for the PM-facing scorecard methodology, including rating-bucket allocation, bucket expected return, historical analog blending, marginal allocation advice, and spread shock sensitivity.

## Portfolio Interpretation

Regime labels should map to implementable credit actions:

- **Risk-On:** overweight credit beta, prefer HY over IG, moderate cash, neutral duration.
- **Neutral:** balanced IG/HY exposure, avoid aggressive beta timing.
- **Caution:** reduce HY and CCC risk, prefer IG quality, raise cash, avoid weak liquidity names.
- **Risk-Off:** protect capital, minimize HY beta, prefer cash/Treasuries/high-quality IG, wait for stabilization.

Credit recommendations should explicitly mention quality, spread beta, duration, liquidity, and hedging where relevant.

## Validation Standard

A professional-grade version should report:

- in-sample and out-of-sample forward returns by regime,
- 1M/3M/6M signal decay,
- spread widening probability by regime,
- drawdown probability by regime,
- recession and non-recession splits,
- threshold robustness,
- false-positive and false-negative episodes,
- sample sizes and confidence flags.

No regime conclusion should be shown without sample size and confidence context.

## Known Limitations

- Public ETF proxies are not a substitute for full bond-level index data.
- Public spread series do not fully separate liquidity premium, default premium, and risk premium.
- Thresholds are heuristic unless validated by walk-forward calibration.
- Macro data are revised and released with lags.
- Some yfinance-derived series are best treated as tactical proxies, not official market data.

## Interview Narrative

A concise explanation:

> I built a macro-credit regime framework that decomposes credit risk into macro cycle, liquidity, spread, fundamentals, market technicals, and cross-asset confirmation channels. The model maps each channel to a composite regime score, validates regimes against forward returns and spread moves, and translates the result into credit portfolio actions such as HY/IG tilt, cash level, quality bias, and duration stance.
