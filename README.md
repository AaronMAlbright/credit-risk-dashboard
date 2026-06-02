# Macro Credit Risk Dashboard

An institutional-style macro-credit regime framework built with public market and macro data.

The project classifies the credit environment, decomposes credit risk into economic channels, evaluates spread compensation versus expected loss, and translates regimes into portfolio actions such as HY/IG tilt, quality bias, duration stance, liquidity posture, and hedge posture.

## What This Is

This is a tactical credit risk and credit strategy dashboard. It is designed to answer:

- Are macro and liquidity conditions supportive or deteriorating?
- Are credit spreads compensating investors for expected default loss?
- Is HY cheap or rich versus IG and its own history?
- Which credit-risk channel is driving the current regime?
- What happened historically after similar regimes?
- What portfolio stance is implied: add beta, hold, upgrade quality, or de-risk?

The framework is not presented as a black-box alpha model. It is a transparent risk overlay and credit strategy tool.

## Core Framework

The institutional credit framework is organized around six channels:

1. **Macro Cycle**: growth, labor, recession, and macro deterioration risk.
2. **Rates And Liquidity**: real rates, curve stress, funding stress, Fed liquidity, and financial conditions.
3. **Credit Market**: HY/IG spread level, spread momentum, spread volatility, distressed pressure, and fallen-angel risk.
4. **Credit Fundamentals**: leverage, profit cycle, lending standards, default-cycle risk, and implied default probability.
5. **Market Technicals**: ETF flows, primary issuance, ETF dislocations, CLO/loan stress, and liquidity pressure.
6. **Cross-Asset Confirmation**: equities, volatility, banks, FX, commodities, sovereign stress, and market internals.

Each channel produces a score, contribution, coverage measure, and one-month change. The channel framework runs alongside the existing production composite so model behavior can be compared before replacing the legacy score.

## Credit-Specific Analytics

The dashboard includes:

- **Spread decomposition**: OAS, expected loss, excess spread, compensation ratio.
- **Relative value**: HY percentile, IG percentile, HY/IG ratio, BBB/IG ratio, excess-spread percentile.
- **Credit market tear sheet**: level, percentile, 1M/3M changes, valuation bucket, and action.
- **Rating bucket proxy view**: IG, BBB, HY, and distressed-quality proxy regimes.
- **Refinancing wall framework**: placeholder structure for issuer or index maturity-bucket data.
- **Positioning playbook**: credit beta, HY tilt, IG tilt, quality bias, duration, liquidity, and hedge posture.
- **Regime performance**: forward S&P returns and HY/IG spread changes by regime and horizon.
- **Strategy memo and credit brief**: downloadable markdown outputs for review or interview demos.

See `docs/credit_compensation_scorecard.md` for the PM-facing methodology and workflow behind the credit compensation scorecard, rating-bucket allocation, expected return/stress framework, marginal allocation advice, spread shock sensitivity, net spread beta, CDX hedge sizing, audit trail, validation replay, and PM review packet.

## Outputs

Main generated artifacts:

- `data/scored_macro_credit_data.csv`
- `outputs/reports/latest_signal_report.txt`
- `outputs/reports/credit_brief.md`
- `outputs/reports/credit_market_tearsheet.md`
- `outputs/charts/`

The Streamlit app includes a dedicated **Credit View** with the brief, memo, positioning table, spread compensation, relative value, channel attribution, rating-bucket proxy view, refinancing wall framework, and regime performance.

## Data Sources

The project currently relies on public data sources and proxies:

- FRED Treasury, unemployment, financial conditions, inflation, and credit series.
- ICE/BofA public spread series where available through FRED.
- yfinance ETF/index proxies for selected market technical and cross-asset modules.
- Committed historical scored dataset for offline testing.

Public proxies are explicitly treated as proxies. A production-grade institutional version should use bond-level index data, TRACE liquidity, issuer fundamentals, rating-bucket OAS, sector OAS, CDX HY/IG, default data, and maturity schedules.

See `docs/data_dictionary.md` for the source registry, quality labels, transforms, and limitations used by the dashboard.

## Validation

Validation modules include:

- forward return and spread-move tables by regime,
- signal decay,
- walk-forward and frozen split analysis,
- tail risk,
- regime validity,
- factor exposure,
- score orthogonality,
- threshold robustness,
- stress episode behavior.

Results include sample-size and confidence context. Thin-regime observations should be treated as directional, not conclusive.

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Run the batch scoring pipeline:

```bash
python app.py
```

Run focused finance tests:

```bash
pytest tests/test_credit_framework.py tests/test_credit_regime_performance.py tests/test_credit_relative_value.py tests/test_credit_presentation.py tests/test_credit_tearsheet.py -q
```

## Interview Framing

A concise project explanation:

> I built a macro-credit regime framework that decomposes credit risk into macro cycle, rates/liquidity, credit-market, fundamentals, technicals, and cross-asset channels. It evaluates whether spreads compensate investors after expected default loss, validates regimes against forward spread and equity behavior, and maps the result to implementable portfolio actions such as HY/IG tilt, quality bias, duration, liquidity, and hedging.

## Known Limitations

- Public ETF and spread series are proxies, not a substitute for institutional bond-level data.
- Expected-loss calculations currently use regime-level PD assumptions and a simplified recovery assumption.
- The refinancing wall module is a framework placeholder until maturity-bucket data are provided.
- The legacy composite remains the production decision score; the institutional channel composite runs alongside it.
- The Streamlit app is feature-rich but still monolithic and should eventually be split into pages/components.

## Governance

See `docs/model_governance.md`.

The important boundary is that the legacy composite remains the production score today. The institutional credit framework is a parallel explanatory layer until composite disagreement periods and forward outcomes are reviewed.
