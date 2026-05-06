# Credit Risk Dashboard

A Python-based macro, credit, liquidity, and market-risk regime engine.

This project pulls public market and macro data, builds risk signals, scores current market conditions, validates historical forward returns, and produces a daily credit/market risk report.

## Purpose

The goal is to create a practical credit and markets-focused dashboard that helps answer:

- Are macro conditions deteriorating?
- Are credit markets pricing stress or complacency?
- Is risk appetite stretched?
- Are markets setting up for mean reversion?
- What is the current portfolio stance?
- What historical regimes look similar?

## Current Features

- FRED data pulls
- Yield curve regime classification
- Macro risk scoring
- Credit market risk scoring
- Liquidity regime scoring
- Treasury stress scoring
- Complacency scoring
- Mean-reversion scoring
- Risk appetite scoring
- Cross-asset divergence scoring
- Signal attribution
- Scenario shock testing
- Historical analog comparison
- Portfolio stance output
- Model health checks
- Forward-return validation
- Strategy backtest summary
- Daily signal report export
- Model run history logging

## Data Sources

Current public data sources include:

- FRED Treasury yields
- FRED unemployment data
- Chicago Fed NFCI
- ICE BofA High Yield spreads
- S&P 500 index level
- VIX index
- 10-year breakeven inflation

Planned future sources:

- WRDS
- Compustat / Capital IQ
- CRSP
- TRACE
- Bank regulatory data
- Cboe options data

## Project Structure

```text
credit-risk-dashboard/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── scored_macro_credit_data.csv
├── history/
│   └── model_run_history.csv
├── outputs/
│   ├── charts/
│   └── reports/
│       └── latest_signal_report.txt
└── src/
    ├── backtester.py
    ├── crisis_similarity.py
    ├── fred_loader.py
    ├── liquidity_engine.py
    ├── market_internals_engine.py
    ├── model_health_check.py
    ├── portfolio_engine.py
    ├── risk_engine.py
    ├── run_logger.py
    ├── scenario_engine.py
    ├── signal_attribution.py
    ├── signal_registry.py
    ├── treasury_engine.py
    ├── utils.py
    └── validation_engine.py
