"""
Portfolio Stress Test.

User defines a portfolio (IG%, HY%, Loans%, EM%, Cash%) and this module
applies the 5 shock scenarios from the shock simulator to compute dollar
P&L per $100 invested.

All P&L values are expressed as % of notional (e.g., -3.5 = -3.5% per $100).
Portfolio P&L = sum(weight * asset_pl) across asset classes.

Public API
----------
  DEFAULT_PORTFOLIO    — dict of default weights
  ASSET_LABELS         — human-readable asset class names
  SHOCK_SCENARIOS      — 5 pre-defined shock scenario definitions
  compute_asset_pl(asset, shocks) -> float
  stress_test_portfolio(portfolio, scenario_name, shocks) -> dict
  run_portfolio_stress_test(df, portfolio=None) -> dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Duration and sensitivity constants
# ---------------------------------------------------------------------------

# Duration assumptions (years)
IG_DURATION    = 7.0
HY_DURATION    = 4.0
LOAN_DURATION  = 0.25   # floating rate — minimal rate sensitivity
EM_DURATION    = 5.5
CASH_DURATION  = 0.0

# HY equity beta (partial equity-like return component)
HY_EQUITY_BETA = 0.3

# IG-spread-to-HY-spread beta (IG tightens/widens ~40% as much as HY)
IG_HY_BETA = 0.4

# Loan spread sensitivity relative to HY bonds (loans less spread-sensitive)
LOAN_SPREAD_BETA = 0.6

# EM spread multiplier vs HY (EM typically wider / more volatile)
EM_SPREAD_MULT = 1.2

# EM FX sensitivity (dollar impact per unit DXY move)
# A rising DXY (USD strengthens) hurts EM dollar-debt issuers
EM_DXY_BETA = -0.5

# ---------------------------------------------------------------------------
# Portfolio defaults
# ---------------------------------------------------------------------------

DEFAULT_PORTFOLIO: dict[str, float] = {
    "ig":    0.40,
    "hy":    0.30,
    "loans": 0.10,
    "em":    0.10,
    "cash":  0.10,
}

ASSET_LABELS: dict[str, str] = {
    "ig":    "Investment Grade",
    "hy":    "High Yield",
    "loans": "Leveraged Loans",
    "em":    "EM Credit",
    "cash":  "Cash",
}

# ---------------------------------------------------------------------------
# Shock scenarios (mirrors shock_simulator.get_default_shocks())
# ---------------------------------------------------------------------------

SHOCK_SCENARIOS: dict[str, dict[str, float]] = {
    "Mild Stress":   {"hy_spread": 50,   "vix": 5,  "sp500": -0.05},
    "Credit Crunch": {"hy_spread": 200,  "vix": 25, "sp500": -0.15},
    "Rate Shock":    {"yield_10y": 1.0,  "yield_3m": 0.5},
    "Risk-Off":      {"hy_spread": 300,  "vix": 40, "sp500": -0.20, "yield_10y": -0.5},
    "Stagflation":   {"yield_10y": 1.5,  "vix": 20, "hy_spread": 150, "sp500": -0.10},
}

# ---------------------------------------------------------------------------
# Regime → portfolio context mapping
# ---------------------------------------------------------------------------

_REGIME_CONTEXT: dict[str, str] = {
    "Risk-On":  (
        "Risk-On regime: HY and EM credit are well supported. "
        "Rate sensitivity is the primary tail risk. "
        "Leverage loans benefit from floating-rate protection."
    ),
    "Neutral":  (
        "Neutral regime: balanced risk/reward. "
        "Spread widening and rate vol are moderate risks. "
        "IG acts as a partial buffer."
    ),
    "Caution":  (
        "Caution regime: spread widening risk elevated. "
        "HY and EM positions are most vulnerable. "
        "Consider reducing high-beta exposure."
    ),
    "Risk-Off": (
        "Risk-Off regime: significant spread widening likely. "
        "HY and EM face the largest drawdowns. "
        "Cash and IG offer relative safety; loans are somewhat insulated by floating rate."
    ),
}

_REGIME_CONTEXT_DEFAULT = (
    "Regime unavailable — stress test results based on mechanical sensitivities only."
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_shock(shocks: dict, key: str, default: float = 0.0) -> float:
    """Return shocks[key] as float, defaulting to 0 if absent or non-numeric."""
    val = shocks.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Public: compute_asset_pl
# ---------------------------------------------------------------------------

def compute_asset_pl(asset: str, shocks: dict) -> float:
    """Compute P&L (% of notional) for a single asset given market shocks.

    Parameters
    ----------
    asset  : one of "ig", "hy", "loans", "em", "cash"
    shocks : dict with keys from {hy_spread, ig_spread, yield_10y,
             yield_3m, sp500, dxy, vix}. Yield changes in %, spread
             changes in bps, sp500 as fraction, dxy in index points.

    Returns
    -------
    float — e.g., -3.5 means a 3.5% loss per $100 invested.
    """
    try:
        d_yield_10y = _safe_shock(shocks, "yield_10y")
        d_hy_spread = _safe_shock(shocks, "hy_spread")
        d_sp500     = _safe_shock(shocks, "sp500")
        d_dxy       = _safe_shock(shocks, "dxy")

        # IG spread shock: use explicit ig_spread if provided, else 40% of HY move
        if "ig_spread" in shocks:
            d_ig_spread = _safe_shock(shocks, "ig_spread")
        else:
            d_ig_spread = IG_HY_BETA * d_hy_spread

        asset = asset.lower()

        if asset == "ig":
            # Rate: -duration * d_yield (yield up = price down)
            # Spread: -dv01 * d_spread, where dv01 = duration/100 (per bp of spread)
            pl = (
                -IG_DURATION * d_yield_10y
                - (IG_DURATION / 100.0) * d_ig_spread
            )

        elif asset == "hy":
            # Rate + spread + partial equity beta
            pl = (
                -HY_DURATION * d_yield_10y
                - (HY_DURATION / 100.0) * d_hy_spread
                + d_sp500 * HY_EQUITY_BETA * 100.0  # d_sp500 is fraction → convert to %
            )

        elif asset == "loans":
            # Floating rate: minimal duration; spread sensitivity dampened
            pl = -(LOAN_DURATION / 100.0) * d_hy_spread * LOAN_SPREAD_BETA

        elif asset == "em":
            # Rate + spread (elevated sensitivity) + FX transmission
            dxy_effect = EM_DXY_BETA * d_dxy if d_dxy != 0.0 else 0.0
            pl = (
                -EM_DURATION * d_yield_10y
                - (EM_DURATION / 100.0) * d_hy_spread * EM_SPREAD_MULT
                + dxy_effect
            )

        elif asset == "cash":
            pl = 0.0

        else:
            # Unknown asset — treat like equities (pure sp500 beta)
            pl = d_sp500 * 100.0

        return round(float(pl), 4)

    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Public: stress_test_portfolio
# ---------------------------------------------------------------------------

def stress_test_portfolio(
    portfolio: dict,
    scenario_name: str,
    shocks: dict,
) -> dict:
    """Apply one shock scenario to a portfolio and compute P&L breakdown.

    Parameters
    ----------
    portfolio     : dict mapping asset key → weight (weights should sum to 1.0)
    scenario_name : str label for this scenario
    shocks        : dict of market variable → delta value

    Returns
    -------
    dict:
        scenario             : str
        portfolio_pl         : float — total weighted P&L %
        asset_pls            : dict  — {asset: pl_pct}
        asset_contributions  : dict  — {asset: weight * pl_pct}
        worst_asset          : str
        best_asset           : str
    """
    try:
        asset_pls: dict[str, float] = {}
        asset_contributions: dict[str, float] = {}

        total_weight = sum(float(w) for w in portfolio.values() if w is not None)
        if total_weight <= 0:
            total_weight = 1.0

        for asset, weight in portfolio.items():
            try:
                w = float(weight)
            except (TypeError, ValueError):
                w = 0.0

            pl = compute_asset_pl(asset, shocks)
            asset_pls[asset] = pl
            asset_contributions[asset] = round(w * pl, 4)

        portfolio_pl = round(sum(asset_contributions.values()), 4)

        # Find best and worst asset by raw P&L (unweighted)
        if asset_pls:
            worst_asset = min(asset_pls, key=lambda a: asset_pls[a])
            best_asset  = max(asset_pls, key=lambda a: asset_pls[a])
        else:
            worst_asset = "N/A"
            best_asset  = "N/A"

        return {
            "scenario":            scenario_name,
            "portfolio_pl":        portfolio_pl,
            "asset_pls":           {k: round(v, 4) for k, v in asset_pls.items()},
            "asset_contributions": asset_contributions,
            "worst_asset":         worst_asset,
            "best_asset":          best_asset,
        }
    except Exception:
        return {
            "scenario":            scenario_name,
            "portfolio_pl":        0.0,
            "asset_pls":           {},
            "asset_contributions": {},
            "worst_asset":         "N/A",
            "best_asset":          "N/A",
        }


# ---------------------------------------------------------------------------
# Public: run_portfolio_stress_test
# ---------------------------------------------------------------------------

def run_portfolio_stress_test(
    df: pd.DataFrame,
    portfolio: dict | None = None,
) -> dict:
    """Top-level portfolio stress test entry point.

    Parameters
    ----------
    df        : Master DataFrame with DatetimeIndex. Used only to read the
                current regime from df["final_decision"] if available.
    portfolio : dict mapping asset key → weight. If None, DEFAULT_PORTFOLIO
                is used.

    Returns
    -------
    dict:
        available          : bool — always True (purely computational)
        portfolio_used     : dict
        scenarios          : dict — {scenario_name: stress_test_portfolio result}
        summary_table      : pd.DataFrame — scenarios × (assets + "Total")
        worst_scenario     : str
        worst_scenario_pl  : float
        best_scenario      : str
        best_scenario_pl   : float
        current_regime     : str
        regime_context     : str
        interpretation     : str
    """
    try:
        if portfolio is None or not portfolio:
            portfolio = DEFAULT_PORTFOLIO.copy()

        # Normalise weights to sum to 1
        total_w = sum(float(w) for w in portfolio.values() if w is not None)
        if total_w > 0:
            portfolio_used = {k: round(float(v) / total_w, 6) for k, v in portfolio.items()}
        else:
            portfolio_used = DEFAULT_PORTFOLIO.copy()

        # Run every scenario
        scenarios: dict[str, dict] = {}
        for name, shocks in SHOCK_SCENARIOS.items():
            scenarios[name] = stress_test_portfolio(portfolio_used, name, shocks)

        # Build summary table: rows=scenarios, cols=assets + Total
        all_assets = list(portfolio_used.keys())
        rows = []
        for name, result in scenarios.items():
            row: dict[str, object] = {"Scenario": name}
            for asset in all_assets:
                label = ASSET_LABELS.get(asset, asset.upper())
                row[label] = round(result["asset_pls"].get(asset, 0.0), 2)
            row["Portfolio Total"] = round(result["portfolio_pl"], 2)
            rows.append(row)
        summary_table = pd.DataFrame(rows).set_index("Scenario")

        # Best/worst overall scenario
        scenario_pls = {name: r["portfolio_pl"] for name, r in scenarios.items()}
        if scenario_pls:
            worst_scenario = min(scenario_pls, key=lambda n: scenario_pls[n])
            best_scenario  = max(scenario_pls, key=lambda n: scenario_pls[n])
            worst_scenario_pl = scenario_pls[worst_scenario]
            best_scenario_pl  = scenario_pls[best_scenario]
        else:
            worst_scenario    = "N/A"
            worst_scenario_pl = 0.0
            best_scenario     = "N/A"
            best_scenario_pl  = 0.0

        # Current regime from df
        current_regime = "Unknown"
        if df is not None and not df.empty and "final_decision" in df.columns:
            try:
                last_valid = df["final_decision"].dropna()
                if not last_valid.empty:
                    current_regime = str(last_valid.iloc[-1])
            except Exception:
                pass

        regime_context = _REGIME_CONTEXT.get(current_regime, _REGIME_CONTEXT_DEFAULT)

        # Portfolio weights summary for interpretation
        weight_strs = [
            f"{ASSET_LABELS.get(k, k.upper())} {v * 100:.0f}%"
            for k, v in portfolio_used.items()
            if v > 0
        ]
        weights_desc = ", ".join(weight_strs)

        interpretation = (
            f"Portfolio ({weights_desc}) under {len(scenarios)} shock scenarios: "
            f"worst case is '{worst_scenario}' at {worst_scenario_pl:+.1f}%, "
            f"best case is '{best_scenario}' at {best_scenario_pl:+.1f}% per $100 invested. "
            f"Current regime: {current_regime}. {regime_context}"
        )

        return {
            "available":          True,
            "portfolio_used":     portfolio_used,
            "scenarios":          scenarios,
            "summary_table":      summary_table,
            "worst_scenario":     worst_scenario,
            "worst_scenario_pl":  worst_scenario_pl,
            "best_scenario":      best_scenario,
            "best_scenario_pl":   best_scenario_pl,
            "current_regime":     current_regime,
            "regime_context":     regime_context,
            "interpretation":     interpretation,
        }
    except Exception:
        return {"available": False}
