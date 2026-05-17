"""
DV01 and spread sensitivity analysis.

DV01 (Dollar Value of a Basis Point) measures how much the portfolio
gains or loses when credit spreads tighten or widen by 1 basis point.

For a credit portfolio:
  DV01 = -portfolio_value × modified_duration × 0.0001

The negative sign captures the inverse relationship between spreads and prices:
when spreads widen (credit deteriorates), bond prices fall and the portfolio
loses value. When spreads tighten (credit improves), prices rise.

Spread sensitivity: if spreads widen by X bps, portfolio P&L =
  P&L ≈ -modified_duration × ΔSpread_decimal × portfolio_value

This is a first-order (linear) approximation. Convexity effects become material
for large spread moves (>100 bps) but are ignored here for simplicity.

Key credit durations (approximations based on index data, as of 2024):
  HY index:   ~4.0 years modified duration
    (shorter because HY bonds have higher coupons, shorter maturities, and
    more callable structures — calls cap duration when spreads tighten)
  IG index:   ~7.0 years modified duration
    (longer because IG bonds are typically 10–30 year maturities, lower
    coupons, and fewer calls)
  BBB:        ~6.5 years modified duration
    (slightly shorter than overall IG due to BBB issuers' preference for
    intermediate maturities and higher coupon structures)

Portfolio blended duration:
  When the regime engine weights HY and IG, the portfolio duration is the
  weighted average: D_blend = w_HY × D_HY + w_IG × D_IG.

  A risk-off regime (high ig_weight) will produce a longer portfolio duration,
  meaning LARGER dollar losses per basis point of spread widening — but IG
  spreads typically move less than HY spreads in stress, so the net P&L impact
  depends on the spread beta of each segment.

Public API
----------
  HY_DURATION, IG_DURATION   — assumed modified durations
  PORTFOLIO_VALUE             — $1,000,000 notional (configurable)
  compute_spread_sensitivity(df) → pd.DataFrame
  run_dv01_analysis(df, portfolio_value) → dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Assumed modified duration for the HY index (Bloomberg US Corporate HY).
#: Historically 3.5–4.5 years; 4.0 is a reasonable mid-cycle estimate.
HY_DURATION: float = 4.0

#: Assumed modified duration for the IG index (Bloomberg US Corporate IG).
#: Historically 6.5–8.0 years; 7.0 is a reasonable mid-cycle estimate.
IG_DURATION: float = 7.0

#: Default portfolio notional for DV01 calculations ($1,000,000).
#: This is just a reference size; run_dv01_analysis accepts an override.
PORTFOLIO_VALUE: float = 1_000_000.0


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def _get_weights(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Extract HY and IG portfolio weights from *df*.

    Falls back to 0.5/0.5 if either column is absent, so the module
    remains useful even without regime-engine output.

    Returns
    -------
    (hy_weight, ig_weight) as float Series aligned to df.index.
    """
    if "hy_weight" in df.columns and "ig_weight" in df.columns:
        hy_w = df["hy_weight"].astype(float).fillna(0.5)
        ig_w = df["ig_weight"].astype(float).fillna(0.5)
    elif "hy_weight" in df.columns:
        hy_w = df["hy_weight"].astype(float).fillna(0.5)
        ig_w = (1.0 - hy_w).clip(0, 1)
    elif "ig_weight" in df.columns:
        ig_w = df["ig_weight"].astype(float).fillna(0.5)
        hy_w = (1.0 - ig_w).clip(0, 1)
    else:
        hy_w = pd.Series(0.5, index=df.index)
        ig_w = pd.Series(0.5, index=df.index)
    return hy_w, ig_w


def _pnl(duration: float | pd.Series,
         spread_change_bps: float,
         portfolio_value: float) -> float | pd.Series:
    """
    First-order P&L approximation for a parallel spread shift.

    P&L = -D_mod × ΔSpread × portfolio_value
        where ΔSpread is in decimal (bps / 10_000).

    Positive spread_change_bps → spreads widen → prices fall → negative P&L.
    Negative spread_change_bps → spreads tighten → prices rise → positive P&L.
    """
    return -duration * (spread_change_bps / 10_000.0) * portfolio_value


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_spread_sensitivity(df: pd.DataFrame,
                                portfolio_value: float = PORTFOLIO_VALUE) -> pd.DataFrame:
    """
    Compute per-row DV01 and spread sensitivity for a blended credit portfolio.

    Uses the regime-engine weights (``hy_weight``, ``ig_weight``) to compute
    a blended portfolio modified duration, then derives DV01 and scenario P&Ls.

    Parameters
    ----------
    df : pd.DataFrame
        May contain ``hy_weight`` and ``ig_weight`` columns (0–1 floats).
        Falls back to 50/50 if absent.
    portfolio_value : float
        Notional portfolio size in dollars. Default $1,000,000.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with additional columns:

        blended_duration : float
            Weighted average modified duration of the portfolio.

        dv01_per_1m : float
            Dollar P&L per 1 bp spread move per $1M notional (always negative
            or zero — spread widening loses money). Formula:
            DV01 = −D_mod × 0.0001 × 1_000_000

        pnl_25bps_widen : float
            Estimated portfolio P&L (for *portfolio_value* notional) if all
            spreads widen 25 bps simultaneously.

        pnl_50bps_widen : float
            P&L for +50 bps parallel spread shift.

        pnl_100bps_widen : float
            P&L for +100 bps parallel spread shift (severe stress scenario).

        pnl_25bps_tighten : float
            P&L for −25 bps parallel spread shift (tightening scenario,
            always positive — portfolio gains as credit rallies).
    """
    out = df.copy()
    hy_w, ig_w = _get_weights(out)

    out["blended_duration"] = hy_w * HY_DURATION + ig_w * IG_DURATION

    # DV01 per $1M notional (always ≤ 0: widening = loss)
    out["dv01_per_1m"] = _pnl(out["blended_duration"], 1.0, 1_000_000.0)

    # Scenario P&Ls at the actual portfolio_value
    out["pnl_25bps_widen"]    = _pnl(out["blended_duration"],  25.0, portfolio_value)
    out["pnl_50bps_widen"]    = _pnl(out["blended_duration"],  50.0, portfolio_value)
    out["pnl_100bps_widen"]   = _pnl(out["blended_duration"], 100.0, portfolio_value)
    out["pnl_25bps_tighten"]  = _pnl(out["blended_duration"], -25.0, portfolio_value)

    return out


# ---------------------------------------------------------------------------
# Scenario table builder
# ---------------------------------------------------------------------------

def _build_scenario_table(blended_duration: float,
                           portfolio_value: float) -> pd.DataFrame:
    """
    Build a scenario P&L table for spread changes from −150 bps to +300 bps
    in 25 bp increments.

    Parameters
    ----------
    blended_duration : float
        Portfolio modified duration (years).
    portfolio_value : float
        Notional portfolio size in dollars.

    Returns
    -------
    pd.DataFrame with columns:
        spread_change_bps : int     — spread change in basis points
        pnl               : float  — estimated P&L in dollars
        pnl_pct           : float  — P&L as % of portfolio_value
    """
    steps = list(range(-150, 301, 25))
    rows = []
    for bps in steps:
        pnl = _pnl(blended_duration, float(bps), portfolio_value)
        rows.append({
            "spread_change_bps": bps,
            "pnl":               round(float(pnl), 2),
            "pnl_pct":           round(float(pnl) / portfolio_value * 100, 4)
                                  if portfolio_value != 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis runner
# ---------------------------------------------------------------------------

def run_dv01_analysis(df: pd.DataFrame,
                      portfolio_value: float = PORTFOLIO_VALUE) -> dict:
    """
    Run a full DV01 / spread sensitivity analysis and return a structured dict.

    Parameters
    ----------
    df : pd.DataFrame
        Historical panel. ``hy_weight`` and ``ig_weight`` columns are used if
        present; otherwise 50/50 split is assumed.
    portfolio_value : float
        Portfolio notional in dollars. Default $1,000,000.

    Returns
    -------
    dict with keys:

        df : pd.DataFrame
            Input dataframe enriched with sensitivity columns.

        current : dict
            Latest observation summary including weights, blended duration,
            DV01, and a full set of P&L scenarios.

        scenario_table : pd.DataFrame
            Spread change from −150 to +300 bps in 25 bp steps, with
            dollar P&L and P&L as % of portfolio.

        available : bool
            Always True (the module falls back to 50/50 weights if needed),
            but set to False only if *df* is empty.
    """
    if df.empty:
        dummy_duration = 0.5 * HY_DURATION + 0.5 * IG_DURATION
        return {
            "df":             df.copy(),
            "current":        {},
            "scenario_table": _build_scenario_table(dummy_duration, portfolio_value),
            "available":      False,
        }

    enriched = compute_spread_sensitivity(df, portfolio_value=portfolio_value)

    last = enriched.iloc[-1]

    def _f(key: str, default: float = np.nan) -> float:
        try:
            v = float(last[key])
            return v if np.isfinite(v) else default
        except (KeyError, TypeError, ValueError):
            return default

    hy_w_val  = _f("hy_weight",        0.5)
    ig_w_val  = _f("ig_weight",        0.5)
    dur_val   = _f("blended_duration", 0.5 * HY_DURATION + 0.5 * IG_DURATION)
    dv01_val  = _f("dv01_per_1m")

    pnl_scenarios = {
        "+25bps":  _f("pnl_25bps_widen"),
        "+50bps":  _f("pnl_50bps_widen"),
        "+100bps": _f("pnl_100bps_widen"),
        "-25bps":  _f("pnl_25bps_tighten"),
        "-50bps":  _pnl(dur_val, -50.0, portfolio_value),
        "-100bps": _pnl(dur_val, -100.0, portfolio_value),
    }

    current = {
        "hy_weight":        hy_w_val,
        "ig_weight":        ig_w_val,
        "blended_duration": dur_val,
        "dv01":             dv01_val,
        "pnl_scenarios":    pnl_scenarios,
        "portfolio_value":  portfolio_value,
    }

    scenario_table = _build_scenario_table(dur_val, portfolio_value)

    return {
        "df":             enriched,
        "current":        current,
        "scenario_table": scenario_table,
        "available":      True,
    }
