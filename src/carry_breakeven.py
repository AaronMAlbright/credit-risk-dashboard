"""
Carry breakeven analysis — how much can spreads widen before carry is lost?

Carry is the income earned from holding a credit bond (coupon yield).
Spread duration measures how much the bond price falls per bp of spread widening.

BREAKEVEN WIDENING = All-in Yield (%) / Duration (years) × 100
  expressed in basis points per year.

Example: HY yield = 8%, Duration = 4yr → breakeven = 200bps/yr
  If spreads widen more than 200bps over the next year, you lose money.

MONTHLY BREAKEVEN = Breakeven / 12 (bps per month)
  → how much spreads can widen per month before you break even

CARRY CUSHION = how many months of carry covers a specific shock:
  cushion_months = carry_per_year / shock_bps × 12

This answers the most common question in credit portfolio management:
"How much do spreads need to move against me before I should have just held cash?"

Public API
----------
  HY_DURATION = 4.0
  IG_DURATION = 7.0
  BBB_DURATION = 6.0
  SHOCK_SCENARIOS = [25, 50, 100, 150, 200, 300]  # bps
  compute_carry_breakeven(df) → pd.DataFrame   adds carry_ columns
  get_current_breakeven(df) → dict
  run_breakeven_analysis(df) → dict
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HY_DURATION = 4.0
IG_DURATION = 7.0
BBB_DURATION = 6.0
SHOCK_SCENARIOS = [25, 50, 100, 150, 200, 300]  # bps


# ---------------------------------------------------------------------------
# compute_carry_breakeven
# ---------------------------------------------------------------------------

def compute_carry_breakeven(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with carry and breakeven columns appended.

    Uses ``hy_yield`` / ``ig_yield`` when available; falls back to
    ``hy_spread + yield_10y`` / ``ig_spread + yield_10y`` otherwise.
    """
    out = df.copy()

    # --- resolve HY yield ---------------------------------------------------
    if "hy_yield" in out.columns and out["hy_yield"].notna().any():
        hy_yield = out["hy_yield"]
    elif "hy_spread" in out.columns and "yield_10y" in out.columns:
        hy_yield = out["hy_spread"] + out["yield_10y"]
    else:
        hy_yield = pd.Series(np.nan, index=out.index)

    # --- resolve IG yield ---------------------------------------------------
    if "ig_yield" in out.columns and out["ig_yield"].notna().any():
        ig_yield = out["ig_yield"]
    elif "ig_spread" in out.columns and "yield_10y" in out.columns:
        ig_yield = out["ig_spread"] + out["yield_10y"]
    else:
        ig_yield = pd.Series(np.nan, index=out.index)

    # --- carry in bps (annual & monthly) ------------------------------------
    out["carry_hy_annual_bps"] = hy_yield * 100
    out["carry_ig_annual_bps"] = ig_yield * 100
    out["carry_hy_monthly_bps"] = out["carry_hy_annual_bps"] / 12
    out["carry_ig_monthly_bps"] = out["carry_ig_annual_bps"] / 12

    # --- breakeven widening (bps per year) ----------------------------------
    # breakeven = yield(%) / duration * 100
    out["breakeven_hy_bps"] = hy_yield / HY_DURATION * 100
    out["breakeven_ig_bps"] = ig_yield / IG_DURATION * 100
    out["breakeven_hy_monthly"] = out["breakeven_hy_bps"] / 12
    out["breakeven_ig_monthly"] = out["breakeven_ig_bps"] / 12

    # --- carry cushion: months of carry that covers a specific shock ---------
    # cushion_months = shock_bps / breakeven_hy_monthly
    # Avoid division-by-zero with replace
    be_hy_m = out["breakeven_hy_monthly"].replace(0, np.nan)
    out["carry_cushion_hy_50bps"] = 50 / be_hy_m
    out["carry_cushion_hy_100bps"] = 100 / be_hy_m
    out["carry_cushion_hy_200bps"] = 200 / be_hy_m

    # --- qualitative regime label for HY carry ------------------------------
    def _carry_regime(bps: float) -> str:
        if pd.isna(bps):
            return "Unknown"
        if bps < 150:
            return "Rich (<150bps)"
        if bps <= 300:
            return "Fair (150-300bps)"
        return "Cheap (>300bps)"

    out["carry_hy_regime"] = out["carry_hy_annual_bps"].apply(_carry_regime)

    return out


# ---------------------------------------------------------------------------
# get_current_breakeven
# ---------------------------------------------------------------------------

def _scenario_decision(months_to_recover: float) -> str:
    """Translate months-to-recover into a portfolio action label."""
    if pd.isna(months_to_recover):
        return "Insufficient data"
    if months_to_recover < 6:
        return "Hold"
    if months_to_recover <= 12:
        return "Reduce"
    return "Exit"


def get_current_breakeven(df: pd.DataFrame) -> dict:
    """Return a dict describing the current (latest row) carry/breakeven state.

    Includes a scenario table covering each entry in SHOCK_SCENARIOS for both
    HY and IG, plus a blended breakeven and plain-English interpretation.
    """
    enriched = compute_carry_breakeven(df)
    if enriched.empty:
        return {"available": False}

    row = enriched.iloc[-1]

    def _safe(col: str):
        return row[col] if col in enriched.columns and pd.notna(row[col]) else None

    # --- latest scalar values -----------------------------------------------
    hy_yield_val = _safe("carry_hy_annual_bps")
    ig_yield_val = _safe("carry_ig_annual_bps")
    be_hy = _safe("breakeven_hy_bps")
    be_ig = _safe("breakeven_ig_bps")
    be_hy_m = _safe("breakeven_hy_monthly")
    be_ig_m = _safe("breakeven_ig_monthly")

    # --- scenario table as list of dicts ------------------------------------
    scenarios = []
    for shock in SHOCK_SCENARIOS:
        # HY
        if be_hy_m and be_hy_m > 0:
            hy_months = shock / be_hy_m
            hy_pl_net = (hy_yield_val / 12 if hy_yield_val else np.nan) - shock * HY_DURATION
        else:
            hy_months = np.nan
            hy_pl_net = np.nan
        hy_decision = _scenario_decision(hy_months)

        # IG
        if be_ig_m and be_ig_m > 0:
            ig_months = shock / be_ig_m
            ig_pl_net = (ig_yield_val / 12 if ig_yield_val else np.nan) - shock * IG_DURATION
        else:
            ig_months = np.nan
            ig_pl_net = np.nan
        ig_decision = _scenario_decision(ig_months)

        scenarios.append({
            "shock_bps": shock,
            "hy_carry_months": round(hy_months, 1) if not np.isnan(hy_months) else None,
            "hy_pl_net_bps": round(hy_pl_net, 1) if not np.isnan(hy_pl_net) else None,
            "hy_decision": hy_decision,
            "ig_carry_months": round(ig_months, 1) if not np.isnan(ig_months) else None,
            "ig_pl_net_bps": round(ig_pl_net, 1) if not np.isnan(ig_pl_net) else None,
            "ig_decision": ig_decision,
        })

    # --- blended breakeven (70% HY, 30% IG) ---------------------------------
    if be_hy is not None and be_ig is not None:
        blended_breakeven_bps = 0.70 * be_hy + 0.30 * be_ig
    elif be_hy is not None:
        blended_breakeven_bps = be_hy
    elif be_ig is not None:
        blended_breakeven_bps = be_ig
    else:
        blended_breakeven_bps = None

    # --- plain-English interpretation ---------------------------------------
    if hy_yield_val is not None and be_hy is not None:
        hy_yield_pct = hy_yield_val / 100
        hy_spread_now = _safe("hy_spread") or 0
        interpretation = (
            f"At {hy_yield_pct:.1f}% yield, HY can absorb {be_hy:.0f}bps of widening "
            f"per year before carry is erased. "
            f"Current HY spread is {hy_spread_now:.0f}bps — "
            f"a move of {be_hy:.0f}bps would push total return negative on a 1-year horizon."
        )
    else:
        interpretation = "Insufficient yield data to compute carry breakeven."

    return {
        "available": True,
        "hy_yield_bps": round(hy_yield_val, 1) if hy_yield_val is not None else None,
        "ig_yield_bps": round(ig_yield_val, 1) if ig_yield_val is not None else None,
        "breakeven_hy_bps": round(be_hy, 1) if be_hy is not None else None,
        "breakeven_ig_bps": round(be_ig, 1) if be_ig is not None else None,
        "breakeven_hy_monthly": round(be_hy_m, 2) if be_hy_m is not None else None,
        "breakeven_ig_monthly": round(be_ig_m, 2) if be_ig_m is not None else None,
        "carry_cushion_50bps_months": _safe("carry_cushion_hy_50bps"),
        "carry_cushion_100bps_months": _safe("carry_cushion_hy_100bps"),
        "carry_cushion_200bps_months": _safe("carry_cushion_hy_200bps"),
        "carry_hy_regime": _safe("carry_hy_regime"),
        "blended_breakeven_bps": round(blended_breakeven_bps, 1) if blended_breakeven_bps is not None else None,
        "scenario_table": scenarios,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# run_breakeven_analysis
# ---------------------------------------------------------------------------

def run_breakeven_analysis(df: pd.DataFrame) -> dict:
    """Top-level entry point.  Returns everything needed to render the UI.

    Keys
    ----
    current          : get_current_breakeven result
    df               : compute_carry_breakeven result (full history)
    scenario_table   : pd.DataFrame — shock scenarios for display
    historical_breakeven : pd.DataFrame — date-indexed breakeven time series
    current_percentile_hy : float — where current HY breakeven sits vs history
    available        : bool
    """
    if df is None or df.empty:
        return {"available": False}

    enriched = compute_carry_breakeven(df)

    # --- availability check -------------------------------------------------
    if enriched["breakeven_hy_bps"].isna().all() and enriched["breakeven_ig_bps"].isna().all():
        return {"available": False, "df": enriched}

    current = get_current_breakeven(df)

    # --- scenario_table as DataFrame ----------------------------------------
    scenario_rows = current.get("scenario_table", [])
    scenario_df = pd.DataFrame(scenario_rows, columns=[
        "shock_bps",
        "hy_carry_months", "hy_pl_net_bps", "hy_decision",
        "ig_carry_months", "ig_pl_net_bps", "ig_decision",
    ]) if scenario_rows else pd.DataFrame()

    # --- historical breakeven time series (index may be DatetimeIndex) ------
    hist_cols = ["breakeven_hy_bps", "breakeven_ig_bps"]
    available_hist_cols = [c for c in hist_cols if c in enriched.columns]
    historical_breakeven = enriched[available_hist_cols].copy()

    # --- current percentile (where does today's HY breakeven sit?) ----------
    current_be_hy = current.get("breakeven_hy_bps")
    if current_be_hy is not None and "breakeven_hy_bps" in enriched.columns:
        series = enriched["breakeven_hy_bps"].dropna()
        if len(series) > 0:
            current_percentile_hy = float((series < current_be_hy).mean() * 100)
        else:
            current_percentile_hy = None
    else:
        current_percentile_hy = None

    return {
        "available": True,
        "current": current,
        "df": enriched,
        "scenario_table": scenario_df,
        "historical_breakeven": historical_breakeven,
        "current_percentile_hy": current_percentile_hy,
    }
