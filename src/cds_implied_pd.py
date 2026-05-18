"""
CDS-implied default probability surface.

Convert HY and IG spread levels into implied annual default probabilities
using standard CDS pricing, building a 1y/3y/5y/10y term structure.

Formula (continuous compounding):
  PD_cumulative(T) = 1 - exp(-T * spread / (1 - RR))
  where spread is in decimal (bps / 10000) and RR is recovery rate.

Recovery rate is cycle-adjusted using HY spread level:
  RR = max(0.10, 0.40 - max(0, hy_spread - 450) * 0.0001)

Expected loss per $100 face value:
  EL = PD_1y * LGD * 100  where LGD = 1 - RR

Public API
----------
  HY_RECOVERY_RATE_DEFAULT  = 0.40
  IG_RECOVERY_RATE_DEFAULT  = 0.40
  HY_HISTORICAL_AVG_PD      = 4.0   (% per year)
  IG_HISTORICAL_AVG_PD      = 0.5   (% per year)
  HORIZONS                  = [1, 3, 5, 10]
  compute_cds_implied_pd(df)   -> pd.DataFrame
  get_current_pd_surface(df)   -> dict
  run_cds_implied_pd(df)       -> dict
"""

import numpy as np
import pandas as pd

HY_RECOVERY_RATE_DEFAULT = 0.40
IG_RECOVERY_RATE_DEFAULT = 0.40
HY_HISTORICAL_AVG_PD = 4.0
IG_HISTORICAL_AVG_PD = 0.5
HORIZONS = [1, 3, 5, 10]

_PD_REGIMES = [
    ("Below Historical Avg", 0,    3.0),
    ("Near Historical Avg",  3.0,  6.0),
    ("Elevated",             6.0, 12.0),
    ("Distressed",          12.0, 999),
]


def _recovery_rate(hy_spread_bps):
    if pd.isna(hy_spread_bps):
        return HY_RECOVERY_RATE_DEFAULT
    compression = max(0.0, float(hy_spread_bps) - 450) * 0.0001
    return max(0.10, HY_RECOVERY_RATE_DEFAULT - compression)


def _pd_cumulative(spread_bps, rr, horizon_years):
    if pd.isna(spread_bps) or spread_bps <= 0:
        return np.nan
    s = float(spread_bps) / 10000.0
    lgd = max(1e-6, 1.0 - rr)
    return (1.0 - np.exp(-horizon_years * s / lgd)) * 100.0


def _pd_regime(pd_1y):
    if pd.isna(pd_1y):
        return "Unknown"
    for name, lo, hi in _PD_REGIMES:
        if lo <= pd_1y < hi:
            return name
    return "Distressed"


def _hy_spread_by_horizon(hy_spread, horizon):
    if pd.isna(hy_spread):
        return np.nan
    scale = {1: 0.70, 3: 0.88, 5: 1.00, 10: 1.08}
    return float(hy_spread) * scale.get(horizon, 1.0)


def _ig_spread_by_horizon(ig_spread, horizon):
    if pd.isna(ig_spread):
        return np.nan
    scale = {1: 0.65, 3: 0.85, 5: 1.00, 10: 1.10}
    return float(ig_spread) * scale.get(horizon, 1.0)


def compute_cds_implied_pd(df):
    out = df.copy()
    hy_col = next((c for c in ["hy_spread", "hy_spread_bps"] if c in out.columns), None)
    ig_col = next((c for c in ["ig_spread", "ig_spread_bps"] if c in out.columns), None)

    if hy_col is None and ig_col is None:
        for h in HORIZONS:
            out[f"pd_hy_{h}y"] = np.nan
            out[f"pd_ig_{h}y"] = np.nan
        out["el_hy_1y"] = np.nan
        out["el_ig_1y"] = np.nan
        out["pd_hy_regime"] = "Unknown"
        return out

    hy_spread = out[hy_col] if hy_col else pd.Series(np.nan, index=out.index)
    ig_spread = out[ig_col] if ig_col else pd.Series(np.nan, index=out.index)
    rr_series = hy_spread.apply(_recovery_rate)

    for h in HORIZONS:
        hy_h = hy_spread.apply(lambda s, _h=h: _hy_spread_by_horizon(s, _h))
        ig_h = ig_spread.apply(lambda s, _h=h: _ig_spread_by_horizon(s, _h))
        out[f"pd_hy_{h}y"] = [_pd_cumulative(s, r, h) for s, r in zip(hy_h, rr_series)]
        out[f"pd_ig_{h}y"] = [_pd_cumulative(s, IG_RECOVERY_RATE_DEFAULT, h) for s in ig_h]

    out["el_hy_1y"] = out["pd_hy_1y"] / 100.0 * (1.0 - rr_series) * 100.0
    out["el_ig_1y"] = out["pd_ig_1y"] / 100.0 * (1.0 - IG_RECOVERY_RATE_DEFAULT) * 100.0
    out["pd_hy_regime"] = out["pd_hy_1y"].apply(_pd_regime)
    return out


def get_current_pd_surface(df):
    try:
        enriched = compute_cds_implied_pd(df)
        if enriched.empty:
            return {"available": False}

        row = enriched.iloc[-1]

        def _safe(col):
            val = row.get(col) if col in enriched.columns else None
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            return float(val)

        hy_spread = _safe("hy_spread") or _safe("hy_spread_bps")
        ig_spread = _safe("ig_spread") or _safe("ig_spread_bps")

        pd_surface_hy = {}
        pd_surface_ig = {}
        for h in HORIZONS:
            v_hy = _safe(f"pd_hy_{h}y")
            v_ig = _safe(f"pd_ig_{h}y")
            if v_hy is not None:
                pd_surface_hy[h] = round(v_hy, 2)
            if v_ig is not None:
                pd_surface_ig[h] = round(v_ig, 2)

        rr_current = _recovery_rate(hy_spread) if hy_spread is not None else HY_RECOVERY_RATE_DEFAULT
        el_hy = _safe("el_hy_1y")
        el_ig = _safe("el_ig_1y")
        pd_hy_1y = pd_surface_hy.get(1)
        pd_regime = _pd_regime(pd_hy_1y) if pd_hy_1y is not None else "Unknown"

        if pd_hy_1y is not None:
            vs_hist = (
                "below" if pd_hy_1y < HY_HISTORICAL_AVG_PD * 0.75
                else "near" if pd_hy_1y < HY_HISTORICAL_AVG_PD * 1.25
                else "above"
            )
            pd_vs_historical = (
                f"HY 1y implied PD of {pd_hy_1y:.1f}% is {vs_hist} the "
                f"long-run average of {HY_HISTORICAL_AVG_PD:.1f}%."
            )
        else:
            pd_vs_historical = "Insufficient spread data."

        if hy_spread is not None and pd_hy_1y is not None:
            interpretation = (
                f"At {hy_spread:.0f}bps HY spread and {rr_current:.0%} implied recovery, "
                f"the market prices a {pd_hy_1y:.1f}% annual default rate "
                f"and {pd_surface_hy.get(5, 0):.1f}% cumulative 5y default probability. "
                f"Expected loss per $100 face: ${el_hy:.2f} for HY"
                + (f", ${el_ig:.2f} for IG." if el_ig is not None else ".")
            )
        else:
            interpretation = "Insufficient spread data to compute default probability surface."

        warning = None
        if pd_hy_1y is not None and pd_hy_1y > 8.0:
            warning = (
                f"HY implied 1y PD at {pd_hy_1y:.1f}% — well above historical average "
                f"of {HY_HISTORICAL_AVG_PD:.1f}%. Distressed market conditions."
            )

        return {
            "available": bool(pd_surface_hy or pd_surface_ig),
            "pd_surface_hy": pd_surface_hy,
            "pd_surface_ig": pd_surface_ig,
            "recovery_rate_current": round(rr_current * 100, 1),
            "el_hy_1y": round(el_hy, 2) if el_hy is not None else None,
            "el_ig_1y": round(el_ig, 2) if el_ig is not None else None,
            "pd_regime": pd_regime,
            "pd_vs_historical": pd_vs_historical,
            "hy_spread": round(hy_spread, 1) if hy_spread is not None else None,
            "ig_spread": round(ig_spread, 1) if ig_spread is not None else None,
            "interpretation": interpretation,
            "warning": warning,
        }
    except Exception:
        return {"available": False}


def run_cds_implied_pd(df):
    try:
        if df is None or df.empty:
            return {"available": False}

        hy_present = any(c in df.columns for c in ["hy_spread", "hy_spread_bps"])
        ig_present = any(c in df.columns for c in ["ig_spread", "ig_spread_bps"])
        if not hy_present and not ig_present:
            return {"available": False}

        enriched = compute_cds_implied_pd(df)
        current  = get_current_pd_surface(df)

        pd_hy_history     = enriched["pd_hy_5y"] if "pd_hy_5y" in enriched.columns else pd.Series(dtype=float)
        pd_ig_history     = enriched["pd_ig_5y"] if "pd_ig_5y" in enriched.columns else pd.Series(dtype=float)
        el_hy_history     = enriched["el_hy_1y"] if "el_hy_1y" in enriched.columns else pd.Series(dtype=float)
        pd_regime_history = enriched["pd_hy_regime"] if "pd_hy_regime" in enriched.columns else pd.Series(dtype=str)

        return {
            "available": current.get("available", False),
            "current": current,
            "df": enriched,
            "pd_hy_history": pd_hy_history,
            "pd_ig_history": pd_ig_history,
            "el_hy_history": el_hy_history,
            "pd_regime_history": pd_regime_history,
        }
    except Exception:
        return {"available": False}
