"""
Distressed debt ratio, recovery rate cycle, and LGD positioning.

The fraction of the HY universe trading below 80 cents on the dollar
("distressed") is the most direct late-cycle credit signal. When this
ratio spikes, actual default rates follow with a 6-12 month lag.

Proxy methodology (bond-level price data not available)
-------------------------------------------------------
Distressed ratio proxy:
  - hy_spread < 400 bps  → 0% distressed (pristine credit)
  - hy_spread 400-600    → linear from 0% → 3%
  - hy_spread > 600      → linear from 3% → 40% at 1400 bps
  Calibrated to GFC peak (~38% distressed) and COVID peak (~27%).

Recovery rate proxy:
  recovery = clip(40 - (hy_spread - 450) * 0.01, 25, 50)
  - Wide spreads → low recovery (distressed overhang, fire sales)
  - Tight spreads → high recovery (high-quality issuers)

LGD = 1 - recovery_rate_proxy

Annual PD (Jarrow-Turnbull):
  PD = (hy_spread_decimal) / (1 - recovery_rate_proxy / 100)
  where hy_spread_decimal = hy_spread / 100

Distressed pipeline (expected losses from distressed bucket):
  pipeline = distressed_ratio_proxy * (lgd_proxy / 100)

Cycle phases:
  < 1%   : Pristine — Late Cycle Complacency
  1–5%   : Normal — Mid Cycle
  5–10%  : Elevated — Early Stress
  10–20% : Distressed Cycle Underway
  > 20%  : Acute Distress — GFC/COVID-level

Historical stress episodes used for cycle comparison:
  GFC 2008-09, COVID 2020, Energy 2015-16, COVID-adjacent 2022-23

Source columns used from master df
------------------------------------
  hy_spread        ICE BofA HY OAS (stored as %, e.g. 5.5 = 550 bps) — required.
  distressed_ratio : optional override if actual data present in df.
  pct_distressed   : optional override alias.
  stressed_issuers : optional override alias.

Public API
----------
  compute_distressed_metrics(df) -> pd.DataFrame
  get_current_distressed_state(df, enriched) -> dict
  compute_cycle_comparison(enriched) -> pd.DataFrame
  run_distressed_debt(df) -> dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Historical GFC and COVID distressed ratio peaks (%)
_GFC_PEAK_DISTRESSED: float = 38.0
_COVID_PEAK_DISTRESSED: float = 27.0

# Historical hy_spread peaks at each episode (stored as % = bps/100)
_STRESS_EPISODES: list[dict] = [
    {
        "episode": "GFC 2008-09",
        "peak_hy_spread": 19.00,   # ~1900 bps in March 2009
        "peak_year": 2009,
        "actual_peak_distressed": 38.0,
    },
    {
        "episode": "COVID 2020",
        "peak_hy_spread": 11.00,   # ~1100 bps in March 2020
        "peak_year": 2020,
        "actual_peak_distressed": 27.0,
    },
    {
        "episode": "Energy 2015-16",
        "peak_hy_spread": 8.50,    # ~850 bps
        "peak_year": 2016,
        "actual_peak_distressed": 15.0,
    },
    {
        "episode": "Post-Hike 2022-23",
        "peak_hy_spread": 5.80,    # ~580 bps
        "peak_year": 2023,
        "actual_peak_distressed": 8.0,
    },
]

# Minimum rows required
_MIN_ROWS: int = 252
_HIST_ROWS: int = 504

# Thresholds for distressed_ratio_proxy computation
_SPREAD_TIGHT_THRESHOLD: float = 4.00    # < 400 bps → 0% distressed
_SPREAD_MODERATE_LOW: float = 4.00       # 400 bps
_SPREAD_MODERATE_HIGH: float = 6.00     # 600 bps
_SPREAD_ACUTE_THRESHOLD: float = 14.00  # 1400 bps → 40% distressed

# Recovery rate parameters (linear with spread)
_RECOVERY_CENTER_SPREAD: float = 4.50   # 450 bps
_RECOVERY_SLOPE: float = 0.01           # percent per bps-unit of hy_spread
_RECOVERY_BASE: float = 40.0
_RECOVERY_FLOOR: float = 25.0
_RECOVERY_CAP: float = 50.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _distressed_ratio_from_spread(hy_spread_series: pd.Series) -> pd.Series:
    """
    Map hy_spread (stored as %, e.g. 5.5 for 550 bps) to distressed ratio proxy (%).

    Segmented linear model:
      spread < 4.00  → 0%
      4.00 to 6.00   → linear 0% → 3%
      > 6.00         → linear 3% → 40% at 14.00 (1400 bps)
    Clipped to [0, 40].
    """
    s = hy_spread_series.copy()
    result = pd.Series(np.nan, index=s.index)

    below_400 = s < _SPREAD_TIGHT_THRESHOLD
    between = (s >= _SPREAD_MODERATE_LOW) & (s <= _SPREAD_MODERATE_HIGH)
    above_600 = s > _SPREAD_MODERATE_HIGH

    result[below_400] = 0.0

    # 400 to 600 bps: linear 0% → 3%
    if between.any():
        t = (s[between] - _SPREAD_MODERATE_LOW) / (_SPREAD_MODERATE_HIGH - _SPREAD_MODERATE_LOW)
        result[between] = t * 3.0

    # > 600 bps: linear 3% → 40% at 1400 bps
    if above_600.any():
        t = (s[above_600] - _SPREAD_MODERATE_HIGH) / (_SPREAD_ACUTE_THRESHOLD - _SPREAD_MODERATE_HIGH)
        result[above_600] = 3.0 + t * 37.0

    return result.clip(0.0, 40.0)


def _recovery_rate_from_spread(hy_spread_series: pd.Series) -> pd.Series:
    """
    Map hy_spread (%) to estimated recovery rate (%).

    recovery = clip(40 - (spread - 4.50) * 0.01 * 100, 25, 50)
    Note: slope applied as percent-per-spread-unit where spread is in %.
    Wider spreads → lower recovery (distressed overhang).
    """
    recovery = _RECOVERY_BASE - (hy_spread_series - _RECOVERY_CENTER_SPREAD) * (_RECOVERY_SLOPE * 100.0)
    return recovery.clip(_RECOVERY_FLOOR, _RECOVERY_CAP)


def _cycle_phase(ratio: float) -> str:
    """Map distressed ratio proxy (%) to a cycle phase label."""
    if np.isnan(ratio):
        return "Unknown"
    if ratio < 1.0:
        return "Pristine — Late Cycle Complacency"
    if ratio < 5.0:
        return "Normal — Mid Cycle"
    if ratio < 10.0:
        return "Elevated — Early Stress"
    if ratio < 20.0:
        return "Distressed Cycle Underway"
    return "Acute Distress — GFC/COVID-level"


def _pd_annual(hy_spread_pct: float, recovery_pct: float) -> float:
    """
    Jarrow-Turnbull annual default probability.

    PD = spread_decimal / (1 - recovery_decimal)
    where spread_decimal = hy_spread / 100 (converting % to decimal)
    and   recovery_decimal = recovery_pct / 100.
    Returns decimal (e.g., 0.092 for 9.2%).
    """
    lgd = 1.0 - recovery_pct / 100.0
    if lgd <= 0:
        return 0.0
    spread_decimal = hy_spread_pct / 100.0
    pd_val = spread_decimal / lgd
    return float(np.clip(pd_val, 0.0, 1.0))


# ---------------------------------------------------------------------------
# compute_distressed_metrics
# ---------------------------------------------------------------------------

def compute_distressed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich a copy of *df* with distressed debt metric columns.

    If 'hy_spread' is absent, returns an unchanged copy.

    New columns
    -----------
    distressed_ratio_proxy  : estimated % of HY universe at distressed levels
    recovery_rate_proxy     : estimated recovery rate (%)
    lgd_proxy               : 1 - recovery_rate_proxy / 100  (as a %)
    distressed_cycle_phase  : phase label string
    distressed_pipeline     : distressed_ratio * lgd (expected loss proxy, %)
    pd_annual_proxy         : annual default probability from Jarrow-Turnbull (decimal)
    """
    out = df.copy()

    if "hy_spread" not in out.columns or out["hy_spread"].notna().sum() < 2:
        for col in [
            "distressed_ratio_proxy",
            "recovery_rate_proxy",
            "lgd_proxy",
            "distressed_cycle_phase",
            "distressed_pipeline",
            "pd_annual_proxy",
        ]:
            out[col] = np.nan
        return out

    spread = out["hy_spread"]

    # Use actual column if available, otherwise compute proxy
    actual_cols = [c for c in ["distressed_ratio", "pct_distressed", "stressed_issuers"] if c in out.columns]
    if actual_cols:
        dr = out[actual_cols[0]].copy()
        # Fill remaining NaNs with proxy
        proxy = _distressed_ratio_from_spread(spread)
        dr = dr.fillna(proxy)
        out["distressed_ratio_proxy"] = dr
    else:
        out["distressed_ratio_proxy"] = _distressed_ratio_from_spread(spread)

    out["recovery_rate_proxy"] = _recovery_rate_from_spread(spread)
    out["lgd_proxy"] = (100.0 - out["recovery_rate_proxy"])  # expressed as %

    out["distressed_cycle_phase"] = out["distressed_ratio_proxy"].apply(
        lambda v: _cycle_phase(float(v)) if pd.notna(v) else "Unknown"
    )

    # Pipeline = distressed_ratio (%) * lgd_proxy (%) / 100  → expressed as %
    out["distressed_pipeline"] = out["distressed_ratio_proxy"] * (out["lgd_proxy"] / 100.0)

    # PD annual proxy — computed row-wise
    pd_vals = [
        _pd_annual(float(s), float(r))
        if pd.notna(s) and pd.notna(r)
        else np.nan
        for s, r in zip(out["hy_spread"], out["recovery_rate_proxy"])
    ]
    out["pd_annual_proxy"] = pd_vals

    return out


# ---------------------------------------------------------------------------
# get_current_distressed_state
# ---------------------------------------------------------------------------

def get_current_distressed_state(
    df: pd.DataFrame,
    enriched: pd.DataFrame = None,
) -> dict:
    """Return the current distressed market state as a flat dict.

    Parameters
    ----------
    df       : raw master DataFrame (DatetimeIndex required).
    enriched : pre-computed output of compute_distressed_metrics(df). If None,
               it will be computed internally.

    Returns
    -------
    dict with keys:
        available            : bool
        distressed_ratio_proxy : float
        recovery_rate_proxy  : float
        lgd_proxy            : float
        pd_annual_proxy      : float
        cycle_phase          : str
        distressed_pipeline  : float
        hy_spread            : float
        vs_gfc_peak          : float  — current as % of GFC peak (38%)
        vs_covid_peak        : float  — current as % of COVID peak (27%)
        warning              : str or None
    """
    try:
        if df is None or df.empty or "hy_spread" not in df.columns:
            return {"available": False}

        if enriched is None:
            enriched = compute_distressed_metrics(df)

        last = enriched.iloc[-1]

        def _f(col: str) -> float | None:
            if col not in enriched.columns:
                return None
            v = last[col]
            try:
                fv = float(v)
                return None if np.isnan(fv) else fv
            except Exception:
                return None

        dr = _f("distressed_ratio_proxy")
        rr = _f("recovery_rate_proxy")
        lgd = _f("lgd_proxy")
        pd_val = _f("pd_annual_proxy")
        pipeline = _f("distressed_pipeline")
        hy = _f("hy_spread")

        phase = str(last.get("distressed_cycle_phase", "Unknown")) if "distressed_cycle_phase" in enriched.columns else "Unknown"

        vs_gfc = round(dr / _GFC_PEAK_DISTRESSED * 100.0, 1) if dr is not None else None
        vs_covid = round(dr / _COVID_PEAK_DISTRESSED * 100.0, 1) if dr is not None else None

        # Warning if elevated or worse
        warning: str | None = None
        if dr is not None and dr >= 10.0:
            warning = (
                f"Distressed ratio proxy ({dr:.1f}%) is in the 'Distressed Cycle' zone. "
                f"This level is {vs_gfc:.0f}% of the GFC peak and {vs_covid:.0f}% of the "
                "COVID peak — historical default waves follow with 6-12 month lag."
            )
        elif dr is not None and dr >= 5.0:
            warning = (
                f"Distressed ratio proxy ({dr:.1f}%) signals early stress. Monitor for "
                "deterioration toward the 10% threshold that historically precedes default cycles."
            )

        return {
            "available": True,
            "distressed_ratio_proxy": dr,
            "recovery_rate_proxy": rr,
            "lgd_proxy": lgd,
            "pd_annual_proxy": pd_val,
            "cycle_phase": phase,
            "distressed_pipeline": pipeline,
            "hy_spread": hy,
            "vs_gfc_peak": vs_gfc,
            "vs_covid_peak": vs_covid,
            "warning": warning,
        }

    except Exception:
        return {"available": False}


# ---------------------------------------------------------------------------
# compute_cycle_comparison
# ---------------------------------------------------------------------------

def compute_cycle_comparison(enriched: pd.DataFrame) -> pd.DataFrame:
    """Compare current distressed ratio proxy to historical stress episode peaks.

    For each episode, the function locates the approximate peak in the
    distressed_ratio_proxy series (within the episode's expected year window),
    then computes where the current level stands relative to that peak.

    Returns
    -------
    pd.DataFrame with columns:
        episode            : str
        peak_hy_spread     : float  — peak HY spread at episode (% as stored)
        peak_distressed_pct: float  — peak distressed ratio at episode (%)
        current_pct        : float  — current distressed_ratio_proxy (%)
        pct_of_peak        : float  — current as % of episode peak
    """
    rows = []
    current_ratio = float("nan")
    if "distressed_ratio_proxy" in enriched.columns:
        last_valid = enriched["distressed_ratio_proxy"].dropna()
        if not last_valid.empty:
            current_ratio = float(last_valid.iloc[-1])

    for episode in _STRESS_EPISODES:
        ep_name = episode["episode"]
        peak_spread = episode["peak_hy_spread"]
        actual_peak = episode["actual_peak_distressed"]
        peak_year = episode["peak_year"]

        # Try to find the episode peak in the actual data (±1 year window)
        computed_peak = actual_peak  # default to known value
        if "distressed_ratio_proxy" in enriched.columns and hasattr(enriched.index, "year"):
            window = enriched[
                (enriched.index.year >= peak_year - 1) & (enriched.index.year <= peak_year + 1)
            ]
            if not window.empty and window["distressed_ratio_proxy"].notna().any():
                computed_peak = float(window["distressed_ratio_proxy"].max())

        pct_of_peak = (
            round(current_ratio / computed_peak * 100.0, 1)
            if computed_peak > 0 and not np.isnan(current_ratio)
            else float("nan")
        )

        rows.append(
            {
                "episode": ep_name,
                "peak_hy_spread": peak_spread,
                "peak_distressed_pct": round(computed_peak, 1),
                "current_pct": round(current_ratio, 1) if not np.isnan(current_ratio) else float("nan"),
                "pct_of_peak": pct_of_peak,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# run_distressed_debt
# ---------------------------------------------------------------------------

def run_distressed_debt(df: pd.DataFrame) -> dict:
    """Top-level entry point for distressed debt analysis.

    Requires 'hy_spread' column and at least 252 rows.

    Returns
    -------
    dict with keys:
        available          : bool
        current            : dict — from get_current_distressed_state()
        historical         : pd.DataFrame — last 504 rows with distressed columns
        signal_series      : pd.Series — distressed_ratio_proxy for last 504 rows
        cycle_comparison   : pd.DataFrame — episode comparison table
        percentile_current : float — where current ratio sits in full history (0-100)
        recovery_outlook   : str — narrative on recovery rate vs spread level
        interpretation     : str
        warning            : str or None
    """
    try:
        if df is None or df.empty:
            return {"available": False}

        if "hy_spread" not in df.columns:
            return {"available": False}

        valid_rows = df["hy_spread"].notna().sum()
        if valid_rows < _MIN_ROWS:
            return {"available": False}

        enriched = compute_distressed_metrics(df)

        # ---- current state -------------------------------------------------
        current = get_current_distressed_state(df, enriched)
        if not current.get("available"):
            return {"available": False}

        # ---- historical slice ----------------------------------------------
        hist_cols = [
            "hy_spread",
            "distressed_ratio_proxy",
            "recovery_rate_proxy",
            "lgd_proxy",
            "distressed_cycle_phase",
            "distressed_pipeline",
            "pd_annual_proxy",
        ]
        hist_cols_avail = [c for c in hist_cols if c in enriched.columns]
        historical = enriched[hist_cols_avail].tail(_HIST_ROWS).copy()

        # ---- signal series -------------------------------------------------
        if "distressed_ratio_proxy" in enriched.columns:
            signal_series = enriched["distressed_ratio_proxy"].tail(_HIST_ROWS).copy()
        else:
            signal_series = pd.Series(dtype=float)

        # ---- cycle comparison ----------------------------------------------
        cycle_comparison = compute_cycle_comparison(enriched)

        # ---- percentile vs full history ------------------------------------
        percentile_current = float("nan")
        if "distressed_ratio_proxy" in enriched.columns:
            dr_series = enriched["distressed_ratio_proxy"].dropna()
            current_dr = current.get("distressed_ratio_proxy")
            if current_dr is not None and len(dr_series) > 0:
                percentile_current = float((dr_series < current_dr).mean() * 100.0)

        # ---- recovery outlook narrative ------------------------------------
        rr = current.get("recovery_rate_proxy")
        hy = current.get("hy_spread")
        lgd = current.get("lgd_proxy")
        if rr is not None and hy is not None:
            hy_bps = round(hy * 100)
            if rr >= 45:
                recovery_outlook = (
                    f"At {hy_bps} bps, spreads are tight and expected recovery rates are "
                    f"elevated (~{rr:.0f}%). Lenders are pricing in high collateral quality "
                    "and limited default risk. LGD is compressed — loss severity would be "
                    "modest if defaults occurred."
                )
            elif rr >= 38:
                lgd_display = lgd if lgd is not None else (100.0 - rr)
                recovery_outlook = (
                    f"At {hy_bps} bps, recovery expectations are near the long-run average "
                    f"(~{rr:.0f}%). Default risk is present but loss severity is historically "
                    f"typical. LGD of approximately {lgd_display:.0f}% is consistent with mid-cycle credit conditions."
                )
            else:
                recovery_outlook = (
                    f"At {hy_bps} bps, expected recovery rates have fallen to ~{rr:.0f}% — "
                    "consistent with distressed cycles where fire-sale dynamics suppress "
                    f"recoveries. LGD of ~{lgd:.0f}% implies severe loss severity if defaults "
                    "materialise. This level is typically seen in the Energy 2015-16 to GFC range."
                )
        else:
            recovery_outlook = "Insufficient spread data to project recovery rate outlook."

        # ---- interpretation ------------------------------------------------
        dr = current.get("distressed_ratio_proxy")
        phase = current.get("cycle_phase", "Unknown")
        pd_val = current.get("pd_annual_proxy")
        pipeline = current.get("distressed_pipeline")

        vs_gfc = current.get("vs_gfc_peak")
        vs_covid = current.get("vs_covid_peak")

        if dr is not None:
            interpretation = (
                f"The distressed ratio proxy stands at {dr:.1f}% — phase: '{phase}'. "
                f"This represents {vs_gfc:.0f}% of the GFC peak and {vs_covid:.0f}% of the "
                f"COVID peak distressed levels. "
            )
            if pd_val is not None:
                interpretation += (
                    f"Jarrow-Turnbull implied annual default probability: {pd_val * 100:.1f}%. "
                )
            if pipeline is not None:
                interpretation += (
                    f"Distressed pipeline (expected loss proxy): {pipeline:.2f}% of face value. "
                )
        else:
            interpretation = "Distressed debt proxy unavailable — hy_spread data required."

        warning = current.get("warning")

        return {
            "available": True,
            "current": current,
            "historical": historical,
            "signal_series": signal_series,
            "cycle_comparison": cycle_comparison,
            "percentile_current": percentile_current,
            "recovery_outlook": recovery_outlook,
            "interpretation": interpretation,
            "warning": warning,
        }

    except Exception:
        return {"available": False}
