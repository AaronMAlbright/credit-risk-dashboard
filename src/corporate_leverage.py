"""
Corporate leverage cycle — US nonfinancial corporate debt relative to GDP and earnings.

High leverage entering a downturn amplifies credit spread widening; deleveraging
enables tightening. This is the fundamental credit quality backdrop — the structural
context for all market-based signals.

When FRED-sourced leverage fundamentals are absent (corp_debt_to_gdp, gdp, corp_debt,
interest_coverage), the module falls back to a synthetic leverage proxy built from
hy_spread: the rolling 252-day percentile rank of HY spreads captures implied leverage
stress with high practical fidelity.

Historical reference levels (corp_debt_to_gdp, nonfinancial corporate, % of GDP):
  Pre-GFC peak   : ~73%  (Q3 2008)
  GFC trough     : ~65%  (Q1 2011)
  2019 late-cycle: ~75%  (Q4 2019)
  COVID peak     : ~85%  (Q2 2020, due to GDP collapse)
  Post-COVID norm: ~70–72%

NOTE: LEVERAGE_THRESHOLDS below use % units consistent with corp_debt_to_gdp as
stored (e.g., 48.5 means 48.5%). These are conservative thresholds suited to a
shorter/normalised series; tune as needed once live FRED data are ingested.

Public API
----------
  LEVERAGE_THRESHOLDS : dict
  compute_corporate_leverage(df) -> pd.DataFrame
  run_corporate_leverage_analysis(df) -> dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEVERAGE_THRESHOLDS: dict[str, float] = {
    "low":      45.0,   # corp_debt_to_gdp % — historical "safe" zone
    "moderate": 50.0,
    "high":     55.0,
    "extreme":  60.0,
}

_HIST_ROWS: int = 504          # ~2 trading years of history returned
_ROLL_WINDOW: int = 252        # rolling window for percentile rank / change
_MIN_PERIODS: int = 63         # minimum observations before computing rolling stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rolling_pctrank(series: pd.Series,
                     window: int = _ROLL_WINDOW,
                     min_periods: int = _MIN_PERIODS) -> pd.Series:
    """Compute rolling percentile rank (0–100) using pure numpy.

    For each position i, counts what fraction of the preceding `window`
    observations the current value exceeds, then multiplies by 100.
    """
    arr = series.to_numpy(dtype=float)
    n = len(arr)
    ranks: list[float] = []
    for i in range(n):
        if np.isnan(arr[i]):
            ranks.append(np.nan)
            continue
        start = max(0, i - window + 1)
        window_vals = arr[start: i + 1]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) < min_periods:
            ranks.append(np.nan)
        else:
            pct = float((valid < arr[i]).sum()) / len(valid) * 100.0
            ranks.append(pct)
    return pd.Series(ranks, index=series.index)


def _leverage_regime(ratio: float) -> str:
    """Map a corp_debt_to_gdp ratio to a leverage regime label."""
    if np.isnan(ratio):
        return "Unknown"
    if ratio < LEVERAGE_THRESHOLDS["low"]:
        return "Low"
    if ratio < LEVERAGE_THRESHOLDS["moderate"]:
        return "Moderate"
    if ratio < LEVERAGE_THRESHOLDS["high"]:
        return "High"
    if ratio < LEVERAGE_THRESHOLDS["extreme"]:
        return "Extreme"
    return "Extreme"


def _stress_signal_from_ratio(ratio: float, change_1y: float) -> float:
    """Convert leverage ratio (and its 1-year change) to a 0–100 stress score.

    Base score by level
    -------------------
    <45  → 20
    45–50 → 40
    50–55 → 60
    55–60 → 75
    >60  → 88

    Momentum addon
    --------------
    +8 if change_1y > 3 percentage points (rising fast)
    """
    if np.isnan(ratio):
        return np.nan

    thresholds = LEVERAGE_THRESHOLDS
    if ratio < thresholds["low"]:
        base = 20.0
    elif ratio < thresholds["moderate"]:
        base = 40.0
    elif ratio < thresholds["high"]:
        base = 60.0
    elif ratio < thresholds["extreme"]:
        base = 75.0
    else:
        base = 88.0

    addon = 8.0 if (not np.isnan(change_1y) and change_1y > 3.0) else 0.0
    return min(100.0, base + addon)


# ---------------------------------------------------------------------------
# compute_corporate_leverage
# ---------------------------------------------------------------------------

def compute_corporate_leverage(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich *df* with corporate leverage analysis columns.

    Added columns
    -------------
    leverage_ratio          : corp_debt_to_gdp if available, else NaN
    leverage_regime         : "Low" / "Moderate" / "High" / "Extreme" / "Unknown"
    leverage_change_1y      : 252-day change in leverage_ratio (pp); NaN when
                              leverage_ratio is unavailable
    leverage_stress_signal  : 0–100 score (see module docstring for mapping)
    synthetic_leverage_proxy: rolling 252-day percentile rank of hy_spread × 100
                              (always computed when hy_spread is present)

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex. The date must be the index, not a column.

    Returns
    -------
    pd.DataFrame — copy of *df* with the five columns above appended.
    """
    out = df.copy()

    # ------------------------------------------------------------------
    # 1. Resolve leverage ratio source
    # ------------------------------------------------------------------
    has_ratio = (
        "corp_debt_to_gdp" in out.columns
        and out["corp_debt_to_gdp"].notna().any()
    )

    if has_ratio:
        leverage_ratio = out["corp_debt_to_gdp"].copy()
    else:
        # Attempt synthetic construction from corp_debt / gdp if both present
        has_debt = "corp_debt" in out.columns and out["corp_debt"].notna().any()
        has_gdp = "gdp" in out.columns and out["gdp"].notna().any()
        if has_debt and has_gdp:
            # Forward-fill quarterly GDP to daily frequency before dividing
            gdp_ff = out["gdp"].ffill()
            gdp_safe = gdp_ff.replace(0.0, np.nan)
            leverage_ratio = (out["corp_debt"] / gdp_safe) * 100.0
            has_ratio = leverage_ratio.notna().any()
        else:
            leverage_ratio = pd.Series(np.nan, index=out.index)

    out["leverage_ratio"] = leverage_ratio

    # ------------------------------------------------------------------
    # 2. Regime label (vectorised via map on scalar values)
    # ------------------------------------------------------------------
    out["leverage_regime"] = out["leverage_ratio"].apply(
        lambda v: _leverage_regime(float(v)) if pd.notna(v) else "Unknown"
    )

    # ------------------------------------------------------------------
    # 3. 1-year (252-day) change in leverage ratio
    # ------------------------------------------------------------------
    out["leverage_change_1y"] = leverage_ratio.diff(_ROLL_WINDOW)

    # ------------------------------------------------------------------
    # 4. Synthetic leverage proxy — always compute when hy_spread present
    # ------------------------------------------------------------------
    has_hy = "hy_spread" in out.columns and out["hy_spread"].notna().any()
    if has_hy:
        out["synthetic_leverage_proxy"] = _rolling_pctrank(out["hy_spread"])
    else:
        out["synthetic_leverage_proxy"] = np.nan

    # ------------------------------------------------------------------
    # 5. Leverage stress signal (0–100)
    # ------------------------------------------------------------------
    if has_ratio and out["leverage_ratio"].notna().any():
        # Use fundamental ratio path
        stress = []
        for ratio_val, change_val in zip(
            out["leverage_ratio"].to_numpy(dtype=float),
            out["leverage_change_1y"].to_numpy(dtype=float),
        ):
            stress.append(_stress_signal_from_ratio(ratio_val, change_val))
        out["leverage_stress_signal"] = stress
    else:
        # Fall back to synthetic proxy (already on 0–100 scale)
        out["leverage_stress_signal"] = out["synthetic_leverage_proxy"]

    return out


# ---------------------------------------------------------------------------
# run_corporate_leverage_analysis
# ---------------------------------------------------------------------------

def run_corporate_leverage_analysis(df: pd.DataFrame) -> dict:
    """Full corporate leverage analysis packaged for dashboard consumption.

    Always returns ``available: True`` because the synthetic proxy provides a
    fallback whenever hy_spread is present. If neither FRED fundamentals nor
    hy_spread are available, still returns a graceful result.

    Returns
    -------
    dict with keys:
        df              : enriched DataFrame (full history, all new columns)
        current         : dict of latest scalar values
        historical      : last 504 rows with leverage_stress_signal &
                          synthetic_leverage_proxy columns
        cycle_phase     : str — "Leveraging Up" / "Leveraging Up (from low base)" /
                          "Deleveraging" / "Peak Leverage" / "Trough / Stable"
        available       : bool — always True (uses synthetic proxy as fallback)
        interpretation  : one-sentence summary
        warning         : str — non-empty if leverage_stress_signal > 70
    """
    if df is None or df.empty:
        return {
            "available": False,
            "df": pd.DataFrame() if df is None else df,
            "current": {},
            "historical": pd.DataFrame(),
            "cycle_phase": "Unknown",
            "interpretation": "No data available.",
            "warning": "",
        }

    enriched = compute_corporate_leverage(df)
    last = enriched.iloc[-1]

    # ------------------------------------------------------------------
    # Extract latest scalar values (convert numpy scalars to Python types)
    # ------------------------------------------------------------------
    def _scalar(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (np.floating, np.integer)):
            return float(val)
        return val

    leverage_ratio_cur = _scalar(last.get("leverage_ratio", np.nan))
    leverage_regime_cur = last.get("leverage_regime", "Unknown")
    leverage_change_1y_cur = _scalar(last.get("leverage_change_1y", np.nan))
    leverage_stress_cur = _scalar(last.get("leverage_stress_signal", np.nan))
    synthetic_proxy_cur = _scalar(last.get("synthetic_leverage_proxy", np.nan))

    has_fundamental = not np.isnan(leverage_ratio_cur)
    using_synthetic = not has_fundamental

    current = {
        "leverage_ratio": leverage_ratio_cur,
        "leverage_regime": leverage_regime_cur,
        "leverage_change_1y": leverage_change_1y_cur,
        "leverage_stress_signal": leverage_stress_cur,
        "synthetic_leverage_proxy": synthetic_proxy_cur,
        "using_synthetic": using_synthetic,
    }

    # ------------------------------------------------------------------
    # Historical slice
    # ------------------------------------------------------------------
    hist_cols = [c for c in ["leverage_stress_signal", "synthetic_leverage_proxy"]
                 if c in enriched.columns]
    historical = enriched[hist_cols].tail(_HIST_ROWS).copy()

    # ------------------------------------------------------------------
    # Cycle phase
    # ------------------------------------------------------------------
    change = leverage_change_1y_cur
    ratio = leverage_ratio_cur

    if np.isnan(change):
        # Derive a rough synthetic phase from the synthetic proxy trend
        if "synthetic_leverage_proxy" in enriched.columns:
            proxy_series = enriched["synthetic_leverage_proxy"].dropna()
            if len(proxy_series) >= _ROLL_WINDOW:
                change_proxy = float(proxy_series.iloc[-1]) - float(proxy_series.iloc[-_ROLL_WINDOW])
                change = change_proxy / 10.0   # rescale: proxy is 0-100, treat ~3pp change = 30 points
                ratio = float(proxy_series.iloc[-1]) / 2.0   # rescale to leverage-like level
            else:
                change = 0.0
                ratio = 50.0
        else:
            change = 0.0
            ratio = 50.0

    high_level = not np.isnan(ratio) and ratio >= LEVERAGE_THRESHOLDS["moderate"]

    if change > 2.0 and high_level:
        cycle_phase = "Leveraging Up"
    elif change > 2.0 and not high_level:
        cycle_phase = "Leveraging Up (from low base)"
    elif change < -2.0:
        cycle_phase = "Deleveraging"
    elif abs(change) <= 2.0 and high_level:
        cycle_phase = "Peak Leverage"
    else:
        cycle_phase = "Trough / Stable"

    # ------------------------------------------------------------------
    # Warning
    # ------------------------------------------------------------------
    warning = ""
    stress_val = leverage_stress_cur if not np.isnan(leverage_stress_cur) else 0.0
    if stress_val > 70:
        source_note = "synthetic HY-spread proxy" if using_synthetic else "corp_debt_to_gdp"
        warning = (
            f"Leverage stress signal is elevated at {stress_val:.0f}/100 "
            f"(source: {source_note}). High leverage amplifies spread widening risk "
            f"in a downturn — current phase: {cycle_phase}."
        )

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    if using_synthetic:
        interp = (
            f"No FRED leverage fundamentals available; synthetic proxy "
            f"(HY-spread percentile rank) reads {synthetic_proxy_cur:.0f}/100 — "
            f"cycle phase: {cycle_phase}."
        )
    else:
        change_str = (
            f"{leverage_change_1y_cur:+.1f}pp over the past year"
            if not np.isnan(leverage_change_1y_cur)
            else "change n/a"
        )
        interp = (
            f"Corporate leverage (debt/GDP) stands at {leverage_ratio_cur:.1f}% "
            f"({leverage_regime_cur} regime, {change_str}), "
            f"placing the economy in a '{cycle_phase}' phase of the leverage cycle."
        )

    return {
        "df": enriched,
        "current": current,
        "historical": historical,
        "cycle_phase": cycle_phase,
        "available": True,
        "interpretation": interp,
        "warning": warning,
    }
