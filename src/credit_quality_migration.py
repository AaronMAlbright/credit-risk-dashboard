"""
Credit Quality Migration — upgrade/downgrade ratio proxy and BBB cliff risk.

Track the upgrade/downgrade balance as a leading indicator for spread direction.
Since actual rating-action data is unavailable, we proxy migration pressure
through the relative velocity of HY vs IG spread changes and the
IG/HY level ratio as a BBB cliff risk signal.

Upgrade / Downgrade proxy
-------------------------
  When HY spreads are tightening *faster* than IG spreads, the net flow of
  rating actions is biased toward upgrades (quality migration up).  When HY
  widens faster than IG, downgrades and fallen-angel risk dominate.

  hy_ig_relative_velocity = hy_spread.pct_change(21) - ig_spread.pct_change(21)
    < 0  →  HY outperforming  →  upgrade pressure
    > 0  →  HY underperforming →  downgrade pressure

  Rolling 252-day z-score → migration_zscore.

BBB Cliff proxy
---------------
  bbb_cliff_proxy = ig_spread / hy_spread
    Higher ratio → IG is relatively expensive vs HY → IG buyers crowd into
    BBBs → on any downgrade they become forced sellers → elevated cliff risk.

  Rolling 504-day percentile rank → cliff_pct.

Migration regimes (based on migration_zscore)
---------------------------------------------
  "Upgrade Cycle"       zscore < -1.0
  "Stable"              -1.0 <= zscore <= 1.0
  "Downgrade Pressure"  zscore > 1.0
  "Fallen Angel Risk"   zscore > 2.0  (override)

Cliff risk flag
---------------
  True when cliff_pct > 70th percentile AND hy_spread is within 15% of
  its trailing 52-week high.

Public API
----------
  MIGRATION_WINDOWS     = [21, 63, 126]
  CLIFF_RISK_THRESHOLD  = 0.70

  compute_migration_proxy(df)       -> pd.DataFrame
  get_current_migration(df, enriched) -> dict
  run_credit_quality_migration(df)  -> dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIGRATION_WINDOWS: list[int] = [21, 63, 126]   # 1M, 3M, 6M lookback
CLIFF_RISK_THRESHOLD: float = 0.70              # 70th percentile triggers flag

_ZSCORE_WINDOW: int = 252    # rolling window for migration z-score
_CLIFF_WINDOW: int  = 504    # rolling window for cliff percentile rank
_HIST_ROWS: int     = 504    # rows returned in "historical" / signal keys
_REGIME_ROWS: int   = 252    # rows returned in regime_history
_HY_52W: int        = 252    # 52-week window for HY high check
_HY_CLIFF_PCT: float = 0.15  # within 15 % of 52-week high = stressed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return the named column if it exists and has data; else NaN series."""
    if col in df.columns and df[col].notna().any():
        return df[col].copy()
    return pd.Series(np.nan, index=df.index)


def _rolling_zscore(series: pd.Series, window: int = _ZSCORE_WINDOW) -> pd.Series:
    """Rolling z-score with a trailing window; NaN where fewer than window//2 obs."""
    min_p = max(20, window // 2)
    roll  = series.rolling(window, min_periods=min_p)
    mean  = roll.mean()
    std   = roll.std(ddof=1).replace(0, np.nan)
    return (series - mean) / std


def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling percentile rank of the current observation within the trailing
    *window* values (including the current value).

    Returns values in [0, 1].  NaN where fewer than window // 2 obs exist.
    """
    min_p = max(20, window // 2)

    def _rank(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        if len(valid) < 2:
            return np.nan
        # how many values <= current (last element), divided by count
        current = valid[-1]
        return float((valid <= current).mean())

    return series.rolling(window, min_periods=min_p).apply(_rank, raw=True)


def _migration_regime(zscore: float) -> str:
    """Map migration_zscore to a qualitative regime label."""
    if np.isnan(zscore):
        return "Stable"
    if zscore > 2.0:
        return "Fallen Angel Risk"
    if zscore > 1.0:
        return "Downgrade Pressure"
    if zscore < -1.0:
        return "Upgrade Cycle"
    return "Stable"


def _safe_float(val) -> float | None:
    """Convert a scalar to Python float; return None for NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# compute_migration_proxy
# ---------------------------------------------------------------------------

def compute_migration_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with credit quality migration columns appended.

    Required source columns
    -----------------------
    hy_spread : HY credit spread in bps
    ig_spread : IG credit spread in bps

    New columns
    -----------
    migration_velocity_21d   : pct_change(21) of HY minus pct_change(21) of IG
    migration_velocity_63d   : pct_change(63) of HY minus pct_change(63) of IG
    migration_velocity_126d  : pct_change(126) of HY minus pct_change(126) of IG
    migration_zscore         : rolling 252d z-score of migration_velocity_21d
    migration_regime         : "Upgrade Cycle" / "Stable" / "Downgrade Pressure"
                               / "Fallen Angel Risk"
    bbb_cliff_proxy          : ig_spread / hy_spread
    cliff_pct                : rolling 504d percentile rank of bbb_cliff_proxy (0–1)
    cliff_risk_flag          : bool — True when cliff_pct > threshold AND hy near 52w high
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index)

    hy = _safe_col(out, "hy_spread")
    ig = _safe_col(out, "ig_spread")

    hy_safe = hy.replace(0, np.nan)
    ig_safe = ig.replace(0, np.nan)

    # ---- migration velocity for each window -----------------------------------
    for w in MIGRATION_WINDOWS:
        hy_pc = hy_safe.pct_change(w)
        ig_pc = ig_safe.pct_change(w)
        out[f"migration_velocity_{w}d"] = hy_pc - ig_pc

    # ---- migration z-score (on 21d velocity) ----------------------------------
    vel_21 = out["migration_velocity_21d"]
    if vel_21.notna().sum() >= _ZSCORE_WINDOW // 2:
        out["migration_zscore"] = _rolling_zscore(vel_21, _ZSCORE_WINDOW)
    else:
        out["migration_zscore"] = np.nan

    # ---- migration regime -----------------------------------------------------
    out["migration_regime"] = out["migration_zscore"].apply(
        lambda v: _migration_regime(v) if pd.notna(v) else "Stable"
    )

    # ---- BBB cliff proxy = ig / hy --------------------------------------------
    out["bbb_cliff_proxy"] = ig_safe / hy_safe

    # ---- rolling percentile rank (504 days) -----------------------------------
    cliff_series = out["bbb_cliff_proxy"]
    if cliff_series.notna().sum() >= _CLIFF_WINDOW // 2:
        out["cliff_pct"] = _rolling_percentile_rank(cliff_series, _CLIFF_WINDOW)
    else:
        out["cliff_pct"] = np.nan

    # ---- cliff risk flag ------------------------------------------------------
    # Condition 1: cliff_pct exceeds threshold
    cliff_pct_flag = out["cliff_pct"] > CLIFF_RISK_THRESHOLD

    # Condition 2: HY spread is within HY_CLIFF_PCT of its trailing 52-week high
    hy_52w_high = hy_safe.rolling(_HY_52W, min_periods=max(20, _HY_52W // 4)).max()
    hy_near_high = hy_safe >= (hy_52w_high * (1.0 - _HY_CLIFF_PCT))

    out["cliff_risk_flag"] = (cliff_pct_flag & hy_near_high).fillna(False).astype(bool)

    return out


# ---------------------------------------------------------------------------
# get_current_migration
# ---------------------------------------------------------------------------

def get_current_migration(
    df: pd.DataFrame,
    enriched: pd.DataFrame | None = None,
) -> dict:
    """Return the current (latest-row) migration state as a flat dict.

    Parameters
    ----------
    df       : original master DataFrame (used if *enriched* is None)
    enriched : pre-computed result of compute_migration_proxy(df); avoids
               recomputation when the caller already has it.

    Returns
    -------
    {
        available            : bool,
        migration_zscore     : float | None,
        migration_regime     : str,
        migration_velocity_21d : float | None,
        bbb_cliff_proxy      : float | None,
        cliff_pct            : float | None,
        cliff_risk_flag      : bool,
        hy_spread            : float | None,
        ig_spread            : float | None,
        hy_ig_ratio          : float | None,
        warning              : str | None,
    }
    """
    try:
        if df is None or df.empty:
            return {"available": False}

        has_hy = "hy_spread" in df.columns and df["hy_spread"].notna().any()
        has_ig = "ig_spread" in df.columns and df["ig_spread"].notna().any()
        if not (has_hy and has_ig):
            return {"available": False}

        if enriched is None:
            enriched = compute_migration_proxy(df)

        last = enriched.iloc[-1]

        def _get(col: str) -> float | None:
            if col not in enriched.columns:
                return None
            return _safe_float(last[col])

        def _get_bool(col: str) -> bool:
            if col not in enriched.columns:
                return False
            v = last[col]
            if isinstance(v, (bool, np.bool_)):
                return bool(v)
            return False

        def _get_str(col: str, default: str = "Unknown") -> str:
            if col not in enriched.columns:
                return default
            v = last[col]
            return str(v) if pd.notna(v) else default

        hy_val  = _get("hy_spread")
        ig_val  = _get("ig_spread")

        # hy_ig_ratio for convenience (hy / ig)
        hy_ig_ratio: float | None = None
        if hy_val is not None and ig_val is not None and ig_val != 0:
            hy_ig_ratio = hy_val / ig_val

        zscore       = _get("migration_zscore")
        regime       = _get_str("migration_regime", "Stable")
        vel_21       = _get("migration_velocity_21d")
        cliff_proxy  = _get("bbb_cliff_proxy")
        cliff_pct    = _get("cliff_pct")
        cliff_flag   = _get_bool("cliff_risk_flag")

        # ---- warning string ---------------------------------------------------
        warning: str | None = None
        if cliff_flag:
            parts = []
            if cliff_pct is not None:
                parts.append(
                    f"BBB cliff proxy is at the {cliff_pct * 100:.0f}th historical "
                    f"percentile (>{CLIFF_RISK_THRESHOLD * 100:.0f}th threshold)"
                )
            if hy_val is not None:
                parts.append(
                    f"HY spreads ({hy_val:.0f} bps) are near their 52-week high"
                )
            warning = (
                "Cliff risk flag is active: "
                + "; ".join(parts)
                + " — elevated BBB-to-HY downgrade pressure."
            )

        return {
            "available": True,
            "migration_zscore": zscore,
            "migration_regime": regime,
            "migration_velocity_21d": vel_21,
            "bbb_cliff_proxy": cliff_proxy,
            "cliff_pct": cliff_pct,
            "cliff_risk_flag": cliff_flag,
            "hy_spread": hy_val,
            "ig_spread": ig_val,
            "hy_ig_ratio": hy_ig_ratio,
            "warning": warning,
        }

    except Exception:
        return {"available": False}


# ---------------------------------------------------------------------------
# run_credit_quality_migration
# ---------------------------------------------------------------------------

def run_credit_quality_migration(df: pd.DataFrame) -> dict:
    """Top-level entry point for credit quality migration analysis.

    Minimum requirements
    --------------------
    - Columns ``hy_spread`` and ``ig_spread`` must be present with data.
    - At least 252 rows of non-NaN overlap between the two columns.

    Returns
    -------
    dict with keys:
        available              : bool
        current                : dict  (see get_current_migration)
        historical             : pd.DataFrame  — last 504 rows with migration cols
        signal_series          : pd.Series     — migration_zscore, last 504 rows
        regime_history         : pd.Series     — migration_regime, last 252 rows
        percentile_migration   : float         — where current zscore sits vs history
        interpretation         : str
    """
    try:
        if df is None or df.empty:
            return {"available": False}

        has_hy = "hy_spread" in df.columns and df["hy_spread"].notna().any()
        has_ig = "ig_spread" in df.columns and df["ig_spread"].notna().any()

        if not (has_hy and has_ig):
            return {"available": False}

        # Check minimum row requirement (252 rows with both spreads available)
        both_valid = df["hy_spread"].notna() & df["ig_spread"].notna()
        if both_valid.sum() < 252:
            return {"available": False}

        # ---- enrich -----------------------------------------------------------
        enriched = compute_migration_proxy(df)

        # ---- current state ----------------------------------------------------
        current = get_current_migration(df, enriched=enriched)
        if not current.get("available", False):
            return {"available": False}

        # ---- historical slice (last 504 rows) ---------------------------------
        migration_cols = [
            "migration_velocity_21d", "migration_velocity_63d",
            "migration_velocity_126d", "migration_zscore",
            "migration_regime", "bbb_cliff_proxy", "cliff_pct", "cliff_risk_flag",
        ]
        hist_cols = [c for c in migration_cols if c in enriched.columns]
        historical = enriched[hist_cols].tail(_HIST_ROWS).copy()

        # ---- signal series (zscore, last 504 rows) ----------------------------
        signal_series: pd.Series
        if "migration_zscore" in enriched.columns:
            signal_series = enriched["migration_zscore"].tail(_HIST_ROWS).copy()
            signal_series.name = "migration_zscore"
        else:
            signal_series = pd.Series(dtype=float, name="migration_zscore")

        # ---- regime history (last 252 rows) -----------------------------------
        regime_history: pd.Series
        if "migration_regime" in enriched.columns:
            regime_history = enriched["migration_regime"].tail(_REGIME_ROWS).copy()
            regime_history.name = "migration_regime"
        else:
            regime_history = pd.Series(dtype=str, name="migration_regime")

        # ---- percentile of current zscore vs full history --------------------
        percentile_migration: float = float("nan")
        zscore_now = current.get("migration_zscore")
        if zscore_now is not None and "migration_zscore" in enriched.columns:
            full_z = enriched["migration_zscore"].dropna()
            if len(full_z) > 0:
                percentile_migration = float((full_z < zscore_now).mean() * 100.0)

        # ---- interpretation ---------------------------------------------------
        regime    = current.get("migration_regime", "Stable")
        zscore    = current.get("migration_zscore")
        vel_21    = current.get("migration_velocity_21d")
        cliff_pct = current.get("cliff_pct")
        cliff_flag = current.get("cliff_risk_flag", False)

        parts: list[str] = []

        # Regime headline
        if zscore is not None:
            parts.append(
                f"The migration z-score of {zscore:.2f} places credit quality in "
                f"the '{regime}' regime"
                + (
                    f" ({percentile_migration:.0f}th historical percentile)"
                    if not np.isnan(percentile_migration)
                    else ""
                )
                + "."
            )
        else:
            parts.append(f"Credit quality migration regime: {regime}.")

        # Velocity context
        if vel_21 is not None:
            direction = (
                "HY spreads are compressing faster than IG (upgrade pressure)"
                if vel_21 < 0
                else "HY spreads are widening faster than IG (downgrade pressure)"
            )
            parts.append(f"Over the past month, {direction} (velocity: {vel_21:+.3f}).")

        # Cliff risk
        if cliff_flag:
            cp_str = (
                f" at the {cliff_pct * 100:.0f}th percentile" if cliff_pct is not None else ""
            )
            parts.append(
                f"The BBB cliff risk flag is active — IG/HY ratio{cp_str}, "
                "with HY spreads near their 52-week high, signalling elevated "
                "fallen-angel downgrade pressure."
            )
        elif cliff_pct is not None:
            cp_val = cliff_pct * 100.0
            if cp_val > 50:
                parts.append(
                    f"The BBB cliff proxy is elevated ({cp_val:.0f}th percentile) "
                    "but has not triggered the cliff risk flag."
                )

        interpretation = " ".join(parts)

        return {
            "available": True,
            "current": current,
            "historical": historical,
            "signal_series": signal_series,
            "regime_history": regime_history,
            "percentile_migration": percentile_migration,
            "interpretation": interpretation,
        }

    except Exception:
        return {"available": False}
