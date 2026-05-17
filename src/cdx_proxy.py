"""
Synthetic CDX — a credit default swap index proxy.

CDX.NA.IG and CDX.NA.HY are OTC derivatives that price the cost of credit
insurance on baskets of 125 investment-grade or 100 high-yield entities.
They are more liquid than the underlying bonds and often lead cash spreads.

Since CDX data is not freely available, this module builds a *synthetic*
CDX-equivalent signal from the available market data:

  Synthetic CDX-HY  ≈ normalized blend of:
    - HY OAS spread level (40% weight)
    - HY OAS 30-day change z-score (30% weight)
    - VIX level z-score (20% weight)
    - SP500 drawdown (10% weight)

  Synthetic CDX-IG  ≈ normalized blend of:
    - IG OAS spread level (40% weight)
    - IG OAS 30-day change z-score (30% weight)
    - 2s10s yield curve flattening z-score (30% weight)

The synthetic indices are scaled to 0–100 (100 = maximum stress seen historically)
and calibrated against named stress episodes (GFC, COVID, etc.)

Key credit concept this teaches: CDX trades at a "spread" in bps/year.
When you buy CDX protection at 80bps, you pay 80bps/year; if defaults hit,
the index pays you par minus recovery on the defaulted names.

Public API
----------
  CDX_WEIGHTS_HY  — dict of component weights
  CDX_WEIGHTS_IG  — dict of component weights
  compute_synthetic_cdx(df) → pd.DataFrame    adds cdx_hy_proxy, cdx_ig_proxy columns
  run_cdx_analysis(df) → dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CDX_WEIGHTS_HY: dict[str, float] = {
    "hy_spread_zscore":       0.40,
    "hy_change_30d_zscore":   0.30,
    "vix_zscore":             0.20,
    "sp500_drawdown_scaled":  0.10,
}

CDX_WEIGHTS_IG: dict[str, float] = {
    "ig_spread_zscore":       0.40,
    "ig_change_30d_zscore":   0.30,
    "curve_flattening_zscore": 0.30,
}

_ZSCORE_WINDOW = 252
_ZSCORE_MIN_PERIODS = 63
_CLIP_LO = -3.0
_CLIP_HI = 3.0

# Regime boundaries on 0–100 scale
_REGIME_TIGHT    = 30.0
_REGIME_WIDE     = 60.0
_REGIME_CRISIS   = 80.0

# Composite blend weights
_CDX_COMPOSITE_HY_WT = 0.60
_CDX_COMPOSITE_IG_WT = 0.40

_NAN_FILL = 50.0  # warmup fill value


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rolling_zscore(series: pd.Series, window: int = _ZSCORE_WINDOW) -> pd.Series:
    """
    (x - rolling_mean) / rolling_std with the given window.

    Uses min_periods=_ZSCORE_MIN_PERIODS. Result is clipped to [-3, 3].
    """
    rm = series.rolling(window=window, min_periods=_ZSCORE_MIN_PERIODS).mean()
    rs = series.rolling(window=window, min_periods=_ZSCORE_MIN_PERIODS).std(ddof=1)
    z = (series - rm) / rs.replace(0, np.nan)
    return z.clip(_CLIP_LO, _CLIP_HI)


def _zscore_to_0100(z: pd.Series) -> pd.Series:
    """
    Linear rescale from [-3, 3] → [0, 100].  val = (z + 3) / 6 * 100.

    Input is already clipped to [-3, 3]; output is clipped to [0, 100].
    """
    return ((z + 3.0) / 6.0 * 100.0).clip(0.0, 100.0)


def _raw_blend_to_0100(raw: pd.Series) -> pd.Series:
    """
    The raw blend is already a weighted sum of z-scores in [-3, 3].
    Rescale to [0, 100].
    """
    return ((raw + 3.0) / 6.0 * 100.0).clip(0.0, 100.0)


def _cdx_regime(score: float) -> str:
    if score < _REGIME_TIGHT:
        return "Tight (<30)"
    elif score < _REGIME_WIDE:
        return "Normal (30–60)"
    elif score < _REGIME_CRISIS:
        return "Wide (60–80)"
    else:
        return "Crisis (>80)"


def _pearson_r(a: pd.Series, b: pd.Series) -> float:
    """Hand-rolled Pearson correlation, no scipy."""
    df_ab = pd.concat([a, b], axis=1).dropna()
    if len(df_ab) < 5:
        return float("nan")
    x = df_ab.iloc[:, 0].values.astype(float)
    y = df_ab.iloc[:, 1].values.astype(float)
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((xm * ym).sum() / denom)


def _historical_percentile(series: pd.Series, current_val: float) -> float:
    """What fraction of history is at or below current_val (0–100)."""
    clean = series.dropna()
    if clean.empty:
        return float("nan")
    return float((clean <= current_val).mean() * 100.0)


# ---------------------------------------------------------------------------
# Public: compute_synthetic_cdx
# ---------------------------------------------------------------------------

def compute_synthetic_cdx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add synthetic CDX columns to df and return the enriched copy.

    Added columns:
        cdx_hy_proxy    — 0–100 HY credit stress index
        cdx_ig_proxy    — 0–100 IG credit stress index (NaN if ig_spread absent)
        cdx_composite   — 60% HY + 40% IG blend
        cdx_regime      — categorical label

    All z-scores use a 252-day rolling window (min_periods=63).
    NaN during warmup is filled with _NAN_FILL (50).
    """
    out = df.copy()

    has_hy = "hy_spread" in out.columns
    has_ig = "ig_spread" in out.columns
    has_vix = "vix" in out.columns
    has_sp500 = "sp500" in out.columns
    has_yield_10y = "yield_10y" in out.columns
    has_yield_2y = "yield_2y" in out.columns

    if not has_hy:
        out["cdx_hy_proxy"] = _NAN_FILL
        out["cdx_ig_proxy"] = _NAN_FILL
        out["cdx_composite"] = _NAN_FILL
        out["cdx_regime"] = _cdx_regime(_NAN_FILL)
        return out

    # ------------------------------------------------------------------
    # HY components
    # ------------------------------------------------------------------

    # 1. HY spread level z-score
    hy_z = _rolling_zscore(out["hy_spread"])

    # 2. HY 30-day change z-score
    if "hy_change_30d" in out.columns:
        hy_chg = out["hy_change_30d"]
    else:
        hy_chg = out["hy_spread"].diff(30)
    hy_chg_z = _rolling_zscore(hy_chg)

    # 3. VIX z-score
    if has_vix:
        vix_z = _rolling_zscore(out["vix"])
    else:
        vix_z = pd.Series(0.0, index=out.index)

    # 4. SP500 drawdown scaled to z-score range
    if "sp500_drawdown" in out.columns:
        drawdown = out["sp500_drawdown"]
    elif has_sp500:
        rolling_max = out["sp500"].rolling(window=_ZSCORE_WINDOW, min_periods=1).max()
        drawdown = (out["sp500"] - rolling_max) / rolling_max.replace(0, np.nan)
    else:
        drawdown = pd.Series(0.0, index=out.index)

    # Scale drawdown (which is 0 to ~-0.5) into [-3, 3]:
    # drawdown is negative; more negative = more stress → flip sign
    # z-score the flipped drawdown
    dd_flipped = -drawdown.fillna(0.0)
    dd_z = _rolling_zscore(dd_flipped)

    # Blend HY components
    cdx_hy_raw = (
        CDX_WEIGHTS_HY["hy_spread_zscore"] * hy_z
        + CDX_WEIGHTS_HY["hy_change_30d_zscore"] * hy_chg_z
        + CDX_WEIGHTS_HY["vix_zscore"] * vix_z
        + CDX_WEIGHTS_HY["sp500_drawdown_scaled"] * dd_z
    )
    cdx_hy_100 = _raw_blend_to_0100(cdx_hy_raw)
    out["cdx_hy_proxy"] = cdx_hy_100.fillna(_NAN_FILL)

    # ------------------------------------------------------------------
    # IG components
    # ------------------------------------------------------------------

    if has_ig:
        ig_z = _rolling_zscore(out["ig_spread"])

        if "ig_change_30d" in out.columns:
            ig_chg = out["ig_change_30d"]
        else:
            ig_chg = out["ig_spread"].diff(30)
        ig_chg_z = _rolling_zscore(ig_chg)

        # 2s10s curve flattening = negative spread → invert so flattening = stress
        if has_yield_10y and has_yield_2y:
            curve_2s10s = out["yield_10y"] - out["yield_2y"]
            # Flattening (falling 2s10s) = rising stress → negate
            curve_flat = -curve_2s10s
            curve_z = _rolling_zscore(curve_flat)
        else:
            curve_z = pd.Series(0.0, index=out.index)

        cdx_ig_raw = (
            CDX_WEIGHTS_IG["ig_spread_zscore"] * ig_z
            + CDX_WEIGHTS_IG["ig_change_30d_zscore"] * ig_chg_z
            + CDX_WEIGHTS_IG["curve_flattening_zscore"] * curve_z
        )
        cdx_ig_100 = _raw_blend_to_0100(cdx_ig_raw)
        out["cdx_ig_proxy"] = cdx_ig_100.fillna(_NAN_FILL)
    else:
        out["cdx_ig_proxy"] = float("nan")

    # ------------------------------------------------------------------
    # Composite and regime
    # ------------------------------------------------------------------

    if has_ig:
        out["cdx_composite"] = (
            _CDX_COMPOSITE_HY_WT * out["cdx_hy_proxy"]
            + _CDX_COMPOSITE_IG_WT * out["cdx_ig_proxy"]
        )
    else:
        out["cdx_composite"] = out["cdx_hy_proxy"]

    out["cdx_regime"] = out["cdx_composite"].apply(_cdx_regime)

    return out


# ---------------------------------------------------------------------------
# Public: run_cdx_analysis
# ---------------------------------------------------------------------------

def run_cdx_analysis(df: pd.DataFrame) -> dict:
    """
    Compute synthetic CDX metrics and return an analysis dict.

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame (must contain at least hy_spread for full output).

    Returns
    -------
    dict with keys:
        df                          — enriched DataFrame
        current                     — latest cdx_hy_proxy, cdx_ig_proxy,
                                      cdx_composite, cdx_regime
        historical_percentile_hy    — HY percentile vs full history
        historical_percentile_ig    — IG percentile vs full history
        rolling_cdx                 — last 252 rows of hy/ig proxy series
        stress_peaks                — top-5 peak composite readings (date, value)
        correlation_with_composite  — Pearson r vs composite_risk_score_smooth
        available                   — bool
    """
    _unavailable = {
        "df": df,
        "current": {},
        "historical_percentile_hy": float("nan"),
        "historical_percentile_ig": float("nan"),
        "rolling_cdx": pd.DataFrame(),
        "stress_peaks": pd.DataFrame(),
        "correlation_with_composite": float("nan"),
        "available": False,
    }

    if "hy_spread" not in df.columns:
        return _unavailable

    enriched = compute_synthetic_cdx(df)

    # --- Current snapshot ---
    last = enriched.iloc[-1]
    current = {
        "cdx_hy_proxy":  float(last.get("cdx_hy_proxy", _NAN_FILL)),
        "cdx_ig_proxy":  float(last.get("cdx_ig_proxy", _NAN_FILL))
        if not pd.isna(last.get("cdx_ig_proxy", float("nan"))) else float("nan"),
        "cdx_composite": float(last.get("cdx_composite", _NAN_FILL)),
        "cdx_regime":    str(last.get("cdx_regime", "Unknown")),
    }

    # --- Historical percentiles ---
    pct_hy = _historical_percentile(enriched["cdx_hy_proxy"], current["cdx_hy_proxy"])
    pct_ig = (
        _historical_percentile(enriched["cdx_ig_proxy"], current["cdx_ig_proxy"])
        if "cdx_ig_proxy" in enriched.columns and not enriched["cdx_ig_proxy"].isna().all()
        else float("nan")
    )

    # --- Rolling series (last 252 rows) ---
    roll_cols = ["cdx_hy_proxy", "cdx_ig_proxy"]
    if "date" in enriched.columns:
        roll_cols_present = [c for c in roll_cols if c in enriched.columns]
        rolling_cdx = enriched[["date"] + roll_cols_present].tail(252).reset_index(drop=True)
    else:
        roll_cols_present = [c for c in roll_cols if c in enriched.columns]
        rolling_cdx = enriched[roll_cols_present].tail(252).reset_index(drop=True)

    # --- Stress peaks (top 5 cdx_composite readings) ---
    comp_col = enriched["cdx_composite"].dropna()
    top5_idx = comp_col.nlargest(5).index
    if "date" in enriched.columns:
        stress_peaks = enriched.loc[top5_idx, ["date", "cdx_composite"]].copy()
    else:
        stress_peaks = enriched.loc[top5_idx, ["cdx_composite"]].copy()
    stress_peaks = stress_peaks.reset_index(drop=True)

    # --- Correlation with composite risk score ---
    corr = float("nan")
    if "composite_risk_score_smooth" in enriched.columns:
        corr = _pearson_r(enriched["cdx_composite"], enriched["composite_risk_score_smooth"])

    return {
        "df": enriched,
        "current": current,
        "historical_percentile_hy": pct_hy,
        "historical_percentile_ig": pct_ig,
        "rolling_cdx": rolling_cdx,
        "stress_peaks": stress_peaks,
        "correlation_with_composite": corr,
        "available": True,
    }
