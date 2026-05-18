"""
Systematic Deleveraging Detector.

Identifies when quantitative/systematic strategies are force-selling by
detecting the "correlation-to-1" phenomenon: cross-asset correlation spike,
vol expansion, and simultaneous drawdown across equities, credit, and rates.

The signal is triggered when:
  1. Rolling 21d cross-asset correlation spikes (assets that normally move
     independently start moving together).
  2. Realized volatility is elevated above its 63d average.
  3. Multiple assets are simultaneously in drawdown.

Public API
----------
  compute_deleveraging_composite(df)    -> pd.DataFrame
  run_systematic_deleveraging(df)       -> dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

DELEV_ASSETS = [
    ("sp500",     "Equities",  True),    # True  = drawdown when price falls
    ("hy_spread", "HY Credit", False),   # False = drawdown when spread rises
    ("ig_spread", "IG Credit", False),
    ("yield_10y", "10y Rates", True),    # rates falling = flight-to-safety context
]

CORR_WINDOW      = 21    # trading days for rolling correlation
VOL_WINDOW       = 21    # realized vol window
VOL_LOOKBACK     = 63    # baseline window for vol comparison
DRAWDOWN_THRESHOLD = 0.02  # 2% move counts as "in drawdown"

_MIN_ROWS        = 126   # minimum rows required for the analysis
_MIN_ASSETS      = 2     # minimum asset columns required
_HIST_ROWS       = 504   # ~2 trading years for historical slices
_ALERT_HIST_ROWS = 252   # ~1 trading year for alert history
_PEAK_EVENTS     = 5     # number of top events to surface


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return df[col] if present, else a NaN Series aligned to df.index."""
    if col in df.columns:
        return df[col].copy()
    return pd.Series(float("nan"), index=df.index, name=col)


def _percentile_rank_series(s: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling percentile rank (0–100) of s within its trailing *window* values.
    Uses expanding window until at least *window* observations are available.
    """
    def _rank_scalar(arr: np.ndarray) -> float:
        v = arr[-1]
        if np.isnan(v):
            return float("nan")
        valid = arr[~np.isnan(arr)]
        if len(valid) < 2:
            return float("nan")
        return float(np.sum(valid < v) / (len(valid) - 1) * 100.0)

    out = s.rolling(window, min_periods=max(10, window // 10)).apply(
        _rank_scalar, raw=True
    )
    return out


def _rolling_pearson(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Rolling Pearson correlation — pure pandas, no scipy."""
    return x.rolling(window, min_periods=max(5, window // 4)).corr(y)


# ---------------------------------------------------------------------------
# Asset returns
# ---------------------------------------------------------------------------

def _asset_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily returns for each deleveraging asset.

    For price assets (invert=True): pct_change().
    For spread assets (invert=False): diff() / prior level.

    Returns a DataFrame with one column per available asset, named by asset
    name (e.g. "Equities", "HY Credit").
    """
    frames: dict[str, pd.Series] = {}
    for col, label, invert in DELEV_ASSETS:
        if col not in df.columns:
            continue
        raw = df[col].copy()
        if invert:
            ret = raw.pct_change()
        else:
            prior = raw.shift(1)
            ret = raw.diff() / prior.where(prior != 0).abs()
        frames[label] = ret
    if not frames:
        return pd.DataFrame(index=df.index)
    return pd.DataFrame(frames, index=df.index)


# ---------------------------------------------------------------------------
# Sub-score: cross-asset correlation
# ---------------------------------------------------------------------------

def compute_correlation_score(returns_df: pd.DataFrame) -> pd.Series:
    """
    Rolling 21d average absolute pairwise correlation across assets,
    percentile-ranked to 0–100.

    Steps:
      1. For every pair of asset return columns, compute 21d rolling Pearson r.
      2. Take abs(r) for each pair, average all pairs.
      3. Percentile-rank within trailing 252d to normalise to 0–100.
    """
    cols = returns_df.columns.tolist()
    if len(cols) < 2:
        return pd.Series(float("nan"), index=returns_df.index, name="delev_corr_score")

    pair_corrs: list[pd.Series] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = _rolling_pearson(returns_df[cols[i]], returns_df[cols[j]], CORR_WINDOW)
            pair_corrs.append(r.abs())

    avg_abs_corr = pd.concat(pair_corrs, axis=1).mean(axis=1)
    score = _percentile_rank_series(avg_abs_corr, window=252)
    score.name = "delev_corr_score"
    return score


# ---------------------------------------------------------------------------
# Sub-score: vol expansion
# ---------------------------------------------------------------------------

def compute_vol_expansion_score(returns_df: pd.DataFrame) -> pd.Series:
    """
    Vol expansion: 21d realised vol / 63d realised vol − 1, averaged across
    assets, then linearly scaled 0–100 where:
      ratio > +0.5  → 100
      ratio < -0.2  → 0
      linear between
    """
    cols = returns_df.columns.tolist()
    if not cols:
        return pd.Series(float("nan"), index=returns_df.index, name="delev_vol_score")

    expansion_series: list[pd.Series] = []
    for c in cols:
        s = returns_df[c].dropna()
        if s.empty:
            continue
        aligned = returns_df[c]
        vol_short = aligned.rolling(VOL_WINDOW, min_periods=max(5, VOL_WINDOW // 2)).std()
        vol_long  = aligned.rolling(VOL_LOOKBACK, min_periods=max(10, VOL_LOOKBACK // 3)).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                vol_long > 0,
                vol_short / vol_long - 1.0,
                float("nan"),
            )
        expansion_series.append(pd.Series(ratio, index=returns_df.index))

    if not expansion_series:
        return pd.Series(float("nan"), index=returns_df.index, name="delev_vol_score")

    avg_expansion = pd.concat(expansion_series, axis=1).mean(axis=1)

    def _scale(x: float) -> float:
        if np.isnan(x):
            return float("nan")
        if x >= 0.5:
            return 100.0
        if x <= -0.2:
            return 0.0
        # linear: -0.2 → 0, 0.5 → 100
        return float((x - (-0.2)) / (0.5 - (-0.2)) * 100.0)

    score = avg_expansion.map(_scale)
    score.name = "delev_vol_score"
    return score


# ---------------------------------------------------------------------------
# Sub-score: simultaneous drawdown
# ---------------------------------------------------------------------------

def compute_drawdown_score(df: pd.DataFrame) -> pd.Series:
    """
    Count assets simultaneously in drawdown, scaled 0–100.

    Drawdown definitions:
      price assets (invert=True):  current < 21d rolling max * (1 - threshold)
      spread assets (invert=False): current > 21d rolling min * (1 + threshold)
    """
    available_assets = [(col, label, inv) for col, label, inv in DELEV_ASSETS
                        if col in df.columns]

    if not available_assets:
        return pd.Series(float("nan"), index=df.index, name="delev_drawdown_score")

    max_count = len(available_assets)
    in_dd_frames: list[pd.Series] = []

    for col, label, invert in available_assets:
        raw = df[col]
        if invert:
            rolling_high = raw.rolling(CORR_WINDOW, min_periods=max(5, CORR_WINDOW // 2)).max()
            in_dd = (raw < rolling_high * (1.0 - DRAWDOWN_THRESHOLD)).astype(float)
        else:
            rolling_low = raw.rolling(CORR_WINDOW, min_periods=max(5, CORR_WINDOW // 2)).min()
            in_dd = (raw > rolling_low * (1.0 + DRAWDOWN_THRESHOLD)).astype(float)
        # Where rolling window couldn't compute, leave NaN
        in_dd = in_dd.where(raw.notna())
        in_dd_frames.append(in_dd.rename(label))

    dd_count = pd.concat(in_dd_frames, axis=1).sum(axis=1, min_count=1)
    score = (dd_count / max_count * 100.0).clip(0.0, 100.0)
    score.name = "delev_drawdown_score"
    return score


# ---------------------------------------------------------------------------
# Alert level
# ---------------------------------------------------------------------------

def _alert_level(composite: float) -> str:
    if np.isnan(composite):
        return "No Signal"
    if composite >= 80.0:
        return "Systematic Deleveraging Alert"
    if composite >= 60.0:
        return "Deleveraging Warning"
    if composite >= 30.0:
        return "Elevated Correlation"
    return "No Signal"


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def compute_deleveraging_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with columns appended:
      delev_corr_score, delev_vol_score, delev_drawdown_score,
      delev_composite, delev_alert_level
    """
    out = df.copy()
    returns_df = _asset_returns(df)

    corr_score = compute_correlation_score(returns_df)
    vol_score  = compute_vol_expansion_score(returns_df)
    dd_score   = compute_drawdown_score(df)

    out["delev_corr_score"]     = corr_score
    out["delev_vol_score"]      = vol_score
    out["delev_drawdown_score"] = dd_score

    sub_scores = pd.concat(
        [out["delev_corr_score"], out["delev_vol_score"], out["delev_drawdown_score"]],
        axis=1,
    )
    out["delev_composite"] = sub_scores.mean(axis=1, skipna=True)
    out["delev_alert_level"] = out["delev_composite"].map(
        lambda x: _alert_level(x) if not pd.isna(x) else "No Signal"
    )
    return out


# ---------------------------------------------------------------------------
# Assets currently in drawdown (for current snapshot)
# ---------------------------------------------------------------------------

def _assets_in_drawdown_now(df: pd.DataFrame) -> list[str]:
    """Return labels of assets currently in drawdown per their last available row."""
    result: list[str] = []
    for col, label, invert in DELEV_ASSETS:
        if col not in df.columns:
            continue
        raw = df[col].dropna()
        if len(raw) < CORR_WINDOW:
            continue
        current = float(raw.iloc[-1])
        if invert:
            high_21 = float(raw.tail(CORR_WINDOW).max())
            if current < high_21 * (1.0 - DRAWDOWN_THRESHOLD):
                result.append(label)
        else:
            low_21 = float(raw.tail(CORR_WINDOW).min())
            if current > low_21 * (1.0 + DRAWDOWN_THRESHOLD):
                result.append(label)
    return result


# ---------------------------------------------------------------------------
# Peak events
# ---------------------------------------------------------------------------

def _find_peak_events(enriched: pd.DataFrame, n: int = _PEAK_EVENTS) -> list[dict]:
    """
    Return the top *n* highest delev_composite readings, deduplicated by a
    21-day minimum gap so they represent distinct stress episodes.
    """
    if "delev_composite" not in enriched.columns:
        return []

    series = enriched["delev_composite"].dropna()
    if series.empty:
        return []

    peaks: list[dict] = []
    already_used: list[pd.Timestamp] = []

    sorted_idx = series.sort_values(ascending=False).index

    for ts in sorted_idx:
        if len(peaks) >= n:
            break
        # check 21-day gap from already-used dates
        too_close = any(
            abs((ts - used).days) < CORR_WINDOW for used in already_used
        )
        if too_close:
            continue
        already_used.append(ts)
        row = enriched.loc[ts]
        peaks.append(
            {
                "date": ts,
                "composite": round(float(series.loc[ts]), 1),
                "alert_level": str(row.get("delev_alert_level", _alert_level(series.loc[ts]))),
            }
        )

    return peaks


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_systematic_deleveraging(df: pd.DataFrame) -> dict:
    """
    Top-level entry. Requires >= 2 available asset columns and >= 126 rows.

    Returns
    -------
    {
        available: bool,
        current: {
            delev_composite: float,
            alert_level: str,
            corr_score: float,
            vol_score: float,
            drawdown_score: float,
            assets_in_drawdown: list[str],
            n_assets: int,
        },
        historical: pd.DataFrame,   # last 504 rows with all delev_ columns
        signal_series: pd.Series,   # delev_composite last 504 rows
        alert_history: pd.Series,   # delev_alert_level last 252 rows
        peak_events: list[dict],    # top 5 highest delev_composite events
        interpretation: str,
        warning: str or None,
    }
    """
    try:
        # ---- Guard: minimum viable data ------------------------------------
        available_cols = [col for col, _, _ in DELEV_ASSETS if col in df.columns]
        n_assets = len(available_cols)

        if n_assets < _MIN_ASSETS or len(df) < _MIN_ROWS:
            return {"available": False}

        # ---- Compute all deleveraging columns ------------------------------
        enriched = compute_deleveraging_composite(df)

        # ---- Current snapshot ----------------------------------------------
        def _last_valid(col: str) -> float:
            s = enriched[col].dropna()
            return float(s.iloc[-1]) if not s.empty else float("nan")

        composite    = _last_valid("delev_composite")
        corr_score   = _last_valid("delev_corr_score")
        vol_score    = _last_valid("delev_vol_score")
        dd_score     = _last_valid("delev_drawdown_score")
        alert        = _alert_level(composite)
        assets_in_dd = _assets_in_drawdown_now(df)

        current = {
            "delev_composite": round(composite, 1) if not np.isnan(composite) else None,
            "alert_level":     alert,
            "corr_score":      round(corr_score, 1) if not np.isnan(corr_score) else None,
            "vol_score":       round(vol_score, 1)  if not np.isnan(vol_score) else None,
            "drawdown_score":  round(dd_score, 1)   if not np.isnan(dd_score) else None,
            "assets_in_drawdown": assets_in_dd,
            "n_assets": n_assets,
        }

        # ---- Historical slices ---------------------------------------------
        delev_cols = [
            "delev_corr_score", "delev_vol_score",
            "delev_drawdown_score", "delev_composite", "delev_alert_level",
        ]
        present_cols = [c for c in delev_cols if c in enriched.columns]
        historical    = enriched[present_cols].tail(_HIST_ROWS).copy()
        signal_series = enriched["delev_composite"].tail(_HIST_ROWS).copy()
        alert_history = enriched["delev_alert_level"].tail(_ALERT_HIST_ROWS).copy()

        # ---- Peak events ---------------------------------------------------
        peak_events = _find_peak_events(enriched)

        # ---- Interpretation ------------------------------------------------
        if np.isnan(composite):
            interp = (
                "Insufficient data to compute the systematic deleveraging signal."
            )
        else:
            asset_labels = [label for _, label, _ in DELEV_ASSETS if _ or True
                            if (_ is True or _ is False)
                            if (col := next((c for c, l, _ in DELEV_ASSETS if l == label), None))
                            and col in df.columns]
            # simpler rebuild:
            asset_labels = [label for col, label, _ in DELEV_ASSETS if col in df.columns]
            dd_str = (
                f"{', '.join(assets_in_dd)} in drawdown" if assets_in_dd
                else "no assets in drawdown"
            )
            interp = (
                f"Systematic deleveraging composite is {composite:.0f}/100 "
                f"({alert}). "
                f"Correlation score {corr_score:.0f}, vol expansion {vol_score:.0f}, "
                f"drawdown score {dd_score:.0f} — {dd_str}. "
                f"Monitoring {n_assets} asset{'s' if n_assets != 1 else ''}: "
                f"{', '.join(asset_labels)}."
            )

        # ---- Warning -------------------------------------------------------
        warning: str | None = None
        if alert in ("Deleveraging Warning", "Systematic Deleveraging Alert"):
            n_dd = len(assets_in_dd)
            warning = (
                f"{alert}: cross-asset correlations spiking (score {corr_score:.0f}), "
                f"{n_dd} asset{'s' if n_dd != 1 else ''} in simultaneous drawdown. "
                f"Forced selling by systematic strategies may amplify spread widening."
            )

        return {
            "available":    True,
            "current":      current,
            "historical":   historical,
            "signal_series": signal_series,
            "alert_history": alert_history,
            "peak_events":  peak_events,
            "interpretation": interp,
            "warning":      warning,
        }

    except Exception:
        return {"available": False}
