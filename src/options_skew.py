"""
Options market skew — tail risk pricing from equity derivatives.

SKEW Index (CBOE): Measures the market's implied probability of outlier moves
  in the S&P 500. Unlike VIX (which measures level of ATM vol), SKEW measures
  the SHAPE of the implied vol curve — specifically how much more expensive
  OTM puts are relative to ATM options.

  SKEW = 100 → log-normal distribution (no tail premium)
  SKEW = 130 → elevated tail risk
  SKEW = 150 → high concern about left-tail events
  SKEW > 150 → extreme tail pricing (rare; seen before major dislocations)

VVIX (Vol of Vol): Measures the 30-day expected volatility of VIX.
  High VVIX = market is uncertain about future fear levels.
  VVIX > 100 typically precedes VIX spikes.

Key insight for credit: High SKEW + high VIX = "we see the tail risk AND we're
already scared." High SKEW + low VIX = "hidden danger" — market is complacent
on surface but pricing extreme tails. The latter is often more dangerous.

Public API
----------
  SKEW_NORMAL   = 115.0
  SKEW_ELEVATED = 130.0
  SKEW_HIGH     = 145.0
  SKEW_EXTREME  = 155.0
  get_skew_snapshot() → dict
  compute_skew_signal(df) → pd.DataFrame   adds skew_ columns
  run_skew_analysis(df) → dict
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

SKEW_NORMAL: float = 115.0
SKEW_ELEVATED: float = 130.0
SKEW_HIGH: float = 145.0
SKEW_EXTREME: float = 155.0

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_close(tickers: list[str], period: str = "60d") -> pd.DataFrame:
    """Download Close prices for *tickers*, return a DataFrame indexed by date."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = tickers

    close.index = pd.to_datetime(close.index).normalize()
    return close


def _latest(series: pd.Series) -> Optional[float]:
    """Return the last non-NaN value from a Series, or None."""
    s = series.dropna()
    return float(s.iloc[-1]) if not s.empty else None


def _classify_skew_regime(skew: float) -> str:
    if skew < SKEW_NORMAL:
        return "Normal (<115)"
    elif skew < SKEW_ELEVATED:
        return "Elevated (115–130)"
    elif skew < SKEW_HIGH:
        return "High (130–145)"
    else:
        return "Extreme (>145)"


def _classify_vvix_regime(vvix: float) -> str:
    if vvix < 80:
        return "Normal (<80)"
    elif vvix < 100:
        return "Elevated (80–100)"
    else:
        return "High (>100)"


def _skew_to_signal(skew: float) -> float:
    """
    Piecewise-linear mapping of SKEW level → 0–100 stress score.

    Breakpoints (chosen to match credit stress thresholds):
      SKEW ≤ 100  → 0
      100–115     → 0–15   (approaching normal tail concern)
      115–130     → 15–40  (elevated)
      130–145     → 40–70  (high)
      145–160     → 70–90  (extreme)
      ≥ 160       → 90–100 (capped)
    """
    if skew <= 100:
        return 0.0
    elif skew < 115:
        return (skew - 100) / 15.0 * 15.0
    elif skew < 130:
        return 15.0 + (skew - 115) / 15.0 * 25.0
    elif skew < 145:
        return 40.0 + (skew - 130) / 15.0 * 30.0
    elif skew < 160:
        return 70.0 + (skew - 145) / 15.0 * 20.0
    else:
        return min(90.0 + (skew - 160) / 10.0 * 10.0, 100.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_skew_snapshot() -> dict:
    """
    Fetch current SKEW and VVIX snapshot from live market data.

    Returns
    -------
    dict with keys:
        skew              : float — current SKEW level
        vvix              : float — current VVIX level
        skew_regime       : str   — regime label
        skew_30d_avg      : float — 30-day rolling average of SKEW
        skew_vs_30d_avg   : float — current SKEW minus 30d average
        skew_percentile_60d : float — current SKEW percentile vs last 60 days
        hidden_danger     : bool  — True when SKEW > 140 and VIX < 20
        vvix_regime       : str   — VVIX regime label
        available         : bool
        error             : str or None
    """
    try:
        closes = _fetch_close(["^SKEW", "^VVIX", "^VIX"], period="60d")
        if closes.empty:
            return _error_snapshot("yfinance returned empty data")

        skew_series = closes.get("^SKEW", pd.Series(dtype=float)).dropna()
        vvix_series = closes.get("^VVIX", pd.Series(dtype=float)).dropna()
        vix_series = closes.get("^VIX", pd.Series(dtype=float)).dropna()

        skew = _latest(skew_series)
        vvix = _latest(vvix_series)
        vix = _latest(vix_series)

        if skew is None:
            return _error_snapshot("Could not retrieve SKEW value")

        # 30-day average (use last 21 trading days ≈ 30 calendar days)
        skew_30d = skew_series.tail(21)
        skew_30d_avg = float(skew_30d.mean()) if not skew_30d.empty else None
        skew_vs_30d_avg = (skew - skew_30d_avg) if skew_30d_avg is not None else None

        # Percentile of current SKEW vs full 60-day history
        all_skew = skew_series.values
        if len(all_skew) > 0:
            skew_percentile_60d = float(np.mean(all_skew <= skew) * 100)
        else:
            skew_percentile_60d = None

        # Hidden danger: high tail pricing with low VIX
        hidden_danger = bool(skew > 140 and vix is not None and vix < 20)

        return {
            "skew": skew,
            "vvix": vvix,
            "skew_regime": _classify_skew_regime(skew),
            "skew_30d_avg": skew_30d_avg,
            "skew_vs_30d_avg": skew_vs_30d_avg,
            "skew_percentile_60d": skew_percentile_60d,
            "hidden_danger": hidden_danger,
            "vvix_regime": _classify_vvix_regime(vvix) if vvix is not None else "Unknown",
            "available": True,
            "error": None,
        }

    except Exception as exc:  # noqa: BLE001
        return _error_snapshot(str(exc))


def _error_snapshot(msg: str) -> dict:
    return {
        "skew": None,
        "vvix": None,
        "skew_regime": None,
        "skew_30d_avg": None,
        "skew_vs_30d_avg": None,
        "skew_percentile_60d": None,
        "hidden_danger": None,
        "vvix_regime": None,
        "available": False,
        "error": msg,
    }


def compute_skew_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge SKEW-based signals onto *df* by date.

    Expects *df* to have a "date" column (or DatetimeIndex).  Adds columns:
        skew_level          — SKEW index level on that date
        skew_regime         — regime label string
        skew_signal         — 0–100 stress score
        skew_hidden_danger  — bool: SKEW > 140 and VIX < 20
        skew_zscore_60d     — rolling 60-day z-score of SKEW level

    If SKEW data cannot be fetched, the above columns are added as NaN and
    the original DataFrame is returned otherwise unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a "date" column or DatetimeIndex.
        Optionally contains a "vix" column for hidden_danger computation.

    Returns
    -------
    pd.DataFrame — copy of df with skew_* columns appended.
    """
    df = df.copy()

    if "date" in df.columns:
        df_dates = pd.to_datetime(df["date"]).dt.normalize()
    else:
        df_dates = pd.to_datetime(df.index).normalize()

    nan_cols = [
        "skew_level",
        "skew_regime",
        "skew_signal",
        "skew_hidden_danger",
        "skew_zscore_60d",
    ]

    try:
        closes = _fetch_close(["^SKEW"], period="60d")
        if closes.empty or "^SKEW" not in closes.columns:
            raise ValueError("SKEW data unavailable")

        skew_series = closes["^SKEW"].dropna()
        skew_series.index = pd.to_datetime(skew_series.index).normalize()

        skew_map = skew_series.to_dict()
        df["skew_level"] = df_dates.map(skew_map)

        df["skew_regime"] = df["skew_level"].apply(
            lambda s: _classify_skew_regime(s) if pd.notna(s) else np.nan
        )
        df["skew_signal"] = df["skew_level"].apply(
            lambda s: _skew_to_signal(s) if pd.notna(s) else np.nan
        )

        # Hidden danger: skew > 140 AND vix < 20
        if "vix" in df.columns:
            vix_num = pd.to_numeric(df["vix"], errors="coerce")
            df["skew_hidden_danger"] = (df["skew_level"] > 140) & (vix_num < 20)
        else:
            df["skew_hidden_danger"] = np.nan

        # Rolling 60-day z-score of SKEW level
        window = 60
        if df["skew_level"].notna().sum() >= 10:
            rolling_mean = df["skew_level"].rolling(window, min_periods=10).mean()
            rolling_std = df["skew_level"].rolling(window, min_periods=10).std()
            # Avoid division by zero when std is 0
            df["skew_zscore_60d"] = (df["skew_level"] - rolling_mean) / rolling_std.replace(
                0, np.nan
            )
        else:
            df["skew_zscore_60d"] = np.nan

    except Exception:  # noqa: BLE001
        for col in nan_cols:
            df[col] = np.nan

    return df


def run_skew_analysis(df: pd.DataFrame) -> dict:
    """
    Run a full SKEW analysis combining a live snapshot with historical signals.

    Parameters
    ----------
    df : pd.DataFrame
        Historical data frame.  Expected columns (used when present):
            date, vix, hy_spread, composite_risk_score_smooth

    Returns
    -------
    dict with keys:
        snapshot                  : get_skew_snapshot() result
        df                        : compute_skew_signal(df) result
        available                 : bool
        correlation_with_composite : float or None — Pearson r vs composite score
        correlation_with_hy       : float or None — Pearson r vs HY spread
        hidden_danger_episodes    : pd.DataFrame — past hidden-danger periods
        interpretation            : str — plain-English summary
    """
    snapshot = get_skew_snapshot()
    enriched_df = compute_skew_signal(df)
    available = snapshot["available"]

    # -----------------------------------------------------------------------
    # Correlation: skew_level vs composite_risk_score_smooth (last 252 rows)
    # -----------------------------------------------------------------------
    corr_composite: Optional[float] = None
    if "skew_level" in enriched_df.columns and "composite_risk_score_smooth" in enriched_df.columns:
        window = enriched_df.tail(252)[["skew_level", "composite_risk_score_smooth"]].dropna()
        if len(window) >= 10:
            corr_composite = float(window["skew_level"].corr(window["composite_risk_score_smooth"]))

    # -----------------------------------------------------------------------
    # Correlation: skew_level vs hy_spread (last 252 rows)
    # -----------------------------------------------------------------------
    corr_hy: Optional[float] = None
    if "skew_level" in enriched_df.columns and "hy_spread" in enriched_df.columns:
        window = enriched_df.tail(252)[["skew_level", "hy_spread"]].dropna()
        if len(window) >= 10:
            corr_hy = float(window["skew_level"].corr(window["hy_spread"]))

    # -----------------------------------------------------------------------
    # Hidden danger episodes with subsequent HY spread change
    # -----------------------------------------------------------------------
    hidden_danger_episodes = _extract_hidden_danger_episodes(enriched_df)

    # -----------------------------------------------------------------------
    # Plain-English interpretation
    # -----------------------------------------------------------------------
    interpretation = _build_interpretation(snapshot)

    return {
        "snapshot": snapshot,
        "df": enriched_df,
        "available": available,
        "correlation_with_composite": corr_composite,
        "correlation_with_hy": corr_hy,
        "hidden_danger_episodes": hidden_danger_episodes,
        "interpretation": interpretation,
    }


def _extract_hidden_danger_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame of hidden-danger=True rows with the date and
    the subsequent 5-day change in hy_spread (if available).

    Columns: date, skew_level, vix (if present), hy_spread_change_5d
    """
    if "skew_hidden_danger" not in df.columns:
        return pd.DataFrame()

    # Identify hidden-danger rows
    danger_mask = df["skew_hidden_danger"].fillna(False).astype(bool)
    if not danger_mask.any():
        return pd.DataFrame()

    # Build output
    cols = ["skew_hidden_danger"]
    if "date" in df.columns:
        cols = ["date"] + cols
    if "skew_level" in df.columns:
        cols.append("skew_level")
    if "vix" in df.columns:
        cols.append("vix")

    episodes = df.loc[danger_mask, [c for c in cols if c in df.columns]].copy()

    # Add forward 5-day HY spread change
    if "hy_spread" in df.columns:
        hy = df["hy_spread"].reset_index(drop=True)
        fwd_change = []
        for idx in df.index[danger_mask]:
            iloc_pos = df.index.get_loc(idx)
            fwd_pos = iloc_pos + 5
            if fwd_pos < len(hy):
                fwd_change.append(float(hy.iloc[fwd_pos] - hy.iloc[iloc_pos]))
            else:
                fwd_change.append(np.nan)
        episodes["hy_spread_change_5d"] = fwd_change
    else:
        episodes["hy_spread_change_5d"] = np.nan

    return episodes.reset_index(drop=True)


def _build_interpretation(snapshot: dict) -> str:
    """
    Compose a one-sentence plain-English summary of the current SKEW/VVIX state.
    """
    if not snapshot.get("available"):
        err = snapshot.get("error", "data unavailable")
        return f"SKEW data could not be retrieved ({err})."

    skew = snapshot["skew"]
    vvix = snapshot["vvix"]
    regime = snapshot["skew_regime"]
    hidden = snapshot["hidden_danger"]
    vvix_regime = snapshot["vvix_regime"]
    vs_avg = snapshot.get("skew_vs_30d_avg")

    # Build sentence components
    skew_part = f"SKEW is {skew:.1f} ({regime})"
    avg_part = ""
    if vs_avg is not None:
        direction = "above" if vs_avg > 0 else "below"
        avg_part = f", {abs(vs_avg):.1f} pts {direction} its 30-day average"

    vvix_part = ""
    if vvix is not None:
        vvix_part = f"; VVIX is {vvix:.1f} ({vvix_regime})"

    if hidden:
        tail_part = (
            " — a 'hidden danger' setup where tail risk is aggressively priced "
            "despite low realized fear (VIX<20), historically a precursor to sharp credit spread widening"
        )
    else:
        tail_part = ""

    return f"{skew_part}{avg_part}{vvix_part}{tail_part}."
