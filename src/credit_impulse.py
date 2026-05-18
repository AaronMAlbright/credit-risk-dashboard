"""
Credit impulse — rate of change of new credit flow relative to GDP.

Credit impulse = Δ(New Credit Flow / GDP), not the level.
Concept from Michael Biggs (Deutsche Bank): measures the *acceleration* of credit
growth, not the stock. A slowing of credit expansion is contractionary even if
total credit is still rising.

Lead time: 6–18 months ahead of corporate default rates and spread cycles.
Positive impulse → credit being extended faster → growth/spread supportive.
Negative impulse → credit contraction → leading indicator of recession and defaults.

Data sourcing (FRED via fredapi, lazy-imported):
  TOTLL          — Total Loans and Leases (US commercial banks, weekly, NSA)
  CONSUMER       — Total Consumer Credit (monthly)
  GDP            — Nominal GDP (quarterly, forward-filled to match credit frequency)
  BOGZ1FL144104005Q — Business debt, nonfinancial corporate (quarterly, optional)

Fallback: if no FRED key or fetch fails, use df columns total_credit / gdp if
present, or proxy from available macro columns (hy_spread inversion as last resort).

Computation (pure numpy/pandas):
  1. total_credit = TOTLL + CONSUMER (or best available)
  2. new_credit_flow = total_credit.diff(63)              — quarterly change in stock
  3. credit_impulse_raw = new_credit_flow / GDP           — as fraction of GDP
  4. credit_impulse = credit_impulse_raw.rolling(63).mean() — smoothed
  5. impulse_zscore = rolling z-score vs 252-day window
  6. impulse_score = percentile rank vs full history × 100

Public API
----------
  FRED_SERIES            : dict
  IMPULSE_WINDOW         : int
  compute_credit_impulse(df) -> pd.DataFrame
  get_current_impulse(df)    -> dict
  run_credit_impulse_analysis(df) -> dict
"""

from __future__ import annotations

from datetime import datetime, date

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRED_SERIES: dict[str, str] = {
    "total_loans":      "TOTLL",
    "consumer_credit":  "CONSUMER",
    "gdp":              "GDP",
    "business_debt":    "BOGZ1FL144104005Q",
}

IMPULSE_WINDOW: int = 63

_MIN_ROWS: int = 252
_HIST_ROWS: int = 504
_ZSCORE_WINDOW: int = 252
_MOMENTUM_WINDOW: int = 21
_LEAD_MONTHS: int = 6     # 6-month forward shift for correlation (≈126 trading days)
_LEAD_DAYS: int = 126

# Score thresholds
_SCORE_EXPANDING: float = 50.0
_SCORE_STRONG: float = 70.0
_SCORE_WEAK: float = 30.0

_FRED_START: str = "1990-01-01"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except Exception:
        return None


def _rolling_zscore(series: pd.Series, window: int = _ZSCORE_WINDOW) -> pd.Series:
    roll = series.rolling(window, min_periods=window // 3)
    mean = roll.mean()
    std = roll.std(ddof=1).replace(0, np.nan)
    return (series - mean) / std


def _percentile_rank(series: pd.Series) -> pd.Series:
    """Expanding percentile rank (0–100) — where each value stands vs full history to that point."""
    def _pct(x: pd.Series) -> float:
        if x.isna().all():
            return np.nan
        v = x.iloc[-1]
        if np.isnan(v):
            return np.nan
        clean = x.dropna()
        return float((clean < v).sum() / len(clean) * 100.0)

    return series.expanding(min_periods=max(10, IMPULSE_WINDOW)).apply(_pct, raw=False)


def _impulse_direction(chg_63: float | None) -> str:
    if chg_63 is None or np.isnan(chg_63):
        return "Unknown"
    return "Expanding" if chg_63 > 0 else "Contracting"


def _impulse_momentum(chg_21: float | None) -> str:
    if chg_21 is None or np.isnan(chg_21):
        return "Stable"
    if chg_21 > 0.0002:
        return "Accelerating"
    if chg_21 < -0.0002:
        return "Decelerating"
    return "Stable"


def _lead_signal(score: float) -> str:
    if np.isnan(score):
        return "Insufficient Data"
    if score < _SCORE_WEAK:
        return "Spread Widening Risk (6-18mo lead)"
    if score > _SCORE_STRONG:
        return "Spread Tightening Tailwind"
    return "Neutral — No Strong Directional Signal"


# ---------------------------------------------------------------------------
# FRED fetch
# ---------------------------------------------------------------------------

def _fetch_fred_series() -> dict[str, pd.Series]:
    """Attempt to fetch TOTLL, CONSUMER, GDP from FRED.

    Returns dict of {key: pd.Series with DatetimeIndex}, may be partially filled.
    Returns empty dict on any failure.
    """
    try:
        import os
        api_key = os.getenv("FRED_API_KEY", "")
        if not api_key:
            return {}

        from fredapi import Fred
        fred = Fred(api_key=api_key)

        results: dict[str, pd.Series] = {}

        for key, series_id in [
            ("total_loans",     FRED_SERIES["total_loans"]),
            ("consumer_credit", FRED_SERIES["consumer_credit"]),
            ("gdp",             FRED_SERIES["gdp"]),
        ]:
            try:
                raw = fred.get_series(series_id, observation_start=_FRED_START)
                if raw is not None and len(raw) > 0:
                    results[key] = raw.astype(float)
            except Exception:
                continue

        return results

    except Exception:
        return {}


def _build_credit_series_from_fred(fred_data: dict[str, pd.Series]) -> tuple[pd.Series | None, pd.Series | None]:
    """Combine FRED series into daily (total_credit, gdp) aligned series.

    Returns (total_credit, gdp) — either may be None if data missing.
    """
    try:
        loans = fred_data.get("total_loans")
        consumer = fred_data.get("consumer_credit")
        gdp = fred_data.get("gdp")

        if loans is None and consumer is None:
            return None, None

        # Build a daily date range covering available data
        all_series = [s for s in [loans, consumer, gdp] if s is not None]
        idx_min = min(s.index.min() for s in all_series)
        idx_max = max(s.index.max() for s in all_series)
        daily_idx = pd.date_range(idx_min, idx_max, freq="B")

        def _to_daily(s: pd.Series | None) -> pd.Series | None:
            if s is None:
                return None
            s = s.copy()
            s.index = pd.to_datetime(s.index)
            s = s.reindex(daily_idx).ffill()
            return s

        loans_d = _to_daily(loans)
        consumer_d = _to_daily(consumer)
        gdp_d = _to_daily(gdp)

        if loans_d is not None and consumer_d is not None:
            total_credit = (loans_d + consumer_d).dropna()
        elif loans_d is not None:
            total_credit = loans_d.dropna()
        elif consumer_d is not None:
            total_credit = consumer_d.dropna()
        else:
            return None, None

        return total_credit, gdp_d

    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# compute_credit_impulse
# ---------------------------------------------------------------------------

def compute_credit_impulse(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich a copy of *df* with credit impulse columns.

    Tries FRED first, then falls back to df columns.

    New columns added
    -----------------
    credit_impulse_raw        : Δ(new_credit_flow / GDP), smoothed 63d rolling mean
    credit_impulse_zscore     : rolling z-score vs 252d window
    credit_impulse_score      : 0–100 percentile rank vs full history
    credit_impulse_direction  : "Expanding" / "Contracting" per row
    credit_impulse_regime     : same — alias kept for regime_history extraction
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index)

    impulse_raw: pd.Series | None = None

    # --- attempt 1: FRED -------------------------------------------------------
    try:
        fred_data = _fetch_fred_series()
        if fred_data:
            total_credit, gdp = _build_credit_series_from_fred(fred_data)
            if total_credit is not None and gdp is not None and len(total_credit) > IMPULSE_WINDOW * 2:
                new_credit_flow = total_credit.diff(IMPULSE_WINDOW)
                gdp_aligned = gdp.reindex(new_credit_flow.index).ffill().replace(0, np.nan)
                raw_ratio = new_credit_flow / gdp_aligned
                smoothed = raw_ratio.rolling(IMPULSE_WINDOW, min_periods=IMPULSE_WINDOW // 3).mean()
                smoothed.index = pd.to_datetime(smoothed.index)
                # Align to df index — reindex then ffill up to 63 business days
                impulse_raw = smoothed.reindex(out.index.union(smoothed.index)).ffill(limit=IMPULSE_WINDOW).reindex(out.index)
    except Exception:
        impulse_raw = None

    # --- attempt 2: df columns total_credit + gdp --------------------------------
    if impulse_raw is None or impulse_raw.notna().sum() < _MIN_ROWS // 2:
        try:
            if "total_credit" in out.columns and "gdp" in out.columns:
                tc = out["total_credit"].astype(float)
                gdp_col = out["gdp"].astype(float).replace(0, np.nan).ffill()
                new_flow = tc.diff(IMPULSE_WINDOW)
                raw_ratio = new_flow / gdp_col
                impulse_raw = raw_ratio.rolling(IMPULSE_WINDOW, min_periods=IMPULSE_WINDOW // 3).mean()
        except Exception:
            pass

    # --- attempt 3: proxy from hy_spread (sign-inverted; credit tightening = tighter spreads) --
    if impulse_raw is None or impulse_raw.notna().sum() < _MIN_ROWS // 2:
        try:
            if "hy_spread" in out.columns and out["hy_spread"].notna().sum() >= _MIN_ROWS // 2:
                # Invert: spread tightening = credit expanding → positive impulse proxy
                s = -out["hy_spread"].astype(float)
                chg = s.diff(IMPULSE_WINDOW)
                impulse_raw = chg.rolling(IMPULSE_WINDOW, min_periods=IMPULSE_WINDOW // 3).mean()
                # Normalise to small magnitudes consistent with real impulse
                impulse_std = impulse_raw.std()
                if impulse_std and impulse_std > 0:
                    impulse_raw = impulse_raw / (impulse_std * 10.0)
        except Exception:
            pass

    if impulse_raw is None:
        impulse_raw = pd.Series(np.nan, index=out.index)

    impulse_raw = impulse_raw.reindex(out.index)

    out["credit_impulse_raw"] = impulse_raw
    out["credit_impulse_zscore"] = _rolling_zscore(impulse_raw, _ZSCORE_WINDOW)
    out["credit_impulse_score"] = _percentile_rank(impulse_raw)

    chg_63 = impulse_raw.diff(IMPULSE_WINDOW)
    out["credit_impulse_direction"] = chg_63.apply(
        lambda v: _impulse_direction(_safe_float(v))
    )
    out["credit_impulse_regime"] = out["credit_impulse_direction"]

    return out


# ---------------------------------------------------------------------------
# get_current_impulse
# ---------------------------------------------------------------------------

def get_current_impulse(df: pd.DataFrame) -> dict:
    """Return the current credit impulse state as a flat dict.

    Parameters
    ----------
    df : master DataFrame with DatetimeIndex. If credit_impulse_* columns are
         already present they are used directly; otherwise compute_credit_impulse
         is called internally.

    Returns
    -------
    dict — see module docstring for key list.
    """
    try:
        if df is None or df.empty:
            return {"available": False}

        if "credit_impulse_raw" not in df.columns:
            enriched = compute_credit_impulse(df)
        else:
            enriched = df

        if enriched["credit_impulse_raw"].notna().sum() < 10:
            return {"available": False}

        last = enriched.iloc[-1]

        score = _safe_float(last.get("credit_impulse_score"))
        raw = _safe_float(last.get("credit_impulse_raw"))
        zscore = _safe_float(last.get("credit_impulse_zscore"))

        chg_63 = None
        if "credit_impulse_raw" in enriched.columns:
            impulse_series = enriched["credit_impulse_raw"].dropna()
            if len(impulse_series) >= IMPULSE_WINDOW + 1:
                last_val = impulse_series.iloc[-1]
                prev_val = impulse_series.iloc[-IMPULSE_WINDOW - 1] if len(impulse_series) > IMPULSE_WINDOW else impulse_series.iloc[0]
                chg_63 = float(last_val - prev_val)

        chg_21 = None
        if "credit_impulse_raw" in enriched.columns:
            impulse_series = enriched["credit_impulse_raw"].dropna()
            if len(impulse_series) >= _MOMENTUM_WINDOW + 1:
                chg_21 = float(impulse_series.iloc[-1] - impulse_series.iloc[-_MOMENTUM_WINDOW - 1])

        direction = _impulse_direction(chg_63)
        momentum = _impulse_momentum(chg_21)
        l_signal = _lead_signal(score if score is not None else float("nan"))

        # Percentile vs history — same as score but stated explicitly
        pct_vs_history = score

        # Build interpretation
        if score is not None and raw is not None:
            state_word = "expansionary" if direction == "Expanding" else "contractionary"
            interpretation = (
                f"Credit impulse is {state_word} (score {score:.0f}/100, "
                f"raw impulse {raw:.4f}). "
                f"Momentum is {momentum.lower()}. "
                f"Lead signal: {l_signal}. "
                f"The impulse Z-score vs 252-day history is {zscore:.2f}."
                if zscore is not None else
                f"Credit impulse is {state_word} (score {score:.0f}/100). "
                f"Momentum is {momentum.lower()}. Lead signal: {l_signal}."
            )
        else:
            interpretation = "Credit impulse data unavailable — insufficient history or missing macro series."

        warning: str | None = None
        if score is not None and score < _SCORE_WEAK:
            warning = (
                f"Credit impulse score is {score:.0f}/100, in the bottom third of history. "
                "Contractionary credit impulse historically precedes spread widening and "
                "elevated default rates with a 6–18 month lead time."
            )

        return {
            "available": True,
            "credit_impulse_score": score,
            "credit_impulse_raw": raw,
            "credit_impulse_zscore": zscore,
            "impulse_direction": direction,
            "impulse_momentum": momentum,
            "lead_signal": l_signal,
            "impulse_pct_vs_history": pct_vs_history,
            "interpretation": interpretation,
            "warning": warning,
        }

    except Exception:
        return {"available": False}


# ---------------------------------------------------------------------------
# run_credit_impulse_analysis
# ---------------------------------------------------------------------------

def run_credit_impulse_analysis(df: pd.DataFrame) -> dict:
    """Top-level entry point for credit impulse analysis.

    Parameters
    ----------
    df : master DataFrame with DatetimeIndex and at least 252 rows.

    Returns
    -------
    dict with keys:
        available            : bool
        current              : dict from get_current_impulse()
        df                   : pd.DataFrame — enriched copy with impulse columns
        impulse_history      : pd.Series — raw impulse values (DatetimeIndex)
        impulse_score_history: pd.Series — 0–100 score (DatetimeIndex)
        lead_correlation     : float — corr(impulse, hy_spread shifted -126 days)
        regime_history       : pd.Series — "Expanding"/"Contracting" (DatetimeIndex)
        interpretation       : str
    """
    try:
        if df is None or df.empty:
            return {"available": False}

        if len(df) < _MIN_ROWS:
            return {"available": False}

        enriched = compute_credit_impulse(df)

        valid_impulse = enriched["credit_impulse_raw"].notna().sum()
        if valid_impulse < _MIN_ROWS // 2:
            return {"available": False}

        current = get_current_impulse(enriched)
        if not current.get("available"):
            return {"available": False}

        impulse_history = enriched["credit_impulse_raw"].tail(_HIST_ROWS).copy()
        impulse_history.name = "credit_impulse_raw"

        impulse_score_history = enriched["credit_impulse_score"].tail(_HIST_ROWS).copy()
        impulse_score_history.name = "credit_impulse_score"

        regime_history = enriched["credit_impulse_regime"].tail(_HIST_ROWS).copy()
        regime_history.name = "credit_impulse_regime"

        # Lead correlation: impulse vs hy_spread 6 months forward
        lead_correlation: float = float("nan")
        try:
            if "hy_spread" in enriched.columns:
                impulse_clean = enriched["credit_impulse_raw"].dropna()
                fwd_spread = enriched["hy_spread"].shift(-_LEAD_DAYS)
                aligned = pd.concat(
                    {"impulse": impulse_clean, "fwd_spread": fwd_spread},
                    axis=1,
                ).dropna()
                if len(aligned) >= 126:
                    lead_correlation = float(
                        np.corrcoef(aligned["impulse"].values, aligned["fwd_spread"].values)[0, 1]
                    )
        except Exception:
            pass

        score_now = current.get("credit_impulse_score")
        direction = current.get("impulse_direction", "Unknown")
        l_signal = current.get("lead_signal", "")
        momentum = current.get("impulse_momentum", "Stable")

        parts: list[str] = []
        if score_now is not None:
            parts.append(
                f"Credit impulse score: {score_now:.0f}/100 ({direction}, {momentum}). "
                f"Lead signal: {l_signal}."
            )
        else:
            parts.append("Credit impulse: data marginal.")

        if not np.isnan(lead_correlation):
            direction_word = "positive" if lead_correlation > 0 else "negative"
            parts.append(
                f"Historical correlation of impulse with 6-month forward HY spread: "
                f"{lead_correlation:+.3f} ({direction_word} relationship, "
                f"n={_LEAD_DAYS}-day lead)."
            )

        interpretation = " ".join(parts)

        return {
            "available": True,
            "current": current,
            "df": enriched,
            "impulse_history": impulse_history,
            "impulse_score_history": impulse_score_history,
            "lead_correlation": lead_correlation,
            "regime_history": regime_history,
            "interpretation": interpretation,
        }

    except Exception:
        return {"available": False}
