"""
Rolling equity-credit correlation regime detector.

The correlation between equity returns and credit spread changes is normally
negative (stocks up → spreads tight). When this correlation breaks down
or goes positive, it signals a risk-off dislocation — historically this
precedes credit stress episodes by 2–4 weeks.

"Correlation regime" breakdown:
  Strongly negative (< -0.3):  Normal, equity-credit moving together (flight to quality works)
  Weakly negative (-0.3 to 0): Decoupling — early warning
  Positive (> 0):              Dislocation — both moving adversely; systemic risk elevated

Public API
----------
  WINDOWS = [21, 63, 90, 252]   rolling windows in trading days
  compute_rolling_correlations(df) → pd.DataFrame    adds corr_ columns
  detect_correlation_regimes(df) → pd.DataFrame      adds corr_regime columns
  run_correlation_analysis(df) → dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOWS: list[int] = [21, 63, 90, 252]

_MIN_PERIODS = 20          # minimum observations before computing correlation
_PRIMARY_WINDOW = 90       # window used for regime classification
_DISLOCATION_THRESHOLD = 0.0    # corr > this → dislocation
_DECOUPLING_THRESHOLD = -0.3    # corr > this AND <= 0 → decoupling


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rolling_pearson(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling Pearson correlation between two series using only
    numpy/pandas (no scipy).

    Returns NaN wherever fewer than _MIN_PERIODS joint observations exist.
    """
    # Align on index, drop where either is NaN
    combined = pd.concat([x.rename("x"), y.rename("y")], axis=1)

    def _pearson_window(arr: np.ndarray) -> float:
        """Compute Pearson r for a 2-column array, returning NaN as needed."""
        valid = arr[~np.isnan(arr).any(axis=1)]
        if len(valid) < _MIN_PERIODS:
            return float("nan")
        xv = valid[:, 0]
        yv = valid[:, 1]
        xd = xv - xv.mean()
        yd = yv - yv.mean()
        denom = np.sqrt((xd ** 2).sum() * (yd ** 2).sum())
        if denom == 0:
            return float("nan")
        return float(np.dot(xd, yd) / denom)

    r = combined["x"].rolling(window, min_periods=_MIN_PERIODS).corr(combined["y"])
    return r


def _get_return_series(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Derive the equity return series and HY spread change series from df.

    Equity returns
    --------------
    Uses ``sp500_daily_return`` if present; otherwise computes pct_change
    from ``sp500``.

    HY spread change
    ----------------
    Daily diff (level change in basis points) of ``hy_spread``.

    Returns (equity_returns, hy_change).  Either may be all-NaN if the
    required columns are missing.
    """
    # Equity returns
    if "sp500_daily_return" in df.columns:
        eq_ret = df["sp500_daily_return"].copy()
    elif "sp500" in df.columns:
        eq_ret = df["sp500"].pct_change()
    else:
        eq_ret = pd.Series(float("nan"), index=df.index)

    # HY spread change (daily diff in spread level)
    if "hy_spread" in df.columns:
        hy_chg = df["hy_spread"].diff()
    else:
        hy_chg = pd.Series(float("nan"), index=df.index)

    return eq_ret, hy_chg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rolling_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling equity-credit correlation columns to a copy of ``df``.

    For each window ``w`` in WINDOWS:
        corr_eq_hy_{w}d   : rolling Pearson r(sp500 return, hy_spread diff)

    Additional column:
        corr_eq_vix_90d   : rolling 90d Pearson r(sp500 return, vix pct change)
                            Normally negative; positive = stress signal.

    Minimum 20 joint observations required before a value is returned.
    """
    df = df.copy()

    eq_ret, hy_chg = _get_return_series(df)

    # Equity-HY correlations at multiple windows
    for w in WINDOWS:
        col = f"corr_eq_hy_{w}d"
        df[col] = _rolling_pearson(eq_ret, hy_chg, w)

    # Equity-VIX 90d correlation
    if "vix" in df.columns:
        vix_chg = df["vix"].pct_change()
        df["corr_eq_vix_90d"] = _rolling_pearson(eq_ret, vix_chg, 90)
    else:
        df["corr_eq_vix_90d"] = float("nan")

    return df


def detect_correlation_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add correlation-regime columns based on the 90-day equity-HY correlation.

    Requires ``corr_eq_hy_90d`` — calls ``compute_rolling_correlations`` if
    the column is absent.

    New columns
    -----------
    corr_regime_90d          : "Normal (neg)" / "Decoupling" / "Dislocation (pos)"
    corr_dislocation_flag    : 1 where corr_eq_hy_90d > 0, else 0
    corr_dislocation_streak  : consecutive days of dislocation (resets to 0 otherwise)
    """
    df = df.copy()

    if "corr_eq_hy_90d" not in df.columns:
        df = compute_rolling_correlations(df)

    corr = df["corr_eq_hy_90d"]

    # Regime label
    def _label(c: float) -> str:
        if pd.isna(c):
            return "Unknown"
        if c > _DISLOCATION_THRESHOLD:
            return "Dislocation (pos)"
        if c > _DECOUPLING_THRESHOLD:
            return "Decoupling"
        return "Normal (neg)"

    df["corr_regime_90d"] = corr.apply(_label)

    # Dislocation flag
    df["corr_dislocation_flag"] = (corr > _DISLOCATION_THRESHOLD).astype(int)

    # Dislocation streak — count consecutive days of flag == 1
    flag = df["corr_dislocation_flag"].values
    streak = np.zeros(len(flag), dtype=int)
    for i in range(len(flag)):
        if flag[i] == 1:
            streak[i] = streak[i - 1] + 1 if i > 0 else 1
        else:
            streak[i] = 0
    df["corr_dislocation_streak"] = streak

    return df


def run_correlation_analysis(df: pd.DataFrame) -> dict:
    """
    Full equity-credit correlation regime analysis.

    Returns
    -------
    dict with keys:
        df                   : DataFrame enriched by detect_correlation_regimes()
        current              : dict of latest corr_ column values + regime label
        current_streak       : int, days of active dislocation (0 if none)
        dislocation_episodes : DataFrame of historical dislocation episodes:
                               columns: start_date, end_date, duration_days,
                                        max_correlation, subsequent_hy_chg_4w
        available            : True if sp500 and hy_spread columns exist
        warning              : non-empty string if dislocation_streak > 10 days
    """
    available = (
        ("sp500" in df.columns or "sp500_daily_return" in df.columns)
        and ("hy_spread" in df.columns)
    )

    enriched = detect_correlation_regimes(df)

    # ---- Current snapshot ----
    corr_cols = [c for c in enriched.columns if c.startswith("corr_")]
    latest_row = enriched[corr_cols].iloc[-1].to_dict()

    # Scalar-safe conversion
    for k, v in latest_row.items():
        if isinstance(v, (np.floating, np.integer)):
            latest_row[k] = float(v)

    current_streak = int(enriched["corr_dislocation_streak"].iloc[-1])
    current_regime = enriched["corr_regime_90d"].iloc[-1]

    current: dict = {**latest_row, "regime": current_regime}

    # ---- Dislocation episodes ----
    episodes = _build_dislocation_episodes(enriched)

    # ---- Warning string ----
    if current_streak > 10:
        corr_val = latest_row.get("corr_eq_hy_90d", float("nan"))
        corr_str = f"{corr_val:.2f}" if not pd.isna(corr_val) else "N/A"
        warning = (
            f"DISLOCATION ACTIVE: equity-credit correlation has been positive "
            f"for {current_streak} consecutive trading days "
            f"(90d corr = {corr_str}). Historically this precedes HY spread "
            "widening by 2–4 weeks. Review portfolio positioning."
        )
    else:
        warning = ""

    return {
        "df": enriched,
        "current": current,
        "current_streak": current_streak,
        "dislocation_episodes": episodes,
        "available": available,
        "warning": warning,
    }


def _build_dislocation_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify contiguous runs where corr_dislocation_flag == 1 and summarise
    each episode.

    Returns a DataFrame with columns:
        start_date           : first date of the dislocation episode
        end_date             : last date of the dislocation episode
        duration_days        : length of episode in calendar days
        max_correlation      : peak (highest positive) 90d correlation during episode
        subsequent_hy_chg_4w : HY spread change in the 4 weeks *after* the episode
                               ends (NaN if insufficient history)
    """
    flag = df["corr_dislocation_flag"].values
    index = df.index

    episodes: list[dict] = []
    n = len(flag)
    i = 0
    while i < n:
        if flag[i] == 1:
            start_i = i
            while i < n and flag[i] == 1:
                i += 1
            end_i = i - 1  # last day of dislocation

            start_date = index[start_i]
            end_date = index[end_i]

            # Peak correlation within episode
            ep_corr = df["corr_eq_hy_90d"].iloc[start_i : end_i + 1]
            max_corr = float(ep_corr.max()) if ep_corr.notna().any() else float("nan")

            # HY spread change in the 4 weeks (~20 trading days) after episode ends
            subsequent_hy_chg: float = float("nan")
            if "hy_spread" in df.columns:
                fwd_end = min(end_i + 21, n - 1)
                if fwd_end > end_i:
                    hy_at_end = df["hy_spread"].iloc[end_i]
                    hy_fwd = df["hy_spread"].iloc[fwd_end]
                    if not (pd.isna(hy_at_end) or pd.isna(hy_fwd)):
                        subsequent_hy_chg = float(hy_fwd - hy_at_end)

            episodes.append(
                {
                    "start_date": start_date,
                    "end_date": end_date,
                    "duration_days": (end_date - start_date).days + 1
                    if hasattr(end_date - start_date, "days")
                    else int(end_i - start_i + 1),
                    "max_correlation": max_corr,
                    "subsequent_hy_chg_4w": subsequent_hy_chg,
                }
            )
        else:
            i += 1

    if not episodes:
        return pd.DataFrame(
            columns=[
                "start_date",
                "end_date",
                "duration_days",
                "max_correlation",
                "subsequent_hy_chg_4w",
            ]
        )

    return pd.DataFrame(episodes).reset_index(drop=True)
