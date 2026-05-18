"""
Primary credit market issuance — new bond supply as a spread driver.

When issuance is heavy, new issue concessions (NIC) widen secondary spreads.
When issuance is light (or absorbed by strong demand), secondary spreads tighten.
Since real-time deal data is unavailable, we use ETF volume and spread-momentum
proxies to estimate where the primary market stands.

Proxy logic
-----------
1. NIC proxy      : hy_spread - hy_spread.rolling(21).mean()
                    Premium above MA ≈ concession being offered to place new deals.
2. Seasonal factor: Month-of-year calendar; January and September are heavy supply.
3. Volume proxy   : HYG/LQD average daily volume vs their own 63-day averages.
                    Elevated volume = issuance absorption demand → positive for spreads.
                    Fetched live via yfinance with graceful fallback.
4. ETF premium    : HYG price / HYG.rolling(5).mean() — ETF at premium = demand > supply.

Source columns used from master df
-----------------------------------
  hy_spread   ICE BofA HY OAS (percentage, e.g. 5.5 for 550 bps) — required.

Public API
----------
  ISSUANCE_SEASONS          : dict[int, str]
  get_issuance_snapshot()   -> dict
  compute_issuance_proxies(df) -> pd.DataFrame
  run_primary_market_issuance(df) -> dict
"""

from __future__ import annotations

import calendar
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ISSUANCE_SEASONS: dict[int, str] = {
    1: "Heavy (Jan supply surge)",
    2: "Moderate",
    3: "Moderate",
    4: "Moderate",
    5: "Light (pre-summer)",
    6: "Light",
    7: "Very Light (summer lull)",
    8: "Very Light",
    9: "Heavy (post-summer surge)",
    10: "Moderate",
    11: "Light (pre-holiday)",
    12: "Very Light (holiday)",
}

# NIC proxy thresholds (expressed in same units as hy_spread — % not bps)
# hy_spread is stored as %, e.g. 5.50 = 550 bps, so 10bps = 0.10
_NIC_HEAVY_SUPPLY: float = 0.10    # +10 bps concession
_NIC_MILD: float = 0.05            # +5 bps concession
_NIC_TIGHT: float = -0.05          # -5 bps (demand > supply)

_NIC_WINDOW: int = 21              # MA window for NIC proxy
_VOL_SHORT_WINDOW: int = 21        # short-term volume average
_VOL_LONG_WINDOW: int = 63         # longer-term volume average
_HIST_ROWS: int = 504              # rows returned in "historical" key
_FETCH_PERIOD: str = "1y"

_ETF_TICKERS: list[str] = ["HYG", "LQD", "JNK"]

# Seasonal issuance score contribution (0 = very light, 100 = very heavy)
_SEASONAL_SCORE: dict[int, float] = {
    1: 80.0,  # Heavy
    2: 50.0,
    3: 50.0,
    4: 50.0,
    5: 30.0,  # Light
    6: 30.0,
    7: 10.0,  # Very Light
    8: 10.0,
    9: 80.0,  # Heavy
    10: 50.0,
    11: 25.0,
    12: 10.0,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nic_signal(nic_value: float) -> str:
    """Map NIC proxy value to a supply/demand signal string."""
    if np.isnan(nic_value):
        return "Unknown"
    if nic_value > _NIC_HEAVY_SUPPLY:
        return "Concession Required — Heavy Supply"
    if nic_value > _NIC_MILD:
        return "Mild Concession"
    if nic_value > _NIC_TIGHT:
        return "Balanced"
    return "Demand Exceeds Supply (Tight)"


def _issuance_score(nic_value: float, month: int) -> float:
    """
    Combine NIC proxy with seasonal factor into a 0-100 issuance pressure score.

    NIC component: maps linearly from -0.20 → 0 to +0.30 → 100.
    Seasonal component: from SEASONAL_SCORE dict.
    Final = 0.6 * nic_component + 0.4 * seasonal_component, clipped to [0, 100].
    """
    if np.isnan(nic_value):
        seasonal = _SEASONAL_SCORE.get(month, 40.0)
        return float(np.clip(seasonal, 0.0, 100.0))

    # NIC linear map: -0.20 → 0, 0 → 50, +0.30 → 100
    nic_norm = (nic_value + 0.20) / 0.50 * 100.0
    nic_norm = float(np.clip(nic_norm, 0.0, 100.0))
    seasonal = _SEASONAL_SCORE.get(month, 40.0)
    score = 0.6 * nic_norm + 0.4 * seasonal
    return float(np.clip(score, 0.0, 100.0))


def _volume_signal(ratio: float | None) -> str:
    """Classify 21d/63d volume ratio into a demand label."""
    if ratio is None or np.isnan(ratio):
        return "Normal"
    if ratio >= 1.25:
        return "High Demand"
    if ratio <= 0.80:
        return "Low Demand"
    return "Normal"


def _safe_float(val) -> float | None:
    """Return float or None if conversion fails."""
    try:
        v = float(val)
        return None if np.isnan(v) else v
    except Exception:
        return None


# ---------------------------------------------------------------------------
# get_issuance_snapshot
# ---------------------------------------------------------------------------

def get_issuance_snapshot() -> dict:
    """Fetch live ETF volume data via yfinance.

    Returns
    -------
    dict with keys:
        available            : bool
        hyg_volume_21d_avg   : float or None
        hyg_volume_vs_63d    : float or None  — ratio of 21d avg to 63d avg
        lqd_volume_21d_avg   : float or None
        lqd_volume_vs_63d    : float or None
        volume_signal        : str  — "High Demand" / "Normal" / "Low Demand"
        as_of                : str
    """
    try:
        import yfinance as yf

        raw = yf.download(
            _ETF_TICKERS,
            period=_FETCH_PERIOD,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if raw.empty:
            return {"available": False}

        # Extract volume; yfinance returns MultiIndex when >1 ticker
        if isinstance(raw.columns, pd.MultiIndex):
            if "Volume" not in raw.columns.get_level_values(0):
                return {"available": False}
            volume = raw["Volume"]
        else:
            if "Volume" not in raw.columns:
                return {"available": False}
            volume = raw[["Volume"]]

        volume = volume.dropna(how="all")
        if volume.empty:
            return {"available": False}

        def _vol_stats(ticker: str) -> tuple[float | None, float | None]:
            if ticker not in volume.columns:
                return None, None
            v = volume[ticker].dropna()
            if len(v) < _VOL_LONG_WINDOW:
                return None, None
            avg_21 = float(v.tail(_VOL_SHORT_WINDOW).mean())
            avg_63 = float(v.tail(_VOL_LONG_WINDOW).mean())
            ratio = avg_21 / avg_63 if avg_63 > 0 else None
            return avg_21, ratio

        hyg_21, hyg_ratio = _vol_stats("HYG")
        lqd_21, lqd_ratio = _vol_stats("LQD")

        # Use HYG ratio as the primary volume signal
        signal = _volume_signal(hyg_ratio)

        as_of = str(volume.index[-1].date()) if hasattr(volume.index[-1], "date") else str(volume.index[-1])

        return {
            "available": True,
            "hyg_volume_21d_avg": hyg_21,
            "hyg_volume_vs_63d": hyg_ratio,
            "lqd_volume_21d_avg": lqd_21,
            "lqd_volume_vs_63d": lqd_ratio,
            "volume_signal": signal,
            "as_of": as_of,
        }

    except Exception:
        return {"available": False}


# ---------------------------------------------------------------------------
# compute_issuance_proxies
# ---------------------------------------------------------------------------

def compute_issuance_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Compute issuance proxy columns and append them to a copy of *df*.

    Requires 'hy_spread' in df.columns. If absent, returns unchanged copy.

    New columns
    -----------
    nic_proxy        : hy_spread - hy_spread.rolling(21).mean()  (same units as hy_spread)
    issuance_season  : month-based label from ISSUANCE_SEASONS
    issuance_signal  : NIC-based supply/demand signal string
    issuance_score   : 0-100 composite pressure score
    """
    out = df.copy()

    if "hy_spread" not in out.columns or out["hy_spread"].notna().sum() < _NIC_WINDOW:
        out["nic_proxy"] = np.nan
        out["issuance_season"] = out.index.month.map(ISSUANCE_SEASONS) if hasattr(out.index, "month") else np.nan
        out["issuance_signal"] = "Unknown"
        out["issuance_score"] = np.nan
        return out

    spread = out["hy_spread"]
    nic = spread - spread.rolling(_NIC_WINDOW, min_periods=10).mean()
    out["nic_proxy"] = nic

    if hasattr(out.index, "month"):
        out["issuance_season"] = out.index.month.map(ISSUANCE_SEASONS)
        months = out.index.month
    else:
        out["issuance_season"] = "Unknown"
        months = pd.Series([datetime.today().month] * len(out), index=out.index)

    out["issuance_signal"] = out["nic_proxy"].apply(lambda v: _nic_signal(float(v) if pd.notna(v) else float("nan")))

    scores = []
    for nic_val, m in zip(out["nic_proxy"], months):
        n = float(nic_val) if pd.notna(nic_val) else float("nan")
        scores.append(_issuance_score(n, int(m)))
    out["issuance_score"] = scores

    return out


# ---------------------------------------------------------------------------
# run_primary_market_issuance
# ---------------------------------------------------------------------------

def run_primary_market_issuance(df: pd.DataFrame) -> dict:
    """Top-level entry point for primary market issuance analysis.

    Requires 'hy_spread' column and at least 63 rows of data.

    Returns
    -------
    dict with keys:
        available        : bool
        snapshot         : dict from get_issuance_snapshot()
        current          : dict of latest scalar values
        historical       : pd.DataFrame — last 504 rows with issuance columns
        signal_series    : pd.Series — nic_proxy for last 504 rows
        seasonal_calendar: dict — {month_name: season_label} for all 12 months
        spread_vs_nic    : dict — correlation of nic_proxy with fwd 21d spread change
        interpretation   : str
    """
    try:
        if df is None or df.empty:
            return {"available": False}

        if "hy_spread" not in df.columns:
            return {"available": False}

        valid_rows = df["hy_spread"].notna().sum()
        if valid_rows < 63:
            return {"available": False}

        enriched = compute_issuance_proxies(df)

        # ---- snapshot (live fetch) -----------------------------------------
        snapshot = get_issuance_snapshot()

        # ---- current scalars -----------------------------------------------
        last = enriched.iloc[-1]

        def _scalar(col: str):
            if col not in enriched.columns:
                return None
            val = last[col]
            if isinstance(val, (np.floating, np.integer)):
                return float(val)
            if isinstance(val, (bool, np.bool_)):
                return bool(val)
            return val if pd.notna(val) else None

        current_month = datetime.today().month if not hasattr(enriched.index, "month") else int(enriched.index[-1].month)
        current_month_name = calendar.month_name[current_month]

        hy_last = _safe_float(last.get("hy_spread"))
        nic_last = _safe_float(last.get("nic_proxy"))
        hy_21d_avg = None
        if "hy_spread" in enriched.columns:
            ma_val = enriched["hy_spread"].rolling(_NIC_WINDOW, min_periods=10).mean().iloc[-1]
            hy_21d_avg = _safe_float(ma_val)

        current = {
            "nic_proxy": nic_last,
            "issuance_signal": _scalar("issuance_signal"),
            "issuance_score": _scalar("issuance_score"),
            "current_month": current_month_name,
            "issuance_season": ISSUANCE_SEASONS.get(current_month, "Unknown"),
            "hy_spread": hy_last,
            "hy_spread_21d_avg": hy_21d_avg,
        }

        # ---- historical slice ----------------------------------------------
        hist_cols = ["nic_proxy", "issuance_season", "issuance_signal", "issuance_score", "hy_spread"]
        hist_cols_avail = [c for c in hist_cols if c in enriched.columns]
        historical = enriched[hist_cols_avail].tail(_HIST_ROWS).copy()

        # ---- signal series -------------------------------------------------
        if "nic_proxy" in enriched.columns:
            signal_series = enriched["nic_proxy"].tail(_HIST_ROWS).copy()
        else:
            signal_series = pd.Series(dtype=float)

        # ---- seasonal calendar (all 12 months) -----------------------------
        seasonal_calendar = {
            calendar.month_name[m]: label for m, label in ISSUANCE_SEASONS.items()
        }

        # ---- spread vs NIC correlation (fwd 21d spread change) -------------
        spread_vs_nic: dict = {"available": False}
        if "nic_proxy" in enriched.columns and "hy_spread" in enriched.columns:
            nic_clean = enriched["nic_proxy"].dropna()
            if len(nic_clean) >= 63:
                fwd_spread_chg = enriched["hy_spread"].shift(-21) - enriched["hy_spread"]
                aligned = pd.concat(
                    {"nic": enriched["nic_proxy"], "fwd_chg": fwd_spread_chg},
                    axis=1,
                ).dropna()
                if len(aligned) >= 40:
                    corr = float(np.corrcoef(aligned["nic"].values, aligned["fwd_chg"].values)[0, 1])
                    spread_vs_nic = {
                        "available": True,
                        "correlation": round(corr, 4),
                        "n_observations": len(aligned),
                        "interpretation": (
                            "Positive correlation: higher NIC proxy (concession) tends to "
                            "precede spread widening over the next 21 trading days."
                            if corr > 0.05 else
                            "Negative correlation: NIC proxy inversely related to "
                            "forward spread change — demand absorbs supply pressure."
                            if corr < -0.05 else
                            "Weak or no linear relationship between NIC proxy and "
                            "21-day forward spread changes."
                        ),
                    }

        # ---- interpretation ------------------------------------------------
        nic_bps = round(nic_last * 100, 1) if nic_last is not None else None
        signal_str = current.get("issuance_signal", "Unknown")
        season_str = current.get("issuance_season", "Unknown")
        score_val = current.get("issuance_score")

        if nic_bps is not None:
            interpretation = (
                f"The NIC proxy stands at {nic_bps:+.1f} bps ({signal_str}). "
                f"Seasonal context for {current_month_name} is '{season_str}'. "
                f"Composite issuance pressure score: {score_val:.0f}/100. "
            )
        else:
            interpretation = (
                f"NIC proxy unavailable. Seasonal context for {current_month_name} "
                f"is '{season_str}'."
            )

        vol_sig = snapshot.get("volume_signal", "N/A") if snapshot.get("available") else "N/A"
        hyg_ratio = snapshot.get("hyg_volume_vs_63d")
        if vol_sig != "N/A":
            interpretation += (
                f"Live ETF volume signal is '{vol_sig}'"
                + (f" (HYG 21d/63d volume ratio: {hyg_ratio:.2f}x)" if hyg_ratio else "")
                + "."
            )

        return {
            "available": True,
            "snapshot": snapshot,
            "current": current,
            "historical": historical,
            "signal_series": signal_series,
            "seasonal_calendar": seasonal_calendar,
            "spread_vs_nic": spread_vs_nic,
            "interpretation": interpretation,
        }

    except Exception:
        return {"available": False}
