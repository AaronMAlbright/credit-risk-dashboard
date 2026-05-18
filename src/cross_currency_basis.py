"""
Cross-Currency Basis Swap Monitor.

EUR/USD and JPY/USD cross-currency basis = the premium/discount to borrow USD
via FX swap versus borrowing USD directly in the interbank market.

  Negative EUR/USD basis → EUR investors pay extra to swap EUR into USD
                          → USD funding is scarce for non-US investors
                          → foreign demand for USD credit falls → spreads widen

  Basis near 0 or positive → foreign demand for USD credit is intact

Because live cross-currency basis data (e.g., Bloomberg EURUSD 3m basis) is
not freely available, this module builds a proxy from publicly available data:

  Priority stack (EUR basis proxy):
    1. "eurusd_basis", "xccy_basis", or "eur_usd_basis" column in df (direct)
    2. DXY z-score scaling: EUR_basis_proxy ≈ −DXY_zscore_1y × 15  (bps)
    3. FXE/UUP live fetch via yfinance (returns-based proxy)

  Priority stack (JPY basis proxy):
    1. "jpyusd_basis" column in df (direct)
    2. JPYUSD=X (or inverse of JPY=X) combined with DXY z-score

Basis signal labels (bps equivalent):
  < −30 bps  →  "Negative Basis (Headwind)"
  −30 to −5  →  "Moderately Negative"
   −5 to +5  →  "Neutral"
       > +5  →  "Positive Basis (Tailwind)"

Public API
----------
  get_basis_snapshot()             -> dict
  compute_historical_basis(df)     -> pd.DataFrame
  run_cross_currency_basis(df)     -> dict
"""
from __future__ import annotations

import datetime
import functools

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HIST_ROWS = 504          # ~2 trading years of history to surface
_ZSCORE_WINDOW = 252      # 1 trading year
_MIN_PERIODS = 63         # minimum ~3 months

_LIVE_PERIOD = "2y"

# DXY scaling factor: 1-std DXY move ≈ 15 bps of EUR/USD basis
_DXY_SCALE_BPS = 15.0

# Basis signal thresholds (bps)
_NEG_HEADWIND = -30.0
_NEG_MODERATE = -5.0
_POS_TAILWIND =  5.0

# Lead time in weeks that cross-currency basis typically precedes spread moves
_LEAD_WEEKS = 3

# Column candidates in master DataFrame
_EUR_BASIS_CANDIDATES = ("eurusd_basis", "xccy_basis", "eur_usd_basis")
_JPY_BASIS_CANDIDATES = ("jpyusd_basis", "jpy_usd_basis", "jpy_basis")
_DXY_CANDIDATES = ("dxy", "dx_y_nyb", "dollar_index", "dtwexbgs")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _rolling_zscore(s: pd.Series, window: int = _ZSCORE_WINDOW) -> pd.Series:
    rm = s.rolling(window=window, min_periods=_MIN_PERIODS).mean()
    rs = s.rolling(window=window, min_periods=_MIN_PERIODS).std(ddof=1)
    return ((s - rm) / rs.replace(0, np.nan)).clip(-3.0, 3.0)


def _eur_basis_signal(basis_bps: float) -> str:
    if np.isnan(basis_bps):
        return "Unknown"
    if basis_bps < _NEG_HEADWIND:
        return "Negative Basis (Headwind)"
    if basis_bps < _NEG_MODERATE:
        return "Moderately Negative"
    if basis_bps <= _POS_TAILWIND:
        return "Neutral"
    return "Positive Basis (Tailwind)"


def _jpy_basis_signal(basis_bps: float) -> str:
    """JPY basis labeling uses same thresholds as EUR."""
    return _eur_basis_signal(basis_bps)


def _usd_credit_demand_signal(
    eur_basis_bps: float, jpy_basis_bps: float
) -> str:
    """
    Composite USD credit demand signal from EUR and JPY basis proxies.

    Returns "Falling" / "Stable" / "Rising".
    """
    signals = []
    for b in (eur_basis_bps, jpy_basis_bps):
        if np.isnan(b):
            continue
        if b < _NEG_HEADWIND:
            signals.append(-2)
        elif b < _NEG_MODERATE:
            signals.append(-1)
        elif b <= _POS_TAILWIND:
            signals.append(0)
        else:
            signals.append(1)

    if not signals:
        return "Unknown"
    score = float(np.mean(signals))
    if score <= -1.0:
        return "Falling"
    if score >= 0.75:
        return "Rising"
    return "Stable"


# ---------------------------------------------------------------------------
# Live yfinance fetch (cached per calendar day)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _cached_fx_history(date_key: str) -> dict:  # noqa: ARG001
    """Download 2-year daily closes for FX and ETF tickers. Cached per day."""
    try:
        import yfinance as yf
        tickers = ["EURUSD=X", "JPYUSD=X", "FXE", "FXY", "UUP", "SPY"]
        raw = yf.download(
            tickers, period=_LIVE_PERIOD, auto_adjust=True, progress=False
        )
        if raw.empty:
            return {"available": False}

        # Handle both MultiIndex and flat column structures
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"].copy()
        else:
            closes = raw.copy()

        closes = closes.dropna(how="all")
        if closes.empty:
            return {"available": False}

        return {"available": True, "closes": closes}
    except Exception:
        return {"available": False}


def _get_fx_closes() -> dict:
    date_key = datetime.date.today().isoformat()
    return _cached_fx_history(date_key)


# ---------------------------------------------------------------------------
# Public: get_basis_snapshot
# ---------------------------------------------------------------------------

def get_basis_snapshot() -> dict:
    """
    Fetch live FX and basis proxy data via yfinance.

    Returns
    -------
    dict with keys:
        available               : bool
        eurusd_spot             : float or None
        jpyusd_spot             : float or None
        dxy_level               : float or None  (UUP used as DXY proxy)
        eur_basis_proxy_bps     : float or None  (estimated EUR/USD basis in bps)
        jpy_basis_proxy_bps     : float or None
        eur_basis_signal        : str
        jpy_basis_signal        : str
        usd_credit_demand_signal: str
        as_of                   : str
        data_source             : str  ("Direct" or "Proxy (DXY-based)")
    """
    _now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    _unavail: dict = {
        "available": False,
        "eurusd_spot": None,
        "jpyusd_spot": None,
        "dxy_level": None,
        "eur_basis_proxy_bps": None,
        "jpy_basis_proxy_bps": None,
        "eur_basis_signal": "Unknown",
        "jpy_basis_signal": "Unknown",
        "usd_credit_demand_signal": "Unknown",
        "as_of": _now,
        "data_source": "Unavailable",
    }

    try:
        data = _get_fx_closes()
        if not data.get("available"):
            return _unavail

        closes: pd.DataFrame = data["closes"]
        if closes.empty or len(closes) < _MIN_PERIODS:
            return _unavail

        # ---- Spot rates -------------------------------------------------------
        eurusd_spot: float | None = None
        if "EURUSD=X" in closes.columns:
            v = closes["EURUSD=X"].dropna()
            if not v.empty:
                eurusd_spot = float(v.iloc[-1])

        jpyusd_spot: float | None = None
        if "JPYUSD=X" in closes.columns:
            v = closes["JPYUSD=X"].dropna()
            if not v.empty:
                jpyusd_spot = float(v.iloc[-1])

        # ---- DXY proxy via UUP ------------------------------------------------
        uup_level: float | None = None
        uup_zscore: float | None = None
        if "UUP" in closes.columns:
            uup = closes["UUP"].dropna()
            if len(uup) >= _MIN_PERIODS:
                uup_level = float(uup.iloc[-1])
                uup_z = _rolling_zscore(uup, window=_ZSCORE_WINDOW)
                uup_zscore = float(uup_z.iloc[-1]) if not np.isnan(uup_z.iloc[-1]) else None

        # ---- EUR basis proxy --------------------------------------------------
        # Method 1: FXE return vs interest-rate-parity-implied return (63-day)
        # Method 2: −UUP_zscore × 15 bps (simpler, always available if UUP has data)
        eur_basis_bps: float | None = None
        data_source = "Proxy (DXY-based)"

        if "FXE" in closes.columns and "UUP" in closes.columns:
            fxe = closes["FXE"].dropna()
            uup_s = closes["UUP"].dropna()
            # Align on common index
            common_idx = fxe.index.intersection(uup_s.index)
            if len(common_idx) >= 63:
                fxe_a = fxe.reindex(common_idx)
                uup_a = uup_s.reindex(common_idx)
                # FXE and UUP move inversely — basis proxy = FXE return + UUP return
                # When USD strengthens (UUP up), FXE falls, basis turns negative
                fxe_63d = float(fxe_a.iloc[-1] / fxe_a.iloc[-64] - 1) if len(fxe_a) >= 64 else float("nan")
                uup_63d = float(uup_a.iloc[-1] / uup_a.iloc[-64] - 1) if len(uup_a) >= 64 else float("nan")
                if not (np.isnan(fxe_63d) or np.isnan(uup_63d)):
                    # Net of carry: positive UUP (USD strong) → negative EUR demand for USD
                    # Scale: 1% UUP return ≈ 5 bps of basis change
                    eur_basis_bps = float(-(uup_63d * 500))
                    data_source = "Proxy (FXE/UUP returns)"

        if eur_basis_bps is None and uup_zscore is not None:
            eur_basis_bps = float(-uup_zscore * _DXY_SCALE_BPS)
            data_source = "Proxy (DXY-based)"

        # ---- JPY basis proxy --------------------------------------------------
        # JPY basis is correlated with DXY but JPY carry dynamics differ:
        # A very weak JPY (high USDJPY) → JPY investors piling into USD carry
        # but also signals potential disorderly unwind → basis becomes more negative
        jpy_basis_bps: float | None = None

        if "FXY" in closes.columns and uup_zscore is not None:
            fxy = closes["FXY"].dropna()
            if len(fxy) >= 64:
                fxy_63d = float(fxy.iloc[-1] / fxy.iloc[-64] - 1)
                if not np.isnan(fxy_63d):
                    # Similar logic: FXY (JPY ETF in USD) performance vs DXY direction
                    jpy_basis_bps = float(-(uup_zscore * _DXY_SCALE_BPS * 0.7)
                                          - (fxy_63d * 300))
        elif uup_zscore is not None:
            jpy_basis_bps = float(-uup_zscore * _DXY_SCALE_BPS * 0.7)

        # ---- Signals ----------------------------------------------------------
        eur_bps_val = eur_basis_bps if eur_basis_bps is not None else float("nan")
        jpy_bps_val = jpy_basis_bps if jpy_basis_bps is not None else float("nan")

        eur_sig = _eur_basis_signal(eur_bps_val)
        jpy_sig = _jpy_basis_signal(jpy_bps_val)
        demand_sig = _usd_credit_demand_signal(eur_bps_val, jpy_bps_val)

        def _r(v: float | None, n: int = 2) -> float | None:
            if v is None or np.isnan(v):
                return None
            return round(v, n)

        return {
            "available": True,
            "eurusd_spot": _r(eurusd_spot, 5),
            "jpyusd_spot": _r(jpyusd_spot, 6),
            "dxy_level": _r(uup_level, 4),
            "eur_basis_proxy_bps": _r(eur_basis_bps, 1),
            "jpy_basis_proxy_bps": _r(jpy_basis_bps, 1),
            "eur_basis_signal": eur_sig,
            "jpy_basis_signal": jpy_sig,
            "usd_credit_demand_signal": demand_sig,
            "as_of": _now,
            "data_source": data_source,
        }
    except Exception:
        return _unavail


# ---------------------------------------------------------------------------
# Public: compute_historical_basis
# ---------------------------------------------------------------------------

def compute_historical_basis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute historical EUR/USD and JPY/USD basis proxies from master DataFrame.

    Column resolution (checked in order):
      EUR basis: "eurusd_basis", "xccy_basis", "eur_usd_basis"
      JPY basis: "jpyusd_basis", "jpy_usd_basis", "jpy_basis"
      DXY proxy: "dxy", "dx_y_nyb", "dollar_index", "dtwexbgs"

    Appended columns:
      basis_eur_proxy  — EUR/USD basis in bps equivalent
      basis_jpy_proxy  — JPY/USD basis in bps equivalent (if DXY available)
      basis_signal     — categorical label (based on EUR proxy)
      basis_usd_demand — "Falling" / "Stable" / "Rising"

    Returns a copy of df (never raises).
    """
    try:
        out = df.copy()

        # ---- EUR basis --------------------------------------------------------
        eur_col = _resolve_column(out, _EUR_BASIS_CANDIDATES)
        if eur_col is not None:
            out["basis_eur_proxy"] = out[eur_col].copy()
        else:
            # Build from DXY z-score
            dxy_col = _resolve_column(out, _DXY_CANDIDATES)
            if dxy_col is not None:
                dxy_z = _rolling_zscore(out[dxy_col].copy())
                out["basis_eur_proxy"] = -(dxy_z * _DXY_SCALE_BPS)
            else:
                out["basis_eur_proxy"] = pd.Series(float("nan"), index=out.index)

        # ---- JPY basis --------------------------------------------------------
        jpy_col = _resolve_column(out, _JPY_BASIS_CANDIDATES)
        if jpy_col is not None:
            out["basis_jpy_proxy"] = out[jpy_col].copy()
        else:
            dxy_col = _resolve_column(out, _DXY_CANDIDATES)
            if dxy_col is not None:
                dxy_z = _rolling_zscore(out[dxy_col].copy())
                out["basis_jpy_proxy"] = -(dxy_z * _DXY_SCALE_BPS * 0.7)
            else:
                out["basis_jpy_proxy"] = pd.Series(float("nan"), index=out.index)

        # ---- Signals ----------------------------------------------------------
        def _row_eur_signal(v: float) -> str:
            return _eur_basis_signal(v)

        def _row_demand_signal(row: pd.Series) -> str:
            eur = row["basis_eur_proxy"]
            jpy = row["basis_jpy_proxy"]
            return _usd_credit_demand_signal(
                eur if not np.isnan(eur) else float("nan"),
                jpy if not np.isnan(jpy) else float("nan"),
            )

        out["basis_signal"] = out["basis_eur_proxy"].apply(
            lambda v: _eur_basis_signal(float(v)) if not np.isnan(v) else "Unknown"
        )
        out["basis_usd_demand"] = out[["basis_eur_proxy", "basis_jpy_proxy"]].apply(
            _row_demand_signal, axis=1
        )

        return out
    except Exception:
        return df.copy()


# ---------------------------------------------------------------------------
# Public: run_cross_currency_basis
# ---------------------------------------------------------------------------

def run_cross_currency_basis(df: pd.DataFrame) -> dict:
    """
    Top-level entry point for the cross-currency basis module.

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame with DatetimeIndex.

    Returns
    -------
    dict with keys:
        available           : bool
        snapshot            : dict  (from get_basis_snapshot)
        historical          : pd.DataFrame  (last 504 rows with basis columns)
        signal_series       : pd.Series    (basis_eur_proxy, last 504 rows)
        current_signal      : str
        credit_implication  : str
        interpretation      : str
        lead_weeks          : int  (typically 2–4)
    """
    _unavail: dict = {
        "available": False,
        "snapshot": {"available": False},
        "historical": pd.DataFrame(),
        "signal_series": pd.Series(dtype=float),
        "current_signal": "Unknown",
        "credit_implication": "Insufficient data to assess cross-currency basis impact.",
        "interpretation": "Cross-currency basis data unavailable.",
        "lead_weeks": _LEAD_WEEKS,
    }

    try:
        # ---- Live snapshot ----------------------------------------------------
        snapshot = get_basis_snapshot()

        # ---- Historical computation ------------------------------------------
        historical_df = compute_historical_basis(df)
        tail = historical_df.tail(_HIST_ROWS)

        # Check whether we have meaningful data
        has_eur_proxy = (
            "basis_eur_proxy" in tail.columns
            and tail["basis_eur_proxy"].notna().any()
        )
        dxy_col = _resolve_column(df, _DXY_CANDIDATES)
        has_dxy = dxy_col is not None
        eur_basis_col = _resolve_column(df, _EUR_BASIS_CANDIDATES)

        snap_available = snapshot.get("available", False)
        data_available = snap_available or has_dxy or eur_basis_col is not None

        if not data_available:
            return _unavail

        # ---- Current signal --------------------------------------------------
        signal_series = (
            tail["basis_eur_proxy"]
            if has_eur_proxy
            else pd.Series(dtype=float)
        )

        # Determine current signal: prefer live snapshot, fallback to historical
        if snap_available and snapshot.get("eur_basis_signal"):
            current_signal = snapshot["eur_basis_signal"]
            current_bps = snapshot.get("eur_basis_proxy_bps")
        elif has_eur_proxy and signal_series.notna().any():
            last_val = float(signal_series.dropna().iloc[-1])
            current_signal = _eur_basis_signal(last_val)
            current_bps = round(last_val, 1)
        else:
            current_signal = "Unknown"
            current_bps = None

        demand_signal = (
            snapshot.get("usd_credit_demand_signal", "Unknown")
            if snap_available
            else "Unknown"
        )

        # ---- Credit implication ----------------------------------------------
        implication_map = {
            "Negative Basis (Headwind)": (
                "Foreign USD credit demand is being suppressed by expensive FX hedging "
                "costs. EUR- and JPY-based investors are paying a significant premium to "
                "swap into USD, making USD HY and IG credit less attractive on a hedged "
                "basis. Historically this condition has preceded spread widening of 30-80 "
                "bps in HY over the following 2-4 weeks."
            ),
            "Moderately Negative": (
                "FX hedging costs are slightly elevated, creating a modest headwind for "
                "foreign USD credit demand. Spreads may face upward pressure, but the "
                "effect is not yet dislocating. Monitor for widening to below -30 bps."
            ),
            "Neutral": (
                "Cross-currency basis is near zero; FX hedging costs are not a meaningful "
                "barrier to foreign participation in USD credit markets. Foreign demand is "
                "supportive of spreads at current levels."
            ),
            "Positive Basis (Tailwind)": (
                "Foreign investors are receiving a net premium to access USD credit on a "
                "hedged basis. This is a structural tailwind for USD credit demand and is "
                "spread-compressive. May indicate excess global USD liquidity."
            ),
            "Unknown": (
                "Cross-currency basis signal is not determinable from available data."
            ),
        }
        credit_implication = implication_map.get(current_signal, implication_map["Unknown"])

        # ---- Interpretation --------------------------------------------------
        bps_str = f"{current_bps:+.0f} bps" if current_bps is not None else "N/A"
        src = snapshot.get("data_source", "Proxy") if snap_available else "Historical proxy"
        interpretation = (
            f"EUR/USD basis proxy: {bps_str} ({current_signal}). "
            f"USD credit demand: {demand_signal}. "
            f"Source: {src}. "
            f"Basis typically leads spread moves by {_LEAD_WEEKS} weeks."
        )

        return {
            "available": True,
            "snapshot": snapshot,
            "historical": tail,
            "signal_series": signal_series,
            "current_signal": current_signal,
            "credit_implication": credit_implication,
            "interpretation": interpretation,
            "lead_weeks": _LEAD_WEEKS,
        }

    except Exception:
        return _unavail
