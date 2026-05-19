"""
Unified market data loader with local parquet caching.

Design principles
-----------------
- SP500 sourced from yfinance (^GSPC) for 1999+ history; the FRED SP500
  series only starts 2016-05-09.
- All series reindexed to business-day frequency; weekly/monthly FRED data
  is forward-filled with a per-series limit (prevents stale data propagating
  over genuine gaps like survey release cycles).
- Parquet cache stored under data/cache/. TTL: 12 hours. Each series is
  cached independently so one stale series doesn't force a full re-fetch.
- Optional series fail gracefully: a warning is printed and the column is
  left out of the returned DataFrame rather than aborting the pipeline.
- T10YIE (10Y breakeven) starts 2003-01-02; pre-2003 observations are filled
  with 2.5% (long-run inflation consensus) and flagged via breakeven_imputed.

Public API
----------
  fetch_fred(series_id, start, cache_dir)  →  pd.Series
  fetch_sp500(start, cache_dir)            →  pd.Series
  load_all_series(start, cache_dir)        →  pd.DataFrame  (business-day index)
  get_freshness(start, cache_dir)          →  dict[str, dict]
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_DIR    = Path("data/cache")
_CACHE_TTL_S  = 12 * 3600          # 12 hours before a re-fetch is triggered
_DEFAULT_START = "1999-01-01"

# Forward-fill limits (calendar days) per series.
# Prevents a monthly release from propagating stale values over weeks.
_FFILL_LIMITS: dict[str, int] = {
    # Daily series — allow up to 1 full week (covers long-weekend closures)
    "DGS10": 5, "DGS2": 5, "DGS3MO": 5, "T10Y3M": 5, "T10Y2Y": 5,
    "DFF": 5, "VIXCLS": 5, "BAA10Y": 5, "T10YIE": 10,
    "DCOILWTICO": 5, "DEXUSEU": 5, "DEXJPUS": 5, "TEDRATE": 5,
    "SP500": 5, "BAMLH0A0HYM2": 5, "MOVE": 5,
    # Weekly series — allow up to 7 calendar days
    "NFCI": 7, "ANFCI": 7, "STLFSI4": 7,
    "ICSA": 7, "CCSA": 7,
    "DPSACBW027SBOG": 7, "TOTLL": 7, "WALCL": 7, "WRESBAL": 7,
    "BOGMBASEW": 7,
    # Monthly series — allow up to 35 calendar days (~1 month + buffer)
    "UNRATE": 35, "CPIAUCSL": 35, "CPILFESL": 35, "PCEPILFE": 35,
    "USREC": 35,
    # Credit spread complex — daily ICE BofA series
    "BAMLC0A0CM":      5,   # IG OAS
    "BAMLC0A4CBBB":    5,   # BBB OAS
    "BAMLH0A0HYM2EY": 5,  # HY effective yield (all-in carry)
    "BAMLC0A0CMEY":    5,   # IG effective yield
    # Quarterly SLOOS survey — ffill up to one quarter + buffer
    "DRTSCILM": 95,
}

# Calibration ratio: median(BAMLH0A0HYM2 / BAA10Y) over 748-row overlap (2023-05-08 onward).
# Pre-ICE period uses BAA10Y * this ratio as a scaled HY proxy.
_ICE_BAA_RATIO = 1.9513

# Core daily series whose absence disqualifies a row entirely
_REQUIRED_COLS = ["yield_10y", "yield_2y", "vix", "hy_spread", "sp500"]

# Mapping from FRED series ID → clean column name used in the DataFrame
# Note: hy_spread is set by fetch_hy_spliced (ICE OAS 2023+ / scaled BAA10Y pre-2023).
# BAA10Y mapped to hy_spread_baa for diagnostics if splice succeeded.
_COL_NAMES: dict[str, str] = {
    "DGS10":          "yield_10y",
    "DGS2":           "yield_2y",
    "DGS3MO":         "yield_3m",
    "T10Y3M":         "spread_10y3m",     # precomputed 10Y-3M; better recession signal
    "DFF":            "fed_funds_rate",
    "VIXCLS":         "vix",
    "BAA10Y":         "hy_spread",        # fallback if fetch_hy_spliced fails
    "UNRATE":         "unemployment",
    "NFCI":           "nfci",
    "ANFCI":          "anfci",            # adjusted NFCI (removes macro trends)
    "STLFSI4":        "stlfsi",           # St. Louis Financial Stress Index
    "T10YIE":         "breakeven_10y",
    "DCOILWTICO":     "oil_wti",
    "DEXUSEU":        "eurusd",
    "DEXJPUS":        "usdjpy",
    "ICSA":           "initial_claims",   # weekly, thousands
    "DPSACBW027SBOG": "bank_deposits",    # weekly
    "TOTLL":          "total_loans",      # weekly
    "WALCL":          "fed_balance_sheet",# weekly, millions
    "CPIAUCSL":       "cpi",              # monthly
    "USREC":          "nber_recession",   # monthly, 0/1
    # Credit spread complex
    "BAMLC0A0CM":      "ig_spread",        # ICE BofA IG OAS (daily)
    "BAMLC0A4CBBB":    "bbb_spread",       # ICE BofA BBB OAS (daily)
    "BAMLH0A0HYM2EY": "hy_yield",        # HY all-in effective yield
    "BAMLC0A0CMEY":    "ig_yield",         # IG all-in effective yield
    "DRTSCILM":        "sloos_ci",         # SLOOS — % banks tightening C&I standards
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cache_path(series_id: str, start_year: str) -> Path:
    safe = series_id.replace("/", "_").replace("^", "")
    return _CACHE_DIR / f"{safe}_{start_year}.parquet"


def _cache_valid(path: Path) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < _CACHE_TTL_S


def _to_daily(s: pd.Series, ffill_limit: int, bday_index: pd.DatetimeIndex) -> pd.Series:
    """
    Reindex a Series to a business-day DatetimeIndex and forward-fill gaps.
    Duplicates are resolved by keeping the last value for each date.

    Monthly FRED series use observation dates like "2026-01-01" which fall on
    weekends or public holidays and would be silently dropped when reindexing
    to the business-day spine. This function snaps each observation date to the
    next available business day (look-ahead of up to 7 calendar days) before
    reindexing, so no monthly observation is lost.
    """
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")]

    # Snap any non-business-day observation dates to the next business day.
    # We build a frozenset of business-day dates for O(1) membership testing.
    bday_dates = frozenset(bday_index.normalize().date)
    snapped: list[pd.Timestamp] = []
    for d in s.index:
        for delta in range(8):
            candidate = (d + pd.Timedelta(days=delta)).normalize()
            if candidate.date() in bday_dates:
                snapped.append(candidate)
                break
        else:
            snapped.append(d)  # fallback: keep original date (rare edge case)

    s.index = pd.DatetimeIndex(snapped)
    s = s[~s.index.duplicated(keep="last")]  # deduplicate again after snapping
    s = s.reindex(bday_index)
    s = s.ffill(limit=ffill_limit)
    return s


# ---------------------------------------------------------------------------
# Public fetch functions
# ---------------------------------------------------------------------------

def fetch_fred(
    series_id: str,
    start: str = _DEFAULT_START,
    cache_dir: Path = _CACHE_DIR,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> pd.Series:
    """
    Fetch a single FRED series with parquet caching.

    Returns a Series with DatetimeIndex (raw observation frequency — not yet
    aligned to business days). Missing observations are NaN.
    Cache is invalidated after _CACHE_TTL_S seconds.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(series_id, start[:4])

    if _cache_valid(path):
        cached = pd.read_parquet(path).squeeze()
        cached.name = series_id
        return cached

    from fredapi import Fred
    fred = Fred(api_key=os.environ.get("FRED_API_KEY", ""))

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = fred.get_series(series_id, observation_start=start)
            s = raw.dropna()
            s.name = series_id
            s.to_frame().to_parquet(path)
            return s
        except Exception as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(retry_delay)

    raise RuntimeError(
        f"FRED fetch failed for {series_id} after {retries} attempts: {last_err}"
    )


def fetch_move(
    start: str = _DEFAULT_START,
    cache_dir: Path = _CACHE_DIR,
) -> pd.Series:
    """
    Fetch the ICE BofA MOVE index (^MOVE) from yfinance with parquet caching.

    MOVE measures implied volatility on US Treasuries (1-month options across
    the 2Y/5Y/10Y/30Y curve, weighted by open interest). It is the bond-market
    analogue of VIX: elevated readings (>100) signal funding stress and forced
    deleveraging risk. Available from 2002-11-12; pre-2003 rows are NaN.

    Returns a Series with DatetimeIndex (market calendar dates).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path("MOVE_yf", start[:4])

    if _cache_valid(path):
        cached = pd.read_parquet(path).squeeze()
        cached.name = "MOVE"
        if not cached.dropna().empty:
            return cached

    try:
        import yfinance as yf
        raw = yf.download("^MOVE", start=start, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            s = raw["Close"].iloc[:, 0]
        else:
            s = raw["Close"]
        s.index = pd.DatetimeIndex(s.index).normalize()
        s = s.dropna()
        if s.empty:
            raise RuntimeError("yfinance returned empty MOVE series")
        s.name = "MOVE"
        s.to_frame().to_parquet(path)
        return s
    except Exception as exc:
        raise RuntimeError(f"[market_data] MOVE fetch failed: {exc}") from exc


def fetch_hy_spliced(
    start: str = _DEFAULT_START,
    cache_dir: Path = _CACHE_DIR,
) -> tuple[pd.Series, pd.Series]:
    """
    Build a spliced HY spread series combining ICE BofA OAS (BAMLH0A0HYM2,
    2023-05-08+) with a scaled BAA10Y proxy for the pre-2023 period.

    ICE BofA HY OAS is the preferred series: it measures the option-adjusted
    spread of the full US HY universe and is the industry benchmark. BAA10Y
    (Moody's Baa minus 10Y Treasury) is investment-grade and structurally
    lower, so we calibrate it using the median ratio over the overlap period.

    Returns
    -------
    hy_spliced  : pd.Series — spliced HY spread (use as primary hy_spread)
    hy_baa_raw  : pd.Series — raw BAA10Y (kept for diagnostics / signal continuity)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    splice_path = _cache_path("HY_spliced", start[:4])

    baa = fetch_fred("BAA10Y", start=start, cache_dir=cache_dir)

    # Try ICE HY OAS — available from 2023-05-08
    try:
        ice = fetch_fred("BAMLH0A0HYM2", start="2020-01-01", cache_dir=cache_dir)
    except Exception as exc:
        print(f"[market_data] WARNING: BAMLH0A0HYM2 unavailable ({exc}); using BAA10Y proxy")
        return baa.rename("hy_spread"), baa.rename("hy_spread_baa")

    # Compute calibration ratio from overlap (ICE / BAA10Y)
    baa_idx  = set(baa.index.normalize().date)
    ice_idx  = set(ice.index.normalize().date)
    overlap  = sorted(baa_idx & ice_idx)
    if len(overlap) >= 30:
        baa_ov = baa[baa.index.normalize().map(lambda d: d.date() in set(overlap))]
        ice_ov = ice[ice.index.normalize().map(lambda d: d.date() in set(overlap))]
        baa_ov = baa_ov[~baa_ov.index.duplicated(keep="last")]
        ice_ov = ice_ov[~ice_ov.index.duplicated(keep="last")]
        common = baa_ov.index.intersection(ice_ov.index)
        if len(common) >= 30:
            ratio = float((ice_ov.reindex(common) / baa_ov.reindex(common)).median())
        else:
            ratio = _ICE_BAA_RATIO
    else:
        ratio = _ICE_BAA_RATIO

    # Splice: BAA10Y * ratio for pre-ICE period; ICE OAS directly afterward
    spliced = (baa * ratio).rename("hy_spread")
    ice_clean = ice.dropna()
    if len(ice_clean) > 0:
        splice_start = ice_clean.index.normalize().min()
        mask = spliced.index.normalize() >= splice_start
        # Reindex ICE to BAA's index for the post-splice period
        ice_reindexed = ice_clean.reindex(spliced.index[mask], method="pad", limit=5)
        spliced[mask] = ice_reindexed.values

    spliced.to_frame().to_parquet(splice_path)
    return spliced, baa.rename("hy_spread_baa")


def fetch_sp500(
    start: str = _DEFAULT_START,
    cache_dir: Path = _CACHE_DIR,
) -> pd.Series:
    """
    Fetch S&P 500 daily closes from yfinance (^GSPC).

    yfinance provides history back to 1999 while FRED's SP500 series only
    starts 2016-05-09. Falls back to the FRED series if yfinance fails.

    Returns a Series with DatetimeIndex (business-day dates already implied
    by market trading calendar).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path("SP500_yf", start[:4])

    if _cache_valid(path):
        cached = pd.read_parquet(path).squeeze()
        cached.name = "SP500"
        if not cached.dropna().empty:
            return cached

    try:
        import yfinance as yf
        raw = yf.download("^GSPC", start=start, auto_adjust=True, progress=False)
        # Handle MultiIndex columns from newer yfinance versions
        if isinstance(raw.columns, pd.MultiIndex):
            s = raw["Close"].iloc[:, 0]
        else:
            s = raw["Close"]
        s.index = pd.DatetimeIndex(s.index).normalize()
        s = s.dropna()
        if s.empty:
            raise RuntimeError("yfinance returned empty SP500 series")
        s.name = "SP500"
        s.to_frame().to_parquet(path)
        return s
    except Exception as exc:
        print(f"[market_data] yfinance SP500 failed ({exc}); falling back to FRED SP500")
        return fetch_fred("SP500", start=start, cache_dir=cache_dir)


# ---------------------------------------------------------------------------
# Main assembly function
# ---------------------------------------------------------------------------

def load_all_series(
    start: str = _DEFAULT_START,
    cache_dir: Path = _CACHE_DIR,
) -> pd.DataFrame:
    """
    Load all core and cross-asset market series, align to a business-day index,
    and return a single DataFrame ready for the feature pipeline.

    Column names follow the clean naming in _COL_NAMES; the index is a
    DatetimeIndex of business days.

    Missing optional series are silently omitted (printed warning only).
    Rows where ALL required daily series are missing are dropped.

    New columns vs. legacy app.py:
      yield_10y  (was value_10y from join lsuffix)
      yield_2y   (was value_2y from join rsuffix)
      spread_10y3m     — 10Y minus 3M yield spread (better recession predictor)
      fed_funds_rate   — effective fed funds rate
      anfci, stlfsi    — enhanced liquidity stress indices
      oil_wti, eurusd, usdjpy — cross-asset risk-off signals
      initial_claims, bank_deposits, total_loans, fed_balance_sheet
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Establish business-day date spine
    bdays = pd.bdate_range(start=start, end=pd.Timestamp.now().normalize())

    frames: dict[str, pd.Series] = {}

    # SP500 via yfinance
    try:
        sp = fetch_sp500(start=start, cache_dir=cache_dir)
        frames["sp500"] = _to_daily(sp, ffill_limit=5, bday_index=bdays)
    except Exception as exc:
        print(f"[market_data] ERROR: SP500 load failed — {exc}")

    # MOVE index via yfinance (bond vol; available 2002-11-12+)
    try:
        mv = fetch_move(start=start, cache_dir=cache_dir)
        frames["move_index"] = _to_daily(mv, ffill_limit=5, bday_index=bdays)
    except Exception as exc:
        print(f"[market_data] WARNING: MOVE index load failed — {exc}")

    # Spliced HY spread: ICE BofA OAS (2023+) spliced onto scaled BAA10Y proxy
    try:
        hy_spliced, hy_baa = fetch_hy_spliced(start=start, cache_dir=cache_dir)
        frames["hy_spread"]     = _to_daily(hy_spliced, ffill_limit=5, bday_index=bdays)
        frames["hy_spread_baa"] = _to_daily(hy_baa,     ffill_limit=5, bday_index=bdays)
    except Exception as exc:
        print(f"[market_data] WARNING: HY splice failed ({exc}); falling back to raw BAA10Y")

    # FRED series — BAA10Y still fetched (used by HY splice above); skip if already loaded
    fred_ids_required = ["DGS10", "DGS2", "DGS3MO", "T10Y3M", "DFF", "VIXCLS"]
    # BAA10Y loaded via fetch_hy_spliced; add it to required list only if splice failed
    if "hy_spread" not in frames:
        fred_ids_required.append("BAA10Y")
    fred_ids_optional = [
        "UNRATE", "NFCI", "ANFCI", "STLFSI4", "T10YIE",
        "DCOILWTICO", "DEXUSEU", "DEXJPUS", "ICSA",
        "DPSACBW027SBOG", "TOTLL", "WALCL", "CPIAUCSL", "USREC",
        # Credit spread complex + SLOOS
        "BAMLC0A0CM", "BAMLC0A4CBBB", "BAMLH0A0HYM2EY", "BAMLC0A0CMEY",
        "DRTSCILM",
    ]

    for sid in fred_ids_required + fred_ids_optional:
        try:
            s = fetch_fred(sid, start=start, cache_dir=cache_dir)
            lim = _FFILL_LIMITS.get(sid, 5)
            clean_name = _COL_NAMES.get(sid, sid)
            frames[clean_name] = _to_daily(s, ffill_limit=lim, bday_index=bdays)
        except Exception as exc:
            is_required = sid in fred_ids_required
            tag = "ERROR" if is_required else "WARNING"
            print(f"[market_data] {tag}: could not load {sid}: {exc}")

    df = pd.DataFrame(frames, index=bdays)
    df.index.name = "date"

    # Derived columns that belong to the data layer (not feature logic)
    if "yield_10y" in df.columns and "yield_2y" in df.columns:
        df["spread_10y2y"] = df["yield_10y"] - df["yield_2y"]

    # Breakeven imputation for pre-2003 (TIPS market didn't exist before 1997;
    # liquid data only from 2003). Fill with 2.5% as a neutral long-run proxy.
    # The imputed flag lets downstream code down-weight or exclude this period.
    if "breakeven_10y" in df.columns:
        df["breakeven_imputed"] = df["breakeven_10y"].isna().astype(int)
        df["breakeven_10y"] = df["breakeven_10y"].fillna(2.5)

    # Drop rows where all required daily series are missing simultaneously
    # (e.g. market holidays where every series is NaN)
    req_present = [c for c in _REQUIRED_COLS if c in df.columns]
    if req_present:
        df = df.dropna(subset=req_present, how="all")

    return df


# ---------------------------------------------------------------------------
# Freshness report
# ---------------------------------------------------------------------------

def get_freshness(
    start: str = _DEFAULT_START,
    cache_dir: Path = _CACHE_DIR,
) -> dict[str, dict]:
    """
    Return per-series freshness metadata without triggering a re-fetch.
    Reads only from cache if available.

    Returns dict:
      {series_id: {label, last_date, days_stale, cached, cache_age_h}}
    """
    today = pd.Timestamp.now().normalize()
    result: dict[str, dict] = {}

    for sid, label in {v: k for k, v in _COL_NAMES.items()}.items():
        path = _cache_path(label, start[:4])
        if path.exists():
            try:
                s = pd.read_parquet(path).squeeze()
                last = pd.Timestamp(s.index[-1]).normalize()
                bdays_stale = max(0, len(pd.bdate_range(last + pd.Timedelta(days=1), today)))
                age_h = (time.time() - path.stat().st_mtime) / 3600
                result[sid] = {
                    "label": label,
                    "last_date": str(last.date()),
                    "days_stale": bdays_stale,
                    "cached": True,
                    "cache_age_h": round(age_h, 1),
                }
            except Exception:
                result[sid] = {"label": label, "last_date": None,
                               "days_stale": None, "cached": True,
                               "cache_age_h": None}
        else:
            result[sid] = {"label": label, "last_date": None,
                           "days_stale": None, "cached": False,
                           "cache_age_h": None}

    # SP500 (yfinance)
    sp_path = _cache_path("SP500_yf", start[:4])
    if sp_path.exists():
        try:
            s = pd.read_parquet(sp_path).squeeze()
            last = pd.Timestamp(s.index[-1]).normalize()
            bdays_stale = max(0, len(pd.bdate_range(last + pd.Timedelta(days=1), today)))
            age_h = (time.time() - sp_path.stat().st_mtime) / 3600
            result["SP500"] = {
                "label": "sp500 (yfinance)",
                "last_date": str(last.date()),
                "days_stale": bdays_stale,
                "cached": True,
                "cache_age_h": round(age_h, 1),
            }
        except Exception:
            pass

    return result
