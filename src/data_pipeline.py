"""
Live data pipeline.

Orchestrates a full data refresh by invoking the existing app.py scoring
pipeline as a subprocess, verifying the output CSV was updated, and reporting
per-series FRED freshness without running the full pipeline.

Public API
----------
  check_api_key        — verifies FRED_API_KEY is available in environment
  fetch_source_dates   — lightweight per-series latest-observation dates
  run_pipeline         — runs app.py and returns structured status dict
  get_pipeline_status  — combined freshness + key status for sidebar display
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data_sources import fred_source_labels

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):  # type: ignore[misc]
        pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FRED_SERIES: dict[str, str] = fred_source_labels()

_DEFAULT_TIMEOUT  = 360     # seconds — pipeline can be slow on first run
_CSV_PATH         = Path("data/scored_macro_credit_data.csv")
_APP_SCRIPT       = "app.py"
_CORE_REFRESH_COLUMNS = ["yield_10y", "yield_2y", "vix", "hy_spread", "sp500", "unemployment", "nfci"]


# ---------------------------------------------------------------------------
# Pre-flight helpers
# ---------------------------------------------------------------------------

def check_api_key() -> dict:
    """
    Verify FRED_API_KEY is present in the environment (via .env or shell).

    Returns dict: {available: bool, key_preview: str | None}
    key_preview shows first 6 chars + '...' — never the full key.
    """
    load_dotenv()
    key = os.environ.get("FRED_API_KEY", "")
    if key:
        return {"available": True, "key_preview": key[:6] + "..."}
    return {"available": False, "key_preview": None}


def fetch_source_dates(start_date: str = "2020-01-01") -> dict[str, dict]:
    """
    Pull the latest observation date for each FRED series without running the
    full scoring pipeline. Useful for the sidebar freshness panel.

    Returns dict: {series_id: {label, last_date, days_stale, available}}
    'days_stale' is business-days since last observation vs. today.
    If a series cannot be fetched, available=False and last_date=None.
    """
    try:
        load_dotenv()
        from fredapi import Fred
        fred = Fred(api_key=os.environ.get("FRED_API_KEY", ""))
    except Exception:
        return {sid: {"label": lbl, "last_date": None, "days_stale": None,
                      "available": False}
                for sid, lbl in _FRED_SERIES.items()}

    today = pd.Timestamp.now().normalize()
    result: dict[str, dict] = {}

    for sid, label in _FRED_SERIES.items():
        try:
            data = fred.get_series(sid, observation_start=start_date)
            data = data.dropna()
            if data.empty:
                result[sid] = {"label": label, "last_date": None,
                               "days_stale": None, "available": False}
                continue
            last = pd.Timestamp(data.index[-1]).normalize()
            bdays_stale = max(0, len(pd.bdate_range(
                last + pd.Timedelta(days=1), today
            )))
            result[sid] = {
                "label":       label,
                "last_date":   str(last.date()),
                "days_stale":  bdays_stale,
                "available":   True,
            }
        except Exception as exc:
            result[sid] = {"label": label, "last_date": None,
                           "days_stale": None, "available": False,
                           "error": str(exc)}

    return result


def get_csv_freshness(csv_path: Path | str = _CSV_PATH) -> dict:
    """Return latest scored CSV date and business-day staleness."""
    path = Path(csv_path)
    if not path.exists():
        return {"csv_last_date": None, "csv_bdays_stale": None, "row_count": 0}

    try:
        df = pd.read_csv(path, usecols=["date"])
        last = pd.to_datetime(df["date"]).max().normalize()
        today = pd.Timestamp.now().normalize()
        bdays_stale = max(0, len(pd.bdate_range(last + pd.Timedelta(days=1), today)))
        return {
            "csv_last_date": str(last.date()),
            "csv_bdays_stale": bdays_stale,
            "row_count": int(len(df)),
        }
    except Exception as exc:
        return {
            "csv_last_date": None,
            "csv_bdays_stale": None,
            "row_count": 0,
            "error": str(exc),
        }


def diagnose_refresh_limit(
    start: str = "1999-01-01",
    csv_path: Path | str = _CSV_PATH,
) -> dict:
    """
    Diagnose whether scored data freshness is limited by raw source columns or
    by a pipeline output that has not been regenerated.

    This is intentionally separate from the fast sidebar status because loading
    raw market data can touch cache/network-backed data sources.
    """
    csv = get_csv_freshness(csv_path)
    try:
        from src.market_data import load_all_series

        raw = load_all_series(start=start)
    except Exception as exc:
        return {
            **csv,
            "available": False,
            "reason": f"raw source load failed: {exc}",
            "raw_last_date": None,
            "latest_complete_core_date": None,
            "limiting_columns": [],
            "column_last_dates": {},
        }

    column_last_dates: dict[str, str | None] = {}
    for col in _CORE_REFRESH_COLUMNS:
        if col not in raw.columns:
            column_last_dates[col] = None
            continue
        s = raw[col].dropna()
        column_last_dates[col] = None if s.empty else str(pd.Timestamp(s.index[-1]).date())

    present_core = [col for col in _CORE_REFRESH_COLUMNS if col in raw.columns]
    latest_complete = None
    if present_core:
        complete = raw.dropna(subset=present_core)
        if not complete.empty:
            latest_complete = pd.Timestamp(complete.index[-1]).normalize()

    raw_last = pd.Timestamp(raw.index[-1]).normalize() if not raw.empty else None
    csv_last = pd.Timestamp(csv["csv_last_date"]).normalize() if csv.get("csv_last_date") else None

    limiting_columns: list[str] = []
    reason = "current"
    if raw_last is None or latest_complete is None:
        reason = "raw sources unavailable"
    elif latest_complete < raw_last:
        reason = "core source column limited"
        limiting_columns = [
            col for col, last in column_last_dates.items()
            if last == str(latest_complete.date())
        ]
    elif csv_last is not None and csv_last < latest_complete:
        reason = "pipeline output stale"
    elif csv_last is None:
        reason = "scored CSV unavailable"

    return {
        **csv,
        "available": True,
        "reason": reason,
        "raw_last_date": None if raw_last is None else str(raw_last.date()),
        "latest_complete_core_date": None if latest_complete is None else str(latest_complete.date()),
        "limiting_columns": limiting_columns,
        "column_last_dates": column_last_dates,
    }


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    timeout: int = _DEFAULT_TIMEOUT,
    project_root: str | None = None,
) -> dict:
    """
    Execute app.py as a subprocess and verify the output CSV was updated.

    Parameters
    ----------
    timeout      : seconds before the subprocess is killed
    project_root : working directory for the subprocess; defaults to cwd

    Returns dict:
      success        — bool
      elapsed_s      — float, wall-clock seconds
      csv_last_date  — str | None, last date in the refreshed CSV
      stdout         — str (truncated to 4000 chars)
      stderr         — str (truncated to 2000 chars)
      error          — str | None, human-readable failure reason
      refreshed_at   — ISO timestamp of when the run completed
    """
    cwd = project_root or str(Path.cwd())
    csv_mtime_before = _CSV_PATH.stat().st_mtime if _CSV_PATH.exists() else None
    t0 = time.monotonic()

    try:
        proc = subprocess.run(
            ["python", _APP_SCRIPT],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        elapsed = time.monotonic() - t0
        stdout = proc.stdout[-4000:] if proc.stdout else ""
        stderr = proc.stderr[-2000:] if proc.stderr else ""

        if proc.returncode != 0:
            return {
                "success":       False,
                "elapsed_s":     round(elapsed, 1),
                "csv_last_date": None,
                "stdout":        stdout,
                "stderr":        stderr,
                "error":         f"app.py exited with code {proc.returncode}",
                "refreshed_at":  datetime.now().isoformat(),
            }

        # Verify the CSV was actually updated
        csv_mtime_after = _CSV_PATH.stat().st_mtime if _CSV_PATH.exists() else None
        csv_updated = (csv_mtime_after is not None and csv_mtime_after != csv_mtime_before)

        csv_last_date = None
        if _CSV_PATH.exists():
            try:
                df = pd.read_csv(_CSV_PATH, usecols=["date"])
                csv_last_date = str(pd.to_datetime(df["date"]).max().date())
            except Exception:
                pass

        return {
            "success":       csv_updated or csv_mtime_before is None,
            "elapsed_s":     round(elapsed, 1),
            "csv_last_date": csv_last_date,
            "stdout":        stdout,
            "stderr":        stderr,
            "error":         None if (csv_updated or csv_mtime_before is None)
                             else "Pipeline ran but CSV was not updated.",
            "refreshed_at":  datetime.now().isoformat(),
        }

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        return {
            "success":       False,
            "elapsed_s":     round(elapsed, 1),
            "csv_last_date": None,
            "stdout":        "",
            "stderr":        "",
            "error":         f"Pipeline timed out after {timeout}s.",
            "refreshed_at":  datetime.now().isoformat(),
        }
    except Exception as exc:
        elapsed = time.monotonic() - t0
        return {
            "success":       False,
            "elapsed_s":     round(elapsed, 1),
            "csv_last_date": None,
            "stdout":        "",
            "stderr":        "",
            "error":         str(exc),
            "refreshed_at":  datetime.now().isoformat(),
        }


def get_pipeline_status(fast: bool = True, include_diagnostic: bool = False) -> dict:
    """
    Combined status for the sidebar display.

    Parameters
    ----------
    fast : if True, skip FRED per-series fetch (key check only).
           Set False to show per-series staleness (adds ~3-5s).

    Returns dict:
      key_status     — output of check_api_key()
      csv_last_date  — str | None
      csv_bdays_stale— int
      source_dates   — dict from fetch_source_dates() or {} when fast=True
      refresh_diagnostic — optional diagnose_refresh_limit() result
    """
    key_status = check_api_key()
    csv = get_csv_freshness()

    source_dates = fetch_source_dates() if not fast else {}
    refresh_diagnostic = diagnose_refresh_limit() if include_diagnostic else {}

    return {
        "key_status":      key_status,
        "csv_last_date":   csv["csv_last_date"],
        "csv_bdays_stale": csv["csv_bdays_stale"],
        "row_count":       csv["row_count"],
        "source_dates":    source_dates,
        "refresh_diagnostic": refresh_diagnostic,
    }
