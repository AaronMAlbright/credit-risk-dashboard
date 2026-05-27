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


def get_pipeline_status(fast: bool = True) -> dict:
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
    """
    key_status = check_api_key()

    csv_last_date = None
    csv_bdays_stale = None
    if _CSV_PATH.exists():
        try:
            df = pd.read_csv(_CSV_PATH, usecols=["date"])
            last = pd.to_datetime(df["date"]).max().normalize()
            csv_last_date = str(last.date())
            today = pd.Timestamp.now().normalize()
            csv_bdays_stale = max(0, len(pd.bdate_range(
                last + pd.Timedelta(days=1), today
            )))
        except Exception:
            pass

    source_dates = fetch_source_dates() if not fast else {}

    return {
        "key_status":      key_status,
        "csv_last_date":   csv_last_date,
        "csv_bdays_stale": csv_bdays_stale,
        "source_dates":    source_dates,
    }
