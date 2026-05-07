"""
Tests for src/data_pipeline.py.

All subprocess and external API calls are mocked — no real FRED calls or
app.py executions happen in tests.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_pipeline import (
    _CSV_PATH,
    _DEFAULT_TIMEOUT,
    _FRED_SERIES,
    check_api_key,
    fetch_source_dates,
    get_pipeline_status,
    run_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_proc(returncode=0, stdout="Done.", stderr=""):
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


# ---------------------------------------------------------------------------
# check_api_key
# ---------------------------------------------------------------------------


class TestCheckApiKey:
    def test_returns_dict(self):
        result = check_api_key()
        assert isinstance(result, dict)

    def test_has_available_key(self):
        result = check_api_key()
        assert "available" in result

    def test_available_true_when_key_set(self):
        with patch.dict(os.environ, {"FRED_API_KEY": "abc123xyz"}):
            result = check_api_key()
        assert result["available"] is True

    def test_available_false_when_key_missing(self):
        env = {k: v for k, v in os.environ.items() if k != "FRED_API_KEY"}
        with patch("src.data_pipeline.load_dotenv"), \
             patch.dict(os.environ, env, clear=True):
            result = check_api_key()
        assert result["available"] is False

    def test_key_preview_truncated(self):
        with patch.dict(os.environ, {"FRED_API_KEY": "secretkey123"}):
            result = check_api_key()
        assert result["key_preview"].endswith("...")
        assert len(result["key_preview"]) < len("secretkey123")

    def test_key_preview_none_when_missing(self):
        env = {k: v for k, v in os.environ.items() if k != "FRED_API_KEY"}
        with patch("src.data_pipeline.load_dotenv"), \
             patch.dict(os.environ, env, clear=True):
            result = check_api_key()
        assert result["key_preview"] is None


# ---------------------------------------------------------------------------
# fetch_source_dates
# ---------------------------------------------------------------------------


class TestFetchSourceDates:
    """Patch fredapi.Fred since it is imported inside the function body."""

    def _mock_fred_cls(self, dates_by_series: dict[str, str]):
        """Return a mock Fred class whose instances return known series."""
        def get_series(sid, observation_start=None):
            if sid not in dates_by_series:
                raise ValueError(f"Unknown series {sid}")
            last = pd.Timestamp(dates_by_series[sid])
            idx = pd.date_range("2024-01-01", last, freq="B")
            return pd.Series(1.0, index=idx)

        mock_instance = MagicMock()
        mock_instance.get_series.side_effect = get_series
        mock_cls = MagicMock(return_value=mock_instance)
        return mock_cls

    def _all_dates(self, date="2025-01-10"):
        return {sid: date for sid in _FRED_SERIES}

    def test_returns_dict(self):
        with patch("fredapi.Fred", self._mock_fred_cls(self._all_dates())), \
             patch("src.data_pipeline.load_dotenv"):
            result = fetch_source_dates()
        assert isinstance(result, dict)

    def test_all_series_in_result(self):
        with patch("fredapi.Fred", self._mock_fred_cls(self._all_dates())), \
             patch("src.data_pipeline.load_dotenv"):
            result = fetch_source_dates()
        for sid in _FRED_SERIES:
            assert sid in result

    def test_each_entry_has_expected_keys(self):
        with patch("fredapi.Fred", self._mock_fred_cls(self._all_dates())), \
             patch("src.data_pipeline.load_dotenv"):
            result = fetch_source_dates()
        for sid in _FRED_SERIES:
            for k in ["label", "last_date", "days_stale", "available"]:
                assert k in result[sid], f"Missing key {k} in {sid}"

    def test_available_true_for_fetched_series(self):
        with patch("fredapi.Fred", self._mock_fred_cls(self._all_dates())), \
             patch("src.data_pipeline.load_dotenv"):
            result = fetch_source_dates()
        assert all(result[sid]["available"] for sid in _FRED_SERIES)

    def test_days_stale_is_nonneg_int(self):
        # Use a date within the mock's date_range start ("2024-01-01")
        with patch("fredapi.Fred", self._mock_fred_cls(self._all_dates("2024-06-01"))), \
             patch("src.data_pipeline.load_dotenv"):
            result = fetch_source_dates()
        for sid in _FRED_SERIES:
            ds = result[sid]["days_stale"]
            assert isinstance(ds, int) and ds >= 0

    def test_unavailable_when_fred_raises_on_init(self):
        bad_cls = MagicMock(side_effect=ImportError("no fredapi"))
        with patch("fredapi.Fred", bad_cls), \
             patch("src.data_pipeline.load_dotenv"):
            result = fetch_source_dates()
        for sid in _FRED_SERIES:
            assert result[sid]["available"] is False
            assert result[sid]["last_date"] is None

    def test_series_error_marked_unavailable(self):
        mock_instance = MagicMock()
        mock_instance.get_series.side_effect = RuntimeError("API error")
        with patch("fredapi.Fred", MagicMock(return_value=mock_instance)), \
             patch("src.data_pipeline.load_dotenv"):
            result = fetch_source_dates()
        for sid in _FRED_SERIES:
            assert result[sid]["available"] is False

    def test_label_matches_registry(self):
        with patch("fredapi.Fred", self._mock_fred_cls(self._all_dates())), \
             patch("src.data_pipeline.load_dotenv"):
            result = fetch_source_dates()
        for sid, label in _FRED_SERIES.items():
            assert result[sid]["label"] == label


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def test_returns_dict(self, tmp_path):
        csv = tmp_path / "data" / "scored_macro_credit_data.csv"
        csv.parent.mkdir()
        csv.write_text("date,val\n2025-01-10,1.0\n")
        with patch("subprocess.run", return_value=_make_proc()) as mock_run, \
             patch("src.data_pipeline._CSV_PATH", csv):
            result = run_pipeline(project_root=str(tmp_path))
        assert isinstance(result, dict)

    def test_all_keys_present(self, tmp_path):
        csv = tmp_path / "data" / "scored_macro_credit_data.csv"
        csv.parent.mkdir()
        csv.write_text("date,val\n2025-01-10,1.0\n")
        with patch("subprocess.run", return_value=_make_proc()), \
             patch("src.data_pipeline._CSV_PATH", csv):
            result = run_pipeline(project_root=str(tmp_path))
        for key in ["success", "elapsed_s", "csv_last_date",
                    "stdout", "stderr", "error", "refreshed_at"]:
            assert key in result, f"Missing key: {key}"

    def test_no_crash_on_zero_returncode(self, tmp_path):
        # returncode=0 should not raise; result must have all required keys
        csv = tmp_path / "data" / "scored_macro_credit_data.csv"
        csv.parent.mkdir()
        csv.write_text("date,val\n2025-05-01,1.0\n")
        with patch("subprocess.run", return_value=_make_proc(returncode=0)), \
             patch("src.data_pipeline._CSV_PATH", csv):
            result = run_pipeline(project_root=str(tmp_path))
        for key in ["success", "elapsed_s", "csv_last_date", "error", "refreshed_at"]:
            assert key in result

    def test_failure_on_nonzero_returncode(self, tmp_path):
        csv = tmp_path / "data" / "scored_macro_credit_data.csv"
        csv.parent.mkdir()
        csv.write_text("date,val\n2025-01-10,1.0\n")
        with patch("subprocess.run", return_value=_make_proc(returncode=1, stderr="Error!")), \
             patch("src.data_pipeline._CSV_PATH", csv):
            result = run_pipeline(project_root=str(tmp_path))
        assert result["success"] is False
        assert result["error"] is not None

    def test_timeout_returns_failure(self, tmp_path):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("app.py", 10)):
            result = run_pipeline(timeout=10, project_root=str(tmp_path))
        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    def test_exception_returns_failure(self, tmp_path):
        with patch("subprocess.run", side_effect=OSError("python not found")):
            result = run_pipeline(project_root=str(tmp_path))
        assert result["success"] is False
        assert result["error"] is not None

    def test_elapsed_s_is_nonneg_float(self, tmp_path):
        csv = tmp_path / "data" / "scored_macro_credit_data.csv"
        csv.parent.mkdir()
        csv.write_text("date,val\n2025-01-10,1.0\n")
        with patch("subprocess.run", return_value=_make_proc()), \
             patch("src.data_pipeline._CSV_PATH", csv):
            result = run_pipeline(project_root=str(tmp_path))
        assert isinstance(result["elapsed_s"], float)
        assert result["elapsed_s"] >= 0

    def test_stdout_in_result(self, tmp_path):
        csv = tmp_path / "data" / "scored_macro_credit_data.csv"
        csv.parent.mkdir()
        csv.write_text("date,val\n2025-01-10,1.0\n")
        with patch("subprocess.run", return_value=_make_proc(stdout="Pipeline complete")), \
             patch("src.data_pipeline._CSV_PATH", csv):
            result = run_pipeline(project_root=str(tmp_path))
        assert "Pipeline complete" in result["stdout"]

    def test_refreshed_at_is_iso_string(self, tmp_path):
        csv = tmp_path / "data" / "scored_macro_credit_data.csv"
        csv.parent.mkdir()
        csv.write_text("date,val\n2025-01-10,1.0\n")
        with patch("subprocess.run", return_value=_make_proc()), \
             patch("src.data_pipeline._CSV_PATH", csv):
            result = run_pipeline(project_root=str(tmp_path))
        from datetime import datetime
        # Should parse without error
        datetime.fromisoformat(result["refreshed_at"])

    def test_subprocess_called_with_python_app(self, tmp_path):
        csv = tmp_path / "data" / "scored_macro_credit_data.csv"
        csv.parent.mkdir()
        csv.write_text("date,val\n2025-01-10,1.0\n")
        with patch("subprocess.run", return_value=_make_proc()) as mock_run, \
             patch("src.data_pipeline._CSV_PATH", csv):
            run_pipeline(project_root=str(tmp_path))
        args = mock_run.call_args[0][0]
        assert args[0] == "python"
        assert "app.py" in args[1]

    def test_timeout_passed_to_subprocess(self, tmp_path):
        csv = tmp_path / "data" / "scored_macro_credit_data.csv"
        csv.parent.mkdir()
        csv.write_text("date,val\n2025-01-10,1.0\n")
        with patch("subprocess.run", return_value=_make_proc()) as mock_run, \
             patch("src.data_pipeline._CSV_PATH", csv):
            run_pipeline(timeout=999, project_root=str(tmp_path))
        kwargs = mock_run.call_args[1]
        assert kwargs.get("timeout") == 999

    def test_csv_last_date_parsed(self, tmp_path):
        csv = tmp_path / "data" / "scored_macro_credit_data.csv"
        csv.parent.mkdir()
        csv.write_text("date,val\n2025-05-01,1.0\n2025-05-06,2.0\n")
        with patch("subprocess.run", return_value=_make_proc()), \
             patch("src.data_pipeline._CSV_PATH", csv):
            result = run_pipeline(project_root=str(tmp_path))
        assert result["csv_last_date"] == "2025-05-06"


# ---------------------------------------------------------------------------
# get_pipeline_status
# ---------------------------------------------------------------------------


class TestGetPipelineStatus:
    def test_returns_dict(self):
        result = get_pipeline_status(fast=True)
        assert isinstance(result, dict)

    def test_all_keys_present(self):
        result = get_pipeline_status(fast=True)
        for key in ["key_status", "csv_last_date", "csv_bdays_stale", "source_dates"]:
            assert key in result

    def test_fast_mode_has_empty_source_dates(self):
        result = get_pipeline_status(fast=True)
        assert result["source_dates"] == {}

    def test_key_status_is_dict(self):
        result = get_pipeline_status(fast=True)
        assert isinstance(result["key_status"], dict)
        assert "available" in result["key_status"]

    def test_csv_last_date_uses_real_file(self):
        result = get_pipeline_status(fast=True)
        if _CSV_PATH.exists():
            assert result["csv_last_date"] is not None
            # Should be a date string
            pd.Timestamp(result["csv_last_date"])

    def test_csv_bdays_stale_nonneg_when_file_exists(self):
        result = get_pipeline_status(fast=True)
        if _CSV_PATH.exists():
            assert isinstance(result["csv_bdays_stale"], int)
            assert result["csv_bdays_stale"] >= 0

    def test_slow_mode_calls_fetch_source_dates(self):
        mock_dates = {sid: {"label": lbl, "last_date": "2025-01-10",
                            "days_stale": 5, "available": True}
                      for sid, lbl in _FRED_SERIES.items()}
        with patch("src.data_pipeline.fetch_source_dates", return_value=mock_dates):
            result = get_pipeline_status(fast=False)
        assert result["source_dates"] == mock_dates


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_fred_series_is_dict(self):
        assert isinstance(_FRED_SERIES, dict)

    def test_fred_series_non_empty(self):
        assert len(_FRED_SERIES) > 0

    def test_default_timeout_reasonable(self):
        assert 60 <= _DEFAULT_TIMEOUT <= 600

    def test_csv_path_is_pathlib(self):
        assert isinstance(_CSV_PATH, Path)
