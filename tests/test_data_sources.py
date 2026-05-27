from pathlib import Path

import pandas as pd

from src.data_pipeline import _FRED_SERIES
from src.data_sources import (
    SOURCES,
    column_quality,
    format_source_note,
    registered_columns,
    source_by_column,
    source_rows,
)


def test_pipeline_freshness_series_have_source_metadata():
    assert set(_FRED_SERIES) == set(SOURCES)


def test_source_rows_are_table_ready():
    rows = source_rows()
    assert rows
    required = {
        "series_id", "label", "provider", "column",
        "frequency", "quality", "transform", "limitation",
    }
    assert required.issubset(rows[0])


def test_default_cycle_sources_are_observed_quarterly_series():
    for series_id in ["DRBLACBS", "CORBLACBS"]:
        source = SOURCES[series_id]
        assert source.frequency == "quarterly"
        assert source.quality == "observed"


def test_format_source_note_includes_ids_and_quality():
    note = format_source_note("DRBLACBS", "CORBLACBS")
    assert "DRBLACBS" in note
    assert "CORBLACBS" in note
    assert "observed" in note


def test_registered_columns_exist_in_scored_csv():
    csv_columns = set(pd.read_csv("data/scored_macro_credit_data.csv", nrows=0).columns)
    missing = registered_columns() - csv_columns
    assert not missing


def test_column_quality_identifies_registered_and_derived_columns():
    assert column_quality("hy_spread") == "observed_with_proxy_history"
    assert column_quality("composite_risk_score") == "derived/unregistered"


def test_source_lookup_by_column():
    source = source_by_column("business_chargeoff_rate")
    assert source is not None
    assert source.series_id == "CORBLACBS"


def test_data_dictionary_mentions_every_registered_series():
    text = Path("docs/data_dictionary.md").read_text()
    missing = [series_id for series_id in SOURCES if series_id not in text]
    assert not missing
