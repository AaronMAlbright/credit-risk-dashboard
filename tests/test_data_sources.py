from src.data_pipeline import _FRED_SERIES
from src.data_sources import SOURCES, format_source_note, source_rows


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
