"""
Macro chronology overlays.

Canonical registry of named macro events and Fed policy cycles for
overlaying on charts (regime timeline, composite score, drawdown).

Public API
----------
  STRESS_EVENTS      — named market stress episodes
  FED_CYCLES         — Fed hiking / cutting / QE / QT periods
  get_events_df      — stress events as DataFrame
  get_fed_cycles_df  — Fed cycles as DataFrame
  get_all_overlays   — combined sorted list for chart annotation
"""

from __future__ import annotations
import pandas as pd

# ---------------------------------------------------------------------------
# Named stress events (point-in-time markers)
# ---------------------------------------------------------------------------

STRESS_EVENTS: list[dict] = [
    {"date": "2008-09-15", "label": "Lehman failure",           "category": "crisis",    "color": "#e74c3c"},
    {"date": "2008-10-10", "label": "GFC market bottom",        "category": "crisis",    "color": "#e74c3c"},
    {"date": "2009-03-09", "label": "SP500 bottom (GFC)",       "category": "recovery",  "color": "#27ae60"},
    {"date": "2010-05-06", "label": "Flash Crash",              "category": "episode",   "color": "#e67e22"},
    {"date": "2011-08-05", "label": "US debt downgrade",        "category": "episode",   "color": "#e67e22"},
    {"date": "2011-10-04", "label": "EU crisis low",            "category": "episode",   "color": "#e67e22"},
    {"date": "2015-08-24", "label": "China devaluation shock",  "category": "episode",   "color": "#e67e22"},
    {"date": "2016-02-11", "label": "China/oil trough",         "category": "recovery",  "color": "#27ae60"},
    {"date": "2018-12-24", "label": "Q4 2018 bottom",           "category": "episode",   "color": "#e67e22"},
    {"date": "2020-02-19", "label": "COVID peak (SP500)",       "category": "crisis",    "color": "#e74c3c"},
    {"date": "2020-03-23", "label": "COVID trough",             "category": "crisis",    "color": "#e74c3c"},
    {"date": "2020-03-23", "label": "COVID bottom / Fed QE∞",  "category": "recovery",  "color": "#27ae60"},
    {"date": "2022-01-03", "label": "2022 rate shock begins",   "category": "episode",   "color": "#e67e22"},
    {"date": "2022-10-13", "label": "2022 bear market low",     "category": "recovery",  "color": "#27ae60"},
    {"date": "2023-03-10", "label": "SVB collapse",             "category": "episode",   "color": "#e67e22"},
    {"date": "2024-08-05", "label": "Yen carry unwind",         "category": "episode",   "color": "#e67e22"},
    {"date": "2025-04-02", "label": "Liberation Day tariffs",   "category": "episode",   "color": "#e67e22"},
]

# ---------------------------------------------------------------------------
# Fed policy cycles (shaded bands)
# ---------------------------------------------------------------------------

FED_CYCLES: list[dict] = [
    # Hiking cycles
    {"start": "2004-06-30", "end": "2006-06-29", "label": "Fed hikes 2004–06",     "type": "hike",  "color": "rgba(231,76,60,0.10)"},
    {"start": "2015-12-16", "end": "2018-12-19", "label": "Fed hikes 2015–18",     "type": "hike",  "color": "rgba(231,76,60,0.10)"},
    {"start": "2022-03-16", "end": "2023-07-26", "label": "Fed hikes 2022–23",     "type": "hike",  "color": "rgba(231,76,60,0.13)"},

    # Cutting cycles
    {"start": "2007-09-18", "end": "2008-12-16", "label": "Fed cuts 2007–08",      "type": "cut",   "color": "rgba(39,174,96,0.10)"},
    {"start": "2019-07-31", "end": "2019-10-30", "label": "Fed mid-cycle cuts",    "type": "cut",   "color": "rgba(39,174,96,0.08)"},
    {"start": "2020-03-03", "end": "2020-03-15", "label": "COVID emergency cuts",  "type": "cut",   "color": "rgba(39,174,96,0.10)"},
    {"start": "2024-09-18", "end": "2025-01-29", "label": "Fed cuts 2024–25",      "type": "cut",   "color": "rgba(39,174,96,0.10)"},

    # QE phases
    {"start": "2008-11-25", "end": "2010-03-31", "label": "QE1",                   "type": "qe",    "color": "rgba(52,152,219,0.08)"},
    {"start": "2010-11-03", "end": "2011-06-30", "label": "QE2",                   "type": "qe",    "color": "rgba(52,152,219,0.08)"},
    {"start": "2012-09-13", "end": "2014-10-29", "label": "QE3",                   "type": "qe",    "color": "rgba(52,152,219,0.08)"},
    {"start": "2020-03-15", "end": "2022-03-16", "label": "QE∞ (COVID)",           "type": "qe",    "color": "rgba(52,152,219,0.10)"},

    # QT phases
    {"start": "2017-10-01", "end": "2019-09-30", "label": "QT1",                   "type": "qt",    "color": "rgba(155,89,182,0.08)"},
    {"start": "2022-06-01", "end": "2025-05-01", "label": "QT2",                   "type": "qt",    "color": "rgba(155,89,182,0.08)"},

    # ZIRP periods
    {"start": "2008-12-16", "end": "2015-12-16", "label": "ZIRP 2008–15",          "type": "zirp",  "color": "rgba(243,156,18,0.05)"},
    {"start": "2020-03-15", "end": "2022-03-16", "label": "ZIRP 2020–22",          "type": "zirp",  "color": "rgba(243,156,18,0.05)"},
]


def get_events_df() -> pd.DataFrame:
    """Return STRESS_EVENTS as a DataFrame with parsed date index."""
    df = pd.DataFrame(STRESS_EVENTS)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def get_fed_cycles_df() -> pd.DataFrame:
    """Return FED_CYCLES as a DataFrame with parsed start/end."""
    df = pd.DataFrame(FED_CYCLES)
    df["start"] = pd.to_datetime(df["start"])
    df["end"]   = pd.to_datetime(df["end"])
    return df.sort_values("start").reset_index(drop=True)


def get_all_overlays(start: str | None = None, end: str | None = None) -> dict:
    """
    Return both events and cycles filtered to [start, end].

    Returns {events: DataFrame, cycles: DataFrame}.
    """
    evts = get_events_df()
    cycs = get_fed_cycles_df()

    if start:
        s = pd.Timestamp(start)
        evts = evts[evts.index >= s]
        cycs = cycs[cycs["end"] >= s]

    if end:
        e = pd.Timestamp(end)
        evts = evts[evts.index <= e]
        cycs = cycs[cycs["start"] <= e]

    return {"events": evts, "cycles": cycs}
