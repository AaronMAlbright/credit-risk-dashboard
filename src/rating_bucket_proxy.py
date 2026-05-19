"""
Public-data rating bucket proxy view.

This is not a substitute for bond-level rating bucket data. It gives the app a
clean placeholder framework for discussing IG, BBB, HY, and distressed-quality
pressure with available public spread/proxy columns.
"""

from __future__ import annotations

import pandas as pd


def _latest(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return None if s.empty else float(s.iloc[-1])


def _chg(df: pd.DataFrame, col: str, days: int):
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) <= days:
        return None
    return float(s.iloc[-1] - s.iloc[-days - 1])


def _regime(level, chg_1m, tight, wide):
    if level is None:
        return "Unavailable"
    if level >= wide:
        return "Stressed"
    if chg_1m is not None and chg_1m > 25:
        return "Deteriorating"
    if level <= tight:
        return "Rich / Compressed"
    return "Normal"


def rating_bucket_proxy_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    specs = [
        ("IG", "ig_spread", "Investment-grade spread proxy", 90, 180),
        ("BBB", "bbb_spread", "Lower-quality IG / fallen-angel pressure proxy", 120, 250),
        ("HY", "hy_spread", "High-yield spread proxy", 300, 600),
        ("Distressed", "distressed_ratio", "Distressed debt pressure proxy", 5, 20),
    ]
    for bucket, col, desc, tight, wide in specs:
        level = _latest(df, col)
        chg_1m = _chg(df, col, 21)
        if level is not None and col.endswith("_spread") and abs(level) < 50:
            level *= 100
            chg_1m = chg_1m * 100 if chg_1m is not None else None
        rows.append(
            {
                "bucket": bucket,
                "proxy": col,
                "level": level,
                "change_1m": chg_1m,
                "regime": _regime(level, chg_1m, tight, wide),
                "interpretation": desc,
            }
        )

    return pd.DataFrame(rows)


def rating_bucket_summary(df: pd.DataFrame) -> str:
    table = rating_bucket_proxy_table(df)
    if table.empty:
        return "Rating bucket proxies unavailable."
    available = table[table["regime"] != "Unavailable"]
    if available.empty:
        return "Rating bucket proxies unavailable with current public dataset."
    stressed = available[available["regime"].isin(["Stressed", "Deteriorating"])]
    if not stressed.empty:
        names = ", ".join(stressed["bucket"].tolist())
        return f"Quality pressure is concentrated in: {names}."
    return "No public rating-bucket proxy is currently flagging acute pressure."

