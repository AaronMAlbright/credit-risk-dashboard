"""
Credit relative-value and quality segmentation.

Adds finance-native valuation context around spread compensation:
- HY and IG spread percentile ranks
- HY/IG spread ratio
- BBB/IG quality pressure where BBB spread is available
- excess-spread percentile where spread decomposition is present
- quality tilt recommendation
"""

from __future__ import annotations

import pandas as pd

from src.spread_decomposition import decompose_spreads


MIN_PERCENTILE_OBS = 60


def _spread_bps(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    med = vals.dropna().abs().median()
    if pd.notna(med) and med < 50:
        return vals * 100.0
    return vals


def _expanding_percentile(series: pd.Series, min_obs: int = MIN_PERCENTILE_OBS) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")

    def pct(window: pd.Series) -> float:
        current = window.iloc[-1]
        hist = window.dropna()
        if pd.isna(current) or len(hist) < min_obs:
            return float("nan")
        return float((hist <= current).mean() * 100.0)

    return s.expanding(min_periods=min_obs).apply(pct, raw=False)


def _valuation_bucket(percentile: float) -> str:
    if pd.isna(percentile):
        return "Unavailable"
    if percentile >= 90:
        return "Crisis / Very Cheap"
    if percentile >= 75:
        return "Cheap"
    if percentile >= 40:
        return "Fair"
    if percentile >= 20:
        return "Rich"
    return "Very Rich"


def _quality_tilt(latest: pd.Series) -> str:
    hy_pct = latest.get("hy_spread_percentile")
    ig_pct = latest.get("ig_spread_percentile")
    bbb_ig = latest.get("bbb_ig_ratio")

    if pd.notna(bbb_ig) and bbb_ig > 1.35:
        return "Upgrade quality; BBB pressure is elevated."
    if pd.notna(hy_pct) and pd.notna(ig_pct):
        if hy_pct - ig_pct >= 20:
            return "HY screens cheaper than IG, but confirm defaults and liquidity first."
        if ig_pct - hy_pct >= 20:
            return "Prefer IG; HY is not offering enough relative compensation."
    if pd.notna(hy_pct) and hy_pct >= 75:
        return "HY compensation is elevated; add risk only with improving technicals."
    if pd.notna(hy_pct) and hy_pct <= 25:
        return "Avoid chasing HY beta; spreads are historically tight."
    return "Balanced quality stance."


def compute_credit_relative_value(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(index=df.index)

    out = pd.DataFrame(index=df.index)

    if "hy_spread" in df.columns:
        out["hy_spread_bps"] = _spread_bps(df["hy_spread"])
        out["hy_spread_percentile"] = _expanding_percentile(out["hy_spread_bps"])
        out["hy_spread_valuation"] = out["hy_spread_percentile"].apply(_valuation_bucket)

    if "ig_spread" in df.columns:
        out["ig_spread_bps"] = _spread_bps(df["ig_spread"])
        out["ig_spread_percentile"] = _expanding_percentile(out["ig_spread_bps"])
        out["ig_spread_valuation"] = out["ig_spread_percentile"].apply(_valuation_bucket)

    if {"hy_spread_bps", "ig_spread_bps"}.issubset(out.columns):
        out["hy_ig_spread_ratio"] = out["hy_spread_bps"] / out["ig_spread_bps"].replace(0, pd.NA)
        out["hy_ig_ratio_percentile"] = _expanding_percentile(out["hy_ig_spread_ratio"])

    if "bbb_spread" in df.columns and "ig_spread_bps" in out.columns:
        out["bbb_spread_bps"] = _spread_bps(df["bbb_spread"])
        out["bbb_ig_ratio"] = out["bbb_spread_bps"] / out["ig_spread_bps"].replace(0, pd.NA)
        out["bbb_ig_ratio_percentile"] = _expanding_percentile(out["bbb_ig_ratio"])

    decomp = decompose_spreads(df)
    if not decomp.empty and "excess_spread_bps" in decomp.columns:
        out["excess_spread_bps"] = decomp["excess_spread_bps"]
        out["excess_spread_percentile"] = _expanding_percentile(out["excess_spread_bps"])
        out["excess_spread_valuation"] = out["excess_spread_percentile"].apply(_valuation_bucket)

    if out.empty:
        return out

    out["quality_tilt"] = out.apply(_quality_tilt, axis=1)
    return out


def latest_relative_value_snapshot(df: pd.DataFrame) -> dict:
    rv = compute_credit_relative_value(df)
    if rv.empty:
        return {"available": False, "reason": "No spread columns available"}

    valid = rv.dropna(how="all")
    if valid.empty:
        return {"available": False, "reason": "No valid relative-value observations"}

    row = valid.iloc[-1]

    def val(key):
        x = row.get(key)
        return None if pd.isna(x) else x

    return {
        "available": True,
        "hy_spread_bps": val("hy_spread_bps"),
        "hy_spread_percentile": val("hy_spread_percentile"),
        "hy_spread_valuation": val("hy_spread_valuation"),
        "ig_spread_bps": val("ig_spread_bps"),
        "ig_spread_percentile": val("ig_spread_percentile"),
        "ig_spread_valuation": val("ig_spread_valuation"),
        "hy_ig_spread_ratio": val("hy_ig_spread_ratio"),
        "hy_ig_ratio_percentile": val("hy_ig_ratio_percentile"),
        "bbb_ig_ratio": val("bbb_ig_ratio"),
        "bbb_ig_ratio_percentile": val("bbb_ig_ratio_percentile"),
        "excess_spread_percentile": val("excess_spread_percentile"),
        "excess_spread_valuation": val("excess_spread_valuation"),
        "quality_tilt": val("quality_tilt"),
    }


def relative_value_table(df: pd.DataFrame) -> pd.DataFrame:
    snap = latest_relative_value_snapshot(df)
    if not snap.get("available"):
        return pd.DataFrame()

    rows = [
        ("HY OAS", snap.get("hy_spread_bps"), snap.get("hy_spread_percentile"), snap.get("hy_spread_valuation")),
        ("IG OAS", snap.get("ig_spread_bps"), snap.get("ig_spread_percentile"), snap.get("ig_spread_valuation")),
        ("HY / IG Ratio", snap.get("hy_ig_spread_ratio"), snap.get("hy_ig_ratio_percentile"), None),
        ("BBB / IG Ratio", snap.get("bbb_ig_ratio"), snap.get("bbb_ig_ratio_percentile"), None),
        ("Excess Spread", None, snap.get("excess_spread_percentile"), snap.get("excess_spread_valuation")),
    ]
    return pd.DataFrame(rows, columns=["metric", "level", "percentile", "valuation"])

