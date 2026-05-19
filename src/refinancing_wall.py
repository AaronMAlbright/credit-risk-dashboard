"""
Refinancing wall framework.

Public macro datasets do not provide a true issuer-level maturity wall. This
module creates a presentable placeholder framework and optional calculation if
future maturity-bucket columns are added.
"""

from __future__ import annotations

import pandas as pd


MATURITY_BUCKETS = (
    ("debt_due_1y", "Due <1Y"),
    ("debt_due_1_3y", "Due 1-3Y"),
    ("debt_due_3_5y", "Due 3-5Y"),
    ("debt_due_5y_plus", "Due 5Y+"),
)


def refinancing_wall_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return framework_placeholder()

    latest = df.iloc[-1]
    rows = []
    total = 0.0
    available = False
    for col, label in MATURITY_BUCKETS:
        val = latest.get(col)
        if pd.notna(val):
            available = True
            total += float(val)
            rows.append({"bucket": label, "amount": float(val)})

    if not available or total <= 0:
        return framework_placeholder()

    out = pd.DataFrame(rows)
    out["share"] = out["amount"] / total
    out["risk_note"] = out["share"].apply(
        lambda x: "High near-term wall" if x >= 0.35 else "Manageable share"
    )
    return out


def framework_placeholder() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Due <1Y", None, None, "Needs issuer/index maturity data"),
            ("Due 1-3Y", None, None, "Key refinancing wall bucket"),
            ("Due 3-5Y", None, None, "Medium-term maturity pressure"),
            ("Due 5Y+", None, None, "Longer runway; lower near-term pressure"),
        ],
        columns=["bucket", "amount", "share", "risk_note"],
    )


def refinancing_wall_summary(df: pd.DataFrame) -> str:
    table = refinancing_wall_table(df)
    if table["amount"].isna().all():
        return (
            "Refinancing wall requires issuer/index maturity data. "
            "Production version should track debt due by rating, sector, and maturity bucket."
        )
    near = table.loc[table["bucket"].isin(["Due <1Y", "Due 1-3Y"]), "share"].sum()
    return f"Near-term refinancing wall is {near:.0%} of mapped maturities."

