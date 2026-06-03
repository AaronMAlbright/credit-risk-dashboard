"""Override tracking for the credit compensation scorecard."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.credit_compensation_scorecard import RECOMMENDATIONS


OVERRIDE_PATH = Path("history") / "credit_compensation_overrides.csv"

OVERRIDE_COLUMNS = [
    "override_id",
    "created_at",
    "model_recommendation",
    "override_recommendation",
    "rationale",
    "owner",
    "effective_date",
    "expiration_date",
    "status",
]


def _date_string(value) -> str:
    return str(pd.Timestamp(value).date())


def load_overrides(path: Path | str = OVERRIDE_PATH) -> pd.DataFrame:
    """Load scorecard overrides, returning an empty typed frame if absent."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=OVERRIDE_COLUMNS)
    try:
        overrides = pd.read_csv(p)
    except Exception:
        return pd.DataFrame(columns=OVERRIDE_COLUMNS)
    for col in OVERRIDE_COLUMNS:
        if col not in overrides.columns:
            overrides[col] = pd.NA
    return overrides[OVERRIDE_COLUMNS]


def save_overrides(overrides: pd.DataFrame, path: Path | str = OVERRIDE_PATH) -> None:
    """Persist overrides to CSV."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    output = overrides.copy()
    for col in OVERRIDE_COLUMNS:
        if col not in output.columns:
            output[col] = pd.NA
    output[OVERRIDE_COLUMNS].to_csv(p, index=False)


def save_override(
    *,
    model_recommendation: str,
    override_recommendation: str,
    rationale: str,
    owner: str,
    effective_date: date | str,
    expiration_date: date | str,
    path: Path | str = OVERRIDE_PATH,
    created_at: datetime | None = None,
    override_id: str | None = None,
) -> dict:
    """Append a new active override and supersede prior active overrides."""
    if override_recommendation not in RECOMMENDATIONS:
        return {"available": False, "reason": f"Unknown override recommendation: {override_recommendation}"}
    if not str(rationale).strip():
        return {"available": False, "reason": "Override rationale is required"}
    if not str(owner).strip():
        return {"available": False, "reason": "Override owner is required"}

    eff = pd.Timestamp(effective_date).normalize()
    exp = pd.Timestamp(expiration_date).normalize()
    if exp < eff:
        return {"available": False, "reason": "Expiration date must be on or after effective date"}

    overrides = load_overrides(path)
    if not overrides.empty:
        overrides.loc[overrides["status"] == "active", "status"] = "superseded"

    row = {
        "override_id": override_id or uuid4().hex,
        "created_at": (created_at or datetime.now()).isoformat(timespec="seconds"),
        "model_recommendation": model_recommendation,
        "override_recommendation": override_recommendation,
        "rationale": str(rationale).strip(),
        "owner": str(owner).strip(),
        "effective_date": _date_string(eff),
        "expiration_date": _date_string(exp),
        "status": "active",
    }
    output = pd.concat([overrides, pd.DataFrame([row])], ignore_index=True)
    save_overrides(output, path)
    return {"available": True, "override": row, "path": str(path), "row_count": int(len(output))}


def get_active_override(
    *,
    as_of: date | str | None = None,
    path: Path | str = OVERRIDE_PATH,
) -> dict:
    """Return the active override for as_of, or unavailable metadata."""
    overrides = load_overrides(path)
    if overrides.empty:
        return {"available": False, "reason": "No overrides recorded", "override": None}

    ref_date = pd.Timestamp(as_of or date.today()).normalize()
    work = overrides.copy()
    work["effective_dt"] = pd.to_datetime(work["effective_date"], errors="coerce")
    work["expiration_dt"] = pd.to_datetime(work["expiration_date"], errors="coerce")
    active = work[
        (work["status"] == "active")
        & (work["effective_dt"] <= ref_date)
        & (work["expiration_dt"] >= ref_date)
    ].sort_values(["effective_dt", "created_at"])
    if active.empty:
        return {"available": False, "reason": "No active override", "override": None}

    row = active.iloc[-1].drop(labels=["effective_dt", "expiration_dt"]).to_dict()
    row["days_to_expiration"] = int((pd.Timestamp(row["expiration_date"]) - ref_date).days)
    return {"available": True, "override": row}


def expire_override(
    override_id: str,
    *,
    path: Path | str = OVERRIDE_PATH,
) -> dict:
    """Mark a stored override as expired."""
    overrides = load_overrides(path)
    if overrides.empty or override_id not in set(overrides["override_id"].astype(str)):
        return {"available": False, "reason": "Override not found"}
    overrides.loc[overrides["override_id"].astype(str) == str(override_id), "status"] = "expired"
    save_overrides(overrides, path)
    return {"available": True, "override_id": override_id}


def override_status_summary(
    *,
    as_of: date | str | None = None,
    path: Path | str = OVERRIDE_PATH,
    expiring_days: int = 7,
) -> dict:
    """Summarize active, expired, and expiring override status."""
    overrides = load_overrides(path)
    ref_date = pd.Timestamp(as_of or date.today()).normalize()
    if overrides.empty:
        return {
            "available": True,
            "active_count": 0,
            "expired_count": 0,
            "expiring_soon_count": 0,
            "table": overrides,
        }

    work = overrides.copy()
    work["expiration_dt"] = pd.to_datetime(work["expiration_date"], errors="coerce")
    work["effective_dt"] = pd.to_datetime(work["effective_date"], errors="coerce")
    active_mask = (
        (work["status"] == "active")
        & (work["effective_dt"] <= ref_date)
        & (work["expiration_dt"] >= ref_date)
    )
    expired_mask = (work["status"] == "expired") | (work["expiration_dt"] < ref_date)
    expiring_mask = active_mask & (work["expiration_dt"] <= ref_date + pd.Timedelta(days=expiring_days))
    return {
        "available": True,
        "active_count": int(active_mask.sum()),
        "expired_count": int(expired_mask.sum()),
        "expiring_soon_count": int(expiring_mask.sum()),
        "table": work.drop(columns=["effective_dt", "expiration_dt"]),
    }
