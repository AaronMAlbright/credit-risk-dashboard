from datetime import datetime

import pandas as pd

from src.credit_compensation_overrides import (
    OVERRIDE_COLUMNS,
    expire_override,
    get_active_override,
    load_overrides,
    override_status_summary,
    save_override,
)


def test_load_overrides_missing_path_returns_typed_empty_frame(tmp_path):
    overrides = load_overrides(tmp_path / "missing.csv")

    assert overrides.empty
    assert list(overrides.columns) == OVERRIDE_COLUMNS


def test_save_override_persists_active_override(tmp_path):
    path = tmp_path / "overrides.csv"
    result = save_override(
        model_recommendation="Upgrade Quality",
        override_recommendation="Hold",
        rationale="Mandate exposure is already underweight credit.",
        owner="IC",
        effective_date="2026-06-02",
        expiration_date="2026-06-30",
        path=path,
        created_at=datetime(2026, 6, 2, 12, 0, 0),
        override_id="ovr-1",
    )

    active = get_active_override(as_of="2026-06-10", path=path)

    assert result["available"] is True
    assert active["available"] is True
    assert active["override"]["override_recommendation"] == "Hold"
    assert active["override"]["days_to_expiration"] == 20


def test_active_override_unavailable_after_expiration(tmp_path):
    path = tmp_path / "overrides.csv"
    save_override(
        model_recommendation="Upgrade Quality",
        override_recommendation="Hold",
        rationale="Temporary liquidity constraint.",
        owner="PM",
        effective_date="2026-06-02",
        expiration_date="2026-06-03",
        path=path,
        override_id="ovr-1",
    )

    active = get_active_override(as_of="2026-06-04", path=path)

    assert active["available"] is False
    assert active["reason"] == "No active override"


def test_save_override_supersedes_prior_active_override(tmp_path):
    path = tmp_path / "overrides.csv"
    save_override(
        model_recommendation="Upgrade Quality",
        override_recommendation="Hold",
        rationale="Initial override.",
        owner="PM",
        effective_date="2026-06-02",
        expiration_date="2026-06-30",
        path=path,
        override_id="ovr-1",
    )
    save_override(
        model_recommendation="Upgrade Quality",
        override_recommendation="Hedge",
        rationale="Liquidity deteriorated.",
        owner="IC",
        effective_date="2026-06-10",
        expiration_date="2026-07-10",
        path=path,
        override_id="ovr-2",
    )

    overrides = load_overrides(path)
    active = get_active_override(as_of="2026-06-15", path=path)

    assert list(overrides["status"]) == ["superseded", "active"]
    assert active["override"]["override_id"] == "ovr-2"


def test_expire_override_marks_override_expired(tmp_path):
    path = tmp_path / "overrides.csv"
    save_override(
        model_recommendation="Upgrade Quality",
        override_recommendation="Hold",
        rationale="Temporary override.",
        owner="PM",
        effective_date="2026-06-02",
        expiration_date="2026-06-30",
        path=path,
        override_id="ovr-1",
    )

    result = expire_override("ovr-1", path=path)
    active = get_active_override(as_of="2026-06-10", path=path)

    assert result["available"] is True
    assert active["available"] is False


def test_override_status_summary_counts_active_expiring_and_expired(tmp_path):
    path = tmp_path / "overrides.csv"
    pd.DataFrame([
        {
            "override_id": "active",
            "created_at": "2026-06-02T12:00:00",
            "model_recommendation": "Upgrade Quality",
            "override_recommendation": "Hold",
            "rationale": "Active.",
            "owner": "PM",
            "effective_date": "2026-06-01",
            "expiration_date": "2026-06-05",
            "status": "active",
        },
        {
            "override_id": "expired",
            "created_at": "2026-05-01T12:00:00",
            "model_recommendation": "Add",
            "override_recommendation": "Hold",
            "rationale": "Old.",
            "owner": "IC",
            "effective_date": "2026-05-01",
            "expiration_date": "2026-05-05",
            "status": "active",
        },
    ]).to_csv(path, index=False)

    summary = override_status_summary(as_of="2026-06-02", path=path, expiring_days=7)

    assert summary["active_count"] == 1
    assert summary["expired_count"] == 1
    assert summary["expiring_soon_count"] == 1


def test_save_override_rejects_missing_rationale(tmp_path):
    result = save_override(
        model_recommendation="Upgrade Quality",
        override_recommendation="Hold",
        rationale="",
        owner="PM",
        effective_date="2026-06-02",
        expiration_date="2026-06-30",
        path=tmp_path / "overrides.csv",
    )

    assert result["available"] is False
    assert "rationale" in result["reason"].lower()
