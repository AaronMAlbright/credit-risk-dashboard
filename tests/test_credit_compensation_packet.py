import io
import zipfile

import pandas as pd

from src.credit_compensation_packet import build_pm_review_packet


def _packet_df() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=160, freq="B")
    hy = []
    pct = []
    score = []
    sloos = []
    charge = []
    for i in range(160):
        block = i // 40
        if block == 0:
            hy.append(6.0 - i * 0.01)
            pct.append(80)
            score.append(35)
            sloos.append(-3)
            charge.append(0.0)
        elif block == 1:
            hy.append(3.0 + (i - 40) * 0.005)
            pct.append(5)
            score.append(35)
            sloos.append(-1)
            charge.append(0.0)
        elif block == 2:
            hy.append(4.0 + (i - 80) * 0.02)
            pct.append(45)
            score.append(35)
            sloos.append(8)
            charge.append(0.0)
        else:
            hy.append(5.6 + (i - 120) * 0.03)
            pct.append(15)
            score.append(80)
            sloos.append(8)
            charge.append(0.1)

    return pd.DataFrame(
        {
            "hy_spread": hy,
            "ig_spread_bps": [120 + i * 0.1 for i in range(160)],
            "hy_spread_percentile": pct,
            "final_decision": ["Neutral"] * 160,
            "composite_risk_score_smooth": score,
            "sloos_change_90d": sloos,
            "chargeoff_change_90d": charge,
            "delinquency_change_90d": [0.0] * 160,
        },
        index=idx,
    )


def test_build_pm_review_packet_exports_markdown_and_zip_bundle():
    result = build_pm_review_packet(_packet_df())

    assert result["available"] is True
    assert "# Credit Compensation PM Review Packet" in result["markdown"]
    assert "PM Attribution: What Changed" in result["markdown"]
    assert "Active Override / IC Memo" in result["markdown"]
    assert "CDX Hedge Sizing" in result["markdown"]
    assert "Audit Rules" in result["markdown"]
    assert result["zip"]
    assert "override_history" in result["tables"]

    with zipfile.ZipFile(io.BytesIO(result["zip"])) as zf:
        names = set(zf.namelist())
        assert "pm_review_packet.md" in names
        assert "rating_bucket_allocation.csv" in names
        assert "pm_attribution.csv" in names
        assert "net_spread_beta.csv" in names
        assert "cdx_hedge_sizing.csv" in names
        assert "audit_rules.csv" in names
        assert "recommendation" in zf.read("pm_review_packet.md").decode("utf-8")


def test_build_pm_review_packet_unavailable_without_spreads():
    result = build_pm_review_packet(pd.DataFrame(index=pd.date_range("2024-01-01", periods=3)))

    assert result["available"] is False
