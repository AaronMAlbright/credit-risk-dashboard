import pandas as pd

from src.credit_tearsheet import credit_market_tearsheet, credit_tearsheet_markdown
from src.rating_bucket_proxy import rating_bucket_proxy_table, rating_bucket_summary
from src.refinancing_wall import refinancing_wall_summary, refinancing_wall_table


def _df() -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=100, freq="B")
    return pd.DataFrame(
        {
            "hy_spread": [3.0 + i * 0.01 for i in range(100)],
            "ig_spread": [1.1 + i * 0.002 for i in range(100)],
            "bbb_spread": [1.5 + i * 0.003 for i in range(100)],
            "final_decision": ["Neutral"] * 50 + ["Caution"] * 50,
            "debt_due_1y": [10.0] * 100,
            "debt_due_1_3y": [30.0] * 100,
            "debt_due_3_5y": [40.0] * 100,
            "debt_due_5y_plus": [20.0] * 100,
        },
        index=idx,
    )


def test_credit_market_tearsheet():
    ts = credit_market_tearsheet(_df())
    assert not ts.empty
    assert {"metric", "level", "percentile", "change_1m", "action"}.issubset(ts.columns)
    assert "HY OAS" in set(ts["metric"])


def test_credit_tearsheet_markdown():
    md = credit_tearsheet_markdown(_df())
    assert "# Credit Market Tear Sheet" in md
    assert "HY OAS" in md


def test_rating_bucket_proxy():
    table = rating_bucket_proxy_table(_df())
    assert not table.empty
    assert {"bucket", "proxy", "level", "regime", "interpretation"}.issubset(table.columns)
    assert rating_bucket_summary(_df())


def test_refinancing_wall_with_data():
    table = refinancing_wall_table(_df())
    assert not table.empty
    assert table["share"].sum() == 1.0
    assert "Near-term refinancing wall" in refinancing_wall_summary(_df())


def test_refinancing_wall_placeholder():
    table = refinancing_wall_table(pd.DataFrame({"hy_spread": [3.0]}))
    assert table["amount"].isna().all()
    assert "requires issuer/index maturity data" in refinancing_wall_summary(pd.DataFrame({"hy_spread": [3.0]}))

