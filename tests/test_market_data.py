import pandas as pd

from src.market_data import _FFILL_LIMITS, _to_daily


def test_monthly_unemployment_fill_bridges_release_lag_business_days():
    bdays = pd.bdate_range("2026-04-01", "2026-06-02")
    series = pd.Series([4.3], index=[pd.Timestamp("2026-04-01")])

    daily = _to_daily(series, ffill_limit=_FFILL_LIMITS["UNRATE"], bday_index=bdays)

    assert daily.loc[pd.Timestamp("2026-06-02")] == 4.3
