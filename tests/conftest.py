import pytest
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/scored_macro_credit_data.csv")

# Columns produced by feature_engine.build_features() — everything before scoring.
# These are the exact columns run_scoring_pipeline() expects as input.
_FEATURE_COLS = [
    "value_10y", "value_2y", "spread", "yield_curve_regime",
    "unemployment", "nfci", "nfci_90d_avg", "hy_spread", "sp500", "vix", "breakeven_10y",
    "unemployment_change_90d", "spread_change_90d", "spread_change_5d",
    "hy_change_30d", "hy_change_90d", "hy_change_5d", "hy_change_30d_prior", "credit_impulse",
    "nfci_change_90d", "vix_change_30d", "vix_change_5d",
    "sp500_return_5d", "sp500_return_30d",
    "unemployment_12m_low", "sahm_like", "sp500_peak", "sp500_drawdown",
    "real_yield_proxy", "real_yield_change_90d", "curve_steepening_velocity_90d", "real_yield_z",
    "credit_regime", "labor_warning",
    "credit_equity_divergence", "vol_credit_mismatch",
    "cross_asset_divergence_score", "market_internals_score", "shock_flag",
]


@pytest.fixture(scope="session")
def featured_df():
    """
    Feature-engine output reconstructed from the pre-computed CSV.
    Contains only the columns that precede scoring so it can be passed
    directly to run_scoring_pipeline() without touching the FRED API.
    """
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df[_FEATURE_COLS].copy()


@pytest.fixture(scope="session")
def pipeline_df(featured_df):
    """featured_df after a full run of run_scoring_pipeline()."""
    from src.pipeline import run_scoring_pipeline
    return run_scoring_pipeline(featured_df)


@pytest.fixture(scope="session")
def analytics_result(pipeline_df):
    """
    Tuple of (df, health_check, backtest_summary) from run_analytics().
    Session-scoped so the backtest runs once for the whole test session.
    """
    from src.pipeline import run_analytics
    return run_analytics(pipeline_df)
