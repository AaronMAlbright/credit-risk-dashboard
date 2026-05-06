import pandas as pd

from src.treasury_engine import compute_treasury_features
from src.risk_engine import (
    classify_yield_curve_regime,
    classify_credit_regime,
    classify_labor_warning,
    classify_credit_equity_divergence,
    classify_vol_credit_mismatch,
    detect_shock,
)
from src.market_internals_engine import (
    compute_cross_asset_divergence_score,
    compute_market_internals_score,
)


def build_raw_dataset(series_dict: dict) -> pd.DataFrame:
    """
    Join raw FRED series into a single aligned DataFrame.

    Expected keys: "DGS10", "DGS2", "UNRATE", "NFCI",
                   "BAMLH0A0HYM2", "SP500", "VIXCLS", "T10YIE"
    """
    ten_year = series_dict["DGS10"]
    two_year = series_dict["DGS2"]
    unrate = series_dict["UNRATE"]
    nfci = series_dict["NFCI"]
    hy_spread = series_dict["BAMLH0A0HYM2"]
    sp500 = series_dict["SP500"]
    vix = series_dict["VIXCLS"]
    breakeven_10y = series_dict["T10YIE"]

    df = ten_year.join(two_year, lsuffix="_10y", rsuffix="_2y")
    df["spread"] = df["value_10y"] - df["value_2y"]
    df["yield_curve_regime"] = df["spread"].apply(classify_yield_curve_regime)

    df = df.join(unrate)
    df.rename(columns={"value": "unemployment"}, inplace=True)
    df["unemployment"] = df["unemployment"].ffill()

    df = df.join(nfci)
    df.rename(columns={"value": "nfci"}, inplace=True)
    df["nfci"] = df["nfci"].ffill()
    df["nfci_90d_avg"] = df["nfci"].rolling(90).mean()

    df = df.join(hy_spread)
    df.rename(columns={"value": "hy_spread"}, inplace=True)
    df["hy_spread"] = df["hy_spread"].ffill()

    df = df.join(sp500)
    df.rename(columns={"value": "sp500"}, inplace=True)
    df["sp500"] = df["sp500"].ffill()

    df = df.join(vix)
    df.rename(columns={"value": "vix"}, inplace=True)
    df["vix"] = df["vix"].ffill()

    df = df.join(breakeven_10y)
    df.rename(columns={"value": "breakeven_10y"}, inplace=True)
    df["breakeven_10y"] = df["breakeven_10y"].ffill()

    df = df.dropna()
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum, rate-of-change, and level-derived features to the raw dataset.
    Calls compute_treasury_features for real-yield and curve-velocity columns.
    """
    df = df.copy()

    df["unemployment_change_90d"] = df["unemployment"].diff(90)
    df["spread_change_90d"] = df["spread"].diff(90)
    df["spread_change_5d"] = df["spread"].diff(5)

    df["hy_change_30d"] = df["hy_spread"].diff(30)
    df["hy_change_90d"] = df["hy_spread"].diff(90)
    df["hy_change_5d"] = df["hy_spread"].diff(5)
    df["hy_change_30d_prior"] = df["hy_change_30d"].shift(30)
    df["credit_impulse"] = df["hy_change_30d"] - df["hy_change_30d_prior"]

    df["nfci_change_90d"] = df["nfci_90d_avg"].diff(90)

    df["vix_change_30d"] = df["vix"].diff(30)
    df["vix_change_5d"] = df["vix"].diff(5)

    df["sp500_return_5d"] = df["sp500"].pct_change(5)
    df["sp500_return_30d"] = df["sp500"].pct_change(30)

    df["unemployment_12m_low"] = df["unemployment"].rolling(252).min()
    df["sahm_like"] = df["unemployment"] - df["unemployment_12m_low"]

    df["sp500_peak"] = df["sp500"].cummax()
    df["sp500_drawdown"] = df["sp500"] / df["sp500_peak"] - 1

    df = compute_treasury_features(df)
    df = df.dropna()
    return df


def add_classification_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply regime classifiers and component scorers that produce columns used
    as inputs to the main scoring pipeline (e.g. credit_equity_divergence feeds
    into compute_credit_market_risk_score; market_internals_score feeds into
    compute_complacency_score).
    """
    df = df.copy()

    df["credit_regime"] = df.apply(
        lambda row: classify_credit_regime(row["spread"], row["unemployment"]),
        axis=1,
    )

    df["labor_warning"] = df.apply(
        lambda row: classify_labor_warning(
            row["sahm_like"],
            row["unemployment_change_90d"],
        ),
        axis=1,
    )

    df["credit_equity_divergence"] = df.apply(
        lambda row: classify_credit_equity_divergence(
            row["sp500_return_30d"],
            row["hy_change_30d"],
        ),
        axis=1,
    )

    df["vol_credit_mismatch"] = df.apply(
        lambda row: classify_vol_credit_mismatch(
            row["vix"],
            row["vix_change_30d"],
            row["hy_change_30d"],
        ),
        axis=1,
    )

    df["cross_asset_divergence_score"] = df.apply(
        lambda row: compute_cross_asset_divergence_score(
            row["sp500_return_30d"],
            row["sp500_drawdown"],
            row["hy_change_30d"],
            row["vix"],
            row["vix_change_30d"],
        ),
        axis=1,
    )

    df["market_internals_score"] = df.apply(
        lambda row: compute_market_internals_score(
            row["sp500_return_30d"],
            row["sp500_return_5d"],
            row["sp500_drawdown"],
            row["vix"],
            row["vix_change_30d"],
        ),
        axis=1,
    )

    df["shock_flag"] = df.apply(
        lambda row: detect_shock(
            row["vix_change_5d"],
            row["hy_change_5d"],
            row["sp500_return_5d"],
            row["spread_change_5d"],
        ),
        axis=1,
    )

    return df


def build_features(series_dict: dict) -> pd.DataFrame:
    """
    Full feature pipeline: raw join → derived features → classification features.
    This is the single entry point pipeline.py calls.
    """
    df = build_raw_dataset(series_dict)
    df = add_derived_features(df)
    df = add_classification_features(df)
    return df
