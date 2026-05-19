"""
Observable proxies for credit fundamentals and market technicals.

The dashboard does not have issuer-level financial statements, dealer runs, or
live mutual-fund-flow feeds. These proxies keep those channels explicit while
grounding each score in available market and macro data.
"""

from __future__ import annotations

import pandas as pd


def _empty(index: pd.Index) -> pd.Series:
    return pd.Series(pd.NA, index=index, dtype="Float64")


def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return _empty(df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _rolling_percentile(series: pd.Series, window: int = 756, min_periods: int = 126) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")

    def pct_rank(x: pd.Series) -> float:
        x = x.dropna()
        if x.empty:
            return float("nan")
        return float(x.rank(pct=True).iloc[-1] * 100)

    return values.rolling(window=window, min_periods=min_periods).apply(pct_rank, raw=False)


def _mean_present(parts: list[pd.Series]) -> pd.Series:
    present = [part for part in parts if part.notna().any()]
    if not present:
        return _empty(parts[0].index if parts else pd.Index([]))
    return pd.concat(present, axis=1).mean(axis=1).clip(0, 100)


def add_credit_feature_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interview-ready proxy scores for missing credit channels.

    Scores are 0-100 where higher means greater risk. Existing columns are
    preserved, so callers can override these proxies with richer data later.
    """
    out = df.copy()
    index = out.index

    hy = _safe_numeric(out, "hy_spread")
    ig = _safe_numeric(out, "ig_spread")
    bbb = _safe_numeric(out, "bbb_spread")
    vix = _safe_numeric(out, "vix")
    sp500_dd = _safe_numeric(out, "sp500_drawdown")
    unemployment = _safe_numeric(out, "unemployment")
    sloos = _safe_numeric(out, "sloos_ci")
    hy_total_return = _safe_numeric(out, "hy_total_return_daily")
    ig_total_return = _safe_numeric(out, "ig_total_return_daily")
    loan_growth = _safe_numeric(out, "loan_growth_90d")

    hy_pct = _rolling_percentile(hy)
    ig_pct = _rolling_percentile(ig)
    bbb_pct = _rolling_percentile(bbb)
    vix_pct = _rolling_percentile(vix)
    hy_ig_ratio = hy / ig.replace(0, pd.NA)
    hy_ig_pct = _rolling_percentile(hy_ig_ratio)
    bbb_ig_ratio = bbb / ig.replace(0, pd.NA)
    bbb_ig_pct = _rolling_percentile(bbb_ig_ratio)

    if "default_cycle_score" not in out.columns:
        out["default_cycle_score"] = _mean_present(
            [
                hy_pct,
                _safe_numeric(out, "credit_market_risk_score_smooth"),
                _safe_numeric(out, "sahm_like").clip(lower=0).mul(100).clip(0, 100),
            ]
        )

    if "corporate_leverage_score" not in out.columns:
        out["corporate_leverage_score"] = _mean_present([hy_ig_pct, bbb_ig_pct, hy_pct])

    if "corporate_profit_cycle_score" not in out.columns:
        drawdown_score = (-sp500_dd * 500).clip(0, 100)
        unemployment_pct = _rolling_percentile(unemployment)
        out["corporate_profit_cycle_score"] = _mean_present([drawdown_score, unemployment_pct])

    if "sloos_stress_score" not in out.columns:
        out["sloos_stress_score"] = _mean_present([_rolling_percentile(sloos), _safe_numeric(out, "liquidity_regime_score_smooth")])

    if "cds_implied_pd_score" not in out.columns:
        out["cds_implied_pd_score"] = _mean_present([hy_pct, _safe_numeric(out, "expected_loss_bps").div(12).clip(0, 100)])

    if "primary_market_score" not in out.columns:
        concession_proxy = ((hy - hy.rolling(21, min_periods=10).mean()) * 250).clip(0, 100)
        out["primary_market_score"] = _mean_present([concession_proxy, hy_pct])

    if "etf_fund_flow_score" not in out.columns:
        hy_flow_proxy = (-hy_total_return.rolling(21, min_periods=10).sum() * 500).clip(0, 100)
        ig_flow_proxy = (-ig_total_return.rolling(21, min_periods=10).sum() * 500).clip(0, 100)
        out["etf_fund_flow_score"] = _mean_present([hy_flow_proxy, ig_flow_proxy, vix_pct])

    if "etf_dislocation_score" not in out.columns:
        hy_vol = hy.diff().rolling(21, min_periods=10).std()
        out["etf_dislocation_score"] = _mean_present([_rolling_percentile(hy_vol, min_periods=63), vix_pct])

    if "loan_market_score" not in out.columns:
        loan_contraction_score = (-loan_growth * 1000).clip(0, 100)
        out["loan_market_score"] = _mean_present([loan_contraction_score, _rolling_percentile(sloos), hy_pct])

    if "clo_stress_score" not in out.columns:
        out["clo_stress_score"] = _mean_present([hy_ig_pct, bbb_ig_pct, hy_pct])

    return out
