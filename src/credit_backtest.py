"""
Credit-native portfolio backtest.

Computes daily total returns for the credit portfolio allocation using
carry + duration-adjusted price return:
  HY total return ≈ (hy_yield_prev / 252) - HY_DURATION × d(hy_spread) / 100
  IG total return ≈ (ig_yield_prev / 252) - IG_DURATION × d(ig_spread) / 100

Credit portfolio return = hy_weight_lagged × hy_return + ig_weight_lagged × ig_return

Regime-conditional analysis answers the core credit question:
  Does Risk-Off correctly anticipate HY spread widening?
  Does Risk-On correctly position for HY tightening / carry capture?

Public API
----------
  HY_DURATION            — assumed modified duration for HY index
  IG_DURATION            — assumed modified duration for IG index
  ONE_WAY_COST           — one-way transaction cost assumption
  compute_credit_returns — compute daily HY/IG/blended credit returns
  regime_credit_stats    — per-regime credit spread and return summary
  run_credit_backtest    — orchestrator
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HY_DURATION:  float = 4.0    # years — typical HY index modified duration
IG_DURATION:  float = 7.0    # years — typical IG investment-grade index
ONE_WAY_COST: float = 0.001  # 10 bps per one-way turn

_ANNUALIZE  = 252
_MIN_OBS    = 20
_OOS_START  = "2016-01-01"

_DATE_COL    = "date"
_REGIME_COL  = "final_decision"
_HY_RET_COL  = "hy_total_return_daily"
_IG_RET_COL  = "ig_total_return_daily"
_HY_W_COL    = "hy_weight"
_IG_W_COL    = "ig_weight"
_HY_SPR_COL  = "hy_spread"
_IG_SPR_COL  = "ig_spread"
_HY_FWD_COL  = "hy_forward_30d_change"


# ---------------------------------------------------------------------------
# Core return computation
# ---------------------------------------------------------------------------

def compute_credit_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily HY, IG, and blended credit portfolio returns.

    Requires columns: hy_weight, ig_weight (regime-based allocations),
    and either pre-computed hy_total_return_daily / ig_total_return_daily
    (from app.py) or fallback raw spread + yield columns.

    Returns DataFrame with:
      date, hy_return, ig_return, credit_return (blended),
      net_credit_return (after turnover costs),
      cum_hy, cum_ig, cum_credit, cum_net_credit,
      hy_weight_used, ig_weight_used, turnover
    """
    if df.empty:
        return pd.DataFrame()

    n = len(df)

    # HY daily return
    if _HY_RET_COL in df.columns:
        hy_ret = df[_HY_RET_COL].fillna(0).values.astype(float)
    else:
        hy_ret = np.zeros(n)

    # IG daily return
    if _IG_RET_COL in df.columns:
        ig_ret = df[_IG_RET_COL].fillna(0).values.astype(float)
    else:
        ig_ret = np.zeros(n)

    # Regime-based weights (lag by 1 day to avoid look-ahead)
    hy_w = df[_HY_W_COL].fillna(0).values.astype(float) if _HY_W_COL in df.columns else np.zeros(n)
    ig_w = df[_IG_W_COL].fillna(0).values.astype(float) if _IG_W_COL in df.columns else np.zeros(n)

    hy_w_lag = np.concatenate([[hy_w[0]], hy_w[:-1]])
    ig_w_lag = np.concatenate([[ig_w[0]], ig_w[:-1]])

    credit_ret = hy_w_lag * hy_ret + ig_w_lag * ig_ret

    # Turnover cost
    turnover = np.abs(np.diff(hy_w, prepend=hy_w[0])) + np.abs(np.diff(ig_w, prepend=ig_w[0]))
    cost = np.concatenate([[0.0], turnover[:-1]]) * ONE_WAY_COST
    net_ret = credit_ret - cost

    out = pd.DataFrame()
    if _DATE_COL in df.columns:
        out["date"] = df[_DATE_COL].values
    out["hy_return"]         = hy_ret
    out["ig_return"]         = ig_ret
    out["credit_return"]     = credit_ret
    out["net_credit_return"] = net_ret
    out["hy_weight_used"]    = hy_w_lag
    out["ig_weight_used"]    = ig_w_lag
    out["turnover"]          = turnover

    for col, src in [("cum_hy",         hy_ret),
                     ("cum_ig",         ig_ret),
                     ("cum_credit",     credit_ret),
                     ("cum_net_credit", net_ret)]:
        out[col] = (1 + pd.Series(src).fillna(0)).cumprod().values

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-regime credit statistics
# ---------------------------------------------------------------------------

def regime_credit_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-regime summary of credit market conditions and forward returns.

    For each regime (Risk-On, Neutral, Caution, Risk-Off):
      - avg_hy_spread, avg_ig_spread    — level context
      - avg_hy_forward_30d              — avg HY OAS change over next 30 days
      - hit_rate_tightening             — % of rows where HY tightened next 30d
      - avg_hy_return, avg_ig_return    — mean daily returns within regime
      - sharpe_credit                   — annualised Sharpe on blended credit return
      - n_days, pct_time                — regime frequency

    Separates IS (pre-2016) vs OOS (2016+) if date column is present.
    """
    if df.empty or _REGIME_COL not in df.columns:
        return pd.DataFrame()

    rows = []
    regimes = ["Risk-On", "Neutral", "Caution", "Risk-Off"]

    # IS/OOS split
    if _DATE_COL in df.columns:
        dates    = pd.to_datetime(df[_DATE_COL].values)
        oos_mask = dates >= pd.Timestamp(_OOS_START)
    else:
        oos_mask = np.zeros(len(df), dtype=bool)

    for split_label, mask in [("IS", ~oos_mask), ("OOS", oos_mask)]:
        sub = df[mask]
        if len(sub) < _MIN_OBS:
            continue
        for regime in regimes:
            rsub = sub[sub[_REGIME_COL] == regime]
            if len(rsub) < _MIN_OBS:
                continue

            row: dict = {"split": split_label, "regime": regime, "n_days": len(rsub)}
            row["pct_time"] = round(len(rsub) / max(len(sub), 1) * 100, 1)

            if _HY_SPR_COL in rsub.columns:
                row["avg_hy_spread"] = round(float(rsub[_HY_SPR_COL].mean()), 2)
            if _IG_SPR_COL in rsub.columns:
                row["avg_ig_spread"] = round(float(rsub[_IG_SPR_COL].mean()), 2)

            if _HY_FWD_COL in rsub.columns:
                fwd = rsub[_HY_FWD_COL].dropna()
                if len(fwd) >= _MIN_OBS:
                    row["avg_hy_fwd_30d"]        = round(float(fwd.mean()), 3)
                    row["hit_rate_tightening"]   = round(float((fwd < 0).mean()), 3)

            if _HY_RET_COL in rsub.columns:
                r = rsub[_HY_RET_COL].dropna()
                if len(r) >= _MIN_OBS:
                    row["avg_hy_daily_return"] = round(float(r.mean() * _ANNUALIZE), 4)

            if _IG_RET_COL in rsub.columns:
                r = rsub[_IG_RET_COL].dropna()
                if len(r) >= _MIN_OBS:
                    row["avg_ig_daily_return"] = round(float(r.mean() * _ANNUALIZE), 4)

            # Blended credit return Sharpe
            if _HY_W_COL in rsub.columns and _HY_RET_COL in rsub.columns:
                hy_r = rsub[_HY_RET_COL].fillna(0).values
                ig_r = rsub[_IG_RET_COL].fillna(0).values if _IG_RET_COL in rsub.columns else np.zeros(len(rsub))
                hw   = rsub[_HY_W_COL].fillna(0).values
                iw   = rsub[_IG_W_COL].fillna(0).values if _IG_W_COL in rsub.columns else np.zeros(len(rsub))
                cr   = hw * hy_r + iw * ig_r
                cr   = cr[~np.isnan(cr)]
                if len(cr) >= _MIN_OBS and cr.std(ddof=1) > 1e-9:
                    row["credit_sharpe"] = round(
                        float(cr.mean() * _ANNUALIZE / (cr.std(ddof=1) * np.sqrt(_ANNUALIZE))), 3
                    )

            rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Spread-conditioned forward return table
# ---------------------------------------------------------------------------

def hy_spread_bucket_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket HY spread levels and show forward 30d HY change in each bucket.
    Answers: 'At what spread level does HY become attractive to add?'

    Returns DataFrame with: spread_bucket, n, avg_fwd_30d, hit_rate_tight, pct_risk_off
    """
    if _HY_SPR_COL not in df.columns or _HY_FWD_COL not in df.columns:
        return pd.DataFrame()

    sub = df[[_HY_SPR_COL, _HY_FWD_COL, _REGIME_COL]].dropna(subset=[_HY_SPR_COL, _HY_FWD_COL])
    if len(sub) < _MIN_OBS:
        return pd.DataFrame()

    bins   = [0, 3.0, 4.0, 5.0, 6.5, 8.0, 100]
    labels = ["<3%", "3-4%", "4-5%", "5-6.5%", "6.5-8%", ">8%"]
    sub = sub.copy()
    sub["spread_bucket"] = pd.cut(sub[_HY_SPR_COL], bins=bins, labels=labels, right=False)

    rows = []
    for bucket in labels:
        b = sub[sub["spread_bucket"] == bucket]
        if len(b) < 5:
            continue
        fwd = b[_HY_FWD_COL].dropna()
        rows.append({
            "spread_bucket":       bucket,
            "n":                   len(b),
            "avg_fwd_30d":         round(float(fwd.mean()), 3) if len(fwd) >= 5 else None,
            "hit_rate_tightening": round(float((fwd < 0).mean()), 3) if len(fwd) >= 5 else None,
            "pct_risk_off":        round(float((b[_REGIME_COL] == "Risk-Off").mean() * 100), 1)
                                   if _REGIME_COL in b.columns else None,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Excess return decomposition
# ---------------------------------------------------------------------------

def decompose_excess_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose portfolio credit return into carry and spread P&L components.

    Carry   = hy_w × (hy_yield_prev / 252) + ig_w × (ig_yield_prev / 252)
    Spread  = hy_w × (−HY_DUR × Δhy_spread / 100) + ig_w × (−IG_DUR × Δig_spread / 100)
    Total   ≈ Carry + Spread P&L

    Returns DataFrame: date, carry_daily, spread_pnl_daily, cum_carry, cum_spread_pnl
    """
    if df.empty:
        return pd.DataFrame()

    n = len(df)
    hy_w = df[_HY_W_COL].fillna(0).values.astype(float) if _HY_W_COL in df.columns else np.zeros(n)
    ig_w = df[_IG_W_COL].fillna(0).values.astype(float) if _IG_W_COL in df.columns else np.zeros(n)
    hy_w_lag = np.concatenate([[hy_w[0]], hy_w[:-1]])
    ig_w_lag = np.concatenate([[ig_w[0]], ig_w[:-1]])

    # Carry: all-in yield / 252, weight lagged 1 day
    hy_carry = np.zeros(n)
    ig_carry = np.zeros(n)
    if "hy_yield" in df.columns:
        hy_y = df["hy_yield"].ffill().values.astype(float)
        hy_carry = hy_w_lag * np.concatenate([[hy_y[0]], hy_y[:-1]]) / 100.0 / _ANNUALIZE
    if "ig_yield" in df.columns:
        ig_y = df["ig_yield"].ffill().values.astype(float)
        ig_carry = ig_w_lag * np.concatenate([[ig_y[0]], ig_y[:-1]]) / 100.0 / _ANNUALIZE
    port_carry = hy_carry + ig_carry

    # Spread P&L: −duration × Δspread / 100
    hy_spnl = np.zeros(n)
    ig_spnl = np.zeros(n)
    if "hy_spread" in df.columns:
        d_hy = pd.Series(df["hy_spread"].ffill().values).diff().fillna(0).values.astype(float)
        hy_spnl = hy_w_lag * (-HY_DURATION * d_hy / 100.0)
    if "ig_spread" in df.columns:
        d_ig = pd.Series(df["ig_spread"].ffill().values).diff().fillna(0).values.astype(float)
        ig_spnl = ig_w_lag * (-IG_DURATION * d_ig / 100.0)
    port_spnl = hy_spnl + ig_spnl

    out = pd.DataFrame()
    if _DATE_COL in df.columns:
        out["date"] = df[_DATE_COL].values
    out["carry_daily"]      = port_carry
    out["spread_pnl_daily"] = port_spnl
    out["cum_carry"]        = (1 + pd.Series(port_carry).fillna(0)).cumprod().values
    out["cum_spread_pnl"]   = (1 + pd.Series(port_spnl).fillna(0)).cumprod().values
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_credit_backtest(df: pd.DataFrame) -> dict:
    """
    Full credit-native backtest.

    Returns dict:
      returns_df     — daily credit return DataFrame
      regime_stats   — per-regime credit stats (IS + OOS)
      spread_buckets — HY spread level → forward return table
      ann_turnover   — annualised credit portfolio turnover
      has_hy_data    — bool: True if HY return series available
      has_ig_data    — bool: True if IG return series available
    """
    if df.empty:
        return {}

    has_hy = _HY_RET_COL in df.columns and df[_HY_RET_COL].notna().sum() > _MIN_OBS
    has_ig = _IG_RET_COL in df.columns and df[_IG_RET_COL].notna().sum() > _MIN_OBS

    returns_df    = compute_credit_returns(df)
    regime_stats  = regime_credit_stats(df)
    spread_bkts   = hy_spread_bucket_returns(df)
    decomposition = decompose_excess_returns(df)

    ann_turnover = None
    if not returns_df.empty and "turnover" in returns_df.columns:
        ann_turnover = round(float(returns_df["turnover"].mean() * _ANNUALIZE), 2)

    return {
        "returns_df":     returns_df,
        "regime_stats":   regime_stats,
        "spread_buckets": spread_bkts,
        "decomposition":  decomposition,
        "ann_turnover":   ann_turnover,
        "has_hy_data":    has_hy,
        "has_ig_data":    has_ig,
    }
