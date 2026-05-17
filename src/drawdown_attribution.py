"""
Drawdown attribution — which sizing component drove each strategy drawdown.
Public API: run_drawdown_attribution(bt_df) -> dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_COMPONENTS = {
    "score":      "score_weight",
    "regime":     "regime_weight",
    "vol_target": "vol_target_weight",
    "momentum":   "momentum_weight",
}

_DD_THRESHOLD = -0.05   # 5 % drawdown floor
_MIN_DURATION = 10       # calendar days


def _identify_drawdown_periods(bt_df: pd.DataFrame) -> list[tuple[int, int]]:
    """
    Return a list of (start_idx, end_idx) index pairs for drawdown periods
    where ``strategy_drawdown < _DD_THRESHOLD`` for at least ``_MIN_DURATION``
    consecutive rows.
    """
    if "strategy_drawdown" not in bt_df.columns:
        return []

    in_dd = bt_df["strategy_drawdown"] < _DD_THRESHOLD
    periods: list[tuple[int, int]] = []
    start: int | None = None

    for i, flag in enumerate(in_dd):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            end = i - 1
            if end - start + 1 >= _MIN_DURATION:
                periods.append((start, end))
            start = None

    # Close an open period that reaches the end of the frame
    if start is not None:
        end = len(bt_df) - 1
        if end - start + 1 >= _MIN_DURATION:
            periods.append((start, end))

    return periods


def _cumulative_return(returns: pd.Series) -> float:
    """Compound cumulative return from a series of daily returns."""
    return float((1 + returns).prod() - 1)


def run_drawdown_attribution(bt_df: pd.DataFrame) -> dict:
    """
    Attribute strategy drawdowns to individual sizing components.

    Parameters
    ----------
    bt_df : pd.DataFrame
        Output of ``build_strategy_backtest()``.  Expected columns:
        ``date``, ``strategy_drawdown``, ``strategy_daily_return``,
        ``sp500_daily_return``, ``score_weight``, ``regime_weight``,
        ``vol_target_weight``, ``momentum_weight``.

    Returns
    -------
    dict with keys:
      ``drawdowns`` — list of dicts (up to 5 largest drawdowns), each with
                      ``start``, ``end``, ``duration_days``, ``strategy_return``,
                      ``sp500_return``, ``component_returns``.
      ``avg_component_drag`` — mean component return across all identified drawdowns.
    """
    required = {"strategy_drawdown", "strategy_daily_return", "sp500_daily_return"}
    required |= set(_COMPONENTS.values())
    missing = required - set(bt_df.columns)
    if missing:
        raise ValueError(f"bt_df is missing required columns: {missing}")

    bt_df = bt_df.reset_index(drop=True)

    # Ensure date column is parseable
    if "date" in bt_df.columns:
        bt_df["date"] = pd.to_datetime(bt_df["date"])

    periods = _identify_drawdown_periods(bt_df)

    if not periods:
        return {"drawdowns": [], "avg_component_drag": {}}

    # Sort by worst trough drawdown, keep top 5
    def _trough(start: int, end: int) -> float:
        return float(bt_df.loc[start:end, "strategy_drawdown"].min())

    periods_sorted = sorted(periods, key=lambda p: _trough(*p))[:5]

    drawdowns: list[dict] = []
    for start, end in periods_sorted:
        slice_df = bt_df.loc[start:end]

        strategy_ret = _cumulative_return(slice_df["strategy_daily_return"])
        sp500_ret = _cumulative_return(slice_df["sp500_daily_return"])

        component_returns: dict[str, float] = {}
        for label, col in _COMPONENTS.items():
            component_daily = slice_df[col] * slice_df["sp500_daily_return"]
            component_returns[label] = _cumulative_return(component_daily)

        if "date" in bt_df.columns:
            start_str = str(slice_df["date"].iloc[0].date())
            end_str = str(slice_df["date"].iloc[-1].date())
        else:
            start_str = str(start)
            end_str = str(end)

        duration_days = end - start + 1

        drawdowns.append(
            {
                "start": start_str,
                "end": end_str,
                "duration_days": duration_days,
                "strategy_return": strategy_ret,
                "sp500_return": sp500_ret,
                "component_returns": component_returns,
            }
        )

    # Average component drag across all drawdowns
    avg_component_drag: dict[str, float] = {}
    if drawdowns:
        for label in _COMPONENTS:
            values = [d["component_returns"][label] for d in drawdowns]
            avg_component_drag[label] = float(np.mean(values))

    return {"drawdowns": drawdowns, "avg_component_drag": avg_component_drag}
