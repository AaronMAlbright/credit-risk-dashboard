"""
Kelly Criterion for optimal position sizing.

The Kelly Criterion (Kelly 1956) gives the mathematically optimal fraction
of wealth to bet on a favourable bet to maximise long-run geometric growth:

  f* = (p × b - q) / b = (p × b - (1-p)) / b

where:
  p = probability of winning (signal hit rate)
  q = 1 - p = probability of losing
  b = odds ratio = avg_win / avg_loss

For a continuous return distribution (more relevant for portfolios):
  f* = μ / σ²   (Sharpe ratio / σ)
     = E[r] / Var[r]

In practice, use "half-Kelly" (f*/2) to account for estimation error
and reduce drawdown risk. Quarter-Kelly is even more conservative.

This module computes Kelly sizing from the signal's historical hit rate
and avg win/loss, then compares to current strategy weights.

Public API
----------
  compute_kelly_sizing(df, return_col, signal_col) → dict
  run_kelly_analysis(df) → dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OOS_START: str = "2016-01-01"   # out-of-sample split date

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _kelly_stats_from_returns(returns: pd.Series) -> dict:
    """Compute Kelly statistics from a series of daily returns.

    Parameters
    ----------
    returns : pd.Series
        Daily returns (can be strategy or asset returns).

    Returns
    -------
    dict with keys:
        hit_rate, avg_win, avg_loss, odds_ratio,
        kelly_fraction, half_kelly, quarter_kelly,
        continuous_kelly, n_obs
    """
    clean = returns.dropna()
    n = len(clean)

    if n < 10:
        return {
            "hit_rate": float("nan"),
            "avg_win": float("nan"),
            "avg_loss": float("nan"),
            "odds_ratio": float("nan"),
            "kelly_fraction": float("nan"),
            "half_kelly": float("nan"),
            "quarter_kelly": float("nan"),
            "continuous_kelly": float("nan"),
            "n_obs": n,
        }

    wins = clean[clean > 0]
    losses = clean[clean < 0]

    hit_rate = len(wins) / n if n > 0 else float("nan")
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.abs().mean()) if len(losses) > 0 else 1e-12

    # Guard against degenerate cases
    if avg_loss < 1e-12:
        avg_loss = 1e-12

    odds_ratio = avg_win / avg_loss

    # Classic discrete Kelly formula: f* = (p*b - q) / b
    # where p = hit_rate, q = 1-hit_rate, b = odds_ratio
    if odds_ratio > 1e-12:
        kelly_fraction = (hit_rate * odds_ratio - (1.0 - hit_rate)) / odds_ratio
    else:
        kelly_fraction = 0.0

    half_kelly = kelly_fraction / 2.0
    quarter_kelly = kelly_fraction / 4.0

    # Continuous Kelly: f* = μ / σ² (optimal for lognormal returns)
    mean_r = float(clean.mean())
    var_r = float(clean.var(ddof=1)) if n > 1 else float("nan")
    continuous_kelly = mean_r / var_r if (var_r is not None and var_r > 1e-16) else float("nan")

    return {
        "hit_rate": hit_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "odds_ratio": odds_ratio,
        "kelly_fraction": kelly_fraction,
        "half_kelly": half_kelly,
        "quarter_kelly": quarter_kelly,
        "continuous_kelly": continuous_kelly,
        "n_obs": n,
    }


def _kelly_vs_current(kelly_fraction: float, current_weight: float | None) -> str:
    """Compare current portfolio weight to Kelly fraction.

    Thresholds (using full Kelly as reference):
      - current > 1.5 * kelly  → Over-sized
      - 0.5 * kelly <= current <= 1.5 * kelly → Kelly-aligned
      - current < 0.5 * kelly  → Under-sized
      - unknown if any value is missing / non-finite
    """
    if current_weight is None:
        return "Unknown"
    if not np.isfinite(kelly_fraction) or not np.isfinite(current_weight):
        return "Unknown"
    if kelly_fraction <= 0:
        # Kelly says stay out; any positive exposure is over-sized
        return "Over-sized" if current_weight > 0.01 else "Kelly-aligned"

    ratio = current_weight / kelly_fraction
    if ratio > 1.5:
        return "Over-sized"
    elif ratio >= 0.5:
        return "Kelly-aligned"
    else:
        return "Under-sized"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_kelly_sizing(
    df: pd.DataFrame,
    return_col: str = "strategy_daily_return",
    signal_col: str = "composite_risk_score_smooth",
) -> dict:
    """Compute Kelly sizing statistics from the strategy's return history.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``return_col``.  If a ``date`` column is present it is
        used to split full history vs. out-of-sample (post ``OOS_START``).
    return_col : str
        Column name for daily strategy returns.
    signal_col : str
        Not directly used in computation but validated for presence.

    Returns
    -------
    dict with keys:
        ``full`` — Kelly stats computed over the full history
        ``oos``  — Kelly stats computed over the OOS window only
    """
    if return_col not in df.columns:
        empty = {
            "hit_rate": float("nan"),
            "avg_win": float("nan"),
            "avg_loss": float("nan"),
            "odds_ratio": float("nan"),
            "kelly_fraction": float("nan"),
            "half_kelly": float("nan"),
            "quarter_kelly": float("nan"),
            "continuous_kelly": float("nan"),
            "n_obs": 0,
        }
        return {"full": empty, "oos": empty}

    full_stats = _kelly_stats_from_returns(df[return_col])

    # OOS split — requires a date column
    if "date" in df.columns:
        date_col = pd.to_datetime(df["date"])
        oos_mask = date_col >= OOS_START
        oos_returns = df.loc[oos_mask, return_col]
    else:
        oos_returns = pd.Series(dtype=float)

    oos_stats = _kelly_stats_from_returns(oos_returns)

    return {"full": full_stats, "oos": oos_stats}


def run_kelly_analysis(df: pd.DataFrame) -> dict:
    """Run complete Kelly Criterion analysis including per-regime breakdown.

    Parameters
    ----------
    df : pd.DataFrame
        Expected columns: ``strategy_daily_return``, ``final_decision``,
        optionally ``strategy_weight``, ``date``.

    Returns
    -------
    dict with keys:
        ``full``             — Kelly stats over full history
        ``oos``              — Kelly stats over OOS window
        ``current_weight``   — float or None
        ``kelly_vs_current`` — "Over-sized" / "Kelly-aligned" / "Under-sized" / "Unknown"
        ``regime_kelly``     — pd.DataFrame: per-regime Kelly sizing
        ``available``        — bool
    """
    return_col = "strategy_daily_return"
    if return_col not in df.columns:
        return {
            "full": {},
            "oos": {},
            "current_weight": None,
            "kelly_vs_current": "Unknown",
            "regime_kelly": pd.DataFrame(),
            "available": False,
        }

    # ------------------------------------------------------------------
    # Full + OOS stats
    # ------------------------------------------------------------------
    sizing = compute_kelly_sizing(df, return_col=return_col)
    full_stats = sizing["full"]
    oos_stats = sizing["oos"]

    # ------------------------------------------------------------------
    # Current strategy weight (latest row)
    # ------------------------------------------------------------------
    current_weight: float | None = None
    last_row = df.iloc[-1]
    raw_w = last_row.get("strategy_weight", None)
    if raw_w is not None and not pd.isna(raw_w):
        current_weight = float(raw_w)

    # ------------------------------------------------------------------
    # Kelly vs current weight comparison
    # ------------------------------------------------------------------
    full_kelly = full_stats.get("kelly_fraction", float("nan"))
    comparison = _kelly_vs_current(full_kelly, current_weight)

    # ------------------------------------------------------------------
    # Per-regime Kelly
    # ------------------------------------------------------------------
    regime_rows: list[dict] = []

    if "final_decision" in df.columns:
        for regime, group in df.groupby("final_decision"):
            regime_returns = group[return_col].dropna()
            stats = _kelly_stats_from_returns(regime_returns)
            regime_rows.append(
                {
                    "regime": regime,
                    "hit_rate": stats["hit_rate"],
                    "odds_ratio": stats["odds_ratio"],
                    "kelly_fraction": stats["kelly_fraction"],
                    "half_kelly": stats["half_kelly"],
                    "quarter_kelly": stats["quarter_kelly"],
                    "continuous_kelly": stats["continuous_kelly"],
                    "n_obs": stats["n_obs"],
                }
            )

    regime_kelly = pd.DataFrame(regime_rows)
    if not regime_kelly.empty:
        regime_kelly = regime_kelly.sort_values("kelly_fraction", ascending=False).reset_index(
            drop=True
        )

    return {
        "full": full_stats,
        "oos": oos_stats,
        "current_weight": current_weight,
        "kelly_vs_current": comparison,
        "regime_kelly": regime_kelly,
        "available": True,
    }
