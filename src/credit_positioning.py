"""
Credit positioning playbook.

Maps model regimes and spread valuation into implementable credit language:
HY/IG tilt, quality bias, duration stance, liquidity stance, and hedge posture.
"""

from __future__ import annotations

import pandas as pd

from src.spread_decomposition import latest_spread_snapshot


REGIME_PLAYBOOK = {
    "Risk-On": {
        "credit_beta": "Overweight",
        "hy_tilt": "Overweight",
        "ig_tilt": "Neutral",
        "quality_bias": "BB/B with selective CCC exposure",
        "duration_stance": "Neutral",
        "liquidity_stance": "Normal liquidity buffer",
        "hedge_posture": "Minimal index hedges",
    },
    "Neutral": {
        "credit_beta": "Market weight",
        "hy_tilt": "Neutral",
        "ig_tilt": "Neutral",
        "quality_bias": "Balanced quality exposure",
        "duration_stance": "Neutral",
        "liquidity_stance": "Moderate liquidity buffer",
        "hedge_posture": "Hedge only idiosyncratic risks",
    },
    "Caution": {
        "credit_beta": "Underweight",
        "hy_tilt": "Underweight",
        "ig_tilt": "Overweight",
        "quality_bias": "Upgrade quality; avoid CCC and weak liquidity",
        "duration_stance": "Neutral to long",
        "liquidity_stance": "Raise cash and liquid IG exposure",
        "hedge_posture": "Consider CDX HY/IG hedges",
    },
    "Risk-Off": {
        "credit_beta": "Defensive",
        "hy_tilt": "Avoid",
        "ig_tilt": "High-quality IG only",
        "quality_bias": "A/AA quality, Treasuries, cash",
        "duration_stance": "Long duration if rates are rallying",
        "liquidity_stance": "Preserve liquidity",
        "hedge_posture": "Maintain index hedges until stabilization",
    },
}


def _valuation_adjustment(valuation: str) -> str:
    if valuation in {"Very Cheap", "Cheap"}:
        return "Valuation supports adding risk only after technical stabilization."
    if valuation == "Fair":
        return "Valuation is not a standalone reason to add or cut risk."
    if valuation in {"Rich", "Very Rich"}:
        return "Valuation does not compensate strongly for default-cycle risk."
    return "Spread valuation is unavailable."


def current_positioning(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"available": False, "reason": "No data available"}

    latest = df.iloc[-1]
    regime = latest.get("final_decision", "Neutral")
    playbook = dict(REGIME_PLAYBOOK.get(regime, REGIME_PLAYBOOK["Neutral"]))
    spread = latest_spread_snapshot(df)

    playbook.update(
        {
            "available": True,
            "regime": regime,
            "spread_valuation": spread.get("spread_valuation") if spread["available"] else "Unavailable",
            "valuation_note": _valuation_adjustment(
                spread.get("spread_valuation", "Unavailable") if spread["available"] else "Unavailable"
            ),
        }
    )
    return playbook


def positioning_table(df: pd.DataFrame) -> pd.DataFrame:
    current = current_positioning(df)
    if not current.get("available"):
        return pd.DataFrame()

    keys = [
        "credit_beta",
        "hy_tilt",
        "ig_tilt",
        "quality_bias",
        "duration_stance",
        "liquidity_stance",
        "hedge_posture",
        "spread_valuation",
        "valuation_note",
    ]
    return pd.DataFrame(
        [{"dimension": key.replace("_", " ").title(), "stance": current[key]} for key in keys]
    )

