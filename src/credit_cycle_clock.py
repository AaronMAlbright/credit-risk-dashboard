"""credit_cycle_clock.py

Maps current credit market conditions to a position on a circular "credit cycle
clock" and returns a Plotly figure.

Credit cycle phases (clockwise from top, 0° = 12 o'clock):
  0°–90°   : Early Expansion  — low stress, recovering spreads, improving macro
  90°–180° : Late Cycle       — complacency rising, spreads tight, momentum slowing
  180°–270°: Contraction      — stress rising, spreads widening, macro deteriorating
  270°–360°: Recovery         — stress peaking/falling, spreads near highs, macro bottoming

The clock is driven by ``composite_risk_score_smooth`` (0–100) using a piecewise
linear mapping onto the four quadrants, with the current position shown as a
large colored dot and the trailing 90 days rendered as a fading track.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Phase metadata
# ---------------------------------------------------------------------------

_PHASES = [
    {"label": "Early\nExpansion", "angle": 45,  "color": "#00c853"},  # green
    {"label": "Late\nCycle",     "angle": 135, "color": "#ffd600"},  # yellow
    {"label": "Contraction",     "angle": 225, "color": "#d50000"},  # red
    {"label": "Recovery",        "angle": 315, "color": "#2979ff"},  # blue
]

# Score → (angle_start, angle_end) piecewise bands
# score 0–30   → 315° to 45°  (expansion, wraps through 0°)
# score 30–50  → 45° to 135°  (late cycle)
# score 50–70  → 135° to 225° (contraction)
# score 70–100 → 225° to 315° (recovery/bottom)
_BANDS = [
    (0,  30,  315, 405),   # 315→360→45 — use 405 so lerp stays monotone; mod 360 after
    (30, 50,   45, 135),
    (50, 70,  135, 225),
    (70, 100, 225, 315),
]


def compute_cycle_angle(row: dict) -> float:
    """Return the credit cycle clock angle (0–360°) for a single scored row.

    The mapping is driven primarily by ``composite_risk_score_smooth``.
    Values outside [0, 100] are clipped before mapping.

    Parameters
    ----------
    row:
        A dict (or dict-like) containing at minimum the key
        ``composite_risk_score_smooth``.

    Returns
    -------
    float
        Angle in degrees [0, 360), where 0° / 360° is the top of the clock
        (12 o'clock) and values increase clockwise.
    """
    score = float(row.get("composite_risk_score_smooth", 50))
    score = float(np.clip(score, 0, 100))

    for s_lo, s_hi, a_lo, a_hi in _BANDS:
        if s_lo <= score <= s_hi:
            t = (score - s_lo) / (s_hi - s_lo) if s_hi != s_lo else 0.0
            raw = a_lo + t * (a_hi - a_lo)
            return float(raw % 360)

    # Fallback — should not reach here given clip above
    return 0.0


def _angle_to_color(angle: float) -> str:
    """Map a cycle clock angle to its phase color."""
    angle = angle % 360
    if angle < 90 or angle >= 315:
        return "#00c853"   # expansion — green
    if 90 <= angle < 180:
        return "#ffd600"   # late cycle — yellow
    if 180 <= angle < 270:
        return "#d50000"   # contraction — red
    return "#2979ff"       # recovery — blue


def _angles_for_df(df: pd.DataFrame) -> np.ndarray:
    """Compute cycle angles for every row in *df* and return as a numpy array."""
    return np.array([compute_cycle_angle(row) for _, row in df.iterrows()])


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def build_credit_cycle_clock(df: pd.DataFrame) -> go.Figure:
    """Build a polar "credit cycle clock" Plotly figure.

    Parameters
    ----------
    df:
        A scored DataFrame that must contain ``composite_risk_score_smooth`` and
        (optionally) ``complacency_score_smooth``, ``macro_risk_score_smooth``,
        and ``credit_market_risk_score_smooth``.  The **last row** is treated as
        the current market position.  The previous 90 rows (trading days) are
        rendered as a fading trail.

    Returns
    -------
    go.Figure
        A dark-themed Plotly scatterpolar figure, 380 px tall.
    """
    if df.empty:
        # Return an empty figure with a message rather than crashing.
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            annotations=[
                dict(
                    text="No data available",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(color="#888", size=14),
                )
            ],
        )
        return fig

    # -----------------------------------------------------------------------
    # Compute angles
    # -----------------------------------------------------------------------
    all_angles = _angles_for_df(df)
    trail_angles = all_angles[-90:]           # last 90 rows (or fewer)
    current_angle = float(all_angles[-1])
    current_color = _angle_to_color(current_angle)

    n_trail = len(trail_angles)

    # -----------------------------------------------------------------------
    # Build opacities: oldest point ≈ 0.08, newest (non-current) ≈ 0.55
    # -----------------------------------------------------------------------
    if n_trail > 1:
        opacities = np.linspace(0.08, 0.55, n_trail)
    else:
        opacities = np.array([0.55])

    # -----------------------------------------------------------------------
    # Quadrant ring — four colored arcs represented as dense point bands
    # -----------------------------------------------------------------------
    arc_traces = []
    arc_defs = [
        (315, 405, "#00c853", "Early Expansion"),   # wraps through 0°
        (45,  135, "#ffd600", "Late Cycle"),
        (135, 225, "#d50000", "Contraction"),
        (225, 315, "#2979ff", "Recovery"),
    ]
    for a_start, a_end, color, name in arc_defs:
        arc_theta = np.linspace(a_start % 360, a_end % 360, 60)
        # For the wrap-around arc (expansion), linspace from 315→45 going
        # through 360 needs special handling.
        if a_start == 315 and a_end == 405:
            arc_theta = np.linspace(315, 405, 60) % 360
        arc_traces.append(
            go.Scatterpolar(
                r=np.full(len(arc_theta), 1.08),
                theta=arc_theta,
                mode="lines",
                line=dict(color=color, width=10),
                opacity=0.35,
                showlegend=False,
                hoverinfo="skip",
                name=name,
            )
        )

    # -----------------------------------------------------------------------
    # Trail trace — individual points so each can have its own opacity
    # -----------------------------------------------------------------------
    trail_traces = []
    for i, (ang, opa) in enumerate(zip(trail_angles[:-1], opacities[:-1])):
        col = _angle_to_color(ang)
        trail_traces.append(
            go.Scatterpolar(
                r=[1.0],
                theta=[ang],
                mode="markers",
                marker=dict(
                    size=5,
                    color=col,
                    opacity=float(opa),
                ),
                showlegend=False,
                hoverinfo="skip",
                name="",
            )
        )

    # -----------------------------------------------------------------------
    # Current position — large dot + crosshair line from center
    # -----------------------------------------------------------------------
    current_trace = go.Scatterpolar(
        r=[0, 1.0],
        theta=[current_angle, current_angle],
        mode="lines+markers",
        line=dict(color=current_color, width=2, dash="dot"),
        marker=dict(
            size=[0, 18],
            color=current_color,
            line=dict(color="white", width=2),
        ),
        showlegend=False,
        hovertemplate=(
            f"<b>Cycle angle:</b> {current_angle:.1f}°<br>"
            f"<b>Phase:</b> {_phase_name(current_angle)}"
            "<extra></extra>"
        ),
        name="Current",
    )

    # -----------------------------------------------------------------------
    # Assemble figure
    # -----------------------------------------------------------------------
    fig = go.Figure(data=[*arc_traces, *trail_traces, current_trace])

    # Phase label annotations (placed just outside the arc ring)
    annotations = []
    for phase in _PHASES:
        angle_rad = np.radians(phase["angle"])
        # In polar coordinates the annotation x/y live in paper space;
        # we approximate using the known geometry of the plot area.
        annotations.append(
            dict(
                text=f"<b>{phase['label']}</b>",
                x=0.5 + 0.42 * np.sin(angle_rad),
                y=0.5 + 0.42 * np.cos(angle_rad),
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color=phase["color"], size=11),
                align="center",
            )
        )

    # Title annotation (center of polar plot)
    annotations.append(
        dict(
            text=(
                f"<b style='font-size:13px'>Credit Cycle Clock</b><br>"
                f"<span style='font-size:11px;color:{current_color}'>"
                f"{_phase_name(current_angle)}</span><br>"
                f"<span style='font-size:10px;color:#888'>{current_angle:.0f}°</span>"
            ),
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="center",
            font=dict(color="white", size=11),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        annotations=annotations,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            angularaxis=dict(
                tickmode="array",
                tickvals=[0, 90, 180, 270],
                ticktext=["12", "3", "6", "9"],
                direction="clockwise",
                rotation=90,           # 0° at top (12 o'clock)
                showgrid=True,
                gridcolor="rgba(255,255,255,0.08)",
                linecolor="rgba(255,255,255,0.15)",
                tickfont=dict(color="rgba(255,255,255,0.4)", size=9),
            ),
            radialaxis=dict(
                visible=False,
                range=[0, 1.2],
            ),
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phase_name(angle: float) -> str:
    """Return the phase name for a given clock angle."""
    angle = angle % 360
    if angle < 90 or angle >= 315:
        return "Early Expansion"
    if 90 <= angle < 180:
        return "Late Cycle"
    if 180 <= angle < 270:
        return "Contraction"
    return "Recovery"
