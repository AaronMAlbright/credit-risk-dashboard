"""
Alert engine for the Macro Credit Risk Dashboard.

Compares the current dashboard state against a persisted previous state
and sends HTML email notifications when configured trigger conditions fire.

Trigger conditions (each independently configurable):
  1. regime_change   — final_decision changed from previous run
  2. blend_below     — blend position weight crosses below threshold (default 50%)
  3. composite_spike — composite score moves > N points since last run (default 10)
  4. shock_flag      — shock_flag transitions from "No Shock" to an active shock
  5. composite_cross — composite crosses stress_threshold upward (default 50)

State is persisted to a JSON file so consecutive runs only alert on new changes.
Email is sent via SMTP (TLS on port 587 by default; configurable).
All SMTP credentials are read from environment variables:
  ALERT_SMTP_HOST, ALERT_SMTP_PORT, ALERT_SMTP_USER, ALERT_SMTP_PASSWORD,
  ALERT_FROM_ADDR, ALERT_TO_ADDRS  (comma-separated)

Public API
----------
  AlertConfig     — dataclass holding all configuration
  ALERT_LEVELS    — {"INFO": 0, "WARNING": 1, "ALERT": 2}
  DEFAULT_STATE   — empty/sentinel state dict
  load_alert_state  — read state JSON from disk
  save_alert_state  — write state JSON to disk
  extract_current_state — pull relevant fields from scored df + sizing result
  check_alerts    — compare current vs. previous state → list of alert dicts
  format_alert_email — render HTML alert email body
  send_alert_email  — connect to SMTP and send
  run_alerts      — orchestrator: load → check → send → save
"""

from __future__ import annotations

import json
import os
import smtplib
import ssl
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALERT_LEVELS: dict[str, int] = {"INFO": 0, "WARNING": 1, "ALERT": 2}

_STRESS_DECISIONS = {"Risk-Off", "Caution"}

_DECISION_COLORS: dict[str, str] = {
    "Risk-On":  "#27ae60",
    "Neutral":  "#95a5a6",
    "Caution":  "#e67e22",
    "Risk-Off": "#e74c3c",
}

_LEVEL_COLORS = {"INFO": "#3498db", "WARNING": "#e67e22", "ALERT": "#e74c3c"}

DEFAULT_STATE: dict = {
    "last_run":    "",
    "last_date":   "",
    "decision":    "",
    "composite":   None,
    "blend":       None,
    "shock_flag":  "No Shock",
}

_DEFAULT_STATE_PATH = Path("history/alert_state.json")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AlertConfig:
    """Alert engine configuration. All fields have safe defaults."""

    # ── SMTP ──────────────────────────────────────────────────────────────
    smtp_host:     str  = ""
    smtp_port:     int  = 587
    smtp_user:     str  = ""
    smtp_password: str  = ""
    from_addr:     str  = ""
    to_addrs:      list = field(default_factory=list)  # list[str]

    # ── Triggers ──────────────────────────────────────────────────────────
    check_regime_change:   bool  = True
    check_blend_below:     bool  = True
    check_composite_spike: bool  = True
    check_shock_flag:      bool  = True
    check_composite_cross: bool  = True

    # ── Thresholds ────────────────────────────────────────────────────────
    blend_threshold:            float = 0.50   # alert when blend drops below this
    composite_spike_pts:        float = 10.0   # abs change in composite triggering alert
    composite_stress_threshold: float = 50.0   # upward crossing triggers ALERT

    # ── Behaviour ─────────────────────────────────────────────────────────
    min_level: str  = "INFO"   # suppress alerts below this level
    dry_run:   bool = False    # if True, build email but don't send


def config_from_env() -> AlertConfig:
    """Build an AlertConfig from environment variables."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    to_raw = os.environ.get("ALERT_TO_ADDRS", "")
    to_addrs = [a.strip() for a in to_raw.split(",") if a.strip()]

    return AlertConfig(
        smtp_host=os.environ.get("ALERT_SMTP_HOST",     "smtp.gmail.com"),
        smtp_port=int(os.environ.get("ALERT_SMTP_PORT", "587")),
        smtp_user=os.environ.get("ALERT_SMTP_USER",     ""),
        smtp_password=os.environ.get("ALERT_SMTP_PASSWORD", ""),
        from_addr=os.environ.get("ALERT_FROM_ADDR",     ""),
        to_addrs=to_addrs,
    )


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def load_alert_state(path: Path | str = _DEFAULT_STATE_PATH) -> dict:
    """Load previous alert state from JSON. Returns DEFAULT_STATE if absent."""
    p = Path(path)
    if not p.exists():
        return dict(DEFAULT_STATE)
    try:
        data = json.loads(p.read_text())
        return {**DEFAULT_STATE, **data}
    except Exception:
        return dict(DEFAULT_STATE)


def save_alert_state(state: dict, path: Path | str = _DEFAULT_STATE_PATH) -> None:
    """Persist alert state to JSON, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2, default=str))


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------

def extract_current_state(
    df: pd.DataFrame,
    sizing: dict | None = None,
) -> dict:
    """
    Pull the relevant fields from the latest scored row + position sizing result.

    Returns dict:
      last_date, decision, composite, blend,
      shock_flag, regime, vix, hy_spread, sp500_drawdown
    """
    if df.empty:
        return dict(DEFAULT_STATE)

    row = df.iloc[-1]
    blend = None
    if sizing:
        cur = sizing.get("current", {})
        blend = cur.get("blend")

    return {
        "last_run":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "last_date":    str(row.get("date", ""))[:10],
        "decision":     str(row.get("final_decision",          "")),
        "composite":    float(row.get("composite_risk_score_smooth", 0)),
        "blend":        float(blend) if blend is not None else None,
        "shock_flag":   str(row.get("shock_flag",              "No Shock")),
        "vix":          float(row.get("vix",                   0)),
        "hy_spread":    float(row.get("hy_spread",             0)),
        "sp500_drawdown": float(row.get("sp500_drawdown",      0)),
        "environment":  str(row.get("final_environment",       "")),
        "action":       str(row.get("final_action",            "")),
    }


# ---------------------------------------------------------------------------
# Alert detection
# ---------------------------------------------------------------------------

def _alert(trigger: str, level: str, message: str, details: dict | None = None) -> dict:
    return {"trigger": trigger, "level": level, "message": message,
            "details": details or {}}


def check_alerts(
    current: dict,
    previous: dict,
    config: AlertConfig,
) -> list[dict]:
    """
    Compare current vs. previous state and return a list of fired alert dicts.

    Each alert dict: {trigger, level, message, details}
    Alerts are ordered by severity (ALERT first).
    """
    alerts: list[dict] = []

    curr_decision  = current.get("decision",  "")
    prev_decision  = previous.get("decision", "")
    curr_composite = float(current.get("composite", 0) or 0)
    prev_composite = float(previous.get("composite") or 0)
    curr_blend     = current.get("blend")
    prev_blend     = previous.get("blend")
    curr_shock     = current.get("shock_flag", "No Shock")
    prev_shock     = previous.get("shock_flag", "No Shock")

    # 1. Regime change
    if config.check_regime_change and prev_decision and curr_decision != prev_decision:
        level = "ALERT" if curr_decision in _STRESS_DECISIONS else "INFO"
        alerts.append(_alert(
            "regime_change", level,
            f"Regime changed: {prev_decision} → {curr_decision}",
            {"prev": prev_decision, "curr": curr_decision},
        ))

    # 2. Blend drops below threshold (crossing event only)
    if (config.check_blend_below
            and curr_blend is not None
            and prev_blend is not None):
        thr = config.blend_threshold
        if float(prev_blend) >= thr > float(curr_blend):
            level = "WARNING" if float(curr_blend) >= 0.30 else "ALERT"
            alerts.append(_alert(
                "blend_below", level,
                f"Blend sizing dropped below {thr:.0%}: now {float(curr_blend):.0%}",
                {"blend": float(curr_blend), "threshold": thr,
                 "prev_blend": float(prev_blend)},
            ))

    # 3. Composite spike
    if config.check_composite_spike and prev_composite:
        delta = curr_composite - prev_composite
        if abs(delta) >= config.composite_spike_pts:
            level = "ALERT" if delta > 0 else "WARNING"
            alerts.append(_alert(
                "composite_spike", level,
                f"Composite moved {delta:+.1f} pts: "
                f"{prev_composite:.1f} → {curr_composite:.1f}",
                {"delta": delta, "prev": prev_composite, "curr": curr_composite},
            ))

    # 4. Shock flag transition (No Shock → active shock)
    if config.check_shock_flag:
        if prev_shock == "No Shock" and curr_shock != "No Shock":
            alerts.append(_alert(
                "shock_flag", "ALERT",
                f"Market shock detected: {curr_shock}",
                {"shock": curr_shock},
            ))

    # 5. Composite crosses stress threshold upward
    if config.check_composite_cross and prev_composite:
        thr = config.composite_stress_threshold
        if prev_composite < thr <= curr_composite:
            alerts.append(_alert(
                "composite_cross", "ALERT",
                f"Composite crossed stress threshold "
                f"({thr:.0f}): now {curr_composite:.1f}",
                {"threshold": thr, "curr": curr_composite, "prev": prev_composite},
            ))

    # Sort: ALERT first, then WARNING, then INFO
    alerts.sort(key=lambda a: -ALERT_LEVELS.get(a["level"], 0))
    return alerts


# ---------------------------------------------------------------------------
# Email formatting
# ---------------------------------------------------------------------------

def format_alert_email(
    alerts: list[dict],
    current: dict,
    previous: dict | None = None,
) -> str:
    """
    Render a self-contained HTML email body for the given alerts.

    Returns an HTML string suitable for MIMEText('html').
    """
    now_str   = datetime.now().strftime("%Y-%m-%d %H:%M")
    data_date = current.get("last_date", "—")
    decision  = current.get("decision", "—")
    composite = current.get("composite")
    blend     = current.get("blend")
    shock     = current.get("shock_flag", "No Shock")

    badge_color  = _DECISION_COLORS.get(decision, "#95a5a6")
    comp_str     = f"{float(composite):.1f}" if composite is not None else "—"
    blend_str    = f"{float(blend):.0%}" if blend is not None else "—"

    top_level = alerts[0]["level"] if alerts else "INFO"
    top_color = _LEVEL_COLORS.get(top_level, "#888")

    # Alert rows
    alert_rows = ""
    for a in alerts:
        lv_color = _LEVEL_COLORS.get(a["level"], "#888")
        alert_rows += f"""
        <tr>
          <td style="padding:8px 12px">
            <span style="background:{lv_color};color:white;padding:2px 8px;
                         border-radius:10px;font-size:11px;font-weight:600">
              {a["level"]}
            </span>
          </td>
          <td style="padding:8px 12px;font-size:13px">{a["message"]}</td>
        </tr>"""

    # Previous state row
    prev_row = ""
    if previous and previous.get("decision"):
        prev_composite = previous.get("composite")
        prev_blend     = previous.get("blend")
        prev_row = f"""
        <p style="font-size:12px;color:#888;margin-top:8px">
          Previous: <b>{previous.get("decision","—")}</b> &nbsp;·&nbsp;
          Composite {f"{float(prev_composite):.1f}" if prev_composite else "—"} &nbsp;·&nbsp;
          Blend {f"{float(prev_blend):.0%}" if prev_blend else "—"} &nbsp;·&nbsp;
          Data {previous.get("last_date","—")}
        </p>"""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
             background:#f8f9fa;padding:0;margin:0">
  <div style="max-width:640px;margin:0 auto;padding:24px">

    <!-- Header -->
    <div style="background:#1a1a2e;color:white;padding:20px 28px;border-radius:8px 8px 0 0">
      <div style="font-size:18px;font-weight:700">Macro Credit Risk Dashboard</div>
      <div style="font-size:12px;color:#aab;margin-top:4px">
        Alert &nbsp;·&nbsp; {data_date} &nbsp;·&nbsp; Generated {now_str}
      </div>
    </div>

    <!-- Alert banner -->
    <div style="background:{top_color};color:white;padding:12px 28px">
      <b>{len(alerts)} trigger{'s' if len(alerts)!=1 else ''} fired</b>
      &nbsp;·&nbsp; Highest severity: {top_level}
    </div>

    <!-- Current state -->
    <div style="background:white;padding:20px 28px;border-bottom:1px solid #eee">
      <span style="background:{badge_color};color:white;padding:5px 14px;
                   border-radius:20px;font-weight:600;font-size:13px">
        {decision}
      </span>
      <div style="display:flex;gap:24px;margin-top:14px;flex-wrap:wrap">
        <div>
          <div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.5px">
            Composite</div>
          <div style="font-size:20px;font-weight:700;color:#1a1a2e">{comp_str}</div>
        </div>
        <div>
          <div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.5px">
            Blend Sizing</div>
          <div style="font-size:20px;font-weight:700;color:#1a1a2e">{blend_str}</div>
        </div>
        <div>
          <div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.5px">
            Shock Flag</div>
          <div style="font-size:20px;font-weight:700;color:#{'e74c3c' if shock!='No Shock' else '1a1a2e'}">
            {shock}
          </div>
        </div>
        <div>
          <div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.5px">
            VIX</div>
          <div style="font-size:20px;font-weight:700;color:#1a1a2e">
            {f"{float(current.get('vix',0)):.1f}"}
          </div>
        </div>
        <div>
          <div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.5px">
            HY Spread</div>
          <div style="font-size:20px;font-weight:700;color:#1a1a2e">
            {f"{float(current.get('hy_spread',0)):.2f}%"}
          </div>
        </div>
      </div>
      {prev_row}
    </div>

    <!-- Alert table -->
    <div style="background:white;padding:0 0 4px">
      <table style="width:100%;border-collapse:collapse">
        <tr style="background:#f8f9fa">
          <th style="text-align:left;padding:8px 12px;font-size:11px;
                     color:#888;text-transform:uppercase">Level</th>
          <th style="text-align:left;padding:8px 12px;font-size:11px;
                     color:#888;text-transform:uppercase">Trigger</th>
        </tr>
        {alert_rows}
      </table>
    </div>

    <!-- Action / environment -->
    <div style="background:white;padding:14px 28px;border-top:1px solid #eee;
                border-radius:0 0 8px 8px">
      <div style="font-size:12px;color:#555">
        <b>Environment:</b> {current.get("environment","—")}
      </div>
      <div style="font-size:12px;color:#555;margin-top:4px">
        <b>Action:</b> {current.get("action","—")}
      </div>
    </div>

    <!-- Footer -->
    <div style="text-align:center;color:#aaa;font-size:11px;padding:16px 0">
      <p>Macro Credit Risk Dashboard &nbsp;·&nbsp; For research purposes only.</p>
      <p style="margin-top:4px">Not investment advice. Past performance ≠ future results.</p>
    </div>
  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Email sending
# ---------------------------------------------------------------------------

def send_alert_email(
    html_body: str,
    alerts: list[dict],
    config: AlertConfig,
) -> bool:
    """
    Send the HTML alert email via SMTP (TLS).

    Returns True on success, False on failure.
    Raises ValueError if SMTP config is incomplete.
    """
    if not config.smtp_host or not config.smtp_user or not config.to_addrs:
        raise ValueError(
            "SMTP config incomplete. Set smtp_host, smtp_user, "
            "smtp_password, from_addr, to_addrs."
        )

    top_level = alerts[0]["level"] if alerts else "INFO"
    subject   = (
        f"[{top_level}] Macro Credit Risk — "
        f"{alerts[0]['message'][:60] if alerts else 'Alert'}"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = config.from_addr or config.smtp_user
    msg["To"]      = ", ".join(config.to_addrs)
    msg.attach(MIMEText(html_body, "html"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls(context=context)
            server.login(config.smtp_user, config.smtp_password)
            server.sendmail(
                config.from_addr or config.smtp_user,
                config.to_addrs,
                msg.as_string(),
            )
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_alerts(
    df: pd.DataFrame,
    config: AlertConfig | None = None,
    sizing: dict | None = None,
    state_path: Path | str = _DEFAULT_STATE_PATH,
) -> dict:
    """
    Full alert pipeline: load state → check → send → save.

    Parameters
    ----------
    df         : scored DataFrame
    config     : AlertConfig (defaults built from env vars if None)
    sizing     : position sizing result for blend weight (optional)
    state_path : path to JSON state file

    Returns dict:
      alerts       — list of fired alert dicts (empty if none)
      current      — current state dict
      previous     — previous state dict
      email_sent   — True if email was sent, False otherwise, None if not attempted
      email_html   — HTML body (None if no alerts fired)
      state_saved  — True if state was written
    """
    if config is None:
        config = config_from_env()

    previous = load_alert_state(state_path)
    current  = extract_current_state(df, sizing)

    if df.empty:
        return {
            "alerts":     [],
            "current":    current,
            "previous":   previous,
            "email_sent": None,
            "email_html": None,
            "state_saved": False,
        }

    alerts = check_alerts(current, previous, config)

    # Filter by min_level
    min_rank = ALERT_LEVELS.get(config.min_level, 0)
    alerts   = [a for a in alerts if ALERT_LEVELS.get(a["level"], 0) >= min_rank]

    email_html = None
    email_sent = None

    if alerts:
        email_html = format_alert_email(alerts, current, previous)
        if config.dry_run:
            email_sent = False
        else:
            try:
                email_sent = send_alert_email(email_html, alerts, config)
            except ValueError:
                email_sent = False

    save_alert_state(current, state_path)

    return {
        "alerts":      alerts,
        "current":     current,
        "previous":    previous,
        "email_sent":  email_sent,
        "email_html":  email_html,
        "state_saved": True,
    }
