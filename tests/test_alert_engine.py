"""
Tests for src/alert_engine.py

Coverage:
  - load_alert_state / save_alert_state (state persistence)
  - extract_current_state (field extraction from df + sizing)
  - check_alerts (all 5 triggers, disabled triggers, severity sorting)
  - format_alert_email (HTML structure)
  - send_alert_email (SMTP mocked, ValueError on bad config)
  - run_alerts (orchestrator paths: empty df, dry_run, no alerts, normal)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.alert_engine import (
    ALERT_LEVELS,
    DEFAULT_STATE,
    AlertConfig,
    check_alerts,
    check_scorecard_alerts,
    config_from_env,
    extract_current_state,
    format_alert_email,
    load_alert_state,
    run_alerts,
    save_alert_state,
    send_alert_email,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(**overrides) -> pd.DataFrame:
    """Minimal scored DataFrame row."""
    base = {
        "date":                        "2026-05-07",
        "final_decision":              "Neutral",
        "composite_risk_score_smooth": 42.0,
        "shock_flag":                  "No Shock",
        "final_environment":           "Late Cycle",
        "final_action":                "Reduce Duration",
        "vix":                         18.5,
        "hy_spread":                   3.50,
        "sp500_drawdown":              -0.05,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def _make_sizing(blend: float = 0.65) -> dict:
    return {"current": {"blend": blend}}


def _cfg(**kw) -> AlertConfig:
    """AlertConfig with all checks enabled and safe defaults."""
    defaults = dict(
        smtp_host="smtp.example.com",
        smtp_user="user@example.com",
        smtp_password="secret",
        from_addr="user@example.com",
        to_addrs=["recipient@example.com"],
        check_scorecard_alerts=False,
    )
    defaults.update(kw)
    return AlertConfig(**defaults)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

class TestLoadSaveState:
    def test_load_missing_file_returns_default(self, tmp_path):
        state = load_alert_state(tmp_path / "nonexistent.json")
        assert state == DEFAULT_STATE

    def test_load_corrupt_json_returns_default(self, tmp_path):
        p = tmp_path / "state.json"
        p.write_text("{ not valid json }")
        state = load_alert_state(p)
        assert state == DEFAULT_STATE

    def test_save_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "a" / "b" / "state.json"
        save_alert_state({"decision": "Hold"}, p)
        assert p.exists()

    def test_round_trip(self, tmp_path):
        p = tmp_path / "state.json"
        original = {**DEFAULT_STATE, "decision": "Risk-On", "composite": 61.0}
        save_alert_state(original, p)
        loaded = load_alert_state(p)
        assert loaded["decision"] == "Risk-On"
        assert loaded["composite"] == 61.0

    def test_load_merges_with_defaults(self, tmp_path):
        """Partial JSON should be filled in from DEFAULT_STATE."""
        p = tmp_path / "state.json"
        p.write_text(json.dumps({"decision": "Neutral"}))
        state = load_alert_state(p)
        assert state["shock_flag"] == "No Shock"   # from DEFAULT_STATE
        assert state["decision"] == "Neutral"       # from file

    def test_save_persists_all_fields(self, tmp_path):
        p = tmp_path / "state.json"
        data = {"last_run": "2026-05-07 09:00", "composite": 55.0}
        save_alert_state(data, p)
        raw = json.loads(p.read_text())
        assert raw["last_run"] == "2026-05-07 09:00"
        assert raw["composite"] == 55.0


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------

class TestExtractCurrentState:
    def test_empty_df_returns_default(self):
        state = extract_current_state(pd.DataFrame())
        assert state == DEFAULT_STATE

    def test_basic_fields(self):
        df = _make_df()
        state = extract_current_state(df)
        assert state["decision"] == "Neutral"
        assert abs(state["composite"] - 42.0) < 0.01
        assert state["shock_flag"] == "No Shock"
        assert state["last_date"] == "2026-05-07"

    def test_blend_from_sizing(self):
        df = _make_df()
        state = extract_current_state(df, _make_sizing(0.72))
        assert abs(state["blend"] - 0.72) < 0.001

    def test_blend_none_when_no_sizing(self):
        state = extract_current_state(_make_df())
        assert state["blend"] is None

    def test_blend_none_when_sizing_missing_current(self):
        state = extract_current_state(_make_df(), {"other": 1})
        assert state["blend"] is None

    def test_vix_and_spread_extracted(self):
        df = _make_df(vix=25.3, hy_spread=4.10)
        state = extract_current_state(df)
        assert abs(state["vix"] - 25.3) < 0.01
        assert abs(state["hy_spread"] - 4.10) < 0.01

    def test_last_run_is_set(self):
        state = extract_current_state(_make_df())
        assert state["last_run"] != ""

    def test_environment_and_action(self):
        df = _make_df(final_environment="Recession", final_action="Exit Risk")
        state = extract_current_state(df)
        assert state["environment"] == "Recession"
        assert state["action"] == "Exit Risk"


# ---------------------------------------------------------------------------
# check_alerts — trigger 1: regime_change
# ---------------------------------------------------------------------------

class TestCheckAlertsRegimeChange:
    def _curr(self, decision="Risk-On") -> dict:
        return {**DEFAULT_STATE, "decision": decision, "composite": 55.0,
                "blend": 0.60, "shock_flag": "No Shock"}

    def _prev(self, decision="Neutral") -> dict:
        return {**DEFAULT_STATE, "decision": decision, "composite": 50.0,
                "blend": 0.65, "shock_flag": "No Shock"}

    def test_stress_decision_fires_alert(self):
        cfg = _cfg(check_blend_below=False, check_composite_spike=False,
                   check_shock_flag=False, check_composite_cross=False)
        alerts = check_alerts(self._curr("Risk-Off"), self._prev(), cfg)
        assert any(a["trigger"] == "regime_change" for a in alerts)
        match = next(a for a in alerts if a["trigger"] == "regime_change")
        assert match["level"] == "ALERT"

    def test_non_stress_decision_fires_info(self):
        cfg = _cfg(check_blend_below=False, check_composite_spike=False,
                   check_shock_flag=False, check_composite_cross=False)
        alerts = check_alerts(self._curr("Neutral"), self._prev("Caution"), cfg)
        match = next(a for a in alerts if a["trigger"] == "regime_change")
        assert match["level"] == "INFO"

    def test_no_change_no_fire(self):
        cfg = _cfg(check_blend_below=False, check_composite_spike=False,
                   check_shock_flag=False, check_composite_cross=False)
        alerts = check_alerts(self._curr("Neutral"),
                               self._prev("Neutral"), cfg)
        assert not any(a["trigger"] == "regime_change" for a in alerts)

    def test_disabled_does_not_fire(self):
        cfg = _cfg(check_regime_change=False, check_blend_below=False,
                   check_composite_spike=False, check_shock_flag=False,
                   check_composite_cross=False)
        alerts = check_alerts(self._curr("Risk-On"), self._prev(), cfg)
        assert alerts == []

    def test_empty_previous_decision_no_fire(self):
        cfg = _cfg(check_blend_below=False, check_composite_spike=False,
                   check_shock_flag=False, check_composite_cross=False)
        prev = {**DEFAULT_STATE, "decision": ""}
        alerts = check_alerts(self._curr("Risk-On"), prev, cfg)
        assert not any(a["trigger"] == "regime_change" for a in alerts)


# ---------------------------------------------------------------------------
# check_alerts — trigger 2: blend_below
# ---------------------------------------------------------------------------

class TestCheckAlertsBlendBelow:
    def _state(self, blend, decision="Neutral", composite=42.0,
               shock="No Shock") -> dict:
        return {**DEFAULT_STATE, "decision": decision, "composite": composite,
                "blend": blend, "shock_flag": shock}

    def test_crossing_below_threshold_fires(self):
        cfg = _cfg(check_regime_change=False, check_composite_spike=False,
                   check_shock_flag=False, check_composite_cross=False,
                   blend_threshold=0.50)
        alerts = check_alerts(self._state(0.45), self._state(0.55), cfg)
        assert any(a["trigger"] == "blend_below" for a in alerts)

    def test_already_below_no_fire(self):
        """Crossing must happen — already below last run shouldn't re-trigger."""
        cfg = _cfg(check_regime_change=False, check_composite_spike=False,
                   check_shock_flag=False, check_composite_cross=False,
                   blend_threshold=0.50)
        alerts = check_alerts(self._state(0.40), self._state(0.45), cfg)
        assert not any(a["trigger"] == "blend_below" for a in alerts)

    def test_above_threshold_no_fire(self):
        cfg = _cfg(check_regime_change=False, check_composite_spike=False,
                   check_shock_flag=False, check_composite_cross=False,
                   blend_threshold=0.50)
        alerts = check_alerts(self._state(0.65), self._state(0.70), cfg)
        assert not any(a["trigger"] == "blend_below" for a in alerts)

    def test_level_warning_when_blend_ge_30pct(self):
        cfg = _cfg(check_regime_change=False, check_composite_spike=False,
                   check_shock_flag=False, check_composite_cross=False,
                   blend_threshold=0.50)
        alerts = check_alerts(self._state(0.35), self._state(0.55), cfg)
        match = next(a for a in alerts if a["trigger"] == "blend_below")
        assert match["level"] == "WARNING"

    def test_level_alert_when_blend_lt_30pct(self):
        cfg = _cfg(check_regime_change=False, check_composite_spike=False,
                   check_shock_flag=False, check_composite_cross=False,
                   blend_threshold=0.50)
        alerts = check_alerts(self._state(0.20), self._state(0.55), cfg)
        match = next(a for a in alerts if a["trigger"] == "blend_below")
        assert match["level"] == "ALERT"

    def test_none_blend_skipped(self):
        cfg = _cfg(check_regime_change=False, check_composite_spike=False,
                   check_shock_flag=False, check_composite_cross=False)
        curr = {**DEFAULT_STATE, "blend": None}
        prev = {**DEFAULT_STATE, "blend": 0.70}
        alerts = check_alerts(curr, prev, cfg)
        assert not any(a["trigger"] == "blend_below" for a in alerts)


# ---------------------------------------------------------------------------
# check_alerts — trigger 3: composite_spike
# ---------------------------------------------------------------------------

class TestCheckAlertsCompositeSpike:
    def _state(self, composite, blend=0.60) -> dict:
        return {**DEFAULT_STATE, "composite": composite, "blend": blend,
                "decision": "Neutral", "shock_flag": "No Shock"}

    def test_upward_spike_fires_alert(self):
        cfg = _cfg(check_regime_change=False, check_blend_below=False,
                   check_shock_flag=False, check_composite_cross=False,
                   composite_spike_pts=10.0)
        alerts = check_alerts(self._state(62.0), self._state(50.0), cfg)
        match = next(a for a in alerts if a["trigger"] == "composite_spike")
        assert match["level"] == "ALERT"

    def test_downward_spike_fires_warning(self):
        cfg = _cfg(check_regime_change=False, check_blend_below=False,
                   check_shock_flag=False, check_composite_cross=False,
                   composite_spike_pts=10.0)
        alerts = check_alerts(self._state(38.0), self._state(50.0), cfg)
        match = next(a for a in alerts if a["trigger"] == "composite_spike")
        assert match["level"] == "WARNING"

    def test_small_move_does_not_fire(self):
        cfg = _cfg(check_regime_change=False, check_blend_below=False,
                   check_shock_flag=False, check_composite_cross=False,
                   composite_spike_pts=10.0)
        alerts = check_alerts(self._state(53.0), self._state(50.0), cfg)
        assert not any(a["trigger"] == "composite_spike" for a in alerts)

    def test_exact_threshold_fires(self):
        cfg = _cfg(check_regime_change=False, check_blend_below=False,
                   check_shock_flag=False, check_composite_cross=False,
                   composite_spike_pts=10.0)
        alerts = check_alerts(self._state(60.0), self._state(50.0), cfg)
        assert any(a["trigger"] == "composite_spike" for a in alerts)

    def test_zero_prev_composite_skipped(self):
        """Guard: prev_composite=0 treated as falsy → no spike check."""
        cfg = _cfg(check_regime_change=False, check_blend_below=False,
                   check_shock_flag=False, check_composite_cross=False)
        alerts = check_alerts(self._state(60.0), self._state(0.0), cfg)
        assert not any(a["trigger"] == "composite_spike" for a in alerts)


# ---------------------------------------------------------------------------
# check_alerts — trigger 4: shock_flag
# ---------------------------------------------------------------------------

class TestCheckAlertsShockFlag:
    def _state(self, shock) -> dict:
        return {**DEFAULT_STATE, "shock_flag": shock, "composite": 45.0,
                "decision": "Neutral"}

    def test_no_shock_to_shock_fires_alert(self):
        cfg = _cfg(check_regime_change=False, check_blend_below=False,
                   check_composite_spike=False, check_composite_cross=False)
        alerts = check_alerts(self._state("VIX Spike"), self._state("No Shock"), cfg)
        match = next(a for a in alerts if a["trigger"] == "shock_flag")
        assert match["level"] == "ALERT"
        assert "VIX Spike" in match["message"]

    def test_shock_to_shock_no_fire(self):
        cfg = _cfg(check_regime_change=False, check_blend_below=False,
                   check_composite_spike=False, check_composite_cross=False)
        alerts = check_alerts(self._state("HY Widening"), self._state("VIX Spike"), cfg)
        assert not any(a["trigger"] == "shock_flag" for a in alerts)

    def test_shock_to_no_shock_no_fire(self):
        cfg = _cfg(check_regime_change=False, check_blend_below=False,
                   check_composite_spike=False, check_composite_cross=False)
        alerts = check_alerts(self._state("No Shock"), self._state("HY Widening"), cfg)
        assert not any(a["trigger"] == "shock_flag" for a in alerts)

    def test_disabled_does_not_fire(self):
        cfg = _cfg(check_regime_change=False, check_blend_below=False,
                   check_composite_spike=False, check_shock_flag=False,
                   check_composite_cross=False)
        alerts = check_alerts(self._state("VIX Spike"), self._state("No Shock"), cfg)
        assert alerts == []


# ---------------------------------------------------------------------------
# check_alerts — trigger 5: composite_cross
# ---------------------------------------------------------------------------

class TestCheckAlertsCompositeCross:
    def _state(self, composite) -> dict:
        return {**DEFAULT_STATE, "composite": composite,
                "decision": "Neutral", "shock_flag": "No Shock"}

    def test_upward_cross_fires_alert(self):
        cfg = _cfg(check_regime_change=False, check_blend_below=False,
                   check_composite_spike=False, check_shock_flag=False,
                   composite_stress_threshold=50.0)
        alerts = check_alerts(self._state(52.0), self._state(48.0), cfg)
        match = next(a for a in alerts if a["trigger"] == "composite_cross")
        assert match["level"] == "ALERT"

    def test_already_above_no_fire(self):
        cfg = _cfg(check_regime_change=False, check_blend_below=False,
                   check_composite_spike=False, check_shock_flag=False,
                   composite_stress_threshold=50.0)
        alerts = check_alerts(self._state(55.0), self._state(52.0), cfg)
        assert not any(a["trigger"] == "composite_cross" for a in alerts)

    def test_downward_cross_no_fire(self):
        cfg = _cfg(check_regime_change=False, check_blend_below=False,
                   check_composite_spike=False, check_shock_flag=False,
                   composite_stress_threshold=50.0)
        alerts = check_alerts(self._state(47.0), self._state(53.0), cfg)
        assert not any(a["trigger"] == "composite_cross" for a in alerts)

    def test_exact_threshold_fires(self):
        cfg = _cfg(check_regime_change=False, check_blend_below=False,
                   check_composite_spike=False, check_shock_flag=False,
                   composite_stress_threshold=50.0)
        alerts = check_alerts(self._state(50.0), self._state(49.9), cfg)
        assert any(a["trigger"] == "composite_cross" for a in alerts)


# ---------------------------------------------------------------------------
# check_alerts — severity ordering and min_level
# ---------------------------------------------------------------------------

class TestCheckAlertsSorting:
    def test_alert_before_warning_before_info(self):
        """Multiple triggers firing — ALERT sorted to front."""
        curr = {
            "decision":   "Risk-On",
            "composite":  62.0,
            "blend":      0.20,
            "shock_flag": "VIX Spike",
            "last_date":  "2026-05-07",
            "last_run":   "2026-05-07 09:00",
        }
        prev = {
            "decision":   "Neutral",
            "composite":  50.0,
            "blend":      0.55,
            "shock_flag": "No Shock",
            "last_date":  "2026-05-06",
            "last_run":   "2026-05-06 09:00",
        }
        cfg = _cfg(composite_spike_pts=5.0, blend_threshold=0.50,
                   composite_stress_threshold=60.0)
        alerts = check_alerts(curr, prev, cfg)
        levels = [a["level"] for a in alerts]
        alert_idx = [i for i, l in enumerate(levels) if l == "ALERT"]
        info_idx   = [i for i, l in enumerate(levels) if l == "INFO"]
        if alert_idx and info_idx:
            assert max(alert_idx) < min(info_idx)

    def test_no_alerts_returns_empty(self):
        same = {**DEFAULT_STATE, "decision": "Neutral",
                "composite": 45.0, "blend": 0.65, "shock_flag": "No Shock"}
        alerts = check_alerts(same, same, _cfg())
        assert alerts == []


# ---------------------------------------------------------------------------
# scorecard history alerts
# ---------------------------------------------------------------------------

class TestScorecardAlerts:
    def _current(self, date="2026-06-02") -> dict:
        return {**DEFAULT_STATE, "last_date": date, "decision": "Neutral", "composite": 40.0}

    def _history(self, rows: list[dict]) -> pd.DataFrame:
        defaults = {
            "as_of": "2026-06-02",
            "recorded_at": "2026-06-02T12:00:00",
            "recommendation": "Hold",
            "net_spread_beta": 0.25,
            "target_net_spread_beta": 0.30,
            "incremental_cdx_hy_protection_pct": 0.0,
            "constraint_breach_count": 0,
            "breached_constraints": "",
            "spread_compensation_ratio": 1.30,
        }
        return pd.DataFrame([{**defaults, **row} for row in rows])

    def _scorecard_cfg(self, **kw) -> AlertConfig:
        return _cfg(
            check_regime_change=False,
            check_blend_below=False,
            check_composite_spike=False,
            check_shock_flag=False,
            check_composite_cross=False,
            check_scorecard_alerts=True,
            **kw,
        )

    def test_missing_history_warns(self):
        alerts = check_scorecard_alerts(self._current(), pd.DataFrame(), self._scorecard_cfg())
        match = next(a for a in alerts if a["trigger"] == "scorecard_history_stale")
        assert match["level"] == "WARNING"

    def test_stale_history_warns(self):
        history = self._history([{"as_of": "2026-06-01"}])
        alerts = check_scorecard_alerts(self._current("2026-06-02"), history, self._scorecard_cfg())
        assert any(a["trigger"] == "scorecard_history_stale" for a in alerts)

    def test_recommendation_change_fires(self):
        history = self._history([
            {"as_of": "2026-06-01", "recommendation": "Hold"},
            {"as_of": "2026-06-02", "recommendation": "Upgrade Quality"},
        ])
        alerts = check_scorecard_alerts(self._current(), history, self._scorecard_cfg())
        match = next(a for a in alerts if a["trigger"] == "scorecard_recommendation_change")
        assert match["level"] == "WARNING"

    def test_defensive_shift_fires_alert(self):
        history = self._history([
            {"as_of": "2026-06-01", "recommendation": "Add"},
            {"as_of": "2026-06-02", "recommendation": "De-risk"},
        ])
        alerts = check_scorecard_alerts(self._current(), history, self._scorecard_cfg())
        assert any(a["trigger"] == "scorecard_defensive_shift" and a["level"] == "ALERT" for a in alerts)

    def test_beta_breach_fires_when_crossing_tolerance(self):
        history = self._history([
            {"as_of": "2026-06-01", "net_spread_beta": 0.32, "target_net_spread_beta": 0.30},
            {"as_of": "2026-06-02", "net_spread_beta": 0.38, "target_net_spread_beta": 0.30},
        ])
        alerts = check_scorecard_alerts(self._current(), history, self._scorecard_cfg(scorecard_beta_tolerance=0.05))
        assert any(a["trigger"] == "scorecard_beta_breach" for a in alerts)

    def test_beta_under_tolerance_does_not_fire_with_single_row(self):
        history = self._history([
            {"as_of": "2026-06-02", "net_spread_beta": 0.29, "target_net_spread_beta": 0.30},
        ])
        alerts = check_scorecard_alerts(self._current(), history, self._scorecard_cfg(scorecard_beta_tolerance=0.05))
        assert not any(a["trigger"] == "scorecard_beta_breach" for a in alerts)

    def test_hedge_gap_fires_when_crossing_threshold(self):
        history = self._history([
            {"as_of": "2026-06-01", "incremental_cdx_hy_protection_pct": 0.0},
            {"as_of": "2026-06-02", "incremental_cdx_hy_protection_pct": 8.0},
        ])
        alerts = check_scorecard_alerts(self._current(), history, self._scorecard_cfg(scorecard_hedge_gap_pct=5.0))
        assert any(a["trigger"] == "scorecard_hedge_gap" for a in alerts)

    def test_constraint_breach_fires_when_new(self):
        history = self._history([
            {"as_of": "2026-06-01", "constraint_breach_count": 0},
            {"as_of": "2026-06-02", "constraint_breach_count": 1, "breached_constraints": "Max HY beta"},
        ])
        alerts = check_scorecard_alerts(self._current(), history, self._scorecard_cfg())
        match = next(a for a in alerts if a["trigger"] == "scorecard_constraint_breach")
        assert match["level"] == "ALERT"

    def test_underpaid_credit_fires_when_crossing_threshold(self):
        history = self._history([
            {"as_of": "2026-06-01", "spread_compensation_ratio": 1.20},
            {"as_of": "2026-06-02", "spread_compensation_ratio": 1.05},
        ])
        alerts = check_scorecard_alerts(self._current(), history, self._scorecard_cfg(scorecard_comp_ratio_min=1.10))
        assert any(a["trigger"] == "scorecard_underpaid_credit" for a in alerts)

    def test_run_alerts_includes_scorecard_history_alerts(self, tmp_path):
        state_path = tmp_path / "state.json"
        history_path = tmp_path / "scorecard_history.csv"
        self._history([
            {"as_of": "2026-06-01", "recommendation": "Hold"},
            {"as_of": "2026-06-02", "recommendation": "De-risk"},
        ]).to_csv(history_path, index=False)
        df = _make_df(date="2026-06-02", final_decision="Neutral")
        result = run_alerts(
            df,
            config=self._scorecard_cfg(dry_run=True),
            state_path=state_path,
            scorecard_history_path=history_path,
        )
        assert any(a["trigger"] == "scorecard_defensive_shift" for a in result["alerts"])


# ---------------------------------------------------------------------------
# format_alert_email
# ---------------------------------------------------------------------------

class TestFormatAlertEmail:
    def _alerts(self) -> list[dict]:
        return [{"trigger": "regime_change", "level": "ALERT",
                 "message": "Regime changed: Neutral → Risk-On",
                 "details": {"prev": "Neutral", "curr": "Risk-On"}}]

    def test_returns_html_string(self):
        html = format_alert_email(self._alerts(), extract_current_state(_make_df()))
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html

    def test_alert_message_present(self):
        html = format_alert_email(self._alerts(), extract_current_state(_make_df()))
        assert "Regime changed" in html

    def test_level_badge_present(self):
        html = format_alert_email(self._alerts(), extract_current_state(_make_df()))
        assert "ALERT" in html

    def test_composite_displayed(self):
        curr = extract_current_state(_make_df(composite_risk_score_smooth=61.5))
        html = format_alert_email(self._alerts(), curr)
        assert "61.5" in html

    def test_handles_no_previous(self):
        html = format_alert_email(self._alerts(), extract_current_state(_make_df()),
                                   previous=None)
        assert isinstance(html, str)

    def test_previous_decision_shown_when_provided(self):
        curr = extract_current_state(_make_df())
        prev = {**DEFAULT_STATE, "decision": "Risk-Off",
                "composite": 58.0, "blend": 0.40, "last_date": "2026-05-06"}
        html = format_alert_email(self._alerts(), curr, previous=prev)
        assert "Risk-Off" in html

    def test_trigger_count_in_banner(self):
        alerts = self._alerts() + [{"trigger": "shock_flag", "level": "ALERT",
                                     "message": "Shock detected", "details": {}}]
        html = format_alert_email(alerts, extract_current_state(_make_df()))
        assert "2 trigger" in html

    def test_shock_flag_highlighted_red_when_active(self):
        curr = extract_current_state(_make_df(shock_flag="VIX Spike"))
        html = format_alert_email(self._alerts(), curr)
        assert "e74c3c" in html


# ---------------------------------------------------------------------------
# send_alert_email
# ---------------------------------------------------------------------------

class TestSendAlertEmail:
    def _alerts(self) -> list[dict]:
        return [{"trigger": "regime_change", "level": "WARNING",
                 "message": "Test alert", "details": {}}]

    def test_raises_on_missing_smtp_host(self):
        cfg = AlertConfig(smtp_host="", smtp_user="u", to_addrs=["r@x.com"])
        with pytest.raises(ValueError, match="SMTP config incomplete"):
            send_alert_email("<html/>", self._alerts(), cfg)

    def test_raises_on_missing_smtp_user(self):
        cfg = AlertConfig(smtp_host="smtp.x.com", smtp_user="", to_addrs=["r@x.com"])
        with pytest.raises(ValueError):
            send_alert_email("<html/>", self._alerts(), cfg)

    def test_raises_on_empty_to_addrs(self):
        cfg = AlertConfig(smtp_host="smtp.x.com", smtp_user="u", to_addrs=[])
        with pytest.raises(ValueError):
            send_alert_email("<html/>", self._alerts(), cfg)

    def test_sends_email_via_smtp(self):
        cfg = _cfg()
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)

        with patch("smtplib.SMTP", return_value=mock_smtp):
            result = send_alert_email("<html>test</html>", self._alerts(), cfg)

        assert result is True
        mock_smtp.sendmail.assert_called_once()

    def test_returns_false_on_smtp_error(self):
        cfg = _cfg()
        with patch("smtplib.SMTP", side_effect=ConnectionRefusedError):
            result = send_alert_email("<html/>", self._alerts(), cfg)
        assert result is False

    def test_subject_includes_top_level(self):
        """Subject line should include the alert level."""
        cfg = _cfg()
        captured_args = {}

        def fake_sendmail(from_addr, to_addrs, msg_str):
            captured_args["msg"] = msg_str

        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)
        mock_smtp.sendmail = fake_sendmail

        with patch("smtplib.SMTP", return_value=mock_smtp):
            send_alert_email("<html/>", self._alerts(), cfg)

        assert "WARNING" in captured_args.get("msg", "")


# ---------------------------------------------------------------------------
# run_alerts (orchestrator)
# ---------------------------------------------------------------------------

class TestRunAlerts:
    def test_empty_df_returns_early(self, tmp_path):
        cfg = _cfg(dry_run=True)
        result = run_alerts(pd.DataFrame(), config=cfg,
                            state_path=tmp_path / "state.json")
        assert result["alerts"] == []
        assert result["email_sent"] is None
        assert result["state_saved"] is False

    def test_no_alerts_when_state_unchanged(self, tmp_path):
        sp = tmp_path / "state.json"
        df = _make_df()
        state = extract_current_state(df)
        save_alert_state(state, sp)

        cfg = _cfg(dry_run=True)
        result = run_alerts(df, config=cfg, state_path=sp)
        assert result["alerts"] == []
        assert result["email_html"] is None

    def test_state_always_saved_after_run(self, tmp_path):
        sp = tmp_path / "state.json"
        cfg = _cfg(dry_run=True)
        run_alerts(_make_df(), config=cfg, state_path=sp)
        assert sp.exists()

    def test_dry_run_does_not_send_email(self, tmp_path):
        sp = tmp_path / "state.json"
        # Seed previous with a different decision so regime_change fires
        save_alert_state({**DEFAULT_STATE, "decision": "Risk-On",
                          "composite": 50.0}, sp)

        cfg = _cfg(dry_run=True)
        with patch("smtplib.SMTP") as mock_smtp:
            result = run_alerts(_make_df(), config=cfg, state_path=sp)

        mock_smtp.assert_not_called()
        assert result["email_sent"] is False
        assert result["email_html"] is not None   # still formatted

    def test_alerts_fired_on_regime_change(self, tmp_path):
        sp = tmp_path / "state.json"
        save_alert_state({**DEFAULT_STATE, "decision": "Neutral",
                          "composite": 42.0}, sp)

        df = _make_df(final_decision="Risk-On",
                      composite_risk_score_smooth=62.0)
        cfg = _cfg(dry_run=True)
        result = run_alerts(df, config=cfg, state_path=sp)
        assert any(a["trigger"] == "regime_change" for a in result["alerts"])

    def test_min_level_filters_info(self, tmp_path):
        sp = tmp_path / "state.json"
        save_alert_state({**DEFAULT_STATE, "decision": "Caution",
                          "composite": 42.0}, sp)

        df = _make_df(final_decision="Neutral",
                      composite_risk_score_smooth=44.0)
        cfg = _cfg(dry_run=True, min_level="WARNING",
                   check_composite_spike=False, check_blend_below=False,
                   check_shock_flag=False, check_composite_cross=False)
        result = run_alerts(df, config=cfg, state_path=sp)
        # regime_change should be INFO (→ "Neutral"), filtered out by min_level
        for a in result["alerts"]:
            assert ALERT_LEVELS[a["level"]] >= ALERT_LEVELS["WARNING"]

    def test_result_keys_present(self, tmp_path):
        cfg = _cfg(dry_run=True)
        result = run_alerts(_make_df(), config=cfg,
                            state_path=tmp_path / "state.json")
        for key in ("alerts", "current", "previous", "email_sent",
                    "email_html", "state_saved"):
            assert key in result

    def test_current_state_in_result(self, tmp_path):
        cfg = _cfg(dry_run=True)
        df = _make_df(final_decision="Risk-Off")
        result = run_alerts(df, config=cfg, state_path=tmp_path / "state.json")
        assert result["current"]["decision"] == "Risk-Off"

    def test_sizing_blend_passed_through(self, tmp_path):
        cfg = _cfg(dry_run=True)
        df = _make_df()
        result = run_alerts(df, config=cfg, sizing=_make_sizing(0.45),
                            state_path=tmp_path / "state.json")
        assert abs(result["current"]["blend"] - 0.45) < 0.001
