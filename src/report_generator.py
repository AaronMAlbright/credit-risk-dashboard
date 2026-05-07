"""
Signal report generator.

Produces a self-contained HTML report from live dashboard data.
All dependencies are stdlib + pandas/numpy — no Jinja2, WeasyPrint, or
other heavy rendering libraries.

Public API
----------
  generate_html_report  — returns an HTML string ready for download
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# Decision color map (matches regime_charts.py)
# ---------------------------------------------------------------------------

_DECISION_COLORS: dict[str, str] = {
    "Buy Stress":                   "#27ae60",
    "Watch Entry":                  "#2ecc71",
    "Hold / Do Not Chase":          "#f1c40f",
    "Neutral":                      "#95a5a6",
    "Wait":                         "#e67e22",
    "Avoid Chasing Risk":           "#e74c3c",
    "Divergence Warning":           "#9b59b6",
    "Credit Warning":               "#c0392b",
    "Stress / Stabilization Watch": "#8e44ad",
}

_CONFIDENCE_COLORS = {
    "Robust":      "#27ae60",
    "Indicative":  "#e67e22",
    "Exploratory": "#e74c3c",
}

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 14px; color: #1a1a2e; background: #f8f9fa;
}
.header {
    background: #1a1a2e; color: white;
    padding: 28px 40px 20px; margin-bottom: 24px;
}
.header h1 { font-size: 22px; font-weight: 700; letter-spacing: 0.3px; }
.header .sub { font-size: 13px; color: #aab; margin-top: 6px; }
.container { max-width: 960px; margin: 0 auto; padding: 0 24px 40px; }
.section { background: white; border-radius: 8px; padding: 20px 24px;
           margin-bottom: 18px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
.section h2 { font-size: 15px; font-weight: 600; color: #1a1a2e;
              border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 14px; }
.badge {
    display: inline-block; padding: 5px 14px; border-radius: 20px;
    font-weight: 600; font-size: 13px; color: white; margin-bottom: 10px;
}
.signal-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin-top: 12px; }
.signal-cell { background: #f8f9fa; border-radius: 6px; padding: 12px 14px; }
.signal-cell .label { font-size: 11px; color: #888; text-transform: uppercase;
                       letter-spacing: 0.5px; margin-bottom: 4px; }
.signal-cell .value { font-size: 14px; font-weight: 600; color: #1a1a2e; }
.conf-badge {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 12px; font-weight: 600; color: white;
}
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; padding: 7px 10px; background: #f1f3f5;
     font-weight: 600; color: #444; font-size: 12px; }
td { padding: 7px 10px; border-bottom: 1px solid #f1f3f5; }
tr:last-child td { border-bottom: none; }
.warn-list { list-style: none; }
.warn-list li { padding: 6px 10px; margin-bottom: 6px;
                background: #fff8e1; border-left: 3px solid #e67e22;
                border-radius: 0 4px 4px 0; font-size: 13px; }
.warn-list li.ok { background: #e8f5e9; border-color: #27ae60; }
.driver-card { display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; }
.driver { background: #f8f9fa; border-radius: 6px; padding: 12px; }
.driver .name { font-size: 12px; color: #666; margin-bottom: 4px; }
.driver .score { font-size: 20px; font-weight: 700; color: #1a1a2e; }
.driver .status { font-size: 11px; margin-top: 4px; }
.elevated { color: #e74c3c; font-weight: 600; }
.normal   { color: #888; }
.neg { color: #e74c3c; }
.pos { color: #27ae60; }
.footer { text-align: center; color: #aaa; font-size: 11px;
          padding: 24px; border-top: 1px solid #e0e0e0; margin-top: 8px; }
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(v, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{float(v):.{decimals}%}"


def _fmt(v, decimals: int = 1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{float(v):.{decimals}f}"


def _color_class(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return "pos" if float(v) > 0 else "neg" if float(v) < 0 else ""


def _safe_str(v, fallback: str = "N/A") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return fallback
    return str(v)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_signal_section(latest: pd.Series) -> str:
    decision = _safe_str(latest.get("final_decision"), "Unknown")
    environment = _safe_str(latest.get("final_environment"), "—")
    action = _safe_str(latest.get("final_action"), "—")
    composite = latest.get("composite_risk_score_smooth")
    date_val = _safe_str(latest.get("date"), "—")

    badge_color = _DECISION_COLORS.get(decision, "#95a5a6")
    composite_str = _fmt(composite) if composite is not None else "—"

    return f"""
    <div class="section">
      <h2>Current Signal</h2>
      <span class="badge" style="background:{badge_color}">{decision}</span>
      <div class="signal-grid">
        <div class="signal-cell">
          <div class="label">Environment</div>
          <div class="value">{environment}</div>
        </div>
        <div class="signal-cell">
          <div class="label">Action</div>
          <div class="value">{action}</div>
        </div>
        <div class="signal-cell">
          <div class="label">Composite Risk Score</div>
          <div class="value">{composite_str} / 100</div>
        </div>
      </div>
      <p style="margin-top:12px;font-size:12px;color:#888">Data as of {date_val}</p>
    </div>"""


def _build_confidence_section(audit: dict | None) -> str:
    if not audit:
        return ""
    summary = audit.get("summary", {})
    overall = summary.get("overall_confidence", "—")
    n_robust = summary.get("n_robust", 0)
    n_ind = summary.get("n_indicative", 0)
    n_exp = summary.get("n_exploratory", 0)
    conf_color = _CONFIDENCE_COLORS.get(overall, "#888")
    warnings = audit.get("warnings", [])

    warn_html = ""
    if warnings:
        items = "".join(f"<li>{w}</li>" for w in warnings)
        warn_html = f"<ul class='warn-list' style='margin-top:12px'>{items}</ul>"
    else:
        warn_html = "<ul class='warn-list' style='margin-top:12px'><li class='ok'>No confidence warnings — all layers meet robust thresholds.</li></ul>"

    return f"""
    <div class="section">
      <h2>Statistical Confidence</h2>
      <span class="conf-badge" style="background:{conf_color}">{overall}</span>
      &nbsp;
      <span style="font-size:12px;color:#888">
        Robust: <b>{n_robust}</b> &nbsp;·&nbsp;
        Indicative: <b>{n_ind}</b> &nbsp;·&nbsp;
        Exploratory: <b>{n_exp}</b>
      </span>
      {warn_html}
    </div>"""


def _build_drivers_section(drivers: list | None) -> str:
    if not drivers:
        return ""
    cards = ""
    for d in drivers[:3]:
        name = d.get("name", "—")
        level = _fmt(d.get("level"))
        excess = d.get("excess", 0.0)
        elevated = d.get("elevated", False)
        status_cls = "elevated" if elevated else "normal"
        status_txt = f"⬆ Elevated (+{excess:.1f} vs 75th pct)" if elevated else f"Normal ({excess:+.1f} vs 75th pct)"
        cards += f"""
        <div class="driver">
          <div class="name">{name}</div>
          <div class="score">{level}</div>
          <div class="status {status_cls}">{status_txt}</div>
        </div>"""

    return f"""
    <div class="section">
      <h2>Top Drivers</h2>
      <div class="driver-card">{cards}</div>
    </div>"""


def _build_decay_section(decay_results: dict | None, current_regime: str | None) -> str:
    if not decay_results or not current_regime:
        return ""
    decay_df = decay_results.get("decay_by_regime")
    if decay_df is None or decay_df.empty:
        return ""
    if current_regime not in decay_df.index.get_level_values("regime"):
        return ""

    grp = decay_df.loc[current_regime].reset_index()
    rows = ""
    for _, row in grp.iterrows():
        h = int(row["horizon"])
        mr = _pct(row.get("mean_return"))
        hr = _pct(row.get("hit_rate"))
        n = int(row.get("n_obs", 0))
        ci_lo = _pct(row.get("ci_lower"))
        ci_hi = _pct(row.get("ci_upper"))
        mr_cls = _color_class(row.get("mean_return"))
        rows += f"""
        <tr>
          <td><b>{h}d</b></td>
          <td class="{mr_cls}">{mr}</td>
          <td>{ci_lo} – {ci_hi}</td>
          <td>{hr}</td>
          <td>{n}</td>
        </tr>"""

    return f"""
    <div class="section">
      <h2>Signal Decay — {current_regime}</h2>
      <p style="font-size:12px;color:#888;margin-bottom:12px">
        Mean forward SP500 return at each holding horizon while in this regime.</p>
      <table>
        <tr><th>Horizon</th><th>Mean Return</th><th>95% CI</th><th>Hit Rate</th><th>N Obs</th></tr>
        {rows}
      </table>
    </div>"""


def _build_tail_section(tail_risk: dict | None, current_regime: str | None) -> str:
    if not tail_risk:
        return ""
    vsb = tail_risk.get("vs_benchmark", {})
    rts = tail_risk.get("regime_tail_stats")

    rows = ""
    if vsb:
        for label, stats in vsb.items():
            disp = "Strategy" if label == "strategy" else "SP500 Buy & Hold"
            cvar = _pct(stats.get("cvar"))
            var = _pct(stats.get("var"))
            vol = _pct(stats.get("ann_vol"))
            ml = _pct(stats.get("max_loss"))
            cvar_cls = _color_class(stats.get("cvar"))
            rows += f"""
            <tr>
              <td><b>{disp}</b></td>
              <td class="{cvar_cls}">{cvar}</td>
              <td>{var}</td>
              <td>{vol}</td>
              <td class="neg">{ml}</td>
            </tr>"""

    regime_row = ""
    if rts is not None and not rts.empty and current_regime and current_regime in rts.index:
        r = rts.loc[current_regime]
        cvar_r = _pct(r.get("cvar_95"))
        var_r = _pct(r.get("var_95"))
        ml_r = _pct(r.get("max_loss"))
        regime_row = f"""
        <tr style="background:#f8f9fa">
          <td><b>{current_regime} (regime)</b></td>
          <td class="neg">{cvar_r}</td>
          <td>{var_r}</td>
          <td>—</td>
          <td class="neg">{ml_r}</td>
        </tr>"""

    return f"""
    <div class="section">
      <h2>Tail Risk (95% CVaR)</h2>
      <table>
        <tr><th>Series</th><th>CVaR 95%</th><th>VaR 95%</th><th>Ann. Vol</th><th>Max Loss</th></tr>
        {rows}
        {regime_row}
      </table>
    </div>"""


def _build_weights_section(weight_opt: dict | None) -> str:
    if not weight_opt:
        return ""
    curr_s = weight_opt.get("current_sharpe")
    opt_s = weight_opt.get("optimal_sharpe")
    opt_w = weight_opt.get("optimal_weights", {})
    from src.weight_optimizer import CURRENT_WEIGHTS

    _DISPLAY = {
        "macro_risk": "Macro Risk", "credit_risk": "Credit Risk",
        "complacency": "Complacency", "liquidity": "Liquidity",
        "treasury": "Treasury", "mean_reversion": "Mean Reversion",
    }

    rows = ""
    for k, label in _DISPLAY.items():
        cw = CURRENT_WEIGHTS.get(k, 0.0)
        ow = opt_w.get(k, float("nan"))
        diff = ow - cw if not np.isnan(ow) else float("nan")
        diff_cls = _color_class(diff)
        diff_str = f"{diff:+.1%}" if not np.isnan(diff) else "—"
        rows += f"""
        <tr>
          <td>{label}</td>
          <td>{cw:.0%}</td>
          <td>{_pct(ow, 1) if not np.isnan(ow) else '—'}</td>
          <td class="{diff_cls}">{diff_str}</td>
        </tr>"""

    sharpe_row = ""
    if curr_s is not None and not np.isnan(curr_s):
        delta = (opt_s - curr_s) if (opt_s is not None and not np.isnan(opt_s)) else float("nan")
        delta_str = f"({delta:+.2f})" if not np.isnan(delta) else ""
        sharpe_row = f"""
        <p style="margin-top:12px;font-size:12px;color:#888">
          In-sample Sharpe — Current: <b>{curr_s:.2f}</b> &nbsp;·&nbsp;
          Best found: <b>{_fmt(opt_s, 2)}</b> {delta_str}
          &nbsp;(2,000 Monte Carlo samples)
        </p>"""

    return f"""
    <div class="section">
      <h2>Composite Weight Context</h2>
      <table>
        <tr><th>Score</th><th>Current</th><th>Optimal (MC)</th><th>Δ</th></tr>
        {rows}
      </table>
      {sharpe_row}
      <p style="margin-top:6px;font-size:11px;color:#bbb">
        Optimal = top-Sharpe sample from Dirichlet grid search. In-sample only — treat as directional.</p>
    </div>"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _build_stress_section(stress: dict | None) -> str:
    if not stress:
        return ""
    stats = stress.get("episode_stats")
    if stats is None or stats.empty:
        return ""

    covered = stats[stats["covered"] == True]
    if covered.empty:
        return ""

    rows = ""
    for name, row in covered.iterrows():
        dd = _pct(row.get("sp500_drawdown"))
        mc = _fmt(row.get("mean_composite"))
        pc = _fmt(row.get("peak_composite"))
        ha = _pct(row.get("high_alert_pct"), 0)
        pd_val = _safe_str(row.get("primary_decision"), "—")
        sev = _safe_str(row.get("severity"), "—")
        rows += f"""
        <tr>
          <td><b>{name}</b></td>
          <td>{sev}</td>
          <td>{mc}</td>
          <td>{pc}</td>
          <td class="neg">{dd}</td>
          <td>{ha}</td>
          <td style="font-size:11px">{pd_val}</td>
        </tr>"""

    n_cov = int(stress.get("n_covered", 0))
    n_hist = int(stress.get("n_historical", 0))

    return f"""
    <div class="section">
      <h2>Stress Episode Behaviour</h2>
      <p style="font-size:12px;color:#888;margin-bottom:12px">
        {n_cov} episode(s) validated against dataset &nbsp;·&nbsp;
        {n_hist} historical reference(s) outside dataset range.
        High Alert = composite ≥ 60.</p>
      <table>
        <tr>
          <th>Episode</th><th>Severity</th><th>Mean Score</th>
          <th>Peak Score</th><th>SP500 DD</th><th>High Alert %</th><th>Primary Decision</th>
        </tr>
        {rows}
      </table>
    </div>"""


def _build_regime_prob_section(regime_prob: dict | None) -> str:
    if not regime_prob:
        return ""
    current = regime_prob.get("current", {})
    probs   = current.get("probs", {})
    if not probs:
        return ""
    top_regime  = _safe_str(current.get("top_regime"), "—")
    top_prob    = current.get("top_prob", 0.0)
    second      = _safe_str(current.get("second"), "")
    second_prob = current.get("second_prob", 0.0)
    entropy     = current.get("entropy", float("nan"))

    sorted_probs = sorted(probs.items(), key=lambda x: -x[1])[:6]
    rows = ""
    for regime, p in sorted_probs:
        bar_w = int(float(p) * 200)
        rows += f"""
        <tr>
          <td style="width:45%">{regime}</td>
          <td style="width:15%">{float(p):.1%}</td>
          <td style="width:40%">
            <div style="background:#e0e0e0;border-radius:3px;height:10px">
              <div style="background:#1f77b4;width:{bar_w}px;height:10px;border-radius:3px"></div>
            </div>
          </td>
        </tr>"""

    entropy_str = f"{float(entropy):.2f} bits" if not np.isnan(float(entropy) if entropy is not None else float("nan")) else "—"
    return f"""
    <div class="section">
      <h2>Regime Probability Nowcast</h2>
      <p style="margin-bottom:10px">
        <b>{top_prob:.0%}</b> probability of
        <span style="font-weight:600;color:#1a1a2e">{top_regime}</span>
        &nbsp;·&nbsp;
        Runner-up: {second} ({second_prob:.0%})
        &nbsp;·&nbsp; Entropy: {entropy_str}
      </p>
      <table><tr><th>Regime</th><th>Probability</th><th></th></tr>{rows}</table>
    </div>"""


def _build_position_sizing_section(sizing: dict | None) -> str:
    if not sizing:
        return ""
    current = sizing.get("current", {})
    if not current:
        return ""
    methods = [
        ("score_sizing",       "Score-Based"),
        ("regime_prob_sizing", "Regime-Prob"),
        ("kelly_sizing",       "Half-Kelly"),
        ("vol_target_sizing",  "Vol-Target"),
        ("blend",              "Blend"),
    ]
    cells = ""
    for key, label in methods:
        val = current.get(key)
        val_str = f"{float(val):.0%}" if val is not None else "—"
        bold = " font-weight:700;" if key == "blend" else ""
        cells += f"""
        <div class="signal-cell" style="{bold}">
          <div class="label">{label}</div>
          <div class="value">{val_str}</div>
        </div>"""

    regime = _safe_str(current.get("current_regime"), "—")
    score  = current.get("current_composite")
    score_str = f"{float(score):.1f}" if score is not None else "—"
    return f"""
    <div class="section">
      <h2>Position Sizing</h2>
      <div class="signal-grid" style="grid-template-columns:repeat(5,1fr)">{cells}</div>
      <p style="margin-top:10px;font-size:12px;color:#888">
        Current regime: <b>{regime}</b> &nbsp;·&nbsp; Composite: <b>{score_str}</b> / 100 &nbsp;·&nbsp;
        Blend = mean of four methods. Half-Kelly and Vol-Target use rolling return history.
      </p>
    </div>"""


def _build_scenario_section(scenario_grid: "pd.DataFrame | None") -> str:
    if scenario_grid is None or (hasattr(scenario_grid, "empty") and scenario_grid.empty):
        return ""
    rows = ""
    for name, row in scenario_grid.iterrows():
        composite = row.get("composite_risk_score_smooth")
        delta     = row.get("delta_composite", 0.0)
        decision  = _safe_str(row.get("final_decision"), "—")
        blend     = row.get("blend_sizing")
        delta_str = f"{float(delta):+.1f}" if delta is not None and not np.isnan(float(delta)) else "—"
        delta_cls = "neg" if (delta is not None and float(delta) > 0) else "pos"
        comp_str  = _fmt(composite)
        blend_str = _pct(blend, 0) if blend is not None else "—"
        rows += f"""
        <tr>
          <td><b>{name}</b></td>
          <td>{comp_str}</td>
          <td class="{delta_cls}">{delta_str}</td>
          <td style="font-size:12px">{decision}</td>
          <td>{blend_str}</td>
        </tr>"""
    return f"""
    <div class="section">
      <h2>Scenario Analysis</h2>
      <p style="font-size:12px;color:#888;margin-bottom:10px">
        Composite scores and positioning under preset market stress scenarios.
        Δ is relative to the zero-shock baseline (instantaneous scoring, no 10d smoothing lag).</p>
      <table>
        <tr><th>Scenario</th><th>Composite</th><th>Δ Composite</th><th>Decision</th><th>Blend Weight</th></tr>
        {rows}
      </table>
    </div>"""


def _build_subperiod_section(subperiod_table: "pd.DataFrame | None") -> str:
    if subperiod_table is None or (hasattr(subperiod_table, "empty") and subperiod_table.empty):
        return ""
    show_cols = ["n_obs", "strat_ann_ret", "strat_sharpe", "strat_max_dd",
                 "sp500_sharpe", "excess_ann_ret", "beta", "ann_alpha"]
    avail = [c for c in show_cols if c in subperiod_table.columns]
    labels = {
        "n_obs": "N", "strat_ann_ret": "Strat Ann Ret", "strat_sharpe": "Sharpe",
        "strat_max_dd": "Max DD", "sp500_sharpe": "SP500 Sharpe",
        "excess_ann_ret": "Excess Ret", "beta": "Beta", "ann_alpha": "Alpha",
    }
    header = "".join(f"<th>{labels.get(c, c)}</th>" for c in avail)
    rows = ""
    for period, row in subperiod_table.iterrows():
        cells = ""
        for c in avail:
            v = row.get(c)
            if c == "n_obs":
                cells += f"<td>{int(v) if v is not None else '—'}</td>"
            elif c in ("strat_ann_ret", "strat_max_dd", "excess_ann_ret", "ann_alpha"):
                cls = _color_class(v)
                cells += f"<td class='{cls}'>{_pct(v, 1)}</td>"
            else:
                cells += f"<td>{_fmt(v, 2)}</td>"
        bold = " font-weight:600" if period == "Full Period" else ""
        rows += f"<tr style='{bold}'><td><b>{period}</b></td>{cells}</tr>"
    return f"""
    <div class="section">
      <h2>Sub-period Attribution</h2>
      <table>
        <tr><th>Period</th>{header}</tr>
        {rows}
      </table>
    </div>"""


def _build_monte_carlo_section(monte_carlo: dict | None) -> str:
    if not monte_carlo:
        return ""
    # summary lives at monte_carlo["simulation"]["summary"]
    summary = monte_carlo.get("simulation", {}).get("summary", {})
    if not summary:
        return ""
    horizons = sorted(summary.keys())
    header = "".join(f"<th>{h}d</th>" for h in horizons)

    def _row(label, key, is_pct=False):
        cells = ""
        for h in horizons:
            v = summary.get(h, {}).get(key)
            if v is None:
                cells += "<td>—</td>"
            elif is_pct:
                cells += f"<td>{float(v):.1%}</td>"
            else:
                cells += f"<td>{_fmt(v, 1)}</td>"
        return f"<tr><td><b>{label}</b></td>{cells}</tr>"

    return f"""
    <div class="section">
      <h2>Monte Carlo Forward Simulation</h2>
      <p style="font-size:12px;color:#888;margin-bottom:10px">
        1,000 Markov-chain paths seeded from current regime probabilities.</p>
      <table>
        <tr><th>Metric</th>{header}</tr>
        {_row("Median Composite",  "median")}
        {_row("P5 (Optimistic)",   "p5")}
        {_row("P95 (Stress)",      "p95")}
        {_row("P(Composite > 50)", "prob_above_50", is_pct=True)}
      </table>
    </div>"""


def generate_html_report(
    df: pd.DataFrame,
    audit: dict | None = None,
    decay_results: dict | None = None,
    tail_risk: dict | None = None,
    weight_opt: dict | None = None,
    attr: dict | None = None,
    stress: dict | None = None,
    regime_prob: dict | None = None,
    position_sizing: dict | None = None,
    scenario_grid: "pd.DataFrame | None" = None,
    subperiod_table: "pd.DataFrame | None" = None,
    monte_carlo: dict | None = None,
) -> str:
    """
    Generate a self-contained HTML signal report.

    Parameters
    ----------
    df              : scored DataFrame (must have 'date' and 'final_decision')
    audit           : output of run_validation_audit  (optional)
    decay_results   : output of run_signal_decay       (optional)
    tail_risk       : output of run_tail_risk_analysis (optional)
    weight_opt      : output of run_weight_optimization (optional)
    attr            : output of run_regime_attribution  (optional)
    stress          : output of run_stress_analysis     (optional)
    regime_prob     : output of run_regime_probability  (optional)
    position_sizing : output of run_position_sizing     (optional)
    scenario_grid   : DataFrame from run_scenario_grid  (optional)
    subperiod_table : DataFrame from compute_subperiod_table (optional)
    monte_carlo     : output of run_monte_carlo         (optional)

    Returns
    -------
    str — complete HTML document
    """
    latest = df.iloc[-1] if not df.empty else pd.Series(dtype=object)
    current_regime = str(latest.get("final_decision", "")) if not df.empty else ""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    last_date = str(latest.get("date", ""))[:10]

    drivers = attr.get("top_drivers_current", []) if attr else []

    sections = "".join([
        _build_signal_section(latest),
        _build_regime_prob_section(regime_prob),
        _build_position_sizing_section(position_sizing),
        _build_confidence_section(audit),
        _build_drivers_section(drivers),
        _build_monte_carlo_section(monte_carlo),
        _build_scenario_section(scenario_grid),
        _build_subperiod_section(subperiod_table),
        _build_stress_section(stress),
        _build_decay_section(decay_results, current_regime),
        _build_tail_section(tail_risk, current_regime),
        _build_weights_section(weight_opt),
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Macro Credit Risk Signal Report — {last_date}</title>
  <style>{_CSS}</style>
</head>
<body>
  <div class="header">
    <h1>Macro Credit Risk Dashboard</h1>
    <div class="sub">Signal Report &nbsp;·&nbsp; {last_date} &nbsp;·&nbsp; Generated {now_str}</div>
  </div>
  <div class="container">
    {sections}
    <div class="footer">
      <p>Generated by Macro Credit Risk Dashboard &nbsp;·&nbsp; {now_str}</p>
      <p style="margin-top:6px">
        For research purposes only. Not investment advice.
        Past performance is not indicative of future results.
      </p>
    </div>
  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

def generate_excel_report(
    df: pd.DataFrame,
    position_sizing: dict | None = None,
    scenario_grid: "pd.DataFrame | None" = None,
    subperiod_table: "pd.DataFrame | None" = None,
    regime_prob: dict | None = None,
    monte_carlo: dict | None = None,
    lookback_days: int = 90,
) -> bytes:
    """
    Generate a multi-sheet Excel workbook from dashboard data.

    Sheets
    ------
    Summary          — current signal snapshot (key metrics)
    Component Scores — last <lookback_days> rows of component score history
    Position Sizing  — last <lookback_days> rows of sizing time series
    Scenario Grid    — preset scenario comparison table
    Attribution      — sub-period performance stats table
    Monte Carlo      — forward simulation percentiles

    Returns
    -------
    bytes — Excel file content (write directly to st.download_button data)
    """
    import io
    import openpyxl  # noqa: F401  (ensures import-time check)

    buf = io.BytesIO()

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:

        # ── Sheet 1: Summary ──────────────────────────────────────────────
        if not df.empty:
            latest = df.iloc[-1]
            summary_rows = [
                ("Date",              str(latest.get("date", ""))[:10]),
                ("Decision",          str(latest.get("final_decision", "—"))),
                ("Environment",       str(latest.get("final_environment", "—"))),
                ("Action",            str(latest.get("final_action", "—"))),
                ("Composite Score",   latest.get("composite_risk_score_smooth")),
                ("Macro Risk",        latest.get("macro_risk_score_smooth")),
                ("Credit Risk",       latest.get("credit_market_risk_score_smooth")),
                ("Liquidity Regime",  latest.get("liquidity_regime_score_smooth")),
                ("Treasury Stress",   latest.get("treasury_stress_score_smooth")),
                ("Complacency",       latest.get("complacency_score_smooth")),
                ("Mean Reversion",    latest.get("mean_reversion_score_smooth")),
                ("VIX",               latest.get("vix")),
                ("HY Spread",         latest.get("hy_spread")),
                ("SP500 Drawdown",    latest.get("sp500_drawdown")),
                ("NFCI",              latest.get("nfci")),
                ("Unemployment",      latest.get("unemployment")),
                ("Model Confidence",  str(latest.get("model_confidence", "—"))),
            ]
            if regime_prob:
                cur = regime_prob.get("current", {})
                summary_rows += [
                    ("Top Regime",       cur.get("top_regime", "—")),
                    ("Top Regime Prob",  cur.get("top_prob")),
                    ("Regime Entropy",   cur.get("entropy")),
                ]
            if position_sizing:
                cur_sz = position_sizing.get("current", {})
                summary_rows += [
                    ("Score Sizing",       cur_sz.get("score_sizing")),
                    ("Regime-Prob Sizing", cur_sz.get("regime_prob_sizing")),
                    ("Half-Kelly Sizing",  cur_sz.get("kelly_sizing")),
                    ("Vol-Target Sizing",  cur_sz.get("vol_target_sizing")),
                    ("Blend Sizing",       cur_sz.get("blend")),
                ]
            pd.DataFrame(summary_rows, columns=["Metric", "Value"]).to_excel(
                writer, sheet_name="Summary", index=False
            )

        # ── Sheet 2: Component Score History ──────────────────────────────
        score_cols = [
            "date",
            "macro_risk_score_smooth", "credit_market_risk_score_smooth",
            "liquidity_regime_score_smooth", "treasury_stress_score_smooth",
            "complacency_score_smooth", "mean_reversion_score_smooth",
            "composite_risk_score_smooth", "final_decision",
        ]
        if not df.empty:
            hist_cols = [c for c in score_cols if c in df.columns]
            df[hist_cols].tail(lookback_days).to_excel(
                writer, sheet_name="Component Scores", index=False
            )

        # ── Sheet 3: Position Sizing ──────────────────────────────────────
        if position_sizing:
            sizes = position_sizing.get("sizes")
            if sizes is not None and not sizes.empty:
                sizes_out = sizes.tail(lookback_days).copy()
                sizes_out.index.name = "date"
                sizes_out.reset_index().to_excel(
                    writer, sheet_name="Position Sizing", index=False
                )
            bt = position_sizing.get("backtest")
            if bt is not None and not bt.empty:
                bt_cols = ["date", "strategy_return", "sp500_return",
                           "cum_full", "cum_blend", "cum_sp500"]
                bt_avail = [c for c in bt_cols if c in bt.columns]
                bt[bt_avail].tail(lookback_days).to_excel(
                    writer, sheet_name="Sized Backtest", index=False
                )

        # ── Sheet 4: Scenario Grid ────────────────────────────────────────
        if scenario_grid is not None and not scenario_grid.empty:
            sg_cols = [
                "composite_risk_score_smooth", "delta_composite",
                "final_decision", "blend_sizing",
                "macro_risk_score_smooth", "credit_market_risk_score_smooth",
                "complacency_score_smooth", "vix", "hy_spread", "sp500_drawdown",
            ]
            sg_avail = [c for c in sg_cols if c in scenario_grid.columns]
            scenario_grid[sg_avail].to_excel(writer, sheet_name="Scenarios")

        # ── Sheet 5: Attribution ──────────────────────────────────────────
        if subperiod_table is not None and not subperiod_table.empty:
            subperiod_table.to_excel(writer, sheet_name="Attribution")

        # ── Sheet 6: Monte Carlo ──────────────────────────────────────────
        if monte_carlo:
            summary_mc = monte_carlo.get("simulation", {}).get("summary", {})
            if summary_mc:
                mc_rows = []
                for h, stats in sorted(summary_mc.items()):
                    mc_rows.append({
                        "Horizon (days)": h,
                        "P5":             stats.get("p5"),
                        "Median":         stats.get("median"),
                        "P95":            stats.get("p95"),
                        "P(>50)":         stats.get("prob_above_50"),
                        "P(>70)":         stats.get("prob_above_70"),
                    })
                pd.DataFrame(mc_rows).to_excel(
                    writer, sheet_name="Monte Carlo", index=False
                )

    buf.seek(0)
    return buf.read()
