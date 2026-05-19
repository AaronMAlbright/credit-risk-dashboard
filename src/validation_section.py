"""
Streamlit renderer for the walk-forward validation page.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.validation_guard import (
    CONFIDENCE_EXPLORATORY,
    CONFIDENCE_INDICATIVE,
    CONFIDENCE_ROBUST,
    CONFIDENCE_SIGILS,
)


def _sample_flag(n: int) -> str:
    if n < 20:
        return "Exploratory"
    if n < 50:
        return "Indicative"
    return "Reliable"


def render_validation_section(
    df: pd.DataFrame,
    wf_windows: pd.DataFrame | None,
    wf_regimes: pd.DataFrame | None,
    load_regime_transition,
    load_validation_audit,
    load_bootstrap,
    check_missing_values,
    check_score_bounds,
    check_sample_sizes,
    _cfg_equity_floor: int,
    _cfg_equity_cap: int,
    _cfg_target_vol: int,
    _cfg_momentum_lookback: int,
):
    st.header("Walk-Forward Validation")

    _regime_results = load_regime_transition(df)
    _trans_counts = _regime_results.get("transition_counts") if _regime_results else None
    _audit = load_validation_audit(df, wf_windows, _trans_counts)
    _summary = _audit.get("summary", {})
    _overall = _summary.get("overall_confidence", CONFIDENCE_EXPLORATORY)

    _badge_color = {
        CONFIDENCE_ROBUST: "green",
        CONFIDENCE_INDICATIVE: "orange",
        CONFIDENCE_EXPLORATORY: "red",
    }.get(_overall, "gray")

    st.markdown(
        f"**System Confidence:** :{_badge_color}[{CONFIDENCE_SIGILS[_overall]} {_overall}]  "
        f"&nbsp;&nbsp; Robust: **{_summary.get('n_robust', 0)}** · "
        f"Indicative: **{_summary.get('n_indicative', 0)}** · "
        f"Exploratory: **{_summary.get('n_exploratory', 0)}**"
    )

    _warnings = _audit.get("warnings", [])
    if _warnings:
        with st.expander(f"Confidence Warnings ({len(_warnings)})", expanded=(_overall == CONFIDENCE_EXPLORATORY)):
            for w in _warnings:
                st.warning(w)
    else:
        st.success("No confidence warnings. All analytical layers meet robust thresholds.")

    with st.expander("Regime Confidence Detail", expanded=False):
        _regime_stats_audit = _audit.get("regime_stats")
        _trans_audit = _audit.get("transition_matrix")
        _wf_audit = _audit.get("walk_forward", {})
        _corr_audit = _audit.get("correlations")

        if "grouped_regime" in df.columns and "sp500_forward_30d_return" in df.columns:
            _grp = (
                df.groupby("grouped_regime")["sp500_forward_30d_return"]
                .agg(n="count", mean_fwd=("mean"), hit_rate=lambda x: (x > 0).mean())
                .sort_values("mean_fwd")
                .rename(columns={"n": "N Obs", "mean_fwd": "Mean Fwd 30d", "hit_rate": "Hit Rate"})
            )
            _grp["Sample Flag"] = _grp["N Obs"].apply(_sample_flag)
            _ORDER = ["Risk-On", "Neutral", "Caution", "Risk-Off"]
            _grp = _grp.reindex([r for r in _ORDER if r in _grp.index])
            st.markdown("**Regime Forward Returns (4 regimes)**")
            st.caption("Sample reliability: Exploratory=n<20 · Indicative=n<50 · Reliable=n≥50")
            st.dataframe(_grp.style.format({"Mean Fwd 30d": "{:.2%}", "Hit Rate": "{:.0%}"}), use_container_width=True)

        if _regime_stats_audit is not None and not _regime_stats_audit.empty:
            with st.expander("Detailed regime stats"):
                st.caption("Sample reliability: Exploratory = n<20 · Indicative = n<50 · Reliable = n≥50")
                _rs_disp = _regime_stats_audit[["n_obs", "mean_return", "hit_rate", "confidence"]].copy()
                _rs_disp["mean_return"] = _rs_disp["mean_return"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
                _rs_disp["hit_rate"] = _rs_disp["hit_rate"].map(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
                _rs_disp["confidence"] = _rs_disp["confidence"].map(lambda c: f"{CONFIDENCE_SIGILS[c]} {c}")
                _rs_disp["Sample Flag"] = _regime_stats_audit["n_obs"].apply(_sample_flag)
                _rs_disp.columns = ["N Obs", "Mean Return", "Hit Rate", "Confidence", "Sample Flag"]
                st.dataframe(_rs_disp, use_container_width=True)

                if _trans_audit is not None and not _trans_audit.empty:
                    st.markdown("**Transition Confidence by From-Regime**")
                    _td_disp = _trans_audit[["n_outgoing", "confidence"]].copy()
                    _td_disp["confidence"] = _td_disp["confidence"].map(lambda c: f"{CONFIDENCE_SIGILS[c]} {c}")
                    _td_disp.columns = ["N Outgoing", "Confidence"]
                    st.dataframe(_td_disp, use_container_width=True)

                if _corr_audit is not None and not _corr_audit.empty:
                    st.markdown("**Signal–Return Correlations**")
                    _ca_disp = _corr_audit[["correlation", "n_obs", "informative", "confidence"]].copy()
                    _ca_disp["correlation"] = _ca_disp["correlation"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
                    _ca_disp["informative"] = _ca_disp["informative"].map(lambda x: "Yes" if x else "No")
                    _ca_disp["confidence"] = _ca_disp["confidence"].map(lambda c: f"{CONFIDENCE_SIGILS[c]} {c}")
                    _ca_disp.columns = ["Correlation", "N Obs", "Informative", "Confidence"]
                    st.dataframe(_ca_disp, use_container_width=True)

        _wf_conf = _wf_audit.get("confidence", CONFIDENCE_EXPLORATORY)
        st.markdown(f"**Walk-Forward:** {CONFIDENCE_SIGILS[_wf_conf]} {_wf_conf} ({_wf_audit.get('n_windows', 0)} windows)")

    st.divider()

    st.subheader("Regime Persistence")
    st.caption(
        "How many days does the model typically stay in each regime before transitioning? "
        "Regimes with very short mean duration (< 3 days) are likely label noise — "
        "treat their forward-return stats with extra caution even when n-obs is adequate."
    )

    import plotly.graph_objects as _pers_go

    _fd_results = _regime_results.get("final_decision", {}) if _regime_results else {}
    _dur = _fd_results.get("durations")
    _fwd = _fd_results.get("forward_returns")

    if _dur is not None and not _dur.empty:
        _dur_sorted = _dur.sort_values("mean_days", ascending=True)
        _pers_colors = ["#e74c3c" if v < 3 else "#f39c12" if v < 10 else "#27ae60" for v in _dur_sorted["mean_days"]]
        _p_col1, _p_col2 = st.columns([3, 2])

        with _p_col1:
            st.caption("Mean episode duration by regime (trading days) — error bars = ±1 SE")
            _fig_pers = _pers_go.Figure(_pers_go.Bar(
                y=_dur_sorted.index.tolist(),
                x=_dur_sorted["mean_days"].round(1).tolist(),
                orientation="h",
                marker_color=_pers_colors,
                error_x=dict(
                    type="data",
                    array=(_dur_sorted["std_days"] / _dur_sorted["count"].pow(0.5)).round(1).tolist(),
                    visible=True,
                    color="rgba(200,200,200,0.35)",
                ),
                hovertemplate="%{y}<br>Mean: %{x:.1f}d · Episodes: %{customdata}<extra></extra>",
                customdata=_dur_sorted["count"].tolist(),
            ))
            _fig_pers.add_vline(
                x=3, line_color="rgba(231,76,60,0.45)", line_dash="dot", line_width=1.5,
                annotation_text="Noise threshold",
                annotation_font=dict(color="rgba(231,76,60,0.65)", size=9),
                annotation_position="top right",
            )
            _fig_pers.update_layout(
                height=max(240, len(_dur_sorted) * 34 + 60),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                margin=dict(l=8, r=8, t=8, b=8),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="Mean Duration (trading days)"),
                yaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig_pers, use_container_width=True)

        with _p_col2:
            st.caption("Persistence vs. signal reliability")
            _dur_tbl = _dur[["mean_days", "count"]].copy()
            if _fwd is not None and not _fwd.empty:
                _fwd_col = _fwd.get("sp500_forward_30d_return", pd.Series(dtype=float))
                _dur_tbl = _dur_tbl.join(_fwd_col.rename("mean_fwd_30d"), how="left")
            _dur_tbl["reliability"] = _dur_tbl["mean_days"].apply(lambda d: "Stable" if d >= 10 else ("Moderate" if d >= 3 else "Noisy"))
            _dur_tbl = _dur_tbl.sort_values("mean_days", ascending=False)
            _dur_tbl.index.name = "Regime"
            _fmt = {"mean_days": "{:.1f}", "count": "{:.0f}"}
            if "mean_fwd_30d" in _dur_tbl.columns:
                _fmt["mean_fwd_30d"] = "{:.2%}"
                _dur_tbl.columns = ["Mean Days", "Episodes", "Mean Fwd 30D", "Reliability"]
            else:
                _dur_tbl.columns = ["Mean Days", "Episodes", "Reliability"]
            st.dataframe(_dur_tbl.style.format(_fmt), use_container_width=True)
            st.caption("🟢 Stable ≥10d · 🟡 Moderate 3–10d · 🔴 Noisy <3d")
    else:
        st.info("Run `python app.py` to generate regime transition data.")

    st.divider()

    if wf_windows is None:
        st.warning("No walk-forward results found. Run `python app.py` to generate them.")
    else:
        n_windows = len(wf_windows)
        strategy_beat = (wf_windows["strategy_sharpe"] > wf_windows["sp500_sharpe"]).sum()
        avg_strat_sharpe = wf_windows["strategy_sharpe"].mean()
        avg_sp500_sharpe = wf_windows["sp500_sharpe"].mean()
        avg_strat_return = wf_windows["strategy_total_return"].mean()
        avg_sp500_return = wf_windows["sp500_total_return"].mean()
        avg_strat_dd = wf_windows["strategy_max_drawdown"].mean()
        avg_sp500_dd = wf_windows["sp500_max_drawdown"].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test Windows", n_windows)
        col2.metric("Windows: Strategy Beat SP500", f"{strategy_beat} / {n_windows}")
        col3.metric("Avg Strategy Sharpe", f"{avg_strat_sharpe:.2f}")
        col4.metric("Avg SP500 Sharpe", f"{avg_sp500_sharpe:.2f}")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Avg Strategy Return", f"{avg_strat_return:.1%}")
        col6.metric("Avg SP500 Return", f"{avg_sp500_return:.1%}")
        col7.metric("Avg Strategy Max Drawdown", f"{avg_strat_dd:.1%}")
        col8.metric("Avg SP500 Max Drawdown", f"{avg_sp500_dd:.1%}")

        st.subheader("Sharpe Ratio by Window")
        sharpe_chart = wf_windows[["test_start", "strategy_sharpe", "sp500_sharpe"]].copy()
        sharpe_chart["test_start"] = sharpe_chart["test_start"].dt.strftime("%b %Y")
        st.bar_chart(sharpe_chart.set_index("test_start"), color=["#4C9BE8", "#E8834C"])

        st.subheader("Per-Window Performance")
        disp = wf_windows[[
            "window_id", "test_start", "test_end", "dominant_regime",
            "strategy_total_return", "strategy_sharpe", "strategy_max_drawdown",
            "sp500_total_return", "sp500_sharpe", "sp500_max_drawdown",
        ]].copy()
        disp["test_start"] = disp["test_start"].dt.strftime("%Y-%m-%d")
        disp["test_end"] = disp["test_end"].dt.strftime("%Y-%m-%d")
        for c in ["strategy_total_return", "strategy_max_drawdown", "sp500_total_return", "sp500_max_drawdown"]:
            disp[c] = disp[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        for c in ["strategy_sharpe", "sp500_sharpe"]:
            disp[c] = disp[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        st.dataframe(disp, use_container_width=True, hide_index=True)

        st.subheader("Regime Breakdown")
        rd = wf_regimes[[
            "window_id", "test_start", "regime", "n_obs",
            "avg_forward_30d_return", "hit_rate", "worst_5pct", "avg_equity_weight",
        ]].copy()
        rd["test_start"] = rd["test_start"].dt.strftime("%Y-%m-%d")
        for c in ["avg_forward_30d_return", "worst_5pct"]:
            rd[c] = rd[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        rd["hit_rate"] = rd["hit_rate"].map(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
        rd["avg_equity_weight"] = rd["avg_equity_weight"].map(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
        st.dataframe(rd, use_container_width=True, hide_index=True)

    st.subheader("Validation Dataset")
    validation_cols = [
        "date",
        "macro_risk_score_smooth",
        "credit_market_risk_score_smooth",
        "liquidity_regime_score_smooth",
        "complacency_score_smooth",
        "mean_reversion_score_smooth",
        "composite_risk_score_smooth",
        "composite_risk_label",
        "final_decision",
        "sp500_forward_30d_return",
        "sp500_forward_60d_return",
        "hy_forward_30d_change",
        "sp500_future_drawdown_30d",
    ]
    existing_cols = [c for c in validation_cols if c in df.columns]
    st.dataframe(df[existing_cols].tail(100), use_container_width=True)

    st.subheader("Correlation Snapshot")
    corr_cols = [
        "macro_risk_score_smooth",
        "credit_market_risk_score_smooth",
        "liquidity_regime_score_smooth",
        "complacency_score_smooth",
        "mean_reversion_score_smooth",
        "composite_risk_score_smooth",
        "sp500_forward_30d_return",
        "sp500_forward_60d_return",
        "hy_forward_30d_change",
        "sp500_future_drawdown_30d",
    ]
    available = [c for c in corr_cols if c in df.columns]
    st.dataframe(df[available].corr(), use_container_width=True)

    with st.expander("Bootstrap Confidence Intervals (95%)", expanded=False):
        st.caption(
            "Percentile bootstrap with 1,000 resamples. "
            "⚠️ = flagged: fewer than 30 obs for regime metrics, "
            "fewer than 5 transitions for probability cells, "
            "fewer than 10 windows for walk-forward metrics."
        )

        boot = load_bootstrap(df, wf_windows)

        def _flag(val):
            return "⚠️" if val else ""

        def _fmt_ci_row(mean, lo, hi, pct=True):
            if pd.isna(lo):
                return f"{mean:.2%}" if pct else f"{mean:.3f}", "—", "—"
            if pct:
                return f"{mean:.2%}", f"{lo:.2%}", f"{hi:.2%}"
            return f"{mean:.3f}", f"{lo:.3f}", f"{hi:.3f}"

        wm = boot.get("window_metrics", pd.DataFrame())
        if not wm.empty:
            st.markdown("**Walk-Forward Window Metrics**")
            pct_metrics = {"strategy_total_return", "sp500_total_return", "strategy_max_drawdown", "sp500_max_drawdown", "strategy_hit_rate"}
            rows = []
            for metric, row in wm.iterrows():
                pct = metric in pct_metrics
                m, lo, hi = _fmt_ci_row(row["mean"], row.get("ci_lower"), row.get("ci_upper"), pct)
                rows.append({"Metric": metric, "Mean": m, "CI Lower": lo, "CI Upper": hi, "N Windows": int(row["n_windows"]), "": _flag(row["flagged"])})
            st.dataframe(pd.DataFrame(rows).set_index("Metric"), use_container_width=True)

        rr = boot.get("regime_returns", pd.DataFrame())
        if not rr.empty:
            st.markdown("**Mean 30D Forward Return by Regime**")
            rows = []
            for regime, row in rr.iterrows():
                m, lo, hi = _fmt_ci_row(row["mean"], row.get("ci_lower"), row.get("ci_upper"), pct=True)
                rows.append({"Regime": regime, "Mean": m, "CI Lower": lo, "CI Upper": hi, "N Obs": int(row["n_obs"]), "": _flag(row["flagged"])})
            st.dataframe(pd.DataFrame(rows).set_index("Regime"), use_container_width=True)

        hr = boot.get("regime_hit_rates", pd.DataFrame())
        if not hr.empty:
            st.markdown("**Hit Rate (% Positive 30D Returns) by Regime**")
            rows = []
            for regime, row in hr.iterrows():
                m, lo, hi = _fmt_ci_row(row["hit_rate"], row.get("ci_lower"), row.get("ci_upper"), pct=True)
                rows.append({"Regime": regime, "Hit Rate": m, "CI Lower": lo, "CI Upper": hi, "N Obs": int(row["n_obs"]), "": _flag(row["flagged"])})
            st.dataframe(pd.DataFrame(rows).set_index("Regime"), use_container_width=True)

        tp = boot.get("transition_probs_final_decision", pd.DataFrame())
        if not tp.empty:
            st.markdown("**Transition Probability CIs (Final Decision)**")
            rows = []
            for (frm, to), row in tp.iterrows():
                m, lo, hi = _fmt_ci_row(row["mean_prob"], row.get("ci_lower"), row.get("ci_upper"), pct=True)
                rows.append({"From": frm, "To": to, "Prob": m, "CI Lower": lo, "CI Upper": hi, "N Transitions": int(row["n_obs"]), "": _flag(row["flagged"])})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
