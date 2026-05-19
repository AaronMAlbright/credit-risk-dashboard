"""
Streamlit renderer for the institutional credit view.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_credit_view_section(credit_view: dict) -> None:
    brief = credit_view["brief"]

    if brief.get("available"):
        st.markdown(f"### {brief['headline']}")
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Regime", brief["regime"])
        b2.metric("Risk Score", f"{brief['score']:.1f}" if brief["score"] is not None else "n/a")
        b3.metric("Credit Beta", brief["credit_beta"])
        b4.metric("HY Tilt", brief["hy_tilt"])
        b5.metric("Valuation", brief["valuation"] or "n/a")

        st.download_button(
            "Download Credit Brief",
            data=credit_view["brief_markdown"],
            file_name="credit_brief.md",
            mime="text/markdown",
        )
        st.download_button(
            "Download Tear Sheet",
            data=credit_view["tearsheet_markdown"],
            file_name="credit_market_tearsheet.md",
            mime="text/markdown",
        )
    else:
        st.warning(brief.get("headline", "Credit brief unavailable."))

    st.subheader("Strategy Memo")
    st.code(credit_view["memo"], language="text")

    _cv1, _cv2 = st.columns([1, 1])

    with _cv1:
        st.subheader("Credit Positioning")
        pos_tbl = credit_view["positioning_table"]
        if isinstance(pos_tbl, pd.DataFrame) and not pos_tbl.empty:
            st.dataframe(pos_tbl, use_container_width=True, hide_index=True)
        else:
            st.info("Positioning unavailable.")

    with _cv2:
        st.subheader("Spread Compensation")
        spread = credit_view["spread"]
        if spread.get("available"):
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("HY OAS", f"{spread['spread_oas_bps']:.0f} bps")
            sc2.metric("Expected Loss", f"{spread['expected_loss_bps']:.0f} bps")
            sc3.metric("Excess Spread", f"{spread['excess_spread_bps']:.0f} bps")
            st.caption(
                f"PD {spread['default_probability']:.1%} · "
                f"Recovery {spread['recovery_rate']:.0%} · "
                f"Valuation: {spread['spread_valuation']}"
            )
        else:
            st.info(spread.get("reason", "Spread decomposition unavailable."))

    st.subheader("Credit Risk Channels")
    channels = credit_view["channels"]
    if channels.get("available"):
        channel_df = pd.DataFrame(channels["channels"])
        display_cols = ["name", "weight", "score", "coverage", "description"]
        st.dataframe(
            channel_df[display_cols].rename(
                columns={
                    "name": "Channel",
                    "weight": "Weight",
                    "score": "Score",
                    "coverage": "Coverage",
                    "description": "Description",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Institutional channel composite: {channels['composite']:.1f} / 100")
    else:
        st.info("Channel scores unavailable. Run a fresh pipeline to persist the new channel columns.")

    st.subheader("Channel Contribution Attribution")
    contrib = credit_view["channel_contributions"]
    if isinstance(contrib, pd.DataFrame) and not contrib.empty:
        contrib_disp = contrib.copy()
        for col in ["weight", "coverage", "contribution_share"]:
            if col in contrib_disp.columns:
                contrib_disp[col] = contrib_disp[col].map(lambda x: None if pd.isna(x) else f"{x:.0%}")
        for col in ["score", "contribution", "score_change_1m", "contribution_change_1m"]:
            if col in contrib_disp.columns:
                contrib_disp[col] = contrib_disp[col].map(lambda x: None if pd.isna(x) else f"{x:.1f}")
        st.dataframe(
            contrib_disp[
                [
                    "channel",
                    "weight",
                    "score",
                    "contribution",
                    "contribution_share",
                    "score_change_1m",
                    "coverage",
                ]
            ].rename(
                columns={
                    "channel": "Channel",
                    "weight": "Weight",
                    "score": "Score",
                    "contribution": "Contribution",
                    "contribution_share": "Share",
                    "score_change_1m": "1M Change",
                    "coverage": "Coverage",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        drivers = credit_view["top_channel_drivers"]
        if drivers:
            st.caption("Top drivers: " + " · ".join(drivers))
    else:
        st.info("Channel contribution attribution unavailable.")

    st.subheader("Credit Relative Value")
    rv_tbl = credit_view["relative_value_table"]
    rv = credit_view["relative_value"]
    if isinstance(rv_tbl, pd.DataFrame) and not rv_tbl.empty:
        rv_disp = rv_tbl.copy()
        if "level" in rv_disp.columns:
            rv_disp["level"] = rv_disp["level"].map(lambda x: None if pd.isna(x) else f"{x:.2f}")
        if "percentile" in rv_disp.columns:
            rv_disp["percentile"] = rv_disp["percentile"].map(lambda x: None if pd.isna(x) else f"{x:.0f}")
        st.dataframe(
            rv_disp.rename(
                columns={
                    "metric": "Metric",
                    "level": "Level",
                    "percentile": "Percentile",
                    "valuation": "Valuation",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        if rv.get("quality_tilt"):
            st.caption(f"Quality tilt: {rv['quality_tilt']}")
    else:
        st.info("Relative-value table unavailable. IG/BBB columns may be missing from the current dataset.")

    st.subheader("Credit Market Tear Sheet")
    ts = credit_view["tearsheet"]
    if isinstance(ts, pd.DataFrame) and not ts.empty:
        ts_disp = ts.copy()
        for col in ["level", "percentile", "change_1m", "change_3m"]:
            if col in ts_disp.columns:
                ts_disp[col] = ts_disp[col].map(lambda x: None if pd.isna(x) else f"{x:.1f}")
        st.dataframe(
            ts_disp.rename(
                columns={
                    "metric": "Metric",
                    "level": "Level",
                    "percentile": "Percentile",
                    "change_1m": "1M Change",
                    "change_3m": "3M Change",
                    "valuation": "Valuation",
                    "action": "Action",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Credit tear sheet unavailable.")

    st.subheader("Rating Bucket Proxy View")
    rb = credit_view["rating_bucket_proxy"]
    if isinstance(rb, pd.DataFrame) and not rb.empty:
        rb_disp = rb.copy()
        for col in ["level", "change_1m"]:
            if col in rb_disp.columns:
                rb_disp[col] = rb_disp[col].map(lambda x: None if pd.isna(x) else f"{x:.1f}")
        st.caption(credit_view["rating_bucket_summary"])
        st.dataframe(
            rb_disp.rename(
                columns={
                    "bucket": "Bucket",
                    "proxy": "Proxy",
                    "level": "Level",
                    "change_1m": "1M Change",
                    "regime": "Regime",
                    "interpretation": "Interpretation",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Refinancing Wall Framework")
    st.caption(credit_view["refinancing_wall_summary"])
    refi = credit_view["refinancing_wall"]
    if isinstance(refi, pd.DataFrame) and not refi.empty:
        refi_disp = refi.copy()
        if "share" in refi_disp.columns:
            refi_disp["share"] = refi_disp["share"].map(lambda x: None if pd.isna(x) else f"{x:.0%}")
        st.dataframe(
            refi_disp.rename(
                columns={
                    "bucket": "Maturity Bucket",
                    "amount": "Amount",
                    "share": "Share",
                    "risk_note": "Risk Note",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Regime-Conditioned Forward Performance")
    perf = credit_view["regime_performance"]
    if isinstance(perf, pd.DataFrame) and not perf.empty:
        perf_disp = perf.copy()
        for col in ["avg_sp500_return", "hit_rate_sp500_positive", "worst_decile_sp500_return", "prob_hy_widening", "prob_ig_widening"]:
            if col in perf_disp.columns:
                perf_disp[col] = perf_disp[col].map(lambda x: None if pd.isna(x) else f"{x:.1%}")
        for col in ["avg_hy_spread_change", "avg_ig_spread_change"]:
            if col in perf_disp.columns:
                perf_disp[col] = perf_disp[col].map(lambda x: None if pd.isna(x) else f"{x:.2f}")
        st.dataframe(perf_disp, use_container_width=True, hide_index=True)

        note = credit_view["current_regime_note"]
        if note.get("available"):
            st.caption(
                f"Current regime historical sample: n={note['n_obs']} "
                f"({note['confidence']}) over {note['horizon_days']} trading days."
            )
    else:
        st.info("Regime performance table unavailable.")

    st.subheader("Channel Forward Validation")
    ch_val = credit_view["channel_validation"]
    if isinstance(ch_val, pd.DataFrame) and not ch_val.empty:
        ch_val_disp = ch_val.copy()
        for col in ["score_threshold", "avg_hy_spread_change", "avg_ig_spread_change"]:
            if col in ch_val_disp.columns:
                ch_val_disp[col] = ch_val_disp[col].map(lambda x: None if pd.isna(x) else f"{x:.2f}")
        for col in ["avg_sp500_return", "worst_5pct_sp500_return", "hit_rate_sp500_down", "hit_rate_hy_widening", "hit_rate_ig_widening"]:
            if col in ch_val_disp.columns:
                ch_val_disp[col] = ch_val_disp[col].map(lambda x: None if pd.isna(x) else f"{x:.1%}")
        st.dataframe(
            ch_val_disp.rename(
                columns={
                    "channel": "Channel",
                    "horizon_days": "Horizon",
                    "score_threshold": "Score Threshold",
                    "observations": "Observations",
                    "confidence": "Confidence",
                    "avg_sp500_return": "Avg SP500 Return",
                    "worst_5pct_sp500_return": "Worst 5%",
                    "avg_hy_spread_change": "Avg HY Δ",
                    "avg_ig_spread_change": "Avg IG Δ",
                    "hit_rate_sp500_down": "SP500 Down Hit Rate",
                    "hit_rate_hy_widening": "HY Widen Hit Rate",
                    "hit_rate_ig_widening": "IG Widen Hit Rate",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        snap = credit_view["channel_validation_snapshot"]
        if snap.get("available"):
            st.caption(
                "Validation summary: "
                + ", ".join(
                    f"{row.channel}: n={int(row.observations)}"
                    for row in snap["summary"].itertuples()
                )
            )
    else:
        st.info("Channel validation unavailable.")

    with st.expander("Framework Assumptions And Glossary", expanded=False):
        a1, a2 = st.columns([1, 1])
        with a1:
            st.markdown("**Key Assumptions**")
            st.dataframe(credit_view["assumptions"], use_container_width=True, hide_index=True)
        with a2:
            st.markdown("**Credit Glossary**")
            st.dataframe(credit_view["glossary"], use_container_width=True, hide_index=True)

    with st.expander("Model Governance", expanded=False):
        st.markdown(credit_view["governance_markdown"])
        st.markdown("**Production vs Research Status**")
        st.dataframe(credit_view["governance_status"], use_container_width=True, hide_index=True)
        st.markdown("**Known Limitations**")
        st.dataframe(credit_view["limitations"], use_container_width=True, hide_index=True)
        st.markdown("**Institutional Data Needed For Production Grade**")
        st.dataframe(credit_view["institutional_data"], use_container_width=True, hide_index=True)

    with st.expander("Model Spec", expanded=False):
        st.markdown(credit_view["model_spec_markdown"])
        st.markdown("**Channel Specification**")
        st.dataframe(credit_view["model_spec"], use_container_width=True, hide_index=True)
        st.markdown("**Validation Boundary**")
        st.dataframe(credit_view["validation_boundary"], use_container_width=True, hide_index=True)
        st.markdown("**Limitations**")
        st.dataframe(credit_view["model_spec_limitations"], use_container_width=True, hide_index=True)
        st.markdown("**Institutional Data Needed**")
        st.dataframe(credit_view["model_spec_institutional_data"], use_container_width=True, hide_index=True)

    with st.expander("Composite Comparison And Case Studies", expanded=False):
        comp = credit_view["composite_comparison"]
        st.caption(credit_view["composite_note"])
        if comp.get("available"):
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Correlation", f"{comp['correlation']:.2f}")
            cc2.metric("Latest Gap", f"{comp['latest_gap']:+.1f}")
            cc3.metric("Mean Abs Gap", f"{comp['mean_abs_gap']:.1f}")
            cc4.metric("Large Gaps", f"{comp['large_gap_count']}")
            gaps = comp.get("largest_gaps")
            if isinstance(gaps, pd.DataFrame) and not gaps.empty:
                st.dataframe(gaps, use_container_width=True, hide_index=True)
        cases = credit_view["case_studies"]
        if isinstance(cases, pd.DataFrame) and not cases.empty:
            cases_disp = cases.copy()
            if "sp500_return" in cases_disp.columns:
                cases_disp["sp500_return"] = cases_disp["sp500_return"].map(lambda x: None if pd.isna(x) else f"{x:.1%}")
            for col in ["peak_score", "hy_spread_change"]:
                if col in cases_disp.columns:
                    cases_disp[col] = cases_disp[col].map(lambda x: None if pd.isna(x) else f"{x:.1f}")
            st.markdown("**Historical Episode Case Studies**")
            st.dataframe(cases_disp, use_container_width=True, hide_index=True)
