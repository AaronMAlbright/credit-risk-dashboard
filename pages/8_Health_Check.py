"""
Health Check — view error monitor, data freshness, and system status.
"""
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Health Check — Credit Risk Dashboard",
    page_icon="🩺",
    layout="wide",
)

from utils.shared import load_data, _ANALYTICS_VIEWS
from src.alert_engine import AlertConfig, check_scorecard_alerts, extract_current_state
from src.credit_compensation_history import load_scorecard_history
from src.data_pipeline import diagnose_refresh_limit, get_pipeline_status
from src.data_sources import source_rows

st.title("🩺 Dashboard Health Check")

# ── Data freshness ───────────────────────────────────────────────────────────
st.subheader("Data Freshness")
try:
    df = load_data()
    _pipeline_status = get_pipeline_status(fast=True)
    _last_date = pd.to_datetime(df["date"]).max() if "date" in df.columns else df.index.max()
    _age_days  = (pd.Timestamp.now() - pd.Timestamp(_last_date)).days
    _nrows, _ncols = df.shape
    _c1, _c2, _c3, _c4, _c5 = st.columns(5)
    _c1.metric("Last data point", str(_last_date.date()))
    _c2.metric("Data age", f"{_age_days} day(s)")
    _c3.metric("Rows", f"{_nrows:,}")
    _c4.metric("Columns", str(_ncols))
    _c5.metric("Trading-day stale", str(_pipeline_status.get("csv_bdays_stale", "-")))
    if _age_days > 3:
        st.warning(f"Data is {_age_days} days old — daily refresh may have failed. Check GitHub Actions.")
    elif _age_days > 1:
        st.info(f"Data is {_age_days} day(s) old (expected if weekend or holiday).")
    else:
        st.success("Data is current.")
    with st.expander("Refresh limit diagnostic", expanded=False):
        if st.button("Run refresh diagnostic", key="health_refresh_diagnostic"):
            with st.spinner("Checking raw source limits..."):
                _diag = diagnose_refresh_limit()
            if not _diag.get("available"):
                st.warning(_diag.get("reason", "Refresh diagnostic unavailable."))
            else:
                _d1, _d2, _d3 = st.columns(3)
                _d1.metric("CSV latest", _diag.get("csv_last_date") or "-")
                _d2.metric("Raw latest", _diag.get("raw_last_date") or "-")
                _d3.metric("Complete core row", _diag.get("latest_complete_core_date") or "-")
                st.caption(f"Status: {_diag.get('reason')}")
                _limits = _diag.get("limiting_columns") or []
                if _limits:
                    st.warning("Limiting columns: " + ", ".join(_limits))
                _col_dates = _diag.get("column_last_dates") or {}
                if _col_dates:
                    st.dataframe(
                        pd.DataFrame(
                            [{"column": col, "last_date": last} for col, last in _col_dates.items()]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
        else:
            st.caption("Run this when the scored CSV is stale and you need to see whether raw source coverage or pipeline output is the limiter.")
except Exception as _de:
    st.error(f"Could not load data: {_de}")

st.divider()

# -- Scorecard alert status ---------------------------------------------------
st.subheader("Scorecard Alert Status")
try:
    _alert_df = load_data()
    _history = load_scorecard_history()
    _current_state = extract_current_state(_alert_df)
    _scorecard_alerts = check_scorecard_alerts(
        _current_state,
        _history,
        AlertConfig(
            check_regime_change=False,
            check_blend_below=False,
            check_composite_spike=False,
            check_shock_flag=False,
            check_composite_cross=False,
            check_scorecard_alerts=True,
            dry_run=True,
        ),
    )
    if _history.empty:
        st.warning("Scorecard history is unavailable.")
    else:
        _latest_hist = _history.tail(1).iloc[0]
        _h1, _h2, _h3, _h4 = st.columns(4)
        _h1.metric("Scorecard date", str(_latest_hist.get("as_of", "-")))
        _h2.metric("Recommendation", str(_latest_hist.get("recommendation", "-")))
        _h3.metric("Net beta", f"{float(_latest_hist.get('net_spread_beta', 0)):.2f}x")
        _h4.metric("CDX gap", f"{float(_latest_hist.get('incremental_cdx_hy_protection_pct', 0)):.1f}% NAV")
    if _scorecard_alerts:
        st.warning(f"{len(_scorecard_alerts)} scorecard alert(s) would fire.")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "level": alert["level"],
                        "trigger": alert["trigger"],
                        "message": alert["message"],
                    }
                    for alert in _scorecard_alerts
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No scorecard alerts would fire.")
except Exception as _sae:
    st.caption(f"Scorecard alert status unavailable: {_sae}")

st.divider()

# -- Data source registry -----------------------------------------------------
st.subheader("Data Source Registry")
_source_df = pd.DataFrame(source_rows())
st.caption("Quality marks distinguish observed series from proxy or synthetic history.")
st.dataframe(
    _source_df,
    use_container_width=True,
    hide_index=True,
    height=320,
)

st.divider()

# ── View error log ───────────────────────────────────────────────────────────
st.subheader("View Error Log (this session)")
_errs = st.session_state.get("_view_errors", {})
if not _errs:
    st.success("No view errors recorded this session. Navigate to analytics sections to run views.")
else:
    import pandas as _pd_hc
    # Build name lookup
    _name_map = {}
    for _sec, _vlist in _ANALYTICS_VIEWS.items():
        for _name, _sid in _vlist:
            _name_map[str(_sid)] = (_sec, _name)
    _rows = []
    for _sid, _emsg in _errs.items():
        _sec, _nm = _name_map.get(_sid, ("Unknown", _sid))
        _rows.append({"Sub ID": _sid, "Section": _sec, "View": _nm, "Error": _emsg})
    _err_df = _pd_hc.DataFrame(_rows)
    st.error(f"{len(_errs)} view(s) encountered errors this session.")
    st.dataframe(_err_df, use_container_width=True, hide_index=True)
    if st.button("Clear error log"):
        st.session_state["_view_errors"] = {}
        st.rerun()

st.divider()

# ── View inventory ───────────────────────────────────────────────────────────
st.subheader("View Inventory")
_inv_rows = []
for _sec, _vlist in _ANALYTICS_VIEWS.items():
    for _name, _sid in _vlist:
        _inv_rows.append({"Section": _sec, "View": _name, "ID": str(_sid),
                          "Status": "Error" if str(_sid) in _errs else "—"})
import pandas as _pd_inv
_inv_df = _pd_inv.DataFrame(_inv_rows)
_total   = len(_inv_df)
_n_errs  = (_inv_df["Status"] == "Error").sum()
st.caption(f"{_total} total views · {_n_errs} errors recorded this session")
st.dataframe(
    _inv_df.style.map(
        lambda v: "color: #ef4444; font-weight: bold" if v == "Error" else "",
        subset=["Status"]
    ),
    use_container_width=True,
    hide_index=True,
    height=400,
)

st.divider()

# ── Session state summary ─────────────────────────────────────────────────────
st.subheader("Session State")
_ss_keys = {k: type(v).__name__ for k, v in st.session_state.items()}
if _ss_keys:
    import pandas as _pd_ss
    st.dataframe(_pd_ss.DataFrame(list(_ss_keys.items()), columns=["Key", "Type"]),
                 use_container_width=True, hide_index=True, height=200)
else:
    st.caption("Session state is empty.")
