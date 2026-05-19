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

st.title("🩺 Dashboard Health Check")

# ── Data freshness ───────────────────────────────────────────────────────────
st.subheader("Data Freshness")
try:
    df = load_data()
    _last_date = pd.to_datetime(df["date"]).max() if "date" in df.columns else df.index.max()
    _age_days  = (pd.Timestamp.now() - pd.Timestamp(_last_date)).days
    _nrows, _ncols = df.shape
    _c1, _c2, _c3, _c4 = st.columns(4)
    _c1.metric("Last data point", str(_last_date.date()))
    _c2.metric("Data age", f"{_age_days} day(s)")
    _c3.metric("Rows", f"{_nrows:,}")
    _c4.metric("Columns", str(_ncols))
    if _age_days > 3:
        st.warning(f"Data is {_age_days} days old — daily refresh may have failed. Check GitHub Actions.")
    elif _age_days > 1:
        st.info(f"Data is {_age_days} day(s) old (expected if weekend or holiday).")
    else:
        st.success("Data is current.")
except Exception as _de:
    st.error(f"Could not load data: {_de}")

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
