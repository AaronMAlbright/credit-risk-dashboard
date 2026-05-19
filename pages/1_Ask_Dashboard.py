"""
Ask the Dashboard — Claude-powered Q&A about current market conditions.

Synthesizes signals from all loaded modules and answers natural-language
questions about credit risk, positioning, and market conditions.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ask the Dashboard",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def _load_df() -> pd.DataFrame:
    """Load and score the full pipeline DataFrame, returning empty DF on failure."""
    # Inject Streamlit Cloud secrets so downstream loaders see API keys
    try:
        for _k, _v in st.secrets.items():
            if isinstance(_v, str):
                os.environ.setdefault(_k, _v)
    except Exception:
        pass

    try:
        from src.market_data import load_all_series
        from src.feature_engine import build_raw_dataset, add_derived_features
        from src.pipeline import run_scoring_pipeline

        raw = load_all_series()
        featured = add_derived_features(raw)
        return run_scoring_pipeline(featured)
    except Exception:
        pass

    # Fallback: try the higher-level pipeline path used by the main app
    try:
        from src.data_pipeline import run_pipeline

        result = run_pipeline()
        csv_path = "data/scored_macro_credit_data.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            return df
    except Exception:
        pass

    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a senior credit portfolio manager analyzing a real-time credit risk dashboard.
You have access to the current state of 35+ market signals covering credit spreads,
rates, volatility, macro conditions, and regime indicators.

Answer questions concisely and directly. Lead with the key insight, then supporting evidence.
Be specific about numbers. Flag if a signal is ambiguous or if data may be stale.
Do not hedge excessively — give a clear view.

If asked about positioning, frame it in terms of credit portfolio decisions
(HY vs IG allocation, duration, carry vs spread risk).
"""


def _safe_val(df: pd.DataFrame, col: str) -> float:
    """Return the last valid value for *col*, or NaN if absent."""
    s = df.get(col, pd.Series([float("nan")]))
    if s.empty:
        return float("nan")
    return float(s.iloc[-1]) if not pd.isna(s.iloc[-1]) else float("nan")


def _fmt(val: float, decimals: int = 2, suffix: str = "") -> str:
    if pd.isna(val):
        return "N/A"
    return f"{val:.{decimals}f}{suffix}"


def build_context(df: pd.DataFrame) -> str:
    """Extract the last row of key signals into a structured text block for Claude."""
    if df.empty:
        return "Dashboard data is unavailable — unable to extract current signal state."

    # Date
    last_date = df.index[-1].date() if isinstance(df.index, pd.DatetimeIndex) else "unknown"

    # Core composite
    composite = _safe_val(df, "composite_risk_score_smooth")
    regime = df.get("final_decision", pd.Series(["N/A"])).iloc[-1]

    # Credit
    hy_spread = _safe_val(df, "hy_spread")
    ig_spread = _safe_val(df, "ig_spread")
    hy_ig_ratio = (hy_spread / ig_spread) if not (pd.isna(hy_spread) or pd.isna(ig_spread) or ig_spread == 0) else float("nan")
    hy_mom = _safe_val(df, "hy_change_90d")       # 90d momentum = ~63 trading days
    # Also try 63d-specific momentum col if present
    for _mom_col in ("hy_mom_63d", "hy_change_63d", "hy_change_90d"):
        _v = _safe_val(df, _mom_col)
        if not pd.isna(_v):
            hy_mom = _v
            break

    # Rates
    yield_10y = _safe_val(df, "yield_10y")
    breakeven_10y = _safe_val(df, "breakeven_10y")
    real_rate_10y = (yield_10y - breakeven_10y) if not (pd.isna(yield_10y) or pd.isna(breakeven_10y)) else _safe_val(df, "real_rate_10y")
    yield_3m = _safe_val(df, "yield_3m")
    yield_curve_spread = (yield_10y - yield_3m) if not (pd.isna(yield_10y) or pd.isna(yield_3m)) else _safe_val(df, "spread_10y3m")

    # Volatility
    vix = _safe_val(df, "vix")
    vrp = _safe_val(df, "vrp_21d")

    # Macro / regime context
    policy_stance = df.get("policy_stance", pd.Series(["N/A"])).iloc[-1]
    recession_prob = _safe_val(df, "recession_prob")
    if pd.isna(recession_prob):
        recession_prob = _safe_val(df, "recession_prob_12m")
    funding_regime = df.get("funding_regime", pd.Series(["N/A"])).iloc[-1]

    lines = [
        f"Current dashboard state (as of {last_date}):",
        "",
        f"COMPOSITE SCORE: {_fmt(composite, 1)}/100 | Regime: {regime}",
        "",
        "CREDIT:",
        f"  HY Spread: {_fmt(hy_spread, 0)} bps",
        f"  IG Spread: {_fmt(ig_spread, 0)} bps",
        f"  HY/IG Ratio: {_fmt(hy_ig_ratio, 2)}x",
        f"  HY 3M Momentum: {_fmt(hy_mom, 0, ' bps') if not pd.isna(hy_mom) else 'N/A'}",
        "",
        "RATES & MACRO:",
        f"  10y Yield: {_fmt(yield_10y, 2)}%",
        f"  Real Rate (10y): {_fmt(real_rate_10y, 2)}%",
        f"  Yield Curve (10y-3m): {_fmt(yield_curve_spread, 2)}%",
        "",
        "VOLATILITY:",
        f"  VIX: {_fmt(vix, 1)}",
        f"  VRP (21d): {_fmt(vrp, 1, '%') if not pd.isna(vrp) else 'N/A'}",
        "",
        "REGIME CONTEXT:",
        f"  Policy Stance: {policy_stance}",
        f"  Recession Prob (12m): {_fmt(recession_prob * 100 if not pd.isna(recession_prob) and recession_prob <= 1 else recession_prob, 1, '%') if not pd.isna(recession_prob) else 'N/A'}",
        f"  Funding Regime: {funding_regime}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------------------------

_SUGGESTED_QUESTIONS = [
    "What are the top 3 credit risks right now?",
    "Should I be adding HY exposure or reducing?",
    "How does the current environment compare to 2022?",
    "What signals are flashing warning?",
    "Is the yield curve telling us something different from credit spreads?",
]

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("Ask the Dashboard")
    st.caption("Claude-powered Q&A — synthesizes current credit risk signals into natural-language answers.")

    # --- API key guard ---
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
        except Exception:
            pass

    if not api_key:
        st.error("Set ANTHROPIC_API_KEY to enable Q&A")
        st.info(
            "Add `ANTHROPIC_API_KEY = 'sk-ant-...'` to your `.env` file or "
            "Streamlit Cloud secrets, then reload the page."
        )
        return

    # --- Load data ---
    with st.spinner("Loading dashboard data..."):
        df = _load_df()

    # --- Build context ---
    context_text = build_context(df)

    # --- Sidebar ---
    with st.sidebar:
        st.subheader("Controls")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        with st.expander("Current Signal State", expanded=False):
            st.code(context_text, language=None)

        if not df.empty:
            st.caption(
                f"Data: {len(df):,} rows | "
                f"Latest: {df.index[-1].date() if isinstance(df.index, pd.DatetimeIndex) else 'N/A'}"
            )

    # --- Session state ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Suggested questions (shown when chat is empty) ---
    if not st.session_state.messages:
        st.markdown("**Suggested questions:**")
        cols = st.columns(len(_SUGGESTED_QUESTIONS))
        for col, question in zip(cols, _SUGGESTED_QUESTIONS):
            with col:
                if st.button(question, use_container_width=True, key=f"sq_{question[:20]}"):
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.rerun()

    # --- Render existing chat history ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Chat input ---
    user_input = st.chat_input("Ask about current market conditions...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        _run_claude(context_text)

    # If a suggested question was just added (messages end with a user turn but
    # no assistant reply yet), run Claude for it.
    elif (
        st.session_state.messages
        and st.session_state.messages[-1]["role"] == "user"
    ):
        with st.chat_message("user"):
            st.markdown(st.session_state.messages[-1]["content"])
        _run_claude(context_text)


def _run_claude(context_text: str) -> None:
    """Call Claude with the current message history and stream the response."""
    import anthropic

    client = anthropic.Anthropic()

    # Build the messages list for the API call.
    # Prepend the context block to the first user message only.
    api_messages: list[dict] = []
    for i, msg in enumerate(st.session_state.messages):
        if i == 0 and msg["role"] == "user":
            content = f"{context_text}\n\n---\n\nUser question: {msg['content']}"
        else:
            content = msg["content"]
        api_messages.append({"role": msg["role"], "content": content})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            with client.messages.stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=api_messages,
            ) as stream:
                for text_chunk in stream.text_stream:
                    full_response += text_chunk
                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        except anthropic.APIStatusError as exc:
            full_response = f"API error ({exc.status_code}): {exc.message}"
            response_placeholder.error(full_response)
        except anthropic.APIConnectionError:
            full_response = "Connection error — check your network and try again."
            response_placeholder.error(full_response)
        except Exception as exc:
            full_response = f"Unexpected error: {exc}"
            response_placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
else:
    main()
