"""
Shared nav-state helper: initialise session-state variables before any
conditional use, preventing NameError when a page is loaded cold.
"""
import streamlit as st


def init_nav_state() -> None:
    """Ensure all nav/section variables exist in st.session_state."""
    defaults = {
        "_active_section": "Overview",
        "_nav_type":        "Core",
        "_active_sub":      None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get_active_section(default: str = "Overview") -> str:
    return st.session_state.get("_active_section", default)


def get_nav_type(default: str = "Core") -> str:
    return st.session_state.get("_nav_type", default)
