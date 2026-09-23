"""Small reusable Streamlit rendering helpers shared across pages."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.utils import STATUS_BADGES, STATUS_COLORS


def style_status_column(df: pd.DataFrame, status_col: str = "status"):
    """Return a pandas Styler that colors the status column's background,
    without relying on color alone (the text itself also names the status).
    """

    def _color(val):
        color = STATUS_COLORS.get(val, "#888888")
        return f"background-color: {color}22; color: {color}; font-weight: 600;"

    return df.style.map(_color, subset=[status_col])


def render_kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    """Render a row of st.metric KPIs. ``items`` is a list of
    (label, value, help_text_or_None) tuples.
    """
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        with col:
            st.metric(label, value, help=help_text)


def guard_no_data() -> bool:
    """Show a friendly message and return True if data hasn't been loaded yet."""
    from src.state import has_data

    if not has_data():
        st.warning(
            "No requirements/test cases loaded yet. Go to **Data Upload** to "
            "load the sample data set or upload your own files."
        )
        safe_page_link("pages/2_Data_Upload.py", "Go to Data Upload", "📤")
        return True
    return False


def guard_no_analysis() -> bool:
    """Show a friendly message and return True if analysis hasn't been run yet."""
    from src.state import has_analysis

    if not has_analysis():
        st.warning(
            "Analysis has not been run yet. Go to **Data Upload** and click "
            "**Run Traceability Analysis**."
        )
        safe_page_link("pages/2_Data_Upload.py", "Go to Data Upload", "📤")
        return True
    return False


def status_badge(status: str) -> str:
    return STATUS_BADGES.get(status, status)


def safe_page_link(path: str, label: str, icon: str | None = None) -> None:
    """st.page_link, guarded against environments where Streamlit's page
    registry isn't available (e.g. a page script executed standalone by a
    test harness rather than through the multipage app). Falls back to a
    plain text hint so the app never crashes because of a navigation link.
    """
    try:
        st.page_link(path, label=label, icon=icon)
    except Exception:  # noqa: BLE001 - navigation is a "nice to have", never fatal
        st.caption(f"{icon or ''} {label} ({path})".strip())
