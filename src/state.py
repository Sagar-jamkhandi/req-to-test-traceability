"""Centralized Streamlit ``session_state`` initialization and helpers.

Streamlit's multipage apps rerun a fresh script top-to-bottom for every
page, so anything that needs to persist across pages (uploaded data,
computed matches, configuration) lives in ``st.session_state``. Keeping
the keys and defaults in one place avoids typos and makes the data flow
easy to follow.
"""
from __future__ import annotations

import streamlit as st

from src.utils import (
    DEFAULT_PARTIAL_THRESHOLD,
    DEFAULT_SEMANTIC_WEIGHT,
    DEFAULT_STRONG_THRESHOLD,
    DEFAULT_TFIDF_WEIGHT,
    DEFAULT_TOP_N,
)

DEFAULTS = {
    "requirements_raw": None,       # raw uploaded/sample DataFrame
    "test_cases_raw": None,         # raw uploaded/sample DataFrame
    "requirements_clean": None,     # preprocessed DataFrame
    "test_cases_clean": None,       # preprocessed DataFrame
    "requirements_warnings": [],
    "test_cases_warnings": [],
    "match_table": None,            # long-format table w/ status
    "best_match_table": None,       # one row per requirement
    "traceability_matrix": None,
    "coverage_summary": None,
    "coverage_gaps": None,
    "semantic_model_loaded": None,
    "semantic_load_error": None,
    "tfidf_weight": DEFAULT_TFIDF_WEIGHT,
    "semantic_weight": DEFAULT_SEMANTIC_WEIGHT,
    "strong_threshold": DEFAULT_STRONG_THRESHOLD,
    "partial_threshold": DEFAULT_PARTIAL_THRESHOLD,
    "top_n": DEFAULT_TOP_N,
    "adequate_includes_partial": True,
    "data_source": None,            # "sample" or "upload"
    "analysis_has_run": False,
}


def init_session_state() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_analysis_state() -> None:
    """Clear computed results (but keep uploaded data / config) so a
    re-run is forced after data or settings change."""
    for key in (
        "match_table",
        "best_match_table",
        "traceability_matrix",
        "coverage_summary",
        "coverage_gaps",
        "analysis_has_run",
    ):
        st.session_state[key] = DEFAULTS[key]


def has_data() -> bool:
    return (
        st.session_state.get("requirements_clean") is not None
        and st.session_state.get("test_cases_clean") is not None
        and len(st.session_state["requirements_clean"]) > 0
        and len(st.session_state["test_cases_clean"]) > 0
    )


def has_analysis() -> bool:
    return bool(st.session_state.get("analysis_has_run"))
