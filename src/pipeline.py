"""End-to-end analysis pipeline, wired up with Streamlit caching so the
(relatively) expensive semantic model load and embedding calls are not
repeated on every rerun.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from src.coverage import (
    add_status_column,
    best_match_per_requirement,
    build_coverage_gaps,
    build_traceability_matrix,
    compute_coverage_summary,
)
from src.matching_engine import build_match_table, compute_match_matrices
from src.semantic_matcher import SemanticMatcher


@st.cache_resource(show_spinner=False)
def get_semantic_matcher() -> SemanticMatcher:
    """Load (and cache) the sentence-transformer model once per session/process."""
    return SemanticMatcher()


@st.cache_data(show_spinner=False)
def cached_match_matrices(
    requirements_clean: pd.DataFrame, test_cases_clean: pd.DataFrame
):
    """Cache the raw TF-IDF/semantic score matrices for a given data set,
    so changing only thresholds/weights does not recompute embeddings.
    """
    matcher = get_semantic_matcher()
    tfidf_matrix, semantic_matrix, model_loaded = compute_match_matrices(
        requirements_clean, test_cases_clean, semantic_matcher=matcher
    )
    return tfidf_matrix, semantic_matrix, model_loaded


def run_full_analysis(
    requirements_clean: pd.DataFrame,
    test_cases_clean: pd.DataFrame,
    tfidf_weight: float,
    semantic_weight: float,
    strong_threshold: float,
    partial_threshold: float,
    top_n: int,
    adequate_includes_partial: bool,
) -> Dict[str, object]:
    """Run the complete matching + coverage pipeline and return every
    artifact the UI pages need, keyed by name.
    """
    tfidf_matrix, semantic_matrix, model_loaded = cached_match_matrices(
        requirements_clean, test_cases_clean
    )

    match_table = build_match_table(
        requirements_clean,
        test_cases_clean,
        tfidf_matrix,
        semantic_matrix,
        tfidf_weight,
        semantic_weight,
    )
    match_table = add_status_column(match_table, strong_threshold, partial_threshold)

    best_match = best_match_per_requirement(match_table)
    traceability_matrix = build_traceability_matrix(
        requirements_clean, test_cases_clean, match_table, top_n=top_n
    )
    coverage_summary = compute_coverage_summary(
        requirements_clean, best_match, adequate_includes_partial
    )
    coverage_gaps = build_coverage_gaps(
        requirements_clean, test_cases_clean, best_match
    )

    matcher = get_semantic_matcher()

    return {
        "match_table": match_table,
        "best_match_table": best_match,
        "traceability_matrix": traceability_matrix,
        "coverage_summary": coverage_summary,
        "coverage_gaps": coverage_gaps,
        "semantic_model_loaded": model_loaded,
        "semantic_load_error": matcher.load_error,
    }
