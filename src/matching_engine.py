"""Combines the TF-IDF and semantic matchers into a single weighted score,
and produces the long-format "every requirement x every test case" match
table that the rest of the app (traceability matrix, coverage, gap
analysis, matching-analysis page) is built on.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.semantic_matcher import SemanticMatcher
from src.tfidf_matcher import TfidfMatcher
from src.utils import clamp

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to",
    "of", "and", "or", "in", "on", "for", "with", "that", "this", "it",
    "as", "by", "at", "from", "shall", "should", "will", "can", "must",
    "user", "system", "able",
}


def compute_match_matrices(
    requirements_df: pd.DataFrame,
    test_cases_df: pd.DataFrame,
    semantic_matcher: Optional[SemanticMatcher] = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Compute the raw TF-IDF and semantic score matrices.

    Returns (tfidf_matrix, semantic_matrix, semantic_model_loaded).
    """
    req_texts = requirements_df["combined_text_clean"].tolist()
    test_texts = test_cases_df["combined_text_clean"].tolist()

    tfidf_matrix = TfidfMatcher().score_matrix(req_texts, test_texts)

    if semantic_matcher is None:
        semantic_matcher = SemanticMatcher()
    semantic_matrix = semantic_matcher.score_matrix(req_texts, test_texts)

    return tfidf_matrix, semantic_matrix, semantic_matcher.model_loaded


def build_match_table(
    requirements_df: pd.DataFrame,
    test_cases_df: pd.DataFrame,
    tfidf_matrix: np.ndarray,
    semantic_matrix: np.ndarray,
    tfidf_weight: float,
    semantic_weight: float,
) -> pd.DataFrame:
    """Build the long-format match table: one row per (requirement, test case).

    ``final_score`` is the weighted combination of TF-IDF and semantic
    similarity. Weights are normalized to sum to 1.0 so the score always
    stays in [0, 1] regardless of what the user entered in the UI.
    """
    total_weight = tfidf_weight + semantic_weight
    if total_weight <= 0:
        tfidf_weight, semantic_weight = 0.5, 0.5
        total_weight = 1.0
    w_tfidf = tfidf_weight / total_weight
    w_semantic = semantic_weight / total_weight

    final_matrix = w_tfidf * tfidf_matrix + w_semantic * semantic_matrix

    rows = []
    req_ids = requirements_df["requirement_id"].tolist()
    test_ids = test_cases_df["test_case_id"].tolist()

    for i, req_id in enumerate(req_ids):
        for j, test_id in enumerate(test_ids):
            rows.append(
                {
                    "requirement_id": req_id,
                    "test_case_id": test_id,
                    "tfidf_score": round(float(tfidf_matrix[i, j]), 4),
                    "semantic_score": round(float(semantic_matrix[i, j]), 4),
                    "final_score": round(float(clamp(final_matrix[i, j])), 4),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "requirement_id",
            "test_case_id",
            "tfidf_score",
            "semantic_score",
            "final_score",
        ],
    )


def top_matches_for_requirement(
    match_table: pd.DataFrame, requirement_id: str, top_n: int = 5
) -> pd.DataFrame:
    """Return the top-N test-case candidates for a single requirement,
    ranked by ``final_score`` descending.
    """
    subset = match_table[match_table["requirement_id"] == requirement_id]
    return subset.sort_values("final_score", ascending=False).head(top_n).reset_index(
        drop=True
    )


def _overlapping_terms(text_a: str, text_b: str, max_terms: int = 6) -> List[str]:
    words_a = {w for w in text_a.split() if w not in STOP_WORDS and len(w) > 2}
    words_b = {w for w in text_b.split() if w not in STOP_WORDS and len(w) > 2}
    overlap = sorted(words_a & words_b)
    return overlap[:max_terms]


def explain_match(
    requirement_text: str,
    test_case_text: str,
    tfidf_score: float,
    semantic_score: float,
    final_score: float,
    status: str,
) -> str:
    """Produce a short, honest, human-readable explanation of a match.

    Explanations describe textual relevance - they never claim functional
    correctness or "proof" of coverage.
    """
    shared_terms = _overlapping_terms(requirement_text, test_case_text)

    if final_score >= 0.75:
        strength_phrase = "closely align"
    elif final_score >= 0.50:
        strength_phrase = "partially overlap"
    else:
        strength_phrase = "show limited overlap"

    if shared_terms:
        terms_phrase = (
            f" They share key terms such as {', '.join(shared_terms)}."
        )
    else:
        terms_phrase = (
            " They do not share many exact terms; any similarity is "
            "driven mainly by semantic (meaning-based) similarity rather "
            "than shared vocabulary."
        )

    return (
        f"The requirement and test case {strength_phrase} based on "
        f"AI-assisted relevance matching (TF-IDF: {tfidf_score:.2f}, "
        f"semantic: {semantic_score:.2f}, combined: {final_score:.2f})."
        f"{terms_phrase} Status: {status} candidate coverage. "
        f"Human validation is recommended before treating this as "
        f"confirmed test coverage."
    )
