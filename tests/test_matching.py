import numpy as np
import pandas as pd

from src.matching_engine import (
    build_match_table,
    explain_match,
    top_matches_for_requirement,
)
from src.preprocessing import preprocess_requirements, preprocess_test_cases
from src.semantic_matcher import SemanticMatcher


def test_semantic_matcher_falls_back_gracefully_when_model_unavailable():
    # In this sandboxed test environment there is no network access to
    # huggingface.co, so the matcher is expected to fail to load and fall
    # back to TF-IDF automatically rather than raising.
    matcher = SemanticMatcher(model_name="a-model-name-that-does-not-exist-xyz")
    assert matcher.model_loaded is False
    assert matcher.load_error is not None

    matrix = matcher.score_matrix(
        ["user can reset password"], ["verify password reset flow"]
    )
    assert matrix.shape == (1, 1)
    assert 0.0 <= matrix[0, 0] <= 1.0


def test_build_match_table_shape_and_columns(sample_requirements_df, sample_test_cases_df):
    req = preprocess_requirements(sample_requirements_df)
    tc = preprocess_test_cases(sample_test_cases_df)

    n_req, n_tc = len(req), len(tc)
    tfidf_matrix = np.full((n_req, n_tc), 0.4)
    semantic_matrix = np.full((n_req, n_tc), 0.6)

    table = build_match_table(req, tc, tfidf_matrix, semantic_matrix, 0.3, 0.7)

    assert len(table) == n_req * n_tc
    assert set(table.columns) == {
        "requirement_id",
        "test_case_id",
        "tfidf_score",
        "semantic_score",
        "final_score",
    }
    # 0.3*0.4 + 0.7*0.6 = 0.54
    assert abs(table["final_score"].iloc[0] - 0.54) < 1e-6


def test_build_match_table_normalizes_weights(sample_requirements_df, sample_test_cases_df):
    req = preprocess_requirements(sample_requirements_df)
    tc = preprocess_test_cases(sample_test_cases_df)
    n_req, n_tc = len(req), len(tc)
    tfidf_matrix = np.full((n_req, n_tc), 1.0)
    semantic_matrix = np.full((n_req, n_tc), 1.0)

    # weights sum to 2, not 1 -> should be normalized and still cap at 1.0
    table = build_match_table(req, tc, tfidf_matrix, semantic_matrix, 1.0, 1.0)
    assert np.allclose(table["final_score"], 1.0)


def test_top_matches_for_requirement_ranks_descending():
    table = pd.DataFrame(
        {
            "requirement_id": ["REQ-001"] * 3,
            "test_case_id": ["TC-001", "TC-002", "TC-003"],
            "tfidf_score": [0.1, 0.2, 0.3],
            "semantic_score": [0.1, 0.2, 0.3],
            "final_score": [0.2, 0.9, 0.5],
        }
    )
    top = top_matches_for_requirement(table, "REQ-001", top_n=2)
    assert list(top["test_case_id"]) == ["TC-002", "TC-003"]


def test_explain_match_mentions_status_and_scores():
    text = explain_match(
        requirement_text="user can reset password using email",
        test_case_text="verify password reset through email",
        tfidf_score=0.6,
        semantic_score=0.8,
        final_score=0.74,
        status="Partial",
    )
    assert "Partial" in text
    assert "0.74" in text
    assert "recommended" in text.lower()
