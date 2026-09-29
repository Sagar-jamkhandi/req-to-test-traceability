import pandas as pd

from src.coverage import (
    add_status_column,
    best_match_per_requirement,
    build_coverage_gaps,
    build_traceability_matrix,
    classify_status,
    compute_coverage_summary,
)
from src.utils import STATUS_MISSING, STATUS_PARTIAL, STATUS_STRONG


def test_classify_status_strong():
    assert classify_status(0.9, strong_threshold=0.75, partial_threshold=0.5) == STATUS_STRONG


def test_classify_status_partial():
    assert classify_status(0.6, strong_threshold=0.75, partial_threshold=0.5) == STATUS_PARTIAL


def test_classify_status_missing():
    assert classify_status(0.2, strong_threshold=0.75, partial_threshold=0.5) == STATUS_MISSING


def test_classify_status_boundary_values():
    assert classify_status(0.75, 0.75, 0.5) == STATUS_STRONG
    assert classify_status(0.5, 0.75, 0.5) == STATUS_PARTIAL
    assert classify_status(0.4999, 0.75, 0.5) == STATUS_MISSING


def _match_table():
    return pd.DataFrame(
        {
            "requirement_id": ["REQ-001", "REQ-001", "REQ-002", "REQ-003"],
            "test_case_id": ["TC-001", "TC-002", "TC-003", "TC-004"],
            "tfidf_score": [0.9, 0.3, 0.6, 0.1],
            "semantic_score": [0.9, 0.3, 0.6, 0.1],
            "final_score": [0.9, 0.3, 0.6, 0.1],
        }
    )


def _requirements_df():
    return pd.DataFrame(
        {
            "requirement_id": ["REQ-001", "REQ-002", "REQ-003"],
            "title": ["Login", "Reset", "Lockout"],
            "description": ["d1", "d2", "d3"],
            "priority": ["High", "Medium", "Low"],
        }
    )


def _test_cases_df():
    return pd.DataFrame(
        {
            "test_case_id": ["TC-001", "TC-002", "TC-003", "TC-004"],
            "title": ["T1", "T2", "T3", "T4"],
            "description": ["d1", "d2", "d3", "d4"],
            "priority": ["High", "High", "Medium", "Low"],
        }
    )


def test_best_match_picks_highest_score_per_requirement():
    table = add_status_column(_match_table(), 0.75, 0.5)
    best = best_match_per_requirement(table)
    req1_row = best[best["requirement_id"] == "REQ-001"].iloc[0]
    assert req1_row["test_case_id"] == "TC-001"
    assert req1_row["status"] == STATUS_STRONG


def test_coverage_summary_counts_and_percentage():
    table = add_status_column(_match_table(), 0.75, 0.5)
    best = best_match_per_requirement(table)
    summary = compute_coverage_summary(_requirements_df(), best, adequate_includes_partial=True)
    assert summary["total_requirements"] == 3
    assert summary["strong"] == 1
    assert summary["partial"] == 1
    assert summary["missing"] == 1
    assert summary["coverage_pct"] == round(2 / 3 * 100, 1)


def test_coverage_summary_strong_only_definition():
    table = add_status_column(_match_table(), 0.75, 0.5)
    best = best_match_per_requirement(table)
    summary = compute_coverage_summary(_requirements_df(), best, adequate_includes_partial=False)
    assert summary["coverage_pct"] == round(1 / 3 * 100, 1)


def test_coverage_summary_handles_zero_requirements():
    empty_req = pd.DataFrame(columns=["requirement_id", "title", "description", "priority"])
    summary = compute_coverage_summary(empty_req, pd.DataFrame(), True)
    assert summary["total_requirements"] == 0
    assert summary["coverage_pct"] == 0.0


def test_build_traceability_matrix_includes_missing_row_when_no_candidates():
    reqs = _requirements_df()
    tcs = _test_cases_df()
    table = pd.DataFrame(
        {
            "requirement_id": ["REQ-001"],
            "test_case_id": ["TC-001"],
            "tfidf_score": [0.01],
            "semantic_score": [0.01],
            "final_score": [0.01],
        }
    )
    table = add_status_column(table, 0.75, 0.5)
    matrix = build_traceability_matrix(reqs, tcs, table, top_n=5)
    # REQ-002 and REQ-003 have no rows at all -> should appear as explicit gaps
    assert set(matrix["requirement_id"]) == {"REQ-001", "REQ-002", "REQ-003"}
    req2_row = matrix[matrix["requirement_id"] == "REQ-002"].iloc[0]
    assert req2_row["status"] == STATUS_MISSING
    assert req2_row["test_case_id"] == "—"


def test_build_coverage_gaps_sorted_by_priority_then_score():
    table = add_status_column(_match_table(), 0.75, 0.5)
    best = best_match_per_requirement(table)
    gaps = build_coverage_gaps(_requirements_df(), _test_cases_df(), best)
    # REQ-002 (Medium, partial) and REQ-003 (Low, missing) should both appear
    assert set(gaps["requirement_id"]) == {"REQ-002", "REQ-003"}
    assert "recommended_action" in gaps.columns
