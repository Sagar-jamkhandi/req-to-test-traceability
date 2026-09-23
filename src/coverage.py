"""Coverage classification, traceability matrix construction, and summary
metrics.

Key assumption (documented on the Assumptions page too): a requirement's
coverage status is based on its **single best-matching test case**
("best-match" aggregation), not an average across all candidates. This is
the most common traceability convention: if one strong test exists, the
requirement is considered covered even if other candidate tests are weak.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.utils import STATUS_MISSING, STATUS_PARTIAL, STATUS_STRONG


def classify_status(score: float, strong_threshold: float, partial_threshold: float) -> str:
    if score >= strong_threshold:
        return STATUS_STRONG
    if score >= partial_threshold:
        return STATUS_PARTIAL
    return STATUS_MISSING


def add_status_column(
    match_table: pd.DataFrame, strong_threshold: float, partial_threshold: float
) -> pd.DataFrame:
    out = match_table.copy()
    out["status"] = out["final_score"].apply(
        lambda s: classify_status(s, strong_threshold, partial_threshold)
    )
    return out


def build_traceability_matrix(
    requirements_df: pd.DataFrame,
    test_cases_df: pd.DataFrame,
    match_table_with_status: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Build the many-to-many traceability matrix: for every requirement,
    list up to ``top_n`` candidate test cases above a nominal relevance
    floor, each with its own score and status.

    Requirements with no candidate above the floor still get one row
    (Test Case = "-", Status = Missing) so every requirement is represented.
    """
    req_lookup = requirements_df.set_index("requirement_id")
    test_lookup = test_cases_df.set_index("test_case_id")

    rows = []
    for req_id, req_row in req_lookup.iterrows():
        candidates = (
            match_table_with_status[
                match_table_with_status["requirement_id"] == req_id
            ]
            .sort_values("final_score", ascending=False)
            .head(top_n)
        )
        # Only keep candidates that are at least weakly relevant (score > 0.05)
        # for display purposes; if none qualify, show an explicit gap row.
        relevant = candidates[candidates["final_score"] > 0.05]

        if relevant.empty:
            rows.append(
                {
                    "requirement_id": req_id,
                    "requirement_title": req_row.get("title", ""),
                    "priority": req_row.get("priority", "Not set"),
                    "test_case_id": "—",
                    "test_case_title": "—",
                    "tfidf_score": 0.0,
                    "semantic_score": 0.0,
                    "final_score": 0.0,
                    "status": STATUS_MISSING,
                }
            )
            continue

        for _, cand in relevant.iterrows():
            test_row = test_lookup.loc[cand["test_case_id"]]
            rows.append(
                {
                    "requirement_id": req_id,
                    "requirement_title": req_row.get("title", ""),
                    "priority": req_row.get("priority", "Not set"),
                    "test_case_id": cand["test_case_id"],
                    "test_case_title": test_row.get("title", ""),
                    "tfidf_score": cand["tfidf_score"],
                    "semantic_score": cand["semantic_score"],
                    "final_score": cand["final_score"],
                    "status": cand["status"],
                }
            )

    return pd.DataFrame(rows)


def best_match_per_requirement(match_table_with_status: pd.DataFrame) -> pd.DataFrame:
    """Return one row per requirement: its single best-scoring test case.

    This is the table coverage classification/aggregate metrics are based
    on (see module docstring).
    """
    if match_table_with_status.empty:
        return match_table_with_status
    idx = match_table_with_status.groupby("requirement_id")["final_score"].idxmax()
    return match_table_with_status.loc[idx].reset_index(drop=True)


def compute_coverage_summary(
    requirements_df: pd.DataFrame,
    best_match_df: pd.DataFrame,
    adequate_includes_partial: bool = True,
) -> Dict[str, float]:
    """Compute headline coverage metrics.

    ``adequate_includes_partial`` controls whether the overall coverage %
    counts Strong-only, or Strong + Partial, as "adequately covered". This
    is a documented, configurable assumption (see Assumptions page).
    """
    total_requirements = len(requirements_df)
    if total_requirements == 0:
        return {
            "total_requirements": 0,
            "strong": 0,
            "partial": 0,
            "missing": 0,
            "coverage_pct": 0.0,
        }

    status_counts = best_match_df["status"].value_counts().to_dict()
    strong = status_counts.get(STATUS_STRONG, 0)
    partial = status_counts.get(STATUS_PARTIAL, 0)
    missing = status_counts.get(STATUS_MISSING, 0)

    # Requirements that have no row at all in best_match_df (shouldn't
    # normally happen, but guard against it) count as missing.
    accounted = strong + partial + missing
    missing += max(0, total_requirements - accounted)

    adequate = strong + partial if adequate_includes_partial else strong
    coverage_pct = round((adequate / total_requirements) * 100, 1)

    return {
        "total_requirements": total_requirements,
        "strong": strong,
        "partial": partial,
        "missing": missing,
        "coverage_pct": coverage_pct,
    }


def build_coverage_gaps(
    requirements_df: pd.DataFrame,
    test_cases_df: pd.DataFrame,
    best_match_df: pd.DataFrame,
    include_statuses: List[str] = None,
) -> pd.DataFrame:
    """Build the Coverage Gaps table: requirements that are Partial or
    Missing, with their best candidate test case (if any), sorted by
    priority then by lowest score.
    """
    if include_statuses is None:
        include_statuses = [STATUS_MISSING, STATUS_PARTIAL]

    test_lookup = test_cases_df.set_index("test_case_id")
    req_lookup = requirements_df.set_index("requirement_id")

    gaps = best_match_df[best_match_df["status"].isin(include_statuses)].copy()

    def _resolve_action(status: str) -> str:
        if status == STATUS_MISSING:
            return "Create explicit positive and boundary test cases for this requirement."
        return "Review existing candidate test and add missing edge cases to strengthen coverage."

    rows = []
    for _, row in gaps.iterrows():
        req_id = row["requirement_id"]
        req_row = req_lookup.loc[req_id]
        test_title = "—"
        if row["test_case_id"] in test_lookup.index:
            test_title = test_lookup.loc[row["test_case_id"], "title"]
        rows.append(
            {
                "requirement_id": req_id,
                "requirement": req_row.get("title", ""),
                "priority": req_row.get("priority", "Not set"),
                "best_match_id": row["test_case_id"] if row["final_score"] > 0.05 else "—",
                "best_match_title": test_title if row["final_score"] > 0.05 else "—",
                "similarity": row["final_score"],
                "status": row["status"],
                "recommended_action": _resolve_action(row["status"]),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    priority_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3, "not set": 4}
    result["_priority_rank"] = (
        result["priority"].str.lower().map(priority_rank).fillna(4)
    )
    result = result.sort_values(
        by=["_priority_rank", "similarity"], ascending=[True, True]
    ).drop(columns=["_priority_rank"]).reset_index(drop=True)
    return result
