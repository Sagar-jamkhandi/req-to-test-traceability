"""Validation for uploaded requirement / test-case data.

Validation never raises on "expected" bad input - it always returns a
result object so the Streamlit UI can show a friendly message instead of
crashing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from src.utils import (
    REQUIRED_REQUIREMENT_COLUMNS,
    REQUIRED_TEST_CASE_COLUMNS,
    find_missing_columns,
    safe_str,
)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def _validate_generic(
    df: pd.DataFrame,
    required_columns: List[str],
    id_column: str,
    record_label: str,
) -> ValidationResult:
    result = ValidationResult(is_valid=True)

    if df is None or df.empty:
        result.add_error(
            f"No {record_label} were found in the uploaded file. "
            f"Please upload a non-empty CSV/XLSX file."
        )
        return result

    missing = find_missing_columns(df, required_columns)
    if missing:
        result.add_error(
            f"The {record_label} file is missing required column(s): "
            f"{', '.join(missing)}. Expected columns (or recognizable "
            f"variants of them): {', '.join(required_columns)}."
        )
        return result

    # Empty / null critical fields
    empty_id_rows = df[df[id_column].apply(lambda v: safe_str(v) == "")]
    if len(empty_id_rows) > 0:
        result.add_warning(
            f"{len(empty_id_rows)} {record_label} row(s) have a blank "
            f"{id_column} and were dropped."
        )

    empty_title_rows = df[df["title"].apply(lambda v: safe_str(v) == "")]
    if len(empty_title_rows) > 0:
        result.add_warning(
            f"{len(empty_title_rows)} {record_label} row(s) have a blank "
            f"title."
        )

    combined_len = (
        df["title"].apply(safe_str) + " " + df["description"].apply(safe_str)
    ).str.strip()
    very_short = combined_len[combined_len.str.len() < 8]
    if len(very_short) > 0:
        result.add_warning(
            f"{len(very_short)} {record_label} row(s) have very little text "
            f"(title + description under 8 characters), which may reduce "
            f"matching quality."
        )

    # Duplicate IDs
    ids = df[id_column].apply(safe_str)
    duplicate_ids = ids[ids.duplicated() & (ids != "")]
    if len(duplicate_ids) > 0:
        result.add_warning(
            f"{duplicate_ids.nunique()} duplicate {id_column} value(s) found "
            f"({', '.join(sorted(set(duplicate_ids))[:5])}"
            f"{'...' if duplicate_ids.nunique() > 5 else ''}). "
            f"Later rows with the same ID will be kept and earlier ones "
            f"dropped."
        )

    # Duplicate content (same title+description)
    dup_content = combined_len.str.lower()
    dup_mask = dup_content.duplicated() & (dup_content.str.len() > 0)
    if dup_mask.sum() > 0:
        result.add_warning(
            f"{int(dup_mask.sum())} {record_label} row(s) appear to be "
            f"textual duplicates of another row."
        )

    return result


def validate_requirements(df: pd.DataFrame) -> ValidationResult:
    return _validate_generic(
        df, REQUIRED_REQUIREMENT_COLUMNS, "requirement_id", "requirement"
    )


def validate_test_cases(df: pd.DataFrame) -> ValidationResult:
    return _validate_generic(
        df, REQUIRED_TEST_CASE_COLUMNS, "test_case_id", "test case"
    )


def validate_thresholds(strong: float, partial: float) -> ValidationResult:
    result = ValidationResult(is_valid=True)
    if not (0.0 <= partial <= 1.0) or not (0.0 <= strong <= 1.0):
        result.add_error("Thresholds must be between 0.0 and 1.0.")
    elif partial >= strong:
        result.add_error(
            "The 'Partial' threshold must be lower than the 'Strong' "
            "threshold."
        )
    return result


def validate_weights(tfidf_weight: float, semantic_weight: float) -> ValidationResult:
    result = ValidationResult(is_valid=True)
    total = tfidf_weight + semantic_weight
    if tfidf_weight < 0 or semantic_weight < 0:
        result.add_error("Weights cannot be negative.")
    elif abs(total - 1.0) > 0.01:
        result.add_warning(
            f"TF-IDF weight + semantic weight = {total:.2f}, which does not "
            f"sum to 1.0. Scores will be normalized automatically."
        )
    return result
