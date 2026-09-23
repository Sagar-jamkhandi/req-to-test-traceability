import pandas as pd

from src.validation import (
    validate_requirements,
    validate_test_cases,
    validate_thresholds,
    validate_weights,
)


def test_valid_requirements_pass(sample_requirements_df):
    result = validate_requirements(sample_requirements_df)
    assert result.is_valid
    assert result.errors == []


def test_missing_columns_reported():
    df = pd.DataFrame({"requirement_id": ["REQ-001"], "title": ["Login"]})
    result = validate_requirements(df)
    assert not result.is_valid
    assert any("description" in e for e in result.errors)


def test_empty_dataframe_reported():
    df = pd.DataFrame(columns=["requirement_id", "title", "description"])
    result = validate_requirements(df)
    assert not result.is_valid
    assert any("No requirement" in e for e in result.errors)


def test_duplicate_ids_warns():
    df = pd.DataFrame(
        {
            "requirement_id": ["REQ-001", "REQ-001"],
            "title": ["A", "B"],
            "description": ["Description one here", "Description two here"],
        }
    )
    result = validate_requirements(df)
    assert result.is_valid  # duplicates are a warning, not a hard error
    assert any("duplicate" in w.lower() for w in result.warnings)


def test_test_cases_valid(sample_test_cases_df):
    result = validate_test_cases(sample_test_cases_df)
    assert result.is_valid


def test_threshold_validation_rejects_partial_above_strong():
    result = validate_thresholds(strong=0.5, partial=0.6)
    assert not result.is_valid


def test_threshold_validation_accepts_valid_values():
    result = validate_thresholds(strong=0.75, partial=0.5)
    assert result.is_valid


def test_weight_validation_warns_on_non_unit_sum():
    result = validate_weights(tfidf_weight=0.3, semantic_weight=0.3)
    assert result.is_valid  # still valid, just a warning
    assert result.warnings
