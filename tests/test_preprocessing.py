import pandas as pd

from src.preprocessing import clean_text, preprocess_requirements


def test_clean_text_lowercases_and_strips_punctuation():
    assert clean_text("User's Login!!") == "user s login"


def test_clean_text_normalizes_whitespace():
    assert clean_text("  too   many    spaces  ") == "too many spaces"


def test_clean_text_handles_none_and_nan():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_preprocess_requirements_drops_blank_ids():
    df = pd.DataFrame(
        {
            "requirement_id": ["REQ-001", ""],
            "title": ["Login", "Something"],
            "description": ["Users can log in", "Some description"],
        }
    )
    out = preprocess_requirements(df)
    assert len(out) == 1
    assert out.iloc[0]["requirement_id"] == "REQ-001"


def test_preprocess_requirements_dedupes_ids_keeping_last():
    df = pd.DataFrame(
        {
            "requirement_id": ["REQ-001", "REQ-001"],
            "title": ["Old title", "New title"],
            "description": ["Old description here", "New description here"],
        }
    )
    out = preprocess_requirements(df)
    assert len(out) == 1
    assert out.iloc[0]["title"] == "New title"


def test_preprocess_requirements_adds_combined_text(sample_requirements_df):
    out = preprocess_requirements(sample_requirements_df)
    assert "combined_text_clean" in out.columns
    assert all(out["combined_text_clean"].str.len() > 0)


def test_preprocess_requirements_drops_empty_text_rows():
    df = pd.DataFrame(
        {
            "requirement_id": ["REQ-001", "REQ-002"],
            "title": ["", ""],
            "description": ["", ""],
        }
    )
    out = preprocess_requirements(df)
    assert len(out) == 0
