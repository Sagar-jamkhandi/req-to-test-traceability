"""File ingestion: reading uploaded CSV/XLSX files and bundled sample data."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.utils import (
    REQUIREMENT_COLUMN_ALIASES,
    TEST_CASE_COLUMN_ALIASES,
    normalize_columns,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_REQUIREMENTS_PATH = DATA_DIR / "sample_requirements.csv"
SAMPLE_TEST_CASES_PATH = DATA_DIR / "sample_test_cases.csv"


class DataLoadError(Exception):
    """Raised when a file cannot be parsed at all (bad format, unreadable)."""


def _read_any(file_obj_or_path: Union[str, Path, io.BytesIO], filename: Optional[str] = None) -> pd.DataFrame:
    """Read a CSV or XLSX file from a path or an in-memory buffer.

    ``filename`` is used to detect the format when reading from an
    in-memory buffer (e.g. a Streamlit ``UploadedFile``), which exposes its
    own ``.name`` attribute we fall back to if not provided.
    """
    name = filename
    if name is None:
        name = getattr(file_obj_or_path, "name", None) or str(file_obj_or_path)
    ext = Path(name).suffix.lower()

    try:
        if ext in (".xlsx", ".xlsm", ".xls"):
            return pd.read_excel(file_obj_or_path)
        # default to CSV (also handles .txt/.csv or no visible extension)
        return pd.read_csv(file_obj_or_path)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError("The uploaded file is empty.") from exc
    except Exception as exc:  # noqa: BLE001 - surfaced as a friendly UI message
        raise DataLoadError(f"Could not read '{name}': {exc}") from exc


def load_requirements(file_obj_or_path, filename: Optional[str] = None) -> pd.DataFrame:
    df = _read_any(file_obj_or_path, filename)
    return normalize_columns(df, REQUIREMENT_COLUMN_ALIASES)


def load_test_cases(file_obj_or_path, filename: Optional[str] = None) -> pd.DataFrame:
    df = _read_any(file_obj_or_path, filename)
    return normalize_columns(df, TEST_CASE_COLUMN_ALIASES)


def load_sample_requirements() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_REQUIREMENTS_PATH)
    return normalize_columns(df, REQUIREMENT_COLUMN_ALIASES)


def load_sample_test_cases() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_TEST_CASES_PATH)
    return normalize_columns(df, TEST_CASE_COLUMN_ALIASES)


def requirement_template_csv() -> bytes:
    df = pd.DataFrame(
        {
            "requirement_id": ["REQ-001"],
            "title": ["Short requirement title"],
            "description": ["Full description of the requirement / user story."],
            "priority": ["High"],
        }
    )
    return df.to_csv(index=False).encode("utf-8")


def test_case_template_csv() -> bytes:
    df = pd.DataFrame(
        {
            "test_case_id": ["TC-001"],
            "title": ["Short test case title"],
            "description": ["Steps / expected result summary for the test case."],
            "priority": ["High"],
        }
    )
    return df.to_csv(index=False).encode("utf-8")
