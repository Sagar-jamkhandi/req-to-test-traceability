"""Shared constants and small helper utilities used across the app.

Keeping these in one place avoids magic strings scattered through the
codebase and makes the coverage/threshold vocabulary consistent between
the matching engine, the UI pages and the tests.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

import pandas as pd

# --------------------------------------------------------------------------
# Coverage status vocabulary
# --------------------------------------------------------------------------
STATUS_STRONG = "Strong"
STATUS_PARTIAL = "Partial"
STATUS_MISSING = "Missing"

STATUS_ORDER = [STATUS_STRONG, STATUS_PARTIAL, STATUS_MISSING]

STATUS_COLORS = {
    STATUS_STRONG: "#1E8E5A",    # green
    STATUS_PARTIAL: "#B8860B",   # amber
    STATUS_MISSING: "#C0392B",   # red
}

STATUS_BADGES = {
    STATUS_STRONG: "🟢 Strong",
    STATUS_PARTIAL: "🟠 Partial",
    STATUS_MISSING: "🔴 Missing",
}

# Default configuration. All of these are exposed as adjustable controls in
# the UI - they are heuristics, not universal QA standards.
DEFAULT_TFIDF_WEIGHT = 0.30
DEFAULT_SEMANTIC_WEIGHT = 0.70
DEFAULT_STRONG_THRESHOLD = 0.75
DEFAULT_PARTIAL_THRESHOLD = 0.50
DEFAULT_TOP_N = 5

# Requirement/test-case column aliases -> canonical column name.
# Matching is case-insensitive and ignores surrounding whitespace/punctuation.
REQUIREMENT_COLUMN_ALIASES: Dict[str, List[str]] = {
    "requirement_id": ["requirement_id", "req_id", "requirement id", "req id", "id"],
    "title": ["title", "requirement title", "name", "summary"],
    "description": ["description", "requirement description", "details", "desc", "text"],
    "priority": ["priority", "req_priority", "importance"],
}

TEST_CASE_COLUMN_ALIASES: Dict[str, List[str]] = {
    "test_case_id": ["test_case_id", "test_id", "tc_id", "test case id", "tc id", "id"],
    "title": ["title", "test title", "name", "summary"],
    "description": ["description", "test description", "details", "desc", "text"],
    "priority": ["priority", "test_priority", "importance"],
}

REQUIRED_REQUIREMENT_COLUMNS = ["requirement_id", "title", "description"]
REQUIRED_TEST_CASE_COLUMNS = ["test_case_id", "title", "description"]


def _normalize_header(name: str) -> str:
    """Lowercase and collapse punctuation/whitespace for fuzzy header matching."""
    return re.sub(r"[\s_\-]+", " ", str(name).strip().lower()).strip()


def normalize_columns(df: pd.DataFrame, alias_map: Dict[str, List[str]]) -> pd.DataFrame:
    """Rename columns of ``df`` to their canonical names using ``alias_map``.

    Unrecognized columns are left untouched (they simply won't be used).
    If several source columns resolve to the same canonical name, the first
    one encountered wins.
    """
    normalized_lookup: Dict[str, str] = {}
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            normalized_lookup[_normalize_header(alias)] = canonical

    rename_map: Dict[str, str] = {}
    seen_canonical = set()
    for col in df.columns:
        key = _normalize_header(col)
        canonical = normalized_lookup.get(key)
        if canonical and canonical not in seen_canonical:
            rename_map[col] = canonical
            seen_canonical.add(canonical)

    return df.rename(columns=rename_map)


def find_missing_columns(df: pd.DataFrame, required: Iterable[str]) -> List[str]:
    return [col for col in required if col not in df.columns]


def safe_str(value) -> str:
    """Coerce a possibly-NaN/None cell value to a clean string."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
