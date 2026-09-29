"""Text preprocessing for requirement / test-case records.

The cleaning here is intentionally light-touch: QA requirement text is
short and domain-term-dense ("OAuth", "SSO", "2FA", ...), so we avoid
stemming, stop-word removal, or lemmatization that could destroy meaning
that the matching models rely on. We only:

  * lowercase
  * normalize whitespace
  * strip most punctuation (keeping characters that carry meaning such as
    numbers) while collapsing repeated punctuation
  * drop rows with no usable text
  * flag / drop duplicate records
"""
from __future__ import annotations

import re
from typing import Tuple

import pandas as pd

from src.utils import safe_str

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def clean_text(text: str) -> str:
    """Lowercase, normalize whitespace and strip punctuation from ``text``.

    Numbers and words are preserved so that domain terms (e.g. "2FA",
    "OAuth2") remain understandable to both the TF-IDF and semantic
    matchers.
    """
    text = safe_str(text)
    if not text:
        return ""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def build_combined_text(title: str, description: str) -> str:
    """Combine title + description into one text blob for matching."""
    title = safe_str(title)
    description = safe_str(description)
    combined = f"{title}. {description}".strip(". ").strip()
    return combined


def preprocess_dataframe(
    df: pd.DataFrame,
    id_column: str,
    drop_empty_text: bool = True,
    drop_duplicate_ids: bool = True,
    drop_duplicate_content: bool = False,
) -> pd.DataFrame:
    """Clean and de-duplicate a requirements/test-cases dataframe.

    Adds two columns:
      - ``combined_text_raw``: title + description, human readable
      - ``combined_text_clean``: lowercased/punctuation-stripped version fed
        to the matchers

    Returns a new dataframe (input is not mutated).
    """
    out = df.copy()

    # Normalize identifier and text columns to plain strings.
    out[id_column] = out[id_column].apply(safe_str)
    out["title"] = out["title"].apply(safe_str) if "title" in out.columns else ""
    out["description"] = (
        out["description"].apply(safe_str) if "description" in out.columns else ""
    )
    if "priority" in out.columns:
        out["priority"] = out["priority"].apply(
            lambda v: safe_str(v) or "Not set"
        )
    else:
        out["priority"] = "Not set"

    # Drop rows with a blank ID - they cannot be traced.
    out = out[out[id_column] != ""].reset_index(drop=True)

    out["combined_text_raw"] = out.apply(
        lambda r: build_combined_text(r["title"], r["description"]), axis=1
    )
    out["combined_text_clean"] = out["combined_text_raw"].apply(clean_text)

    if drop_empty_text:
        out = out[out["combined_text_clean"].str.len() > 0].reset_index(drop=True)

    if drop_duplicate_ids:
        out = out.drop_duplicates(subset=[id_column], keep="last").reset_index(
            drop=True
        )

    if drop_duplicate_content:
        out = out.drop_duplicates(
            subset=["combined_text_clean"], keep="first"
        ).reset_index(drop=True)

    return out


def preprocess_requirements(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return preprocess_dataframe(df, id_column="requirement_id", **kwargs)


def preprocess_test_cases(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return preprocess_dataframe(df, id_column="test_case_id", **kwargs)
