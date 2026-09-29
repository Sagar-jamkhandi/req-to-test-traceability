"""Baseline lexical matching: TF-IDF + cosine similarity."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfMatcher:
    """Fits a single TF-IDF vocabulary across requirements + test cases,
    then scores every requirement against every test case.

    Fitting on the *combined* corpus (rather than fitting separately on
    each side) ensures both sets share the same vocabulary/IDF weights,
    which is required for cosine similarity to be meaningful.
    """

    def __init__(self, min_df: int = 1, ngram_range: Tuple[int, int] = (1, 2)):
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.vectorizer: TfidfVectorizer | None = None

    def score_matrix(
        self, requirement_texts: List[str], test_case_texts: List[str]
    ) -> np.ndarray:
        """Return an (n_requirements x n_test_cases) similarity matrix.

        Scores are cosine similarities in [0, 1] (TF-IDF vectors are
        non-negative so similarity cannot be negative).
        """
        if len(requirement_texts) == 0 or len(test_case_texts) == 0:
            return np.zeros((len(requirement_texts), len(test_case_texts)))

        corpus = list(requirement_texts) + list(test_case_texts)
        # min_df must not exceed the number of documents in tiny corpora
        effective_min_df = min(self.min_df, max(1, len(corpus) - 1))
        self.vectorizer = TfidfVectorizer(
            min_df=effective_min_df, ngram_range=self.ngram_range
        )

        try:
            tfidf = self.vectorizer.fit_transform(corpus)
        except ValueError:
            # Empty vocabulary (e.g. all-empty strings after cleaning)
            return np.zeros((len(requirement_texts), len(test_case_texts)))

        n_req = len(requirement_texts)
        req_vectors = tfidf[:n_req]
        test_vectors = tfidf[n_req:]

        similarity = cosine_similarity(req_vectors, test_vectors)
        return np.clip(similarity, 0.0, 1.0)
