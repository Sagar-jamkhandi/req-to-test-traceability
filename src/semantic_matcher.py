"""Semantic matching using Sentence Transformer embeddings.

If the model cannot be loaded (no internet access, disk issues, etc.) the
matcher degrades gracefully to the TF-IDF score instead of crashing the
app - the caller is told this happened via ``SemanticMatcher.model_loaded``
so the UI can be transparent about it.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.tfidf_matcher import TfidfMatcher

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticMatcher:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.model_loaded = False
        self.load_error: Optional[str] = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            self.model_loaded = True
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see fallback
            self.model = None
            self.model_loaded = False
            self.load_error = str(exc)

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self.model_loaded or self.model is None:
            raise RuntimeError(
                "Semantic model is not loaded; call score_matrix() which "
                "handles the TF-IDF fallback automatically."
            )
        if len(texts) == 0:
            return np.zeros((0, 384))
        return self.model.encode(list(texts), show_progress_bar=False, normalize_embeddings=True)

    def score_matrix(
        self, requirement_texts: List[str], test_case_texts: List[str]
    ) -> np.ndarray:
        """Return an (n_requirements x n_test_cases) semantic similarity matrix.

        Falls back to TF-IDF similarity if the transformer model failed to
        load, so the app always produces a result.
        """
        if len(requirement_texts) == 0 or len(test_case_texts) == 0:
            return np.zeros((len(requirement_texts), len(test_case_texts)))

        if not self.model_loaded:
            return TfidfMatcher().score_matrix(requirement_texts, test_case_texts)

        req_emb = self.encode(requirement_texts)
        test_emb = self.encode(test_case_texts)
        similarity = cosine_similarity(req_emb, test_emb)
        # Embeddings are normalized so cosine similarity is in [-1, 1];
        # clip negative "opposite meaning" scores to 0 for a 0-1 UI scale.
        return np.clip(similarity, 0.0, 1.0)
