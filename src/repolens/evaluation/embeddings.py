"""Embedding utilities for evaluation baselines.

Provides deterministic, reproducible embedding generation without external
network dependencies, as well as factory support for SentenceTransformerProvider.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from repolens.retrieval.semantic_retriever import (
    EmbeddingProvider,
    SentenceTransformerProvider,
)


class DeterministicHashingEmbeddingProvider:
    """Reproducible, self-contained embedding provider using feature hashing.

    Produces fixed-dimension unit-normalized dense vectors based on term
    frequencies and character n-grams. Does not require downloading model weights
    or GPU acceleration, making benchmark tests fast, isolated, and deterministic.
    """

    def __init__(self, dimension: int = 64) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be a positive integer.")
        self.dimension = dimension

    def _hash_token(self, token: str) -> tuple[int, float]:
        """Map token to a feature index and sign using md5."""
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % self.dimension
        sign = 1.0 if (digest[4] % 2 == 0) else -1.0
        return idx, sign

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode each text into a normalized dense embedding vector."""
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dimension
            if not isinstance(text, str) or not text.strip():
                # Zero vector fallback
                embeddings.append(vec)
                continue

            # Tokenize into words and sub-tokens
            tokens = re.findall(r"[A-Za-z0-9_]{2,}", text.lower())
            if not tokens:
                tokens = [text.strip().lower()]

            for token in tokens:
                idx, sign = self._hash_token(token)
                vec[idx] += sign

                # Also hash 3-grams for subword matching
                if len(token) >= 4:
                    for i in range(len(token) - 2):
                        tri = token[i : i + 3]
                        t_idx, t_sign = self._hash_token(tri)
                        vec[t_idx] += t_sign * 0.5

            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 1e-12:
                vec = [x / norm for x in vec]
            embeddings.append(vec)

        return embeddings


def get_default_embedding_provider(
    prefer_sentence_transformers: bool = False,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> EmbeddingProvider:
    """Return embedding provider based on availability and preferences."""
    if prefer_sentence_transformers:
        try:
            return SentenceTransformerProvider(model_name=model_name)
        except Exception:
            pass
    return DeterministicHashingEmbeddingProvider(dimension=64)
