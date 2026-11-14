from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from repolens.ingestion.code_unit import CodeUnit


class EmbeddingProvider(Protocol):
    """Provider-independent interface for generating text embeddings."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Generate one embedding vector for each input text."""
        ...


class SentenceTransformerProvider:
    """Local embedding provider backed by Sentence Transformers.

    The model is loaded only when the first embedding request is made.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty.")

        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Semantic retrieval requires the optional dependencies. "
                    'Install them with: pip install -e ".[semantic]"'
                ) from exc

            self._model = SentenceTransformer(self.model_name)

        return self._model

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Encode text using the configured local embedding model."""
        if not texts:
            return []

        model = self._get_model()

        return model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


@dataclass(frozen=True)
class SemanticSearchResult:
    """A ranked semantic retrieval result."""

    code_unit: CodeUnit
    score: float
    rank: int


def _make_document(unit: CodeUnit) -> str:
    """Build a searchable semantic document from a CodeUnit."""
    parts = [
        f"Name: {unit.name}",
        f"Qualified name: {unit.qualified_name}",
        f"Kind: {unit.kind}",
        f"File: {unit.file_path}",
    ]

    if unit.docstring:
        parts.append(f"Docstring: {unit.docstring}")

    if unit.imports:
        parts.append("Imports: " + " ".join(imp.module for imp in unit.imports))

    if unit.calls:
        parts.append("Calls: " + " ".join(call.target for call in unit.calls))

    for key, value in unit.metadata.items():
        if isinstance(value, str) and value.strip():
            parts.append(f"{key}: {value}")

    if unit.source_code:
        parts.append(f"Source code: {unit.source_code}")

    return "\n".join(parts)


class EmbeddingSemanticRetriever:
    """Semantic search over repository CodeUnits using text embeddings.

    Embeddings are generated lazily and cached for subsequent queries.
    Similarity is calculated using cosine similarity.
    """

    def __init__(
        self,
        code_units: Iterable[CodeUnit],
        embedding_provider: EmbeddingProvider,
    ) -> None:
        units = list(code_units)

        if not all(isinstance(unit, CodeUnit) for unit in units):
            raise TypeError("Expected CodeUnit instances.")

        if not callable(getattr(embedding_provider, "encode", None)):
            raise TypeError("Embedding provider must implement encode().")

        seen_ids: set[str] = set()
        self._code_units: list[CodeUnit] = []

        for unit in units:
            if unit.id in seen_ids:
                continue

            seen_ids.add(unit.id)
            self._code_units.append(unit)

        self._provider = embedding_provider
        self._documents = [_make_document(unit) for unit in self._code_units]
        self._document_embeddings: list[list[float]] | None = None

    @property
    def document_count(self) -> int:
        """Number of indexed CodeUnits."""
        return len(self._code_units)

    @property
    def is_indexed(self) -> bool:
        """Whether document embeddings have been generated."""
        return self._document_embeddings is not None

    def _encode(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Generate and validate embedding vectors."""
        raw_embeddings = self._provider.encode(texts)

        if len(raw_embeddings) != len(texts):
            raise ValueError(
                "Embedding provider returned an unexpected number of vectors."
            )

        embeddings: list[list[float]] = []

        expected_dimension: int | None = None

        for vector in raw_embeddings:
            values = [float(value) for value in vector]

            if not values:
                raise ValueError("Embedding vectors cannot be empty.")

            if not all(math.isfinite(value) for value in values):
                raise ValueError("Embedding vectors must contain finite values.")

            if expected_dimension is None:
                expected_dimension = len(values)
            elif len(values) != expected_dimension:
                raise ValueError("Embedding vectors must have equal dimensions.")

            embeddings.append(values)

        return embeddings

    def _ensure_index(self) -> None:
        """Generate document embeddings once and cache them."""
        if self._document_embeddings is None:
            if not self._documents:
                self._document_embeddings = []
                return

            self._document_embeddings = self._encode(self._documents)

    @staticmethod
    def _cosine_similarity(
        left: Sequence[float],
        right: Sequence[float],
    ) -> float:
        """Calculate cosine similarity between two embedding vectors."""
        if len(left) != len(right):
            raise ValueError("Embedding dimensions do not match.")

        dot_product = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))

        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0

        return dot_product / (left_norm * right_norm)

    def query(
        self,
        query_text: str,
        *,
        top_k: int = 10,
    ) -> list[SemanticSearchResult]:
        """Return the top-k CodeUnits ranked by semantic similarity."""
        if not isinstance(query_text, str):
            raise TypeError("query_text must be a string.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        if not query_text.strip() or not self._code_units:
            return []

        self._ensure_index()

        query_embedding = self._encode([query_text])[0]

        if not self._document_embeddings:
            return []

        if len(query_embedding) != len(self._document_embeddings[0]):
            raise ValueError("Query and document embedding dimensions do not match.")

        scored = [
            (
                self._cosine_similarity(query_embedding, document_embedding),
                index,
            )
            for index, document_embedding in enumerate(self._document_embeddings)
        ]

        scored.sort(key=lambda item: item[0], reverse=True)

        return [
            SemanticSearchResult(
                code_unit=self._code_units[index],
                score=score,
                rank=rank,
            )
            for rank, (score, index) in enumerate(scored[:top_k], start=1)
        ]
