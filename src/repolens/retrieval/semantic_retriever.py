from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from repolens.ingestion.code_unit import CodeUnit

logger = logging.getLogger(__name__)

# Optional FAISS and NumPy imports
try:
    import faiss
    import numpy as np

    _FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    _FAISS_AVAILABLE = False


class EmbeddingError(Exception):
    """Raised when text embedding generation or retrieval fails."""


class EmbeddingProvider(Protocol):
    """Provider-independent interface for generating text embeddings."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Generate one embedding vector for each input text."""
        ...


class SentenceTransformerProvider:
    """Local embedding provider backed by Sentence Transformers.

    The model is loaded lazily when the first embedding request is made.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name cannot be empty.")

        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer.")

        self.model_name = model_name.strip()
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Semantic retrieval requires the optional dependencies. "
                    'Install them with: pip install -e ".[semantic]"'
                ) from exc

            kwargs: dict[str, Any] = {}
            if self.device is not None:
                kwargs["device"] = self.device

            self._model = SentenceTransformer(self.model_name, **kwargs)

        return self._model

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Encode text using the configured local embedding model."""
        if not texts:
            return []

        for item in texts:
            if not isinstance(item, str):
                raise TypeError("All items to encode must be strings.")

        model = self._get_model()

        try:
            embeddings = model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return embeddings
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to generate embeddings using {self.model_name}: {exc}"
            ) from exc


class VectorStore(Protocol):
    """Protocol for vector stores supporting similarity search."""

    def add(
        self,
        vectors: Sequence[Sequence[float]],
        code_units: Sequence[CodeUnit],
    ) -> None:
        """Add embedding vectors and associated CodeUnits to the store."""
        ...

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 10,
    ) -> list[tuple[CodeUnit, float]]:
        """Search for top_k most similar CodeUnits, returning (code_unit, score) pairs."""
        ...

    @property
    def count(self) -> int:
        """Total number of items in the vector store."""
        ...

    def clear(self) -> None:
        """Clear all indexed vectors and CodeUnits."""
        ...


class FaissVectorStore:
    """FAISS-backed vector store for CodeUnit embeddings using cosine similarity.

    Vectors are L2-normalized so that inner product corresponds to cosine similarity
    in the range [-1.0, 1.0]. Source metadata is preserved for all indexed units.
    """

    def __init__(
        self,
        dimension: int | None = None,
        *,
        normalize: bool = True,
    ) -> None:
        if not _FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is required for FaissVectorStore. "
                "Install it with: pip install faiss-cpu"
            )

        if dimension is not None and (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 1
        ):
            raise ValueError("dimension must be a positive integer.")

        self.dimension = dimension
        self.normalize = normalize
        self._index: Any = None
        self._code_units: list[CodeUnit] = []
        self._metadata: list[dict[str, Any]] = []

        if dimension is not None:
            self._index = faiss.IndexFlatIP(dimension)

    @property
    def count(self) -> int:
        """Total number of items stored in the FAISS index."""
        if self._index is None:
            return 0
        return int(self._index.ntotal)

    def _normalize_vectors(self, vectors_array: np.ndarray) -> np.ndarray:
        """L2-normalize vectors so inner product equals cosine similarity."""
        if not self.normalize:
            return vectors_array
        norms = np.linalg.norm(vectors_array, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return vectors_array / norms

    def add(
        self,
        vectors: Sequence[Sequence[float]],
        code_units: Sequence[CodeUnit],
    ) -> None:
        """Add embedding vectors and associated CodeUnits to the FAISS index."""
        if len(vectors) != len(code_units):
            raise ValueError("Number of vectors and CodeUnits must match.")

        if not vectors:
            return

        arr = np.array(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("Vectors must be a 2D sequence of numbers.")

        if not np.all(np.isfinite(arr)):
            raise ValueError("Embedding vectors must contain finite values.")

        dim = arr.shape[1]
        if dim == 0:
            raise ValueError("Embedding vectors cannot be empty.")

        if self.dimension is None:
            self.dimension = dim
            self._index = faiss.IndexFlatIP(dim)
        elif dim != self.dimension:
            raise ValueError(
                f"Vector dimension {dim} does not match store dimension {self.dimension}."
            )

        norm_arr = self._normalize_vectors(arr)
        contiguous_arr = np.ascontiguousarray(norm_arr, dtype=np.float32)
        self._index.add(contiguous_arr)

        for unit in code_units:
            self._code_units.append(unit)
            self._metadata.append(
                {
                    "id": unit.id,
                    "name": unit.name,
                    "qualified_name": unit.qualified_name,
                    "kind": unit.kind,
                    "file_path": unit.file_path,
                    "start_line": unit.start_line,
                    "end_line": unit.end_line,
                    "docstring": unit.docstring,
                    "source_code": unit.source_code,
                    "metadata": unit.metadata,
                }
            )

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 10,
    ) -> list[tuple[CodeUnit, float]]:
        """Search for top_k most similar CodeUnits using FAISS inner product."""
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        q_arr = np.array(query_vector, dtype=np.float32)
        if q_arr.ndim == 1:
            q_arr = q_arr.reshape(1, -1)
        elif q_arr.ndim != 2 or q_arr.shape[0] != 1:
            raise ValueError("Query vector must be a 1D sequence or (1, D) array.")

        if not np.all(np.isfinite(q_arr)):
            raise ValueError("Query vector must contain finite values.")

        if self.dimension is not None and q_arr.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {q_arr.shape[1]} does not match "
                f"index dimension {self.dimension}."
            )

        if self.count == 0 or self._index is None:
            return []

        norm_q = self._normalize_vectors(q_arr)
        contiguous_q = np.ascontiguousarray(norm_q, dtype=np.float32)

        k = min(top_k, self.count)
        distances, indices = self._index.search(contiguous_q, k)

        results: list[tuple[CodeUnit, float]] = []
        for dist, idx in zip(distances[0], indices[0], strict=True):
            if idx < 0 or idx >= len(self._code_units):
                continue
            results.append((self._code_units[idx], float(dist)))

        return results

    def get_metadata(self, index: int) -> dict[str, Any]:
        """Return preserved source metadata for an indexed CodeUnit."""
        if not 0 <= index < len(self._metadata):
            raise IndexError("Metadata index out of bounds.")
        return self._metadata[index]

    def clear(self) -> None:
        """Reset the index and all stored CodeUnits and metadata."""
        if self.dimension is not None:
            self._index = faiss.IndexFlatIP(self.dimension)
        else:
            self._index = None
        self._code_units.clear()
        self._metadata.clear()


class InMemoryVectorStore:
    """Pure-Python vector store fallback computing exact cosine similarity."""

    def __init__(self) -> None:
        self._vectors: list[list[float]] = []
        self._code_units: list[CodeUnit] = []
        self._metadata: list[dict[str, Any]] = []

    @property
    def count(self) -> int:
        """Total number of items in the vector store."""
        return len(self._code_units)

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        """Calculate cosine similarity between two embedding vectors."""
        if len(left) != len(right):
            raise ValueError("Embedding dimensions do not match.")

        dot_product = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))

        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0

        return dot_product / (left_norm * right_norm)

    def add(
        self,
        vectors: Sequence[Sequence[float]],
        code_units: Sequence[CodeUnit],
    ) -> None:
        """Add embedding vectors and associated CodeUnits to the in-memory store."""
        if len(vectors) != len(code_units):
            raise ValueError("Number of vectors and CodeUnits must match.")

        for vec, unit in zip(vectors, code_units, strict=True):
            vals = [float(v) for v in vec]
            if not vals:
                raise ValueError("Embedding vectors cannot be empty.")
            if not all(math.isfinite(v) for v in vals):
                raise ValueError("Embedding vectors must contain finite values.")
            if self._vectors and len(vals) != len(self._vectors[0]):
                raise ValueError("Embedding vectors must have equal dimensions.")

            self._vectors.append(vals)
            self._code_units.append(unit)
            self._metadata.append(
                {
                    "id": unit.id,
                    "name": unit.name,
                    "qualified_name": unit.qualified_name,
                    "kind": unit.kind,
                    "file_path": unit.file_path,
                    "start_line": unit.start_line,
                    "end_line": unit.end_line,
                    "docstring": unit.docstring,
                    "source_code": unit.source_code,
                    "metadata": unit.metadata,
                }
            )

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 10,
    ) -> list[tuple[CodeUnit, float]]:
        """Search for top_k most similar CodeUnits using cosine similarity."""
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        if not self._vectors:
            return []

        q_vals = [float(v) for v in query_vector]
        if len(q_vals) != len(self._vectors[0]):
            raise ValueError("Query and document embedding dimensions do not match.")

        scored = [
            (self._code_units[i], self._cosine_similarity(q_vals, doc_vec))
            for i, doc_vec in enumerate(self._vectors)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def get_metadata(self, index: int) -> dict[str, Any]:
        """Return preserved source metadata for an indexed CodeUnit."""
        if not 0 <= index < len(self._metadata):
            raise IndexError("Metadata index out of bounds.")
        return self._metadata[index]

    def clear(self) -> None:
        """Reset the vector store."""
        self._vectors.clear()
        self._code_units.clear()
        self._metadata.clear()


@dataclass(frozen=True)
class SemanticSearchResult:
    """A ranked semantic retrieval result preserving CodeUnit source metadata."""

    code_unit: CodeUnit
    score: float
    rank: int

    @property
    def code_unit_id(self) -> str:
        """Unique identifier of the CodeUnit."""
        return self.code_unit.id

    @property
    def name(self) -> str:
        """Symbol name of the CodeUnit."""
        return self.code_unit.name

    @property
    def symbol(self) -> str:
        """Symbol name alias."""
        return self.code_unit.name

    @property
    def qualified_name(self) -> str:
        """Fully-qualified symbol name."""
        return self.code_unit.qualified_name

    @property
    def kind(self) -> str:
        """Kind of code construct (e.g., function, class, method, module)."""
        return self.code_unit.kind

    @property
    def file_path(self) -> str:
        """Relative file path of the source file containing this unit."""
        return self.code_unit.file_path

    @property
    def start_line(self) -> int:
        """1-based start line number in the source file."""
        return self.code_unit.start_line

    @property
    def end_line(self) -> int:
        """1-based end line number in the source file."""
        return self.code_unit.end_line

    @property
    def source_code(self) -> str:
        """Source code snippet of the unit."""
        return self.code_unit.source_code

    @property
    def docstring(self) -> str:
        """Docstring associated with the unit."""
        return self.code_unit.docstring

    @property
    def metadata(self) -> dict[str, Any]:
        """Custom metadata associated with the CodeUnit."""
        return self.code_unit.metadata


def _make_document(unit: CodeUnit) -> str:
    """Build a searchable semantic document representation from a CodeUnit."""
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

    Embeddings are generated lazily and stored in a VectorStore (FAISS by default
    when available, or InMemoryVectorStore fallback).
    """

    def __init__(
        self,
        code_units: Iterable[CodeUnit],
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore | None = None,
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
        self._is_indexed = False

        if vector_store is not None:
            self._vector_store = vector_store
        elif _FAISS_AVAILABLE:
            self._vector_store = FaissVectorStore()
        else:
            self._vector_store = InMemoryVectorStore()

    @property
    def document_count(self) -> int:
        """Number of indexed CodeUnits."""
        return len(self._code_units)

    @property
    def is_indexed(self) -> bool:
        """Whether document embeddings have been generated and indexed."""
        return self._is_indexed

    @property
    def vector_store(self) -> VectorStore:
        """The underlying vector store instance."""
        return self._vector_store

    def _encode(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Generate and validate embedding vectors."""
        if not texts:
            return []

        try:
            raw_embeddings = self._provider.encode(texts)
        except Exception as exc:
            if isinstance(exc, (ValueError, TypeError)):
                raise
            raise EmbeddingError(
                f"Embedding provider failed during encoding: {exc}"
            ) from exc

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
        """Generate document embeddings once and cache them in the vector store."""
        if not self._is_indexed:
            if not self._documents:
                self._is_indexed = True
                return

            embeddings = self._encode(self._documents)
            self._vector_store.add(embeddings, self._code_units)
            self._is_indexed = True

    def encode_code_units(
        self,
        code_units: Sequence[CodeUnit],
    ) -> list[list[float]]:
        """Generate embeddings for a sequence of extracted CodeUnits."""
        if not code_units:
            return []

        if not all(isinstance(unit, CodeUnit) for unit in code_units):
            raise TypeError("Expected CodeUnit instances.")

        documents = [_make_document(unit) for unit in code_units]
        return self._encode(documents)

    def encode_query(self, query_text: str) -> list[float]:
        """Generate an embedding vector for an incoming user query."""
        if not isinstance(query_text, str):
            raise TypeError("query_text must be a string.")

        if not query_text.strip():
            raise ValueError("query_text cannot be empty.")

        vectors = self._encode([query_text])
        return vectors[0]

    def query(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        safe: bool = False,
        min_score: float | None = None,
    ) -> list[SemanticSearchResult]:
        """Return the top-k CodeUnits ranked by semantic similarity.

        If safe is True, embedding provider failures are logged and an empty
        result list is returned gracefully instead of raising.
        """
        if not isinstance(query_text, str):
            raise TypeError("query_text must be a string.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        if not query_text.strip() or not self._code_units:
            return []

        try:
            self._ensure_index()
            query_embedding = self._encode([query_text])[0]
            raw_results = self._vector_store.search(query_embedding, top_k=top_k)
        except Exception as exc:
            if not safe:
                raise
            logger.warning("Semantic retrieval query failed: %s", exc)
            return []

        results: list[SemanticSearchResult] = []
        rank = 1

        for unit, score in raw_results:
            if min_score is not None and score < min_score:
                continue

            results.append(
                SemanticSearchResult(
                    code_unit=unit,
                    score=score,
                    rank=rank,
                )
            )
            rank += 1

        return results

    def safe_query(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        min_score: float | None = None,
    ) -> list[SemanticSearchResult]:
        """Graceful query that catches embedding failures and returns an empty list."""
        return self.query(query_text, top_k=top_k, safe=True, min_score=min_score)
