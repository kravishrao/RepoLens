from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from repolens.ingestion.code_unit import CodeUnit
from repolens.retrieval.bm25_retriever import RetrievalResult
from repolens.retrieval.hybrid_retriever import HybridSearchResult
from repolens.retrieval.semantic_retriever import SemanticSearchResult

logger = logging.getLogger(__name__)


class RerankingError(Exception):
    """Raised when query-candidate reranking fails."""


class RerankerProvider(Protocol):
    """Provider-independent interface for scoring (query, candidate_document) pairs."""

    def predict(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        """Predict relevance scores for each (query, document) pair."""
        ...


class CrossEncoderProvider:
    """Local cross-encoder backed by Sentence Transformers CrossEncoder.

    The model is loaded lazily when the first predict request is made.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name cannot be empty.")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer.")
        if (
            not isinstance(max_length, int)
            or isinstance(max_length, bool)
            or max_length < 1
        ):
            raise ValueError("max_length must be a positive integer.")

        self.model_name = model_name.strip()
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "Cross-encoder reranking requires sentence-transformers. "
                    'Install it with: pip install -e ".[semantic]"'
                ) from exc

            kwargs: dict[str, Any] = {"max_length": self.max_length}
            if self.device is not None:
                kwargs["device"] = self.device

            self._model = CrossEncoder(self.model_name, **kwargs)

        return self._model

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Score (query, text) pairs using the cross-encoder model."""
        if not pairs:
            return []

        for pair in pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise TypeError(
                    "Each item must be a (query, text) tuple or 2-element list."
                )
            if not isinstance(pair[0], str) or not isinstance(pair[1], str):
                raise TypeError("Query and text must both be strings.")

        model = self._get_model()

        try:
            raw_scores = model.predict(
                list(pairs),
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            # Handle 1D or scalar outputs
            if hasattr(raw_scores, "tolist"):
                raw_scores = raw_scores.tolist()
            if isinstance(raw_scores, (int, float)):
                return [float(raw_scores)]
            return [float(s) for s in raw_scores]
        except Exception as exc:
            raise RerankingError(
                f"Cross-encoder prediction failed with model {self.model_name}: {exc}"
            ) from exc


class SymbolAwareRerankerProvider:
    """Lightweight, fast, deterministic heuristic reranker for code intelligence.

    Scores candidate relevance based on exact symbol matching, identifier token
    overlap, docstring matching, and qualified name matching. Requires zero
    external neural dependencies.
    """

    _WORD_PATTERN = re.compile(r"[a-zA-Z0-9]+")

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Score (query, text) pairs using symbol and lexical overlap heuristics."""
        if not pairs:
            return []

        scores: list[float] = []
        for query, text in pairs:
            q_tokens = {
                t.lower() for t in self._WORD_PATTERN.findall(query) if len(t) > 1
            }
            if not q_tokens:
                scores.append(0.0)
                continue

            text_lower = text.lower()
            text_tokens = set(self._WORD_PATTERN.findall(text_lower))

            # Exact phrase match bonus
            phrase_bonus = 2.0 if query.lower() in text_lower else 0.0

            # Symbol name line bonus (first line usually starts with Name:)
            first_line = text_lower.split("\n", 1)[0]
            symbol_matches = sum(1 for t in q_tokens if t in first_line)
            symbol_score = (symbol_matches / len(q_tokens)) * 3.0

            # Overall overlap
            overlap = len(q_tokens & text_tokens) / len(q_tokens)

            final_score = symbol_score + phrase_bonus + overlap
            scores.append(float(final_score))

        return scores


@dataclass(frozen=True)
class RerankResult:
    """A single reranked retrieval result preserving CodeUnit source metadata."""

    code_unit: CodeUnit
    score: float
    rank: int
    initial_score: float | None = None
    initial_rank: int | None = None
    source: str = "reranked"

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
        """Construct kind (e.g., function, class, method, module)."""
        return self.code_unit.kind

    @property
    def file_path(self) -> str:
        """Relative file path containing the symbol."""
        return self.code_unit.file_path

    @property
    def start_line(self) -> int:
        """1-based start line number."""
        return self.code_unit.start_line

    @property
    def end_line(self) -> int:
        """1-based end line number."""
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


def _make_rerank_text(unit: CodeUnit) -> str:
    """Build a rich text document for a candidate CodeUnit for reranker scoring."""
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
        # Use first 500 characters of source code to keep cross-encoder context focused
        code_snippet = (
            unit.source_code[:500] + "..."
            if len(unit.source_code) > 500
            else unit.source_code
        )
        parts.append(f"Source code: {code_snippet}")

    return "\n".join(parts)


def _extract_candidate(
    item: Any,
    default_rank: int,
) -> tuple[CodeUnit, float | None, int | None]:
    """Extract (code_unit, initial_score, initial_rank) from candidate items."""
    if isinstance(item, (RetrievalResult, SemanticSearchResult, HybridSearchResult)):
        return item.code_unit, float(item.score), int(item.rank)
    if isinstance(item, RerankResult):
        return item.code_unit, float(item.score), int(item.rank)
    if isinstance(item, CodeUnit):
        return item, None, default_rank
    if hasattr(item, "code_unit"):
        unit = item.code_unit
        score = float(item.score) if hasattr(item, "score") else None
        rank = int(getattr(item, "rank", default_rank))
        return unit, score, rank
    raise TypeError(
        f"Unsupported candidate type '{type(item).__name__}'. "
        "Expected CodeUnit, HybridSearchResult, RetrievalResult, or SemanticSearchResult."
    )


class CodeUnitReranker:
    """Reranking component to improve the ordering and relevance of candidate CodeUnits.

    Accepts candidates from the hybrid retrieval pipeline (or lexical/semantic
    retrievers) and evaluates joint query-code relevance using a cross-encoder
    or scoring model.
    """

    def __init__(
        self,
        provider: RerankerProvider | None = None,
        *,
        max_candidates: int | None = None,
        top_k: int = 10,
        fallback_on_failure: bool = True,
    ) -> None:
        if max_candidates is not None and (
            not isinstance(max_candidates, int)
            or isinstance(max_candidates, bool)
            or max_candidates < 1
        ):
            raise ValueError("max_candidates must be a positive integer.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        self.provider: RerankerProvider = (
            provider if provider is not None else SymbolAwareRerankerProvider()
        )
        self.max_candidates = max_candidates
        self.top_k = top_k
        self.fallback_on_failure = fallback_on_failure

    def rerank(
        self,
        query: str,
        candidates: Sequence[Any],
        *,
        top_k: int | None = None,
        max_candidates: int | None = None,
        safe: bool = True,
        min_score: float | None = None,
    ) -> list[RerankResult]:
        """Rerank candidate CodeUnits for a given query string.

        Parameters
        ----------
        query:
            The search query.
        candidates:
            Sequence of candidate CodeUnits or retrieval results (HybridSearchResult,
            RetrievalResult, SemanticSearchResult).
        top_k:
            Number of top reranked results to return. Defaults to instance top_k.
        max_candidates:
            Maximum number of initial candidates to evaluate. Defaults to instance max_candidates.
        safe:
            If True and the scoring provider fails, falls back gracefully to
            initial candidate ordering rather than raising.
        min_score:
            Optional minimum reranking score threshold.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        k_top = top_k if top_k is not None else self.top_k
        if not isinstance(k_top, int) or isinstance(k_top, bool) or k_top < 1:
            raise ValueError("top_k must be a positive integer.")

        k_max = max_candidates if max_candidates is not None else self.max_candidates
        if k_max is not None and (
            not isinstance(k_max, int) or isinstance(k_max, bool) or k_max < 1
        ):
            raise ValueError("max_candidates must be a positive integer.")

        if not query.strip() or not candidates:
            return []

        # Extract candidates and deduplicate by CodeUnit ID
        seen_ids: set[str] = set()
        extracted: list[tuple[CodeUnit, float | None, int | None]] = []

        for idx, item in enumerate(candidates, start=1):
            unit, initial_score, initial_rank = _extract_candidate(item, idx)
            if unit.id in seen_ids:
                continue
            seen_ids.add(unit.id)
            extracted.append((unit, initial_score, initial_rank))

        # Truncate candidates to k_max if configured
        if k_max is not None:
            extracted = extracted[:k_max]

        if not extracted:
            return []

        # Build (query, document) pairs
        pairs = [(query, _make_rerank_text(unit)) for unit, _, _ in extracted]

        # Score pairs
        try:
            scores = list(self.provider.predict(pairs))
            if len(scores) != len(pairs):
                raise ValueError(
                    f"Reranker provider returned {len(scores)} scores for {len(pairs)} pairs."
                )
        except Exception as exc:
            if not safe and not self.fallback_on_failure:
                if isinstance(exc, (ValueError, TypeError)):
                    raise
                raise RerankingError(f"Reranking failed: {exc}") from exc

            logger.warning(
                "Reranking scoring failed (%s); falling back to initial ranking.",
                exc,
            )
            # Graceful fallback: maintain initial candidate order
            return [
                RerankResult(
                    code_unit=unit,
                    score=initial_score if initial_score is not None else 0.0,
                    rank=new_rank,
                    initial_score=initial_score,
                    initial_rank=initial_rank,
                    source="fallback",
                )
                for new_rank, (unit, initial_score, initial_rank) in enumerate(
                    extracted[:k_top], start=1
                )
            ]

        # Sort candidates descending by rerank score
        scored_candidates = [
            (score, unit, init_score, init_rank)
            for score, (unit, init_score, init_rank) in zip(
                scores, extracted, strict=True
            )
        ]
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Apply min_score filter and top_k limit
        results: list[RerankResult] = []
        rank = 1
        for score, unit, init_score, init_rank in scored_candidates:
            if min_score is not None and score < min_score:
                continue
            results.append(
                RerankResult(
                    code_unit=unit,
                    score=score,
                    rank=rank,
                    initial_score=init_score,
                    initial_rank=init_rank,
                    source="reranked",
                )
            )
            rank += 1
            if len(results) >= k_top:
                break

        return results

    def safe_rerank(
        self,
        query: str,
        candidates: Sequence[Any],
        *,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RerankResult]:
        """Graceful reranking that catches failures and returns fallback results."""
        return self.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
            safe=True,
            min_score=min_score,
        )
