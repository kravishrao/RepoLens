from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from repolens.ingestion.code_unit import CodeUnit
from repolens.retrieval.bm25_retriever import BM25Retriever, RetrievalResult
from repolens.retrieval.semantic_retriever import (
    EmbeddingProvider,
    EmbeddingSemanticRetriever,
    SemanticSearchResult,
)

logger = logging.getLogger(__name__)

FusionStrategy = Literal["rrf", "weighted"]


@dataclass(frozen=True)
class HybridSearchResult:
    """A single ranked result combining lexical and semantic retrieval outputs."""

    code_unit: CodeUnit
    score: float
    rank: int
    bm25_score: float | None = None
    bm25_rank: int | None = None
    semantic_score: float | None = None
    semantic_rank: int | None = None
    source: str = "hybrid"

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
        """Alias for symbol name."""
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
        """Docstring of the unit."""
        return self.code_unit.docstring

    @property
    def metadata(self) -> dict[str, Any]:
        """Custom metadata associated with the CodeUnit."""
        return self.code_unit.metadata


def _extract_item_data(
    item: RetrievalResult | SemanticSearchResult | HybridSearchResult | CodeUnit,
    default_rank: int,
) -> tuple[CodeUnit, float, int]:
    """Extract (code_unit, score, rank) from heterogeneous retrieval items."""
    if isinstance(item, (RetrievalResult, SemanticSearchResult, HybridSearchResult)):
        return item.code_unit, float(item.score), int(item.rank)
    if isinstance(item, CodeUnit):
        return item, 0.0, default_rank
    if hasattr(item, "code_unit") and hasattr(item, "score"):
        code_unit = item.code_unit
        score = float(item.score)
        rank = int(getattr(item, "rank", default_rank))
        return code_unit, score, rank
    raise TypeError(
        f"Unsupported retrieval result type: {type(item).__name__}. "
        "Expected RetrievalResult, SemanticSearchResult, or CodeUnit."
    )


def reciprocal_rank_fusion(
    bm25_results: Sequence[Any],
    semantic_results: Sequence[Any],
    *,
    rrf_k: int = 60,
    bm25_weight: float = 1.0,
    semantic_weight: float = 1.0,
    top_k: int = 10,
) -> list[HybridSearchResult]:
    """Combine lexical and semantic rankings using Reciprocal Rank Fusion (RRF).

    Formula for each document d:
        RRF(d) = bm25_weight * (1 / (rrf_k + rank_bm25(d)))
               + semantic_weight * (1 / (rrf_k + rank_semantic(d)))
    """
    if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k < 1:
        raise ValueError("rrf_k must be a positive integer.")
    if bm25_weight < 0.0 or semantic_weight < 0.0:
        raise ValueError("Weights must be non-negative.")
    if bm25_weight == 0.0 and semantic_weight == 0.0:
        raise ValueError("At least one weight must be greater than zero.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    scores: dict[str, float] = {}
    units: dict[str, CodeUnit] = {}
    bm25_info: dict[str, tuple[float, int]] = {}
    semantic_info: dict[str, tuple[float, int]] = {}

    for default_idx, item in enumerate(bm25_results, start=1):
        unit, score, rank = _extract_item_data(item, default_idx)
        if unit.id not in units:
            units[unit.id] = unit
            scores[unit.id] = 0.0
        bm25_info[unit.id] = (score, rank)
        scores[unit.id] += bm25_weight * (1.0 / (rrf_k + rank))

    for default_idx, item in enumerate(semantic_results, start=1):
        unit, score, rank = _extract_item_data(item, default_idx)
        if unit.id not in units:
            units[unit.id] = unit
            scores[unit.id] = 0.0
        semantic_info[unit.id] = (score, rank)
        scores[unit.id] += semantic_weight * (1.0 / (rrf_k + rank))

    sorted_ids = sorted(scores.keys(), key=lambda uid: scores[uid], reverse=True)

    results: list[HybridSearchResult] = []
    for rank, uid in enumerate(sorted_ids[:top_k], start=1):
        has_bm25 = uid in bm25_info
        has_semantic = uid in semantic_info

        if has_bm25 and has_semantic:
            source = "hybrid"
        elif has_bm25:
            source = "bm25"
        else:
            source = "semantic"

        b_score, b_rank = bm25_info.get(uid, (None, None))
        s_score, s_rank = semantic_info.get(uid, (None, None))

        results.append(
            HybridSearchResult(
                code_unit=units[uid],
                score=scores[uid],
                rank=rank,
                bm25_score=b_score,
                bm25_rank=b_rank,
                semantic_score=s_score,
                semantic_rank=s_rank,
                source=source,
            )
        )

    return results


def _normalize_scores(
    items: Sequence[tuple[CodeUnit, float, int]],
) -> dict[str, float]:
    """Min-max normalize raw scores to the range [0.0, 1.0]."""
    if not items:
        return {}

    raw_scores = [score for _, score, _ in items]
    min_score = min(raw_scores)
    max_score = max(raw_scores)

    diff = max_score - min_score
    normalized: dict[str, float] = {}

    for unit, score, _ in items:
        if diff > 1e-12:
            normalized[unit.id] = (score - min_score) / diff
        else:
            normalized[unit.id] = 1.0 if score > 0.0 else 0.0

    return normalized


def weighted_score_fusion(
    bm25_results: Sequence[Any],
    semantic_results: Sequence[Any],
    *,
    bm25_weight: float = 0.5,
    semantic_weight: float = 0.5,
    top_k: int = 10,
    normalize: bool = True,
) -> list[HybridSearchResult]:
    """Combine lexical and semantic outputs using weighted score normalization.

    Scores are min-max normalized to [0, 1] per source, then linearly combined:
        Fused(d) = bm25_weight * norm_bm25(d) + semantic_weight * norm_semantic(d)
    """
    if bm25_weight < 0.0 or semantic_weight < 0.0:
        raise ValueError("Weights must be non-negative.")
    if bm25_weight == 0.0 and semantic_weight == 0.0:
        raise ValueError("At least one weight must be greater than zero.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    bm25_extracted = [
        _extract_item_data(item, idx) for idx, item in enumerate(bm25_results, start=1)
    ]
    semantic_extracted = [
        _extract_item_data(item, idx)
        for idx, item in enumerate(semantic_results, start=1)
    ]

    units: dict[str, CodeUnit] = {}
    bm25_info: dict[str, tuple[float, int]] = {}
    semantic_info: dict[str, tuple[float, int]] = {}

    for unit, score, rank in bm25_extracted:
        units[unit.id] = unit
        bm25_info[unit.id] = (score, rank)

    for unit, score, rank in semantic_extracted:
        units[unit.id] = unit
        semantic_info[unit.id] = (score, rank)

    if normalize:
        norm_bm25 = _normalize_scores(bm25_extracted)
        norm_semantic = _normalize_scores(semantic_extracted)
    else:
        norm_bm25 = {unit.id: score for unit, score, _ in bm25_extracted}
        norm_semantic = {unit.id: score for unit, score, _ in semantic_extracted}

    fused_scores: dict[str, float] = {}
    for uid in units:
        b_val = norm_bm25.get(uid, 0.0)
        s_val = norm_semantic.get(uid, 0.0)
        fused_scores[uid] = (bm25_weight * b_val) + (semantic_weight * s_val)

    sorted_ids = sorted(
        fused_scores.keys(), key=lambda uid: fused_scores[uid], reverse=True
    )

    results: list[HybridSearchResult] = []
    for rank, uid in enumerate(sorted_ids[:top_k], start=1):
        has_bm25 = uid in bm25_info
        has_semantic = uid in semantic_info

        if has_bm25 and has_semantic:
            source = "hybrid"
        elif has_bm25:
            source = "bm25"
        else:
            source = "semantic"

        b_score, b_rank = bm25_info.get(uid, (None, None))
        s_score, s_rank = semantic_info.get(uid, (None, None))

        results.append(
            HybridSearchResult(
                code_unit=units[uid],
                score=fused_scores[uid],
                rank=rank,
                bm25_score=b_score,
                bm25_rank=b_rank,
                semantic_score=s_score,
                semantic_rank=s_rank,
                source=source,
            )
        )

    return results


def fuse_retrieval_results(
    bm25_results: Sequence[Any],
    semantic_results: Sequence[Any],
    *,
    strategy: FusionStrategy = "rrf",
    bm25_weight: float = 0.5,
    semantic_weight: float = 0.5,
    rrf_k: int = 60,
    top_k: int = 10,
    normalize: bool = True,
) -> list[HybridSearchResult]:
    """Fuse BM25 and semantic retrieval outputs using the specified strategy."""
    if strategy == "rrf":
        return reciprocal_rank_fusion(
            bm25_results=bm25_results,
            semantic_results=semantic_results,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            top_k=top_k,
        )
    if strategy == "weighted":
        return weighted_score_fusion(
            bm25_results=bm25_results,
            semantic_results=semantic_results,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            top_k=top_k,
            normalize=normalize,
        )
    raise ValueError(
        f"Unsupported fusion strategy '{strategy}'. Supported: 'rrf', 'weighted'."
    )


class HybridRetriever:
    """Unified hybrid retrieval pipeline combining BM25 and semantic embeddings."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever | None = None,
        semantic_retriever: EmbeddingSemanticRetriever | None = None,
        *,
        strategy: FusionStrategy = "rrf",
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        if bm25_retriever is None and semantic_retriever is None:
            raise ValueError(
                "At least one retriever (BM25 or semantic) must be provided."
            )

        if strategy not in ("rrf", "weighted"):
            raise ValueError(
                f"Unsupported strategy '{strategy}'. Expected 'rrf' or 'weighted'."
            )

        if bm25_weight < 0.0 or semantic_weight < 0.0:
            raise ValueError("Weights must be non-negative.")

        if bm25_weight == 0.0 and semantic_weight == 0.0:
            raise ValueError("At least one weight must be greater than zero.")

        if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k < 1:
            raise ValueError("rrf_k must be a positive integer.")

        self.bm25_retriever = bm25_retriever
        self.semantic_retriever = semantic_retriever
        self.strategy: FusionStrategy = strategy
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.rrf_k = rrf_k

    @classmethod
    def from_code_units(
        cls,
        code_units: Iterable[CodeUnit],
        embedding_provider: EmbeddingProvider | None = None,
        *,
        strategy: FusionStrategy = "rrf",
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> HybridRetriever:
        """Create a HybridRetriever from a single collection of CodeUnits."""
        units = list(code_units)
        bm25 = BM25Retriever(units)

        semantic = None
        if embedding_provider is not None:
            semantic = EmbeddingSemanticRetriever(units, embedding_provider)

        return cls(
            bm25_retriever=bm25,
            semantic_retriever=semantic,
            strategy=strategy,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            rrf_k=rrf_k,
        )

    def query(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        candidate_k: int | None = None,
        min_score: float | None = None,
    ) -> list[HybridSearchResult]:
        """Execute hybrid search over the repository CodeUnits."""
        if not isinstance(query_text, str):
            raise TypeError("query_text must be a string.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        if not query_text.strip():
            return []

        k_candidates = candidate_k or max(top_k * 3, 20)

        bm25_results: list[RetrievalResult] = []
        if self.bm25_retriever is not None:
            try:
                bm25_results = self.bm25_retriever.query(
                    query_text,
                    top_k=k_candidates,
                )
            except Exception as exc:
                logger.warning("BM25 retrieval failed: %s", exc)

        semantic_results: list[SemanticSearchResult] = []
        if self.semantic_retriever is not None:
            try:
                semantic_results = self.semantic_retriever.safe_query(
                    query_text,
                    top_k=k_candidates,
                )
            except Exception as exc:
                logger.warning("Semantic retrieval failed: %s", exc)

        if not bm25_results and not semantic_results:
            return []

        fused = fuse_retrieval_results(
            bm25_results=bm25_results,
            semantic_results=semantic_results,
            strategy=self.strategy,
            bm25_weight=self.bm25_weight,
            semantic_weight=self.semantic_weight,
            rrf_k=self.rrf_k,
            top_k=top_k,
        )

        if min_score is not None:
            fused = [r for r in fused if r.score >= min_score]
            # Re-rank after filtering
            fused = [
                HybridSearchResult(
                    code_unit=r.code_unit,
                    score=r.score,
                    rank=new_rank,
                    bm25_score=r.bm25_score,
                    bm25_rank=r.bm25_rank,
                    semantic_score=r.semantic_score,
                    semantic_rank=r.semantic_rank,
                    source=r.source,
                )
                for new_rank, r in enumerate(fused, start=1)
            ]

        return fused
