from __future__ import annotations

import pytest

from repolens.ingestion.code_unit import CodeUnit
from repolens.retrieval.bm25_retriever import BM25Retriever, RetrievalResult
from repolens.retrieval.hybrid_retriever import (
    HybridRetriever,
    HybridSearchResult,
    fuse_retrieval_results,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from repolens.retrieval.semantic_retriever import (
    EmbeddingSemanticRetriever,
    SemanticSearchResult,
)


def make_code_unit(
    *,
    unit_id: str,
    name: str,
    source_code: str = "",
    docstring: str = "",
    file_path: str = "module.py",
    start_line: int = 1,
    end_line: int = 10,
    kind: str = "function",
    metadata: dict[str, object] | None = None,
) -> CodeUnit:
    return CodeUnit(
        id=unit_id,
        kind=kind,
        name=name,
        qualified_name=f"module.{name}",
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        source_code=source_code,
        docstring=docstring,
        metadata=metadata or {},
    )


class DummyEmbeddingProvider:
    """Deterministic embedding provider for tests."""

    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "auth" in lowered or "login" in lowered:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


# =============================================================================
# Reciprocal Rank Fusion (RRF) Tests
# =============================================================================


def test_rrf_combines_and_deduplicates():
    unit1 = make_code_unit(unit_id="1", name="auth_service")
    unit2 = make_code_unit(unit_id="2", name="user_model")
    unit3 = make_code_unit(unit_id="3", name="session_mgr")

    # BM25: unit1 ranked 1, unit2 ranked 2
    bm25_res = [
        RetrievalResult(code_unit=unit1, score=12.5, rank=1),
        RetrievalResult(code_unit=unit2, score=8.0, rank=2),
    ]
    # Semantic: unit1 ranked 2, unit3 ranked 1
    semantic_res = [
        SemanticSearchResult(code_unit=unit3, score=0.92, rank=1),
        SemanticSearchResult(code_unit=unit1, score=0.85, rank=2),
    ]

    fused = reciprocal_rank_fusion(
        bm25_res,
        semantic_res,
        rrf_k=60,
        top_k=5,
    )

    # All 3 unique units should be present without duplication
    assert len(fused) == 3
    ids = [r.code_unit_id for r in fused]
    assert len(set(ids)) == 3

    # unit1 appears in both: score = 1/(60+1) + 1/(60+2) ≈ 0.01639 + 0.01613 = 0.03252
    # unit3 appears only in semantic rank 1: score = 1/(60+1) ≈ 0.01639
    # unit2 appears only in BM25 rank 2: score = 1/(60+2) ≈ 0.01613
    assert fused[0].code_unit_id == "1"
    assert isinstance(fused[0], HybridSearchResult)
    assert fused[0].rank == 1
    assert fused[0].source == "hybrid"
    assert fused[0].bm25_score == 12.5
    assert fused[0].bm25_rank == 1
    assert fused[0].semantic_score == 0.85
    assert fused[0].semantic_rank == 2

    assert fused[1].code_unit_id == "3"
    assert fused[1].rank == 2
    assert fused[1].source == "semantic"
    assert fused[1].bm25_score is None

    assert fused[2].code_unit_id == "2"
    assert fused[2].rank == 3
    assert fused[2].source == "bm25"
    assert fused[2].semantic_score is None


def test_rrf_configurable_weights():
    unit1 = make_code_unit(unit_id="1", name="fn1")
    unit2 = make_code_unit(unit_id="2", name="fn2")

    bm25_res = [RetrievalResult(code_unit=unit1, score=10.0, rank=1)]
    semantic_res = [SemanticSearchResult(code_unit=unit2, score=0.9, rank=1)]

    # Give BM25 weight 2.0 and semantic 0.5
    fused = reciprocal_rank_fusion(
        bm25_res,
        semantic_res,
        rrf_k=60,
        bm25_weight=2.0,
        semantic_weight=0.5,
    )

    assert fused[0].code_unit_id == "1"
    assert fused[1].code_unit_id == "2"

    # Now flip weights
    fused_flipped = reciprocal_rank_fusion(
        bm25_res,
        semantic_res,
        rrf_k=60,
        bm25_weight=0.5,
        semantic_weight=2.0,
    )

    assert fused_flipped[0].code_unit_id == "2"
    assert fused_flipped[1].code_unit_id == "1"


def test_rrf_empty_inputs():
    unit = make_code_unit(unit_id="1", name="fn")
    bm25_res = [RetrievalResult(code_unit=unit, score=5.0, rank=1)]

    assert reciprocal_rank_fusion([], []) == []

    # One empty source
    fused_semantic_empty = reciprocal_rank_fusion(bm25_res, [])
    assert len(fused_semantic_empty) == 1
    assert fused_semantic_empty[0].code_unit_id == "1"
    assert fused_semantic_empty[0].source == "bm25"

    fused_bm25_empty = reciprocal_rank_fusion([], bm25_res)
    assert len(fused_bm25_empty) == 1
    assert fused_bm25_empty[0].source == "semantic"


def test_rrf_validation():
    unit = make_code_unit(unit_id="1", name="fn")
    res = [RetrievalResult(code_unit=unit, score=1.0, rank=1)]

    with pytest.raises(ValueError, match="rrf_k"):
        reciprocal_rank_fusion(res, res, rrf_k=0)

    with pytest.raises(ValueError, match="rrf_k"):
        reciprocal_rank_fusion(res, res, rrf_k=-1)

    with pytest.raises(ValueError, match="Weights"):
        reciprocal_rank_fusion(res, res, bm25_weight=-0.5)

    with pytest.raises(ValueError, match="greater than zero"):
        reciprocal_rank_fusion(res, res, bm25_weight=0.0, semantic_weight=0.0)

    with pytest.raises(ValueError, match="top_k"):
        reciprocal_rank_fusion(res, res, top_k=0)


# =============================================================================
# Weighted Score Fusion Tests
# =============================================================================


def test_weighted_score_fusion_normalizes():
    unit1 = make_code_unit(unit_id="1", name="doc1")
    unit2 = make_code_unit(unit_id="2", name="doc2")

    # BM25 scores: 10.0 and 5.0 (min=5, max=10 -> normalized: 1.0 and 0.0)
    bm25_res = [
        RetrievalResult(code_unit=unit1, score=10.0, rank=1),
        RetrievalResult(code_unit=unit2, score=5.0, rank=2),
    ]
    # Semantic scores: 0.9 and 0.1 (min=0.1, max=0.9 -> normalized: 1.0 and 0.0)
    semantic_res = [
        SemanticSearchResult(code_unit=unit2, score=0.9, rank=1),
        SemanticSearchResult(code_unit=unit1, score=0.1, rank=2),
    ]

    # Equal weights 0.5 each:
    # unit1: 0.5 * 1.0 + 0.5 * 0.0 = 0.5
    # unit2: 0.5 * 0.0 + 0.5 * 1.0 = 0.5
    fused = weighted_score_fusion(
        bm25_res,
        semantic_res,
        bm25_weight=0.5,
        semantic_weight=0.5,
    )
    assert len(fused) == 2
    assert fused[0].score == pytest.approx(0.5)
    assert fused[1].score == pytest.approx(0.5)

    # Shift weight toward BM25 (0.8 vs 0.2):
    # unit1: 0.8 * 1.0 + 0.2 * 0.0 = 0.8
    # unit2: 0.8 * 0.0 + 0.2 * 1.0 = 0.2
    fused_bm25_heavy = weighted_score_fusion(
        bm25_res,
        semantic_res,
        bm25_weight=0.8,
        semantic_weight=0.2,
    )
    assert fused_bm25_heavy[0].code_unit_id == "1"
    assert fused_bm25_heavy[0].score == pytest.approx(0.8)
    assert fused_bm25_heavy[1].code_unit_id == "2"
    assert fused_bm25_heavy[1].score == pytest.approx(0.2)


def test_weighted_score_fusion_disjoint_units():
    unit1 = make_code_unit(unit_id="1", name="u1")
    unit2 = make_code_unit(unit_id="2", name="u2")

    bm25_res = [RetrievalResult(code_unit=unit1, score=10.0, rank=1)]
    semantic_res = [SemanticSearchResult(code_unit=unit2, score=0.8, rank=1)]

    fused = weighted_score_fusion(bm25_res, semantic_res)
    assert len(fused) == 2
    sources = {r.code_unit_id: r.source for r in fused}
    assert sources["1"] == "bm25"
    assert sources["2"] == "semantic"


# =============================================================================
# High-Level fuse_retrieval_results Tests
# =============================================================================


def test_fuse_retrieval_results_strategy_selection():
    unit = make_code_unit(unit_id="1", name="u1")
    res = [RetrievalResult(code_unit=unit, score=5.0, rank=1)]

    rrf_out = fuse_retrieval_results(res, res, strategy="rrf")
    assert len(rrf_out) == 1
    assert rrf_out[0].source == "hybrid"

    weighted_out = fuse_retrieval_results(res, res, strategy="weighted")
    assert len(weighted_out) == 1

    with pytest.raises(ValueError, match="Unsupported fusion strategy"):
        fuse_retrieval_results(res, res, strategy="unknown")  # type: ignore[arg-type]


# =============================================================================
# HybridRetriever Class Tests
# =============================================================================


def test_hybrid_retriever_from_code_units():
    units = [
        make_code_unit(
            unit_id="1",
            name="authenticate_user",
            source_code="def authenticate_user(): pass",
            docstring="Authenticate user credentials",
        ),
        make_code_unit(
            unit_id="2",
            name="format_date",
            source_code="def format_date(): pass",
            docstring="Format date string",
        ),
    ]

    retriever = HybridRetriever.from_code_units(
        units,
        embedding_provider=DummyEmbeddingProvider(),
        strategy="rrf",
    )

    results = retriever.query("authenticate user", top_k=2)
    assert len(results) >= 1
    assert results[0].code_unit_id == "1"
    assert results[0].rank == 1
    assert results[0].source in ("hybrid", "bm25", "semantic")


def test_hybrid_retriever_preserves_metadata():
    unit = make_code_unit(
        unit_id="auth-99",
        name="login_handler",
        file_path="src/auth.py",
        start_line=12,
        end_line=24,
        kind="function",
        source_code="def login_handler(): pass",
        docstring="Handle login requests",
        metadata={"route": "/login"},
    )
    bm25 = BM25Retriever([unit])
    semantic = EmbeddingSemanticRetriever([unit], DummyEmbeddingProvider())

    retriever = HybridRetriever(bm25, semantic)
    results = retriever.query("login handler", top_k=1)

    assert len(results) == 1
    res = results[0]
    assert res.code_unit_id == "auth-99"
    assert res.name == "login_handler"
    assert res.symbol == "login_handler"
    assert res.qualified_name == "module.login_handler"
    assert res.file_path == "src/auth.py"
    assert res.start_line == 12
    assert res.end_line == 24
    assert res.kind == "function"
    assert res.source_code == "def login_handler(): pass"
    assert res.docstring == "Handle login requests"
    assert res.metadata == {"route": "/login"}
    assert res.code_unit is unit


def test_hybrid_retriever_empty_and_whitespace_query():
    unit = make_code_unit(unit_id="1", name="calc")
    retriever = HybridRetriever.from_code_units([unit])

    assert retriever.query("") == []
    assert retriever.query("   ") == []
    assert retriever.query("\t\n") == []


def test_hybrid_retriever_validation():
    with pytest.raises(ValueError, match="At least one retriever"):
        HybridRetriever(None, None)

    unit = make_code_unit(unit_id="1", name="fn")
    bm25 = BM25Retriever([unit])

    with pytest.raises(ValueError, match="Unsupported strategy"):
        HybridRetriever(bm25, strategy="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Weights"):
        HybridRetriever(bm25, bm25_weight=-1.0)

    with pytest.raises(ValueError, match="greater than zero"):
        HybridRetriever(bm25, bm25_weight=0.0, semantic_weight=0.0)

    with pytest.raises(ValueError, match="rrf_k"):
        HybridRetriever(bm25, rrf_k=0)

    retriever = HybridRetriever(bm25)
    with pytest.raises(TypeError, match="query_text"):
        retriever.query(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="top_k"):
        retriever.query("test", top_k=0)


def test_hybrid_retriever_min_score_filter():
    unit1 = make_code_unit(unit_id="1", name="search_items")
    unit2 = make_code_unit(unit_id="2", name="other")

    bm25_res = [
        RetrievalResult(code_unit=unit1, score=10.0, rank=1),
        RetrievalResult(code_unit=unit2, score=2.0, rank=2),
    ]
    semantic_res = [
        SemanticSearchResult(code_unit=unit1, score=0.95, rank=1),
    ]

    fused = reciprocal_rank_fusion(bm25_res, semantic_res)
    assert len(fused) == 2

    # Using weighted fusion with min_score
    fused_weighted = weighted_score_fusion(
        bm25_res,
        semantic_res,
        bm25_weight=0.5,
        semantic_weight=0.5,
    )
    # Filter with min_score threshold
    filtered = [r for r in fused_weighted if r.score >= 0.5]
    assert len(filtered) == 1
    assert filtered[0].code_unit_id == "1"
