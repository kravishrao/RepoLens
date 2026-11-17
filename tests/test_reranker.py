from __future__ import annotations

import pytest

from repolens.ingestion.code_unit import CodeUnit
from repolens.retrieval.bm25_retriever import RetrievalResult
from repolens.retrieval.hybrid_retriever import HybridSearchResult
from repolens.retrieval.reranker import (
    CodeUnitReranker,
    CrossEncoderProvider,
    RerankingError,
    RerankResult,
    SymbolAwareRerankerProvider,
    _make_rerank_text,
)
from repolens.retrieval.semantic_retriever import SemanticSearchResult


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


class MockPredictProvider:
    """Mock provider with predictable scores."""

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores = scores or {}

    def predict(self, pairs):
        results = []
        for _query, text in pairs:
            # Score based on unit name in text
            score = 0.0
            for name, val in self.scores.items():
                if f"Name: {name}" in text:
                    score = val
                    break
            results.append(score)
        return results


class FailingRerankProvider:
    """Provider that raises an exception during predict."""

    def predict(self, pairs):
        raise RuntimeError("GPU OOM / service failure")


# =============================================================================
# CodeUnitReranker Tests
# =============================================================================


def test_reranker_reorders_candidates():
    unit1 = make_code_unit(unit_id="1", name="calculate_total")
    unit2 = make_code_unit(unit_id="2", name="authenticate_user")
    unit3 = make_code_unit(unit_id="3", name="send_email")

    # Initial hybrid ordering: unit1 (rank 1), unit2 (rank 2), unit3 (rank 3)
    hybrid_candidates = [
        HybridSearchResult(code_unit=unit1, score=0.8, rank=1),
        HybridSearchResult(code_unit=unit2, score=0.7, rank=2),
        HybridSearchResult(code_unit=unit3, score=0.6, rank=3),
    ]

    # Mock provider gives highest score to authenticate_user
    mock_provider = MockPredictProvider(
        {"authenticate_user": 0.99, "send_email": 0.85, "calculate_total": 0.10}
    )

    reranker = CodeUnitReranker(provider=mock_provider)
    results = reranker.rerank("login credentials", hybrid_candidates, top_k=3)

    assert len(results) == 3
    # unit2 should now be rank 1
    assert results[0].code_unit_id == "2"
    assert results[0].rank == 1
    assert results[0].score == pytest.approx(0.99)
    assert results[0].initial_rank == 2
    assert results[0].initial_score == pytest.approx(0.7)

    # unit3 should now be rank 2
    assert results[1].code_unit_id == "3"
    assert results[1].rank == 2

    # unit1 should now be rank 3
    assert results[2].code_unit_id == "1"
    assert results[2].rank == 3


def test_reranker_accepts_various_candidate_types():
    unit1 = make_code_unit(unit_id="1", name="fn1")
    unit2 = make_code_unit(unit_id="2", name="fn2")
    unit3 = make_code_unit(unit_id="3", name="fn3")
    unit4 = make_code_unit(unit_id="4", name="fn4")

    candidates = [
        unit1,  # Raw CodeUnit
        HybridSearchResult(code_unit=unit2, score=0.5, rank=1),  # HybridSearchResult
        RetrievalResult(code_unit=unit3, score=12.0, rank=1),  # RetrievalResult
        SemanticSearchResult(
            code_unit=unit4, score=0.8, rank=1
        ),  # SemanticSearchResult
    ]

    reranker = CodeUnitReranker(
        provider=MockPredictProvider({"fn3": 1.0, "fn1": 0.8, "fn4": 0.6, "fn2": 0.4})
    )
    results = reranker.rerank("test query", candidates)

    assert len(results) == 4
    assert [r.code_unit_id for r in results] == ["3", "1", "4", "2"]


def test_reranker_deduplicates_candidates():
    unit1 = make_code_unit(unit_id="1", name="fn1")

    candidates = [
        unit1,
        HybridSearchResult(code_unit=unit1, score=0.5, rank=1),
    ]

    reranker = CodeUnitReranker(provider=MockPredictProvider({"fn1": 0.9}))
    results = reranker.rerank("test query", candidates)

    assert len(results) == 1
    assert results[0].code_unit_id == "1"


def test_reranker_configurable_limits():
    units = [make_code_unit(unit_id=str(i), name=f"fn{i}") for i in range(10)]

    mock_scores = {f"fn{i}": float(i) for i in range(10)}
    reranker = CodeUnitReranker(
        provider=MockPredictProvider(mock_scores),
        max_candidates=5,
        top_k=2,
    )

    # With max_candidates=5, only fn0..fn4 are evaluated (highest among them is fn4 with score 4.0)
    results = reranker.rerank("test", units)
    assert len(results) == 2
    assert results[0].code_unit_id == "4"
    assert results[1].code_unit_id == "3"


def test_reranker_min_score_filter():
    units = [
        make_code_unit(unit_id="1", name="auth"),
        make_code_unit(unit_id="2", name="other"),
    ]
    reranker = CodeUnitReranker(
        provider=MockPredictProvider({"auth": 0.95, "other": 0.2})
    )
    results = reranker.rerank("auth", units, min_score=0.5)

    assert len(results) == 1
    assert results[0].code_unit_id == "1"


def test_reranker_preserves_source_metadata():
    unit = make_code_unit(
        unit_id="unit-42",
        name="validate_token",
        file_path="src/security/jwt.py",
        start_line=50,
        end_line=75,
        kind="method",
        source_code="def validate_token(token): return True",
        docstring="Validate an incoming JWT token.",
        metadata={"auth_scheme": "bearer"},
    )
    hybrid_res = HybridSearchResult(
        code_unit=unit,
        score=0.75,
        rank=3,
        bm25_score=5.0,
        semantic_score=0.8,
    )

    reranker = CodeUnitReranker(provider=MockPredictProvider({"validate_token": 0.98}))
    results = reranker.rerank("jwt token", [hybrid_res])

    assert len(results) == 1
    res = results[0]
    assert res.code_unit_id == "unit-42"
    assert res.name == "validate_token"
    assert res.symbol == "validate_token"
    assert res.qualified_name == "module.validate_token"
    assert res.file_path == "src/security/jwt.py"
    assert res.start_line == 50
    assert res.end_line == 75
    assert res.kind == "method"
    assert res.source_code == "def validate_token(token): return True"
    assert res.docstring == "Validate an incoming JWT token."
    assert res.metadata == {"auth_scheme": "bearer"}
    assert res.code_unit is unit
    assert res.score == pytest.approx(0.98)
    assert res.initial_score == pytest.approx(0.75)
    assert res.initial_rank == 3
    assert isinstance(res, RerankResult)


def test_reranker_handles_empty_inputs():
    reranker = CodeUnitReranker()

    assert reranker.rerank("", [make_code_unit(unit_id="1", name="fn")]) == []
    assert reranker.rerank("   ", [make_code_unit(unit_id="1", name="fn")]) == []
    assert reranker.rerank("query", []) == []


def test_reranker_handles_failures_gracefully():
    failing = FailingRerankProvider()
    unit1 = make_code_unit(unit_id="1", name="first")
    unit2 = make_code_unit(unit_id="2", name="second")
    candidates = [
        HybridSearchResult(code_unit=unit1, score=0.9, rank=1),
        HybridSearchResult(code_unit=unit2, score=0.8, rank=2),
    ]

    # In safe mode, falls back to original ordering with source="fallback"
    reranker_safe = CodeUnitReranker(provider=failing, fallback_on_failure=True)
    fallback_res = reranker_safe.rerank("test query", candidates, safe=True)

    assert len(fallback_res) == 2
    assert fallback_res[0].code_unit_id == "1"
    assert fallback_res[0].source == "fallback"
    assert fallback_res[1].code_unit_id == "2"

    # In strict mode (safe=False and fallback_on_failure=False), raises RerankingError
    reranker_strict = CodeUnitReranker(provider=failing, fallback_on_failure=False)
    with pytest.raises(RerankingError, match="Reranking failed"):
        reranker_strict.rerank("test query", candidates, safe=False)

    # safe_rerank method also never raises
    safe_out = reranker_strict.safe_rerank("test query", candidates)
    assert len(safe_out) == 2


def test_reranker_validations():
    with pytest.raises(ValueError, match="max_candidates"):
        CodeUnitReranker(max_candidates=0)

    with pytest.raises(ValueError, match="top_k"):
        CodeUnitReranker(top_k=0)

    reranker = CodeUnitReranker()
    with pytest.raises(TypeError, match="query"):
        reranker.rerank(None, [])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank("test", [make_code_unit(unit_id="1", name="fn")], top_k=0)


# =============================================================================
# SymbolAwareRerankerProvider Tests
# =============================================================================


def test_symbol_aware_reranker_provider():
    provider = SymbolAwareRerankerProvider()
    assert provider.predict([]) == []

    unit_auth = make_code_unit(
        unit_id="1",
        name="authenticate_user",
        docstring="Authenticate user via credentials",
    )
    unit_math = make_code_unit(
        unit_id="2",
        name="calculate_sum",
        docstring="Compute sum of numbers",
    )

    pairs = [
        ("authenticate credentials", _make_rerank_text(unit_auth)),
        ("authenticate credentials", _make_rerank_text(unit_math)),
    ]

    scores = provider.predict(pairs)
    assert len(scores) == 2
    assert scores[0] > scores[1]


# =============================================================================
# CrossEncoderProvider Tests
# =============================================================================


def test_cross_encoder_provider_config():
    provider = CrossEncoderProvider(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=16,
        max_length=256,
        device="cpu",
    )
    assert provider.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert provider.batch_size == 16
    assert provider.max_length == 256
    assert provider.device == "cpu"
    assert provider._model is None


def test_cross_encoder_provider_validations():
    with pytest.raises(ValueError, match="model_name"):
        CrossEncoderProvider(model_name="   ")

    with pytest.raises(ValueError, match="batch_size"):
        CrossEncoderProvider(batch_size=0)

    with pytest.raises(ValueError, match="max_length"):
        CrossEncoderProvider(max_length=0)

    provider = CrossEncoderProvider()
    assert provider.predict([]) == []

    with pytest.raises(TypeError):
        provider.predict(["invalid"])  # type: ignore[list-item]

    with pytest.raises(TypeError):
        provider.predict([(123, "text")])  # type: ignore[list-item]
