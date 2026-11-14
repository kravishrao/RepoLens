from __future__ import annotations

import pytest

from repolens.ingestion.code_unit import CodeUnit
from repolens.retrieval.semantic_retriever import (
    EmbeddingSemanticRetriever,
    SentenceTransformerProvider,
)


class FakeEmbeddingProvider:
    """Deterministic embedding provider for unit tests."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        vectors = []

        for text in texts:
            lowered = text.lower()

            if "query about authentication" in lowered or "authentication" in lowered:
                vectors.append([1.0, 0.0])

            else:
                vectors.append([0.0, 1.0])

        return vectors


def make_code_unit(
    *,
    unit_id: str,
    name: str,
    source_code: str = "",
    docstring: str = "",
) -> CodeUnit:
    return CodeUnit(
        id=unit_id,
        kind="function",
        name=name,
        qualified_name=f"module.{name}",
        file_path="module.py",
        start_line=1,
        end_line=10,
        source_code=source_code,
        docstring=docstring,
    )


def test_semantic_retriever_indexes_code_units():
    units = [
        make_code_unit(unit_id="1", name="login"),
        make_code_unit(unit_id="2", name="logout"),
    ]

    retriever = EmbeddingSemanticRetriever(
        units,
        FakeEmbeddingProvider(),
    )

    assert retriever.document_count == 2
    assert retriever.is_indexed is False


def test_semantic_retriever_ranks_by_cosine_similarity():
    units = [
        make_code_unit(
            unit_id="1",
            name="authenticate_user",
            docstring="Authentication handler",
        ),
        make_code_unit(
            unit_id="2",
            name="close_session",
            docstring="Session cleanup handler",
        ),
    ]

    retriever = EmbeddingSemanticRetriever(
        units,
        FakeEmbeddingProvider(),
    )

    results = retriever.query("query about authentication")

    assert len(results) == 2
    assert results[0].code_unit.id == "1"
    assert results[0].score == pytest.approx(1.0)
    assert results[0].rank == 1
    assert results[1].code_unit.id == "2"
    assert results[1].score == pytest.approx(0.0)


def test_semantic_retriever_caches_document_embeddings():
    provider = FakeEmbeddingProvider()

    retriever = EmbeddingSemanticRetriever(
        [
            make_code_unit(unit_id="1", name="authenticate_user"),
        ],
        provider,
    )

    retriever.query("query about authentication")
    retriever.query("query about authentication")

    assert retriever.is_indexed is True
    assert len(provider.calls) == 3
    assert len(provider.calls[0]) == 1
    assert len(provider.calls[1]) == 1
    assert len(provider.calls[2]) == 1


def test_semantic_retriever_returns_top_k_results():
    units = [
        make_code_unit(unit_id=str(index), name=f"function_{index}")
        for index in range(5)
    ]

    retriever = EmbeddingSemanticRetriever(
        units,
        FakeEmbeddingProvider(),
    )

    results = retriever.query(
        "query about authentication",
        top_k=2,
    )

    assert len(results) == 2
    assert [result.rank for result in results] == [1, 2]


def test_semantic_retriever_returns_empty_for_blank_query():
    retriever = EmbeddingSemanticRetriever(
        [make_code_unit(unit_id="1", name="login")],
        FakeEmbeddingProvider(),
    )

    assert retriever.query("   ") == []
    assert retriever.is_indexed is False


def test_semantic_retriever_handles_empty_repository():
    provider = FakeEmbeddingProvider()
    retriever = EmbeddingSemanticRetriever([], provider)

    assert retriever.query("authentication") == []
    assert retriever.document_count == 0
    assert provider.calls == []


def test_semantic_retriever_deduplicates_code_units():
    unit = make_code_unit(unit_id="1", name="login")

    retriever = EmbeddingSemanticRetriever(
        [unit, unit],
        FakeEmbeddingProvider(),
    )

    assert retriever.document_count == 1


def test_semantic_retriever_rejects_invalid_code_units():
    with pytest.raises(TypeError, match="CodeUnit"):
        EmbeddingSemanticRetriever(
            ["not a code unit"],
            FakeEmbeddingProvider(),
        )


def test_semantic_retriever_rejects_invalid_top_k():
    retriever = EmbeddingSemanticRetriever(
        [make_code_unit(unit_id="1", name="login")],
        FakeEmbeddingProvider(),
    )

    with pytest.raises(ValueError, match="top_k"):
        retriever.query("authentication", top_k=0)


def test_semantic_retriever_rejects_invalid_embedding_dimensions():
    class InvalidProvider:
        def encode(self, texts):
            if len(texts) == 1 and "query" in texts[0].lower():
                return [[1.0, 0.0]]

            return [[1.0], [0.0, 1.0]][: len(texts)]

    retriever = EmbeddingSemanticRetriever(
        [make_code_unit(unit_id="1", name="login")],
        InvalidProvider(),
    )

    with pytest.raises(ValueError, match="dimensions"):
        retriever.query("query")


def test_sentence_transformer_provider_loads_model_lazily():
    provider = SentenceTransformerProvider()

    assert provider._model is None
    assert provider.model_name == "sentence-transformers/all-MiniLM-L6-v2"
