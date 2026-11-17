from __future__ import annotations

import pytest

from repolens.ingestion.code_unit import CodeUnit
from repolens.retrieval.semantic_retriever import (
    EmbeddingError,
    EmbeddingSemanticRetriever,
    FaissVectorStore,
    InMemoryVectorStore,
    SemanticSearchResult,
    SentenceTransformerProvider,
    _make_document,
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


class FailingProvider:
    """Provider that raises an exception during encode."""

    def encode(self, texts):
        raise RuntimeError("Embedding model failed")


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


# =============================================================================
# EmbeddingSemanticRetriever Tests
# =============================================================================


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
    assert results[0].score == pytest.approx(1.0, abs=1e-4)
    assert results[0].rank == 1
    assert results[1].code_unit.id == "2"
    assert results[1].score == pytest.approx(0.0, abs=1e-4)


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
    assert retriever.query("") == []
    assert retriever.query("\t\n") == []
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

    with pytest.raises(ValueError, match="top_k"):
        retriever.query("authentication", top_k=-1)

    with pytest.raises(ValueError, match="top_k"):
        retriever.query("authentication", top_k=True)


def test_semantic_retriever_rejects_invalid_query_type():
    retriever = EmbeddingSemanticRetriever(
        [make_code_unit(unit_id="1", name="login")],
        FakeEmbeddingProvider(),
    )

    with pytest.raises(TypeError, match="query_text"):
        retriever.query(123)  # type: ignore[arg-type]


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

    with pytest.raises(ValueError, match="dimension"):
        retriever.query("query")


def test_semantic_retriever_preserves_source_metadata():
    unit = make_code_unit(
        unit_id="auth-001",
        name="verify_credentials",
        file_path="src/auth/service.py",
        start_line=25,
        end_line=45,
        kind="function",
        source_code="def verify_credentials(u, p): return True",
        docstring="Verify user credentials against storage.",
        metadata={"complexity": 3},
    )

    retriever = EmbeddingSemanticRetriever(
        [unit],
        FakeEmbeddingProvider(),
    )

    results = retriever.query("query about authentication")
    assert len(results) == 1

    res = results[0]
    assert res.code_unit_id == "auth-001"
    assert res.name == "verify_credentials"
    assert res.symbol == "verify_credentials"
    assert res.qualified_name == "module.verify_credentials"
    assert res.kind == "function"
    assert res.file_path == "src/auth/service.py"
    assert res.start_line == 25
    assert res.end_line == 45
    assert res.source_code == "def verify_credentials(u, p): return True"
    assert res.docstring == "Verify user credentials against storage."
    assert res.metadata == {"complexity": 3}
    assert res.code_unit is unit


def test_semantic_retriever_encode_code_units():
    units = [
        make_code_unit(
            unit_id="1",
            name="authenticate_user",
            docstring="Authentication handler",
        ),
        make_code_unit(unit_id="2", name="other"),
    ]
    retriever = EmbeddingSemanticRetriever(units, FakeEmbeddingProvider())

    embeddings = retriever.encode_code_units(units)
    assert len(embeddings) == 2
    assert embeddings[0] == [1.0, 0.0]
    assert embeddings[1] == [0.0, 1.0]

    assert retriever.encode_code_units([]) == []

    with pytest.raises(TypeError, match="CodeUnit"):
        retriever.encode_code_units(["invalid"])  # type: ignore[list-item]


def test_semantic_retriever_encode_query():
    retriever = EmbeddingSemanticRetriever(
        [make_code_unit(unit_id="1", name="auth")],
        FakeEmbeddingProvider(),
    )

    vector = retriever.encode_query("query about authentication")
    assert vector == [1.0, 0.0]

    with pytest.raises(ValueError, match="cannot be empty"):
        retriever.encode_query("   ")

    with pytest.raises(TypeError, match="query_text"):
        retriever.encode_query(None)  # type: ignore[arg-type]


def test_semantic_retriever_min_score_filtering():
    units = [
        make_code_unit(unit_id="1", name="auth", docstring="authentication"),
        make_code_unit(unit_id="2", name="logout", docstring="cleanup"),
    ]
    retriever = EmbeddingSemanticRetriever(units, FakeEmbeddingProvider())

    results = retriever.query("query about authentication", min_score=0.5)
    assert len(results) == 1
    assert results[0].code_unit.id == "1"


def test_semantic_retriever_handles_embedding_failure_gracefully():
    failing = FailingProvider()
    retriever = EmbeddingSemanticRetriever(
        [make_code_unit(unit_id="1", name="test")],
        failing,
    )

    # In safe mode or safe_query, returns empty list without raising
    assert retriever.safe_query("test query") == []
    assert retriever.query("test query", safe=True) == []

    # In default strict mode, raises EmbeddingError
    with pytest.raises(EmbeddingError, match="Embedding provider failed"):
        retriever.query("test query", safe=False)


def test_semantic_retriever_with_in_memory_store():
    store = InMemoryVectorStore()
    units = [
        make_code_unit(unit_id="1", name="auth", docstring="authentication"),
        make_code_unit(unit_id="2", name="other", docstring="session"),
    ]
    retriever = EmbeddingSemanticRetriever(
        units,
        FakeEmbeddingProvider(),
        vector_store=store,
    )

    results = retriever.query("query about authentication")
    assert len(results) == 2
    assert results[0].code_unit.id == "1"
    assert results[0].score == pytest.approx(1.0)


# =============================================================================
# FAISS Vector Store Tests
# =============================================================================


def test_faiss_vector_store_add_and_search():
    store = FaissVectorStore(dimension=2)
    units = [
        make_code_unit(unit_id="1", name="auth", file_path="auth.py"),
        make_code_unit(unit_id="2", name="data", file_path="data.py"),
    ]
    vectors = [[1.0, 0.0], [0.0, 1.0]]

    store.add(vectors, units)
    assert store.count == 2

    # Query with [1.0, 0.0] should match unit 1 first
    results = store.search([1.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0][0].id == "1"
    assert results[0][1] == pytest.approx(1.0, abs=1e-4)
    assert results[1][0].id == "2"
    assert results[1][1] == pytest.approx(0.0, abs=1e-4)


def test_faiss_vector_store_dynamic_dimension():
    store = FaissVectorStore()
    assert store.count == 0
    assert store.dimension is None

    units = [make_code_unit(unit_id="1", name="fn")]
    store.add([[0.5, 0.5, 0.5]], units)

    assert store.dimension == 3
    assert store.count == 1

    results = store.search([0.5, 0.5, 0.5], top_k=1)
    assert len(results) == 1
    assert results[0][0].id == "1"


def test_faiss_vector_store_preserves_metadata():
    store = FaissVectorStore(dimension=2)
    unit = make_code_unit(
        unit_id="10",
        name="process_order",
        file_path="orders.py",
        start_line=15,
        end_line=30,
        kind="function",
        source_code="def process_order(): pass",
        docstring="Process an order.",
    )
    store.add([[1.0, 0.0]], [unit])

    meta = store.get_metadata(0)
    assert meta["id"] == "10"
    assert meta["name"] == "process_order"
    assert meta["file_path"] == "orders.py"
    assert meta["start_line"] == 15
    assert meta["end_line"] == 30
    assert meta["kind"] == "function"
    assert meta["docstring"] == "Process an order."


def test_faiss_vector_store_empty_operations():
    store = FaissVectorStore(dimension=2)
    assert store.search([1.0, 0.0]) == []
    store.add([], [])
    assert store.count == 0


def test_faiss_vector_store_clear():
    store = FaissVectorStore(dimension=2)
    unit = make_code_unit(unit_id="1", name="fn")
    store.add([[1.0, 0.0]], [unit])
    assert store.count == 1

    store.clear()
    assert store.count == 0
    assert store.search([1.0, 0.0]) == []


def test_faiss_vector_store_validation():
    store = FaissVectorStore(dimension=2)
    unit = make_code_unit(unit_id="1", name="fn")

    with pytest.raises(ValueError, match="match"):
        store.add([[1.0, 0.0]], [unit, unit])

    with pytest.raises(ValueError, match="match"):
        store.add([[1.0, 0.0, 0.0]], [unit])

    with pytest.raises(ValueError, match="dimension"):
        store.search([1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="top_k"):
        store.search([1.0, 0.0], top_k=0)

    with pytest.raises(IndexError):
        store.get_metadata(0)


# =============================================================================
# InMemoryVectorStore Tests
# =============================================================================


def test_in_memory_vector_store_operations():
    store = InMemoryVectorStore()
    assert store.count == 0
    assert store.search([1.0, 0.0]) == []

    unit1 = make_code_unit(unit_id="1", name="u1")
    unit2 = make_code_unit(unit_id="2", name="u2")
    store.add([[1.0, 0.0], [0.0, 1.0]], [unit1, unit2])

    assert store.count == 2
    results = store.search([1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0][0].id == "1"

    meta = store.get_metadata(0)
    assert meta["id"] == "1"

    store.clear()
    assert store.count == 0


# =============================================================================
# SentenceTransformerProvider Tests
# =============================================================================


def test_sentence_transformer_provider_loads_model_lazily():
    provider = SentenceTransformerProvider()

    assert provider._model is None
    assert provider.model_name == "sentence-transformers/all-MiniLM-L6-v2"


def test_sentence_transformer_provider_config():
    provider = SentenceTransformerProvider(
        model_name="all-MiniLM-L6-v2",
        batch_size=16,
        device="cpu",
        normalize_embeddings=True,
    )
    assert provider.model_name == "all-MiniLM-L6-v2"
    assert provider.batch_size == 16
    assert provider.device == "cpu"
    assert provider.normalize_embeddings is True


def test_sentence_transformer_provider_validation():
    with pytest.raises(ValueError, match="model_name"):
        SentenceTransformerProvider(model_name="   ")

    with pytest.raises(ValueError, match="batch_size"):
        SentenceTransformerProvider(batch_size=0)

    with pytest.raises(ValueError, match="batch_size"):
        SentenceTransformerProvider(batch_size=-5)

    provider = SentenceTransformerProvider()
    assert provider.encode([]) == []

    with pytest.raises(TypeError, match="strings"):
        provider.encode([123])  # type: ignore[list-item]


def test_sentence_transformer_provider_real_inference():
    """Verify that SentenceTransformerProvider can encode real text using the local model."""
    pytest.importorskip("sentence_transformers")
    provider = SentenceTransformerProvider(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=8,
    )
    texts = [
        "def authenticate_user(username, password): pass",
        "class DatabaseConnection: pass",
    ]
    embeddings = provider.encode(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384


# =============================================================================
# Document Construction & Metadata Tests
# =============================================================================


def test_make_document_captures_all_fields():
    unit = make_code_unit(
        unit_id="test",
        name="calculate",
        docstring="Compute total values.",
        source_code="def calculate(a, b):\n    return a + b",
        metadata={"author": "team", "deprecated": False},
    )
    doc = _make_document(unit)

    assert "Name: calculate" in doc
    assert "Docstring: Compute total values." in doc
    assert "Source code: def calculate" in doc
    assert "author: team" in doc


def test_semantic_search_result_properties():
    unit = make_code_unit(
        unit_id="u1",
        name="handler",
        file_path="pkg/handler.py",
        start_line=10,
        end_line=20,
        kind="function",
        source_code="pass",
        docstring="Handler doc",
        metadata={"k": "v"},
    )
    result = SemanticSearchResult(code_unit=unit, score=0.88, rank=1)

    assert result.code_unit_id == "u1"
    assert result.name == "handler"
    assert result.symbol == "handler"
    assert result.qualified_name == "module.handler"
    assert result.file_path == "pkg/handler.py"
    assert result.start_line == 10
    assert result.end_line == 20
    assert result.kind == "function"
    assert result.source_code == "pass"
    assert result.docstring == "Handler doc"
    assert result.metadata == {"k": "v"}
    assert result.score == 0.88
    assert result.rank == 1
