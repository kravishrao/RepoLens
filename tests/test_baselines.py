from __future__ import annotations

from pathlib import Path

import pytest

from repolens.evaluation.baselines import (
    AgenticRAGPipeline,
    ASTStructureRAGPipeline,
    FullRepoLensPipeline,
    GraphRAGPipeline,
    HybridRAGPipeline,
    NaiveRAGPipeline,
    RepositoryContext,
    create_baseline_pipeline,
)
from repolens.evaluation.embeddings import DeterministicHashingEmbeddingProvider
from repolens.evaluation.models import (
    BaselineType,
    BenchmarkQuery,
    ExperimentConfig,
)
from repolens.ingestion.code_unit import CodeUnit


@pytest.fixture
def sample_code_units() -> list[CodeUnit]:
    """Sample AST CodeUnits for fast isolated baseline tests."""
    return [
        CodeUnit(
            id="auth.py:authenticate",
            name="authenticate",
            qualified_name="auth.authenticate",
            kind="function",
            file_path="auth.py",
            start_line=1,
            end_line=10,
            docstring="Verify user credentials.",
            source_code="def authenticate(user, password):\n    return user == 'admin'",
        ),
        CodeUnit(
            id="server.py:dispatch",
            name="dispatch",
            qualified_name="server.dispatch",
            kind="function",
            file_path="server.py",
            start_line=1,
            end_line=15,
            docstring="Dispatch incoming HTTP request.",
            source_code="def dispatch(request):\n    authenticate(request.user, request.pw)\n    return 200",
        ),
        CodeUnit(
            id="models.py:User",
            name="User",
            qualified_name="models.User",
            kind="class",
            file_path="models.py",
            start_line=1,
            end_line=20,
            docstring="User database model.",
            source_code="class User:\n    def __init__(self, name):\n        self.name = name",
        ),
    ]


@pytest.fixture
def sample_context(sample_code_units: list[CodeUnit]) -> RepositoryContext:
    """Fixture providing an initialized RepositoryContext."""
    provider = DeterministicHashingEmbeddingProvider(dimension=32)
    return RepositoryContext.from_code_units(
        code_units=sample_code_units,
        repository_name="SampleAuthRepo",
        embedding_provider=provider,
    )


def test_create_baseline_pipeline_factory(sample_context: RepositoryContext) -> None:
    """Verify factory instantiates the correct pipeline for all B0 to B5 types."""
    p0 = create_baseline_pipeline(BaselineType.B0_NAIVE_RAG, sample_context)
    assert isinstance(p0, NaiveRAGPipeline)

    p1 = create_baseline_pipeline(BaselineType.B1_AST_RAG, sample_context)
    assert isinstance(p1, ASTStructureRAGPipeline)

    p2 = create_baseline_pipeline(BaselineType.B2_HYBRID_RAG, sample_context)
    assert isinstance(p2, HybridRAGPipeline)

    p3 = create_baseline_pipeline(BaselineType.B3_GRAPH_RAG, sample_context)
    assert isinstance(p3, GraphRAGPipeline)

    p4 = create_baseline_pipeline(BaselineType.B4_AGENTIC_RAG, sample_context)
    assert isinstance(p4, AgenticRAGPipeline)

    p5 = create_baseline_pipeline(BaselineType.B5_FULL_REPOLENS, sample_context)
    assert isinstance(p5, FullRepoLensPipeline)


def test_create_baseline_pipeline_unknown_type(
    sample_context: RepositoryContext,
) -> None:
    """Verify unknown baseline raises ValueError."""
    with pytest.raises(ValueError, match="Unknown baseline type"):
        create_baseline_pipeline("B99: Quantum RAG", sample_context)


def test_b0_naive_rag_execution(sample_context: RepositoryContext) -> None:
    """Verify B0 Naive RAG executes chunk retrieval and synthesis."""
    pipeline = NaiveRAGPipeline(sample_context)
    query = BenchmarkQuery(
        query_id="q_auth",
        question="Where is authenticate defined?",
        repository="SampleAuthRepo",
        gold_symbols=["authenticate"],
        gold_files=["auth.py"],
    )
    res = pipeline.execute(query, top_k=2)

    assert res.baseline == BaselineType.B0_NAIVE_RAG.value
    assert len(res.retrieved_evidence) >= 1
    assert res.metrics.latency_ms > 0
    assert len(res.execution_trace) >= 2


def test_b1_ast_rag_execution(sample_context: RepositoryContext) -> None:
    """Verify B1 AST RAG retrieves CodeUnits and generates answer."""
    pipeline = ASTStructureRAGPipeline(sample_context)
    res = pipeline.execute("How does dispatch work?", top_k=2)

    assert res.baseline == BaselineType.B1_AST_RAG.value
    assert len(res.retrieved_evidence) >= 1
    assert any("dispatch" in e.get("name", "") for e in res.retrieved_evidence)
    assert res.metrics.evidence_count >= 1


def test_b2_hybrid_rag_execution(sample_context: RepositoryContext) -> None:
    """Verify B2 Hybrid RAG combines BM25 and semantic search with reranking."""
    cfg = ExperimentConfig(baseline=BaselineType.B2_HYBRID_RAG, enable_reranker=True)
    pipeline = HybridRAGPipeline(sample_context, cfg)
    res = pipeline.execute("authenticate user credentials", top_k=2)

    assert res.baseline == BaselineType.B2_HYBRID_RAG.value
    assert len(res.retrieved_evidence) >= 1
    trace_tools = [t.get("tool") for t in res.execution_trace]
    assert "cross_encoder_rerank" in trace_tools


def test_b3_graph_rag_execution(sample_context: RepositoryContext) -> None:
    """Verify B3 Graph RAG expands context via CodeGraph."""
    cfg = ExperimentConfig(baseline=BaselineType.B3_GRAPH_RAG, max_hops=1)
    pipeline = GraphRAGPipeline(sample_context, cfg)
    res = pipeline.execute("dispatch request", top_k=2)

    assert res.baseline == BaselineType.B3_GRAPH_RAG.value
    trace_tools = [t.get("tool") for t in res.execution_trace]
    assert "codegraph_expansion" in trace_tools


def test_b4_agentic_rag_execution(sample_context: RepositoryContext) -> None:
    """Verify B4 Agentic RAG runs iterative agent with unverified status."""
    cfg = ExperimentConfig(baseline=BaselineType.B4_AGENTIC_RAG, max_agent_steps=2)
    pipeline = AgenticRAGPipeline(sample_context, cfg)
    res = pipeline.execute("Find authentication logic", top_k=2)

    assert res.baseline == BaselineType.B4_AGENTIC_RAG.value
    assert res.metrics.validation_status in ("unverified", "disabled")


def test_b5_full_repolens_execution(sample_context: RepositoryContext) -> None:
    """Verify B5 Full RepoLens runs agent with ground-truth verification enabled."""
    cfg = ExperimentConfig(baseline=BaselineType.B5_FULL_REPOLENS, max_agent_steps=2)
    pipeline = FullRepoLensPipeline(sample_context, cfg)
    res = pipeline.execute("Where is authenticate defined?", top_k=2)

    assert res.baseline == BaselineType.B5_FULL_REPOLENS.value
    assert "validation_summary" in res.to_dict()


def test_repository_context_from_directory(tmp_path: Path) -> None:
    """Verify RepositoryContext initializes cleanly from a filesystem folder."""
    repo_dir = tmp_path / "mini_repo"
    repo_dir.mkdir()
    (repo_dir / "calc.py").write_text(
        "def add(x, y):\n    '''Add numbers.'''\n    return x + y\n",
        encoding="utf-8",
    )

    ctx = RepositoryContext.from_directory(repo_dir)
    assert ctx.repository_name == "mini_repo"
    assert len(ctx.code_units) >= 1
    assert len(ctx.fixed_chunks) >= 1
    assert ctx.bm25_retriever is not None
    assert ctx.semantic_retriever is not None
