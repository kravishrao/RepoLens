from __future__ import annotations

import json
from pathlib import Path

from repolens.evaluation.models import (
    AblationComparison,
    AblationType,
    BaselineType,
    BenchmarkQuery,
    EvaluationMetrics,
    ExperimentConfig,
    ExperimentRunResult,
    FixedChunk,
    SingleQueryResult,
)


def test_baseline_type_values() -> None:
    """Verify all 6 standardized baselines B0 to B5 exist."""
    assert BaselineType.B0_NAIVE_RAG.value == "B0: Naive RAG"
    assert BaselineType.B1_AST_RAG.value == "B1: AST RAG"
    assert BaselineType.B2_HYBRID_RAG.value == "B2: Hybrid RAG"
    assert BaselineType.B3_GRAPH_RAG.value == "B3: Graph RAG"
    assert BaselineType.B4_AGENTIC_RAG.value == "B4: Agentic RAG"
    assert BaselineType.B5_FULL_REPOLENS.value == "B5: Full RepoLens"


def test_ablation_types() -> None:
    """Verify ablation enum values."""
    assert AblationType.FULL.value == "full"
    assert AblationType.NO_VERIFICATION.value == "no_verification"
    assert AblationType.NO_AGENT.value == "no_agent"
    assert AblationType.NO_GRAPH.value == "no_graph"
    assert AblationType.NO_RERANKER.value == "no_reranker"
    assert AblationType.NO_BM25.value == "no_bm25"
    assert AblationType.NO_AST.value == "no_ast"


def test_fixed_chunk_dataclass() -> None:
    """Verify FixedChunk creation and field access."""
    chunk = FixedChunk(
        chunk_id="file.py:chunk_0",
        file_path="file.py",
        start_char=0,
        end_char=100,
        start_line=1,
        end_line=5,
        content="def hello():\n    return 'world'\n",
        chunk_index=0,
        metadata={"size": 100},
    )
    assert chunk.chunk_id == "file.py:chunk_0"
    assert chunk.start_line == 1
    assert chunk.end_line == 5
    assert "hello" in chunk.content


def test_benchmark_query_dataclass() -> None:
    """Verify BenchmarkQuery fields."""
    query = BenchmarkQuery(
        query_id="q1",
        question="Where is Session defined?",
        repository="requests",
        gold_symbols=["Session"],
        gold_files=["sessions.py"],
        category="symbol_lookup",
    )
    assert query.query_id == "q1"
    assert query.gold_symbols == ["Session"]


def test_experiment_config_to_dict() -> None:
    """Verify ExperimentConfig serialization."""
    config = ExperimentConfig(
        baseline=BaselineType.B5_FULL_REPOLENS,
        repository_name="requests",
        repository_path=Path("dataset/repositories/requests"),
        top_k=10,
        ablation=AblationType.FULL,
    )
    d = config.to_dict()
    assert d["baseline"] == "B5: Full RepoLens"
    assert d["ablation"] == "full"
    assert d["top_k"] == 10
    assert "requests" in d["repository_path"]


def test_evaluation_metrics_defaults() -> None:
    """Verify EvaluationMetrics defaults and dictionary export."""
    metrics = EvaluationMetrics(
        latency_ms=150.5,
        tokens_used=420,
        hit_rate=1.0,
        precision_at_k=0.8,
        citation_precision=1.0,
        validation_status="valid",
    )
    d = metrics.to_dict()
    assert d["latency_ms"] == 150.5
    assert d["tokens_used"] == 420
    assert d["hit_rate"] == 1.0
    assert d["citation_precision"] == 1.0
    assert d["validation_status"] == "valid"


def test_single_query_result_to_dict() -> None:
    """Verify SingleQueryResult dictionary conversion."""
    res = SingleQueryResult(
        query_id="q1",
        query_text="Where is foo?",
        baseline="B5: Full RepoLens",
        answer="Found in file.py",
        retrieved_evidence=[{"file_path": "file.py"}],
        metrics=EvaluationMetrics(hit_rate=1.0),
        execution_trace=[{"step": 1, "tool": "search"}],
        validation_summary={"overall_status": "valid"},
    )
    d = res.to_dict()
    assert d["query_id"] == "q1"
    assert d["metrics"]["hit_rate"] == 1.0
    assert len(d["retrieved_evidence"]) == 1


def test_experiment_run_result_json_and_markdown() -> None:
    """Verify ExperimentRunResult serialization and markdown generation."""
    config = ExperimentConfig(baseline=BaselineType.B5_FULL_REPOLENS)
    run = ExperimentRunResult(
        experiment_id="test_exp_123",
        name="Test Experiment",
        timestamp="2026-11-19T10:00:00",
        config=config,
        system_info={"os": "Windows"},
        query_results=[
            SingleQueryResult(
                query_id="q1",
                query_text="Where is foo?",
                baseline=BaselineType.B5_FULL_REPOLENS.value,
                answer="Answer text",
                retrieved_evidence=[],
                metrics=EvaluationMetrics(hit_rate=1.0, latency_ms=120.0),
            )
        ],
        summary_metrics={"mean_hit_rate": 1.0, "mean_latency_ms": 120.0},
        limitations_notes=["Requires filesystem access for citation checks."],
    )

    json_str = run.to_json()
    parsed = json.loads(json_str)
    assert parsed["experiment_id"] == "test_exp_123"
    assert parsed["summary_metrics"]["mean_hit_rate"] == 1.0

    md = run.to_markdown_summary()
    assert "B5: Full RepoLens" in md
    assert "Mean Hit Rate" in md
    assert "Requires filesystem access" in md


def test_ablation_comparison_table() -> None:
    """Verify AblationComparison markdown table rendering."""
    config = ExperimentConfig(baseline=BaselineType.B0_NAIVE_RAG)
    run1 = ExperimentRunResult(
        experiment_id="e1",
        name="B0 Run",
        timestamp="2026-11-19T10:00:00",
        config=config,
        system_info={},
        query_results=[],
        summary_metrics={
            "mean_hit_rate": 0.6,
            "mean_citation_precision": 0.5,
            "mean_latency_ms": 80.0,
        },
    )
    comparison = AblationComparison(
        comparison_id="comp_1",
        repository_name="requests",
        runs={"B0: Naive RAG": run1},
    )
    table = comparison.to_markdown_table()
    assert "B0: Naive RAG" in table
    assert "60.0%" in table
    assert "Latency (ms)" in table
