"""Comprehensive unit tests for the RepoLens evaluation framework (RL-39)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repolens.evaluation.baselines import (
    B0_NaiveRAG,
    get_baseline_pipeline,
)
from repolens.evaluation.evaluator import RepoLensEvaluator
from repolens.evaluation.metrics import (
    compute_answer_correctness,
    compute_citation_precision,
    compute_evidence_coverage,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    estimate_token_count,
    evidence_matches,
    normalize_path,
    ranges_overlap,
)
from repolens.evaluation.models import (
    ConfigurationMetrics,
    PricingConfig,
    SampleEvaluationResult,
)
from repolens.evaluation.qa_dataset import GoldEvidenceLocation, QuestionSample
from repolens.evaluation.runner import run_evaluation


def make_gold_item(
    file_path: str = "src/requests/api.py",
    start_line: int = 60,
    end_line: int = 80,
    symbol: str = "get",
    role: str = "primary_evidence",
) -> GoldEvidenceLocation:
    """Helper to construct GoldEvidenceLocation."""
    return GoldEvidenceLocation(
        repository="requests",
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        symbol=symbol,
        role=role,
    )


def make_question(
    qid: str = "TEST-01",
    gold: list[GoldEvidenceLocation] | None = None,
    keywords: list[str] | None = None,
) -> QuestionSample:
    """Helper to construct QuestionSample."""
    return QuestionSample(
        id=qid,
        repository="requests",
        category="function",
        scope="single_file",
        complexity="single_hop",
        hop_count=1,
        elements_involved=["functions"],
        question="How does get() work?",
        expected_answer="get() calls Session.request() with method GET.",
        gold_evidence=gold or [make_gold_item()],
        keywords=keywords or ["get", "Session.request", "GET"],
    )


# =============================================================================
# 1. Metric Utilities Tests
# =============================================================================


def test_normalize_path():
    """Verify path normalization handles windows and posix slashes consistently."""
    assert normalize_path(r"src\requests\api.py") == "src/requests/api.py"
    assert normalize_path("./src/flask/app.py") == "src/flask/app.py"
    assert normalize_path("FASTAPI/ROUTING.PY") == "fastapi/routing.py"
    assert normalize_path("") == ""
    assert normalize_path(None) == ""


def test_ranges_overlap():
    """Verify 1-indexed range overlap detection."""
    assert ranges_overlap(10, 20, 15, 25) is True
    assert ranges_overlap(10, 20, 20, 30) is True  # Boundary contact
    assert ranges_overlap(10, 20, 21, 30) is False  # Disjoint
    assert ranges_overlap(50, 60, 1, 100) is True  # Superset
    assert ranges_overlap(1, 100, 50, 60) is True  # Subset


def test_evidence_matches():
    """Verify matching logic between retrieved items and gold evidence locations."""
    gold = make_gold_item(
        file_path="src/requests/api.py", start_line=64, end_line=77, symbol="get"
    )

    # Exact line overlap match
    ret_exact = {
        "file_path": "src/requests/api.py",
        "start_line": 60,
        "end_line": 75,
        "symbol": "get",
    }
    assert evidence_matches(ret_exact, gold) is True

    # Symbol match in same file with different lines
    ret_sym = {
        "file_path": "src/requests/api.py",
        "start_line": 1,
        "end_line": 10,
        "symbol": "api.get",
    }
    assert evidence_matches(ret_sym, gold) is True

    # Mismatched file
    ret_wrong_file = {
        "file_path": "src/requests/models.py",
        "start_line": 64,
        "end_line": 77,
        "symbol": "get",
    }
    assert evidence_matches(ret_wrong_file, gold) is False


# =============================================================================
# 2. Information Retrieval Metrics Tests (Recall, Precision, MRR, nDCG)
# =============================================================================


def test_retrieval_metrics_perfect_ranking():
    """Verify metrics for a perfect ranking where all gold items are in top positions."""
    gold1 = make_gold_item(file_path="a.py", start_line=10, end_line=20)
    gold2 = make_gold_item(file_path="b.py", start_line=30, end_line=40)
    gold = [gold1, gold2]

    retrieved = [
        {"file_path": "a.py", "start_line": 10, "end_line": 20},
        {"file_path": "b.py", "start_line": 30, "end_line": 40},
        {"file_path": "c.py", "start_line": 1, "end_line": 10},
    ]

    assert compute_recall_at_k(retrieved, gold, k=2) == 1.0
    assert compute_precision_at_k(retrieved, gold, k=2) == 1.0
    assert compute_precision_at_k(retrieved, gold, k=3) == round(2 / 3, 4)
    assert compute_mrr(retrieved, gold, k=3) == 1.0
    assert compute_ndcg_at_k(retrieved, gold, k=2) == 1.0


def test_retrieval_metrics_inverted_ranking():
    """Verify MRR and nDCG penalize relevant items placed at lower ranks."""
    gold = [make_gold_item(file_path="target.py", start_line=10, end_line=20)]

    # Item at rank 3
    retrieved = [
        {"file_path": "distractor1.py", "start_line": 1, "end_line": 10},
        {"file_path": "distractor2.py", "start_line": 1, "end_line": 10},
        {"file_path": "target.py", "start_line": 10, "end_line": 20},
    ]

    assert compute_recall_at_k(retrieved, gold, k=1) == 0.0
    assert compute_recall_at_k(retrieved, gold, k=3) == 1.0
    assert compute_precision_at_k(retrieved, gold, k=1) == 0.0
    assert compute_precision_at_k(retrieved, gold, k=3) == round(1 / 3, 4)
    assert compute_mrr(retrieved, gold, k=3) == round(1 / 3, 4)

    # nDCG at k=3 when relevant item is at rank 3
    ndcg3 = compute_ndcg_at_k(retrieved, gold, k=3)
    assert 0.0 < ndcg3 < 1.0


def test_retrieval_metrics_zero_matches_and_empty_lists():
    """Verify metrics handle zero matches and empty inputs gracefully."""
    gold = [make_gold_item(file_path="a.py", start_line=1, end_line=10)]
    distractors = [{"file_path": "x.py", "start_line": 1, "end_line": 10}]

    assert compute_recall_at_k(distractors, gold, k=5) == 0.0
    assert compute_precision_at_k(distractors, gold, k=5) == 0.0
    assert compute_mrr(distractors, gold, k=5) == 0.0
    assert compute_ndcg_at_k(distractors, gold, k=5) == 0.0

    # Empty inputs
    assert compute_recall_at_k([], gold, k=5) == 0.0
    assert compute_precision_at_k([], gold, k=5) == 0.0
    assert compute_mrr([], gold, k=5) == 0.0
    assert compute_ndcg_at_k([], gold, k=5) == 0.0


# =============================================================================
# 3. Answer Correctness Evaluation Tests
# =============================================================================


def test_answer_correctness_identical_strings():
    """Verify answer correctness yields 1.0 for identical answers."""
    ans = "Session.request dispatches the HTTP GET call."
    res = compute_answer_correctness(ans, ans, keywords=["Session.request", "GET"])

    assert res.token_precision == 1.0
    assert res.token_recall == 1.0
    assert res.token_f1 == 1.0
    assert res.keyword_coverage == 1.0
    assert res.overall_correctness == 1.0


def test_answer_correctness_partial_overlap():
    """Verify partial overlap computes balanced token F1 and keyword coverage."""
    generated = "Session.request sends the request."
    expected = "Session.request prepares and dispatches the request using HTTPAdapter."
    keywords = ["Session.request", "HTTPAdapter"]

    res = compute_answer_correctness(generated, expected, keywords=keywords)
    assert 0.0 < res.token_f1 < 1.0
    assert res.keyword_coverage == 0.5  # Session.request present, HTTPAdapter missing
    assert 0.0 < res.overall_correctness < 1.0


def test_answer_correctness_empty_answer():
    """Verify empty generated answer yields 0.0 score."""
    res = compute_answer_correctness("", "Some expected answer.", keywords=["keyword"])
    assert res.token_f1 == 0.0
    assert res.keyword_coverage == 0.0
    assert res.overall_correctness == 0.0


# =============================================================================
# 4. Evidence Coverage & Citation Precision Tests
# =============================================================================


def test_evidence_coverage_and_citation_precision():
    """Verify evidence coverage and citation precision calculations."""
    gold1 = make_gold_item(file_path="src/a.py", start_line=10, end_line=20)
    gold2 = make_gold_item(file_path="src/b.py", start_line=30, end_line=40)
    gold = [gold1, gold2]

    # Only gold1 is cited; one valid citation and one invalid citation
    citations = [
        {"file_path": "src/a.py", "start_line": 10, "end_line": 20},
        {"file_path": "src/hallucinated.py", "start_line": 1, "end_line": 5},
    ]

    cov = compute_evidence_coverage(citations, gold)
    assert cov == 0.5  # 1 of 2 gold locations covered

    prec = compute_citation_precision(citations, gold)
    assert prec == 0.5  # 1 valid citation out of 2 total citations


# =============================================================================
# 5. Efficiency & Cost Estimation Tests
# =============================================================================


def test_token_and_cost_estimation():
    """Verify token counting and USD inference cost calculations."""
    text = "A" * 400
    tokens = estimate_token_count(text)
    assert tokens == 100

    pricing = PricingConfig(
        model_name="test-model",
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
    )
    cost = pricing.estimate_cost(prompt_tokens=1_000_000, completion_tokens=500_000)
    assert pytest.approx(cost, rel=1e-5) == 0.45


# =============================================================================
# 6. Sample & Configuration Evaluation Pipeline Tests
# =============================================================================


def test_evaluate_sample_against_gold_evidence():
    """Verify SampleEvaluationResult evaluates a QuestionSample comprehensively."""
    sample = make_question(
        qid="REQ-01",
        gold=[
            make_gold_item(file_path="src/requests/api.py", start_line=60, end_line=80)
        ],
        keywords=["get", "Session.request"],
    )

    retrieved = [{"file_path": "src/requests/api.py", "start_line": 64, "end_line": 77}]
    generated_answer = "get() calls Session.request() to dispatch the request."

    result = SampleEvaluationResult.evaluate(
        sample=sample,
        config_name="B5: Full RepoLens",
        retrieved_items=retrieved,
        generated_answer=generated_answer,
        citations=retrieved,
        latency_ms=125.0,
    )

    assert result.question_id == "REQ-01"
    assert result.retrieval.recall_at_1 == 1.0
    assert result.retrieval.precision_at_1 == 1.0
    assert result.retrieval.mrr == 1.0
    assert result.evidence.coverage == 1.0
    assert result.evidence.citation_precision == 1.0
    assert result.answer.overall_correctness > 0.5
    assert result.efficiency.latency_ms == 125.0
    assert result.efficiency.estimated_cost_usd > 0.0


def test_configuration_metrics_aggregation():
    """Verify ConfigurationMetrics properly averages sample results."""
    sample1 = make_question(qid="Q1")
    sample2 = make_question(qid="Q2")

    r1 = SampleEvaluationResult.evaluate(
        sample=sample1,
        config_name="B2",
        retrieved_items=[
            {"file_path": "src/requests/api.py", "start_line": 60, "end_line": 80}
        ],
        generated_answer="Answer 1",
        latency_ms=100.0,
    )
    r2 = SampleEvaluationResult.evaluate(
        sample=sample2,
        config_name="B2",
        retrieved_items=[],  # Zero retrieval
        generated_answer="No answer",
        latency_ms=200.0,
    )

    config_metrics = ConfigurationMetrics.aggregate("B2", [r1, r2])

    assert config_metrics.config_name == "B2"
    assert config_metrics.total_samples == 2
    assert config_metrics.mean_recall_at_5 == 0.5  # (1.0 + 0.0) / 2
    assert config_metrics.mean_latency_ms == 150.0  # (100 + 200) / 2


# =============================================================================
# 7. Baseline Pipelines & Reproducibility Tests
# =============================================================================


def test_baseline_pipeline_factory():
    """Verify get_baseline_pipeline correctly instantiates B0 through B5."""
    for key in ["B0", "B1", "B2", "B3", "B4", "B5"]:
        pipeline = get_baseline_pipeline(key)
        assert pipeline is not None
        assert key in pipeline.config_name


def test_reproducible_evaluation_runs_with_seed():
    """Verify evaluations with identical seed produce exactly identical results."""
    sample = make_question(qid="DET-01")

    p1 = B0_NaiveRAG(seed=123)
    p2 = B0_NaiveRAG(seed=123)

    res1 = p1.execute(sample)
    res2 = p2.execute(sample)

    assert res1.retrieved_items == res2.retrieved_items
    assert res1.generated_answer == res2.generated_answer
    assert res1.is_grounded == res2.is_grounded


# =============================================================================
# 8. Benchmark Comparison & Report Export Tests
# =============================================================================


def test_benchmark_comparison_report_markdown_and_json(tmp_path: Path):
    """Verify BenchmarkComparisonReport generates markdown tables and serializes to JSON."""
    sample = make_question(qid="TEST-COMPARE")
    evaluator = RepoLensEvaluator(seed=42)

    report = evaluator.compare_configurations(
        configs=["B0", "B5"],
        dataset=[sample],
    )

    assert len(report.configurations) == 2
    assert report.configurations[0].config_name == "B0: Naive RAG"
    assert report.configurations[1].config_name == "B5: Full RepoLens"

    # Test Markdown generation
    md = report.to_markdown()
    assert "# RepoLens Multi-Configuration Evaluation Benchmark" in md
    assert "| Configuration | Recall@5 | Prec@5 | MRR | nDCG@5 |" in md
    assert "B0: Naive RAG" in md
    assert "B5: Full RepoLens" in md

    # Test JSON serialization roundtrip
    json_path = tmp_path / "report.json"
    report.save_json(json_path)
    assert json_path.exists()

    loaded_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded_data["benchmark_name"] == report.benchmark_name
    assert len(loaded_data["configurations"]) == 2


def test_run_evaluation_runner_helper(tmp_path: Path):
    """Verify run_evaluation end-to-end execution on dataset."""
    json_out = tmp_path / "bench.json"
    md_out = tmp_path / "bench.md"

    report = run_evaluation(
        configs=["B1", "B3"],
        limit=2,
        output_json=json_out,
        output_markdown=md_out,
    )

    assert len(report.configurations) == 2
    assert json_out.exists()
    assert md_out.exists()
    assert "# RepoLens Multi-Configuration Evaluation Benchmark" in md_out.read_text(
        encoding="utf-8"
    )
