"""RepoLens evaluation framework, benchmarks, and dataset loaders."""

from __future__ import annotations

from repolens.evaluation.baselines import (
    B0_NaiveRAG,
    B1_ASTAwareRAG,
    B2_HybridRAG,
    B3_GraphRAG,
    B4_AgenticRAG,
    B5_FullRepoLens,
    BasePipeline,
    PipelineExecutionResult,
    get_baseline_pipeline,
)
from repolens.evaluation.evaluator import RepoLensEvaluator
from repolens.evaluation.metrics import (
    AnswerCorrectnessResult,
    compute_answer_correctness,
    compute_citation_precision,
    compute_evidence_coverage,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    estimate_inference_cost,
    estimate_token_count,
    evidence_matches,
)
from repolens.evaluation.models import (
    AnswerMetrics,
    BenchmarkComparisonReport,
    ConfigurationMetrics,
    EfficiencyMetrics,
    EvidenceMetrics,
    PricingConfig,
    RetrievalMetrics,
    SampleEvaluationResult,
)
from repolens.evaluation.qa_dataset import (
    GoldEvidenceLocation,
    QAEvaluationDataset,
    QuestionSample,
    load_qa_dataset,
    validate_qa_dataset,
)

__all__ = [
    "AnswerCorrectnessResult",
    "AnswerMetrics",
    "B0_NaiveRAG",
    "B1_ASTAwareRAG",
    "B2_HybridRAG",
    "B3_GraphRAG",
    "B4_AgenticRAG",
    "B5_FullRepoLens",
    "BasePipeline",
    "BenchmarkComparisonReport",
    "ConfigurationMetrics",
    "EfficiencyMetrics",
    "EvidenceMetrics",
    "GoldEvidenceLocation",
    "PipelineExecutionResult",
    "PricingConfig",
    "QAEvaluationDataset",
    "QuestionSample",
    "RepoLensEvaluator",
    "RetrievalMetrics",
    "SampleEvaluationResult",
    "compute_answer_correctness",
    "compute_citation_precision",
    "compute_evidence_coverage",
    "compute_mrr",
    "compute_ndcg_at_k",
    "compute_precision_at_k",
    "compute_recall_at_k",
    "estimate_inference_cost",
    "estimate_token_count",
    "evidence_matches",
    "get_baseline_pipeline",
    "load_qa_dataset",
    "validate_qa_dataset",
]
