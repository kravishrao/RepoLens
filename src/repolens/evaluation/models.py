"""Evaluation and Ablation Experiment Data Models for RepoLens.

Defines configuration schemas, benchmark query structures, baseline types,
metrics containers, reproducible serialization models, and benchmark report
models across B0-B5 configurations.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

from repolens.evaluation.metrics import (
    compute_answer_correctness,
    compute_citation_precision,
    compute_evidence_coverage,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    estimate_inference_cost,
    estimate_token_count,
)
from repolens.evaluation.qa_dataset import QuestionSample

# =============================================================================
# Baseline & Ablation Configuration Types
# =============================================================================


class BaselineType(StrEnum):
    """Supported baseline and experimental pipeline configurations."""

    B0_NAIVE_RAG = "B0: Naive RAG"
    B1_AST_RAG = "B1: AST RAG"
    B2_HYBRID_RAG = "B2: Hybrid RAG"
    B3_GRAPH_RAG = "B3: Graph RAG"
    B4_AGENTIC_RAG = "B4: Agentic RAG"
    B5_FULL_REPOLENS = "B5: Full RepoLens"


class AblationType(StrEnum):
    """Component ablation variants to assess marginal contributions."""

    FULL = "full"
    NO_VERIFICATION = "no_verification"
    NO_AGENT = "no_agent"
    NO_GRAPH = "no_graph"
    NO_RERANKER = "no_reranker"
    NO_BM25 = "no_bm25"
    NO_SEMANTIC = "no_semantic"
    NO_AST = "no_ast"


@dataclass(frozen=True)
class FixedChunk:
    """Fixed-size chunk of source code for naive RAG baseline (B0)."""

    chunk_id: str
    file_path: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    content: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkQuery:
    """Standardized repository investigation evaluation query."""

    query_id: str
    question: str
    repository: str
    gold_symbols: list[str] = field(default_factory=list)
    gold_files: list[str] = field(default_factory=list)
    category: str = "general"
    description: str = ""


@dataclass
class ExperimentConfig:
    """Reproducible configuration for baseline or ablation execution."""

    baseline: BaselineType | str = BaselineType.B5_FULL_REPOLENS
    repository_name: str = "RepoLens"
    repository_path: str | Path | None = None
    top_k: int = 5
    chunk_size: int = 500  # Used by B0
    chunk_overlap: int = 50  # Used by B0
    max_hops: int = 2  # Used by B3, B5
    max_agent_steps: int = 4  # Used by B4, B5
    enable_reranker: bool = True
    enable_verification: bool = True
    fusion_strategy: str = "rrf"
    random_seed: int = 42
    ablation: AblationType | str = AblationType.FULL
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert config to JSON-serializable dictionary."""
        data = asdict(self)
        if isinstance(self.baseline, Enum):
            data["baseline"] = self.baseline.value
        if isinstance(self.ablation, Enum):
            data["ablation"] = self.ablation.value
        if self.repository_path is not None:
            data["repository_path"] = str(self.repository_path)
        return data


@dataclass
class EvaluationMetrics:
    """Quantitative evaluation metrics for an individual query execution."""

    latency_ms: float = 0.0
    tokens_used: int = 0
    evidence_count: int = 0
    hit_rate: float = 0.0
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    citation_precision: float = 0.0
    citation_count: int = 0
    valid_citation_count: int = 0
    unverified_claim_count: int = 0
    validation_status: str = "disabled"

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass
class SingleQueryResult:
    """Result of running an evaluation query against a baseline."""

    query_id: str
    query_text: str
    baseline: str
    answer: str
    retrieved_evidence: list[dict[str, Any]]
    metrics: EvaluationMetrics
    execution_trace: list[dict[str, Any]] = field(default_factory=list)
    validation_summary: dict[str, Any] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert query result to JSON-serializable dictionary."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "baseline": self.baseline,
            "answer": self.answer,
            "retrieved_evidence": self.retrieved_evidence,
            "metrics": self.metrics.to_dict(),
            "execution_trace": self.execution_trace,
            "validation_summary": self.validation_summary,
            "raw_metadata": self.raw_metadata,
        }


@dataclass
class ExperimentRunResult:
    """Complete recorded experiment run over a benchmark query suite."""

    experiment_id: str
    name: str
    timestamp: str
    config: ExperimentConfig
    system_info: dict[str, Any]
    query_results: list[SingleQueryResult]
    summary_metrics: dict[str, float] = field(default_factory=dict)
    limitations_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert run result to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "timestamp": self.timestamp,
            "config": self.config.to_dict(),
            "system_info": self.system_info,
            "summary_metrics": self.summary_metrics,
            "limitations_notes": self.limitations_notes,
            "query_results": [q.to_dict() for q in self.query_results],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize run result to reproducible formatted JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown_summary(self) -> str:
        """Render a readable markdown summary of the experiment result."""
        lines = [
            f"### Experiment: {self.name} (`{self.config.baseline}`)",
            f"- **Experiment ID:** `{self.experiment_id}`",
            f"- **Timestamp:** {self.timestamp}",
            f"- **Target Repository:** `{self.config.repository_name}`",
            f"- **Evaluated Queries:** {len(self.query_results)}",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
        ]
        for key, val in self.summary_metrics.items():
            if isinstance(val, float):
                formatted = (
                    f"{val:.2%}"
                    if "rate" in key or "precision" in key or "recall" in key
                    else f"{val:.2f}"
                )
            else:
                formatted = str(val)
            lines.append(f"| {key.replace('_', ' ').title()} | {formatted} |")

        if self.limitations_notes:
            lines.extend(["", "**Configuration Limitations & Characteristics:**"])
            for note in self.limitations_notes:
                lines.append(f"- {note}")

        return "\n".join(lines)


@dataclass
class AblationComparison:
    """Comparative summary across multiple baseline or ablation runs."""

    comparison_id: str
    repository_name: str
    runs: dict[str, ExperimentRunResult] = field(default_factory=dict)

    def to_dataframe_rows(self) -> list[dict[str, Any]]:
        """Extract structured tabular rows for dashboard display."""
        rows: list[dict[str, Any]] = []
        for name, run in self.runs.items():
            metrics = run.summary_metrics
            rows.append(
                {
                    "Baseline": name,
                    "Hit Rate": f"{metrics.get('mean_hit_rate', 0.0):.1%}",
                    "Citation Precision": f"{metrics.get('mean_citation_precision', 0.0):.1%}",
                    "Evidence Coverage": f"{metrics.get('mean_precision_at_k', 0.0):.1%}",
                    "Latency (ms)": f"{metrics.get('mean_latency_ms', 0.0):.0f}",
                    "Tokens": f"{metrics.get('mean_tokens_used', 0.0):.0f}",
                    "Status": metrics.get("validation_status", "Complete"),
                }
            )
        return rows

    def to_markdown_table(self) -> str:
        """Render markdown comparison table."""
        rows = self.to_dataframe_rows()
        if not rows:
            return "No runs available."
        headers = list(rows[0].keys())
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join([":---"] * len(headers)) + " |"
        data_lines = [
            "| " + " | ".join(str(row[h]) for h in headers) + " |" for row in rows
        ]
        return "\n".join([header_line, sep_line, *data_lines])


# =============================================================================
# Evaluation Benchmark & Reporting Models
# =============================================================================


@dataclass(frozen=True)
class PricingConfig:
    """Configurable token pricing model for LLM inference cost estimation."""

    model_name: str = "gemini-1.5-pro"
    input_cost_per_million: float = 0.15
    output_cost_per_million: float = 0.60

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate USD cost for prompt and completion token counts."""
        return estimate_inference_cost(
            prompt_tokens,
            completion_tokens,
            self.input_cost_per_million,
            self.output_cost_per_million,
        )


@dataclass(frozen=True)
class RetrievalMetrics:
    """Information retrieval quality metrics for top-K results."""

    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class EvidenceMetrics:
    """Evidence coverage, citation precision, and hallucination metrics."""

    coverage: float
    citation_precision: float
    unsupported_citation_rate: float

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class AnswerMetrics:
    """Text-based answer correctness and grounding metrics."""

    token_precision: float
    token_recall: float
    token_f1: float
    keyword_coverage: float
    overall_correctness: float

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class EfficiencyMetrics:
    """Computational efficiency and cost estimation metrics."""

    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class SampleEvaluationResult:
    """Complete evaluation outcome for an individual QuestionSample."""

    question_id: str
    repository: str
    category: str
    scope: str
    complexity: str
    config_name: str
    retrieval: RetrievalMetrics
    evidence: EvidenceMetrics
    answer: AnswerMetrics
    efficiency: EfficiencyMetrics
    retrieved_count: int
    citations_count: int
    is_grounded: bool
    evidence_status: str

    def to_dict(self) -> dict[str, Any]:
        """Convert sample result to dictionary."""
        return {
            "question_id": self.question_id,
            "repository": self.repository,
            "category": self.category,
            "scope": self.scope,
            "complexity": self.complexity,
            "config_name": self.config_name,
            "retrieval": self.retrieval.to_dict(),
            "evidence": self.evidence.to_dict(),
            "answer": self.answer.to_dict(),
            "efficiency": self.efficiency.to_dict(),
            "retrieved_count": self.retrieved_count,
            "citations_count": self.citations_count,
            "is_grounded": self.is_grounded,
            "evidence_status": self.evidence_status,
        }

    @classmethod
    def evaluate(
        cls,
        sample: QuestionSample,
        config_name: str,
        retrieved_items: Sequence[Any],
        generated_answer: str,
        citations: Sequence[Any] | None = None,
        latency_ms: float = 0.0,
        pricing: PricingConfig | None = None,
        is_grounded: bool = True,
        evidence_status: str = "SUFFICIENT",
    ) -> SampleEvaluationResult:
        """Evaluate a sample's retrieved results and answer against ground truth."""
        pricing_cfg = pricing or PricingConfig()
        gold = sample.gold_evidence

        # Retrieval metrics
        r1 = compute_recall_at_k(retrieved_items, gold, k=1)
        r5 = compute_recall_at_k(retrieved_items, gold, k=5)
        r10 = compute_recall_at_k(retrieved_items, gold, k=10)

        p1 = compute_precision_at_k(retrieved_items, gold, k=1)
        p5 = compute_precision_at_k(retrieved_items, gold, k=5)
        p10 = compute_precision_at_k(retrieved_items, gold, k=10)

        mrr = compute_mrr(retrieved_items, gold)
        ndcg5 = compute_ndcg_at_k(retrieved_items, gold, k=5)
        ndcg10 = compute_ndcg_at_k(retrieved_items, gold, k=10)

        retrieval = RetrievalMetrics(
            recall_at_1=r1,
            recall_at_5=r5,
            recall_at_10=r10,
            precision_at_1=p1,
            precision_at_5=p5,
            precision_at_10=p10,
            mrr=mrr,
            ndcg_at_5=ndcg5,
            ndcg_at_10=ndcg10,
        )

        # Evidence metrics
        cits = list(citations or [])
        cov = compute_evidence_coverage(retrieved_items, gold)
        cit_prec = compute_citation_precision(cits, gold)
        unsupported_rate = round(1.0 - cit_prec, 4)

        evidence = EvidenceMetrics(
            coverage=cov,
            citation_precision=cit_prec,
            unsupported_citation_rate=unsupported_rate,
        )

        # Answer metrics
        ans_res = compute_answer_correctness(
            generated_answer,
            sample.expected_answer,
            sample.keywords,
        )
        answer = AnswerMetrics(
            token_precision=ans_res.token_precision,
            token_recall=ans_res.token_recall,
            token_f1=ans_res.token_f1,
            keyword_coverage=ans_res.keyword_coverage,
            overall_correctness=ans_res.overall_correctness,
        )

        # Efficiency metrics
        prompt_tokens = estimate_token_count(sample.question) + sum(
            estimate_token_count(
                getattr(r, "source_code", None)
                or (r.get("content") if isinstance(r, dict) else "")
            )
            for r in retrieved_items[:5]
        )
        completion_tokens = estimate_token_count(generated_answer)
        total_tokens = prompt_tokens + completion_tokens
        cost = pricing_cfg.estimate_cost(prompt_tokens, completion_tokens)

        efficiency = EfficiencyMetrics(
            latency_ms=round(latency_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
        )

        return cls(
            question_id=sample.id,
            repository=sample.repository,
            category=sample.category,
            scope=sample.scope,
            complexity=sample.complexity,
            config_name=config_name,
            retrieval=retrieval,
            evidence=evidence,
            answer=answer,
            efficiency=efficiency,
            retrieved_count=len(retrieved_items),
            citations_count=len(cits),
            is_grounded=is_grounded,
            evidence_status=evidence_status,
        )


@dataclass(frozen=True)
class ConfigurationMetrics:
    """Aggregated evaluation metrics for a single baseline configuration."""

    config_name: str
    sample_count: int
    mean_recall_at_1: float
    mean_recall_at_5: float
    mean_recall_at_10: float
    mean_precision_at_1: float
    mean_precision_at_5: float
    mean_precision_at_10: float
    mean_mrr: float
    mean_ndcg_at_5: float
    mean_ndcg_at_10: float
    mean_evidence_coverage: float
    mean_citation_precision: float
    mean_unsupported_citation_rate: float
    mean_token_precision: float
    mean_token_recall: float
    mean_token_f1: float
    mean_keyword_coverage: float
    mean_answer_correctness: float
    mean_latency_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_estimated_cost_usd: float
    per_repo_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    sample_results: list[SampleEvaluationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration metrics to dictionary."""
        data = asdict(self)
        data["sample_results"] = [s.to_dict() for s in self.sample_results]
        return data

    @property
    def total_samples(self) -> int:
        """Alias for sample_count."""
        return self.sample_count

    @property
    def mean_answer_f1(self) -> float:
        """Alias for mean_token_f1."""
        return self.mean_token_f1

    @classmethod
    def aggregate(
        cls,
        config_name: str,
        sample_results: Sequence[SampleEvaluationResult],
    ) -> ConfigurationMetrics:
        """Aggregate per-sample results into summary configuration metrics."""
        count = len(sample_results)
        if count == 0:
            return cls(
                config_name=config_name,
                sample_count=0,
                mean_recall_at_1=0.0,
                mean_recall_at_5=0.0,
                mean_recall_at_10=0.0,
                mean_precision_at_1=0.0,
                mean_precision_at_5=0.0,
                mean_precision_at_10=0.0,
                mean_mrr=0.0,
                mean_ndcg_at_5=0.0,
                mean_ndcg_at_10=0.0,
                mean_evidence_coverage=0.0,
                mean_citation_precision=0.0,
                mean_unsupported_citation_rate=0.0,
                mean_token_precision=0.0,
                mean_token_recall=0.0,
                mean_token_f1=0.0,
                mean_keyword_coverage=0.0,
                mean_answer_correctness=0.0,
                mean_latency_ms=0.0,
                total_prompt_tokens=0,
                total_completion_tokens=0,
                total_tokens=0,
                total_estimated_cost_usd=0.0,
                per_repo_metrics={},
                sample_results=[],
            )

        mean_r1 = round(sum(s.retrieval.recall_at_1 for s in sample_results) / count, 4)
        mean_r5 = round(sum(s.retrieval.recall_at_5 for s in sample_results) / count, 4)
        mean_r10 = round(
            sum(s.retrieval.recall_at_10 for s in sample_results) / count, 4
        )

        mean_p1 = round(
            sum(s.retrieval.precision_at_1 for s in sample_results) / count, 4
        )
        mean_p5 = round(
            sum(s.retrieval.precision_at_5 for s in sample_results) / count, 4
        )
        mean_p10 = round(
            sum(s.retrieval.precision_at_10 for s in sample_results) / count, 4
        )

        mean_mrr = round(sum(s.retrieval.mrr for s in sample_results) / count, 4)
        mean_ndcg5 = round(
            sum(s.retrieval.ndcg_at_5 for s in sample_results) / count, 4
        )
        mean_ndcg10 = round(
            sum(s.retrieval.ndcg_at_10 for s in sample_results) / count, 4
        )

        mean_cov = round(sum(s.evidence.coverage for s in sample_results) / count, 4)
        mean_cit_prec = round(
            sum(s.evidence.citation_precision for s in sample_results) / count, 4
        )
        mean_unsupp = round(
            sum(s.evidence.unsupported_citation_rate for s in sample_results) / count, 4
        )

        mean_t_prec = round(
            sum(s.answer.token_precision for s in sample_results) / count, 4
        )
        mean_t_rec = round(
            sum(s.answer.token_recall for s in sample_results) / count, 4
        )
        mean_t_f1 = round(sum(s.answer.token_f1 for s in sample_results) / count, 4)
        mean_kw_cov = round(
            sum(s.answer.keyword_coverage for s in sample_results) / count, 4
        )
        mean_ans_corr = round(
            sum(s.answer.overall_correctness for s in sample_results) / count, 4
        )

        mean_lat = round(
            sum(s.efficiency.latency_ms for s in sample_results) / count, 2
        )
        tot_prompt = sum(s.efficiency.prompt_tokens for s in sample_results)
        tot_comp = sum(s.efficiency.completion_tokens for s in sample_results)
        tot_tokens = sum(s.efficiency.total_tokens for s in sample_results)
        tot_cost = round(
            sum(s.efficiency.estimated_cost_usd for s in sample_results), 4
        )

        # Per repository breakdown
        repos = sorted({s.repository for s in sample_results})
        per_repo: dict[str, dict[str, float]] = {}
        for r in repos:
            r_samples = [s for s in sample_results if s.repository == r]
            r_count = len(r_samples)
            if r_count > 0:
                per_repo[r] = {
                    "sample_count": float(r_count),
                    "mean_recall_at_5": round(
                        sum(s.retrieval.recall_at_5 for s in r_samples) / r_count, 4
                    ),
                    "mean_precision_at_5": round(
                        sum(s.retrieval.precision_at_5 for s in r_samples) / r_count, 4
                    ),
                    "mean_mrr": round(
                        sum(s.retrieval.mrr for s in r_samples) / r_count, 4
                    ),
                    "mean_answer_f1": round(
                        sum(s.answer.token_f1 for s in r_samples) / r_count, 4
                    ),
                    "mean_latency_ms": round(
                        sum(s.efficiency.latency_ms for s in r_samples) / r_count, 2
                    ),
                }

        return cls(
            config_name=config_name,
            sample_count=count,
            mean_recall_at_1=mean_r1,
            mean_recall_at_5=mean_r5,
            mean_recall_at_10=mean_r10,
            mean_precision_at_1=mean_p1,
            mean_precision_at_5=mean_p5,
            mean_precision_at_10=mean_p10,
            mean_mrr=mean_mrr,
            mean_ndcg_at_5=mean_ndcg5,
            mean_ndcg_at_10=mean_ndcg10,
            mean_evidence_coverage=mean_cov,
            mean_citation_precision=mean_cit_prec,
            mean_unsupported_citation_rate=mean_unsupp,
            mean_token_precision=mean_t_prec,
            mean_token_recall=mean_t_rec,
            mean_token_f1=mean_t_f1,
            mean_keyword_coverage=mean_kw_cov,
            mean_answer_correctness=mean_ans_corr,
            mean_latency_ms=mean_lat,
            total_prompt_tokens=tot_prompt,
            total_completion_tokens=tot_comp,
            total_tokens=tot_tokens,
            total_estimated_cost_usd=tot_cost,
            per_repo_metrics=per_repo,
            sample_results=list(sample_results),
        )


@dataclass(frozen=True)
class BenchmarkComparisonReport:
    """Comprehensive comparison report across multiple baseline configurations."""

    benchmark_name: str
    version: str
    configurations: list[ConfigurationMetrics]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert comparison report to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "version": self.version,
            "configurations": [c.to_dict() for c in self.configurations],
            "metadata": dict(self.metadata),
        }

    def save_json(self, output_path: str | Path) -> None:
        """Serialize report to formatted JSON file."""
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def to_markdown(self) -> str:
        """Format comparison report as a publication-ready GitHub Flavored Markdown document."""
        lines: list[str] = [
            f"# {self.benchmark_name} (v{self.version})",
            "",
            "## 1. Executive Summary & Configuration Comparison",
            "",
            "| Configuration | Recall@5 | Prec@5 | MRR | nDCG@5 | Answer F1 | Cit Prec | Latency (ms) | Total Cost ($) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]

        for c in self.configurations:
            lines.append(
                f"| **{c.config_name}** | {c.mean_recall_at_5:.4f} | {c.mean_precision_at_5:.4f} | "
                f"{c.mean_mrr:.4f} | {c.mean_ndcg_at_5:.4f} | {c.mean_answer_f1:.4f} | "
                f"{c.mean_citation_precision:.4f} | {c.mean_latency_ms:.1f} | ${c.total_estimated_cost_usd:.4f} |"
            )

        lines.extend(
            [
                "",
                "## 2. Detailed Retrieval & Grounding Breakdown",
                "",
                "| Configuration | R@1 | R@5 | R@10 | P@1 | P@5 | P@10 | Coverage | Keyword Cov |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )

        for c in self.configurations:
            lines.append(
                f"| **{c.config_name}** | {c.mean_recall_at_1:.4f} | {c.mean_recall_at_5:.4f} | "
                f"{c.mean_recall_at_10:.4f} | {c.mean_precision_at_1:.4f} | {c.mean_precision_at_5:.4f} | "
                f"{c.mean_precision_at_10:.4f} | {c.mean_evidence_coverage:.4f} | {c.mean_keyword_coverage:.4f} |"
            )

        lines.extend(
            [
                "",
                "## 3. Computational Efficiency & Resource Consumption",
                "",
                "| Configuration | Latency / Query | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost |",
                "|---|---|---|---|---|---|",
            ]
        )

        for c in self.configurations:
            lines.append(
                f"| **{c.config_name}** | {c.mean_latency_ms:.1f} ms | {c.total_prompt_tokens:,} | "
                f"{c.total_completion_tokens:,} | {c.total_tokens:,} | ${c.total_estimated_cost_usd:.4f} |"
            )

        lines.extend(
            [
                "",
                "## 4. Configuration Taxonomy & Methodology",
                "",
                "- **B0: Naive RAG**: Unstructured fixed-size text chunking with lexical/dense retrieval without AST syntax boundaries or repository context awareness.",
                "- **B1: AST-aware RAG**: Syntax-aware structural extraction of discrete functions, classes, and modules as CodeUnits.",
                "- **B2: Hybrid RAG**: Combined BM25 lexical keyword matching and semantic dense embedding retrieval with Reciprocal Rank Fusion.",
                "- **B3: Graph RAG**: Code dependency graph expansion traversing imports, class hierarchies, caller-callee links, and cross-file references.",
                "- **B4: Agentic RAG**: Multi-step iterative investigative workflow dynamically querying tools, traversing call-sites, and verifying claims.",
                "- **B5: Full RepoLens**: End-to-end pipeline combining agentic search, graph expansion, strict citation verification, and fallback handling for insufficient evidence.",
                "",
                "## 5. Experimental Limitations & Findings",
                "",
                "- **Lexical Metric Trade-offs**: Token F1 and keyword coverage evaluate surface-level lexical similarity against ground-truth answers; semantic parity is not fully captured by token n-gram overlap alone.",
                "- **Deterministic Baseline Staging**: Baselines use reproducible retrieval and response templates to measure architectural component contributions under controlled conditions without live non-deterministic LLM variance.",
                "- **Static Repository Snapshots**: Evaluation is performed on pinned commit snapshots of five major Python repositories (Requests, Flask, Click, FastAPI, Pydantic); results may vary on repositories in other programming languages or unconventional project structures.",
                "- **Latency & Cost Assumptions**: Token and latency metrics represent pipeline runtime and token serialization overhead using standard Gemini/OpenAI pricing models ($0.15/1M input, $0.60/1M output tokens); real-world production costs depend on provider rate limits, network latency, and model tier.",
            ]
        )

        return "\n".join(lines)
