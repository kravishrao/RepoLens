"""Data models, report schemas, and view models for RepoLens evaluation benchmark."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
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


@dataclass(frozen=True)
class PricingConfig:
    """Pricing configuration for LLM inference cost estimation."""

    model_name: str = "default-gemini-flash"
    input_cost_per_million: float = 0.15
    output_cost_per_million: float = 0.60

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate total cost in USD for the given token volumes."""
        return estimate_inference_cost(
            prompt_tokens,
            completion_tokens,
            self.input_cost_per_million,
            self.output_cost_per_million,
        )


@dataclass(frozen=True)
class RetrievalMetrics:
    """Aggregated information retrieval quality metrics."""

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
    """Metrics assessing evidence coverage and citation reliability."""

    coverage: float
    citation_precision: float
    unsupported_citation_rate: float

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class AnswerMetrics:
    """Metrics assessing synthesized answer correctness and grounding."""

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
    """Computational efficiency, latency, token usage, and cost metrics."""

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
    """Detailed evaluation outcome for an individual question sample."""

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
    evidence_status: str = "sufficient"

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
        evidence_status: str = "sufficient",
    ) -> SampleEvaluationResult:
        """Evaluate a single question execution against its ground-truth gold evidence."""
        pricing_cfg = pricing or PricingConfig()
        gold = sample.gold_evidence

        # Compute retrieval metrics
        r1 = compute_recall_at_k(retrieved_items, gold, k=1)
        r5 = compute_recall_at_k(retrieved_items, gold, k=5)
        r10 = compute_recall_at_k(retrieved_items, gold, k=10)
        p1 = compute_precision_at_k(retrieved_items, gold, k=1)
        p5 = compute_precision_at_k(retrieved_items, gold, k=5)
        p10 = compute_precision_at_k(retrieved_items, gold, k=10)
        mrr = compute_mrr(retrieved_items, gold, k=10)
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
    total_samples: int
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
    mean_answer_f1: float
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
        return {
            "config_name": self.config_name,
            "total_samples": self.total_samples,
            "mean_recall_at_1": self.mean_recall_at_1,
            "mean_recall_at_5": self.mean_recall_at_5,
            "mean_recall_at_10": self.mean_recall_at_10,
            "mean_precision_at_1": self.mean_precision_at_1,
            "mean_precision_at_5": self.mean_precision_at_5,
            "mean_precision_at_10": self.mean_precision_at_10,
            "mean_mrr": self.mean_mrr,
            "mean_ndcg_at_5": self.mean_ndcg_at_5,
            "mean_ndcg_at_10": self.mean_ndcg_at_10,
            "mean_evidence_coverage": self.mean_evidence_coverage,
            "mean_citation_precision": self.mean_citation_precision,
            "mean_answer_f1": self.mean_answer_f1,
            "mean_keyword_coverage": self.mean_keyword_coverage,
            "mean_answer_correctness": self.mean_answer_correctness,
            "mean_latency_ms": self.mean_latency_ms,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_estimated_cost_usd": self.total_estimated_cost_usd,
            "per_repo_metrics": self.per_repo_metrics,
            "sample_results": [s.to_dict() for s in self.sample_results],
        }

    @classmethod
    def aggregate(
        cls,
        config_name: str,
        sample_results: Sequence[SampleEvaluationResult],
    ) -> ConfigurationMetrics:
        """Aggregate per-sample results into summary configuration metrics."""
        n = len(sample_results)
        if n == 0:
            return cls(
                config_name=config_name,
                total_samples=0,
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
                mean_answer_f1=0.0,
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

        mr1 = round(sum(s.retrieval.recall_at_1 for s in sample_results) / n, 4)
        mr5 = round(sum(s.retrieval.recall_at_5 for s in sample_results) / n, 4)
        mr10 = round(sum(s.retrieval.recall_at_10 for s in sample_results) / n, 4)
        mp1 = round(sum(s.retrieval.precision_at_1 for s in sample_results) / n, 4)
        mp5 = round(sum(s.retrieval.precision_at_5 for s in sample_results) / n, 4)
        mp10 = round(sum(s.retrieval.precision_at_10 for s in sample_results) / n, 4)
        mmrr = round(sum(s.retrieval.mrr for s in sample_results) / n, 4)
        mndcg5 = round(sum(s.retrieval.ndcg_at_5 for s in sample_results) / n, 4)
        mndcg10 = round(sum(s.retrieval.ndcg_at_10 for s in sample_results) / n, 4)

        mcov = round(sum(s.evidence.coverage for s in sample_results) / n, 4)
        mcp = round(sum(s.evidence.citation_precision for s in sample_results) / n, 4)

        mf1 = round(sum(s.answer.token_f1 for s in sample_results) / n, 4)
        mkw = round(sum(s.answer.keyword_coverage for s in sample_results) / n, 4)
        mcorr = round(sum(s.answer.overall_correctness for s in sample_results) / n, 4)

        mlat = round(sum(s.efficiency.latency_ms for s in sample_results) / n, 2)
        tot_pt = sum(s.efficiency.prompt_tokens for s in sample_results)
        tot_ct = sum(s.efficiency.completion_tokens for s in sample_results)
        tot_tokens = tot_pt + tot_ct
        tot_cost = round(
            sum(s.efficiency.estimated_cost_usd for s in sample_results), 6
        )

        # Per repository breakdown
        by_repo: dict[str, list[SampleEvaluationResult]] = {}
        for s in sample_results:
            by_repo.setdefault(s.repository, []).append(s)

        per_repo: dict[str, dict[str, float]] = {}
        for repo_name, r_samples in by_repo.items():
            rn = len(r_samples)
            per_repo[repo_name] = {
                "samples": float(rn),
                "recall_at_5": round(
                    sum(x.retrieval.recall_at_5 for x in r_samples) / rn, 4
                ),
                "mrr": round(sum(x.retrieval.mrr for x in r_samples) / rn, 4),
                "ndcg_at_5": round(
                    sum(x.retrieval.ndcg_at_5 for x in r_samples) / rn, 4
                ),
                "answer_f1": round(sum(x.answer.token_f1 for x in r_samples) / rn, 4),
                "citation_precision": round(
                    sum(x.evidence.citation_precision for x in r_samples) / rn, 4
                ),
            }

        return cls(
            config_name=config_name,
            total_samples=n,
            mean_recall_at_1=mr1,
            mean_recall_at_5=mr5,
            mean_recall_at_10=mr10,
            mean_precision_at_1=mp1,
            mean_precision_at_5=mp5,
            mean_precision_at_10=mp10,
            mean_mrr=mmrr,
            mean_ndcg_at_5=mndcg5,
            mean_ndcg_at_10=mndcg10,
            mean_evidence_coverage=mcov,
            mean_citation_precision=mcp,
            mean_answer_f1=mf1,
            mean_keyword_coverage=mkw,
            mean_answer_correctness=mcorr,
            mean_latency_ms=mlat,
            total_prompt_tokens=tot_pt,
            total_completion_tokens=tot_ct,
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
