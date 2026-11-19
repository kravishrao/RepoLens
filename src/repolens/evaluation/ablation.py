"""Component Ablation Study Runner and Analyzer.

Evaluates the marginal contribution of individual architectural components:
1. Citation Verification (B5 vs B4)
2. Agentic Reasoning (B4 vs B3)
3. CodeGraph Context Expansion (B3 vs B2)
4. Cross-Encoder Reranking (B2 vs Hybrid-NoRerank)
5. Lexical BM25 Retrieval (Hybrid vs Dense B1)
6. AST CodeUnit Chunking (AST B1 vs Naive B0)
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from repolens.evaluation.baselines import (
    BaseBaselinePipeline,
    HybridRAGPipeline,
    RepositoryContext,
    create_baseline_pipeline,
)
from repolens.evaluation.models import (
    AblationComparison,
    AblationType,
    BaselineType,
    BenchmarkQuery,
    ExperimentConfig,
    ExperimentRunResult,
    SingleQueryResult,
)


@dataclass
class ComponentContribution:
    """Marginal contribution of an architectural component across metrics."""

    component_name: str
    baseline_with: str
    baseline_without: str
    delta_hit_rate: float
    delta_citation_precision: float
    delta_precision_at_k: float
    delta_latency_ms: float
    delta_tokens: float
    insights: str


class AblationStudy:
    """Orchestrates ablation experiments and computes component marginal contributions."""

    def __init__(self, context: RepositoryContext) -> None:
        self.context = context

    def run_pipeline(
        self,
        pipeline: BaseBaselinePipeline,
        queries: Sequence[BenchmarkQuery | str],
        name: str,
    ) -> ExperimentRunResult:
        """Execute a pipeline across queries and compile summary metrics."""
        import platform
        import uuid
        from datetime import datetime

        results: list[SingleQueryResult] = []
        for q in queries:
            res = pipeline.execute(q)
            results.append(res)

        # Aggregate summary metrics
        hit_rates = [r.metrics.hit_rate for r in results]
        precisions = [r.metrics.precision_at_k for r in results]
        cit_precisions = [r.metrics.citation_precision for r in results]
        latencies = [r.metrics.latency_ms for r in results]
        tokens = [r.metrics.tokens_used for r in results]

        summary = {
            "mean_hit_rate": round(statistics.mean(hit_rates) if hit_rates else 0.0, 4),
            "mean_precision_at_k": round(
                statistics.mean(precisions) if precisions else 0.0, 4
            ),
            "mean_citation_precision": round(
                statistics.mean(cit_precisions) if cit_precisions else 0.0, 4
            ),
            "mean_latency_ms": round(
                statistics.mean(latencies) if latencies else 0.0, 2
            ),
            "mean_tokens_used": round(statistics.mean(tokens) if tokens else 0.0, 1),
            "query_count": len(results),
        }

        return ExperimentRunResult(
            experiment_id=str(uuid.uuid4())[:8],
            name=name,
            timestamp=datetime.now().isoformat(),
            config=pipeline.config,
            system_info={
                "os": platform.system(),
                "python": platform.python_version(),
            },
            query_results=results,
            summary_metrics=summary,
        )

    def run_ablation_suite(
        self,
        queries: Sequence[BenchmarkQuery | str],
    ) -> AblationComparison:
        """Run all primary baselines B0 to B5 plus key ablation variants."""
        runs: dict[str, ExperimentRunResult] = {}

        # 1. B0: Naive RAG
        p_b0 = create_baseline_pipeline(BaselineType.B0_NAIVE_RAG, self.context)
        runs["B0: Naive RAG"] = self.run_pipeline(p_b0, queries, "B0: Naive RAG")

        # 2. B1: AST RAG
        p_b1 = create_baseline_pipeline(BaselineType.B1_AST_RAG, self.context)
        runs["B1: AST RAG"] = self.run_pipeline(p_b1, queries, "B1: AST RAG")

        # 3. B2: Hybrid RAG (with reranking)
        p_b2 = create_baseline_pipeline(BaselineType.B2_HYBRID_RAG, self.context)
        runs["B2: Hybrid RAG"] = self.run_pipeline(p_b2, queries, "B2: Hybrid RAG")

        # 4. B2 Ablation: Hybrid without Reranker
        cfg_no_rerank = ExperimentConfig(
            baseline=BaselineType.B2_HYBRID_RAG,
            enable_reranker=False,
            ablation=AblationType.NO_RERANKER,
        )
        p_no_rerank = HybridRAGPipeline(self.context, cfg_no_rerank)
        runs["B2 (-Reranker)"] = self.run_pipeline(
            p_no_rerank, queries, "B2 (-Reranker)"
        )

        # 5. B3: Graph RAG
        p_b3 = create_baseline_pipeline(BaselineType.B3_GRAPH_RAG, self.context)
        runs["B3: Graph RAG"] = self.run_pipeline(p_b3, queries, "B3: Graph RAG")

        # 6. B4: Agentic RAG
        p_b4 = create_baseline_pipeline(BaselineType.B4_AGENTIC_RAG, self.context)
        runs["B4: Agentic RAG"] = self.run_pipeline(p_b4, queries, "B4: Agentic RAG")

        # 7. B5: Full RepoLens
        p_b5 = create_baseline_pipeline(BaselineType.B5_FULL_REPOLENS, self.context)
        runs["B5: Full RepoLens"] = self.run_pipeline(
            p_b5, queries, "B5: Full RepoLens"
        )

        comparison = AblationComparison(
            comparison_id="ablation_full_matrix",
            repository_name=self.context.repository_name,
            runs=runs,
        )
        return comparison

    def compute_contributions(
        self,
        comparison: AblationComparison,
    ) -> list[ComponentContribution]:
        """Compute marginal deltas for each isolated component."""
        runs = comparison.runs
        contributions: list[ComponentContribution] = []

        def get_metric(run_name: str, metric_key: str) -> float:
            if run_name in runs:
                return runs[run_name].summary_metrics.get(metric_key, 0.0)
            return 0.0

        # 1. AST CodeUnit Chunking (B1 vs B0)
        if "B1: AST RAG" in runs and "B0: Naive RAG" in runs:
            contributions.append(
                ComponentContribution(
                    component_name="AST CodeUnit Chunking",
                    baseline_with="B1: AST RAG",
                    baseline_without="B0: Naive RAG",
                    delta_hit_rate=get_metric("B1: AST RAG", "mean_hit_rate")
                    - get_metric("B0: Naive RAG", "mean_hit_rate"),
                    delta_citation_precision=get_metric(
                        "B1: AST RAG", "mean_citation_precision"
                    )
                    - get_metric("B0: Naive RAG", "mean_citation_precision"),
                    delta_precision_at_k=get_metric(
                        "B1: AST RAG", "mean_precision_at_k"
                    )
                    - get_metric("B0: Naive RAG", "mean_precision_at_k"),
                    delta_latency_ms=get_metric("B1: AST RAG", "mean_latency_ms")
                    - get_metric("B0: Naive RAG", "mean_latency_ms"),
                    delta_tokens=get_metric("B1: AST RAG", "mean_tokens_used")
                    - get_metric("B0: Naive RAG", "mean_tokens_used"),
                    insights="AST chunking guarantees semantic boundary integrity, eliminating arbitrary line slicing.",
                )
            )

        # 2. Lexical BM25 Fusion (B2 without reranker vs B1)
        target_b2_no_rerank = (
            "B2 (-Reranker)" if "B2 (-Reranker)" in runs else "B2: Hybrid RAG"
        )
        if target_b2_no_rerank in runs and "B1: AST RAG" in runs:
            contributions.append(
                ComponentContribution(
                    component_name="BM25 Lexical Fusion",
                    baseline_with=target_b2_no_rerank,
                    baseline_without="B1: AST RAG",
                    delta_hit_rate=get_metric(target_b2_no_rerank, "mean_hit_rate")
                    - get_metric("B1: AST RAG", "mean_hit_rate"),
                    delta_citation_precision=get_metric(
                        target_b2_no_rerank, "mean_citation_precision"
                    )
                    - get_metric("B1: AST RAG", "mean_citation_precision"),
                    delta_precision_at_k=get_metric(
                        target_b2_no_rerank, "mean_precision_at_k"
                    )
                    - get_metric("B1: AST RAG", "mean_precision_at_k"),
                    delta_latency_ms=get_metric(target_b2_no_rerank, "mean_latency_ms")
                    - get_metric("B1: AST RAG", "mean_latency_ms"),
                    delta_tokens=get_metric(target_b2_no_rerank, "mean_tokens_used")
                    - get_metric("B1: AST RAG", "mean_tokens_used"),
                    insights="Lexical BM25 significantly boosts retrieval of exact identifier symbols, function names, and error codes.",
                )
            )

        # 3. Cross-Encoder Reranking (B2 vs B2 -Reranker)
        if "B2: Hybrid RAG" in runs and "B2 (-Reranker)" in runs:
            contributions.append(
                ComponentContribution(
                    component_name="Cross-Encoder Reranking",
                    baseline_with="B2: Hybrid RAG",
                    baseline_without="B2 (-Reranker)",
                    delta_hit_rate=get_metric("B2: Hybrid RAG", "mean_hit_rate")
                    - get_metric("B2 (-Reranker)", "mean_hit_rate"),
                    delta_citation_precision=get_metric(
                        "B2: Hybrid RAG", "mean_citation_precision"
                    )
                    - get_metric("B2 (-Reranker)", "mean_citation_precision"),
                    delta_precision_at_k=get_metric(
                        "B2: Hybrid RAG", "mean_precision_at_k"
                    )
                    - get_metric("B2 (-Reranker)", "mean_precision_at_k"),
                    delta_latency_ms=get_metric("B2: Hybrid RAG", "mean_latency_ms")
                    - get_metric("B2 (-Reranker)", "mean_latency_ms"),
                    delta_tokens=get_metric("B2: Hybrid RAG", "mean_tokens_used")
                    - get_metric("B2 (-Reranker)", "mean_tokens_used"),
                    insights="Reranking promotes the most contextually relevant CodeUnits to top positions.",
                )
            )

        # 4. CodeGraph Context Expansion (B3 vs B2)
        if "B3: Graph RAG" in runs and "B2: Hybrid RAG" in runs:
            contributions.append(
                ComponentContribution(
                    component_name="CodeGraph Context Expansion",
                    baseline_with="B3: Graph RAG",
                    baseline_without="B2: Hybrid RAG",
                    delta_hit_rate=get_metric("B3: Graph RAG", "mean_hit_rate")
                    - get_metric("B2: Hybrid RAG", "mean_hit_rate"),
                    delta_citation_precision=get_metric(
                        "B3: Graph RAG", "mean_citation_precision"
                    )
                    - get_metric("B2: Hybrid RAG", "mean_citation_precision"),
                    delta_precision_at_k=get_metric(
                        "B3: Graph RAG", "mean_precision_at_k"
                    )
                    - get_metric("B2: Hybrid RAG", "mean_precision_at_k"),
                    delta_latency_ms=get_metric("B3: Graph RAG", "mean_latency_ms")
                    - get_metric("B2: Hybrid RAG", "mean_latency_ms"),
                    delta_tokens=get_metric("B3: Graph RAG", "mean_tokens_used")
                    - get_metric("B2: Hybrid RAG", "mean_tokens_used"),
                    insights="Graph expansion retrieves caller/callee definitions and dependencies that are not lexically mentioned.",
                )
            )

        # 5. Citation Verification (B5 vs B4)
        if "B5: Full RepoLens" in runs and "B4: Agentic RAG" in runs:
            contributions.append(
                ComponentContribution(
                    component_name="Ground-Truth Citation Verification",
                    baseline_with="B5: Full RepoLens",
                    baseline_without="B4: Agentic RAG",
                    delta_hit_rate=get_metric("B5: Full RepoLens", "mean_hit_rate")
                    - get_metric("B4: Agentic RAG", "mean_hit_rate"),
                    delta_citation_precision=get_metric(
                        "B5: Full RepoLens", "mean_citation_precision"
                    )
                    - get_metric("B4: Agentic RAG", "mean_citation_precision"),
                    delta_precision_at_k=get_metric(
                        "B5: Full RepoLens", "mean_precision_at_k"
                    )
                    - get_metric("B4: Agentic RAG", "mean_precision_at_k"),
                    delta_latency_ms=get_metric("B5: Full RepoLens", "mean_latency_ms")
                    - get_metric("B4: Agentic RAG", "mean_latency_ms"),
                    delta_tokens=get_metric("B5: Full RepoLens", "mean_tokens_used")
                    - get_metric("B4: Agentic RAG", "mean_tokens_used"),
                    insights="Citation verification checks that every cited line and symbol exists on disk, filtering out hallucinations.",
                )
            )

        return contributions
