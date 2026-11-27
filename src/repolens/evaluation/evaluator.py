"""RepoLens evaluation orchestrator and benchmark comparative runner."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from repolens.evaluation.baselines import (
    BasePipeline,
    PipelineExecutionResult,
    get_baseline_pipeline,
)
from repolens.evaluation.models import (
    BenchmarkComparisonReport,
    ConfigurationMetrics,
    PricingConfig,
    SampleEvaluationResult,
)
from repolens.evaluation.qa_dataset import QAEvaluationDataset, QuestionSample

logger = logging.getLogger(__name__)


class RepoLensEvaluator:
    """Benchmark evaluation orchestrator measuring retrieval, answer, evidence, and efficiency."""

    def __init__(
        self,
        pricing: PricingConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.pricing = pricing or PricingConfig()
        self.seed = seed

    def evaluate_sample(
        self,
        sample: QuestionSample,
        pipeline: BasePipeline | Callable[[QuestionSample], PipelineExecutionResult],
        repos_root: Path | None = None,
    ) -> SampleEvaluationResult:
        """Run a pipeline on a single question sample and evaluate metrics."""
        if isinstance(pipeline, BasePipeline):
            result = pipeline.execute(sample, repos_root=repos_root)
            config_name = pipeline.config_name
        else:
            result = pipeline(sample)
            config_name = getattr(result, "config_name", "custom_pipeline")

        return SampleEvaluationResult.evaluate(
            sample=sample,
            config_name=config_name,
            retrieved_items=result.retrieved_items,
            generated_answer=result.generated_answer,
            citations=result.citations,
            latency_ms=result.latency_ms,
            pricing=self.pricing,
            is_grounded=result.is_grounded,
            evidence_status=result.evidence_status,
        )

    def evaluate_configuration(
        self,
        pipeline: BasePipeline
        | str
        | Callable[[QuestionSample], PipelineExecutionResult],
        dataset: QAEvaluationDataset | Sequence[QuestionSample],
        limit: int | None = None,
        repos_root: Path | None = None,
    ) -> ConfigurationMetrics:
        """Evaluate a pipeline configuration across all samples in the dataset."""
        if isinstance(pipeline, str):
            pipeline_inst = get_baseline_pipeline(pipeline, seed=self.seed)
            config_name = pipeline_inst.config_name
        elif isinstance(pipeline, BasePipeline):
            pipeline_inst = pipeline
            config_name = pipeline.config_name
        else:
            pipeline_inst = pipeline
            config_name = "custom_pipeline"

        questions: Sequence[QuestionSample] = (
            dataset.questions if isinstance(dataset, QAEvaluationDataset) else dataset
        )

        if limit is not None and limit > 0:
            questions = questions[:limit]

        sample_results: list[SampleEvaluationResult] = []
        for sample in questions:
            res = self.evaluate_sample(
                sample=sample,
                pipeline=pipeline_inst,
                repos_root=repos_root,
            )
            sample_results.append(res)

        return ConfigurationMetrics.aggregate(config_name, sample_results)

    def compare_configurations(
        self,
        configs: Sequence[str | BasePipeline],
        dataset: QAEvaluationDataset | Sequence[QuestionSample],
        limit: int | None = None,
        repos_root: Path | None = None,
        benchmark_name: str = "RepoLens Multi-Configuration Evaluation Benchmark",
        version: str = "1.0.0",
    ) -> BenchmarkComparisonReport:
        """Execute and compare multiple baseline configurations (e.g. B0 to B5)."""
        metrics_list: list[ConfigurationMetrics] = []

        for cfg in configs:
            m = self.evaluate_configuration(
                pipeline=cfg,
                dataset=dataset,
                limit=limit,
                repos_root=repos_root,
            )
            metrics_list.append(m)

        return BenchmarkComparisonReport(
            benchmark_name=benchmark_name,
            version=version,
            configurations=metrics_list,
            metadata={
                "seed": self.seed,
                "sample_limit": limit,
                "pricing_model": self.pricing.model_name,
                "total_configs": len(metrics_list),
            },
        )
