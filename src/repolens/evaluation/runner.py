"""Experiment Runner and Reproducible Results Logger.

Provides reproducible execution, structured logging, environment provenance
capture, and artifact export (JSON, Markdown) for baseline and ablation experiments.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from repolens.evaluation.ablation import AblationStudy
from repolens.evaluation.baselines import (
    RepositoryContext,
    create_baseline_pipeline,
)
from repolens.evaluation.benchmark_suite import (
    get_benchmark_queries,
    prepare_benchmark_context,
)
from repolens.evaluation.documentation import get_baseline_limitations
from repolens.evaluation.evaluator import RepoLensEvaluator
from repolens.evaluation.models import (
    AblationComparison,
    BaselineType,
    BenchmarkComparisonReport,
    BenchmarkQuery,
    ExperimentConfig,
    ExperimentRunResult,
    PricingConfig,
    SingleQueryResult,
)
from repolens.evaluation.qa_dataset import load_qa_dataset

logger = logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """Safely obtain current git commit hash if in a git repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_system_provenance(seed: int | None = 42) -> dict[str, Any]:
    """Capture environment and system metadata for reproducibility."""
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "git_commit": get_git_commit_hash(),
        "pid": os.getpid(),
        "random_seed": seed,
    }


class ExperimentRunner:
    """Manages experiment runs, evaluations, and reproducible result logging."""

    def __init__(
        self,
        context: RepositoryContext | None = None,
        output_dir: str | Path = "experiments/results",
    ) -> None:
        self.context = context
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_experiment(
        self,
        config: ExperimentConfig,
        queries: Sequence[BenchmarkQuery | str] | None = None,
        experiment_name: str | None = None,
    ) -> ExperimentRunResult:
        """Run an evaluation experiment under a specific configuration."""
        # 1. Resolve repository context
        context = self.context
        if context is None:
            context = prepare_benchmark_context(
                config.repository_path or config.repository_name,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )

        # 2. Resolve evaluation queries
        eval_queries = (
            list(queries)
            if queries is not None
            else get_benchmark_queries(config.repository_name)
        )
        if not eval_queries:
            eval_queries = get_benchmark_queries("RepoLens")

        # 3. Instantiate pipeline
        pipeline = create_baseline_pipeline(config.baseline, context, config)

        # 4. Execute queries
        run_id = f"exp_{str(uuid.uuid4())[:8]}"
        name = experiment_name or f"{config.baseline} on {config.repository_name}"
        timestamp = datetime.now(UTC).isoformat()

        query_results: list[SingleQueryResult] = []
        for q in eval_queries:
            res = pipeline.execute(q)
            query_results.append(res)

        # 5. Compute summary metrics
        import statistics

        hit_rates = [r.metrics.hit_rate for r in query_results]
        precisions = [r.metrics.precision_at_k for r in query_results]
        cit_precisions = [r.metrics.citation_precision for r in query_results]
        latencies = [r.metrics.latency_ms for r in query_results]
        tokens = [r.metrics.tokens_used for r in query_results]

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
            "total_queries": len(query_results),
        }

        # 6. Retrieve documented limitations
        b_type_str = (
            config.baseline.value
            if isinstance(config.baseline, BaselineType)
            else str(config.baseline)
        )
        limitations = get_baseline_limitations(b_type_str)

        run_result = ExperimentRunResult(
            experiment_id=run_id,
            name=name,
            timestamp=timestamp,
            config=config,
            system_info=get_system_provenance(config.random_seed),
            query_results=query_results,
            summary_metrics=summary,
            limitations_notes=limitations,
        )

        return run_result

    def run_all_baselines(
        self,
        repository_name: str = "RepoLens",
        queries: Sequence[BenchmarkQuery | str] | None = None,
    ) -> AblationComparison:
        """Run all baselines B0 through B5 on the specified repository."""
        context = self.context
        if context is None:
            context = prepare_benchmark_context(repository_name)

        study = AblationStudy(context)
        eval_queries = (
            list(queries)
            if queries is not None
            else get_benchmark_queries(repository_name)
        )
        if not eval_queries:
            eval_queries = get_benchmark_queries("RepoLens")

        comparison = study.run_ablation_suite(eval_queries)
        return comparison

    def save_run_result(
        self,
        result: ExperimentRunResult,
        filename: str | None = None,
    ) -> Path:
        """Persist experiment result as reproducible JSON on disk."""
        fname = (
            filename
            or f"{result.experiment_id}_{result.config.baseline.replace(' ', '_').replace(':', '')}.json"
        )
        target_path = self.output_dir / fname
        target_path.write_text(result.to_json(), encoding="utf-8")
        return target_path

    def save_comparison(
        self,
        comparison: AblationComparison,
        filename: str | None = None,
    ) -> Path:
        """Persist comparison matrix as markdown table and JSON."""
        fname = (
            filename
            or f"comparison_{comparison.comparison_id}_{comparison.repository_name}.json"
        )
        target_path = self.output_dir / fname
        data = {
            "comparison_id": comparison.comparison_id,
            "repository": comparison.repository_name,
            "runs": {name: run.to_dict() for name, run in comparison.runs.items()},
            "markdown_table": comparison.to_markdown_table(),
        }
        target_path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )
        return target_path

    @staticmethod
    def load_run_result(file_path: str | Path) -> dict[str, Any]:
        """Load a previously saved experiment result JSON."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Result file '{file_path}' does not exist.")
        return json.loads(p.read_text(encoding="utf-8"))


# =============================================================================
# QA Dataset Evaluation Runner & CLI Entry Point
# =============================================================================


def run_evaluation(
    configs: Sequence[str] = ("B0", "B1", "B2", "B3", "B4", "B5"),
    dataset_path: str | Path | None = None,
    limit: int | None = None,
    repos: Sequence[str] | None = None,
    seed: int = 42,
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
    pricing: PricingConfig | None = None,
) -> BenchmarkComparisonReport:
    """Execute benchmark evaluation across selected configurations and export reports."""
    # 1. Load dataset
    dataset = load_qa_dataset(dataset_path)

    # 2. Filter by repository if requested
    questions = dataset.questions
    if repos:
        target_repos = {r.strip().lower() for r in repos}
        questions = [q for q in questions if q.repository.lower() in target_repos]

    evaluator = RepoLensEvaluator(pricing=pricing, seed=seed)
    report = evaluator.compare_configurations(
        configs=configs,
        dataset=questions,
        limit=limit,
    )

    # 3. Export reports
    if output_json:
        report.save_json(output_json)
        print(f"Saved evaluation JSON report to: {output_json}")

    if output_markdown:
        out_p = Path(output_markdown).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(report.to_markdown(), encoding="utf-8")
        print(f"Saved evaluation Markdown report to: {output_markdown}")

    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for repolens evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Run reproducible RepoLens evaluations across baseline configurations B0-B5."
    )
    parser.add_argument(
        "--configs",
        "-c",
        nargs="+",
        default=["B0", "B1", "B2", "B3", "B4", "B5"],
        help="Configurations to evaluate (e.g. B0 B1 B2 B3 B4 B5).",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of questions to evaluate (for quick test runs).",
    )
    parser.add_argument(
        "--repos",
        "-r",
        nargs="+",
        default=None,
        help="Filter evaluation questions to specific repositories.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default=None,
        help="Path to evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--output-json",
        "-o",
        default=None,
        help="Path to output structured JSON evaluation report.",
    )
    parser.add_argument(
        "--output-markdown",
        "-m",
        default=None,
        help="Path to output formatted Markdown comparison table.",
    )

    args = parser.parse_args(argv)

    report = run_evaluation(
        configs=args.configs,
        dataset_path=args.dataset,
        limit=args.limit,
        repos=args.repos,
        seed=args.seed,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )

    print("\n" + report.to_markdown())
    return 0


if __name__ == "__main__":
    sys.exit(main())
