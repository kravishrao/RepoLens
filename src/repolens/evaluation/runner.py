"""Experiment Runner and Reproducible Results Logger.

Provides reproducible execution, structured logging, environment provenance
capture, and artifact export (JSON, Markdown) for baseline and ablation experiments.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
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
from repolens.evaluation.models import (
    AblationComparison,
    BaselineType,
    BenchmarkQuery,
    ExperimentConfig,
    ExperimentRunResult,
    SingleQueryResult,
)


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
