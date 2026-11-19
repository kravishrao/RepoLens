"""Command-line runner and entry point for reproducible RepoLens evaluations."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from repolens.evaluation.evaluator import RepoLensEvaluator
from repolens.evaluation.models import BenchmarkComparisonReport, PricingConfig
from repolens.evaluation.qa_dataset import load_qa_dataset

logger = logging.getLogger(__name__)


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
