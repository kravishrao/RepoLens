"""Evaluation and Ablation Experiment Data Models.

Defines configuration schemas, benchmark query structures, baseline types,
metrics containers, and reproducible serialization models for baseline
and ablation experiments.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any


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
