from __future__ import annotations

from typing import Annotated, TypedDict


def merge_evidence(
    existing: list[dict[str, object]],
    new: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Merge evidence records while preserving unique evidence IDs."""
    merged = list(existing)
    known_ids = {item.get("evidence_id") for item in existing}

    for item in new:
        evidence_id = item.get("evidence_id")
        if evidence_id not in known_ids:
            merged.append(item)
            known_ids.add(evidence_id)

    return merged


def merge_errors(
    existing: list[dict[str, str]],
    new: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Append newly recorded tool errors."""
    return [*existing, *new]


def merge_history(
    existing: list[dict[str, object]],
    new: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Append investigation execution history."""
    return [*existing, *new]


def merge_trace(
    existing: list[dict[str, object]],
    new: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Append investigation execution trace."""
    return [*existing, *new]


class InvestigationState(TypedDict, total=False):
    """Shared state maintained throughout an investigation workflow."""

    query: str
    analyzed_query: str
    query_keywords: list[str]
    investigation_goals: list[str]
    requires_graph_traversal: bool
    requires_source_inspection: bool

    selected_tools: list[str]
    executed_tools: list[str]

    evidence: Annotated[list[dict[str, object]], merge_evidence]
    history: Annotated[list[dict[str, object]], merge_history]
    trace: Annotated[list[dict[str, object]], merge_trace]
    errors: Annotated[list[dict[str, str]], merge_errors]

    iteration: int
    max_iterations: int
    max_evidence: int
    min_evidence: int
    needs_more_investigation: bool
    is_complete: bool

    final_answer: str
    termination_reason: str

    # RL-32: structured source and claim validation output.
    citation_validation_results: list[dict[str, object]]
    claim_validation_results: list[dict[str, object]]
    validation_summary: dict[str, object]
