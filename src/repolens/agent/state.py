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


class InvestigationState(TypedDict, total=False):
    """Shared state maintained throughout an investigation workflow."""

    # Original repository-level question.
    query: str

    # Results of query analysis.
    analyzed_query: str
    query_keywords: list[str]
    investigation_goals: list[str]

    # Tool execution planning.
    selected_tools: list[str]
    executed_tools: list[str]

    # Accumulated investigation evidence.
    evidence: Annotated[
        list[dict[str, object]],
        merge_evidence,
    ]

    # Investigation execution history.
    history: Annotated[
        list[dict[str, object]],
        merge_history,
    ]

    # Tool failures are recorded without crashing the workflow.
    errors: Annotated[
        list[dict[str, str]],
        merge_errors,
    ]

    # Iteration and termination controls.
    iteration: int
    max_iterations: int
    max_evidence: int
    needs_more_investigation: bool
    is_complete: bool

    # Final response.
    final_answer: str
    termination_reason: str
