
from __future__ import annotations

from collections import Counter
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from repolens.agent.nodes import create_nodes
from repolens.agent.state import InvestigationState
from repolens.agent.tools import InvestigationTools
from repolens.verification.citation_extractor import extract_citations
from repolens.verification.citation_validator import CitationValidator
from repolens.verification.claim_validator import ClaimValidator


def route_after_investigation(
    state: InvestigationState,
) -> str:
    """Decide whether investigation should continue or finish."""
    if not state.get("needs_more_investigation", False):
        return "answer_preparation"

    if state.get("iteration", 0) >= state.get("max_iterations", 4):
        return "answer_preparation"

    if len(state.get("evidence", [])) >= state.get("max_evidence", 30):
        return "answer_preparation"

    return "tool_selection"


def _get_overall_validation_status(
    citation_results: list,
    claim_results: list,
) -> str:
    """Determine the overall status from citation and claim results."""
    if not citation_results:
        return "incomplete"

    statuses = [
        result.status for result in citation_results
    ] + [
        result.status for result in claim_results
    ]

    if "invalid" in statuses:
        return "invalid"
    if "incomplete" in statuses:
        return "incomplete"
    if "unverified" in statuses:
        return "unverified"

    return "valid"


def _validate_answer(
    state: InvestigationState,
    tools: InvestigationTools,
    repository_root: str | Path | None,
) -> dict[str, object]:
    """Validate generated citations and evidence-backed code findings."""
    if repository_root is None:
        return {
            "citation_validation_results": [],
            "claim_validation_results": [],
            "validation_summary": {
                "enabled": False,
                "overall_status": "disabled",
                "citation_count": 0,
                "claim_count": 0,
                "message": (
                    "Validation was not run because repository_root was not supplied."
                ),
            },
        }

    answer = state.get("final_answer", "")
    evidence = state.get("evidence", [])

    citation_validator = CitationValidator(
        repository_root=repository_root,
        code_graph=tools.code_graph,
    )
    claim_validator = ClaimValidator()

    citations = extract_citations(answer)
    citation_results = citation_validator.validate_many(citations)

    # The current answer format reports code findings rather than arbitrary
    # natural-language claims. Validate only the factual location of each
    # distinct discovered code element.
    findings: dict[str, dict[str, object]] = {}

    for item in evidence:
        unit_id = item.get("code_unit_id")
        if unit_id is None:
            continue

        findings.setdefault(str(unit_id), item)

    claim_results = []

    for item in findings.values():
        qualified_name = str(item.get("qualified_name") or item.get("name") or "")
        file_path = str(item.get("file_path") or "")

        if not qualified_name or not file_path:
            continue

        claim = f"Code element {qualified_name} is located in {file_path}."

        matching_evidence = [
            record
            for record in evidence
            if record.get("code_unit_id") == item.get("code_unit_id")
        ]

        claim_results.append(claim_validator.validate(claim, matching_evidence))

    citation_counts = Counter(result.status for result in citation_results)
    claim_counts = Counter(result.status for result in claim_results)

    overall_status = _get_overall_validation_status(
        citation_results,
        claim_results,
    )

    issues = [issue.to_dict() for result in citation_results for issue in result.issues]
    issues.extend(
        issue.to_dict() for result in claim_results for issue in result.issues
    )

    return {
        "citation_validation_results": [
            result.to_dict() for result in citation_results
        ],
        "claim_validation_results": [result.to_dict() for result in claim_results],
        "validation_summary": {
            "enabled": True,
            "overall_status": overall_status,
            "citation_count": len(citation_results),
            "claim_count": len(claim_results),
            "citation_status_counts": dict(citation_counts),
            "claim_status_counts": dict(claim_counts),
            "issues": issues,
            "claim_validation_scope": (
                "Lexical evidence checks for code-element location findings; "
                "not semantic entailment verification."
            ),
        },
    }


def build_investigation_workflow(
    tools: InvestigationTools,
    *,
    repository_root: str | Path | None = None,
):
    """Build and compile the repository investigation workflow."""
    nodes = create_nodes(tools)

    workflow = StateGraph(InvestigationState)

    workflow.add_node("query_analysis", nodes["query_analysis"])
    workflow.add_node("tool_selection", nodes["tool_selection"])
    workflow.add_node("investigation", nodes["investigation"])
    workflow.add_node("answer_preparation", nodes["answer_preparation"])

    workflow.add_node(
        "citation_validation",
        lambda state: _validate_answer(state, tools, repository_root),
    )

    workflow.add_edge(START, "query_analysis")
    workflow.add_edge("query_analysis", "tool_selection")
    workflow.add_edge("tool_selection", "investigation")

    workflow.add_conditional_edges(
        "investigation",
        route_after_investigation,
        {
            "tool_selection": "tool_selection",
            "answer_preparation": "answer_preparation",
        },
    )

    workflow.add_edge("answer_preparation", "citation_validation")
    workflow.add_edge("citation_validation", END)

    return workflow.compile()


def investigate_repository(
    query: str,
    tools: InvestigationTools,
    *,
    max_iterations: int = 4,
    max_evidence: int = 30,
    min_evidence: int = 0,
    repository_root: str | Path | None = None,
) -> InvestigationState:
    """Run an investigation and optionally validate its source citations."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Investigation query cannot be empty.")

    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer.")

    if (
        not isinstance(max_evidence, int)
        or isinstance(max_evidence, bool)
        or max_evidence < 1
    ):
        raise ValueError("max_evidence must be a positive integer.")

    if (
        not isinstance(min_evidence, int)
        or isinstance(min_evidence, bool)
        or min_evidence < 0
    ):
        raise ValueError("min_evidence must be a non-negative integer.")

    if min_evidence > max_evidence:
        raise ValueError("min_evidence cannot exceed max_evidence.")

    app = build_investigation_workflow(
        tools,
        repository_root=repository_root,
    )

    initial_state: InvestigationState = {
        "query": query,
        "evidence": [],
        "errors": [],
        "history": [],
        "trace": [],
        "executed_tools": [],
        "selected_tools": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "max_evidence": max_evidence,
        "min_evidence": min_evidence,
        "needs_more_investigation": True,
        "is_complete": False,
    }

    return app.invoke(initial_state)
