from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any
from uuid import uuid4

from repolens.agent.state import InvestigationState
from repolens.agent.tools import InvestigationTools
from repolens.retrieval.bm25_retriever import tokenize

_GRAPH_TERMS = {
    "call",
    "calls",
    "caller",
    "callers",
    "callee",
    "callees",
    "dependency",
    "dependencies",
    "depend",
    "depends",
    "import",
    "imports",
    "relationship",
    "relationships",
    "uses",
    "used",
    "invoke",
    "invokes",
}

_SOURCE_TERMS = {
    "explain",
    "implementation",
    "implement",
    "source",
    "code",
    "logic",
    "functionality",
    "works",
    "work",
}


def analyze_query(state: InvestigationState) -> dict[str, Any]:
    """Analyze a repository question and identify investigation intent."""
    query = state.get("query", "").strip()

    if not query:
        raise ValueError("Investigation query cannot be empty.")

    keywords = list(dict.fromkeys(tokenize(query)))
    keyword_set = set(keywords)

    graph_intent = bool(keyword_set & _GRAPH_TERMS)
    source_intent = bool(keyword_set & _SOURCE_TERMS)

    goals = ["Identify relevant repository code elements."]

    if graph_intent:
        goals.append("Investigate code relationships and dependencies.")
    else:
        goals.append("Discover relevant callers, callees and dependencies.")

    if source_intent:
        goals.append("Inspect implementation details and source code.")
    else:
        goals.append("Inspect source code supporting the investigation.")

    return {
        "analyzed_query": query,
        "query_keywords": keywords,
        "investigation_goals": goals,
        "requires_graph_traversal": graph_intent,
        "requires_source_inspection": source_intent,
        "iteration": 0,
        "max_iterations": state.get("max_iterations", 4),
        "max_evidence": state.get("max_evidence", 30),
        "min_evidence": state.get("min_evidence", 0),
        "needs_more_investigation": True,
        "is_complete": False,
        "termination_reason": "",
    }


def select_tools(
    state: InvestigationState,
    tools: InvestigationTools,
) -> dict[str, Any]:
    """Select tools according to the current investigation phase."""
    executed = set(state.get("executed_tools", []))
    evidence = state.get("evidence", [])
    max_evidence = state.get("max_evidence", 30)
    min_evidence = state.get("min_evidence", 0)

    if len(evidence) >= max_evidence:
        return {"selected_tools": []}

    retrieval_tools = ["keyword_search"]

    if tools.semantic_available:
        retrieval_tools.append("semantic_search")

    pending_retrieval = [name for name in retrieval_tools if name not in executed]

    if pending_retrieval:
        return {"selected_tools": pending_retrieval}

    if not evidence:
        return {"selected_tools": []}

    # With a configured evidence threshold, only perform additional
    # investigation phases explicitly requested by the query.
    if min_evidence > 0 and len(evidence) >= min_evidence:
        if (
            state.get("requires_graph_traversal", False)
            and "graph_traversal" not in executed
        ):
            return {"selected_tools": ["graph_traversal"]}

        if (
            state.get("requires_source_inspection", False)
            and "inspect_source" not in executed
        ):
            return {"selected_tools": ["inspect_source"]}

        return {"selected_tools": []}

    # Preserve the original full investigation behaviour when no
    # minimum evidence threshold has been configured or reached.
    if "graph_traversal" not in executed:
        return {"selected_tools": ["graph_traversal"]}

    if "inspect_source" not in executed:
        return {"selected_tools": ["inspect_source"]}

    return {"selected_tools": []}


def investigate(
    state: InvestigationState,
    tools: InvestigationTools,
) -> dict[str, Any]:
    """Execute selected tools and maintain a detailed investigation trace."""
    query = state.get("analyzed_query", state.get("query", ""))
    selected = state.get("selected_tools", [])
    evidence = state.get("evidence", [])
    executed = list(state.get("executed_tools", []))

    new_evidence: list[dict[str, object]] = []
    new_errors: list[dict[str, str]] = []
    new_history: list[dict[str, object]] = []
    new_trace: list[dict[str, object]] = []

    # Keep the evidence returned by each tool so the trace can distinguish
    # raw results from evidence actually retained after deduplication/capping.
    tool_result_groups: list[tuple[int, list[dict[str, object]]]] = []

    iteration = state.get("iteration", 0) + 1

    known_unit_ids = list(
        dict.fromkeys(
            str(item["code_unit_id"]) for item in evidence if item.get("code_unit_id")
        )
    )

    for tool_name in selected:
        invocation_id = str(uuid4())
        started_at = perf_counter()

        try:
            if tool_name == "keyword_search":
                results = tools.keyword_search(query)
            elif tool_name == "semantic_search":
                results = tools.semantic_search(query)
            elif tool_name == "graph_traversal":
                results = tools.graph_traversal(
                    known_unit_ids,
                    max_depth=2,
                )
            elif tool_name == "inspect_source":
                results = tools.inspect_source(known_unit_ids)
            else:
                raise ValueError(f"Unknown investigation tool: {tool_name}")

            results = list(results)
            new_evidence.extend(results)

            result_ids = [str(item.get("evidence_id", "unknown")) for item in results]

            new_history.append(
                {
                    "iteration": iteration,
                    "tool": tool_name,
                    "status": "success",
                    "results": len(results),
                }
            )

            trace_index = len(new_trace)
            tool_result_groups.append((trace_index, results))

            new_trace.append(
                {
                    "invocation_id": invocation_id,
                    "iteration": iteration,
                    "tool": tool_name,
                    "status": "success",
                    "duration_ms": round(
                        (perf_counter() - started_at) * 1000,
                        3,
                    ),
                    "result_count": len(results),
                    "findings": result_ids,
                    "retained_count": 0,
                    "error": None,
                }
            )

        except Exception as exc:
            error_message = str(exc)

            new_errors.append(
                {
                    "tool": tool_name,
                    "error": error_message,
                    "iteration": str(iteration),
                }
            )

            new_history.append(
                {
                    "iteration": iteration,
                    "tool": tool_name,
                    "status": "failed",
                    "results": 0,
                }
            )

            new_trace.append(
                {
                    "invocation_id": invocation_id,
                    "iteration": iteration,
                    "tool": tool_name,
                    "status": "failed",
                    "duration_ms": round(
                        (perf_counter() - started_at) * 1000,
                        3,
                    ),
                    "result_count": 0,
                    "findings": [],
                    "retained_count": 0,
                    "error": error_message,
                }
            )

        finally:
            # Failed tools are also marked executed to prevent repeated calls.
            if tool_name not in executed:
                executed.append(tool_name)

    existing_ids = {item.get("evidence_id") for item in evidence}
    seen_ids = set(existing_ids)
    unique_new_evidence: list[dict[str, object]] = []

    for item in new_evidence:
        evidence_id = item.get("evidence_id")

        if evidence_id in seen_ids:
            continue

        unique_new_evidence.append(item)
        seen_ids.add(evidence_id)

    max_evidence = state.get("max_evidence", 30)
    remaining_capacity = max(0, max_evidence - len(evidence))
    unique_new_evidence = unique_new_evidence[:remaining_capacity]

    # Allocate retained findings to their originating tool invocation.
    retained_ids = {item.get("evidence_id") for item in unique_new_evidence}

    for trace_index, results in tool_result_groups:
        retained_findings: list[str] = []

        for item in results:
            evidence_id = item.get("evidence_id")

            if evidence_id in retained_ids:
                retained_findings.append(str(evidence_id))
                retained_ids.remove(evidence_id)

        new_trace[trace_index]["findings"] = retained_findings
        new_trace[trace_index]["retained_count"] = len(retained_findings)

    evidence_count = len(evidence) + len(unique_new_evidence)
    min_evidence = state.get("min_evidence", 0)

    retrieval_tools = {"keyword_search"}

    if tools.semantic_available:
        retrieval_tools.add("semantic_search")

    retrieval_finished = retrieval_tools.issubset(set(executed))

    graph_required = state.get("requires_graph_traversal", False)
    source_required = state.get("requires_source_inspection", False)

    required_phases_finished = (
        not graph_required or "graph_traversal" in executed
    ) and (not source_required or "inspect_source" in executed)

    termination_reason = ""
    needs_more = True

    if evidence_count >= max_evidence:
        termination_reason = "maximum_evidence"
        needs_more = False

    elif (
        min_evidence > 0
        and evidence_count >= min_evidence
        and retrieval_finished
        and required_phases_finished
    ):
        termination_reason = "sufficient_evidence"
        needs_more = False

    elif evidence_count == 0 and retrieval_finished:
        retrieval_history = [
            *state.get("history", []),
            *new_history,
        ]

        successful_retrieval = any(
            record.get("tool") in retrieval_tools and record.get("status") == "success"
            for record in retrieval_history
        )

        termination_reason = "no_evidence" if successful_retrieval else "tool_failures"
        needs_more = False

    elif iteration >= state.get("max_iterations", 4):
        termination_reason = "maximum_iterations"
        needs_more = False

    elif (
        retrieval_finished
        and "graph_traversal" in executed
        and "inspect_source" in executed
    ):
        termination_reason = (
            "investigation_completed" if evidence_count > 0 else "no_evidence"
        )
        needs_more = False

    return {
        "evidence": unique_new_evidence,
        "errors": new_errors,
        "history": new_history,
        "trace": new_trace,
        "executed_tools": executed,
        "iteration": iteration,
        "needs_more_investigation": needs_more,
        "termination_reason": termination_reason,
    }


def _append_execution_trace(
    lines: list[str],
    trace: list[dict[str, object]],
) -> None:
    """Append execution trace details to answer lines."""
    if not trace:
        return

    lines.extend(
        [
            "",
            "Execution Trace",
            "---------------",
        ]
    )

    for record in trace:
        lines.append(
            f"- Iteration {record.get('iteration')}: "
            f"{record.get('tool')} — {record.get('status')} "
            f"({record.get('result_count', 0)} raw results, "
            f"{record.get('retained_count', 0)} retained, "
            f"{record.get('duration_ms', 0)} ms)"
        )


def prepare_answer(state: InvestigationState) -> dict[str, Any]:
    """Prepare a structured answer grounded in repository evidence."""
    evidence = state.get("evidence", [])
    errors = state.get("errors", [])
    trace = state.get("trace", [])

    if not evidence:
        lines = ["No relevant code evidence was found for this repository question."]

        if errors:
            lines.extend(
                [
                    "",
                    "Investigation errors",
                    "--------------------",
                ]
            )

            for error in errors:
                lines.append(
                    f"- {error.get('tool', 'unknown')}: "
                    f"{error.get('error', 'Unknown error')}"
                )

        _append_execution_trace(lines, trace)

        return {
            "final_answer": "\n".join(lines),
            "is_complete": True,
            "needs_more_investigation": False,
            "termination_reason": (state.get("termination_reason") or "no_evidence"),
        }

    grouped: dict[str, list[dict[str, object]]] = {}

    for item in evidence:
        unit_id = str(item.get("code_unit_id", "unknown"))
        grouped.setdefault(unit_id, []).append(item)

    lines = [
        "Repository Investigation",
        "========================",
        "",
        f"Question: {state.get('query', '')}",
        "",
        f"Unique code elements investigated: {len(grouped)}",
        f"Total evidence records: {len(evidence)}",
        "",
        "Findings",
        "--------",
    ]

    for index, records in enumerate(grouped.values(), start=1):
        primary = records[0]

        name = primary.get(
            "qualified_name",
            primary.get("name", "Unknown"),
        )

        lines.extend(
            [
                "",
                f"{index}. {name}",
                f"   Type: {primary.get('kind', 'unknown')}",
                f"   File: {primary.get('file_path', 'unknown')}",
                (
                    "   Lines: "
                    f"{primary.get('start_line', '?')}-"
                    f"{primary.get('end_line', '?')}"
                ),
            ]
        )

        sources = list(
            dict.fromkeys(str(record.get("source", "unknown")) for record in records)
        )

        lines.append(f"   Evidence sources: {', '.join(sources)}")

        contents = list(
            dict.fromkeys(
                str(record.get("content", "")).strip()
                for record in records
                if record.get("content")
            )
        )

        for content in contents[:1]:
            lines.append("   Source evidence:")

            for source_line in content.splitlines()[:20]:
                lines.append(f"      {source_line}")

        for record in records:
            metadata = record.get("metadata", {})

            if isinstance(metadata, dict):
                docstring = metadata.get("docstring")

                if docstring:
                    lines.append(f"   Docstring: {docstring}")
                    break

    if errors:
        lines.extend(
            [
                "",
                "Investigation warnings",
                "----------------------",
            ]
        )

        for error in errors:
            lines.append(
                f"- {error.get('tool', 'unknown')}: "
                f"{error.get('error', 'Unknown error')}"
            )

    _append_execution_trace(lines, trace)

    return {
        "final_answer": "\n".join(lines),
        "is_complete": True,
        "needs_more_investigation": False,
        "termination_reason": (
            state.get("termination_reason") or "investigation_completed"
        ),
    }


def create_nodes(
    tools: InvestigationTools,
) -> dict[str, Callable[..., dict[str, Any]]]:
    """Create workflow nodes bound to repository investigation tools."""
    return {
        "query_analysis": analyze_query,
        "tool_selection": lambda state: select_tools(state, tools),
        "investigation": lambda state: investigate(state, tools),
        "answer_preparation": prepare_answer,
    }
