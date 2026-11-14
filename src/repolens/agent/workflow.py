from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from repolens.agent.nodes import create_nodes
from repolens.agent.state import InvestigationState
from repolens.agent.tools import InvestigationTools


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


def build_investigation_workflow(
    tools: InvestigationTools,
):
    """Build and compile the LangGraph repository investigation workflow."""
    nodes = create_nodes(tools)

    workflow = StateGraph(InvestigationState)

    workflow.add_node("query_analysis", nodes["query_analysis"])
    workflow.add_node("tool_selection", nodes["tool_selection"])
    workflow.add_node("investigation", nodes["investigation"])
    workflow.add_node("answer_preparation", nodes["answer_preparation"])

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

    workflow.add_edge("answer_preparation", END)

    return workflow.compile()


def investigate_repository(
    query: str,
    tools: InvestigationTools,
    *,
    max_iterations: int = 4,
    max_evidence: int = 30,
) -> InvestigationState:
    """Run a repository investigation and return its final state."""
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

    app = build_investigation_workflow(tools)

    initial_state: InvestigationState = {
        "query": query,
        "evidence": [],
        "errors": [],
        "history": [],
        "executed_tools": [],
        "selected_tools": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "max_evidence": max_evidence,
        "needs_more_investigation": True,
        "is_complete": False,
    }

    return app.invoke(initial_state)
