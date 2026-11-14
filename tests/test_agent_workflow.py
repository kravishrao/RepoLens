from __future__ import annotations

import pytest

from repolens.agent import InvestigationTools, investigate_repository
from repolens.agent.nodes import analyze_query
from repolens.agent.state import InvestigationState
from repolens.graph.code_graph import CodeGraph
from repolens.ingestion.code_unit import CallReference, CodeUnit
from repolens.retrieval.semantic_retriever import SemanticSearchResult


@pytest.fixture
def repository_units() -> list[CodeUnit]:
    """Create a small repository with a function-call relationship."""
    calculator = CodeUnit(
        id="unit-1",
        kind="function",
        name="calculate_total",
        qualified_name="app.calculate_total",
        file_path="app.py",
        start_line=1,
        end_line=3,
        source_code="def calculate_total(items):\n    return sum(items)",
        docstring="Calculate the total of the supplied items.",
    )

    processor = CodeUnit(
        id="unit-2",
        kind="function",
        name="process_order",
        qualified_name="app.process_order",
        file_path="orders.py",
        start_line=5,
        end_line=8,
        calls=(CallReference(target="app.calculate_total"),),
        source_code=(
            "def process_order(items):\n"
            "    total = calculate_total(items)\n"
            "    return total"
        ),
    )

    return [calculator, processor]


@pytest.fixture
def investigation_tools(
    repository_units: list[CodeUnit],
) -> InvestigationTools:
    """Build investigation tools for the test repository."""
    graph = CodeGraph().build(repository_units)

    return InvestigationTools(
        code_units=repository_units,
        code_graph=graph,
    )


def test_query_analysis_extracts_keywords() -> None:
    state: InvestigationState = {
        "query": "Find calculate_total and its callers",
    }

    result = analyze_query(state)

    assert result["analyzed_query"] == state["query"]
    assert "calculate" in result["query_keywords"]
    assert "total" in result["query_keywords"]
    assert result["iteration"] == 0


def test_keyword_search_returns_matching_code(
    investigation_tools: InvestigationTools,
) -> None:
    results = investigation_tools.keyword_search("calculate total")

    assert results
    assert results[0]["code_unit_id"] == "unit-1"
    assert results[0]["source"] == "keyword_search"


def test_graph_traversal_finds_callers(
    investigation_tools: InvestigationTools,
) -> None:
    results = investigation_tools.graph_traversal(
        ["unit-1"],
        relationships={"CALLS"},
        direction="both",
    )

    assert len(results) == 1
    assert results[0]["code_unit_id"] == "unit-2"
    assert results[0]["metadata"]["relationship"] == "CALLS"


def test_source_inspection_returns_source_and_docstring(
    investigation_tools: InvestigationTools,
) -> None:
    results = investigation_tools.inspect_source(["unit-1"])

    assert len(results) == 1
    assert results[0]["content"].startswith("def calculate_total")
    assert (
        results[0]["metadata"]["docstring"]
        == "Calculate the total of the supplied items."
    )


def test_semantic_search_is_optional(
    investigation_tools: InvestigationTools,
) -> None:
    assert investigation_tools.semantic_available is False
    assert investigation_tools.semantic_search("calculate total") == []


def test_semantic_search_uses_injected_retriever(
    repository_units: list[CodeUnit],
) -> None:
    class FakeSemanticRetriever:
        def query(
            self,
            query_text: str,
            *,
            top_k: int = 10,
        ) -> list[CodeUnit]:
            assert query_text == "calculate total"
            return repository_units[:top_k]

    graph = CodeGraph().build(repository_units)

    tools = InvestigationTools(
        code_units=repository_units,
        code_graph=graph,
        semantic_retriever=FakeSemanticRetriever(),
    )

    results = tools.semantic_search("calculate total")

    assert tools.semantic_available is True
    assert len(results) == 2
    assert results[0]["source"] == "semantic_search"


def test_semantic_search_converts_ranked_results_to_evidence(
    repository_units: list[CodeUnit],
) -> None:
    class FakeSemanticRetriever:
        def query(
            self,
            query_text: str,
            *,
            top_k: int = 10,
        ) -> list[SemanticSearchResult]:
            assert query_text == "calculate total"
            return [
                SemanticSearchResult(
                    code_unit=repository_units[0],
                    score=0.95,
                    rank=1,
                )
            ][:top_k]

    graph = CodeGraph().build(repository_units)

    tools = InvestigationTools(
        code_units=repository_units,
        code_graph=graph,
        semantic_retriever=FakeSemanticRetriever(),
    )

    results = tools.semantic_search("calculate total")

    assert len(results) == 1
    assert results[0]["evidence_id"] == "semantic:unit-1"
    assert results[0]["code_unit_id"] == "unit-1"
    assert results[0]["source"] == "semantic_search"
    assert results[0]["score"] == pytest.approx(0.95)
    assert results[0]["metadata"]["rank"] == 1


def test_workflow_completes_repository_investigation(
    investigation_tools: InvestigationTools,
) -> None:
    result = investigate_repository(
        "Find calculate_total and its callers",
        investigation_tools,
    )

    assert result["is_complete"] is True
    assert result["final_answer"]
    assert result["evidence"]

    evidence_ids = {item["code_unit_id"] for item in result["evidence"]}

    assert "unit-1" in evidence_ids
    assert "unit-2" in evidence_ids

    executed = result["executed_tools"]

    assert "keyword_search" in executed
    assert "graph_traversal" in executed
    assert "inspect_source" in executed

    assert result["termination_reason"] == "investigation_completed"


def test_workflow_inspects_graph_discovered_code_units(
    investigation_tools: InvestigationTools,
) -> None:
    result = investigate_repository(
        "Find calculate_total and its callers",
        investigation_tools,
    )

    source_evidence_ids = {
        item["code_unit_id"]
        for item in result["evidence"]
        if item["source"] == "source_inspection"
    }

    assert "unit-1" in source_evidence_ids
    assert "unit-2" in source_evidence_ids


def test_workflow_executes_investigation_phases_in_order(
    investigation_tools: InvestigationTools,
) -> None:
    result = investigate_repository(
        "Find calculate_total and its callers",
        investigation_tools,
    )

    tool_order = [
        record["tool"] for record in result["history"] if record["status"] == "success"
    ]

    assert tool_order.index("keyword_search") < tool_order.index("graph_traversal")
    assert tool_order.index("graph_traversal") < tool_order.index("inspect_source")


def test_workflow_does_not_execute_tools_repeatedly(
    investigation_tools: InvestigationTools,
) -> None:
    result = investigate_repository(
        "Find calculate_total and its callers",
        investigation_tools,
    )

    executed_tool_history = [record["tool"] for record in result["history"]]

    assert len(executed_tool_history) == len(set(executed_tool_history))


def test_workflow_respects_maximum_iterations(
    investigation_tools: InvestigationTools,
) -> None:
    result = investigate_repository(
        "Find calculate_total",
        investigation_tools,
        max_iterations=1,
    )

    assert result["is_complete"] is True
    assert result["iteration"] == 1
    assert result["termination_reason"] == "maximum_iterations"


def test_workflow_respects_maximum_evidence(
    investigation_tools: InvestigationTools,
) -> None:
    result = investigate_repository(
        "Find calculate_total and its callers",
        investigation_tools,
        max_evidence=1,
    )

    assert result["is_complete"] is True
    assert len(result["evidence"]) == 1
    assert result["termination_reason"] == "maximum_evidence"


def test_workflow_handles_semantic_tool_failure(
    repository_units: list[CodeUnit],
) -> None:
    class FailingSemanticRetriever:
        def query(
            self,
            query_text: str,
            *,
            top_k: int = 10,
        ) -> list[CodeUnit]:
            raise RuntimeError("Semantic service unavailable")

    graph = CodeGraph().build(repository_units)

    tools = InvestigationTools(
        code_units=repository_units,
        code_graph=graph,
        semantic_retriever=FailingSemanticRetriever(),
    )

    result = investigate_repository(
        "Find calculate_total",
        tools,
    )

    assert result["is_complete"] is True
    assert result["evidence"]

    assert any(
        error["tool"] == "semantic_search"
        and "Semantic service unavailable" in error["error"]
        for error in result["errors"]
    )

    assert "keyword_search" in result["executed_tools"]


def test_workflow_handles_complete_retrieval_failure(
    investigation_tools: InvestigationTools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_keyword_search(query: str, top_k: int = 10):
        raise RuntimeError("Keyword index unavailable")

    monkeypatch.setattr(
        investigation_tools,
        "keyword_search",
        failing_keyword_search,
    )

    result = investigate_repository(
        "Find calculate_total",
        investigation_tools,
    )

    assert result["is_complete"] is True
    assert result["evidence"] == []
    assert result["termination_reason"] == "tool_failures"

    assert any(
        error["tool"] == "keyword_search"
        and "Keyword index unavailable" in error["error"]
        for error in result["errors"]
    )


def test_workflow_handles_empty_search_results(
    investigation_tools: InvestigationTools,
) -> None:
    result = investigate_repository(
        "nonexistent_xyz_symbol",
        investigation_tools,
    )

    assert result["is_complete"] is True
    assert result["evidence"] == []
    assert result["termination_reason"] == "no_evidence"


def test_workflow_rejects_empty_query(
    investigation_tools: InvestigationTools,
) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        investigate_repository("", investigation_tools)


def test_tools_reject_mismatched_repository_data(
    repository_units: list[CodeUnit],
) -> None:
    graph = CodeGraph().build(repository_units[:1])

    with pytest.raises(ValueError, match="same CodeUnit collection"):
        InvestigationTools(
            code_units=repository_units,
            code_graph=graph,
        )
