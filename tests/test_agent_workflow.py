from __future__ import annotations

import pytest

from repolens.agent import InvestigationTools, investigate_repository
from repolens.agent.nodes import analyze_query
from repolens.agent.state import InvestigationState
from repolens.graph.code_graph import CodeGraph
from repolens.ingestion.code_unit import CallReference, CodeUnit, ImportReference
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


def test_get_callers_returns_structured_evidence(
    investigation_tools: InvestigationTools,
) -> None:
    results = investigation_tools.get_callers("unit-1")

    assert len(results) == 1
    assert results[0]["code_unit_id"] == "unit-2"
    assert results[0]["source"] == "get_callers"
    assert results[0]["metadata"]["relationship"] == "CALLS"


def test_get_callees_returns_structured_evidence(
    investigation_tools: InvestigationTools,
) -> None:
    results = investigation_tools.get_callees("unit-2")

    assert len(results) == 1
    assert results[0]["code_unit_id"] == "unit-1"
    assert results[0]["source"] == "get_callees"


def test_get_dependencies_returns_module_evidence(tmp_path) -> None:
    module_a = CodeUnit(
        id="module-a",
        kind="module",
        name="a.py",
        qualified_name="a",
        file_path="a.py",
        start_line=1,
        end_line=1,
        imports=(ImportReference(module="b"),),
        repository=str(tmp_path),
        source_code="import b\n",
    )
    module_b = CodeUnit(
        id="module-b",
        kind="module",
        name="b.py",
        qualified_name="b",
        file_path="b.py",
        start_line=1,
        end_line=1,
        repository=str(tmp_path),
        source_code="VALUE = 1\n",
    )
    units = [module_a, module_b]
    tools = InvestigationTools(units, CodeGraph().build(units))

    results = tools.get_dependencies("module-a")

    assert len(results) == 1
    assert results[0]["code_unit_id"] == "module-b"
    assert results[0]["source"] == "get_dependencies"


def test_read_file_returns_requested_source_range(tmp_path) -> None:
    source = "def first():\n    return 1\n\ndef second():\n    return 2\n"
    (tmp_path / "sample.py").write_text(source, encoding="utf-8")
    module = CodeUnit(
        id="module-sample",
        kind="module",
        name="sample.py",
        qualified_name="sample",
        file_path="sample.py",
        start_line=1,
        end_line=6,
        repository=str(tmp_path),
        source_code=source,
    )
    tools = InvestigationTools([module], CodeGraph().build([module]))

    results = tools.read_file("sample.py", start_line=2, end_line=4)

    assert len(results) == 1
    assert results[0]["source"] == "read_file"
    assert results[0]["content"] == "    return 1\n\ndef second():"
    assert results[0]["start_line"] == 2
    assert results[0]["end_line"] == 4


def test_read_file_rejects_path_traversal(tmp_path) -> None:
    module = CodeUnit(
        id="module-sample",
        kind="module",
        name="sample.py",
        qualified_name="sample",
        file_path="sample.py",
        start_line=1,
        end_line=1,
        repository=str(tmp_path),
        source_code="VALUE = 1\n",
    )
    tools = InvestigationTools([module], CodeGraph().build([module]))

    with pytest.raises(ValueError, match="parent traversal"):
        tools.read_file("../outside.py")


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
    assert "get_callers" in executed
    assert "inspect_source" in executed
    assert "read_file" in executed

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


# RL-30: Investigation tracing and execution control tests.


def test_workflow_records_execution_trace(
    investigation_tools: InvestigationTools,
) -> None:
    result = investigate_repository(
        "Find calculate_total and its callers",
        investigation_tools,
    )

    trace = result["trace"]

    assert trace
    assert len(trace) == len(result["history"])

    for record in trace:
        assert record["invocation_id"]
        assert record["iteration"] >= 1
        assert record["tool"]
        assert record["status"] in {"success", "failed"}
        assert record["duration_ms"] >= 0
        assert record["result_count"] >= 0
        assert isinstance(record["findings"], list)
        assert "error" in record

    invocation_ids = [record["invocation_id"] for record in trace]

    assert len(invocation_ids) == len(set(invocation_ids))


def test_execution_trace_records_successful_tool_findings(
    investigation_tools: InvestigationTools,
) -> None:
    result = investigate_repository(
        "Find calculate_total and its callers",
        investigation_tools,
    )

    keyword_trace = next(
        record for record in result["trace"] if record["tool"] == "keyword_search"
    )

    assert keyword_trace["status"] == "success"
    assert keyword_trace["result_count"] > 0
    assert keyword_trace["findings"]
    assert keyword_trace["error"] is None


def test_execution_trace_records_tool_failures(
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

    result = investigate_repository("Find calculate_total", tools)

    semantic_trace = next(
        record for record in result["trace"] if record["tool"] == "semantic_search"
    )

    assert semantic_trace["status"] == "failed"
    assert semantic_trace["result_count"] == 0
    assert semantic_trace["findings"] == []
    assert "Semantic service unavailable" in semantic_trace["error"]


def test_workflow_supports_configurable_minimum_evidence(
    investigation_tools: InvestigationTools,
) -> None:
    result = investigate_repository(
        "Find calculate_total and its callers",
        investigation_tools,
        min_evidence=1,
    )

    assert result["is_complete"] is True
    assert len(result["evidence"]) >= 1
    assert result["termination_reason"] == "sufficient_evidence"


def test_workflow_preserves_default_completion_reason(
    investigation_tools: InvestigationTools,
) -> None:
    result = investigate_repository(
        "Find calculate_total and its callers",
        investigation_tools,
    )

    assert result["termination_reason"] == "investigation_completed"


def test_workflow_rejects_invalid_minimum_evidence(
    investigation_tools: InvestigationTools,
) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        investigate_repository(
            "Find calculate_total",
            investigation_tools,
            min_evidence=-1,
        )


def test_workflow_rejects_minimum_evidence_above_maximum(
    investigation_tools: InvestigationTools,
) -> None:
    with pytest.raises(ValueError, match="cannot exceed max_evidence"):
        investigate_repository(
            "Find calculate_total",
            investigation_tools,
            min_evidence=5,
            max_evidence=4,
        )


def test_workflow_includes_execution_trace_in_final_answer(
    investigation_tools: InvestigationTools,
) -> None:
    result = investigate_repository(
        "Find calculate_total and its callers",
        investigation_tools,
    )

    assert "Execution Trace" in result["final_answer"]
    assert "keyword_search" in result["final_answer"]
    assert "graph_traversal" in result["final_answer"]
    assert "inspect_source" in result["final_answer"]
