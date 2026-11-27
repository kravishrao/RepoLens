from __future__ import annotations

from typing import Any

from repolens.agent.workflow import investigate_repository


class FakeInvestigationTools:
    """Minimal tools for testing RL-30 investigation control."""

    semantic_available = False

    def __init__(self, keyword_results: list[dict[str, Any]]) -> None:
        self.keyword_results = keyword_results
        self.calls: list[str] = []

    def keyword_search(self, query: str) -> list[dict[str, Any]]:
        self.calls.append("keyword_search")
        return self.keyword_results

    def semantic_search(self, query: str) -> list[dict[str, Any]]:
        self.calls.append("semantic_search")
        return []

    def graph_traversal(
        self,
        code_unit_ids: list[str],
        *,
        max_depth: int = 2,
    ) -> list[dict[str, Any]]:
        self.calls.append("graph_traversal")
        return []

    def inspect_source(
        self,
        code_unit_ids: list[str],
    ) -> list[dict[str, Any]]:
        self.calls.append("inspect_source")
        return []


def make_evidence(
    evidence_id: str,
    code_unit_id: str,
) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "code_unit_id": code_unit_id,
        "qualified_name": code_unit_id,
        "kind": "function",
        "file_path": "src/example.py",
        "start_line": 1,
        "end_line": 10,
        "source": "keyword_search",
        "content": "def example(): pass",
    }


def test_minimum_evidence_stops_unrequested_investigation_phases() -> None:
    tools = FakeInvestigationTools([make_evidence("evidence-1", "example")])

    result = investigate_repository(
        "Find example",
        tools,
        min_evidence=1,
    )

    assert result["termination_reason"] == "sufficient_evidence"
    assert result["is_complete"] is True
    assert len(result["evidence"]) == 1
    assert tools.calls == ["keyword_search"]


def test_minimum_evidence_preserves_requested_graph_traversal() -> None:
    tools = FakeInvestigationTools([make_evidence("evidence-1", "example")])

    result = investigate_repository(
        "Find example and its callers",
        tools,
        min_evidence=1,
    )

    assert result["termination_reason"] == "sufficient_evidence"
    assert "graph_traversal" in tools.calls
    assert "inspect_source" not in tools.calls
    assert tools.calls == ["keyword_search", "graph_traversal"]


def test_minimum_evidence_preserves_requested_source_inspection() -> None:
    tools = FakeInvestigationTools([make_evidence("evidence-1", "example")])

    result = investigate_repository(
        "Explain example",
        tools,
        min_evidence=1,
    )

    assert result["termination_reason"] == "sufficient_evidence"
    assert "inspect_source" in tools.calls
    assert "graph_traversal" not in tools.calls
    assert tools.calls == ["keyword_search", "inspect_source"]


def test_trace_distinguishes_raw_and_retained_evidence() -> None:
    tools = FakeInvestigationTools(
        [
            make_evidence("evidence-1", "example_one"),
            make_evidence("evidence-2", "example_two"),
        ]
    )

    result = investigate_repository(
        "Find examples",
        tools,
        max_evidence=1,
    )

    keyword_trace = next(
        record for record in result["trace"] if record["tool"] == "keyword_search"
    )

    assert keyword_trace["result_count"] == 2
    assert keyword_trace["retained_count"] == 1
    assert keyword_trace["findings"] == ["evidence-1"]
    assert len(result["evidence"]) == 1


def test_no_evidence_final_answer_includes_execution_trace() -> None:
    tools = FakeInvestigationTools([])

    result = investigate_repository(
        "Find nonexistent_function",
        tools,
    )

    assert result["termination_reason"] == "no_evidence"
    assert result["evidence"] == []
    assert "Execution Trace" in result["final_answer"]
    assert "keyword_search" in result["final_answer"]
