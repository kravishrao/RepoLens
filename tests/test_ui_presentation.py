"""Unit tests for RepoLens presentation view models and data extraction."""

from __future__ import annotations

from repolens.ui.presentation import (
    extract_presentation_data,
    format_line_reference,
    is_supported_finding,
)


def test_format_line_reference():
    assert format_line_reference(10, 20) == "Lines 10-20"
    assert format_line_reference(15, 15) == "Line 15"
    assert format_line_reference(42, None) == "Line 42+"
    assert format_line_reference(None, 42) == "Lines unknown"
    assert format_line_reference(None, None) == "Lines unknown"


def test_is_supported_finding():
    # Direct inspection sources
    assert is_supported_finding({"source": "inspect_source", "content": ""}) is True
    assert is_supported_finding({"source": "source_inspection", "content": ""}) is True
    assert is_supported_finding({"source": "read_file", "content": ""}) is True

    # Graph relationship sources without direct inspection
    assert is_supported_finding({"source": "context_expansion", "content": ""}) is False
    assert is_supported_finding({"source": "graph_query", "content": ""}) is False
    assert is_supported_finding({"source": "callers", "content": ""}) is False
    assert is_supported_finding({"source": "callees", "content": ""}) is False

    # Retrieval sources with verified content
    assert (
        is_supported_finding({"source": "keyword_search", "content": "def foo(): pass"})
        is True
    )
    assert is_supported_finding({"source": "keyword_search", "content": ""}) is False


def test_extract_presentation_data_from_none():
    data = extract_presentation_data(None)

    assert data.query == ""
    assert "No investigation has been executed" in data.final_answer
    assert data.is_complete is False
    assert data.has_evidence is False
    assert data.has_errors is False
    assert data.all_evidence == []
    assert data.supported_findings == []
    assert data.inferred_findings == []
    assert data.citations == []
    assert data.trace_steps == []
    assert data.executed_tools == []
    assert data.validation_summary.enabled is False


def test_extract_presentation_data_from_empty_dict():
    data = extract_presentation_data({})

    assert data.query == ""
    assert data.final_answer == ""
    assert data.is_complete is False
    assert data.has_evidence is False
    assert data.has_errors is False
    assert data.termination_reason == "unspecified"
    assert data.all_evidence == []


def test_extract_presentation_data_distinguishes_supported_and_inferred():
    state = {
        "query": "How is order calculated?",
        "final_answer": "1. billing.calc\n   File: src/calc.py\n   Lines: 10-25",
        "is_complete": True,
        "termination_reason": "sufficient_evidence",
        "evidence": [
            {
                "evidence_id": "ev-1",
                "code_unit_id": "unit-1",
                "name": "calc",
                "qualified_name": "billing.calc",
                "kind": "function",
                "file_path": "src/calc.py",
                "start_line": 10,
                "end_line": 25,
                "source": "inspect_source",
                "content": "def calc(items):\n    return sum(items)",
                "metadata": {"docstring": "Compute total."},
                "score": 0.95,
            },
            {
                "evidence_id": "ev-2",
                "code_unit_id": "unit-2",
                "name": "apply_discount",
                "qualified_name": "billing.apply_discount",
                "kind": "function",
                "file_path": "src/discounts.py",
                "start_line": 50,
                "end_line": 70,
                "source": "context_expansion",
                "content": "",
                "metadata": {},
            },
        ],
    }

    data = extract_presentation_data(state)

    assert data.has_evidence is True
    assert len(data.all_evidence) == 2
    assert len(data.supported_findings) == 1
    assert len(data.inferred_findings) == 1

    supported = data.supported_findings[0]
    assert supported.qualified_name == "billing.calc"
    assert supported.line_reference == "Lines 10-25"
    assert supported.is_supported is True
    assert supported.docstring == "Compute total."
    assert supported.score == 0.95

    inferred = data.inferred_findings[0]
    assert inferred.qualified_name == "billing.apply_discount"
    assert inferred.line_reference == "Lines 50-70"
    assert inferred.is_supported is False
    assert inferred.score is None


def test_extract_presentation_data_citations_and_validation():
    state = {
        "query": "Find citations",
        "final_answer": "Answer with citations",
        "is_complete": True,
        "citation_validation_results": [
            {
                "status": "valid",
                "citation": {
                    "citation_id": "cit-1",
                    "raw_text": "module.fn (lines 1-10)",
                    "symbol": "module.fn",
                    "file_path": "src/fn.py",
                    "start_line": 1,
                    "end_line": 10,
                },
                "file_exists": True,
                "path_is_safe": True,
                "line_range_valid": True,
                "symbol_found": True,
                "code_unit_found": True,
                "issues": [],
            },
            {
                "status": "invalid",
                "citation": {
                    "citation_id": "cit-2",
                    "raw_text": "bad.fn (lines 99-100)",
                    "symbol": "bad.fn",
                    "file_path": "src/bad.py",
                    "start_line": 99,
                    "end_line": 100,
                },
                "file_exists": False,
                "path_is_safe": True,
                "line_range_valid": False,
                "symbol_found": False,
                "code_unit_found": False,
                "issues": [
                    {
                        "code": "FILE_NOT_FOUND",
                        "message": "File does not exist",
                        "severity": "error",
                    }
                ],
            },
        ],
        "claim_validation_results": [
            {
                "claim": "fn is located in src/fn.py",
                "status": "valid",
                "supporting_evidence_ids": ["ev-1"],
                "issues": [],
            }
        ],
        "validation_summary": {
            "enabled": True,
            "overall_status": "invalid",
            "citation_count": 2,
            "claim_count": 1,
        },
    }

    data = extract_presentation_data(state)

    assert len(data.citations) == 2
    c1, c2 = data.citations
    assert c1.status == "valid"
    assert c1.file_exists is True
    assert c1.symbol == "module.fn"

    assert c2.status == "invalid"
    assert c2.file_exists is False
    assert len(c2.issues) == 1
    assert c2.issues[0]["code"] == "FILE_NOT_FOUND"

    assert len(data.claims) == 1
    assert data.claims[0].claim == "fn is located in src/fn.py"
    assert data.claims[0].status == "valid"

    val = data.validation_summary
    assert val.enabled is True
    assert val.overall_status == "invalid"
    assert val.valid_citation_count == 1
    assert val.invalid_citation_count == 1


def test_extract_presentation_data_trace_and_errors():
    state = {
        "query": "Test trace and error handling",
        "is_complete": False,
        "termination_reason": "tool_failures",
        "trace": [
            {
                "iteration": 1,
                "tool": "keyword_search",
                "status": "success",
                "duration_ms": 12.5,
                "result_count": 10,
                "retained_count": 3,
                "findings": ["ev-1", "ev-2", "ev-3"],
            },
            {
                "iteration": 2,
                "tool": "inspect_source",
                "status": "failed",
                "duration_ms": 5.0,
                "result_count": 0,
                "retained_count": 0,
                "findings": [],
                "error": "Failed to read file: permission denied",
            },
        ],
        "executed_tools": ["keyword_search", "inspect_source"],
        "errors": [
            {
                "tool": "inspect_source",
                "error": "Failed to read file: permission denied",
            }
        ],
    }

    data = extract_presentation_data(state)

    assert data.has_errors is True
    assert len(data.errors) == 1
    assert data.errors[0]["tool"] == "inspect_source"
    assert len(data.trace_steps) == 2
    assert data.total_duration_ms == 17.5
    assert data.executed_tools == ["keyword_search", "inspect_source"]

    step1 = data.trace_steps[0]
    assert step1.iteration == 1
    assert step1.tool == "keyword_search"
    assert step1.status == "success"
    assert step1.retained_count == 3
    assert step1.findings == ["ev-1", "ev-2", "ev-3"]

    step2 = data.trace_steps[1]
    assert step2.status == "failed"
    assert step2.error == "Failed to read file: permission denied"
