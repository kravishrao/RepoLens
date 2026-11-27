"""Integration and end-to-end tests for the RepoLens Streamlit application."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from repolens.ui.app import load_repository_tools


def test_load_repository_tools_valid_path():
    tools, count = load_repository_tools("src/repolens/verification")
    assert count > 0
    assert tools is not None


def test_load_repository_tools_nonexistent_path():
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_repository_tools("path/does/not/exist/anywhere")


def test_load_repository_tools_empty_directory(tmp_path):
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No Python code units discovered"):
        load_repository_tools(str(empty_dir))


def test_app_test_initial_render():
    at = AppTest.from_file("../src/repolens/ui/app.py")
    at.run()

    assert not at.exception
    # Verify title
    assert len(at.title) >= 1
    assert "RepoLens" in at.title[0].value
    # Verify welcome section exists
    assert any("Welcome to RepoLens" in s.value for s in at.subheader)


def test_app_test_with_sample_state():
    at = AppTest.from_file("../src/repolens/ui/app.py")
    # Pre-populate session state with sample investigation data
    at.session_state["investigation_state"] = {
        "query": "Where is extract_citations defined?",
        "final_answer": "1. extract_citations\n   File: src/repolens/verification/citation_extractor.py\n   Lines: 10-25",
        "is_complete": True,
        "termination_reason": "investigation_completed",
        "evidence": [
            {
                "evidence_id": "ev-1",
                "code_unit_id": "u-1",
                "name": "extract_citations",
                "qualified_name": "repolens.verification.citation_extractor.extract_citations",
                "kind": "function",
                "file_path": "src/repolens/verification/citation_extractor.py",
                "start_line": 10,
                "end_line": 25,
                "source": "inspect_source",
                "content": "def extract_citations(text): pass",
                "metadata": {"docstring": "Extract citations from text."},
            }
        ],
        "trace": [
            {
                "iteration": 1,
                "tool": "keyword_search",
                "status": "success",
                "duration_ms": 2.5,
                "result_count": 1,
                "retained_count": 1,
                "findings": ["ev-1"],
            }
        ],
        "citation_validation_results": [
            {
                "status": "valid",
                "citation": {
                    "citation_id": "cit-1",
                    "raw_text": "extract_citations",
                    "symbol": "extract_citations",
                    "file_path": "src/repolens/verification/citation_extractor.py",
                    "start_line": 10,
                    "end_line": 25,
                },
                "file_exists": True,
                "path_is_safe": True,
                "line_range_valid": True,
                "symbol_found": True,
                "code_unit_found": True,
                "issues": [],
            }
        ],
        "validation_summary": {
            "enabled": True,
            "overall_status": "valid",
            "citation_count": 1,
            "claim_count": 0,
        },
    }

    at.run()
    assert not at.exception
    # Tabs and metrics should be rendered
    assert len(at.metric) >= 5
