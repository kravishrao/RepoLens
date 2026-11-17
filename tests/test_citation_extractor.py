"""Tests for repository source citation extraction."""

from __future__ import annotations

import pytest

from repolens.verification.citation_extractor import extract_citations


def test_extracts_repolens_multiline_citation() -> None:
    answer = (
        "1. prepare_answer\n"
        "   Type: function\n"
        "   File: src/repolens/agent/nodes.py\n"
        "   Lines: 10-25\n"
        "   Evidence sources: source inspection\n"
    )

    citations = extract_citations(answer)

    assert len(citations) == 1
    assert citations[0].file_path == "src/repolens/agent/nodes.py"
    assert citations[0].start_line == 10
    assert citations[0].end_line == 25
    assert citations[0].symbol == "prepare_answer"


def test_extracts_inline_file_line_range() -> None:
    citations = extract_citations("See src/repolens/agent/nodes.py:10-25 for details.")

    assert len(citations) == 1
    assert citations[0].start_line == 10
    assert citations[0].end_line == 25


def test_extracts_single_line_reference() -> None:
    citations = extract_citations("See src/repolens/agent/nodes.py:10.")

    assert len(citations) == 1
    assert citations[0].start_line == 10
    assert citations[0].end_line == 10


def test_extracts_backtick_line_reference() -> None:
    citations = extract_citations("See `src/repolens/agent/nodes.py` (lines 10-25).")

    assert len(citations) == 1
    assert citations[0].start_line == 10
    assert citations[0].end_line == 25


def test_extracts_multiple_citations() -> None:
    answer = (
        "See src/repolens/agent/nodes.py:10-25 and src/repolens/agent/state.py:1-15."
    )

    citations = extract_citations(answer)

    assert len(citations) == 2
    assert citations[0].file_path == "src/repolens/agent/nodes.py"
    assert citations[1].file_path == "src/repolens/agent/state.py"


def test_retains_incomplete_file_reference() -> None:
    answer = "1. prepare_answer\n   File: src/repolens/agent/nodes.py\n"

    citations = extract_citations(answer)

    assert len(citations) == 1
    assert citations[0].file_path == "src/repolens/agent/nodes.py"
    assert citations[0].start_line is None
    assert citations[0].end_line is None


def test_does_not_duplicate_complete_multiline_citation() -> None:
    answer = (
        "1. prepare_answer\n   File: src/repolens/agent/nodes.py\n   Lines: 10-25\n"
    )

    citations = extract_citations(answer)

    assert len(citations) == 1
    assert citations[0].start_line == 10


def test_extracts_nearest_preceding_symbol() -> None:
    answer = (
        "1. prepare_answer\n"
        "   File: src/repolens/agent/nodes.py\n"
        "   Lines: 10-25\n"
        "2. create_nodes\n"
        "   File: src/repolens/agent/nodes.py\n"
        "   Lines: 30-45\n"
    )

    citations = extract_citations(answer)

    assert len(citations) == 2
    assert citations[0].symbol == "prepare_answer"
    assert citations[1].symbol == "create_nodes"


def test_rejects_non_string_answer() -> None:
    with pytest.raises(TypeError, match="Answer must be a string"):
        extract_citations(None)  # type: ignore[arg-type]


def test_empty_answer_returns_no_citations() -> None:
    assert extract_citations("") == []


@pytest.mark.parametrize(
    "file_path",
    [
        "src/app.js",
        "src/App.jsx",
        "src/app.ts",
        "src/App.tsx",
        "src/Main.java",
        "src/robot.c",
        "include/robot.h",
        "src/robot.cpp",
        "include/robot.hpp",
        "src/robot.cs",
        "src/main.go",
        "src/lib.rs",
        "src/script.rb",
        "src/index.php",
        "src/App.swift",
        "src/Main.kt",
        "src/query.sql",
    ],
)
def test_extracts_multiple_source_file_extensions(file_path: str) -> None:
    citations = extract_citations(f"See {file_path}:10-20 for details.")

    assert len(citations) == 1
    assert citations[0].file_path == file_path
    assert citations[0].start_line == 10
    assert citations[0].end_line == 20


@pytest.mark.parametrize(
    "file_path",
    [
        "src/app.js",
        "src/App.tsx",
        "src/Main.java",
        "src/robot.cpp",
        "src/main.go",
        "src/lib.rs",
    ],
)
def test_extracts_backtick_references_for_multiple_languages(
    file_path: str,
) -> None:
    citations = extract_citations(f"See `{file_path}` (lines 5-15).")

    assert len(citations) == 1
    assert citations[0].file_path == file_path
    assert citations[0].start_line == 5
    assert citations[0].end_line == 15


@pytest.mark.parametrize(
    "file_path",
    [
        "src/app.js",
        "src/App.tsx",
        "src/Main.java",
        "src/robot.cpp",
        "src/main.go",
        "src/lib.rs",
    ],
)
def test_retains_incomplete_multilanguage_file_references(
    file_path: str,
) -> None:
    answer = f"1. example_function\n   File: {file_path}\n"

    citations = extract_citations(answer)

    assert len(citations) == 1
    assert citations[0].file_path == file_path
    assert citations[0].start_line is None
    assert citations[0].end_line is None


def test_does_not_duplicate_complete_multilanguage_citation() -> None:
    answer = "1. handle_request\n   File: src/server.js\n   Lines: 10-25\n"

    citations = extract_citations(answer)

    assert len(citations) == 1
    assert citations[0].file_path == "src/server.js"
    assert citations[0].start_line == 10


def test_does_not_extract_unsupported_file_extension() -> None:
    citations = extract_citations("See README.md:10-20 for details.")

    assert citations == []
