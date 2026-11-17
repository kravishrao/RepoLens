"""Tests for repository citation validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from repolens.graph.code_graph import CodeGraph
from repolens.ingestion.code_unit import CodeUnit
from repolens.verification.citation_models import Citation
from repolens.verification.citation_validator import CitationValidator


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    source = tmp_path / "src" / "sample.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def calculate():\n    return 42\n\nclass Calculator:\n    pass\n",
        encoding="utf-8",
    )
    return tmp_path


def make_citation(
    file_path: str = "src/sample.py",
    start_line: int | None = 1,
    end_line: int | None = 2,
    symbol: str | None = None,
    code_unit_id: str | None = None,
) -> Citation:
    return Citation(
        citation_id="citation-1",
        raw_text="sample citation",
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        symbol=symbol,
        code_unit_id=code_unit_id,
    )


def make_code_graph() -> CodeGraph:
    unit = CodeUnit(
        id="unit-1",
        kind="function",
        name="calculate",
        qualified_name="sample.calculate",
        file_path="src/sample.py",
        start_line=1,
        end_line=2,
    )
    return CodeGraph().build([unit])


def test_validates_existing_file_and_line_range(repository: Path) -> None:
    validator = CitationValidator(repository)

    result = validator.validate(make_citation())

    assert result.status == "valid"
    assert result.file_exists is True
    assert result.path_is_safe is True
    assert result.line_range_valid is True
    assert result.issues == ()


def test_rejects_missing_file(repository: Path) -> None:
    validator = CitationValidator(repository)

    result = validator.validate(make_citation("src/missing.py"))

    assert result.status == "invalid"
    assert result.file_exists is False
    assert any(
        issue.code == "file_not_found_or_outside_repository" for issue in result.issues
    )


def test_rejects_directory_traversal(repository: Path) -> None:
    validator = CitationValidator(repository)

    result = validator.validate(make_citation("../outside.py"))

    assert result.status == "invalid"
    assert result.path_is_safe is False
    assert any(issue.code == "unsafe_file_path" for issue in result.issues)


def test_rejects_absolute_path(repository: Path) -> None:
    validator = CitationValidator(repository)

    result = validator.validate(make_citation(str(repository / "src/sample.py")))

    assert result.status == "invalid"
    assert result.path_is_safe is False


def test_rejects_windows_absolute_path(repository: Path) -> None:
    validator = CitationValidator(repository)

    result = validator.validate(make_citation(r"C:\outside\sample.py"))

    assert result.status == "invalid"
    assert result.path_is_safe is False


def test_rejects_line_range_beyond_file_boundary(repository: Path) -> None:
    validator = CitationValidator(repository)

    result = validator.validate(make_citation(start_line=1, end_line=100))

    assert result.status == "invalid"
    assert result.line_range_valid is False
    assert any(issue.code == "line_range_out_of_bounds" for issue in result.issues)


def test_rejects_reversed_line_range(repository: Path) -> None:
    validator = CitationValidator(repository)

    result = validator.validate(make_citation(start_line=4, end_line=2))

    assert result.status == "invalid"
    assert result.line_range_valid is False
    assert any(issue.code == "malformed_line_range" for issue in result.issues)


def test_marks_missing_line_range_incomplete(repository: Path) -> None:
    validator = CitationValidator(repository)

    result = validator.validate(make_citation(start_line=None, end_line=None))

    assert result.status == "incomplete"
    assert result.file_exists is True
    assert result.line_range_valid is None
    assert any(issue.code == "missing_line_range" for issue in result.issues)


def test_verifies_symbol_from_source(repository: Path) -> None:
    validator = CitationValidator(repository)

    result = validator.validate(make_citation(symbol="calculate"))

    assert result.status == "valid"
    assert result.symbol_found is True


def test_rejects_unknown_symbol(repository: Path) -> None:
    validator = CitationValidator(repository)

    result = validator.validate(make_citation(symbol="missing_function"))

    assert result.status == "invalid"
    assert result.symbol_found is False
    assert any(issue.code == "symbol_not_found" for issue in result.issues)


def test_verifies_code_unit_from_graph(repository: Path) -> None:
    validator = CitationValidator(repository, make_code_graph())

    result = validator.validate(
        make_citation(
            symbol="calculate",
            code_unit_id="unit-1",
        )
    )

    assert result.status == "valid"
    assert result.code_unit_found is True
    assert result.symbol_found is True


def test_rejects_unknown_code_unit(repository: Path) -> None:
    validator = CitationValidator(repository, make_code_graph())

    result = validator.validate(make_citation(code_unit_id="unknown-unit"))

    assert result.status == "invalid"
    assert result.code_unit_found is False
    assert any(issue.code == "code_unit_not_found" for issue in result.issues)


def test_rejects_code_unit_from_different_file(repository: Path) -> None:
    graph = CodeGraph().build(
        [
            CodeUnit(
                id="unit-2",
                kind="function",
                name="calculate",
                qualified_name="other.calculate",
                file_path="other.py",
                start_line=1,
                end_line=2,
            )
        ]
    )
    validator = CitationValidator(repository, graph)

    result = validator.validate(make_citation(code_unit_id="unit-2"))

    assert result.status == "invalid"
    assert result.code_unit_found is False
    assert any(issue.code == "code_unit_file_mismatch" for issue in result.issues)


def test_validate_many_returns_independent_results(repository: Path) -> None:
    validator = CitationValidator(repository)

    results = validator.validate_many(
        [
            make_citation(),
            make_citation("src/missing.py"),
        ]
    )

    assert len(results) == 2
    assert results[0].status == "valid"
    assert results[1].status == "invalid"


def test_serializes_validation_result(repository: Path) -> None:
    validator = CitationValidator(repository)

    result = validator.validate(make_citation())
    serialized = result.to_dict()

    assert serialized["status"] == "valid"
    assert serialized["citation"]["citation_id"] == "citation-1"
    assert serialized["issues"] == []


def test_rejects_invalid_repository_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="existing directory"):
        CitationValidator(tmp_path / "does-not-exist")


def test_rejects_invalid_citation_type(repository: Path) -> None:
    validator = CitationValidator(repository)

    with pytest.raises(TypeError, match="Citation instance"):
        validator.validate("not a citation")  # type: ignore[arg-type]


def test_symbol_must_exist_in_actual_file_even_when_graph_contains_it(tmp_path):
    from repolens.graph.code_graph import CodeGraph
    from repolens.ingestion.code_unit import CodeUnit
    from repolens.verification.citation_models import Citation
    from repolens.verification.citation_validator import CitationValidator

    source_file = tmp_path / "app.py"
    source_file.write_text(
        "def actual_function():\n    return True\n",
        encoding="utf-8",
    )

    stale_unit = CodeUnit(
        id="stale-unit",
        kind="function",
        name="ghost_function",
        qualified_name="app.ghost_function",
        file_path="app.py",
        start_line=1,
        end_line=2,
        source_code="def ghost_function(): pass",
    )
    graph = CodeGraph().build([stale_unit])

    validator = CitationValidator(tmp_path, graph)
    citation = Citation(
        citation_id="citation-stale",
        raw_text="app.py:1-2",
        file_path="app.py",
        start_line=1,
        end_line=2,
        symbol="app.ghost_function",
        code_unit_id="stale-unit",
    )

    result = validator.validate(citation)

    assert result.status == "invalid"
    assert result.symbol_found is False
    assert any(issue.code == "symbol_not_found" for issue in result.issues)


def test_symbol_must_be_declared_inside_cited_line_range(tmp_path):
    from repolens.verification.citation_models import Citation
    from repolens.verification.citation_validator import CitationValidator

    source_file = tmp_path / "app.py"
    source_file.write_text(
        "def first_function():\n    return 1\n\ndef second_function():\n    return 2\n",
        encoding="utf-8",
    )

    validator = CitationValidator(tmp_path)
    citation = Citation(
        citation_id="citation-range",
        raw_text="app.py:1-2",
        file_path="app.py",
        start_line=1,
        end_line=2,
        symbol="second_function",
    )

    result = validator.validate(citation)

    assert result.status == "invalid"
    assert result.symbol_found is False
