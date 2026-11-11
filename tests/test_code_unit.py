import pytest

from repolens.ingestion.code_unit import (
    CallReference,
    CodeUnit,
    ImportReference,
)


def create_code_unit(**overrides: object) -> CodeUnit:
    values: dict[str, object] = {
        "id": "unit-123",
        "kind": "function",
        "name": "calculate",
        "qualified_name": "calculator.calculate",
        "file_path": "calculator.py",
        "start_line": 1,
        "end_line": 5,
    }
    values.update(overrides)
    return CodeUnit(**values)


def test_code_unit_stores_repository_and_source_metadata() -> None:
    unit = create_code_unit(
        repository="/projects/example",
        source_code="def calculate():\n    return 42",
        docstring="Calculate a value.",
    )

    assert unit.repository == "/projects/example"
    assert unit.source_code == "def calculate():\n    return 42"
    assert unit.docstring == "Calculate a value."


def test_code_unit_alias_properties() -> None:
    unit = create_code_unit()

    assert unit.unit_id == unit.id
    assert unit.symbol == unit.name
    assert unit.type == unit.kind


def test_code_unit_preserves_relationships() -> None:
    unit = create_code_unit(
        imports=(ImportReference(module="math"),),
        calls=(CallReference(target="math.sqrt"),),
    )

    assert unit.imports == (ImportReference(module="math"),)
    assert unit.calls == (CallReference(target="math.sqrt"),)


@pytest.mark.parametrize(
    "kind",
    ["module", "class", "function", "method"],
)
def test_code_unit_accepts_supported_kinds(kind: str) -> None:
    unit = create_code_unit(kind=kind)

    assert unit.kind == kind


def test_code_unit_rejects_invalid_kind() -> None:
    with pytest.raises(ValueError, match="Invalid CodeUnit kind"):
        create_code_unit(kind="variable")


def test_code_unit_rejects_empty_repository_identity() -> None:
    with pytest.raises(ValueError, match="Repository identity"):
        create_code_unit(repository="   ")


def test_code_unit_rejects_invalid_source_code_type() -> None:
    with pytest.raises(TypeError, match="Source code"):
        create_code_unit(source_code=123)


def test_code_unit_rejects_invalid_docstring_type() -> None:
    with pytest.raises(TypeError, match="Docstring"):
        create_code_unit(docstring=123)


def test_code_unit_rejects_invalid_line_range() -> None:
    with pytest.raises(ValueError, match="End line"):
        create_code_unit(start_line=10, end_line=5)
