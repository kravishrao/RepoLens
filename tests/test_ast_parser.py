import logging
from unittest.mock import patch

import pytest

from repolens.ingestion.ast_parser import PythonASTParser


@pytest.fixture
def parser():
    return PythonASTParser()


def test_extracts_module(parser, tmp_path):
    source = tmp_path / "main.py"
    source.write_text("x = 10\n", encoding="utf-8")

    result = parser.parse_file(source)

    assert result.unit_count == 1
    assert result.code_units[0].kind == "module"
    assert result.code_units[0].name == "main.py"


def test_extracts_class(parser, tmp_path):
    source = tmp_path / "main.py"
    source.write_text(
        "class Calculator:\n    pass\n",
        encoding="utf-8",
    )

    result = parser.parse_file(source)

    classes = [unit for unit in result.code_units if unit.kind == "class"]

    assert len(classes) == 1
    assert classes[0].name == "Calculator"
    assert classes[0].qualified_name == "main.Calculator"


def test_extracts_function(parser, tmp_path):
    source = tmp_path / "main.py"
    source.write_text(
        "def calculate():\n    return 10\n",
        encoding="utf-8",
    )

    result = parser.parse_file(source)

    functions = [unit for unit in result.code_units if unit.kind == "function"]

    assert len(functions) == 1
    assert functions[0].name == "calculate"


def test_extracts_class_methods(parser, tmp_path):
    source = tmp_path / "main.py"
    source.write_text(
        "class Calculator:\n    def add(self, a, b):\n        return a + b\n",
        encoding="utf-8",
    )

    result = parser.parse_file(source)

    methods = [unit for unit in result.code_units if unit.kind == "method"]

    assert len(methods) == 1
    assert methods[0].name == "add"
    assert methods[0].qualified_name == "main.Calculator.add"

    class_unit = next(unit for unit in result.code_units if unit.kind == "class")

    assert methods[0].parent_id == class_unit.id


def test_extracts_nested_functions(parser, tmp_path):
    source = tmp_path / "main.py"
    source.write_text(
        "def outer():\n    def inner():\n        return 1\n    return inner()\n",
        encoding="utf-8",
    )

    result = parser.parse_file(source)

    functions = [unit for unit in result.code_units if unit.kind == "function"]

    assert len(functions) == 2

    inner = next(unit for unit in functions if unit.name == "inner")

    assert inner.qualified_name == "main.outer.inner"


def test_extracts_async_functions(parser, tmp_path):
    source = tmp_path / "main.py"
    source.write_text(
        "async def fetch_data():\n    return None\n",
        encoding="utf-8",
    )

    result = parser.parse_file(source)

    functions = [unit for unit in result.code_units if unit.kind == "function"]

    assert len(functions) == 1
    assert functions[0].name == "fetch_data"


def test_extracts_imports(parser, tmp_path):
    source = tmp_path / "main.py"
    source.write_text(
        "import os\nimport json\nfrom pathlib import Path\n",
        encoding="utf-8",
    )

    result = parser.parse_file(source)

    module = result.code_units[0]
    imports = {reference.module for reference in module.imports}

    assert imports == {"os", "json", "pathlib"}


def test_extracts_function_calls(parser, tmp_path):
    source = tmp_path / "main.py"
    source.write_text(
        "def helper():\n    pass\n\ndef main():\n    helper()\n",
        encoding="utf-8",
    )

    result = parser.parse_file(source)

    main = next(unit for unit in result.code_units if unit.name == "main")

    assert [call.target for call in main.calls] == ["helper"]


def test_extracts_attribute_calls(parser, tmp_path):
    source = tmp_path / "main.py"
    source.write_text(
        "def main():\n    service.execute()\n",
        encoding="utf-8",
    )

    result = parser.parse_file(source)

    main = next(unit for unit in result.code_units if unit.name == "main")

    assert main.calls[0].target == "service.execute"


def test_line_numbers_are_recorded(parser, tmp_path):
    source = tmp_path / "main.py"
    source.write_text(
        "class Example:\n    def run(self):\n        return True\n",
        encoding="utf-8",
    )

    result = parser.parse_file(source)

    method = next(unit for unit in result.code_units if unit.kind == "method")

    assert method.start_line == 2
    assert method.end_line == 3


def test_syntax_error_is_recorded(parser, tmp_path):
    source = tmp_path / "broken.py"
    source.write_text(
        "def broken(:\n    pass\n",
        encoding="utf-8",
    )

    result = parser.parse_file(source)

    assert result.unit_count == 0
    assert result.error_count == 1
    assert result.errors[0].line_number == 1


def test_empty_file_produces_module(parser, tmp_path):
    source = tmp_path / "empty.py"
    source.write_text("", encoding="utf-8")

    result = parser.parse_file(source)

    assert result.unit_count == 1
    assert result.code_units[0].kind == "module"
    assert result.code_units[0].start_line == 1
    assert result.code_units[0].end_line == 1


def test_ids_are_deterministic(parser, tmp_path):
    source = tmp_path / "main.py"
    source.write_text(
        "def calculate():\n    return 10\n",
        encoding="utf-8",
    )

    first = parser.parse_file(source)
    second = parser.parse_file(source)

    assert [unit.id for unit in first.code_units] == [
        unit.id for unit in second.code_units
    ]


def test_repository_parsing(parser, tmp_path):
    (tmp_path / "main.py").write_text(
        "def main():\n    pass\n",
        encoding="utf-8",
    )

    (tmp_path / "helper.py").write_text(
        "class Helper:\n    pass\n",
        encoding="utf-8",
    )

    result = parser.parse_repository(tmp_path)

    assert result.error_count == 0

    kinds = {unit.kind for unit in result.code_units}

    assert {"module", "function", "class"} <= kinds


def test_repository_parsing_continues_after_syntax_error(
    parser,
    tmp_path,
):
    (tmp_path / "valid.py").write_text(
        "def valid():\n    pass\n",
        encoding="utf-8",
    )

    (tmp_path / "broken.py").write_text(
        "def broken(:\n",
        encoding="utf-8",
    )

    result = parser.parse_repository(tmp_path)

    assert result.error_count == 1

    valid_functions = [unit for unit in result.code_units if unit.name == "valid"]

    assert len(valid_functions) == 1


def test_invalid_file_path_is_rejected(parser, tmp_path):
    with pytest.raises(FileNotFoundError):
        parser.parse_file(tmp_path / "missing.py")


def test_file_outside_repository_root_is_rejected(
    parser,
    tmp_path,
):
    source = tmp_path / "main.py"
    source.write_text("pass\n", encoding="utf-8")

    other_root = tmp_path / "other"
    other_root.mkdir()

    with pytest.raises(ValueError):
        parser.parse_file(source, repository_root=other_root)


def test_imports_inside_functions_are_not_module_imports(
    parser,
    tmp_path,
):
    source = tmp_path / "main.py"
    source.write_text(
        "def main():\n    import json\n",
        encoding="utf-8",
    )

    result = parser.parse_file(source)

    module = result.code_units[0]

    assert module.imports == ()


def test_nested_function_calls_are_not_attributed_to_parent(
    parser,
    tmp_path,
):
    source = tmp_path / "main.py"
    source.write_text(
        "def outer():\n    def inner():\n        helper()\n    inner()\n",
        encoding="utf-8",
    )

    result = parser.parse_file(source)

    outer = next(unit for unit in result.code_units if unit.name == "outer")
    inner = next(unit for unit in result.code_units if unit.name == "inner")

    assert [call.target for call in outer.calls] == ["inner"]
    assert [call.target for call in inner.calls] == ["helper"]


def test_module_name_for_init_file(parser, tmp_path):
    package = tmp_path / "mypackage"
    package.mkdir()

    source = package / "__init__.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    result = parser.parse_file(
        source,
        repository_root=tmp_path,
    )

    assert result.code_units[0].qualified_name == "mypackage"


def test_extracts_definitions_inside_control_flow(parser, tmp_path):
    source = tmp_path / "main.py"
    source.write_text(
        "if True:\n"
        "    def conditional_function():\n"
        "        pass\n"
        "\n"
        "class Container:\n"
        "    if True:\n"
        "        def nested_method(self):\n"
        "            pass\n",
        encoding="utf-8",
    )

    result = parser.parse_file(source)

    functions = [unit for unit in result.code_units if unit.kind == "function"]
    methods = [unit for unit in result.code_units if unit.kind == "method"]

    assert any(unit.name == "conditional_function" for unit in functions)
    assert any(unit.name == "nested_method" for unit in methods)


def test_duplicate_function_names_have_unique_ids(parser, tmp_path):
    source = tmp_path / "main.py"
    source.write_text(
        "def process():\n    pass\n\ndef process():\n    return True\n",
        encoding="utf-8",
    )

    result = parser.parse_file(source)

    functions = [unit for unit in result.code_units if unit.kind == "function"]

    assert len(functions) == 2
    assert functions[0].id != functions[1].id


def test_parse_file_populates_module_metadata(tmp_path):
    source = '''"""Module documentation."""

import math


def calculate():
    """Calculate a value."""
    return math.sqrt(16)
'''

    file_path = tmp_path / "calculator.py"
    file_path.write_text(source, encoding="utf-8")

    result = PythonASTParser().parse_file(
        file_path,
        repository_root=tmp_path,
    )

    module = next(unit for unit in result.code_units if unit.kind == "module")

    assert module.repository == str(tmp_path.resolve())
    assert module.source_code == source
    assert module.docstring == "Module documentation."


def test_parse_file_populates_function_metadata(tmp_path):
    source = '''def calculate():
    """Calculate a value."""
    return 42
'''

    file_path = tmp_path / "calculator.py"
    file_path.write_text(source, encoding="utf-8")

    result = PythonASTParser().parse_file(
        file_path,
        repository_root=tmp_path,
    )

    function = next(unit for unit in result.code_units if unit.kind == "function")

    assert function.repository == str(tmp_path.resolve())
    assert function.source_code == source.strip()
    assert function.docstring == "Calculate a value."


def test_parse_file_populates_class_and_method_metadata(tmp_path):
    source = '''class Calculator:
    """Calculator class."""

    def add(self, first, second):
        """Add two values."""
        return first + second
'''

    file_path = tmp_path / "calculator.py"
    file_path.write_text(source, encoding="utf-8")

    result = PythonASTParser().parse_file(
        file_path,
        repository_root=tmp_path,
    )

    class_unit = next(unit for unit in result.code_units if unit.kind == "class")
    method_unit = next(unit for unit in result.code_units if unit.kind == "method")

    assert class_unit.docstring == "Calculator class."
    assert class_unit.source_code.startswith("class Calculator:")

    assert method_unit.docstring == "Add two values."
    assert method_unit.source_code.startswith("    def add(")
    assert method_unit.repository == str(tmp_path.resolve())


def test_parse_file_populates_async_function_metadata(tmp_path):
    source = '''async def fetch_data():
    """Fetch data asynchronously."""
    return 42
'''

    file_path = tmp_path / "service.py"
    file_path.write_text(source, encoding="utf-8")

    result = PythonASTParser().parse_file(
        file_path,
        repository_root=tmp_path,
    )

    function = next(unit for unit in result.code_units if unit.kind == "function")

    assert function.docstring == "Fetch data asynchronously."
    assert function.source_code == source.strip()
    assert function.repository == str(tmp_path.resolve())


# ============================================================
# RL-20: AST Extraction Validation and Error Handling
# ============================================================


def test_parse_file_reports_success_statistics(tmp_path):
    """A valid file should count as one successful extraction."""
    source = tmp_path / "valid.py"
    source.write_text(
        "def calculate():\n    return 42\n",
        encoding="utf-8",
    )

    result = PythonASTParser().parse_file(source)

    assert result.files_processed == 1
    assert result.files_succeeded == 1
    assert result.files_failed == 0
    assert result.error_count == 0
    assert result.unit_count == 2


def test_empty_file_reports_success_statistics(tmp_path):
    """An empty Python file is valid and should count as successful."""
    source = tmp_path / "empty.py"
    source.write_text("", encoding="utf-8")

    result = PythonASTParser().parse_file(source)

    assert result.files_processed == 1
    assert result.files_succeeded == 1
    assert result.files_failed == 0
    assert result.unit_count == 1


def test_syntax_error_reports_failure_statistics(tmp_path):
    """Syntax errors should be classified and counted."""
    source = tmp_path / "broken.py"
    source.write_text("def broken(:\n    pass\n", encoding="utf-8")

    result = PythonASTParser().parse_file(source)

    assert result.files_processed == 1
    assert result.files_succeeded == 0
    assert result.files_failed == 1
    assert result.unit_count == 0
    assert result.error_count == 1

    error = result.errors[0]
    assert error.file_path == "broken.py"
    assert error.error_type == "SyntaxError"
    assert error.line_number == 1


def test_invalid_utf8_is_reported(tmp_path):
    """Invalid UTF-8 source files should not crash extraction."""
    source = tmp_path / "invalid_encoding.py"
    source.write_bytes(b"\xff\xfe\xfa")

    result = PythonASTParser().parse_file(source)

    assert result.files_processed == 1
    assert result.files_succeeded == 0
    assert result.files_failed == 1
    assert result.error_count == 1
    assert result.errors[0].error_type == "UnicodeDecodeError"


def test_permission_error_is_reported(tmp_path):
    """A file-reading permission error should be recorded."""
    source = tmp_path / "restricted.py"
    source.write_text("x = 1\n", encoding="utf-8")

    original_read_text = type(source).read_text

    def mocked_read_text(path, *args, **kwargs):
        if path == source:
            raise PermissionError("Permission denied")
        return original_read_text(path, *args, **kwargs)

    with patch.object(type(source), "read_text", mocked_read_text):
        result = PythonASTParser().parse_file(source)

    assert result.files_processed == 1
    assert result.files_succeeded == 0
    assert result.files_failed == 1
    assert result.unit_count == 0
    assert result.errors[0].error_type == "PermissionError"
    assert "Permission denied" in result.errors[0].error


def test_repository_continues_after_unreadable_file(tmp_path):
    """An unreadable file must not prevent other files from being parsed."""
    valid = tmp_path / "valid.py"
    restricted = tmp_path / "restricted.py"

    valid.write_text("def valid():\n    return True\n", encoding="utf-8")
    restricted.write_text("x = 1\n", encoding="utf-8")

    original_read_text = type(valid).read_text

    def mocked_read_text(path, *args, **kwargs):
        if path == restricted:
            raise PermissionError("Permission denied")
        return original_read_text(path, *args, **kwargs)

    with patch.object(type(valid), "read_text", mocked_read_text):
        result = PythonASTParser().parse_repository(tmp_path)

    assert result.files_processed == 2
    assert result.files_succeeded == 1
    assert result.files_failed == 1
    assert result.error_count == 1

    assert any(unit.name == "valid" for unit in result.code_units)


def test_repository_aggregates_extraction_statistics(tmp_path):
    """Repository statistics should reflect all attempted Python files."""
    (tmp_path / "first.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "second.py").write_text(
        "def second():\n    return 2\n",
        encoding="utf-8",
    )
    (tmp_path / "broken.py").write_text(
        "def broken(:\n",
        encoding="utf-8",
    )

    result = PythonASTParser().parse_repository(tmp_path)

    assert result.files_processed == 3
    assert result.files_succeeded == 2
    assert result.files_failed == 1
    assert result.error_count == 1
    assert result.files_processed == (result.files_succeeded + result.files_failed)


def test_syntax_error_is_logged(tmp_path, caplog):
    """Parsing failures should be logged with the affected file path."""
    source = tmp_path / "broken.py"
    source.write_text("def broken(:\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        result = PythonASTParser().parse_file(source)

    assert result.error_count == 1
    assert "broken.py" in caplog.text
    assert "SyntaxError" in caplog.text


def test_repository_extraction_summary_is_logged(tmp_path, caplog):
    """Repository extraction should log its final statistics."""
    (tmp_path / "valid.py").write_text("x = 1\n", encoding="utf-8")

    with caplog.at_level(logging.INFO):
        result = PythonASTParser().parse_repository(tmp_path)

    assert result.files_processed == 1
    assert "Repository AST extraction completed" in caplog.text
    assert "1 processed" in caplog.text
    assert "1 succeeded" in caplog.text


def test_source_line_range_validation_rejects_out_of_bounds(tmp_path):
    """Source-line validation should reject ranges beyond file length."""
    source = tmp_path / "valid.py"
    source.write_text("x = 1\n", encoding="utf-8")

    parser = PythonASTParser()
    result = parser.parse_file(source)
    unit = result.code_units[0]

    with pytest.raises(ValueError, match="exceeds source file length"):
        parser._validate_source_line_range(unit, source_line_count=0)


def test_repository_recovers_from_unexpected_file_exception(tmp_path):
    """Unexpected per-file exceptions must not interrupt the repository."""
    valid = tmp_path / "valid.py"
    broken = tmp_path / "broken.py"

    valid.write_text("def valid():\n    return True\n", encoding="utf-8")
    broken.write_text("x = 1\n", encoding="utf-8")

    parser = PythonASTParser()
    original_parse_file = parser.parse_file

    def mocked_parse_file(file_path, repository_root=None):
        if file_path == broken:
            raise RuntimeError("Unexpected extraction failure")
        return original_parse_file(file_path, repository_root=repository_root)

    with patch.object(parser, "parse_file", side_effect=mocked_parse_file):
        result = parser.parse_repository(tmp_path)

    assert result.files_processed == 2
    assert result.files_succeeded == 1
    assert result.files_failed == 1
    assert result.error_count == 1
    assert result.errors[0].error_type == "RuntimeError"
    assert any(unit.name == "valid" for unit in result.code_units)
