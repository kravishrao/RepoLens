from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from repolens.ingestion.code_unit import (
    CallReference,
    CodeUnit,
    ImportReference,
)
from repolens.ingestion.file_discovery import PythonFileDiscovery


@dataclass(frozen=True)
class ASTParseError:
    """Record a source file that could not be parsed."""

    file_path: str
    error: str
    line_number: int | None = None


@dataclass
class ASTParseResult:
    """Store extracted CodeUnits and parsing errors."""

    code_units: list[CodeUnit] = field(default_factory=list)
    errors: list[ASTParseError] = field(default_factory=list)

    @property
    def unit_count(self) -> int:
        return len(self.code_units)

    @property
    def error_count(self) -> int:
        return len(self.errors)


def _make_id(
    file_path: str,
    qualified_name: str,
    kind: str,
    start_line: int,
) -> str:
    """Generate a deterministic identifier for a code element."""
    value = f"{file_path}:{kind}:{qualified_name}:{start_line}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _attribute_name(node: ast.AST) -> str | None:
    """Resolve a simple dotted name from an AST expression."""
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        parent = _attribute_name(node.value)
        return f"{parent}.{node.attr}" if parent else None

    return None


class _CallCollector(ast.NodeVisitor):
    """Collect calls belonging to one function or method."""

    def __init__(self) -> None:
        self.calls: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        target = _attribute_name(node.func)

        if target:
            self.calls.add(target)

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Nested functions are extracted independently.
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Methods and calls inside nested classes are extracted separately.
        return


class _ModuleImportCollector(ast.NodeVisitor):
    """Collect imports belonging to the module scope."""

    def __init__(self) -> None:
        self.imports: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        prefix = "." * node.level + (node.module or "")
        self.imports.add(prefix)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


class PythonASTParser:
    """Parse Python files and extract structured CodeUnits."""

    def __init__(
        self,
        file_discovery: PythonFileDiscovery | None = None,
    ) -> None:
        self.file_discovery = (
            file_discovery if file_discovery is not None else PythonFileDiscovery()
        )

    @staticmethod
    def _module_name(relative_path: Path) -> str:
        """Convert a relative Python file path into a module name."""
        parts = list(relative_path.with_suffix("").parts)

        if parts and parts[-1] == "__init__":
            parts.pop()

        return ".".join(parts) or relative_path.stem

    def parse_file(
        self,
        file_path: str | Path,
        repository_root: str | Path | None = None,
    ) -> ASTParseResult:
        """Parse one Python file and extract its code elements."""
        path = Path(file_path).resolve()

        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Python source file not found: {path}")

        root = (
            Path(repository_root).resolve()
            if repository_root is not None
            else path.parent
        )

        try:
            relative_path = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(
                "Python source file must be inside the repository root."
            ) from exc

        result = ASTParseResult()

        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=relative_path)
        except (SyntaxError, UnicodeDecodeError) as exc:
            result.errors.append(
                ASTParseError(
                    file_path=relative_path,
                    error=str(exc),
                    line_number=getattr(exc, "lineno", None),
                )
            )
            return result

        module_name = self._module_name(Path(relative_path))
        source_lines = source.splitlines()

        module_end = max(1, len(source_lines))
        module_id = _make_id(
            relative_path,
            module_name,
            "module",
            1,
        )

        import_collector = _ModuleImportCollector()

        for statement in tree.body:
            import_collector.visit(statement)

        module_unit = CodeUnit(
            id=module_id,
            kind="module",
            name=Path(relative_path).name,
            qualified_name=module_name,
            file_path=relative_path,
            start_line=1,
            end_line=module_end,
            imports=tuple(
                ImportReference(module=name)
                for name in sorted(import_collector.imports)
            ),
        )

        result.code_units.append(module_unit)

        def extract(
            node: ast.AST,
            parent_id: str,
            parent_name: str,
            inside_class: bool = False,
        ) -> None:
            """Recursively extract classes and callable definitions."""

            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    kind = "class"
                    name = child.name
                    is_class = True

                elif isinstance(
                    child,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    kind = "method" if inside_class else "function"
                    name = child.name
                    is_class = False

                else:
                    # Traverse intermediate nodes such as If, Try,
                    # With, For and While to discover nested definitions.
                    extract(
                        child,
                        parent_id,
                        parent_name,
                        inside_class=inside_class,
                    )
                    continue

                qualified_name = f"{parent_name}.{name}"

                start_line = min(
                    [child.lineno]
                    + [
                        decorator.lineno
                        for decorator in getattr(
                            child,
                            "decorator_list",
                            [],
                        )
                    ]
                )

                unit_id = _make_id(
                    relative_path,
                    qualified_name,
                    kind,
                    start_line,
                )

                call_collector = _CallCollector()

                if not is_class:
                    for statement in child.body:
                        call_collector.visit(statement)

                calls = tuple(
                    CallReference(target=target)
                    for target in sorted(call_collector.calls)
                )

                end_line = child.end_lineno or child.lineno

                unit = CodeUnit(
                    id=unit_id,
                    kind=kind,
                    name=name,
                    qualified_name=qualified_name,
                    file_path=relative_path,
                    start_line=start_line,
                    end_line=end_line,
                    parent_id=parent_id,
                    calls=calls,
                )

                result.code_units.append(unit)

                extract(
                    child,
                    unit_id,
                    qualified_name,
                    inside_class=is_class,
                )

        extract(tree, module_id, module_name)

        return result

    def parse_repository(
        self,
        repository_path: str | Path,
    ) -> ASTParseResult:
        """Parse every discovered Python file in a repository."""
        root = Path(repository_path).expanduser().resolve()
        discovery = self.file_discovery.discover(root)

        combined_result = ASTParseResult()

        for relative_path in discovery.files:
            file_result = self.parse_file(
                root / relative_path,
                repository_root=root,
            )

            combined_result.code_units.extend(file_result.code_units)
            combined_result.errors.extend(file_result.errors)

        return combined_result


def parse_repository(
    repository_path: str | Path,
) -> ASTParseResult:
    """Convenience function for repository-wide AST extraction."""
    return PythonASTParser().parse_repository(repository_path)
