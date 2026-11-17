"""Validate repository source citations against files and the code graph."""

from __future__ import annotations

import ast
from pathlib import Path, PureWindowsPath
from typing import ClassVar

from repolens.graph.code_graph import CodeGraph
from repolens.ingestion.code_unit import CodeUnit
from repolens.verification.citation_models import (
    Citation,
    CitationValidationResult,
    IssueSeverity,
    ValidationIssue,
)


class CitationValidator:
    """Verify citations against a repository snapshot."""

    _PYTHON_EXTENSIONS: ClassVar[set[str]] = {".py"}

    def __init__(
        self,
        repository_root: str | Path,
        code_graph: CodeGraph | None = None,
    ) -> None:
        root = Path(repository_root).expanduser().resolve()

        if not root.exists() or not root.is_dir():
            raise ValueError("Repository root must be an existing directory.")

        if code_graph is not None and not isinstance(code_graph, CodeGraph):
            raise TypeError("code_graph must be a CodeGraph instance or None.")

        self.repository_root = root
        self.code_graph = code_graph

    def validate(self, citation: Citation) -> CitationValidationResult:
        """Validate one citation against the repository snapshot."""
        if not isinstance(citation, Citation):
            raise TypeError("Expected a Citation instance.")

        issues: list[ValidationIssue] = []

        def issue(
            code: str,
            message: str,
            severity: IssueSeverity = "error",
        ) -> None:
            issues.append(
                ValidationIssue(
                    code=code,
                    message=message,
                    severity=severity,
                    citation_id=citation.citation_id,
                )
            )

        if not citation.file_path:
            issue(
                "missing_file_path",
                "Citation does not contain a file path.",
                severity="warning",
            )
            return CitationValidationResult(
                citation=citation,
                status="incomplete",
                issues=tuple(issues),
            )

        relative_path = self._safe_relative_path(citation.file_path)

        if relative_path is None:
            issue(
                "unsafe_file_path",
                "Citation path is absolute, traverses directories, or is invalid.",
            )
            return CitationValidationResult(
                citation=citation,
                status="invalid",
                file_exists=False,
                path_is_safe=False,
                issues=tuple(issues),
            )

        candidate = self.repository_root / relative_path

        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.repository_root)
        except (OSError, RuntimeError, ValueError):
            issue(
                "file_not_found_or_outside_repository",
                "Referenced file does not exist inside the repository snapshot.",
            )
            return CitationValidationResult(
                citation=citation,
                status="invalid",
                file_exists=False,
                path_is_safe=False,
                issues=tuple(issues),
            )

        if not resolved.is_file():
            issue(
                "not_a_file",
                "Referenced repository path is not a regular file.",
            )
            return CitationValidationResult(
                citation=citation,
                status="invalid",
                file_exists=False,
                path_is_safe=True,
                issues=tuple(issues),
            )

        try:
            source = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError:
            issue("file_unreadable", "Referenced file could not be read.")
            return CitationValidationResult(
                citation=citation,
                status="invalid",
                file_exists=True,
                path_is_safe=True,
                issues=tuple(issues),
            )

        lines = source.splitlines()
        line_range_valid: bool | None = None
        symbol_found: bool | None = None
        code_unit_found: bool | None = None

        if citation.start_line is None or citation.end_line is None:
            issue(
                "missing_line_range",
                "Citation does not contain a complete line range.",
                severity="warning",
            )
        elif citation.start_line < 1 or citation.end_line < citation.start_line:
            line_range_valid = False
            issue(
                "malformed_line_range",
                "Citation line range must start at 1 or later and end "
                "at or after its start.",
            )
        elif citation.end_line > len(lines):
            line_range_valid = False
            issue(
                "line_range_out_of_bounds",
                f"Cited line {citation.end_line} exceeds the file's "
                f"{len(lines)} lines.",
            )
        else:
            line_range_valid = True

        if citation.code_unit_id is not None:
            if self.code_graph is None:
                issue(
                    "code_unit_unverified",
                    "CodeUnit verification requires an available CodeGraph.",
                    severity="warning",
                )
            else:
                unit = self.code_graph.get_code_unit(citation.code_unit_id)
                code_unit_found = unit is not None

                if unit is None:
                    issue(
                        "code_unit_not_found",
                        "Referenced CodeUnit ID could not be verified.",
                    )
                elif not self._same_file(unit.file_path, citation.file_path):
                    code_unit_found = False
                    issue(
                        "code_unit_file_mismatch",
                        "Referenced CodeUnit belongs to a different file.",
                    )
                elif line_range_valid is True and (
                    citation.start_line < unit.start_line
                    or citation.end_line > unit.end_line
                ):
                    code_unit_found = False
                    issue(
                        "code_unit_line_mismatch",
                        "Cited line range falls outside the referenced CodeUnit.",
                    )

        if citation.symbol is not None:
            if resolved.suffix.casefold() not in self._PYTHON_EXTENSIONS:
                symbol_found = None
                issue(
                    "symbol_verification_unsupported",
                    f"Symbol verification is not supported for "
                    f"'{resolved.suffix or 'extensionless'}' files. "
                    "File and line-range checks were still performed.",
                    severity="warning",
                )
            elif line_range_valid is False:
                symbol_found = None
            else:
                symbol_found = self._symbol_in_source(
                    citation.symbol,
                    source,
                    citation.start_line,
                    citation.end_line,
                )

                if not symbol_found:
                    issue(
                        "symbol_not_found",
                        f"Referenced symbol '{citation.symbol}' was not found "
                        "in the cited source.",
                    )

        has_errors = any(item.severity == "error" for item in issues)
        has_unverified_checks = any(
            item.code
            in {
                "code_unit_unverified",
                "symbol_verification_unsupported",
            }
            for item in issues
        )

        if has_errors:
            status = "invalid"
        elif citation.start_line is None or citation.end_line is None:
            status = "incomplete"
        elif has_unverified_checks:
            status = "unverified"
        else:
            status = "valid"

        return CitationValidationResult(
            citation=citation,
            status=status,
            file_exists=True,
            path_is_safe=True,
            line_range_valid=line_range_valid,
            symbol_found=symbol_found,
            code_unit_found=code_unit_found,
            issues=tuple(issues),
        )

    def validate_many(
        self,
        citations: list[Citation],
    ) -> list[CitationValidationResult]:
        """Validate multiple citations independently."""
        return [self.validate(citation) for citation in citations]

    @staticmethod
    def _safe_relative_path(file_path: str) -> Path | None:
        """Reject absolute paths, traversal and invalid path components."""
        if not isinstance(file_path, str) or not file_path.strip():
            return None

        normalized = file_path.replace("\\", "/")

        if "\x00" in normalized:
            return None

        windows_path = PureWindowsPath(file_path)

        if Path(file_path).is_absolute() or windows_path.is_absolute():
            return None

        if windows_path.drive or any(part == ".." for part in normalized.split("/")):
            return None

        parts = [part for part in normalized.split("/") if part not in {"", "."}]

        if not parts:
            return None

        return Path(*parts)

    def _units_for_file(self, file_path: str) -> list[CodeUnit]:
        if self.code_graph is None:
            return []

        return [
            unit
            for unit in self.code_graph.get_code_units()
            if self._same_file(unit.file_path, file_path)
        ]

    @staticmethod
    def _same_file(first: str, second: str) -> bool:
        return (
            first.replace("\\", "/").casefold() == second.replace("\\", "/").casefold()
        )

    @staticmethod
    def _symbol_in_units(
        symbol: str,
        units: list[CodeUnit],
        start_line: int | None,
        end_line: int | None,
    ) -> bool:
        for unit in units:
            if symbol not in {unit.name, unit.qualified_name}:
                continue

            if start_line is not None and start_line < unit.start_line:
                continue

            if end_line is not None and end_line > unit.end_line:
                continue

            return True

        return False

    @staticmethod
    def _symbol_in_source(
        symbol: str,
        source: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> bool:
        """Verify a Python declaration in the actual cited source lines."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False

        target = symbol.rsplit(".", maxsplit=1)[-1]

        for node in ast.walk(tree):
            if not isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                continue

            if node.name != target:
                continue

            declaration_start = node.lineno
            declaration_end = getattr(node, "end_lineno", node.lineno)

            if start_line is not None and declaration_start < start_line:
                continue

            if end_line is not None and declaration_end > end_line:
                continue

            return True

        return False
