"""Structured models for citation and claim validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

ValidationStatus = Literal["valid", "invalid", "incomplete", "unverified"]

IssueSeverity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class ValidationIssue:
    """A specific issue discovered during citation verification."""

    code: str
    message: str
    severity: IssueSeverity = "error"
    citation_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class Citation:
    """A source citation extracted from a generated answer."""

    citation_id: str
    raw_text: str
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    symbol: str | None = None
    code_unit_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class CitationValidationResult:
    """Validation outcome for one extracted citation."""

    citation: Citation
    status: ValidationStatus
    file_exists: bool | None = None
    path_is_safe: bool | None = None
    line_range_valid: bool | None = None
    symbol_found: bool | None = None
    code_unit_found: bool | None = None
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "citation": self.citation.to_dict(),
            "status": self.status,
            "file_exists": self.file_exists,
            "path_is_safe": self.path_is_safe,
            "line_range_valid": self.line_range_valid,
            "symbol_found": self.symbol_found,
            "code_unit_found": self.code_unit_found,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class ClaimValidationResult:
    """Assessment of whether repository evidence supports a claim."""

    claim: str
    status: ValidationStatus
    supporting_evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "claim": self.claim,
            "status": self.status,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "issues": [issue.to_dict() for issue in self.issues],
        }
