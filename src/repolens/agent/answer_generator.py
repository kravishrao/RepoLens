"""Repository-grounded answer generation component."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from repolens.agent.state import InvestigationState
from repolens.ingestion.code_unit import CodeUnit

logger = logging.getLogger(__name__)


def format_line_reference(start_line: int | None, end_line: int | None) -> str:
    """Format start and end lines into a human-readable range string."""
    if start_line is not None and end_line is not None:
        if start_line == end_line:
            return f"Line {start_line}"
        return f"Lines {start_line}-{end_line}"
    if start_line is not None:
        return f"Line {start_line}+"
    return "Lines unknown"


class AnswerGenerationError(Exception):
    """Raised when answer generation fails in strict mode."""


class GenerationProvider(Protocol):
    """Protocol for pluggable LLM or text generation providers."""

    def generate(self, prompt: str) -> str:
        """Generate response text given a prompt."""
        ...


class EvidenceStatus(StrEnum):
    """Classification of evidence completeness and certainty."""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    CONFLICTING = "conflicting"
    UNCERTAIN = "uncertain"


class SupportType(StrEnum):
    """Nature of evidence grounding for a finding or claim."""

    DIRECT = "direct"
    INFERRED = "inferred"
    CONFLICTING = "conflicting"


@dataclass(frozen=True)
class CodeUnitAssociation:
    """Traceable association between an answer component and supporting CodeUnit."""

    code_unit_id: str
    name: str
    qualified_name: str
    kind: str
    file_path: str
    start_line: int | None
    end_line: int | None
    line_reference: str
    support_type: SupportType
    source: str
    content_snippet: str | None = None
    docstring: str | None = None
    calls: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    relevance_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert association to a dictionary."""
        return {
            "code_unit_id": self.code_unit_id,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "kind": self.kind,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "line_reference": self.line_reference,
            "support_type": self.support_type.value,
            "source": self.source,
            "content_snippet": self.content_snippet,
            "docstring": self.docstring,
            "calls": list(self.calls),
            "imports": list(self.imports),
            "relevance_score": self.relevance_score,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CodeUnitAssociation:
        """Instantiate association from a dictionary."""
        raw_support = str(data.get("support_type", "direct")).lower()
        try:
            support_type = SupportType(raw_support)
        except ValueError:
            support_type = SupportType.DIRECT

        start = data.get("start_line")
        end = data.get("end_line")
        start_line = (
            int(start)
            if isinstance(start, (int, str)) and str(start).isdigit()
            else None
        )
        end_line = (
            int(end) if isinstance(end, (int, str)) and str(end).isdigit() else None
        )

        line_ref = str(data.get("line_reference") or "")
        if not line_ref:
            line_ref = format_line_reference(start_line, end_line)

        score = data.get("relevance_score")
        score_val = float(score) if isinstance(score, (int, float)) else None

        return cls(
            code_unit_id=str(data.get("code_unit_id", "unknown")),
            name=str(data.get("name", "Unknown")),
            qualified_name=str(data.get("qualified_name", data.get("name", "Unknown"))),
            kind=str(data.get("kind", "code_element")),
            file_path=str(data.get("file_path", "unknown_file.py")),
            start_line=start_line,
            end_line=end_line,
            line_reference=line_ref,
            support_type=support_type,
            source=str(data.get("source", "retrieval")),
            content_snippet=(
                str(data["content_snippet"]) if data.get("content_snippet") else None
            ),
            docstring=str(data["docstring"]) if data.get("docstring") else None,
            calls=[str(c) for c in (data.get("calls") or [])],
            imports=[str(i) for i in (data.get("imports") or [])],
            relevance_score=score_val,
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ConflictingEvidenceRecord:
    """Represents a detected conflict between multiple evidence sources."""

    symbol: str
    conflict_type: str
    description: str
    conflicting_units: list[CodeUnitAssociation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert conflict record to dictionary."""
        return {
            "symbol": self.symbol,
            "conflict_type": self.conflict_type,
            "description": self.description,
            "conflicting_units": [u.to_dict() for u in self.conflicting_units],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConflictingEvidenceRecord:
        """Instantiate conflict record from dictionary."""
        units = [
            CodeUnitAssociation.from_dict(u)
            for u in (data.get("conflicting_units") or [])
        ]
        return cls(
            symbol=str(data.get("symbol", "unknown")),
            conflict_type=str(data.get("conflict_type", "conflict")),
            description=str(data.get("description", "")),
            conflicting_units=units,
        )


@dataclass(frozen=True)
class EvidenceAssessment:
    """Structured evaluation of evidence sufficiency, conflicts, and gaps."""

    status: EvidenceStatus
    is_sufficient: bool
    is_conflicting: bool
    is_uncertain: bool
    sufficiency_score: float
    direct_findings_count: int
    inferred_findings_count: int
    conflicts: list[ConflictingEvidenceRecord] = field(default_factory=list)
    missing_aspects: list[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert assessment to dictionary."""
        return {
            "status": self.status.value,
            "is_sufficient": self.is_sufficient,
            "is_conflicting": self.is_conflicting,
            "is_uncertain": self.is_uncertain,
            "sufficiency_score": self.sufficiency_score,
            "direct_findings_count": self.direct_findings_count,
            "inferred_findings_count": self.inferred_findings_count,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "missing_aspects": list(self.missing_aspects),
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceAssessment:
        """Instantiate assessment from dictionary."""
        raw_status = str(data.get("status", "sufficient")).lower()
        try:
            status = EvidenceStatus(raw_status)
        except ValueError:
            status = EvidenceStatus.SUFFICIENT

        conflicts = [
            ConflictingEvidenceRecord.from_dict(c)
            for c in (data.get("conflicts") or [])
        ]
        return cls(
            status=status,
            is_sufficient=bool(data.get("is_sufficient", True)),
            is_conflicting=bool(data.get("is_conflicting", False)),
            is_uncertain=bool(data.get("is_uncertain", False)),
            sufficiency_score=float(data.get("sufficiency_score", 1.0)),
            direct_findings_count=int(data.get("direct_findings_count", 0)),
            inferred_findings_count=int(data.get("inferred_findings_count", 0)),
            conflicts=conflicts,
            missing_aspects=[str(m) for m in (data.get("missing_aspects") or [])],
            explanation=str(data.get("explanation", "")),
        )


@dataclass(frozen=True)
class GroundedAnswer:
    """Structured result of grounded answer generation."""

    content: str
    query: str
    supported_findings: list[dict[str, Any]] = field(default_factory=list)
    inferred_information: list[dict[str, Any]] = field(default_factory=list)
    unsupported_notes: list[str] = field(default_factory=list)
    is_grounded: bool = True
    has_evidence: bool = True
    evidence_status: EvidenceStatus = EvidenceStatus.SUFFICIENT
    associations: list[CodeUnitAssociation] = field(default_factory=list)
    assessment: EvidenceAssessment | None = None
    conflicting_evidence: list[ConflictingEvidenceRecord] = field(default_factory=list)
    traceability_map: dict[str, list[str]] = field(default_factory=dict)


def _normalize_evidence_item(
    item: Mapping[str, Any] | CodeUnit | Any,
) -> dict[str, Any]:
    """Convert CodeUnit or evidence mapping to a standardized dictionary."""
    if isinstance(item, CodeUnit):
        return {
            "evidence_id": f"code_unit:{item.id}",
            "code_unit_id": item.id,
            "name": item.name,
            "qualified_name": item.qualified_name,
            "kind": item.kind,
            "file_path": item.file_path,
            "start_line": item.start_line,
            "end_line": item.end_line,
            "source": "code_unit",
            "content": item.source_code or "",
            "metadata": {
                "docstring": item.docstring,
                "calls": [c.target for c in item.calls],
                "imports": [i.module for i in item.imports],
                **item.metadata,
            },
        }

    if isinstance(item, Mapping):
        return dict(item)

    if hasattr(item, "code_unit"):
        unit = item.code_unit
        if isinstance(unit, CodeUnit):
            base = _normalize_evidence_item(unit)
            if hasattr(item, "score"):
                base["score"] = item.score
            return base

    raise TypeError(
        f"Unsupported evidence item type '{type(item).__name__}'. "
        "Expected Mapping, CodeUnit, or object with 'code_unit' attribute."
    )


def construct_prompt(
    query: str,
    evidence: Sequence[Mapping[str, Any] | CodeUnit | Any],
    *,
    errors: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Construct a grounded prompt using relevant CodeUnits and source context."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Question cannot be empty.")

    normalized_evidence = [_normalize_evidence_item(item) for item in evidence]

    # Group evidence by code_unit_id
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in normalized_evidence:
        uid = str(item.get("code_unit_id", "unknown"))
        grouped.setdefault(uid, []).append(item)

    prompt_lines = [
        "You are RepoLens, an expert repository investigation and code-grounding assistant.",
        "Your task is to answer the user's question using ONLY the provided repository evidence.",
        "",
        "CRITICAL GROUNDING RULES:",
        "1. Ground every claim directly in the evidence. Never speculate or fabricate file paths,",
        "   symbol names, line numbers, or implementation details.",
        "2. Include exact file paths, symbols, and line references for all cited code elements.",
        "3. Format each cited code element with:",
        "      <index>. <qualified_name_or_symbol>",
        "         Type: <kind>",
        "         File: <file_path>",
        "         Lines: <start_line>-<end_line>",
        "4. Explicitly distinguish between:",
        "   - Supported Findings: verified declarations, signatures, and inspected source code.",
        "   - Inferred Information: relationships, callers, or callee targets not fully inspected.",
        "5. If sufficient evidence is unavailable to answer any aspect of the question,",
        "   explicitly state that evidence is lacking and avoid unsupported claims.",
        "",
        f"User Question: {query}",
        "",
        f"Repository Evidence ({len(grouped)} unique code elements, {len(normalized_evidence)} total records):",
        "================================================================================",
    ]

    for index, (uid, records) in enumerate(grouped.items(), start=1):
        primary = records[0]
        name = primary.get("qualified_name") or primary.get("name") or "Unknown"
        kind = primary.get("kind", "unknown")
        file_path = primary.get("file_path", "unknown")
        start_line = primary.get("start_line", "?")
        end_line = primary.get("end_line", "?")

        sources = list(dict.fromkeys(str(r.get("source", "unknown")) for r in records))

        prompt_lines.extend(
            [
                "",
                f"[{index}] Symbol: {name}",
                f"    CodeUnit ID: {uid}",
                f"    Kind: {kind}",
                f"    File: {file_path}:{start_line}-{end_line}",
                f"    Evidence Sources: {', '.join(sources)}",
            ]
        )

        # Source code content
        contents = [
            str(r.get("content", "")).strip()
            for r in records
            if r.get("content") and str(r.get("content", "")).strip()
        ]
        if contents:
            prompt_lines.append("    Source Code:")
            for line in contents[0].splitlines()[:30]:
                prompt_lines.append(f"        {line}")

        # Metadata (docstring, calls, imports)
        for r in records:
            meta = r.get("metadata", {})
            if isinstance(meta, dict):
                if meta.get("docstring"):
                    prompt_lines.append(f"    Docstring: {meta['docstring']}")
                if meta.get("calls"):
                    calls = meta["calls"]
                    if isinstance(calls, list) and calls:
                        prompt_lines.append(
                            f"    Calls: {', '.join(str(c) for c in calls)}"
                        )
                if meta.get("imports"):
                    imports = meta["imports"]
                    if isinstance(imports, list) and imports:
                        prompt_lines.append(
                            f"    Imports: {', '.join(str(i) for i in imports)}"
                        )
                break

    if errors:
        prompt_lines.extend(
            [
                "",
                "Investigation Warnings / Errors Encountered:",
                "--------------------------------------------",
            ]
        )
        for err in errors:
            prompt_lines.append(
                f"- {err.get('tool', 'unknown')}: {err.get('error', 'Unknown error')}"
            )

    prompt_lines.extend(
        [
            "",
            "Instructions for Response Formatting:",
            "- Begin with 'Repository Investigation' summary.",
            "- Provide 'Supported Findings' with citations (file, symbol, lines).",
            "- Provide 'Inferred / Uncertain Information' noting any unverified aspects.",
            "- Provide 'Evidence Limitations' noting questions that cannot be conclusively answered.",
        ]
    )

    return "\n".join(prompt_lines)


class AnswerGenerator:
    """Generates accurate, repository-grounded responses from investigation evidence."""

    def __init__(
        self,
        provider: GenerationProvider | None = None,
        *,
        include_trace: bool = True,
        max_source_lines: int = 20,
    ) -> None:
        self.provider = provider
        self.include_trace = include_trace
        self.max_source_lines = max_source_lines

    def construct_prompt(
        self,
        query: str,
        evidence: Sequence[Mapping[str, Any] | CodeUnit | Any],
        *,
        errors: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        """Construct a grounded prompt using relevant CodeUnits and source context."""
        return construct_prompt(query, evidence, errors=errors)

    def build_associations(
        self,
        evidence: Sequence[Mapping[str, Any] | CodeUnit | Any],
    ) -> list[CodeUnitAssociation]:
        """Convert evidence items into structured, traceable CodeUnitAssociations."""
        normalized_evidence = [_normalize_evidence_item(item) for item in evidence]
        associations: list[CodeUnitAssociation] = []
        for item in normalized_evidence:
            uid = str(item.get("code_unit_id") or item.get("evidence_id") or "unknown")
            name = str(item.get("name") or "Unknown")
            qualified_name = str(item.get("qualified_name") or name)
            kind = str(item.get("kind") or "code_element")
            file_path = str(item.get("file_path") or "unknown_file.py")

            start = item.get("start_line")
            end = item.get("end_line")
            start_line = (
                int(start)
                if isinstance(start, (int, str)) and str(start).isdigit()
                else None
            )
            end_line = (
                int(end) if isinstance(end, (int, str)) and str(end).isdigit() else None
            )
            line_ref = format_line_reference(start_line, end_line)

            source = str(item.get("source") or "retrieval")
            content = str(item.get("content") or "").strip()

            # Determine direct vs inferred support
            is_direct = False
            if source in ("inspect_source", "source_inspection", "read_file") or (
                content
                and len(content.splitlines()) >= 1
                and source
                not in (
                    "context_expansion",
                    "graph_query",
                    "graph_traversal",
                    "callers",
                    "callees",
                    "dependencies",
                )
            ):
                is_direct = True

            support_type = SupportType.DIRECT if is_direct else SupportType.INFERRED

            snippet = None
            if content:
                snippet_lines = content.splitlines()[: self.max_source_lines]
                snippet = "\n".join(snippet_lines)

            meta = dict(item.get("metadata") or {})
            docstring = meta.get("docstring") or item.get("docstring")
            raw_calls = meta.get("calls") or item.get("calls") or []
            calls = [str(c) for c in raw_calls] if isinstance(raw_calls, list) else []
            raw_imports = meta.get("imports") or item.get("imports") or []
            imports = (
                [str(i) for i in raw_imports] if isinstance(raw_imports, list) else []
            )

            score = item.get("score")
            score_val = float(score) if isinstance(score, (int, float)) else None

            associations.append(
                CodeUnitAssociation(
                    code_unit_id=uid,
                    name=name,
                    qualified_name=qualified_name,
                    kind=kind,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    line_reference=line_ref,
                    support_type=support_type,
                    source=source,
                    content_snippet=snippet,
                    docstring=str(docstring) if docstring else None,
                    calls=calls,
                    imports=imports,
                    relevance_score=score_val,
                    metadata=meta,
                )
            )
        return associations

    def detect_conflicts(
        self,
        associations: list[CodeUnitAssociation],
    ) -> list[ConflictingEvidenceRecord]:
        """Identify contradictory or conflicting evidence across code unit associations."""
        conflicts: list[ConflictingEvidenceRecord] = []
        by_symbol: dict[str, list[CodeUnitAssociation]] = {}

        for assoc in associations:
            sym = assoc.qualified_name or assoc.name
            by_symbol.setdefault(sym, []).append(assoc)

        for sym, units in by_symbol.items():
            if len(units) < 2:
                continue

            files = {u.file_path for u in units}
            if len(files) > 1:
                # Check for kind mismatch
                kinds = {u.kind for u in units if u.kind != "code_element"}
                if len(kinds) > 1:
                    conflicts.append(
                        ConflictingEvidenceRecord(
                            symbol=sym,
                            conflict_type="symbol_kind_mismatch",
                            description=(
                                f"Symbol '{sym}' is declared with conflicting types "
                                f"({', '.join(sorted(kinds))}) across files: "
                                + ", ".join(
                                    f"{u.file_path}:{u.line_reference} ({u.kind})"
                                    for u in units
                                )
                            ),
                            conflicting_units=units,
                        )
                    )
                    continue

                # Check for signature/parameter mismatch in declarations
                signatures: dict[str, CodeUnitAssociation] = {}
                for u in units:
                    if u.content_snippet and (
                        "def " in u.content_snippet or "class " in u.content_snippet
                    ):
                        first_line = u.content_snippet.splitlines()[0].strip()
                        norm_sig = re.sub(r"\s+", " ", first_line)
                        signatures[norm_sig] = u

                if len(signatures) > 1:
                    conflicts.append(
                        ConflictingEvidenceRecord(
                            symbol=sym,
                            conflict_type="signature_mismatch",
                            description=(
                                f"Conflicting signatures or declarations for '{sym}' across files: "
                                + "; ".join(
                                    f"'{sig}' in {u.file_path}:{u.line_reference}"
                                    for sig, u in signatures.items()
                                )
                            ),
                            conflicting_units=units,
                        )
                    )
                    continue

                # Check for explicit conflict flags in metadata
                meta_conflict = any(u.metadata.get("is_conflicting") for u in units)
                if meta_conflict:
                    conflicts.append(
                        ConflictingEvidenceRecord(
                            symbol=sym,
                            conflict_type="flagged_conflict",
                            description=(
                                f"Conflicting definitions or implementations flagged for '{sym}' across "
                                + ", ".join(
                                    f"{u.file_path}:{u.line_reference}" for u in units
                                )
                            ),
                            conflicting_units=units,
                        )
                    )

        return conflicts

    def assess_evidence(
        self,
        query: str,
        normalized_evidence: list[dict[str, Any]],
        associations: list[CodeUnitAssociation] | None = None,
    ) -> EvidenceAssessment:
        """Evaluate evidence sufficiency, conflicts, and gaps relative to the query."""
        if associations is None:
            associations = self.build_associations(normalized_evidence)

        if not associations:
            return EvidenceAssessment(
                status=EvidenceStatus.INSUFFICIENT,
                is_sufficient=False,
                is_conflicting=False,
                is_uncertain=False,
                sufficiency_score=0.0,
                direct_findings_count=0,
                inferred_findings_count=0,
                conflicts=[],
                missing_aspects=[
                    "No code units or repository evidence records were retrieved for this query."
                ],
                explanation="The repository search returned zero matching code units or source files.",
            )

        conflicts = self.detect_conflicts(associations)
        direct_count = sum(
            1 for a in associations if a.support_type == SupportType.DIRECT
        )
        inferred_count = sum(
            1 for a in associations if a.support_type == SupportType.INFERRED
        )

        # Look for explicit code identifiers in query (snake_case, CamelCase, backticked, or calls)
        code_symbol_patterns = re.findall(
            r"`([a-zA-Z_][a-zA-Z0-9_]*)`|([a-zA-Z_][a-zA-Z0-9_]*_[a-zA-Z0-9_]+)|([a-z]+[A-Z][a-zA-Z0-9]+)|([a-zA-Z_][a-zA-Z0-9_]*)\(\)",
            query,
        )
        explicit_symbols: list[str] = []
        for match_group in code_symbol_patterns:
            for item in match_group:
                if (
                    item
                    and len(item) > 2
                    and item.lower()
                    not in {
                        "where",
                        "what",
                        "which",
                        "when",
                        "how",
                        "return",
                        "class",
                        "function",
                    }
                ):
                    explicit_symbols.append(item)

        missing_aspects: list[str] = []
        if explicit_symbols:
            evidence_text = " ".join(
                f"{a.name} {a.qualified_name} {a.file_path} {a.content_snippet or ''} {a.docstring or ''}"
                for a in associations
            ).lower()

            for sym in explicit_symbols:
                if sym.lower() not in evidence_text:
                    missing_aspects.append(
                        f"No definitions, implementations, or references matching requested symbol '{sym}' were discovered."
                    )

        # Determine status
        if conflicts:
            status = EvidenceStatus.CONFLICTING
            is_sufficient = False
            is_conflicting = True
            is_uncertain = True
            sufficiency_score = 0.5
            explanation = f"Conflicting evidence was detected across {len(conflicts)} code element(s)."
        elif missing_aspects and len(missing_aspects) == len(explicit_symbols):
            status = EvidenceStatus.INSUFFICIENT
            is_sufficient = False
            is_conflicting = False
            is_uncertain = True
            sufficiency_score = 0.1
            explanation = "Available repository evidence does not contain the code elements requested by the question."
        elif direct_count == 0 and inferred_count > 0:
            status = EvidenceStatus.UNCERTAIN
            is_sufficient = True
            is_conflicting = False
            is_uncertain = True
            sufficiency_score = 0.5
            explanation = (
                "Findings are derived exclusively from call graph or import relationships "
                "without direct source code inspection."
            )
        else:
            status = EvidenceStatus.SUFFICIENT
            is_sufficient = True
            is_conflicting = False
            is_uncertain = False
            sufficiency_score = round(direct_count / max(1, len(associations)), 2)
            explanation = "Direct repository evidence exists supporting the investigation findings."

        return EvidenceAssessment(
            status=status,
            is_sufficient=is_sufficient,
            is_conflicting=is_conflicting,
            is_uncertain=is_uncertain,
            sufficiency_score=sufficiency_score,
            direct_findings_count=direct_count,
            inferred_findings_count=inferred_count,
            conflicts=conflicts,
            missing_aspects=missing_aspects,
            explanation=explanation,
        )

    def build_traceability_map(
        self,
        associations: list[CodeUnitAssociation],
    ) -> dict[str, list[str]]:
        """Construct bidirectional traceability mapping from symbols to supporting CodeUnit IDs."""
        trace_map: dict[str, list[str]] = {}
        for assoc in associations:
            sym = assoc.qualified_name or assoc.name
            trace_map.setdefault(sym, []).append(assoc.code_unit_id)
            trace_map.setdefault(assoc.name, []).append(assoc.code_unit_id)
            trace_map.setdefault(assoc.file_path, []).append(assoc.code_unit_id)

        return {k: list(dict.fromkeys(v)) for k, v in trace_map.items()}

    def _generate_deterministic_grounded_answer(
        self,
        query: str,
        normalized_evidence: list[dict[str, Any]],
        *,
        trace: list[dict[str, Any]] | None = None,
        errors: list[dict[str, Any]] | None = None,
    ) -> GroundedAnswer:
        """Synthesize a structured, evidence-grounded answer deterministically."""
        associations = self.build_associations(normalized_evidence)
        assessment = self.assess_evidence(query, normalized_evidence, associations)
        traceability_map = self.build_traceability_map(associations)

        if not normalized_evidence:
            answer_parts = [
                "No relevant code evidence was found for this repository question.\n\n",
                "### Evidence Assessment: Insufficient Evidence\n",
                "The available repository evidence is insufficient to answer this question.\n\n",
                "**Reason:**\n",
                f"{assessment.explanation}\n\n",
                "**Missing Information:**\n",
                "- No code units or repository evidence records were retrieved for this query.\n\n",
                "**Suggested Search Refinements:**\n",
                "- Verify that the target symbols, classes, or files exist in the repository.\n",
                "- Try searching for related function names, classes, or module paths.\n",
                "- Ensure the repository index is populated and valid .py files are present.\n",
            ]

            if errors:
                answer_parts.append("\nInvestigation errors:\n")
                for err in errors:
                    answer_parts.append(
                        f"- {err.get('tool', 'unknown')}: {err.get('error', 'Unknown error')}\n"
                    )

            lines = "".join(answer_parts).splitlines()
            if self.include_trace and trace:
                self._append_trace(lines, trace)

            return GroundedAnswer(
                content="\n".join(lines),
                query=query,
                supported_findings=[],
                inferred_information=[],
                unsupported_notes=[
                    "No repository evidence was retrieved to support any findings."
                ],
                is_grounded=False,
                has_evidence=False,
                evidence_status=EvidenceStatus.INSUFFICIENT,
                associations=[],
                assessment=assessment,
                conflicting_evidence=[],
                traceability_map={},
            )

        # If evidence was found, but assessed as insufficient to answer the query
        if assessment.status == EvidenceStatus.INSUFFICIENT:
            lines = [
                "Repository Investigation: Insufficient Evidence",
                "==============================================",
                "",
                f"Question: {query}",
                "Evidence Status: Insufficient (Cannot Answer Conclusively)",
                "",
                "The available repository evidence is insufficient to answer this question.",
                "",
                "Reason:",
                f"{assessment.explanation}",
                "",
                "Missing Information:",
            ]
            for missing in assessment.missing_aspects:
                lines.append(f"- {missing}")

            lines.extend(
                [
                    "",
                    "Retrieved Partial Evidence (Not Answering Question):",
                ]
            )
            for idx, a in enumerate(associations, start=1):
                lines.append(
                    f"{idx}. {a.qualified_name} ({a.kind}) in {a.file_path}:{a.line_reference}"
                )

            lines.extend(
                [
                    "",
                    "Suggested Search Refinements:",
                    "- Verify that the target symbols exist in the repository snapshot.",
                    "- Try broader search keywords or check package dependencies.",
                    "- Expand the search scope or inspect related modules.",
                ]
            )

            if errors:
                lines.extend(
                    [
                        "",
                        "Investigation warnings",
                        "----------------------",
                    ]
                )
                for err in errors:
                    lines.append(
                        f"- {err.get('tool', 'unknown')}: {err.get('error', 'Unknown error')}"
                    )

            if self.include_trace and trace:
                self._append_trace(lines, trace)

            return GroundedAnswer(
                content="\n".join(lines),
                query=query,
                supported_findings=[],
                inferred_information=[a.to_dict() for a in associations],
                unsupported_notes=assessment.missing_aspects,
                is_grounded=False,
                has_evidence=True,
                evidence_status=EvidenceStatus.INSUFFICIENT,
                associations=associations,
                assessment=assessment,
                conflicting_evidence=assessment.conflicts,
                traceability_map=traceability_map,
            )

        # Group by code unit ID
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in normalized_evidence:
            uid = str(item.get("code_unit_id", "unknown"))
            grouped.setdefault(uid, []).append(item)

        supported: list[dict[str, Any]] = []
        inferred: list[dict[str, Any]] = []
        unsupported_notes: list[str] = []

        lines: list[str] = [
            "Repository Investigation",
            "========================",
            "",
            f"Question: {query}",
            "",
            f"Unique code elements investigated: {len(grouped)}",
            f"Total evidence records: {len(normalized_evidence)}",
            f"Evidence Status: {assessment.status.value.upper()}",
            "",
            "Findings",
            "--------",
        ]

        # If conflicting evidence was detected, display a prominent warning block
        if assessment.conflicts:
            lines.extend(
                [
                    "",
                    "Conflicting Evidence Identified",
                    "-----------------------------------",
                ]
            )
            for conf in assessment.conflicts:
                lines.append(f"- Conflict in '{conf.symbol}' ({conf.conflict_type}):")
                lines.append(f"  {conf.description}")
            lines.append("")

        for index, (uid, records) in enumerate(grouped.items(), start=1):
            primary = records[0]
            name = primary.get("qualified_name") or primary.get("name") or "Unknown"
            kind = primary.get("kind", "unknown")
            file_path = primary.get("file_path", "unknown")
            start_line = primary.get("start_line", "?")
            end_line = primary.get("end_line", "?")

            lines.extend(
                [
                    "",
                    f"{index}. {name}",
                    f"   Type: {kind}",
                    f"   File: {file_path}",
                    f"   Lines: {start_line}-{end_line}",
                ]
            )

            sources = list(
                dict.fromkeys(str(r.get("source", "unknown")) for r in records)
            )
            lines.append(f"   Evidence sources: {', '.join(sources)}")

            # Extract source content
            contents = list(
                dict.fromkeys(
                    str(r.get("content", "")).strip()
                    for r in records
                    if r.get("content") and str(r.get("content", "")).strip()
                )
            )

            has_source_code = False
            for content in contents[:1]:
                has_source_code = True
                lines.append("   Source evidence:")
                for source_line in content.splitlines()[: self.max_source_lines]:
                    lines.append(f"      {source_line}")

            # Extract docstring and metadata
            docstring_found = False
            calls_list: list[str] = []
            for r in records:
                meta = r.get("metadata", {})
                if isinstance(meta, dict):
                    docstring = meta.get("docstring")
                    if docstring and not docstring_found:
                        lines.append(f"   Docstring: {docstring}")
                        docstring_found = True
                    if meta.get("calls"):
                        raw_calls = meta["calls"]
                        if isinstance(raw_calls, list):
                            calls_list.extend(str(c) for c in raw_calls)

            finding_info = {
                "code_unit_id": uid,
                "name": name,
                "kind": kind,
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "sources": sources,
                "has_source_code": has_source_code,
            }

            if has_source_code or "inspect_source" in sources:
                supported.append(finding_info)
            else:
                inferred.append(finding_info)

            if calls_list:
                calls_unique = list(dict.fromkeys(calls_list))
                inferred.append(
                    {
                        "code_unit_id": uid,
                        "type": "call_relationship",
                        "caller": name,
                        "callees": calls_unique[:5],
                    }
                )

        # Distinguish supported vs inferred information
        if inferred:
            lines.extend(
                [
                    "",
                    "Inferred / Uncertain Information",
                    "--------------------------------",
                ]
            )
            for item in inferred:
                if item.get("type") == "call_relationship":
                    callees_str = ", ".join(item.get("callees", []))
                    lines.append(
                        f"- Inferred call relationship: {item.get('caller')} calls ({callees_str}) "
                        "based on AST / graph analysis."
                    )
                else:
                    lines.append(
                        f"- Inferred element: {item.get('name')} in {item.get('file_path')} "
                        f"(sources: {', '.join(item.get('sources', []))}; "
                        "implementation source code was not directly inspected)."
                    )
        else:
            lines.extend(
                [
                    "",
                    "Inferred / Uncertain Information",
                    "--------------------------------",
                    "- All identified code findings are directly supported by verified source code declarations.",
                ]
            )

        # Avoid unsupported claims section
        lines.extend(
            [
                "",
                "Evidence Limitations",
                "--------------------",
            ]
        )
        if len(supported) < len(grouped):
            unsupported_note = (
                f"{len(inferred)} elements were referenced without complete source inspection. "
                "Runtime execution and dynamic behaviors are not verified by static evidence alone."
            )
            lines.append(f"- {unsupported_note}")
            unsupported_notes.append(unsupported_note)
        else:
            lines.append(
                "- Claims are restricted to static declarations verified within retrieved evidence."
            )

        if errors:
            lines.extend(
                [
                    "",
                    "Investigation warnings",
                    "----------------------",
                ]
            )
            for err in errors:
                lines.append(
                    f"- {err.get('tool', 'unknown')}: {err.get('error', 'Unknown error')}"
                )

        if self.include_trace and trace:
            self._append_trace(lines, trace)

        return GroundedAnswer(
            content="\n".join(lines),
            query=query,
            supported_findings=supported,
            inferred_information=inferred,
            unsupported_notes=unsupported_notes,
            is_grounded=True,
            has_evidence=True,
            evidence_status=assessment.status,
            associations=associations,
            assessment=assessment,
            conflicting_evidence=assessment.conflicts,
            traceability_map=traceability_map,
        )

    @staticmethod
    def _append_trace(
        lines: list[str],
        trace: list[dict[str, Any]],
    ) -> None:
        """Append investigation execution trace to output lines."""
        if not trace:
            return

        lines.extend(
            [
                "",
                "Execution Trace",
                "---------------",
            ]
        )

        for record in trace:
            lines.append(
                f"- Iteration {record.get('iteration')}: "
                f"{record.get('tool')} — {record.get('status')} "
                f"({record.get('result_count', 0)} raw results, "
                f"{record.get('retained_count', 0)} retained, "
                f"{record.get('duration_ms', 0)} ms)"
            )

    def generate(
        self,
        query: str,
        evidence: Sequence[Mapping[str, Any] | CodeUnit | Any],
        *,
        trace: list[dict[str, Any]] | None = None,
        errors: list[dict[str, Any]] | None = None,
        safe: bool = True,
    ) -> GroundedAnswer:
        """Generate a repository-grounded answer for the given query and evidence."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Question cannot be empty.")

        normalized_evidence = [_normalize_evidence_item(item) for item in evidence]

        if not normalized_evidence:
            return self._generate_deterministic_grounded_answer(
                query,
                [],
                trace=trace,
                errors=errors,
            )

        # If a custom provider is configured, use it to generate the answer
        if self.provider is not None:
            prompt = self.construct_prompt(query, normalized_evidence, errors=errors)
            try:
                content = self.provider.generate(prompt)
                if not isinstance(content, str) or not content.strip():
                    raise AnswerGenerationError("Provider returned empty content.")

                lines = content.splitlines()
                if self.include_trace and trace:
                    self._append_trace(lines, trace)
                final_content = "\n".join(lines)

                associations = self.build_associations(normalized_evidence)
                assessment = self.assess_evidence(
                    query, normalized_evidence, associations
                )
                traceability_map = self.build_traceability_map(associations)

                return GroundedAnswer(
                    content=final_content,
                    query=query,
                    supported_findings=normalized_evidence,
                    is_grounded=True,
                    has_evidence=True,
                    evidence_status=assessment.status,
                    associations=associations,
                    assessment=assessment,
                    conflicting_evidence=assessment.conflicts,
                    traceability_map=traceability_map,
                )
            except Exception as exc:
                if not safe:
                    if isinstance(exc, (ValueError, TypeError)):
                        raise
                    raise AnswerGenerationError(
                        f"Answer generation failed: {exc}"
                    ) from exc

                logger.warning(
                    "Answer generation provider failed (%s); falling back to deterministic grounded synthesis.",
                    exc,
                )

        return self._generate_deterministic_grounded_answer(
            query,
            normalized_evidence,
            trace=trace,
            errors=errors,
        )

    def generate_from_state(
        self,
        state: InvestigationState,
        *,
        safe: bool = True,
    ) -> dict[str, Any]:
        """Generate a grounded answer directly from LangGraph InvestigationState."""
        query = state.get("query", "")
        evidence = state.get("evidence", [])
        errors = state.get("errors", [])
        trace = state.get("trace", [])

        if not query or not query.strip():
            # Gracefully handle missing or blank query in state
            query = "Repository query"

        grounded = self.generate(
            query=query,
            evidence=evidence,
            trace=trace,
            errors=errors,
            safe=safe,
        )

        termination_reason = state.get("termination_reason")
        if not termination_reason:
            termination_reason = (
                "investigation_completed" if evidence else "no_evidence"
            )

        return {
            "final_answer": grounded.content,
            "is_complete": True,
            "needs_more_investigation": False,
            "termination_reason": termination_reason,
            "evidence_status": grounded.evidence_status.value,
            "evidence_assessment": (
                grounded.assessment.to_dict() if grounded.assessment else {}
            ),
            "code_unit_associations": [a.to_dict() for a in grounded.associations],
            "conflicting_evidence": [
                c.to_dict() for c in grounded.conflicting_evidence
            ],
            "traceability_map": grounded.traceability_map,
        }
