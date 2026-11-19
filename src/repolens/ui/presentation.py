"""Presentation data extraction and view-models for RepoLens Streamlit interface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# Sources that indicate direct code inspection
DIRECT_INSPECTION_SOURCES = frozenset(
    {
        "inspect_source",
        "source_inspection",
        "read_file",
    }
)

# Sources that indicate indirect graph traversal or uninspected relationships
GRAPH_INFERENCE_SOURCES = frozenset(
    {
        "context_expansion",
        "graph_query",
        "graph_traversal",
        "callers",
        "callees",
        "dependencies",
    }
)


def format_line_reference(start_line: int | None, end_line: int | None) -> str:
    """Format start and end lines into a human-readable range string."""
    if start_line is not None and end_line is not None:
        if start_line == end_line:
            return f"Line {start_line}"
        return f"Lines {start_line}-{end_line}"
    if start_line is not None:
        return f"Line {start_line}+"
    return "Lines unknown"


def is_supported_finding(item: Mapping[str, Any]) -> bool:
    """Determine whether an evidence item represents directly inspected code.

    Directly inspected code units have their source text verified through
    source-inspection tools (`inspect_source`, `read_file`, etc.) or provide
    full verified source code content. Graph-traversal relationships and
    uninspected candidates are treated as inferred / uncertain.
    """
    source = str(item.get("source") or "").strip().lower()
    content = str(item.get("content") or "").strip()

    if source in DIRECT_INSPECTION_SOURCES:
        return True

    if source in GRAPH_INFERENCE_SOURCES:
        return False

    # For general retrieval (keyword/semantic search), if source content is present
    # and has substantial lines, it is inspected evidence.
    return bool(content and len(content.splitlines()) >= 1)


@dataclass(frozen=True)
class EvidenceViewModel:
    """Formatted view-model for an evidence record or code snippet."""

    evidence_id: str
    code_unit_id: str
    name: str
    qualified_name: str
    kind: str
    file_path: str
    start_line: int | None
    end_line: int | None
    line_reference: str
    source: str
    content: str
    docstring: str | None
    score: float | None
    is_supported: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CitationViewModel:
    """Formatted view-model for an answer citation and its verification."""

    citation_id: str
    raw_text: str
    symbol: str | None
    file_path: str | None
    line_reference: str
    status: str
    file_exists: bool | None
    path_is_safe: bool | None
    line_range_valid: bool | None
    symbol_found: bool | None
    code_unit_found: bool | None
    issues: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class ClaimViewModel:
    """Formatted view-model for a factual claim validation."""

    claim: str
    status: str
    supporting_evidence_ids: list[str] = field(default_factory=list)
    issues: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class TraceStepViewModel:
    """Formatted view-model for a single agent tool execution step."""

    iteration: int
    tool: str
    status: str
    duration_ms: float
    result_count: int
    retained_count: int
    findings: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class ValidationSummaryViewModel:
    """Summary of citation and claim validation outcomes."""

    enabled: bool
    overall_status: str
    citation_count: int
    claim_count: int
    valid_citation_count: int
    invalid_citation_count: int
    incomplete_citation_count: int
    unverified_citation_count: int
    issues: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class PresentationData:
    """Normalized, complete presentation data ready for Streamlit rendering."""

    query: str
    final_answer: str
    is_complete: bool
    termination_reason: str
    has_evidence: bool
    has_errors: bool
    supported_findings: list[EvidenceViewModel]
    inferred_findings: list[EvidenceViewModel]
    all_evidence: list[EvidenceViewModel]
    citations: list[CitationViewModel]
    claims: list[ClaimViewModel]
    validation_summary: ValidationSummaryViewModel
    trace_steps: list[TraceStepViewModel]
    executed_tools: list[str]
    errors: list[dict[str, str]]
    total_duration_ms: float
    unique_code_elements_count: int
    evidence_status: str = "sufficient"
    is_sufficient: bool = True
    is_conflicting: bool = False
    is_uncertain: bool = False
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    insufficient_reasons: list[str] = field(default_factory=list)
    code_unit_associations: list[dict[str, Any]] = field(default_factory=list)
    traceability_map: dict[str, list[str]] = field(default_factory=dict)


def extract_presentation_data(
    state: Mapping[str, Any] | None,
) -> PresentationData:
    """Extract and normalize raw `InvestigationState` into presentation view models.

    Handles missing keys, `None` values, and empty states gracefully so the UI
    never crashes on unexpected or partial inputs.
    """
    if not isinstance(state, Mapping):
        return PresentationData(
            query="",
            final_answer="No investigation has been executed yet.",
            is_complete=False,
            termination_reason="unstarted",
            has_evidence=False,
            has_errors=False,
            supported_findings=[],
            inferred_findings=[],
            all_evidence=[],
            citations=[],
            claims=[],
            validation_summary=ValidationSummaryViewModel(
                enabled=False,
                overall_status="disabled",
                citation_count=0,
                claim_count=0,
                valid_citation_count=0,
                invalid_citation_count=0,
                incomplete_citation_count=0,
                unverified_citation_count=0,
                issues=[],
            ),
            trace_steps=[],
            executed_tools=[],
            errors=[],
            total_duration_ms=0.0,
            unique_code_elements_count=0,
            evidence_status="insufficient",
            is_sufficient=False,
            is_conflicting=False,
            is_uncertain=False,
            conflicts=[],
            insufficient_reasons=["No investigation has been executed."],
            code_unit_associations=[],
            traceability_map={},
        )

    query = str(state.get("query") or "")
    final_answer = str(state.get("final_answer") or "").strip()
    is_complete = bool(state.get("is_complete", False))
    termination_reason = str(state.get("termination_reason") or "unspecified")

    # Raw evidence records
    raw_evidence = state.get("evidence") or []
    if not isinstance(raw_evidence, list):
        raw_evidence = []

    all_evidence: list[EvidenceViewModel] = []
    supported_findings: list[EvidenceViewModel] = []
    inferred_findings: list[EvidenceViewModel] = []
    unique_unit_ids: set[str] = set()

    for idx, item in enumerate(raw_evidence):
        if not isinstance(item, Mapping):
            continue

        evidence_id = str(item.get("evidence_id") or f"evidence-{idx}")
        code_unit_id = str(item.get("code_unit_id") or f"unit-{idx}")
        unique_unit_ids.add(code_unit_id)

        name = str(item.get("name") or "Unknown")
        qualified_name = str(item.get("qualified_name") or name)
        kind = str(item.get("kind") or "code_element")
        file_path = str(item.get("file_path") or "unknown_file.py")

        start_line = item.get("start_line")
        end_line = item.get("end_line")
        start_line_int = (
            int(start_line)
            if isinstance(start_line, (int, str)) and str(start_line).isdigit()
            else None
        )
        end_line_int = (
            int(end_line)
            if isinstance(end_line, (int, str)) and str(end_line).isdigit()
            else None
        )

        line_ref = format_line_reference(start_line_int, end_line_int)
        source = str(item.get("source") or "retrieval")
        content = str(item.get("content") or "")

        metadata = dict(item.get("metadata") or {})
        docstring = metadata.get("docstring")
        if not docstring and item.get("docstring"):
            docstring = str(item.get("docstring"))

        score = item.get("score")
        score_val = float(score) if isinstance(score, (int, float)) else None

        supported = is_supported_finding(item)

        vm = EvidenceViewModel(
            evidence_id=evidence_id,
            code_unit_id=code_unit_id,
            name=name,
            qualified_name=qualified_name,
            kind=kind,
            file_path=file_path,
            start_line=start_line_int,
            end_line=end_line_int,
            line_reference=line_ref,
            source=source,
            content=content,
            docstring=str(docstring) if docstring else None,
            score=score_val,
            is_supported=supported,
            metadata=metadata,
        )

        all_evidence.append(vm)
        if supported:
            supported_findings.append(vm)
        else:
            inferred_findings.append(vm)

    has_evidence = len(all_evidence) > 0

    # Errors
    raw_errors = state.get("errors") or []
    errors: list[dict[str, str]] = []
    if isinstance(raw_errors, list):
        for err in raw_errors:
            if isinstance(err, Mapping):
                errors.append(
                    {
                        "tool": str(err.get("tool") or "unknown"),
                        "error": str(err.get("error") or "Unknown error"),
                    }
                )

    has_errors = len(errors) > 0

    # Trace steps
    raw_trace = state.get("trace") or []
    trace_steps: list[TraceStepViewModel] = []
    total_duration_ms = 0.0

    if isinstance(raw_trace, list):
        for t in raw_trace:
            if not isinstance(t, Mapping):
                continue
            duration = float(t.get("duration_ms") or 0.0)
            total_duration_ms += duration
            findings_list = [str(f) for f in (t.get("findings") or [])]

            trace_steps.append(
                TraceStepViewModel(
                    iteration=int(t.get("iteration") or 0),
                    tool=str(t.get("tool") or "unknown_tool"),
                    status=str(t.get("status") or "unknown"),
                    duration_ms=round(duration, 2),
                    result_count=int(t.get("result_count") or 0),
                    retained_count=int(t.get("retained_count") or 0),
                    findings=findings_list,
                    error=str(t.get("error")) if t.get("error") else None,
                )
            )

    # Executed tools list
    raw_executed = state.get("executed_tools") or []
    executed_tools = [str(tool) for tool in raw_executed if str(tool).strip()]
    if not executed_tools and trace_steps:
        executed_tools = list(dict.fromkeys(s.tool for s in trace_steps))

    # Citation Validation
    raw_citation_results = state.get("citation_validation_results") or []
    citations: list[CitationViewModel] = []
    valid_count = 0
    invalid_count = 0
    incomplete_count = 0
    unverified_count = 0

    if isinstance(raw_citation_results, list):
        for c_res in raw_citation_results:
            if not isinstance(c_res, Mapping):
                continue

            status = str(c_res.get("status") or "unverified").lower()
            if status == "valid":
                valid_count += 1
            elif status == "invalid":
                invalid_count += 1
            elif status == "incomplete":
                incomplete_count += 1
            else:
                unverified_count += 1

            citation_data = c_res.get("citation") or {}
            c_issues = []
            for iss in c_res.get("issues") or []:
                if isinstance(iss, Mapping):
                    c_issues.append(
                        {
                            "code": str(iss.get("code") or "ISSUE"),
                            "message": str(iss.get("message") or ""),
                            "severity": str(iss.get("severity") or "error"),
                        }
                    )

            c_start = citation_data.get("start_line")
            c_end = citation_data.get("end_line")
            start_int = (
                int(c_start)
                if isinstance(c_start, (int, str)) and str(c_start).isdigit()
                else None
            )
            end_int = (
                int(c_end)
                if isinstance(c_end, (int, str)) and str(c_end).isdigit()
                else None
            )

            citations.append(
                CitationViewModel(
                    citation_id=str(
                        citation_data.get("citation_id") or f"cit-{len(citations)}"
                    ),
                    raw_text=str(citation_data.get("raw_text") or ""),
                    symbol=citation_data.get("symbol"),
                    file_path=citation_data.get("file_path"),
                    line_reference=format_line_reference(start_int, end_int),
                    status=status,
                    file_exists=c_res.get("file_exists"),
                    path_is_safe=c_res.get("path_is_safe"),
                    line_range_valid=c_res.get("line_range_valid"),
                    symbol_found=c_res.get("symbol_found"),
                    code_unit_found=c_res.get("code_unit_found"),
                    issues=c_issues,
                )
            )

    # Claim Validation
    raw_claim_results = state.get("claim_validation_results") or []
    claims: list[ClaimViewModel] = []
    if isinstance(raw_claim_results, list):
        for cl in raw_claim_results:
            if not isinstance(cl, Mapping):
                continue
            cl_issues = []
            for iss in cl.get("issues") or []:
                if isinstance(iss, Mapping):
                    cl_issues.append(
                        {
                            "code": str(iss.get("code") or "ISSUE"),
                            "message": str(iss.get("message") or ""),
                            "severity": str(iss.get("severity") or "error"),
                        }
                    )
            claims.append(
                ClaimViewModel(
                    claim=str(cl.get("claim") or ""),
                    status=str(cl.get("status") or "unverified"),
                    supporting_evidence_ids=[
                        str(i) for i in (cl.get("supporting_evidence_ids") or [])
                    ],
                    issues=cl_issues,
                )
            )

    # Validation summary
    val_sum = state.get("validation_summary") or {}
    val_enabled = bool(val_sum.get("enabled", False) or len(citations) > 0)
    overall_status = str(
        val_sum.get("overall_status")
        or ("valid" if valid_count > 0 and invalid_count == 0 else "unverified")
    )

    val_issues = []
    for iss in val_sum.get("issues") or []:
        if isinstance(iss, Mapping):
            val_issues.append(
                {
                    "code": str(iss.get("code") or "ISSUE"),
                    "message": str(iss.get("message") or ""),
                    "severity": str(iss.get("severity") or "error"),
                }
            )

    summary_vm = ValidationSummaryViewModel(
        enabled=val_enabled,
        overall_status=overall_status,
        citation_count=len(citations),
        claim_count=len(claims),
        valid_citation_count=valid_count,
        invalid_citation_count=invalid_count,
        incomplete_citation_count=incomplete_count,
        unverified_citation_count=unverified_count,
        issues=val_issues,
    )

    # Evidence status and assessment extraction (RL-36)
    explicit_status = state.get("evidence_status")
    if explicit_status and isinstance(explicit_status, str):
        evidence_status = explicit_status.lower()
    elif not all_evidence:
        evidence_status = "insufficient"
    elif not supported_findings and inferred_findings:
        evidence_status = "uncertain"
    else:
        evidence_status = "sufficient"

    raw_conflicts = state.get("conflicting_evidence") or []
    conflicts = [dict(c) for c in raw_conflicts if isinstance(c, Mapping)]
    is_conflicting = len(conflicts) > 0 or evidence_status == "conflicting"
    is_sufficient = evidence_status == "sufficient" and len(all_evidence) > 0
    is_uncertain = evidence_status in ("uncertain", "conflicting")

    raw_assessment = state.get("evidence_assessment") or {}
    insufficient_reasons: list[str] = []
    if isinstance(raw_assessment, Mapping):
        insufficient_reasons = [
            str(r) for r in (raw_assessment.get("missing_aspects") or [])
        ]
    if not insufficient_reasons and not all_evidence:
        insufficient_reasons = [
            "No code units or repository evidence records were retrieved."
        ]

    raw_associations = state.get("code_unit_associations") or []
    code_unit_associations = [
        dict(a) for a in raw_associations if isinstance(a, Mapping)
    ]

    raw_trace_map = state.get("traceability_map") or {}
    traceability_map = dict(raw_trace_map) if isinstance(raw_trace_map, Mapping) else {}

    return PresentationData(
        query=query,
        final_answer=final_answer,
        is_complete=is_complete,
        termination_reason=termination_reason,
        has_evidence=has_evidence,
        has_errors=has_errors,
        supported_findings=supported_findings,
        inferred_findings=inferred_findings,
        all_evidence=all_evidence,
        citations=citations,
        claims=claims,
        validation_summary=summary_vm,
        trace_steps=trace_steps,
        executed_tools=executed_tools,
        errors=errors,
        total_duration_ms=round(total_duration_ms, 2),
        unique_code_elements_count=len(unique_unit_ids),
        evidence_status=evidence_status,
        is_sufficient=is_sufficient,
        is_conflicting=is_conflicting,
        is_uncertain=is_uncertain,
        conflicts=conflicts,
        insufficient_reasons=insufficient_reasons,
        code_unit_associations=code_unit_associations,
        traceability_map=traceability_map,
    )
