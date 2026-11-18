"""Repository-grounded answer generation component."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from repolens.agent.state import InvestigationState
from repolens.ingestion.code_unit import CodeUnit

logger = logging.getLogger(__name__)


class AnswerGenerationError(Exception):
    """Raised when answer generation fails in strict mode."""


class GenerationProvider(Protocol):
    """Protocol for pluggable LLM or text generation providers."""

    def generate(self, prompt: str) -> str:
        """Generate response text given a prompt."""
        ...


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

    def _generate_deterministic_grounded_answer(
        self,
        query: str,
        normalized_evidence: list[dict[str, Any]],
        *,
        trace: list[dict[str, Any]] | None = None,
        errors: list[dict[str, Any]] | None = None,
    ) -> GroundedAnswer:
        """Synthesize a structured, evidence-grounded answer deterministically."""
        if not normalized_evidence:
            answer_parts = [
                "No relevant code evidence was found for this repository question."
            ]

            if errors:
                answer_parts.append("\n\nInvestigation errors:")
                for err in errors:
                    answer_parts.append(
                        f"\n- {err.get('tool', 'unknown')}: {err.get('error', 'Unknown error')}"
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
            "",
            "Findings",
            "--------",
        ]

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

                return GroundedAnswer(
                    content=final_content,
                    query=query,
                    supported_findings=normalized_evidence,
                    is_grounded=True,
                    has_evidence=True,
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
        }
