"""Conservative validation of claims against repository evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from repolens.verification.citation_models import (
    ClaimValidationResult,
    ValidationIssue,
)

_TOKEN_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "uses",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
}


class ClaimValidator:
    """Check whether claims have identifiable support in repository evidence.

    This validator performs deterministic evidence-reference and lexical
    overlap checks. It does not perform semantic entailment verification.
    A claim marked valid has identifiable evidence with sufficient lexical
    overlap; this is not proof that the evidence logically entails the claim.
    """

    def __init__(self, minimum_overlap: float = 0.5) -> None:
        if not 0.0 <= minimum_overlap <= 1.0:
            raise ValueError("minimum_overlap must be between 0 and 1.")

        self.minimum_overlap = minimum_overlap

    def validate(
        self,
        claim: str,
        evidence: Sequence[Mapping[str, Any]],
        supporting_evidence_ids: Sequence[str] | None = None,
    ) -> ClaimValidationResult:
        """Validate one claim against available evidence records."""
        if not isinstance(claim, str):
            raise TypeError("claim must be a string.")

        if not claim.strip():
            return ClaimValidationResult(
                claim=claim,
                status="incomplete",
                issues=(
                    ValidationIssue(
                        code="empty_claim",
                        message="The claim is empty and cannot be validated.",
                        severity="warning",
                    ),
                ),
            )

        evidence_by_id = {
            str(item["evidence_id"]): item
            for item in evidence
            if item.get("evidence_id") is not None
        }

        if supporting_evidence_ids is not None:
            requested_ids = list(dict.fromkeys(supporting_evidence_ids))
            missing_ids = [
                evidence_id
                for evidence_id in requested_ids
                if evidence_id not in evidence_by_id
            ]

            if missing_ids:
                return ClaimValidationResult(
                    claim=claim,
                    status="invalid",
                    supporting_evidence_ids=tuple(
                        evidence_id
                        for evidence_id in requested_ids
                        if evidence_id in evidence_by_id
                    ),
                    issues=(
                        ValidationIssue(
                            code="missing_evidence_reference",
                            message=(
                                "Referenced evidence IDs were not found: "
                                + ", ".join(missing_ids)
                            ),
                            severity="error",
                        ),
                    ),
                )

            candidates = [evidence_by_id[evidence_id] for evidence_id in requested_ids]
        else:
            candidates = list(evidence)

        if not candidates:
            return ClaimValidationResult(
                claim=claim,
                status="unverified",
                issues=(
                    ValidationIssue(
                        code="no_supporting_evidence",
                        message="No repository evidence is available for this claim.",
                        severity="warning",
                    ),
                ),
            )

        claim_tokens = self._tokens(claim)

        if not claim_tokens:
            return ClaimValidationResult(
                claim=claim,
                status="incomplete",
                issues=(
                    ValidationIssue(
                        code="no_meaningful_claim_terms",
                        message=(
                            "The claim contains no meaningful terms for "
                            "evidence comparison."
                        ),
                        severity="warning",
                    ),
                ),
            )

        scored_candidates: list[tuple[float, str | None]] = []

        for item in candidates:
            evidence_text = self._evidence_text(item)
            evidence_tokens = self._tokens(evidence_text)

            if not evidence_tokens:
                continue

            overlap = len(claim_tokens & evidence_tokens) / len(claim_tokens)
            evidence_id = item.get("evidence_id")
            scored_candidates.append(
                (overlap, str(evidence_id) if evidence_id is not None else None)
            )

        if not scored_candidates:
            return ClaimValidationResult(
                claim=claim,
                status="unverified",
                issues=(
                    ValidationIssue(
                        code="empty_evidence_content",
                        message=(
                            "Available evidence records contain no searchable "
                            "source content or identifying information."
                        ),
                        severity="warning",
                    ),
                ),
            )

        best_overlap = max(score for score, _ in scored_candidates)

        supporting_ids = tuple(
            evidence_id
            for score, evidence_id in scored_candidates
            if score >= self.minimum_overlap and evidence_id is not None
        )

        if supporting_ids:
            return ClaimValidationResult(
                claim=claim,
                status="valid",
                supporting_evidence_ids=supporting_ids,
                issues=(
                    ValidationIssue(
                        code="lexical_support_only",
                        message=(
                            "Evidence has sufficient lexical overlap. "
                            "Semantic entailment has not been verified."
                        ),
                        severity="warning",
                    ),
                ),
            )

        return ClaimValidationResult(
            claim=claim,
            status="unverified",
            supporting_evidence_ids=(),
            issues=(
                ValidationIssue(
                    code="weak_evidence_overlap",
                    message=(
                        "Available evidence does not meet the configured "
                        f"lexical overlap threshold of {self.minimum_overlap:.0%}. "
                        f"Best observed overlap: {best_overlap:.0%}."
                    ),
                    severity="warning",
                ),
            ),
        )

    def validate_many(
        self,
        claims: Sequence[str],
        evidence: Sequence[Mapping[str, Any]],
    ) -> list[ClaimValidationResult]:
        """Validate multiple claims against the same evidence collection."""
        return [self.validate(claim, evidence) for claim in claims]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token.lower()
            for token in _TOKEN_PATTERN.findall(text)
            if token.lower() not in _STOP_WORDS and len(token) > 1
        }

    @staticmethod
    def _evidence_text(item: Mapping[str, Any]) -> str:
        """Collect searchable text from an evidence record."""
        parts = [
            item.get("name", ""),
            item.get("qualified_name", ""),
            item.get("kind", ""),
            item.get("file_path", ""),
            item.get("content", ""),
        ]

        metadata = item.get("metadata", {})
        if isinstance(metadata, Mapping):
            parts.extend(
                str(metadata.get(key, ""))
                for key in ("docstring", "description", "summary")
            )

        return " ".join(str(part) for part in parts if part)


__all__ = ["ClaimValidator"]
