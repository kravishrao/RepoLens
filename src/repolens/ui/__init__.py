from __future__ import annotations

from repolens.ui.adapter import run_repolens_query
from repolens.ui.app import main
from repolens.ui.components.cards import render_feature_card
from repolens.ui.components.header import render_header
from repolens.ui.components.results import (
    render_answer_card,
    render_citation_validation,
    render_empty_or_failed_state,
    render_evidence_viewer,
    render_findings_section,
    render_investigation_trace,
    render_overview_metrics,
)
from repolens.ui.components.sidebar import render_sidebar
from repolens.ui.presentation import (
    CitationViewModel,
    ClaimViewModel,
    EvidenceViewModel,
    PresentationData,
    TraceStepViewModel,
    ValidationSummaryViewModel,
    extract_presentation_data,
    format_line_reference,
    is_supported_finding,
)
from repolens.ui.repository_service import (
    LoadedRepository,
    format_bytes,
    get_bundled_repositories,
    load_repository_pipeline,
    resolve_repository_source,
)
from repolens.ui.styles.theme import apply_theme

__all__ = [
    "CitationViewModel",
    "ClaimViewModel",
    "EvidenceViewModel",
    "LoadedRepository",
    "PresentationData",
    "TraceStepViewModel",
    "ValidationSummaryViewModel",
    "apply_theme",
    "extract_presentation_data",
    "format_bytes",
    "format_line_reference",
    "get_bundled_repositories",
    "is_supported_finding",
    "load_repository_pipeline",
    "main",
    "render_answer_card",
    "render_citation_validation",
    "render_empty_or_failed_state",
    "render_evidence_viewer",
    "render_feature_card",
    "render_findings_section",
    "render_header",
    "render_investigation_trace",
    "render_overview_metrics",
    "render_sidebar",
    "resolve_repository_source",
    "run_repolens_query",
]
