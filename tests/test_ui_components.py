"""Unit tests for RepoLens Streamlit UI components."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from repolens.ui.components import (
    render_answer_card,
    render_citation_validation,
    render_empty_or_failed_state,
    render_evidence_viewer,
    render_findings_section,
    render_investigation_trace,
    render_kpi_card,
    render_kpi_row,
    render_overview_metrics,
)
from repolens.ui.presentation import (
    CitationViewModel,
    EvidenceViewModel,
    PresentationData,
    TraceStepViewModel,
    ValidationSummaryViewModel,
    extract_presentation_data,
)


def make_sample_presentation_data(
    *,
    has_evidence: bool = True,
    has_errors: bool = False,
    is_valid: bool = True,
) -> PresentationData:
    evidence = []
    supported = []
    inferred = []

    if has_evidence:
        ev1 = EvidenceViewModel(
            evidence_id="ev-1",
            code_unit_id="u-1",
            name="calculate_total",
            qualified_name="billing.calculate_total",
            kind="function",
            file_path="src/billing/calc.py",
            start_line=10,
            end_line=25,
            line_reference="Lines 10-25",
            source="inspect_source",
            content="def calculate_total(items):\n    return sum(items)",
            docstring="Compute order total.",
            score=0.92,
            is_supported=True,
            metadata={"calls": ["sum"]},
        )
        ev2 = EvidenceViewModel(
            evidence_id="ev-2",
            code_unit_id="u-2",
            name="OrderSummary",
            qualified_name="billing.OrderSummary",
            kind="class",
            file_path="src/billing/models.py",
            start_line=5,
            end_line=40,
            line_reference="Lines 5-40",
            source="context_expansion",
            content="",
            docstring=None,
            score=None,
            is_supported=False,
            metadata={},
        )
        evidence = [ev1, ev2]
        supported = [ev1]
        inferred = [ev2]

    citations = []
    if has_evidence:
        citations.append(
            CitationViewModel(
                citation_id="cit-1",
                raw_text="billing.calculate_total (Lines 10-25)",
                symbol="billing.calculate_total",
                file_path="src/billing/calc.py",
                line_reference="Lines 10-25",
                status="valid" if is_valid else "invalid",
                file_exists=is_valid,
                path_is_safe=True,
                line_range_valid=is_valid,
                symbol_found=is_valid,
                code_unit_found=is_valid,
                issues=[]
                if is_valid
                else [{"code": "ERR", "message": "Failed", "severity": "error"}],
            )
        )

    val_summary = ValidationSummaryViewModel(
        enabled=has_evidence,
        overall_status="valid" if is_valid and has_evidence else "invalid",
        citation_count=len(citations),
        claim_count=0,
        valid_citation_count=1 if (is_valid and has_evidence) else 0,
        invalid_citation_count=0 if is_valid else 1,
        incomplete_citation_count=0,
        unverified_citation_count=0,
        issues=[],
    )

    trace = [
        TraceStepViewModel(
            iteration=1,
            tool="keyword_search",
            status="success",
            duration_ms=8.5,
            result_count=5,
            retained_count=1,
            findings=["ev-1"],
            error=None,
        )
    ]

    errors = []
    if has_errors:
        errors.append({"tool": "inspect_source", "error": "Disk read timeout"})

    return PresentationData(
        query="Explain billing system",
        final_answer="The billing system computes totals via `calculate_total`.",
        is_complete=True,
        termination_reason="sufficient_evidence" if has_evidence else "no_evidence",
        has_evidence=has_evidence,
        has_errors=has_errors,
        supported_findings=supported,
        inferred_findings=inferred,
        all_evidence=evidence,
        citations=citations,
        claims=[],
        validation_summary=val_summary,
        trace_steps=trace,
        executed_tools=["keyword_search"],
        errors=errors,
        total_duration_ms=8.5,
        unique_code_elements_count=len(evidence),
    )


def _mock_columns(spec):
    count = len(spec) if isinstance(spec, list) else int(spec)
    return [MagicMock() for _ in range(count)]


def test_render_overview_metrics():
    data = make_sample_presentation_data()
    with patch("streamlit.columns", side_effect=_mock_columns) as mock_cols:
        render_overview_metrics(data)
        assert mock_cols.called


def test_render_answer_card_with_evidence():
    data = make_sample_presentation_data(has_evidence=True)
    with patch("streamlit.markdown") as mock_md, patch("streamlit.container") as mock_c:
        mock_c.return_value.__enter__.return_value = MagicMock()
        render_answer_card(data)
        assert mock_md.called


def test_render_answer_card_without_evidence():
    data = make_sample_presentation_data(has_evidence=False)
    with patch("streamlit.warning") as mock_warn:
        render_answer_card(data)
        assert mock_warn.called


def test_render_findings_section():
    data = make_sample_presentation_data(has_evidence=True)
    with (
        patch("streamlit.tabs") as mock_tabs,
        patch("streamlit.container") as mock_c,
        patch("streamlit.columns", side_effect=_mock_columns),
    ):
        tab = MagicMock()
        mock_tabs.return_value = [tab, tab, tab]
        mock_c.return_value.__enter__.return_value = MagicMock()
        render_findings_section(data)
        assert mock_tabs.called


def test_render_evidence_viewer():
    data = make_sample_presentation_data(has_evidence=True)
    with patch("streamlit.expander") as mock_exp, patch("streamlit.code") as mock_code:
        mock_exp.return_value.__enter__.return_value = MagicMock()
        render_evidence_viewer(data)
        assert mock_code.called
        assert mock_exp.called


def test_render_citation_validation_valid():
    data = make_sample_presentation_data(has_evidence=True, is_valid=True)
    with (
        patch("streamlit.success") as mock_succ,
        patch("streamlit.container") as mock_c,
        patch("streamlit.columns", side_effect=_mock_columns),
    ):
        mock_c.return_value.__enter__.return_value = MagicMock()
        render_citation_validation(data)
        assert mock_succ.called


def test_render_citation_validation_invalid():
    data = make_sample_presentation_data(has_evidence=True, is_valid=False)
    with (
        patch("streamlit.warning") as mock_warn,
        patch("streamlit.error") as mock_err,
        patch("streamlit.container") as mock_c,
        patch("streamlit.columns", side_effect=_mock_columns),
    ):
        mock_c.return_value.__enter__.return_value = MagicMock()
        render_citation_validation(data)
        assert mock_warn.called
        assert mock_err.called


def test_render_investigation_trace():
    data = make_sample_presentation_data()
    with (
        patch("streamlit.container") as mock_c,
        patch("streamlit.columns", side_effect=_mock_columns),
    ):
        mock_c.return_value.__enter__.return_value = MagicMock()
        render_investigation_trace(data)
        assert mock_c.called


def test_render_empty_or_failed_state():
    data = extract_presentation_data(
        {
            "errors": [{"tool": "ast_parser", "error": "Syntax error"}],
            "termination_reason": "tool_failures",
        }
    )
    with patch("streamlit.error") as mock_err, patch("streamlit.warning") as mock_warn:
        render_empty_or_failed_state(data)
        assert mock_err.called
        assert mock_warn.called


def test_render_kpi_card():
    card_html = render_kpi_card(
        title="CODEUNITS",
        value="1,240",
        subtitle="AST Functions",
        variant="coral",
        badge_text="AST",
    )
    assert "rl-kpi-card" in card_html
    assert "rl-kpi-coral" in card_html
    assert "CODEUNITS" in card_html
    assert "1,240" in card_html
    assert "AST Functions" in card_html
    assert "AST" in card_html


def test_render_kpi_row():
    cards = [
        {"title": "C1", "value": "10", "subtitle": "S1", "variant": "coral"},
        {"title": "C2", "value": "20", "subtitle": "S2", "variant": "cyan"},
    ]
    with patch("streamlit.markdown") as mock_md:
        render_kpi_row(cards)
        assert mock_md.called
        rendered = mock_md.call_args[0][0]
        assert "rl-kpi-grid" in rendered
        assert "rl-kpi-coral" in rendered
        assert "rl-kpi-cyan" in rendered
        assert "C1" in rendered
        assert "C2" in rendered


def test_format_citation_entry():
    from repolens.ui.presentation import format_citation_entry

    res1 = format_citation_entry(
        index=1,
        file_path="src/advisor/recommender.py",
        symbol="QueryRecommender",
        kind="class",
        start_line=51,
        end_line=292,
    )
    assert (
        res1 == "[1] recommender.py - QueryRecommender (Class definition, lines 51-292)"
    )

    res2 = format_citation_entry(
        index=2,
        file_path="src/advisor/recommender.py",
        symbol="QueryRecommender.analyze",
        kind="method",
        start_line=101,
        end_line=240,
        description="Plan traversal & bottleneck logic",
    )
    assert (
        res2
        == "[2] recommender.py - QueryRecommender.analyze (Plan traversal & bottleneck logic, lines 101-240)"
    )

    res3 = format_citation_entry(
        index=6,
        file_path="src/api/main.py",
        symbol="analyze_query",
        kind="function",
        start_line=200,
        end_line=312,
        description="FastAPI route handler",
    )
    assert res3 == "[6] main.py - analyze_query (FastAPI route handler, lines 200-312)"


def test_render_citation_validation_formats_lines_cleanly():
    data = make_sample_presentation_data()
    with (
        patch("streamlit.subheader"),
        patch("streamlit.success"),
        patch("streamlit.info"),
        patch("streamlit.warning"),
        patch("streamlit.error"),
        patch("streamlit.container"),
        patch(
            "streamlit.columns",
            side_effect=[[MagicMock(), MagicMock()], [MagicMock() for _ in range(5)]],
        ),
        patch("streamlit.markdown") as mock_md,
    ):
        render_citation_validation(data)
        md_calls = [str(c[0][0]) for c in mock_md.call_args_list if c[0]]
        assert any(
            "[1] calc.py - billing.calculate_total" in m and "lines 10-25" in m
            for m in md_calls
        )
        assert not any("lines Lines" in m for m in md_calls)
