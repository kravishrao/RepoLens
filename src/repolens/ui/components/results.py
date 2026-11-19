"""Streamlit UI rendering components for RepoLens presentation interface."""

from __future__ import annotations

import streamlit as st

from repolens.ui.presentation import EvidenceViewModel, PresentationData


def render_overview_metrics(data: PresentationData) -> None:
    """Render high-level metrics for the investigation."""
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        label="Evidence Units",
        value=len(data.all_evidence),
        help="Total relevant code evidence records collected by agent tools",
    )

    col2.metric(
        label="Unique Elements",
        value=data.unique_code_elements_count,
        help="Distinct functions, classes, or modules investigated",
    )

    col3.metric(
        label="Tools Executed",
        value=len(data.executed_tools),
        help="Number of agent tools invoked across iterations",
    )

    duration_s = round(data.total_duration_ms / 1000.0, 2)
    col4.metric(
        label="Duration",
        value=f"{duration_s}s",
        help="Cumulative execution time across all tool runs",
    )

    val_status = data.validation_summary.overall_status.upper()
    col5.metric(
        label="Citation Status",
        value=val_status,
        help="Outcome of source citation and claim verification",
    )


def render_answer_card(data: PresentationData) -> None:
    """Render the main generated answer in a prominent readable container."""
    st.subheader("💡 Grounded Answer")

    if not data.final_answer or not data.has_evidence:
        render_empty_or_failed_state(data)
        return

    # Status pill container
    status_label = "Complete" if data.is_complete else "In Progress"
    st.caption(
        f"**Question:** {data.query or 'General inquiry'} | "
        f"**Status:** `{status_label}` | "
        f"**Termination Reason:** `{data.termination_reason}`"
    )

    # Render formatted markdown content
    with st.container(border=True):
        st.markdown(data.final_answer)


def render_findings_section(data: PresentationData) -> None:
    """Render structured code findings, separating supported from inferred elements."""
    st.subheader("🔍 Code Findings & Citations")

    if not data.has_evidence:
        st.info("No code elements were identified during this investigation.")
        return

    tab_supported, tab_inferred, tab_all = st.tabs(
        [
            f"🟢 Supported Findings ({len(data.supported_findings)})",
            f"🟡 Inferred / Referenced Elements ({len(data.inferred_findings)})",
            f"All Findings ({len(data.all_evidence)})",
        ]
    )

    with tab_supported:
        st.markdown(
            "> **Directly Inspected Evidence**: Source code was directly inspected and verified "
            "using AST extraction, file inspection, or verified source retrieval."
        )
        if data.supported_findings:
            _render_findings_list(data.supported_findings, badge_color="green")
        else:
            st.warning("No findings were directly inspected in source files.")

    with tab_inferred:
        st.markdown(
            "> **Inferred Context / Graph References**: Code elements discovered through code-graph "
            "relationships (callers, callees, dependencies) without direct full-file inspection."
        )
        if data.inferred_findings:
            _render_findings_list(data.inferred_findings, badge_color="orange")
        else:
            st.success("All retrieved findings are backed by direct source inspection.")

    with tab_all:
        _render_findings_list(data.all_evidence, badge_color="blue")


def _render_findings_list(
    findings: list[EvidenceViewModel],
    badge_color: str = "blue",
) -> None:
    """Render a structured list of code findings."""
    for idx, item in enumerate(findings, start=1):
        with st.container(border=True):
            col_sym, col_loc, col_kind = st.columns([3, 3, 2])

            with col_sym:
                badge = "🟢 Inspected" if item.is_supported else "🟡 Inferred"
                st.markdown(f"**{idx}. `{item.qualified_name}`** ({badge})")

            with col_loc:
                st.markdown(f"📁 `{item.file_path}` ({item.line_reference})")

            with col_kind:
                st.caption(f"Kind: `{item.kind}` | Source: `{item.source}`")

            if item.docstring:
                st.caption(f"📝 *Docstring:* {item.docstring}")


def render_evidence_viewer(data: PresentationData) -> None:
    """Provide an expandable evidence viewer with syntax-highlighted code snippets."""
    st.subheader("📄 Relevant Code Evidence Viewer")

    if not data.all_evidence:
        st.info("No source code snippets available.")
        return

    st.markdown(
        "Expand each code unit below to inspect the extracted source text, line boundaries, "
        "and retrieval metadata."
    )

    for idx, item in enumerate(data.all_evidence, start=1):
        support_tag = "🟢 Inspected" if item.is_supported else "🟡 Inferred"
        expander_title = (
            f"[{idx}] {item.qualified_name} ({item.kind}) — "
            f"{item.file_path} ({item.line_reference}) {support_tag}"
        )

        with st.expander(expander_title, expanded=(idx == 1 and item.is_supported)):
            st.markdown(
                f"- **Symbol:** `{item.qualified_name}`\n"
                f"- **File:** `{item.file_path}`\n"
                f"- **Lines:** `{item.line_reference}`\n"
                f"- **Discovered via:** `{item.source}`"
            )

            if item.score is not None:
                st.caption(f"Relevance Score: `{item.score:.4f}`")

            if item.docstring:
                st.info(f"**Docstring:**\n{item.docstring}")

            if item.content and item.content.strip():
                st.markdown("**Source Snippet:**")
                st.code(item.content, language="python", line_numbers=True)
            else:
                st.caption(
                    "*(No source code snippet retained for this graph reference)*"
                )

            if item.metadata:
                with st.expander("Show Raw Metadata", expanded=False):
                    st.json(item.metadata)


def render_citation_validation(data: PresentationData) -> None:
    """Display citation-validation outcomes, checks checklist, and detected issues."""
    st.subheader("🛡️ Source Citation Validation")

    val = data.validation_summary

    if not val.enabled and not data.citations:
        st.info(
            "Citation validation is disabled or was not performed. "
            "To enable validation, provide a valid repository root."
        )
        return

    # Validation summary banner
    if val.overall_status == "valid":
        st.success(
            f"✅ **Validation Passed**: All {val.citation_count} extracted citations are verified "
            "against source files and AST symbols."
        )
    elif val.overall_status in {"invalid", "incomplete"}:
        st.warning(
            f"⚠️ **Validation Status: {val.overall_status.upper()}**: "
            f"{val.valid_citation_count}/{val.citation_count} citations valid, "
            f"{val.invalid_citation_count} invalid, {val.incomplete_citation_count} incomplete."
        )
    else:
        st.info(f"Validation Status: `{val.overall_status}`")

    # Detailed Citation Cards
    if data.citations:
        st.markdown("#### Extracted Citations")

        for idx, cit in enumerate(data.citations, start=1):
            with st.container(border=True):
                col_head, col_status = st.columns([4, 1])

                with col_head:
                    st.markdown(
                        f"**Citation #{idx}:** `{cit.symbol or 'Symbol'}` at "
                        f"`{cit.file_path or 'unknown'}` ({cit.line_reference})"
                    )
                with col_status:
                    if cit.status == "valid":
                        st.success("VALID")
                    elif cit.status == "invalid":
                        st.error("INVALID")
                    elif cit.status == "incomplete":
                        st.warning("INCOMPLETE")
                    else:
                        st.info("UNVERIFIED")

                # Checklist
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.markdown(
                    f"File Exists: {'✅' if cit.file_exists else '❌' if cit.file_exists is False else '⚪'}"
                )
                c2.markdown(
                    f"Safe Path: {'✅' if cit.path_is_safe else '❌' if cit.path_is_safe is False else '⚪'}"
                )
                c3.markdown(
                    f"Lines Valid: {'✅' if cit.line_range_valid else '❌' if cit.line_range_valid is False else '⚪'}"
                )
                c4.markdown(
                    f"Symbol Found: {'✅' if cit.symbol_found else '❌' if cit.symbol_found is False else '⚪'}"
                )
                c5.markdown(
                    f"Unit Matched: {'✅' if cit.code_unit_found else '❌' if cit.code_unit_found is False else '⚪'}"
                )

                # Issues
                if cit.issues:
                    st.markdown("**Issues detected:**")
                    for iss in cit.issues:
                        st.error(f"- `[{iss['code']}]` {iss['message']}")

    # Claim Validation
    if data.claims:
        st.markdown("#### Factual Location Claims")
        for cl in data.claims:
            icon = "✅" if cl.status == "valid" else "❌"
            st.markdown(f"- {icon} **Claim:** {cl.claim} (`{cl.status}`)")
            if cl.issues:
                for iss in cl.issues:
                    st.caption(f"  * Issue: {iss.get('message')}")


def render_investigation_trace(data: PresentationData) -> None:
    """Show the investigation trace, including executed tools and relevant findings."""
    st.subheader("⏱️ Agent Investigation Trace")

    # Executed tools overview
    if data.executed_tools:
        st.markdown("**Tools Executed in Workflow:**")
        tools_html = " ".join(f"`{tool}`" for tool in data.executed_tools)
        st.markdown(tools_html)

    if not data.trace_steps:
        st.info("No detailed execution trace records available.")
        return

    st.markdown("#### Step-by-Step Tool Invocations")

    for step in data.trace_steps:
        status_icon = "✅" if step.status == "success" else "❌"
        duration_s = round(step.duration_ms / 1000.0, 3)

        with st.container(border=True):
            col_step, col_tool, col_perf, col_results = st.columns([1, 2, 2, 2])

            with col_step:
                st.markdown(f"**Iter {step.iteration}**")

            with col_tool:
                st.markdown(f"{status_icon} `{step.tool}`")

            with col_perf:
                st.caption(f"Duration: `{duration_s}s` ({step.duration_ms} ms)")

            with col_results:
                st.caption(
                    f"Results: {step.result_count} raw, **{step.retained_count} retained**"
                )

            if step.findings:
                st.caption(f"Retained Evidence IDs: {', '.join(step.findings)}")

            if step.error:
                st.error(f"Error: {step.error}")


def render_empty_or_failed_state(data: PresentationData) -> None:
    """Render graceful warning or error message when evidence is missing or investigation failed."""
    if data.has_errors:
        st.error("⚠️ **Investigation Encountered Errors**")
        st.markdown("The agent encountered errors while executing repository tools:")
        for err in data.errors:
            st.markdown(f"- **`{err['tool']}`**: {err['error']}")

    if not data.has_evidence:
        st.warning("⚠️ **No Relevant Code Evidence Discovered**")
        st.markdown(
            f"The repository investigation completed with termination reason: `{data.termination_reason}`.\n\n"
            "**Suggestions to refine your search:**\n"
            "- Check that the repository directory exists and contains valid `.py` Python files.\n"
            "- Try query keywords matching function, class, or module names (e.g., `calculate_total`, `Client`).\n"
            "- Increase the maximum iterations or evidence capacity in the sidebar controls."
        )
