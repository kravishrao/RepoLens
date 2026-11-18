"""Streamlit Result Presentation Application for RepoLens.

Displays repository-grounded answers, supporting source citations, relevant
code evidence, validation outcomes, and agent investigation execution traces.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st

from repolens.agent.tools import InvestigationTools
from repolens.agent.workflow import investigate_repository
from repolens.graph.code_graph import CodeGraph
from repolens.ingestion.ast_parser import parse_repository
from repolens.ui.components import (
    render_answer_card,
    render_citation_validation,
    render_evidence_viewer,
    render_findings_section,
    render_investigation_trace,
    render_overview_metrics,
)
from repolens.ui.presentation import extract_presentation_data

SAMPLE_QUERIES = [
    "Where is extract_citations defined and how does it parse citations?",
    "How does CodeGraph build relationships between functions and classes?",
    "Explain how BM25 lexical retriever tokenizes source code and scores queries",
    "How does the investigation workflow route after each iteration?",
]


@st.cache_resource(show_spinner=False)
def load_repository_tools(repo_path: str) -> tuple[InvestigationTools, int]:
    """Parse repository source and build cached InvestigationTools."""
    resolved_path = Path(repo_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Repository directory '{repo_path}' does not exist.")

    parse_result = parse_repository(str(resolved_path))
    units = list(parse_result.code_units)
    if not units:
        raise ValueError(
            f"No Python code units discovered in directory '{resolved_path}'."
        )

    graph = CodeGraph()
    graph.build(units)
    tools = InvestigationTools(units, graph, repository_root=str(resolved_path))
    return tools, len(units)


def run_investigation_safe(
    query: str,
    repo_path: str,
    max_iterations: int = 4,
    max_evidence: int = 20,
    min_evidence: int = 0,
) -> dict[str, Any]:
    """Execute investigation workflow with progress updates and error handling."""
    with st.status("🔍 Investigating repository...", expanded=True) as status_box:
        st.write("📂 Loading repository AST and constructing CodeGraph...")
        tools, unit_count = load_repository_tools(repo_path)
        st.write(f"✓ Repository code graph ready ({unit_count} CodeUnits indexed).")

        st.write(f"🤖 Agent executing iterative investigation for: *'{query}'*...")
        state = investigate_repository(
            query=query,
            tools=tools,
            repository_root=str(Path(repo_path).resolve()),
            max_iterations=max_iterations,
            max_evidence=max_evidence,
            min_evidence=min_evidence,
        )

        st.write("🛡️ Citation extractor validating source references...")
        cit_count = len(state.get("citation_validation_results") or [])
        st.write(f"✓ Validated {cit_count} source citations.")

        status_box.update(
            label="✅ Investigation complete!",
            state="complete",
            expanded=False,
        )

    return state


def render_sidebar() -> dict[str, Any]:
    """Render sidebar navigation, repository selector, and investigation parameters."""
    st.sidebar.title("🔍 RepoLens")
    st.sidebar.caption("Agentic Code-Graph Intelligence & Verified Q&A")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Repository Configuration")

    # Default to repolens src directory if available, else current directory
    default_dir = "src/repolens" if os.path.isdir("src/repolens") else "."
    repo_path = st.sidebar.text_input(
        "Repository Directory",
        value=default_dir,
        help="Path to the Python project directory to analyze",
    )

    st.sidebar.subheader("Investigation Parameters")
    max_iterations = st.sidebar.slider(
        "Max Iterations",
        min_value=1,
        max_value=10,
        value=4,
        help="Maximum agent investigation loops",
    )
    max_evidence = st.sidebar.slider(
        "Max Evidence Units",
        min_value=5,
        max_value=50,
        value=20,
        help="Maximum code evidence items to retain",
    )
    min_evidence = st.sidebar.slider(
        "Min Evidence Units",
        min_value=0,
        max_value=20,
        value=0,
        help="Minimum evidence required before terminating early",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Samples")
    selected_sample = st.sidebar.selectbox(
        "Select Sample Query",
        options=["(Custom Query)", *SAMPLE_QUERIES],
        index=0,
    )

    if st.sidebar.button("🧹 Clear Results", use_container_width=True):
        st.session_state.pop("investigation_state", None)
        st.rerun()

    return {
        "repo_path": repo_path,
        "max_iterations": max_iterations,
        "max_evidence": max_evidence,
        "min_evidence": min_evidence,
        "selected_sample": (
            "" if selected_sample == "(Custom Query)" else selected_sample
        ),
    }


def main() -> None:
    """Main application entrypoint."""
    st.set_page_config(
        page_title="RepoLens — Repository Code Intelligence",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    config = render_sidebar()

    st.title("RepoLens: Repository Investigation & Grounded Answers")
    st.markdown(
        "Ask questions about code architecture, implementation details, and function locations. "
        "Answers are strictly grounded in retrieved AST symbols and verified citations."
    )

    # Initial query value from quick sample if chosen
    default_query = config["selected_sample"] or ""

    with st.form("query_form", clear_on_submit=False):
        col_query, col_btn = st.columns([5, 1])

        with col_query:
            query = st.text_input(
                "Enter your repository question:",
                value=default_query,
                placeholder="e.g., Where is calculate_total defined and what does it call?",
                label_visibility="collapsed",
            )

        with col_btn:
            submitted = st.form_submit_button(
                "Investigate 🚀",
                use_container_width=True,
            )

    if submitted:
        if not query.strip():
            st.warning("Please enter a question to investigate.")
        else:
            try:
                state = run_investigation_safe(
                    query=query.strip(),
                    repo_path=config["repo_path"],
                    max_iterations=config["max_iterations"],
                    max_evidence=config["max_evidence"],
                    min_evidence=config["min_evidence"],
                )
                st.session_state["investigation_state"] = state
            except Exception as exc:
                st.error(f"Investigation failed: {exc}")
                st.session_state["investigation_state"] = {
                    "query": query,
                    "final_answer": f"Investigation could not complete due to error: {exc}",
                    "is_complete": False,
                    "termination_reason": "exception",
                    "evidence": [],
                    "errors": [{"tool": "investigation_runner", "error": str(exc)}],
                }

    # Render results if available in session_state
    saved_state = st.session_state.get("investigation_state")
    if saved_state:
        data = extract_presentation_data(saved_state)

        st.markdown("---")
        render_overview_metrics(data)
        st.markdown("")

        tab_ans, tab_findings, tab_evidence, tab_val, tab_trace = st.tabs(
            [
                "💡 Grounded Answer",
                f"🔍 Code Findings ({len(data.all_evidence)})",
                f"📄 Code Evidence ({len(data.all_evidence)})",
                f"🛡️ Citation Validation ({len(data.citations)})",
                f"⏱️ Investigation Trace ({len(data.trace_steps)})",
            ]
        )

        with tab_ans:
            render_answer_card(data)

        with tab_findings:
            render_findings_section(data)

        with tab_evidence:
            render_evidence_viewer(data)

        with tab_val:
            render_citation_validation(data)

        with tab_trace:
            render_investigation_trace(data)
    else:
        # Welcome placeholder
        with st.container(border=True):
            st.subheader("👋 Welcome to RepoLens")
            st.markdown(
                "To start exploring, select a sample query from the sidebar or type a question above.\n\n"
                "**Key Capabilities:**\n"
                "- **Hybrid Code Retrieval:** Combines BM25 lexical matching and semantic embeddings.\n"
                "- **Code Graph Traversal:** Explores callers, callees, and module hierarchy.\n"
                "- **Grounded Synthesis:** Produces responses citing verified file paths and line ranges.\n"
                "- **Citation Validation:** Validates that every citation exists in repository files."
            )


if __name__ == "__main__":
    main()
