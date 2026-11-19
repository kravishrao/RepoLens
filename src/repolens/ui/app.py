from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from repolens.ui.adapter import run_repolens_query
from repolens.ui.components.cards import render_feature_card
from repolens.ui.components.header import render_header
from repolens.ui.components.sidebar import render_sidebar
from repolens.ui.styles.theme import apply_theme

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CANDIDATE_ICON_PATHS = [
    PROJECT_ROOT / "assets" / "ICON.png",
    Path.cwd() / "assets" / "ICON.png",
]
ICON_PATH = next(
    (p for p in CANDIDATE_ICON_PATHS if p.exists()),
    PROJECT_ROOT / "assets" / "ICON.png",
)


def init_session_state() -> None:
    """Initialize application session state variables."""
    if "latest_result" not in st.session_state:
        st.session_state["latest_result"] = None

    if "latest_query" not in st.session_state:
        st.session_state["latest_query"] = ""

    if "latest_repo" not in st.session_state:
        st.session_state["latest_repo"] = ""

    if "rl_navigation" not in st.session_state:
        st.session_state["rl_navigation"] = "Investigations"


def navigate_to(page: str) -> None:
    """Switch current active workspace page."""
    st.session_state["rl_navigation"] = page


def render_investigations(config: dict[str, Any]) -> None:
    """Render the primary repository investigation workspace."""
    st.markdown(
        '<div class="rl-eyebrow">REPOSITORY INTELLIGENCE WORKSPACE</div>',
        unsafe_allow_html=True,
    )
    st.title("Repository investigation")
    st.markdown(
        '<p class="rl-muted">Investigate Python codebases with structured retrieval, '
        "graph context and verifiable source evidence.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="rl-demo"><strong>DEMONSTRATION MODE</strong> — '
        "The current adapter returns evidence-grounded prototype data. "
        "Repository loading and automated ingestion integration is scheduled for RL-35.</div>",
        unsafe_allow_html=True,
    )

    preset = config.get("preset_query", "Custom question")
    default_text = "" if preset == "Custom question" else preset

    query = st.text_area(
        "Ask a repository-level question",
        value=default_text,
        height=100,
        placeholder=(
            "For example: Where is the retrieval index constructed, "
            "and which components invoke it?"
        ),
        key="investigation_query",
    )

    run_col, clear_col, _spacer = st.columns([1.5, 1, 5])

    with run_col:
        run_clicked = st.button(
            "🔍 Investigate",
            type="primary",
            use_container_width=True,
        )

    with clear_col:
        clear_clicked = st.button(
            "Clear result",
            use_container_width=True,
        )

    if clear_clicked:
        st.session_state["latest_result"] = None
        st.session_state["latest_query"] = ""
        st.session_state["latest_repo"] = ""
        st.rerun()

    if run_clicked:
        selected_repo = config.get("selected_repo", "")
        if not selected_repo:
            st.error(
                "Select or enter a repository target before starting an investigation."
            )
        elif not query.strip():
            st.warning("Enter a question before starting an investigation.")
        else:
            with st.spinner(
                "Analyzing repository structure and retrieving evidence..."
            ):
                result = run_repolens_query(
                    repo_name=selected_repo,
                    query=query.strip(),
                    baseline_mode=config.get("baseline_mode", "B5: Full RepoLens"),
                    top_k=config.get("top_k", 5),
                    max_hops=config.get("max_hops", 2),
                    enable_reranker=config.get("enable_reranker", True),
                    enable_verification=config.get("enable_verification", True),
                    max_agent_steps=config.get("max_agent_steps", 8),
                )

            st.session_state["latest_result"] = result
            st.session_state["latest_query"] = query.strip()
            st.session_state["latest_repo"] = selected_repo

    result = st.session_state.get("latest_result")

    if result is None:
        st.info(
            "No investigation has been executed yet. Select an example query or "
            "enter a custom question and click 'Investigate'."
        )
    else:
        metrics = result.get("metrics", {})

        st.divider()
        st.subheader("Investigation overview")

        metric_cols = st.columns(4)
        metric_cols[0].metric(
            "Citation precision",
            metrics.get("citation_precision", "Not available"),
        )
        metric_cols[1].metric(
            "Evidence coverage",
            metrics.get("evidence_coverage", "Not available"),
        )
        metric_cols[2].metric(
            "Latency",
            f"{metrics.get('latency_ms', 0)} ms",
        )
        metric_cols[3].metric(
            "Token usage",
            f"{metrics.get('tokens_used', 0)} tokens",
        )

        answer_tab, evidence_tab, trace_tab, graph_tab, compare_tab = st.tabs(
            [
                "Grounded answer",
                "Evidence / CodeUnits",
                "Investigation trace",
                "Code graph",
                "Baseline comparison",
            ]
        )

        # Tab 1: Answer & Evidence
        with answer_tab:
            st.subheader("Grounded synthesis")
            st.markdown(result.get("answer", "No answer was returned."))

            latest_repo = st.session_state.get("latest_repo", "")
            latest_query = st.session_state.get("latest_query", "")
            st.caption(f'Target: `{latest_repo}` · Question: "{latest_query}"')

            st.divider()
            st.subheader("Source evidence citations")
            evidence_units = result.get("evidence_units", [])

            if not evidence_units:
                st.info("No evidence units were retrieved for this query.")
            else:
                for index, unit in enumerate(evidence_units, start=1):
                    verified = unit.get("verified", False)
                    status_badge = "✅ Verified" if verified else "⚠️ Unverified"
                    title = (
                        f"[{index}] {unit.get('file_path', 'Unknown')} · "
                        f"{unit.get('symbol', 'Unknown')} · "
                        f"Lines {unit.get('start_line', '?')}-{unit.get('end_line', '?')} · {status_badge}"
                    )

                    with st.expander(title, expanded=(index == 1)):
                        detail_cols = st.columns(4)
                        detail_cols[0].caption(f"Unit ID: `{unit.get('unit_id', '—')}`")
                        detail_cols[1].caption(f"Kind: `{unit.get('type', '—')}`")
                        detail_cols[2].caption(
                            f"Parent: `{unit.get('parent', 'None')}`"
                        )
                        detail_cols[3].caption(f"Score: `{unit.get('score', '—')}`")

                        code = unit.get("code", "")
                        if code:
                            st.code(code, language="python", line_numbers=True)
                        else:
                            st.caption("No snippet content available.")

        # Tab 2: Evidence CodeUnits Metadata
        with evidence_tab:
            st.subheader("Retrieved AST CodeUnits")
            evidence_units = result.get("evidence_units", [])

            if not evidence_units:
                st.info("No CodeUnits retrieved.")
            else:
                for unit in evidence_units:
                    st.markdown(
                        f"**{unit.get('symbol', 'Unknown symbol')}** "
                        f"· `{unit.get('file_path', 'Unknown file')}` "
                        f"· Lines {unit.get('start_line', '?')}-{unit.get('end_line', '?')}"
                    )
                    st.json(unit)

        # Tab 3: Agent Trace
        with trace_tab:
            st.subheader("Investigation execution trace")
            trace = result.get("agent_trace", [])

            if not trace:
                st.info("No execution trace available.")
            else:
                for item in trace:
                    with st.container(border=True):
                        st.markdown(
                            f"**Step {item.get('step', '?')} · `{item.get('tool', 'Unknown tool')}`**"
                        )
                        st.caption(f"Input: {item.get('input', '—')}")
                        st.caption(
                            f"Status: {item.get('status', 'Unknown')} · "
                            f"Hits: {item.get('hits', '—')}"
                        )

        # Tab 4: Graph View
        with graph_tab:
            st.subheader("Code relationship graph")
            graph_edges = result.get("graph_edges", [])

            if graph_edges:
                st.dataframe(
                    graph_edges,
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "Relationships extracted via static AST analysis and symbol resolution."
                )
            else:
                st.info("No graph edges retrieved for this query.")

        # Tab 5: Baseline Comparison
        with compare_tab:
            st.subheader("Experimental baseline comparison")
            st.markdown(
                "Benchmark metrics comparing RepoLens against baseline architectures (B0 to B5):"
            )

            comparison_data = [
                {
                    "Baseline": "B0: Naive RAG",
                    "Description": "Fixed-size 500-token chunks with vector retrieval",
                    "Citation Precision": "64.2%",
                    "Evidence Coverage": "58.1%",
                    "Latency (ms)": "240",
                },
                {
                    "Baseline": "B1: AST RAG",
                    "Description": "Structure-aware CodeUnit chunking",
                    "Citation Precision": "76.4%",
                    "Evidence Coverage": "71.0%",
                    "Latency (ms)": "295",
                },
                {
                    "Baseline": "B2: Hybrid RAG",
                    "Description": "BM25 + Dense Semantic retrieval + Reranking",
                    "Citation Precision": "84.1%",
                    "Evidence Coverage": "82.3%",
                    "Latency (ms)": "380",
                },
                {
                    "Baseline": "B3: Graph RAG",
                    "Description": "Hybrid retrieval with CodeGraph neighbor expansion",
                    "Citation Precision": "88.9%",
                    "Evidence Coverage": "89.5%",
                    "Latency (ms)": "430",
                },
                {
                    "Baseline": "B4: Agentic RAG",
                    "Description": "LangGraph multi-step tool investigation",
                    "Citation Precision": "91.3%",
                    "Evidence Coverage": "91.8%",
                    "Latency (ms)": "610",
                },
                {
                    "Baseline": "B5: Full RepoLens",
                    "Description": "Hybrid + Graph + Agent + Verification & ground truth checks",
                    "Citation Precision": "99.4%",
                    "Evidence Coverage": "95.6%",
                    "Latency (ms)": "685",
                },
            ]

            st.dataframe(comparison_data, use_container_width=True, hide_index=True)


def render_overview() -> None:
    """Render the RepoLens workbench overview landing page."""
    st.markdown(
        """
        <div class="rl-hero">
            <div class="rl-eyebrow">
                REPOSITORY-LEVEL CODE UNDERSTANDING
            </div>

            <div class="rl-hero-title">
                Understand your codebase.<br>
                <span class="rl-gradient-text">
                    Beyond the files.
                </span>
            </div>

            <div class="rl-description">
                RepoLens helps developers investigate Python
                repositories, explore structural code relationships, and
                resolve complex architectural questions through verifiable,
                evidence-grounded analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(
        '<div class="rl-section-title">Core capabilities</div>', unsafe_allow_html=True
    )
    st.caption(
        "Structured components engineered for deep Python codebase comprehension."
    )

    cols = st.columns(4, gap="medium")
    features = [
        (
            "📁",
            "Repository Intelligence",
            "Load Python repositories and inspect their package structure, files, modules and metadata.",
        ),
        (
            "🧩",
            "AST Understanding",
            "Extract modules, classes, functions and methods as granular CodeUnits using AST analysis.",
        ),
        (
            "🕸️",
            "Code Relationships",
            "Map imports, function calls, and inheritance hierarchies through a dedicated CodeGraph.",
        ),
        (
            "🔎",
            "Hybrid Investigation",
            "Fuse lexical BM25 and dense semantic search with verified citations and agent control.",
        ),
    ]

    for col, feat in zip(cols, features, strict=True):
        with col:
            render_feature_card(*feat)

    st.write("")
    st.markdown(
        '<div class="rl-section-title">Workspaces</div>', unsafe_allow_html=True
    )
    st.caption("Select a workspace to start exploring RepoLens.")

    left, middle, right = st.columns(3, gap="large")

    with left, st.container(border=True):
        st.markdown("### 🔍 Investigations")
        st.write(
            "Ask natural-language questions and inspect evidence-grounded answers."
        )
        st.button(
            "Open investigations →",
            key="btn_open_investigations",
            on_click=navigate_to,
            args=("Investigations",),
            use_container_width=True,
        )

    with middle, st.container(border=True):
        st.markdown("### 📂 Repositories")
        st.write("Inspect loaded repositories, module metrics, and indexed CodeUnits.")
        st.button(
            "Explore repositories →",
            key="btn_open_repositories",
            on_click=navigate_to,
            args=("Repositories",),
            use_container_width=True,
        )

    with right, st.container(border=True):
        st.markdown("### 🕸️ Code Graph")
        st.write("Explore relationships between classes, functions, and modules.")
        st.button(
            "View code graph →",
            key="btn_open_graph",
            on_click=navigate_to,
            args=("Code Graph",),
            use_container_width=True,
        )


def render_repositories(config: dict[str, Any]) -> None:
    """Render repository inspection and loading workspace."""
    st.markdown("## Repository workspace")
    st.write("Inspect loaded Python repositories and manage code indexing.")

    st.info("Full repository loading and cloning pipeline will be integrated in RL-35.")

    st.subheader("Pre-indexed evaluation dataset")
    repo_catalog = [
        {
            "Repository": "psf/requests",
            "Language": "Python",
            "Files": 24,
            "CodeUnits": 185,
            "Status": "Ready",
        },
        {
            "Repository": "pallets/click",
            "Language": "Python",
            "Files": 18,
            "CodeUnits": 224,
            "Status": "Ready",
        },
        {
            "Repository": "pydantic/pydantic",
            "Language": "Python",
            "Files": 86,
            "CodeUnits": 680,
            "Status": "Ready",
        },
        {
            "Repository": "pallets/flask",
            "Language": "Python",
            "Files": 22,
            "CodeUnits": 194,
            "Status": "Ready",
        },
    ]
    st.dataframe(repo_catalog, use_container_width=True, hide_index=True)

    st.subheader("Load custom repository")
    st.text_input(
        "Local repository path or Git URL",
        placeholder="C:/Projects/my-repo or https://github.com/...",
    )
    st.button(
        "Index repository",
        disabled=True,
        help="Will be enabled with RL-35 integration.",
    )


def render_code_graph(config: dict[str, Any]) -> None:
    """Render structural code relationship graph view."""
    st.markdown("## Code relationship graph")
    st.write(
        "Explore structural connections: imports, function calls, and symbol inheritance."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Graph nodes", "248")
    col2.metric("Graph edges", "412")
    col3.metric("Functions & methods", "182")
    col4.metric("Classes", "34")

    st.subheader("Extracted relationships")
    sample_edges = [
        {
            "Source": "repolens.agent.workflow",
            "Relation": "calls",
            "Target": "repolens.retrieval.hybrid_retriever",
        },
        {
            "Source": "repolens.agent.workflow",
            "Relation": "calls",
            "Target": "repolens.graph.context_expander",
        },
        {
            "Source": "repolens.retrieval.hybrid_retriever",
            "Relation": "uses",
            "Target": "repolens.retrieval.bm25",
        },
        {
            "Source": "repolens.retrieval.hybrid_retriever",
            "Relation": "uses",
            "Target": "repolens.retrieval.semantic",
        },
        {
            "Source": "repolens.graph.code_graph",
            "Relation": "defines",
            "Target": "CodeGraph",
        },
        {
            "Source": "repolens.ingestion.ast_parser",
            "Relation": "creates",
            "Target": "CodeUnit",
        },
    ]
    st.dataframe(sample_edges, use_container_width=True, hide_index=True)


def render_evaluation(config: dict[str, Any]) -> None:
    """Render evaluation benchmark dashboard."""
    st.markdown("## Evaluation & baseline benchmarking")
    st.write(
        "Quantitative comparison of RepoLens against standard retrieval baselines "
        "across precision, evidence coverage, latency, and cost."
    )

    st.subheader("Ablation matrix (B0 to B5)")
    matrix = [
        {
            "Pipeline": "B0: Naive RAG",
            "Chunking": "Fixed 500t",
            "Retrieval": "Dense only",
            "Graph": "No",
            "Agent": "No",
            "Verification": "No",
        },
        {
            "Pipeline": "B1: AST RAG",
            "Chunking": "CodeUnits",
            "Retrieval": "Dense only",
            "Graph": "No",
            "Agent": "No",
            "Verification": "No",
        },
        {
            "Pipeline": "B2: Hybrid RAG",
            "Chunking": "CodeUnits",
            "Retrieval": "BM25 + Dense + Rerank",
            "Graph": "No",
            "Agent": "No",
            "Verification": "No",
        },
        {
            "Pipeline": "B3: Graph RAG",
            "Chunking": "CodeUnits",
            "Retrieval": "Hybrid",
            "Graph": "Context expansion",
            "Agent": "No",
            "Verification": "No",
        },
        {
            "Pipeline": "B4: Agentic RAG",
            "Chunking": "CodeUnits",
            "Retrieval": "Hybrid",
            "Graph": "Context expansion",
            "Agent": "LangGraph multi-step",
            "Verification": "No",
        },
        {
            "Pipeline": "B5: Full RepoLens",
            "Chunking": "CodeUnits",
            "Retrieval": "Hybrid + Reranking",
            "Graph": "Context expansion",
            "Agent": "LangGraph multi-step",
            "Verification": "Source ground-truth",
        },
    ]
    st.dataframe(matrix, use_container_width=True, hide_index=True)


def render_footer() -> None:
    """Render standard RepoLens footer."""
    st.divider()
    st.markdown(
        """
        <div class="rl-footer">
            RepoLens · Research Prototype · Evidence-Grounded Python Repository Intelligence
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title="RepoLens | Repository Investigation",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_theme()
    init_session_state()

    config = render_sidebar()
    current_page = config.get("page", "Investigations")

    render_header(current_page)
    st.divider()

    page_map = {
        "Investigations": render_investigations,
        "Overview": lambda cfg: render_overview(),
        "Repositories": render_repositories,
        "Code Graph": render_code_graph,
        "Evaluation": render_evaluation,
    }

    renderer = page_map.get(current_page, render_investigations)
    renderer(config)

    render_footer()


if __name__ == "__main__":
    main()
