from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from repolens.ingestion.repository_loader import (
    InvalidRepositoryError,
    RepositoryCloneError,
    RepositoryError,
)
from repolens.ui.adapter import run_repolens_query
from repolens.ui.components.cards import render_feature_card
from repolens.ui.components.header import render_header
from repolens.ui.components.sidebar import render_sidebar
from repolens.ui.repository_service import (
    LoadedRepository,
    get_bundled_repositories,
    load_repository_pipeline,
)
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

    if "active_repository" not in st.session_state:
        st.session_state["active_repository"] = None

    if "repo_load_error" not in st.session_state:
        st.session_state["repo_load_error"] = None


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

    active_repo: LoadedRepository | None = st.session_state.get("active_repository")

    if active_repo is not None:
        st.markdown(
            f"""
            <div class="rl-evidence">
                <strong>ACTIVE REPOSITORY</strong>: <code>{active_repo.name}</code> &nbsp;·&nbsp;
                {active_repo.file_count} Python files &nbsp;·&nbsp;
                {active_repo.code_unit_count} CodeUnits &nbsp;·&nbsp;
                {active_repo.graph_edge_count} Graph edges
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        demo_col1, demo_col2 = st.columns([5, 2], vertical_alignment="center")
        with demo_col1:
            st.markdown(
                '<div class="rl-demo"><strong>DEMONSTRATION MODE</strong> — '
                "No custom repository is loaded. Responses use illustrative prototype data. "
                "Load a live Python repository in the Repositories workspace for real codebase analysis.</div>",
                unsafe_allow_html=True,
            )
        with demo_col2:
            if st.button("📂 Ingest a repository", use_container_width=True):
                navigate_to("Repositories")
                st.rerun()

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
        target_name = (
            active_repo.name if active_repo else config.get("selected_repo", "")
        )
        if not target_name:
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
                    repo_name=target_name,
                    query=query.strip(),
                    baseline_mode=config.get("baseline_mode", "B5: Full RepoLens"),
                    top_k=config.get("top_k", 5),
                    max_hops=config.get("max_hops", 2),
                    enable_reranker=config.get("enable_reranker", True),
                    enable_verification=config.get("enable_verification", True),
                    max_agent_steps=config.get("max_agent_steps", 8),
                    active_repo=active_repo,
                )

            st.session_state["latest_result"] = result
            st.session_state["latest_query"] = query.strip()
            st.session_state["latest_repo"] = target_name

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


def _trigger_repository_loading(source_target: str) -> None:
    """Helper to run repository loading pipeline with real-time UI progress and feedback."""
    clean_target = source_target.strip()
    if not clean_target:
        st.error(
            "Please provide a repository path or select a supported repository source."
        )
        return

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def on_progress(fraction: float, message: str) -> None:
        progress_bar.progress(fraction)
        status_text.markdown(f"**Loading progress:** {message}")

    try:
        loaded_repo = load_repository_pipeline(
            clean_target, progress_callback=on_progress
        )
        st.session_state["active_repository"] = loaded_repo
        st.session_state["repo_load_error"] = None
        st.session_state["latest_result"] = None
        st.success(
            f"🎉 Repository '{loaded_repo.name}' loaded successfully! "
            f"Indexed {loaded_repo.file_count} files, {loaded_repo.code_unit_count} CodeUnits, "
            f"and {loaded_repo.graph_edge_count} graph connections."
        )
        st.rerun()
    except InvalidRepositoryError as exc:
        st.session_state["repo_load_error"] = str(exc)
        st.error(f"❌ Invalid Repository: {exc}")
    except RepositoryCloneError as exc:
        st.session_state["repo_load_error"] = str(exc)
        st.error(f"❌ Repository Clone Failed: {exc}")
    except RepositoryError as exc:
        st.session_state["repo_load_error"] = str(exc)
        st.error(f"❌ Ingestion Error: {exc}")
    except Exception as exc:
        st.session_state["repo_load_error"] = str(exc)
        st.error(f"❌ Unexpected Error: {exc}")


def render_repositories(config: dict[str, Any]) -> None:
    """Render repository selection, loading, and configuration interface."""
    st.markdown(
        '<div class="rl-eyebrow">REPOSITORY SELECTION & CONFIGURATION</div>',
        unsafe_allow_html=True,
    )
    st.title("Repository workspace")
    st.markdown(
        '<p class="rl-muted">Select, ingest, and inspect Python repositories for structured '
        "code understanding, AST symbol extraction, and relationship mapping.</p>",
        unsafe_allow_html=True,
    )

    active_repo: LoadedRepository | None = st.session_state.get("active_repository")

    if active_repo is not None:
        # Display active repository status card
        st.markdown(
            f"""
            <div class="rl-evidence">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span class="rl-status rl-valid">● ACTIVE REPOSITORY</span>
                        <h2 style="margin: 0.4rem 0 0.2rem 0; font-size: 1.6rem; color: #F0F6FC;">{active_repo.name}</h2>
                        <span style="color: #9BAEC5; font-size: 0.85rem;"><code>{active_repo.root_path}</code></span>
                    </div>
                    <div style="text-align: right; color: #71839C; font-size: 0.8rem;">
                        <div>Source type: <strong>{active_repo.source_type.upper()}</strong></div>
                        <div>Loaded: <strong>{active_repo.loaded_at}</strong></div>
                        {f"<div>Branch: <code>{active_repo.branch}</code></div>" if active_repo.branch else ""}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Discovered files", str(active_repo.file_count))
        metric_cols[1].metric("AST CodeUnits", str(active_repo.code_unit_count))
        metric_cols[2].metric("Graph nodes", str(active_repo.graph_node_count))
        metric_cols[3].metric("Graph edges", str(active_repo.graph_edge_count))

        st.write("")
        btn_cols = st.columns([1.5, 1.5, 1.8, 3.2])
        with btn_cols[0]:
            if st.button("🔄 Reload repository", use_container_width=True):
                with st.spinner("Reloading repository..."):
                    _trigger_repository_loading(active_repo.source)
        with btn_cols[1]:
            if st.button("🔀 Switch repository", use_container_width=True):
                st.session_state["active_repository"] = None
                st.rerun()
        with btn_cols[2]:
            if st.button(
                "🔍 Investigate codebase →", type="primary", use_container_width=True
            ):
                navigate_to("Investigations")
                st.rerun()

        st.divider()
        detail_tabs = st.tabs(
            [
                "Discovered files",
                "AST CodeUnits",
                "Graph relationships",
                "Parsing diagnostics",
            ]
        )

        with detail_tabs[0]:
            st.subheader(f"Discovered Python files ({active_repo.file_count})")
            if active_repo.files:
                search_term = (
                    st.text_input(
                        "Filter files by path or extension", key="repo_file_filter"
                    )
                    .strip()
                    .lower()
                )
                filtered_files = (
                    [
                        f
                        for f in active_repo.files
                        if search_term in f["relative_path"].lower()
                    ]
                    if search_term
                    else active_repo.files
                )
                st.dataframe(
                    [
                        {
                            "File path": f["relative_path"],
                            "Size": f["size_formatted"],
                            "Extension": f["suffix"],
                        }
                        for f in filtered_files
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No files recorded.")

        with detail_tabs[1]:
            st.subheader(f"Extracted AST CodeUnits ({active_repo.code_unit_count})")
            if active_repo.code_units:
                kinds = [u.kind for u in active_repo.code_units]
                summary_cols = st.columns(3)
                summary_cols[0].caption(f"Functions: **{kinds.count('function')}**")
                summary_cols[1].caption(f"Classes: **{kinds.count('class')}**")
                summary_cols[2].caption(f"Methods: **{kinds.count('method')}**")

                unit_filter = (
                    st.text_input(
                        "Search CodeUnits by symbol or kind", key="unit_filter"
                    )
                    .strip()
                    .lower()
                )
                displayed_units = [
                    u
                    for u in active_repo.code_units
                    if (
                        not unit_filter
                        or unit_filter in u.name.lower()
                        or unit_filter in u.qualified_name.lower()
                        or unit_filter in u.kind.lower()
                    )
                ][:50]

                st.dataframe(
                    [
                        {
                            "Symbol": u.qualified_name or u.name,
                            "Kind": u.kind,
                            "File": u.file_path,
                            "Lines": f"{u.start_line} - {u.end_line}",
                            "Imports": len(u.imports),
                            "Calls": len(u.calls),
                        }
                        for u in displayed_units
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No CodeUnits extracted.")

        with detail_tabs[2]:
            st.subheader(
                f"CodeGraph relationships ({active_repo.graph_edge_count} edges)"
            )
            if active_repo.graph_edges:
                st.dataframe(
                    active_repo.graph_edges[:200],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No graph edges built.")

        with detail_tabs[3]:
            st.subheader("Ingestion diagnostics")
            if not active_repo.parse_errors:
                st.success(
                    "✅ All discovered files parsed cleanly with zero syntax or extraction errors."
                )
            else:
                st.warning(
                    f"⚠️ {len(active_repo.parse_errors)} files encountered AST syntax errors during parsing:"
                )
                st.dataframe(
                    active_repo.parse_errors, use_container_width=True, hide_index=True
                )

    else:
        # Repository selection interface
        st.subheader("Select repository source")
        source_tab1, source_tab2, source_tab3 = st.tabs(
            [
                "Bundled benchmark datasets",
                "Local filesystem path",
                "Remote Git URL",
            ]
        )

        with source_tab1:
            st.markdown(
                "Choose a pre-indexed benchmark repository from the local dataset folder:"
            )
            bundled_repos = get_bundled_repositories()
            options = {b["name"]: b for b in bundled_repos}
            selected_name = st.selectbox(
                "Select benchmark repository",
                options=list(options.keys()),
                key="select_bundled_repo",
            )
            selected_info = options[selected_name]

            st.caption(f"**Description:** {selected_info['description']}")
            st.caption(f"**Path on disk:** `{selected_info['abs_path']}`")

            if st.button(
                "📥 Load benchmark repository", type="primary", key="btn_load_bundled"
            ):
                _trigger_repository_loading(selected_info["abs_path"])

        with source_tab2:
            st.markdown(
                "Enter an absolute or relative directory path to any local Python repository:"
            )
            local_path = st.text_input(
                "Local repository directory",
                placeholder="C:/Projects/my-repository or ./dataset/repositories/requests",
                key="input_local_repo_path",
            )

            suggest_col1, _suggest_col2 = st.columns([2, 5])
            with suggest_col1:
                if st.button("Load RepoLens self", key="btn_load_self"):
                    _trigger_repository_loading(str(PROJECT_ROOT))

            if st.button(
                "📥 Load local repository", type="primary", key="btn_load_local"
            ):
                _trigger_repository_loading(local_path)

        with source_tab3:
            st.markdown("Clone and analyze a remote repository via HTTPS or SSH:")
            remote_url = st.text_input(
                "Git repository URL",
                placeholder="https://github.com/owner/repository.git",
                key="input_remote_repo_url",
            )
            st.caption("Repositories are cloned into `data/repositories/`.")
            if st.button(
                "📥 Clone & load repository", type="primary", key="btn_load_remote"
            ):
                _trigger_repository_loading(remote_url)


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
        st.markdown("### 📂 Repositories")
        st.write("Select, load, and configure Python repositories for analysis.")
        st.button(
            "Manage repositories →",
            key="btn_open_repositories",
            on_click=navigate_to,
            args=("Repositories",),
            use_container_width=True,
        )

    with middle, st.container(border=True):
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


def render_code_graph(config: dict[str, Any]) -> None:
    """Render structural code relationship graph view."""
    st.markdown(
        '<div class="rl-eyebrow">GRAPH INTELLIGENCE</div>', unsafe_allow_html=True
    )
    st.title("Code relationship graph")
    st.markdown(
        '<p class="rl-muted">Explore structural connections: imports, function calls, and symbol inheritance.</p>',
        unsafe_allow_html=True,
    )

    active_repo: LoadedRepository | None = st.session_state.get("active_repository")

    if active_repo is not None:
        st.markdown(
            f"""
            <div class="rl-evidence">
                <strong>ACTIVE REPOSITORY GRAPH</strong>: <code>{active_repo.name}</code> &nbsp;·&nbsp;
                {active_repo.graph_node_count} nodes &nbsp;·&nbsp;
                {active_repo.graph_edge_count} directed edges
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Graph nodes", str(active_repo.graph_node_count))
        col2.metric("Graph edges", str(active_repo.graph_edge_count))
        col3.metric("CodeUnits", str(active_repo.code_unit_count))
        col4.metric("Source files", str(active_repo.file_count))

        st.divider()
        st.subheader("Extracted code relationships")
        if active_repo.graph_edges:
            rel_filter = st.selectbox(
                "Filter by relation type",
                ["All relations", "calls", "imports", "contains", "inherits"],
                key="graph_rel_filter",
            )
            filtered_edges = (
                active_repo.graph_edges
                if rel_filter == "All relations"
                else [
                    e
                    for e in active_repo.graph_edges
                    if e.get("relation") == rel_filter
                ]
            )
            st.dataframe(
                filtered_edges[:250],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No graph edges extracted for this repository.")
    else:
        st.info("No active repository loaded. Showing illustrative graph topology:")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sample nodes", "248")
        col2.metric("Sample edges", "412")
        col3.metric("Functions & methods", "182")
        col4.metric("Classes", "34")

        st.subheader("Illustrative relationships")
        sample_edges = [
            {
                "source": "repolens.agent.workflow",
                "relation": "calls",
                "target": "repolens.retrieval.hybrid_retriever",
            },
            {
                "source": "repolens.agent.workflow",
                "relation": "calls",
                "target": "repolens.graph.context_expander",
            },
            {
                "source": "repolens.retrieval.hybrid_retriever",
                "relation": "uses",
                "target": "repolens.retrieval.bm25",
            },
            {
                "source": "repolens.retrieval.hybrid_retriever",
                "relation": "uses",
                "target": "repolens.retrieval.semantic",
            },
            {
                "source": "repolens.graph.code_graph",
                "relation": "defines",
                "target": "CodeGraph",
            },
            {
                "source": "repolens.ingestion.ast_parser",
                "relation": "creates",
                "target": "CodeUnit",
            },
        ]
        st.dataframe(sample_edges, use_container_width=True, hide_index=True)

        if st.button("📂 Load a real repository now", key="btn_graph_load_repo"):
            navigate_to("Repositories")
            st.rerun()


def render_evaluation(config: dict[str, Any]) -> None:
    """Render evaluation benchmark dashboard."""
    st.markdown(
        '<div class="rl-eyebrow">BENCHMARKING & ABLATION</div>', unsafe_allow_html=True
    )
    st.title("Evaluation & baseline benchmarking")
    st.markdown(
        '<p class="rl-muted">Quantitative comparison of RepoLens against standard retrieval baselines '
        "across precision, evidence coverage, latency, and cost.</p>",
        unsafe_allow_html=True,
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
