from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[4]

CANDIDATE_LOGO_PATHS = [
    PROJECT_ROOT / "assets" / "LOGO.png",
    Path.cwd() / "assets" / "LOGO.png",
]
LOGO_PATH = next(
    (p for p in CANDIDATE_LOGO_PATHS if p.exists()),
    PROJECT_ROOT / "assets" / "LOGO.png",
)

CANDIDATE_ICON_PATHS = [
    PROJECT_ROOT / "assets" / "ICON.png",
    Path.cwd() / "assets" / "ICON.png",
]
ICON_PATH = next(
    (p for p in CANDIDATE_ICON_PATHS if p.exists()),
    PROJECT_ROOT / "assets" / "ICON.png",
)

PIPELINE_MODES: dict[str, str] = {
    "B5: Full RepoLens": "Hybrid retrieval + graph + agent + verification",
    "B4: Agentic RAG": "Agent-driven multi-step investigation",
    "B3: Graph RAG": "Retrieval with code graph expansion",
    "B2: Hybrid RAG": "BM25 + semantic retrieval + reranking",
    "B1: AST-aware RAG": "Structure-aware CodeUnit retrieval",
    "B0: Naive RAG": "Fixed-size chunks + embeddings",
}

DEMO_REPOSITORIES: list[str] = [
    "pallets/flask (illustrative snapshot)",
    "psf/requests (illustrative snapshot)",
    "tiangolo/fastapi (illustrative snapshot)",
    "RepoLens (illustrative snapshot)",
]

PRESET_QUERIES: list[str] = [
    "Custom question",
    "Where is the retrieval index constructed?",
    "Which functions call the ranking fusion method?",
    "Trace a query from the agent to source-file reading.",
    "How are AST CodeUnits extracted and validated?",
]


def _set_pending_nav(page: str) -> None:
    """Set navigation page for the next Streamlit run."""
    st.session_state["pending_navigation"] = page
    with contextlib.suppress(Exception):
        st.session_state["rl_navigation"] = page


def _switch_repo() -> None:
    """Clear active repository and navigate to Repositories workspace."""
    st.session_state["active_repository"] = None
    _set_pending_nav("Repositories")


def render_sidebar() -> dict[str, Any]:
    """Render the RepoLens repository and investigation control panel.

    Returns:
        dict[str, Any]: Dictionary containing current navigation page and configuration.
    """
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=180)
            st.caption("Repository Intelligence Workbench")
        elif ICON_PATH.exists():
            st.image(str(ICON_PATH), width=48)
            st.markdown("## RepoLens")
            st.caption("Repository Intelligence Workbench")
        else:
            st.markdown("## RepoLens")
            st.caption("Repository Intelligence Workbench")

        st.divider()

        st.markdown("##### WORKSPACE")
        if st.session_state.get("pending_navigation"):
            st.session_state["rl_navigation"] = st.session_state.pop(
                "pending_navigation"
            )

        page = st.radio(
            "Navigation",
            options=[
                "Investigations",
                "Repositories",
                "Overview",
                "Code Graph",
                "Evaluation",
            ],
            index=0,
            label_visibility="collapsed",
            key="rl_navigation",
        )

        st.divider()
        st.markdown("##### REPOSITORY TARGET")

        active_repo = st.session_state.get("active_repository")

        if active_repo is not None:
            st.markdown(
                f"""
                <div style="
                    background: #10251F;
                    border: 1px solid #285B4C;
                    border-radius: 8px;
                    padding: 0.6rem 0.8rem;
                    margin-bottom: 0.5rem;
                ">
                    <div style="color: #3FB68B; font-weight: 700; font-size: 0.75rem; letter-spacing: 0.05em; text-transform: uppercase;">
                        ● ACTIVE REPOSITORY
                    </div>
                    <div style="color: #F0F6FC; font-size: 0.95rem; font-weight: 600; margin-top: 0.2rem;">
                        {active_repo.name}
                    </div>
                    <div style="color: #71839C; font-size: 0.75rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                        {active_repo.root_path}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_manage, col_switch = st.columns(2)
            with col_manage:
                st.button(
                    "Manage repo",
                    key="sb_btn_manage",
                    on_click=_set_pending_nav,
                    args=("Repositories",),
                    width="stretch",
                )
            with col_switch:
                st.button(
                    "Switch",
                    key="sb_btn_switch",
                    on_click=_switch_repo,
                    width="stretch",
                )

            selected_repo = active_repo.name
            repo_mode = active_repo.source_type
        else:
            repo_mode = st.radio(
                "Repository source",
                ["Demo snapshot", "Local repository", "GitHub URL"],
                key="rl_repo_mode",
                label_visibility="collapsed",
            )

            if repo_mode == "Demo snapshot":
                selected_repo = st.selectbox(
                    "Select demonstration snapshot",
                    DEMO_REPOSITORIES,
                    key="rl_demo_repository",
                )
                st.caption(
                    "Demonstration snapshot selected for illustrative exploration."
                )
            elif repo_mode == "Local repository":
                selected_repo = st.text_input(
                    "Local repository path",
                    placeholder="C:/Projects/my-repository",
                    key="rl_local_repository",
                ).strip()
                st.caption("Load this path in the Repositories workspace.")
            else:
                selected_repo = st.text_input(
                    "GitHub URL",
                    placeholder="https://github.com/owner/repository",
                    key="rl_github_url",
                ).strip()
                st.caption("Clone and load in the Repositories workspace.")

            st.button(
                "📂 Load in Repositories",
                key="sb_btn_goto_repos",
                on_click=_set_pending_nav,
                args=("Repositories",),
                width="stretch",
            )

        st.divider()
        st.markdown("##### INDEXED STATISTICS")
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.metric(
                "Files",
                str(active_repo.file_count) if active_repo else "--",
            )
        with stat_col2:
            st.metric(
                "Code units",
                str(active_repo.code_unit_count) if active_repo else "--",
            )
        st.metric(
            "Graph edges",
            str(active_repo.graph_edge_count) if active_repo else "--",
        )

        st.divider()
        st.markdown("##### PIPELINE CONFIGURATION")
        baseline_mode = st.selectbox(
            "Pipeline configuration",
            options=list(PIPELINE_MODES.keys()),
            index=0,
            key="rl_baseline_mode",
        )
        st.caption(PIPELINE_MODES[baseline_mode])

        with st.expander("Advanced retrieval & graph settings"):
            top_k = st.slider(
                "Retrieval Top-K",
                min_value=1,
                max_value=20,
                value=5,
                key="rl_top_k",
            )
            max_hops = st.slider(
                "Maximum graph traversal hops",
                min_value=0,
                max_value=3,
                value=2,
                key="rl_max_hops",
            )
            enable_reranker = st.checkbox(
                "Enable cross-encoder reranking",
                value=True,
                key="rl_enable_reranker",
            )
            enable_verification = st.checkbox(
                "Enable citation verification",
                value=True,
                key="rl_enable_verification",
            )
            max_agent_steps = st.slider(
                "Maximum agent steps",
                min_value=1,
                max_value=15,
                value=8,
                key="rl_max_agent_steps",
            )

        st.divider()
        st.markdown("##### EXAMPLE QUESTIONS")
        preset_query = st.selectbox(
            "Choose a query preset",
            PRESET_QUERIES,
            key="rl_preset_query",
        )

        st.divider()
        status_color = "#3FB68B" if active_repo else "#E5AA55"
        status_text = (
            f"Ready ({active_repo.name})" if active_repo else "Awaiting Repository"
        )
        st.markdown(
            f"""
            <div style="
                display:flex;
                align-items:center;
                gap:8px;
                color:#9BAEC5;
                font-size:0.8rem;
            ">
                <span style="
                    width:8px;
                    height:8px;
                    background:{status_color};
                    border-radius:50%;
                    display:inline-block;
                "></span>
                {status_text}
            </div>
            <div style="
                color:#71839C;
                font-size:0.7rem;
                margin-top:0.75rem;
            ">
                RepoLens · v0.1.0 (RL-35)
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "page": page,
        "repo_mode": repo_mode,
        "selected_repo": selected_repo,
        "baseline_mode": baseline_mode,
        "top_k": top_k,
        "max_hops": max_hops,
        "enable_reranker": enable_reranker,
        "enable_verification": enable_verification,
        "max_agent_steps": max_agent_steps,
        "preset_query": preset_query,
    }
