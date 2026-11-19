from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[4]
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


def render_sidebar() -> dict[str, Any]:
    """Render the RepoLens repository and investigation control panel.

    Returns:
        dict[str, Any]: Dictionary containing current navigation page and configuration.
    """
    with st.sidebar:
        if ICON_PATH.exists():
            st.image(str(ICON_PATH), width=48)

        st.markdown("## RepoLens")
        st.caption("Repository Intelligence Workbench")
        st.divider()

        st.markdown("##### WORKSPACE")
        page = st.radio(
            "Navigation",
            options=[
                "Investigations",
                "Overview",
                "Repositories",
                "Code Graph",
                "Evaluation",
            ],
            index=0,
            label_visibility="collapsed",
            key="rl_navigation",
        )

        st.divider()
        st.markdown("##### REPOSITORY TARGET")
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
            st.caption("Demonstration snapshot selected for illustrative exploration.")
        elif repo_mode == "Local repository":
            selected_repo = st.text_input(
                "Local repository path",
                placeholder="C:/Projects/my-repository",
                key="rl_local_repository",
            ).strip()
            st.caption("Local repository ingestion will be expanded in RL-35.")
        else:
            selected_repo = st.text_input(
                "GitHub URL",
                placeholder="https://github.com/owner/repository",
                key="rl_github_url",
            ).strip()
            st.caption("Remote repository cloning will be expanded in RL-35.")

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
        st.markdown(
            """
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
                    background:#3FB68B;
                    border-radius:50%;
                    display:inline-block;
                "></span>
                Workbench Ready
            </div>
            <div style="
                color:#71839C;
                font-size:0.7rem;
                margin-top:0.75rem;
            ">
                RepoLens · v0.1.0 (RL-34)
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
