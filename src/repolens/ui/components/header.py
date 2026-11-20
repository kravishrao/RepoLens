"""Modern Application Header for RepoLens."""

from __future__ import annotations

from pathlib import Path

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

CANDIDATE_LOGO_PATHS = [
    PROJECT_ROOT / "assets" / "LOGO.png",
    Path.cwd() / "assets" / "LOGO.png",
]
LOGO_PATH = next(
    (p for p in CANDIDATE_LOGO_PATHS if p.exists()),
    PROJECT_ROOT / "assets" / "LOGO.png",
)

GITHUB_URL = "https://github.com/kravishrao/RepoLens"


def render_header(page_title: str) -> None:
    """Render the RepoLens application header."""
    brand, repository = st.columns([5, 2], vertical_alignment="center")

    with brand:
        logo_col, identity_col = st.columns(
            [1, 8],
            vertical_alignment="center",
            gap="small",
        )

        with logo_col:
            if ICON_PATH.exists():
                st.image(str(ICON_PATH), width=56)
            elif LOGO_PATH.exists():
                st.image(str(LOGO_PATH), width=56)

        with identity_col:
            st.markdown(
                """
                <div class="rl-eyebrow">
                    REPOSITORY INTELLIGENCE ENGINE
                </div>
                <div class="rl-page-title" style="margin: 0.15rem 0; font-size: 1.75rem;">
                    RepoLens
                </div>
                <div class="rl-description" style="font-size: 0.86rem; line-height: 1.4;">
                    Evidence-grounded Python codebase analysis · AST parsing · CodeGraph · Citation verification
                </div>
                """,
                unsafe_allow_html=True,
            )

    with repository:
        active_repo = st.session_state.get("active_repository")
        active_badge = (
            f"""<div style="display:inline-flex; align-items:center; gap:0.4rem; padding:0.25rem 0.65rem; background:#0E201B; border:1px solid rgba(52, 211, 153, 0.4); border-radius:20px; font-size:0.75rem; color:#34D399; font-weight:600; margin-bottom:0.35rem;">
                <span style="font-size:0.6rem;">●</span> ACTIVE: {active_repo.name} ({active_repo.file_count} files)
            </div>"""
            if active_repo
            else """<div style="display:inline-flex; align-items:center; gap:0.4rem; padding:0.25rem 0.65rem; background:rgba(30, 41, 59, 0.5); border:1px solid rgba(255, 255, 255, 0.08); border-radius:20px; font-size:0.75rem; color:#94A3B8; margin-bottom:0.35rem;">
                <span>○</span> DEMO MODE (Select repo below)
            </div>"""
        )

        st.markdown(
            f"""
            <div style="text-align:right;">
                {active_badge}
                <div style="
                    color:#64748B;
                    font-size:0.72rem;
                    letter-spacing:0.08em;
                    text-transform:uppercase;
                    font-weight:600;
                    font-family:'JetBrains Mono', monospace;
                ">
                    WORKSPACE: <span style="color:#F8FAFC;">{page_title}</span>
                </div>
                <a href="{GITHUB_URL}"
                   target="_blank"
                   style="
                       display:inline-block;
                       margin-top:0.35rem;
                       color:#38BDF8;
                       font-size:0.8rem;
                       text-decoration:none;
                       font-weight:500;
                   ">
                    GitHub Repository ↗
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )
