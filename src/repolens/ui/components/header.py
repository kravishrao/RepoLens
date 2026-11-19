from __future__ import annotations

from pathlib import Path

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
            if LOGO_PATH.exists():
                st.image(str(LOGO_PATH), width=48)

        with identity_col:
            st.markdown(
                """
                <div class="rl-eyebrow">
                    REPOSITORY INTELLIGENCE
                </div>
                <div class="rl-page-title">
                    RepoLens
                </div>
                <div class="rl-description">
                    Evidence-grounded Python codebase investigation
                </div>
                """,
                unsafe_allow_html=True,
            )

    with repository:
        st.markdown(
            f"""
            <div style="text-align:right;">
                <div style="
                    color:#71839C;
                    font-size:0.72rem;
                    letter-spacing:0.08em;
                    text-transform:uppercase;
                ">
                    WORKSPACE
                </div>
                <div style="
                    color:#F0F6FC;
                    font-size:1rem;
                    font-weight:600;
                    margin-top:0.25rem;
                ">
                    {page_title}
                </div>
                <a href="{GITHUB_URL}"
                   target="_blank"
                   style="
                       display:inline-block;
                       margin-top:0.5rem;
                       color:#58A6FF;
                       font-size:0.82rem;
                       text-decoration:none;
                   ">
                    GitHub Repository &#8599;
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )
