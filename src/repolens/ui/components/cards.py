from __future__ import annotations

import streamlit as st


def render_feature_card(
    icon: str,
    title: str,
    description: str,
) -> None:
    """Render a reusable RepoLens feature card."""
    st.markdown(
        f"""
        <div class="rl-card">
            <div class="rl-card-icon">{icon}</div>
            <div class="rl-card-title">{title}</div>
            <div class="rl-card-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
