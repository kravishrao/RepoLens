"""Modern card components for RepoLens UI dashboard."""

from __future__ import annotations

import streamlit as st


def render_feature_card(
    icon: str,
    title: str,
    description: str,
) -> None:
    """Render a reusable RepoLens feature card."""
    icon_html = f'<div class="rl-card-icon">{icon}</div>' if icon else ""
    st.markdown(
        f"""
        <div class="rl-card">
            {icon_html}
            <div class="rl-card-title">{title}</div>
            <div class="rl-card-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    title: str,
    value: str,
    subtitle: str,
    variant: str = "coral",
    badge_text: str = "",
) -> str:
    """Generate HTML markup for a signature modern gradient KPI card.

    Args:
        title: Metric label (e.g., 'INDEXED CODEUNITS').
        value: Headline numeric or percentage value (e.g., '1,842').
        subtitle: Secondary descriptive text (e.g., 'AST Functions & Classes').
        variant: Gradient style name ('coral', 'cyan', 'emerald', 'purple').
        badge_text: Optional top-right pill counter or micro-tag.
    """
    badge_html = (
        f'<div class="rl-kpi-badge-circle">{badge_text}</div>'
        if badge_text
        else '<div class="rl-kpi-badge-circle">●</div>'
    )

    return f"""
    <div class="rl-kpi-card rl-kpi-{variant}">
        <div class="rl-kpi-top">
            <span class="rl-kpi-label">{title}</span>
            {badge_html}
        </div>
        <div class="rl-kpi-val">{value}</div>
        <div class="rl-kpi-sub">{subtitle}</div>
    </div>
    """


def render_kpi_row(cards: list[dict[str, str]]) -> None:
    """Render a 4-card signature KPI row at the top of a workspace."""
    cards_html = "".join(
        render_kpi_card(
            title=c.get("title", ""),
            value=c.get("value", "--"),
            subtitle=c.get("subtitle", ""),
            variant=c.get("variant", "coral"),
            badge_text=c.get("badge_text", ""),
        )
        for c in cards
    )
    st.markdown(
        f'<div class="rl-kpi-grid">{cards_html}</div>',
        unsafe_allow_html=True,
    )
