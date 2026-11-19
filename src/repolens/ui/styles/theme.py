from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    """Apply the RepoLens dark engineering design system."""
    st.markdown(
        """
        <style>
        @import url(
            'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800'
            '&family=JetBrains+Mono:wght@400;500;600&display=swap'
        );

        :root {
            --rl-bg: #0B1120;
            --rl-sidebar: #0D1626;
            --rl-surface: #111D30;
            --rl-elevated: #17253A;
            --rl-border: #26364D;

            --rl-text: #F0F6FC;
            --rl-secondary: #9BAEC5;
            --rl-muted: #71839C;

            --rl-blue: #58A6FF;
            --rl-violet: #8B7CFF;
            --rl-green: #3FB68B;
            --rl-amber: #E5AA55;
            --rl-red: #F07178;
        }

        .stApp {
            background: var(--rl-bg);
            color: var(--rl-text);
            font-family: 'Inter', sans-serif;
        }

        [data-testid="stHeader"] {
            background: rgba(11, 17, 32, 0.96);
        }

        [data-testid="stSidebar"] {
            background: var(--rl-sidebar);
            border-right: 1px solid var(--rl-border);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.25rem;
        }

        .block-container {
            max-width: 1600px;
            padding-top: 1.8rem;
            padding-bottom: 2.5rem;
        }

        h1, h2, h3, h4 {
            color: var(--rl-text) !important;
            font-family: 'Inter', sans-serif !important;
            letter-spacing: -0.025em;
        }

        p, label, span, div {
            font-family: 'Inter', sans-serif;
        }

        code, pre, .rl-mono {
            font-family: 'JetBrains Mono', monospace !important;
        }

        .rl-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding-bottom: 1.25rem;
            border-bottom: 1px solid var(--rl-border);
            margin-bottom: 1.75rem;
        }

        .rl-brand {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            color: var(--rl-text);
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.04em;
        }

        .rl-eyebrow {
            color: var(--rl-blue);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .rl-page-title {
            color: var(--rl-text);
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.045em;
            line-height: 1.2;
            margin: 0.4rem 0;
        }

        .rl-description {
            max-width: 780px;
            color: var(--rl-secondary);
            font-size: 0.95rem;
            line-height: 1.7;
        }

        .rl-muted {
            color: var(--rl-secondary);
            font-size: 0.9rem;
        }

        .rl-hero {
            padding: 1.5rem 0 1rem 0;
        }

        .rl-hero-title {
            color: var(--rl-text);
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            line-height: 1.15;
            margin: 0.8rem 0;
        }

        .rl-gradient-text {
            background: linear-gradient(135deg, #58A6FF 0%, #8B7CFF 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .rl-card {
            height: 100%;
            padding: 1.25rem;
            border: 1px solid var(--rl-border);
            border-radius: 12px;
            background: var(--rl-surface);
            transition: border-color 160ms ease, background 160ms ease;
        }

        .rl-card:hover {
            border-color: #42648A;
            background: var(--rl-elevated);
        }

        .rl-card-icon {
            font-size: 1.5rem;
            margin-bottom: 0.6rem;
        }

        .rl-card-title {
            color: var(--rl-text);
            font-size: 1rem;
            font-weight: 600;
        }

        .rl-card-description {
            margin-top: 0.5rem;
            color: var(--rl-secondary);
            font-size: 0.85rem;
            line-height: 1.65;
        }

        .rl-section-title {
            margin-bottom: 0.75rem;
            color: var(--rl-text);
            font-size: 1.2rem;
            font-weight: 650;
            letter-spacing: -0.025em;
        }

        .rl-demo {
            background: rgba(229, 170, 85, 0.10);
            border: 1px solid rgba(229, 170, 85, 0.45);
            border-radius: 8px;
            color: #E5AA55;
            padding: 0.7rem 0.9rem;
            font-size: 0.85rem;
            margin-bottom: 1rem;
        }

        .rl-verified {
            color: #3FB68B;
            font-weight: 600;
        }

        .rl-unverified {
            color: #E5AA55;
            font-weight: 600;
        }

        .rl-evidence {
            padding: 1rem;
            border: 1px solid #285B4C;
            border-left: 3px solid var(--rl-green);
            border-radius: 8px;
            background: #10251F;
        }

        .rl-evidence-inferred {
            border-color: #66502D;
            border-left: 3px solid var(--rl-amber);
            background: #251F15;
        }

        .rl-evidence-invalid {
            border-color: #67343A;
            border-left: 3px solid var(--rl-red);
            background: #28191D;
        }

        .rl-status {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.25rem 0.55rem;
            border: 1px solid var(--rl-border);
            border-radius: 5px;
            font-size: 0.72rem;
            font-weight: 600;
        }

        .rl-valid {
            color: var(--rl-green);
            border-color: #285B4C;
            background: #10251F;
        }

        .rl-warning {
            color: var(--rl-amber);
            border-color: #66502D;
            background: #251F15;
        }

        .rl-invalid {
            color: var(--rl-red);
            border-color: #67343A;
            background: #28191D;
        }

        div[data-testid="stMetric"] {
            background: var(--rl-surface);
            border: 1px solid var(--rl-border);
            border-radius: 10px;
            padding: 12px 14px;
        }

        div[data-testid="stMetricValue"] {
            color: var(--rl-blue);
            font-size: 1.45rem;
        }

        div[data-testid="stTextArea"] textarea,
        div[data-testid="stTextInput"] input {
            background: var(--rl-sidebar);
            border-color: #344B68;
            color: var(--rl-text);
        }

        div[data-testid="stTabs"] button {
            font-weight: 600;
        }

        div[data-testid="stTabs"] [data-baseweb="tab"] {
            color: var(--rl-secondary);
        }

        div[data-testid="stTabs"] [aria-selected="true"] {
            color: var(--rl-blue) !important;
        }

        [data-testid="stExpander"] {
            border-color: var(--rl-border);
            border-radius: 9px;
            background: var(--rl-surface);
        }

        .rl-footer {
            padding-top: 2rem;
            color: var(--rl-muted);
            font-size: 0.8rem;
            text-align: center;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .rl-page-title {
                font-size: 1.6rem;
            }

            .rl-header {
                flex-wrap: wrap;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
