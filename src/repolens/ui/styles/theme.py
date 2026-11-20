"""RepoLens Dark Engineering Design System & Modern UI Theme."""

from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    """Apply the modern, polished RepoLens obsidian engineering theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        :root {
            --rl-bg: #090D16;
            --rl-sidebar: #0D131F;
            --rl-surface: #111A2E;
            --rl-elevated: #18233C;
            --rl-border: rgba(255, 255, 255, 0.08);
            --rl-border-hover: rgba(56, 189, 248, 0.4);

            --rl-text: #F8FAFC;
            --rl-secondary: #94A3B8;
            --rl-muted: #64748B;

            --rl-cyan: #38BDF8;
            --rl-blue: #60A5FA;
            --rl-indigo: #818CF8;
            --rl-emerald: #34D399;
            --rl-amber: #FBBF24;
            --rl-rose: #FB7185;

            --rl-glow-blue: rgba(56, 189, 248, 0.15);
            --rl-glow-emerald: rgba(52, 211, 153, 0.15);
        }

        /* Base Application Layout */
        .stApp {
            background: var(--rl-bg);
            color: var(--rl-text);
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            letter-spacing: -0.01em;
        }

        [data-testid="stHeader"] {
            background: rgba(9, 13, 22, 0.85);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--rl-border);
        }

        [data-testid="stSidebar"] {
            background: var(--rl-sidebar);
            border-right: 1px solid var(--rl-border);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1rem;
            padding-left: 1.1rem;
            padding-right: 1.1rem;
        }

        .block-container {
            max-width: 1540px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: var(--rl-text) !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            letter-spacing: -0.03em;
            font-weight: 700;
        }

        p, label, span, div {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        code, pre, .rl-mono {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Header Components */
        .rl-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.8rem 1.4rem;
            background: rgba(17, 26, 46, 0.6);
            backdrop-filter: blur(10px);
            border: 1px solid var(--rl-border);
            border-radius: 14px;
            margin-bottom: 1.5rem;
        }

        .rl-brand-group {
            display: flex;
            align-items: center;
            gap: 0.9rem;
        }

        .rl-brand-name {
            font-size: 1.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 50%, #94A3B8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.04em;
            margin: 0;
            line-height: 1.1;
        }

        .rl-brand-tagline {
            color: var(--rl-secondary);
            font-size: 0.8rem;
            margin-top: 0.15rem;
        }

        .rl-eyebrow {
            color: var(--rl-cyan);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        .rl-page-title {
            color: var(--rl-text);
            font-size: 1.85rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            line-height: 1.2;
            margin: 0.3rem 0;
        }

        .rl-description {
            max-width: 820px;
            color: var(--rl-secondary);
            font-size: 0.95rem;
            line-height: 1.6;
        }

        .rl-muted {
            color: var(--rl-secondary);
            font-size: 0.88rem;
        }

        /* Hero Prompt Console */
        .rl-console-card {
            background: linear-gradient(180deg, rgba(17, 26, 46, 0.8) 0%, rgba(13, 19, 31, 0.8) 100%);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 16px;
            padding: 1.5rem 1.6rem;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
            margin-bottom: 1.8rem;
        }

        .rl-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.3rem 0.7rem;
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid var(--rl-border);
            border-radius: 20px;
            font-size: 0.78rem;
            color: var(--rl-secondary);
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .rl-chip:hover {
            border-color: var(--rl-cyan);
            color: var(--rl-text);
            background: rgba(56, 189, 248, 0.12);
        }

        /* Repository Status Badges */
        .rl-active-banner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.8rem 1.2rem;
            background: linear-gradient(90deg, rgba(16, 37, 31, 0.8) 0%, rgba(13, 27, 24, 0.8) 100%);
            border: 1px solid rgba(52, 211, 153, 0.3);
            border-left: 4px solid var(--rl-emerald);
            border-radius: 12px;
            margin-bottom: 1.4rem;
        }

        .rl-demo-banner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.8rem 1.2rem;
            background: linear-gradient(90deg, rgba(37, 31, 21, 0.8) 0%, rgba(27, 23, 16, 0.8) 100%);
            border: 1px solid rgba(251, 191, 36, 0.3);
            border-left: 4px solid var(--rl-amber);
            border-radius: 12px;
            margin-bottom: 1.4rem;
        }

        /* Metric Cards */
        div[data-testid="stMetric"] {
            background: rgba(17, 26, 46, 0.75);
            backdrop-filter: blur(8px);
            border: 1px solid var(--rl-border);
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
            transition: transform 0.15s ease, border-color 0.15s ease;
        }

        div[data-testid="stMetric"]:hover {
            border-color: rgba(56, 189, 248, 0.4);
            transform: translateY(-2px);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--rl-secondary);
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        div[data-testid="stMetricValue"] {
            color: var(--rl-text) !important;
            font-size: 1.65rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em;
        }

        /* Inputs & Text Area */
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stTextInput"] input {
            background: #0B101D !important;
            border: 1px solid #1E293B !important;
            border-radius: 10px !important;
            color: var(--rl-text) !important;
            font-size: 0.95rem !important;
            padding: 0.75rem 0.9rem !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }

        div[data-testid="stTextArea"] textarea:focus,
        div[data-testid="stTextInput"] input:focus {
            border-color: var(--rl-cyan) !important;
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15) !important;
        }

        /* Buttons */
        div[data-testid="stButton"] button {
            border-radius: 9px !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            letter-spacing: -0.01em !important;
            padding: 0.5rem 1.1rem !important;
            transition: all 0.18s ease !important;
            border: 1px solid var(--rl-border);
        }

        div[data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(135deg, #0284C7 0%, #4F46E5 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35);
        }

        div[data-testid="stButton"] button[kind="primary"]:hover {
            box-shadow: 0 6px 20px rgba(2, 132, 199, 0.55) !important;
            transform: translateY(-1px);
        }

        div[data-testid="stButton"] button[kind="secondary"] {
            background: rgba(30, 41, 59, 0.6) !important;
            color: var(--rl-text) !important;
        }

        div[data-testid="stButton"] button[kind="secondary"]:hover {
            background: rgba(51, 65, 85, 0.8) !important;
            border-color: rgba(255, 255, 255, 0.2) !important;
        }

        /* Tabs */
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 1.2rem;
            border-bottom: 1px solid var(--rl-border);
            padding-bottom: 0.2rem;
        }

        div[data-testid="stTabs"] [data-baseweb="tab"] {
            color: var(--rl-secondary);
            font-weight: 600;
            font-size: 0.92rem;
            padding: 0.6rem 0.2rem;
            background: transparent !important;
            border: none !important;
        }

        div[data-testid="stTabs"] [aria-selected="true"] {
            color: var(--rl-cyan) !important;
            border-bottom: 2px solid var(--rl-cyan) !important;
        }

        /* Expanders */
        [data-testid="stExpander"] {
            border: 1px solid var(--rl-border) !important;
            border-radius: 12px !important;
            background: rgba(17, 26, 46, 0.6) !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            overflow: hidden;
            margin-bottom: 0.6rem;
        }

        [data-testid="stExpander"] details summary {
            font-weight: 600;
            color: var(--rl-text);
            padding: 0.8rem 1rem;
        }

        /* Images */
        .stImage img {
            border-radius: 8px;
        }

        [data-testid="stSidebar"] img {
            border-radius: 6px;
            margin-bottom: 0.25rem;
        }

        /* Cards & Findings */
        .rl-card {
            padding: 1.3rem;
            border: 1px solid var(--rl-border);
            border-radius: 14px;
            background: var(--rl-surface);
            transition: border-color 0.16s ease, transform 0.16s ease;
        }

        .rl-card:hover {
            border-color: rgba(56, 189, 248, 0.4);
            transform: translateY(-2px);
        }

        .rl-evidence {
            padding: 1rem 1.2rem;
            border: 1px solid #1E3A34;
            border-left: 4px solid var(--rl-emerald);
            border-radius: 10px;
            background: #0E201B;
        }

        .rl-evidence-inferred {
            border-color: #4A3A1F;
            border-left: 4px solid var(--rl-amber);
            background: #1F190E;
        }

        .rl-evidence-invalid {
            border-color: #4D2127;
            border-left: 4px solid var(--rl-rose);
            background: #200E12;
        }

        .rl-status {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.2rem 0.55rem;
            border-radius: 6px;
            font-size: 0.74rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }

        .rl-valid {
            color: var(--rl-emerald);
            background: rgba(52, 211, 153, 0.12);
            border: 1px solid rgba(52, 211, 153, 0.3);
        }

        .rl-warning {
            color: var(--rl-amber);
            background: rgba(251, 191, 36, 0.12);
            border: 1px solid rgba(251, 191, 36, 0.3);
        }

        .rl-invalid {
            color: var(--rl-rose);
            background: rgba(251, 113, 133, 0.12);
            border: 1px solid rgba(251, 113, 133, 0.3);
        }

        .rl-verified {
            color: var(--rl-emerald);
            font-weight: 600;
        }

        .rl-unverified {
            color: var(--rl-amber);
            font-weight: 600;
        }

        /* Footer */
        .rl-footer {
            padding: 2.5rem 0 1rem 0;
            color: var(--rl-muted);
            font-size: 0.8rem;
            text-align: center;
            border-top: 1px solid var(--rl-border);
            margin-top: 3rem;
        }

        #MainMenu, footer {
            visibility: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
