"""RepoLens Dark Engineering Design System & Modern UI Theme.

Designed for high-contrast obsidian dashboards, sleek geometric typography,
elevated surface containers, and vibrant gradient KPI cards.
"""

from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    """Apply the modern, polished RepoLens obsidian engineering theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        :root {
            /* Palette matching modern high-contrast obsidian design */
            --rl-bg: #0B0E17;
            --rl-sidebar: #101422;
            --rl-surface: #141A28;
            --rl-elevated: #1B2337;
            --rl-border: rgba(255, 255, 255, 0.08);
            --rl-border-hover: rgba(56, 189, 248, 0.4);

            --rl-text: #F8FAFC;
            --rl-secondary: #94A3B8;
            --rl-muted: #64748B;

            /* Vibrant Accent Palette */
            --rl-cyan: #38BDF8;
            --rl-blue: #60A5FA;
            --rl-indigo: #818CF8;
            --rl-emerald: #34D399;
            --rl-amber: #FBBF24;
            --rl-rose: #FB7185;
            --rl-coral: #FF6B4A;
            --rl-purple: #A855F7;

            /* Glows & Shadows */
            --rl-shadow-card: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
            --rl-shadow-kpi: 0 12px 28px -6px rgba(0, 0, 0, 0.4);
        }

        /* Base Application Layout */
        .stApp {
            background: var(--rl-bg);
            color: var(--rl-text);
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            letter-spacing: -0.01em;
        }

        [data-testid="stHeader"] {
            background: rgba(11, 14, 23, 0.85);
            backdrop-filter: blur(14px);
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
            background: rgba(20, 26, 40, 0.6);
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

        /* 4 Signature Gradient KPI Cards */
        .rl-kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.1rem;
            margin: 1.2rem 0 1.6rem 0;
        }

        @media (max-width: 992px) {
            .rl-kpi-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (max-width: 600px) {
            .rl-kpi-grid {
                grid-template-columns: 1fr;
            }
        }

        .rl-kpi-card {
            border-radius: 14px;
            padding: 1.25rem 1.4rem;
            position: relative;
            box-shadow: var(--rl-shadow-kpi);
            color: #FFFFFF;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 120px;
        }

        .rl-kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 16px 32px -6px rgba(0, 0, 0, 0.5);
        }

        .rl-kpi-coral {
            background: linear-gradient(135deg, #FF5A36 0%, #FF8533 100%);
        }

        .rl-kpi-cyan {
            background: linear-gradient(135deg, #0284C7 0%, #38BDF8 100%);
        }

        .rl-kpi-emerald {
            background: linear-gradient(135deg, #059669 0%, #34D399 100%);
        }

        .rl-kpi-purple {
            background: linear-gradient(135deg, #7C3AED 0%, #A855F7 100%);
        }

        .rl-kpi-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .rl-kpi-label {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            opacity: 0.92;
        }

        .rl-kpi-badge-circle {
            width: 26px;
            height: 26px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.22);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.72rem;
            font-weight: 700;
        }

        .rl-kpi-val {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            line-height: 1.1;
            margin: 0.55rem 0 0.25rem 0;
            color: #FFFFFF;
        }

        .rl-kpi-sub {
            font-size: 0.78rem;
            font-weight: 500;
            opacity: 0.88;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* Status Pills & Micro-Badges */
        .rl-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.22rem 0.65rem;
            border-radius: 20px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            font-family: 'JetBrains Mono', monospace;
            text-transform: uppercase;
        }

        .rl-pill-active {
            background: rgba(16, 185, 129, 0.16);
            color: #34D399;
            border: 1px solid rgba(52, 211, 153, 0.35);
        }

        .rl-pill-danger {
            background: rgba(239, 68, 68, 0.16);
            color: #F87171;
            border: 1px solid rgba(248, 113, 113, 0.35);
        }

        .rl-pill-warning {
            background: rgba(245, 158, 11, 0.16);
            color: #FBBF24;
            border: 1px solid rgba(251, 191, 36, 0.35);
        }

        .rl-pill-info {
            background: rgba(56, 189, 248, 0.16);
            color: #38BDF8;
            border: 1px solid rgba(56, 189, 248, 0.35);
        }

        .rl-pill-neutral {
            background: rgba(148, 163, 184, 0.12);
            color: #94A3B8;
            border: 1px solid rgba(148, 163, 184, 0.22);
        }

        /* Clean Target Bar & Status Indicators */
        .rl-target-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.7rem 1.1rem;
            background: rgba(18, 23, 34, 0.8);
            border: 1px solid var(--rl-border);
            border-radius: 10px;
            margin-bottom: 1.2rem;
            font-size: 0.85rem;
        }

        .rl-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.3rem 0.75rem;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--rl-border);
            border-radius: 6px;
            font-size: 0.8rem;
            color: var(--rl-secondary);
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .rl-chip:hover {
            background: rgba(56, 189, 248, 0.12);
            border-color: var(--rl-cyan);
            color: var(--rl-text);
        }

        /* Banner Components */
        .rl-active-banner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.85rem 1.25rem;
            background: linear-gradient(90deg, rgba(16, 37, 31, 0.85) 0%, rgba(13, 27, 24, 0.85) 100%);
            border: 1px solid rgba(52, 211, 153, 0.35);
            border-left: 4px solid var(--rl-emerald);
            border-radius: 12px;
            margin-bottom: 1.4rem;
        }

        .rl-demo-banner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.85rem 1.25rem;
            background: linear-gradient(90deg, rgba(37, 31, 21, 0.85) 0%, rgba(27, 23, 16, 0.85) 100%);
            border: 1px solid rgba(251, 191, 36, 0.35);
            border-left: 4px solid var(--rl-amber);
            border-radius: 12px;
            margin-bottom: 1.4rem;
        }

        /* Metric Cards */
        div[data-testid="stMetric"] {
            background: rgba(20, 26, 40, 0.8);
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
            background: #0E121E !important;
            border: 1px solid #1E2638 !important;
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
            background: rgba(20, 26, 40, 0.6) !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            overflow: hidden;
            margin-bottom: 0.6rem;
        }

        [data-testid="stExpander"] details summary {
            font-weight: 600;
            color: var(--rl-text);
            padding: 0.8rem 1rem;
        }

        /* Data Tables */
        [data-testid="stDataFrame"] {
            border: 1px solid var(--rl-border);
            border-radius: 12px;
            overflow: hidden;
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

        .rl-card-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--rl-text);
            margin: 0.4rem 0 0.3rem 0;
        }

        .rl-card-description {
            font-size: 0.86rem;
            color: var(--rl-secondary);
            line-height: 1.5;
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
