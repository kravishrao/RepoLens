"""Tests for RepoLens technical documentation and demonstration guides (RL-44)."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
README_PATH = REPO_ROOT / "README.md"
ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"


def test_readme_exists_and_substantive() -> None:
    """Verify that README.md exists and contains all required sections."""
    assert README_PATH.is_file(), "Root README.md must exist"
    content = README_PATH.read_text(encoding="utf-8")
    assert len(content) > 5000, "README.md should be comprehensive and substantive"

    required_sections = [
        "Executive Summary",
        "System Architecture",
        "Key Features",
        "Prerequisites & Requirements",
        "Installation Guide",
        "Environment Configuration",
        "Core Workflow & Subsystems",
        "CodeUnit Schema",
        "Code Graph Representation",
        "Hybrid Retrieval",
        "Agentic Investigation Pipeline",
        "Citation Validation",
        "Python API Reference",
        "Streamlit Web Application",
        "Example Repository Questions",
        "Experimental Benchmarks",
        "Testing & CI/CD",
        "Faculty Demonstration Guide",
        "Repository Structure",
        "Team & Contribution Workflow",
    ]

    for section in required_sections:
        assert section.lower() in content.lower(), (
            f"README.md missing section: {section}"
        )


def test_technical_architecture_exists_and_substantive() -> None:
    """Verify that docs/technical_architecture.md exists and details all subsystems."""
    arch_path = DOCS_DIR / "technical_architecture.md"
    assert arch_path.is_file(), "docs/technical_architecture.md must exist"
    content = arch_path.read_text(encoding="utf-8")
    assert len(content) > 5000, "technical_architecture.md should be comprehensive"

    required_subsystems = [
        "Ingestion & File Discovery Subsystem",
        "CodeUnit Schema & Representation",
        "Code Graph Representation & Traversal",
        "Hybrid Retrieval & Reranking Subsystem",
        "Agentic Investigation Pipeline",
        "Grounded Answer Generation & Citation Validation",
        "Streamlit Presentation Architecture",
        "Evaluation Framework Specification",
    ]

    for subsystem in required_subsystems:
        assert subsystem.lower() in content.lower(), (
            f"technical_architecture.md missing subsystem: {subsystem}"
        )


def test_faculty_demonstration_guide_exists_and_substantive() -> None:
    """Verify that docs/faculty_demonstration_guide.md covers all demo scenarios and defense points."""
    guide_path = DOCS_DIR / "faculty_demonstration_guide.md"
    assert guide_path.is_file(), "docs/faculty_demonstration_guide.md must exist"
    content = guide_path.read_text(encoding="utf-8")
    assert len(content) > 5000, "faculty_demonstration_guide.md should be comprehensive"

    required_topics = [
        "Executive Presentation Overview",
        "Pre-Flight Demonstration Setup",
        "High-Level Architecture Walkthrough",
        "Live Interactive Demonstration Protocol",
        "Scenario 1",
        "Scenario 2",
        "Scenario 3",
        "Scenario 4",
        "Scenario 5",
        "Scenario 6",
        "Experimental Results & Ablation Summary",
        "Faculty Defense & Technical Q&A Cheat Sheet",
        "Demonstration Wrap-Up Checklist",
    ]

    for topic in required_topics:
        assert topic.lower() in content.lower(), (
            f"faculty_demonstration_guide.md missing topic: {topic}"
        )


def test_env_example_configuration() -> None:
    """Verify that .env.example exists, defines standard variables, and contains no real secrets."""
    assert ENV_EXAMPLE_PATH.is_file(), ".env.example must exist"
    content = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")

    expected_vars = [
        "REPOLENS_CACHE_DIR",
        "REPOLENS_LOG_LEVEL",
        "REPOLENS_DEFAULT_REPO",
        "REPOLENS_EMBEDDING_MODEL",
        "REPOLENS_RETRIEVAL_TOP_K",
        "REPOLENS_MAX_ITERATIONS",
        "STREAMLIT_SERVER_PORT",
    ]

    for var in expected_vars:
        assert var in content, f".env.example missing variable: {var}"

    # Ensure no actual real secret keys are stored
    dangerous_patterns = [
        r"sk-[a-zA-Z0-9]{32,}",
        r"ghp_[a-zA-Z0-9]{36,}",
        r"AIza[0-9A-Za-z-_]{35}",
    ]
    for pattern in dangerous_patterns:
        assert not re.search(pattern, content), (
            f"Found potential real secret matching pattern {pattern}"
        )


def test_benchmark_plots_exist_in_assets() -> None:
    """Verify that all generated benchmark plots and charts exist in assets/."""
    assets_dir = REPO_ROOT / "assets"
    expected_files = [
        "benchmark_baseline_comparison.png",
        "benchmark_baseline_comparison.svg",
        "citation_precision_hallucination.png",
        "citation_precision_hallucination.svg",
        "latency_vs_accuracy_pareto.png",
        "latency_vs_accuracy_pareto.svg",
        "component_ablation_contributions.png",
        "component_ablation_contributions.svg",
        "per_repository_performance.png",
        "per_repository_performance.svg",
    ]

    for fname in expected_files:
        fpath = assets_dir / fname
        assert fpath.is_file(), f"Missing plot file in assets/: {fname}"
        assert fpath.stat().st_size > 1000, f"Plot file {fname} is unexpectedly small"
