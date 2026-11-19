"""Tests for RepoLens research paper and experimental documentation (RL-43)."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DOCS_DIR = REPO_ROOT / "docs"
BENCHMARK_REPORT_PATH = REPO_ROOT / "dataset" / "evaluation" / "benchmark_report.json"


def test_research_paper_markdown_exists():
    """Verify that docs/research_paper.md exists and is substantive."""
    paper_path = DOCS_DIR / "research_paper.md"
    assert paper_path.is_file(), "docs/research_paper.md must exist"
    content = paper_path.read_text(encoding="utf-8")
    assert len(content) > 10000, "Research paper markdown should be substantive"


def test_research_paper_latex_exists():
    """Verify that docs/research_paper.tex exists and contains valid LaTeX structure."""
    tex_path = DOCS_DIR / "research_paper.tex"
    assert tex_path.is_file(), "docs/research_paper.tex must exist"
    content = tex_path.read_text(encoding="utf-8")
    assert "\\documentclass" in content
    assert "\\begin{document}" in content
    assert "\\end{document}" in content
    assert "\\title{" in content
    assert "\\begin{table}" in content


def test_research_paper_sections():
    """Verify that the research paper covers all required sections and acceptance criteria."""
    paper_path = DOCS_DIR / "research_paper.md"
    content = paper_path.read_text(encoding="utf-8")

    required_topics = [
        "Abstract",
        "Introduction",
        "Related Work",
        "Limitations of Conventional Fixed-Chunk RAG",
        "The RepoLens Architecture",
        "Experimental Evaluation Setup",
        "Experimental Results and Findings",
        "Component Ablation Analysis",
        "Limitations and Threats to Validity",
        "Conclusion",
        "References",
    ]
    for topic in required_topics:
        assert topic.lower() in content.lower(), f"Missing required topic: {topic}"

    # Verify research questions RQ1-RQ5
    for rq in ["RQ1", "RQ2", "RQ3", "RQ4", "RQ5"]:
        assert rq in content, f"Missing research question {rq}"

    # Verify baselines B0-B5
    for baseline in ["B0", "B1", "B2", "B3", "B4", "B5"]:
        assert baseline in content, f"Missing baseline {baseline}"

    # Verify architectural components
    assert "codeunit" in content.lower()
    assert "ast" in content.lower()
    assert "bm25" in content.lower()
    assert "graph" in content.lower()
    assert "langgraph" in content.lower()
    assert "citation" in content.lower()


def test_research_paper_data_consistency():
    """Verify that the empirical metrics in the paper accurately reflect benchmark_report.json."""
    paper_path = DOCS_DIR / "research_paper.md"
    paper_content = paper_path.read_text(encoding="utf-8")

    assert BENCHMARK_REPORT_PATH.is_file(), "benchmark_report.json must exist"
    with open(BENCHMARK_REPORT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    baselines = data.get("baselines", {})

    # Check key figures are explicitly cited in the paper
    for b_id, metrics in baselines.items():
        assert b_id in paper_content, f"Baseline {b_id} missing from paper"
        rec = f"{metrics['recall_at_5']:.4f}"
        assert rec in paper_content, (
            f"Recall@5 {rec} for {b_id} not found in paper text"
        )
        mrr = f"{metrics['mrr']:.4f}"
        assert mrr in paper_content, f"MRR {mrr} for {b_id} not found in paper text"
