from __future__ import annotations

from unittest.mock import MagicMock, patch

import repolens.ui as ui
from repolens.ui.adapter import run_repolens_query
from repolens.ui.app import (
    init_session_state,
    navigate_to,
    render_code_graph,
    render_evaluation,
    render_footer,
    render_investigations,
    render_overview,
    render_repositories,
)
from repolens.ui.components.header import render_header
from repolens.ui.components.sidebar import render_sidebar
from repolens.ui.styles.theme import apply_theme


def test_ui_exports() -> None:
    """Verify repolens.ui package exports all required components and helpers."""
    expected_exports = {
        "apply_theme",
        "main",
        "render_feature_card",
        "render_header",
        "render_sidebar",
        "run_repolens_query",
    }
    assert expected_exports.issubset(set(ui.__all__))
    for symbol in expected_exports:
        assert hasattr(ui, symbol)


def test_run_repolens_query_schema() -> None:
    """Verify run_repolens_query returns expected dictionary structure and types."""
    result = run_repolens_query(
        repo_name="psf/requests",
        query="Where is the authentication handler defined?",
        baseline_mode="B5: Full RepoLens",
        top_k=5,
        max_hops=2,
        enable_reranker=True,
        enable_verification=True,
        max_agent_steps=8,
    )

    required_keys = {
        "is_demo",
        "answer",
        "metrics",
        "agent_trace",
        "evidence_units",
        "graph_edges",
        "configuration",
    }
    assert required_keys.issubset(result.keys())
    assert isinstance(result["is_demo"], bool)
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0

    metrics = result["metrics"]
    assert "citation_precision" in metrics
    assert "evidence_coverage" in metrics
    assert "latency_ms" in metrics
    assert "tokens_used" in metrics
    assert "cost_inr" in metrics
    assert metrics["latency_ms"] > 0
    assert metrics["tokens_used"] > 0

    trace = result["agent_trace"]
    assert isinstance(trace, list)
    assert len(trace) >= 4
    for step in trace:
        assert {"step", "tool", "input", "status", "hits"}.issubset(step.keys())

    evidence = result["evidence_units"]
    assert isinstance(evidence, list)
    assert len(evidence) >= 1
    for unit in evidence:
        assert {
            "unit_id",
            "repository",
            "commit",
            "file_path",
            "symbol",
            "type",
            "start_line",
            "end_line",
            "parent",
            "imports",
            "calls",
            "verified",
            "score",
            "code",
        }.issubset(unit.keys())
        assert unit["start_line"] <= unit["end_line"]

    edges = result["graph_edges"]
    assert isinstance(edges, list)
    for edge in edges:
        assert {"source", "relation", "target"}.issubset(edge.keys())

    config = result["configuration"]
    assert config["repository"] == "psf/requests"
    assert config["baseline_mode"] == "B5: Full RepoLens"
    assert config["top_k"] == 5
    assert config["max_hops"] == 2
    assert config["reranker_enabled"] is True
    assert config["verification_enabled"] is True


def test_run_repolens_query_ast_topic() -> None:
    """Verify specialized evidence is returned for AST-related queries."""
    result = run_repolens_query(
        repo_name="RepoLens",
        query="How are AST CodeUnits extracted and validated?",
    )
    symbols = [u["symbol"] for u in result["evidence_units"]]
    assert any("ASTParser" in s for s in symbols)
    assert any("CodeUnit" in s for s in symbols)


def test_run_repolens_query_retrieval_topic() -> None:
    """Verify specialized evidence is returned for retrieval index queries."""
    result = run_repolens_query(
        repo_name="RepoLens",
        query="Where is the retrieval index constructed?",
    )
    symbols = [u["symbol"] for u in result["evidence_units"]]
    assert any("HybridRetriever" in s for s in symbols)
    assert any("CrossEncoderReranker" in s for s in symbols)


def test_run_repolens_query_verification_toggle() -> None:
    """Verify citation verification flag affects result status and precision."""
    verified_result = run_repolens_query(
        repo_name="RepoLens",
        query="Where is the index defined?",
        enable_verification=True,
    )
    unverified_result = run_repolens_query(
        repo_name="RepoLens",
        query="Where is the index defined?",
        enable_verification=False,
    )

    assert verified_result["metrics"]["citation_precision"] == "100%"
    assert unverified_result["metrics"]["citation_precision"] != "100%"

    verification_trace_v = next(
        t
        for t in verified_result["agent_trace"]
        if t["tool"] == "citation_verification"
    )
    verification_trace_u = next(
        t
        for t in unverified_result["agent_trace"]
        if t["tool"] == "citation_verification"
    )
    assert verification_trace_v["status"] == "Verified"
    assert verification_trace_u["status"] == "Skipped"


def test_apply_theme() -> None:
    """Verify theme CSS is injected into Streamlit markdown."""
    with patch("streamlit.markdown") as mock_md:
        apply_theme()
        mock_md.assert_called_once()
        css = mock_md.call_args[0][0]
        assert "--rl-bg" in css
        assert "--rl-blue" in css
        assert ".stApp" in css


def test_render_feature_card() -> None:
    """Verify render_feature_card generates expected HTML structure."""
    with patch("streamlit.markdown") as mock_md:
        ui.render_feature_card("🚀", "Speed", "Fast indexing")
        mock_md.assert_called_once()
        call_content = mock_md.call_args[0][0]
        assert "rl-card" in call_content
        assert "Speed" in call_content
        assert "Fast indexing" in call_content


def test_render_header() -> None:
    """Verify render_header generates title and workspace details."""
    mock_brand = MagicMock()
    mock_repo = MagicMock()
    with patch("streamlit.columns", return_value=(mock_brand, mock_repo)):
        render_header("Investigations")
        assert mock_brand.__enter__.called
        assert mock_repo.__enter__.called


def test_render_sidebar() -> None:
    """Verify render_sidebar returns expected configuration mapping."""
    with (
        patch("streamlit.sidebar") as mock_sb,
        patch("streamlit.radio", return_value="Investigations"),
        patch("streamlit.selectbox", return_value="B5: Full RepoLens"),
        patch("streamlit.slider", return_value=5),
        patch("streamlit.checkbox", return_value=True),
        patch("streamlit.text_input", return_value="RepoLens"),
    ):
        mock_sb.__enter__.return_value = mock_sb
        config = render_sidebar()

        assert isinstance(config, dict)
        assert config["page"] == "Investigations"
        assert config["baseline_mode"] == "B5: Full RepoLens"
        assert config["top_k"] == 5
        assert config["enable_reranker"] is True


def test_init_session_state_and_navigate() -> None:
    """Verify session state defaults and page switching."""
    mock_state: dict[str, object] = {}
    with patch("streamlit.session_state", mock_state):
        init_session_state()
        assert "latest_result" in mock_state
        assert "latest_query" in mock_state
        assert "latest_repo" in mock_state
        assert mock_state["rl_navigation"] == "Investigations"

        navigate_to("Overview")
        assert mock_state["rl_navigation"] == "Overview"


def test_render_pages_smoke() -> None:
    """Verify that page render functions execute without error."""
    config = {
        "page": "Overview",
        "selected_repo": "RepoLens",
        "baseline_mode": "B5: Full RepoLens",
        "top_k": 5,
        "max_hops": 2,
        "enable_reranker": True,
        "enable_verification": True,
        "preset_query": "Custom question",
    }

    mock_state = {
        "latest_result": None,
        "latest_query": "",
        "latest_repo": "",
        "rl_navigation": "Overview",
    }

    def mock_columns(spec, **kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [MagicMock() for _ in range(count)]

    with (
        patch("streamlit.session_state", mock_state),
        patch("streamlit.markdown"),
        patch("streamlit.title"),
        patch("streamlit.caption"),
        patch("streamlit.info"),
        patch("streamlit.dataframe"),
        patch("streamlit.button", return_value=False),
        patch("streamlit.text_area", return_value=""),
        patch("streamlit.text_input", return_value=""),
        patch("streamlit.columns", side_effect=mock_columns),
        patch(
            "streamlit.tabs",
            return_value=[
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
            ],
        ),
    ):
        render_overview()
        render_repositories(config)
        render_code_graph(config)
        render_evaluation(config)
        render_footer()
        render_investigations(config)
