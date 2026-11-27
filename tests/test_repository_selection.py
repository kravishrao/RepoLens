from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from repolens.ingestion.code_unit import CodeUnit
from repolens.ingestion.repository_loader import InvalidRepositoryError
from repolens.ui.adapter import run_repolens_query
from repolens.ui.app import (
    _trigger_repository_loading,
    render_code_graph,
    render_investigations,
    render_repositories,
)
from repolens.ui.repository_service import (
    LoadedRepository,
    format_bytes,
    get_bundled_repositories,
    load_repository_pipeline,
    resolve_repository_source,
)


def test_format_bytes() -> None:
    """Verify byte formatting handles various magnitudes and edge cases."""
    assert format_bytes(None) == "Unknown"
    assert format_bytes(500) == "500 B"
    assert format_bytes(2048) == "2.0 KB"
    assert format_bytes(5 * 1024 * 1024) == "5.00 MB"


def test_get_bundled_repositories() -> None:
    """Verify bundled benchmark catalog contains expected repositories."""
    repos = get_bundled_repositories()
    assert len(repos) >= 4
    ids = {r["id"] for r in repos}
    assert {"requests", "click", "pydantic", "flask"}.issubset(ids)
    for r in repos:
        assert "abs_path" in r
        assert "description" in r


def test_resolve_repository_source_empty() -> None:
    """Verify empty or whitespace source raises InvalidRepositoryError."""
    with pytest.raises(InvalidRepositoryError, match="cannot be empty"):
        resolve_repository_source("")
    with pytest.raises(InvalidRepositoryError, match="cannot be empty"):
        resolve_repository_source("   ")


def test_resolve_repository_source_nonexistent() -> None:
    """Verify non-existent directory raises InvalidRepositoryError."""
    with pytest.raises(InvalidRepositoryError, match="does not exist"):
        resolve_repository_source("C:/non/existent/path/for/repolens/test")


def test_resolve_repository_source_file_not_dir(tmp_path: Path) -> None:
    """Verify pointing to a file instead of a directory raises InvalidRepositoryError."""
    test_file = tmp_path / "single_file.py"
    test_file.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(InvalidRepositoryError, match="not a directory"):
        resolve_repository_source(str(test_file))


def test_resolve_repository_source_bundled() -> None:
    """Verify bundled repo id resolves to target path."""
    target_path, source_type, remote_url = resolve_repository_source("requests")
    assert target_path.exists()
    assert target_path.name == "requests"
    assert source_type == "dataset"
    assert remote_url is None


def test_load_repository_pipeline_no_python_files(tmp_path: Path) -> None:
    """Verify loading directory without Python files raises InvalidRepositoryError."""
    empty_dir = tmp_path / "empty_repo"
    empty_dir.mkdir()
    (empty_dir / "README.txt").write_text("No python code here", encoding="utf-8")

    with pytest.raises(InvalidRepositoryError, match="No Python source files"):
        load_repository_pipeline(str(empty_dir))


def test_load_repository_pipeline_success(tmp_path: Path) -> None:
    """Verify loading repository with Python source files extracts units and builds graph."""
    repo_dir = tmp_path / "sample_project"
    repo_dir.mkdir()
    sample_py = repo_dir / "calculator.py"
    sample_py.write_text(
        "def add(a: int, b: int) -> int:\n"
        "    '''Add two integers.'''\n"
        "    return a + b\n\n"
        "class Engine:\n"
        "    def run(self):\n"
        "        return add(1, 2)\n",
        encoding="utf-8",
    )

    progress_calls: list[tuple[float, str]] = []

    def mock_progress(val: float, msg: str) -> None:
        progress_calls.append((val, msg))

    loaded_repo = load_repository_pipeline(
        str(repo_dir), progress_callback=mock_progress
    )

    assert loaded_repo.name == "sample_project"
    assert loaded_repo.file_count == 1
    assert loaded_repo.code_unit_count >= 2
    assert loaded_repo.graph_node_count >= 2
    assert loaded_repo.status == "ready"
    assert len(progress_calls) >= 4
    assert progress_calls[-1][0] == 1.0


def test_active_repository_integration_with_query() -> None:
    """Verify run_repolens_query uses active LoadedRepository data when present."""
    unit = CodeUnit(
        id="sample-unit-id",
        kind="function",
        name="compute_engine",
        qualified_name="sample.compute_engine",
        file_path="src/sample.py",
        start_line=10,
        end_line=25,
        source_code="def compute_engine():\n    return 42",
        docstring="Compute engine processor",
    )

    loaded_repo = LoadedRepository(
        name="test_active_repo",
        source="C:/fake/path",
        source_type="local",
        root_path="C:/fake/path",
        commit_hash="abc1234",
        branch="main",
        is_git_repository=True,
        file_count=5,
        files=[
            {
                "relative_path": "src/sample.py",
                "size_bytes": 100,
                "size_formatted": "100 B",
                "suffix": ".py",
            }
        ],
        code_unit_count=1,
        code_units=[unit],
        parse_errors=[],
        graph_node_count=1,
        graph_edge_count=2,
        graph_edges=[
            {
                "source": "sample.compute_engine",
                "relation": "calls",
                "target": "math.sqrt",
            }
        ],
        status="ready",
    )

    result = run_repolens_query(
        repo_name="test_active_repo",
        query="Where is compute_engine defined?",
        baseline_mode="B5: Full RepoLens",
        active_repo=loaded_repo,
    )

    assert result["is_demo"] is False
    assert "test_active_repo" in result["answer"]
    assert len(result["evidence_units"]) >= 1
    assert result["evidence_units"][0]["symbol"] == "sample.compute_engine"
    assert result["evidence_units"][0]["file_path"] == "src/sample.py"
    assert len(result["graph_edges"]) >= 1


def test_render_repositories_empty_and_active_state() -> None:
    """Verify render_repositories handles both unloaded state and active loaded state."""
    config = {
        "page": "Repositories",
        "selected_repo": "RepoLens",
        "baseline_mode": "B5: Full RepoLens",
    }

    mock_state: dict[str, object] = {
        "active_repository": None,
        "repo_load_error": None,
        "rl_navigation": "Repositories",
        "latest_result": None,
        "latest_query": "",
        "latest_repo": "",
    }

    def mock_columns(spec, **kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [MagicMock() for _ in range(count)]

    def mock_tabs(tabs, **kwargs):
        return [MagicMock() for _ in range(len(tabs))]

    # Test 1: Empty state (selection form)
    with (
        patch("streamlit.session_state", mock_state),
        patch("streamlit.markdown"),
        patch("streamlit.title"),
        patch("streamlit.subheader"),
        patch("streamlit.caption"),
        patch("streamlit.info"),
        patch("streamlit.success"),
        patch("streamlit.error"),
        patch("streamlit.text_input", return_value=""),
        patch("streamlit.selectbox", return_value="psf/requests"),
        patch("streamlit.button", return_value=False),
        patch("streamlit.columns", side_effect=mock_columns),
        patch("streamlit.tabs", side_effect=mock_tabs),
    ):
        render_repositories(config)

    # Test 2: Active loaded state
    fake_unit = CodeUnit(
        id="unit-1",
        kind="function",
        name="do_work",
        qualified_name="pkg.do_work",
        file_path="pkg/work.py",
        start_line=1,
        end_line=10,
        source_code="def do_work(): pass",
    )
    mock_state["active_repository"] = LoadedRepository(
        name="active_project",
        source="C:/active/project",
        source_type="local",
        root_path="C:/active/project",
        commit_hash="def5678",
        branch="develop",
        is_git_repository=True,
        file_count=3,
        files=[
            {
                "relative_path": "pkg/work.py",
                "size_bytes": 200,
                "size_formatted": "200 B",
                "suffix": ".py",
            }
        ],
        code_unit_count=1,
        code_units=[fake_unit],
        parse_errors=[],
        graph_node_count=1,
        graph_edge_count=0,
        graph_edges=[],
        status="ready",
    )

    with (
        patch("streamlit.session_state", mock_state),
        patch("streamlit.markdown"),
        patch("streamlit.title"),
        patch("streamlit.subheader"),
        patch("streamlit.metric"),
        patch("streamlit.dataframe"),
        patch("streamlit.button", return_value=False),
        patch("streamlit.text_input", return_value=""),
        patch("streamlit.columns", side_effect=mock_columns),
        patch("streamlit.tabs", side_effect=mock_tabs),
    ):
        render_repositories(config)
        render_investigations(config)
        render_code_graph(config)


def test_trigger_repository_loading_empty_input() -> None:
    """Verify _trigger_repository_loading rejects empty input gracefully."""
    with patch("streamlit.error") as mock_err:
        _trigger_repository_loading("   ")
        mock_err.assert_called_once()
        assert "provide a repository path" in mock_err.call_args[0][0]
