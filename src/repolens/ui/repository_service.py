from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from repolens.graph.code_graph import CodeGraph
from repolens.ingestion.ast_parser import parse_repository
from repolens.ingestion.code_unit import CodeUnit
from repolens.ingestion.repository_loader import (
    InvalidRepositoryError,
    load_repository,
)
from repolens.ingestion.repository_metadata import collect_repository_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = PROJECT_ROOT / "dataset" / "repositories"


@dataclass
class LoadedRepository:
    """Represents an ingested repository with its AST CodeUnits and CodeGraph."""

    name: str
    source: str
    source_type: str  # 'local', 'remote', 'dataset'
    root_path: str
    commit_hash: str | None
    branch: str | None
    is_git_repository: bool
    file_count: int
    files: list[dict[str, Any]]
    code_unit_count: int
    code_units: list[CodeUnit]
    parse_errors: list[dict[str, Any]]
    graph_node_count: int
    graph_edge_count: int
    graph_edges: list[dict[str, Any]]
    loaded_at: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    )
    status: str = "ready"

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a lightweight dictionary summary of repository metadata."""
        return {
            "name": self.name,
            "source": self.source,
            "source_type": self.source_type,
            "root_path": self.root_path,
            "commit_hash": self.commit_hash,
            "branch": self.branch,
            "is_git_repository": self.is_git_repository,
            "file_count": self.file_count,
            "code_unit_count": self.code_unit_count,
            "graph_node_count": self.graph_node_count,
            "graph_edge_count": self.graph_edge_count,
            "error_count": len(self.parse_errors),
            "loaded_at": self.loaded_at,
            "status": self.status,
        }


def format_bytes(size: int | None) -> str:
    """Format file size in human-readable bytes string."""
    if size is None:
        return "Unknown"
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.2f} MB"


def get_bundled_repositories() -> list[dict[str, Any]]:
    """Return catalog of benchmark repositories bundled in the local dataset directory."""
    bundled = [
        {
            "id": "requests",
            "name": "psf/requests",
            "description": "Python HTTP for humans - benchmark library",
            "rel_path": "dataset/repositories/requests",
        },
        {
            "id": "click",
            "name": "pallets/click",
            "description": "Composable command line interface toolkit",
            "rel_path": "dataset/repositories/click",
        },
        {
            "id": "flask",
            "name": "pallets/flask",
            "description": "Lightweight WSGI web application framework",
            "rel_path": "dataset/repositories/flask",
        },
        {
            "id": "fastapi",
            "name": "tiangolo/fastapi",
            "description": "High performance async web framework",
            "rel_path": "dataset/repositories/fastapi",
        },
        {
            "id": "pydantic",
            "name": "pydantic/pydantic",
            "description": "Data validation using Python type hints",
            "rel_path": "dataset/repositories/pydantic",
        },
    ]

    for item in bundled:
        abs_path = PROJECT_ROOT / item["rel_path"]
        item["exists"] = abs_path.is_dir()
        item["abs_path"] = str(abs_path)

    return bundled


def resolve_repository_source(source: str) -> tuple[Path, str, str | None]:
    """Resolve a user-provided source into an absolute Path and determine its type.

    Args:
        source: User input string (path, URL, or bundled identifier).

    Returns:
        tuple[Path, str, str | None]:
            - resolved Path to repository root on disk
            - source type ('dataset', 'local', 'remote')
            - remote URL if source is a remote repository, else None
    """
    clean = source.strip()
    if not clean:
        raise InvalidRepositoryError("Repository path or source cannot be empty.")

    norm = (
        clean.replace(" (illustrative snapshot)", "").replace(" (snapshot)", "").strip()
    )

    # Check if target is the RepoLens project codebase itself
    if norm.lower() in {"repolens", "the original repolens"}:
        return PROJECT_ROOT, "local", None

    # Check if user selected a bundled repository identifier or snapshot label
    for bundled in get_bundled_repositories():
        identifiers = {
            bundled["id"].lower(),
            bundled["name"].lower(),
            bundled["rel_path"].lower(),
        }
        if (
            clean in {bundled["id"], bundled["name"], bundled["rel_path"]}
            or norm.lower() in identifiers
        ):
            target_path = Path(bundled["abs_path"])
            if not target_path.is_dir():
                raise InvalidRepositoryError(
                    f"Bundled dataset repository '{bundled['name']}' not found at {target_path}."
                )
            return target_path, "dataset", None

    # Check if source is a remote Git URL
    is_remote = clean.startswith(("https://", "http://", "ssh://", "git://", "git@"))
    if is_remote:
        repo_name = clean.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        existing_clone = PROJECT_ROOT / "data" / "repositories" / repo_name
        if existing_clone.is_dir() and (existing_clone / ".git").exists():
            return existing_clone, "remote", clean

        # load_repository handles cloning into data/repositories
        meta = load_repository(clean, workspace=PROJECT_ROOT / "data" / "repositories")
        return Path(meta.local_path), "remote", clean

    # Otherwise treat as local filesystem path
    candidate_path = Path(clean).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = (PROJECT_ROOT / candidate_path).resolve()
    else:
        candidate_path = candidate_path.resolve()

    if not candidate_path.exists():
        raise InvalidRepositoryError(f"Repository directory does not exist: '{clean}'")

    if not candidate_path.is_dir():
        raise InvalidRepositoryError(f"Provided path is not a directory: '{clean}'")

    return candidate_path, "local", None


def load_repository_pipeline(
    source: str,
    progress_callback: Callable[[float, str], None] | None = None,
) -> LoadedRepository:
    """Execute end-to-end repository loading, inventory collection, AST extraction, and graph building.

    Args:
        source: Local path, remote URL, or bundled repository identifier.
        progress_callback: Optional callable accepting (progress_float, status_message).

    Returns:
        LoadedRepository: Fully populated repository model ready for investigation.

    Raises:
        InvalidRepositoryError: If the source path is empty, invalid, or lacks Python files.
        RepositoryCloneError: If remote cloning fails.
        RepositoryError: If an unrecoverable ingestion failure occurs.
    """

    def update_progress(val: float, msg: str) -> None:
        if progress_callback:
            progress_callback(val, msg)

    update_progress(0.1, "Validating repository target and accessibility...")
    target_path, source_type, remote_url = resolve_repository_source(source)

    update_progress(0.3, f"Scanning Python source files in {target_path.name}...")
    inventory = collect_repository_metadata(
        target_path, source=remote_url or str(target_path)
    )

    if inventory.file_count == 0:
        raise InvalidRepositoryError(
            f"No Python source files (.py, .pyi) found in '{target_path.name}'. "
            "RepoLens requires a repository containing Python code."
        )

    update_progress(
        0.55,
        f"Parsing AST structures across {inventory.file_count} Python files...",
    )
    ast_result = parse_repository(target_path)

    update_progress(
        0.8,
        f"Constructing code relationship graph ({ast_result.unit_count} CodeUnits)...",
    )
    code_graph = CodeGraph()
    try:
        code_graph.build(ast_result.code_units)
        graph_nodes = code_graph.graph.number_of_nodes()
        graph_edges_count = code_graph.graph.number_of_edges()

        # Extract structured edges for UI display
        edges_list: list[dict[str, Any]] = []
        for u, v, data in code_graph.graph.edges(data=True):
            source_node = code_graph.graph.nodes.get(u, {})
            target_node = code_graph.graph.nodes.get(v, {})
            edges_list.append(
                {
                    "source": source_node.get("qualified_name")
                    or source_node.get("name")
                    or str(u),
                    "relation": data.get("type", "relates_to").lower(),
                    "target": target_node.get("qualified_name")
                    or target_node.get("name")
                    or str(v),
                }
            )
    except Exception as exc:
        # Fallback gracefully if graph build encounters unresolved references
        graph_nodes = ast_result.unit_count
        graph_edges_count = 0
        edges_list = []
        update_progress(0.9, f"Notice during graph construction: {exc}")

    files_data = [
        {
            "relative_path": f.relative_path,
            "size_bytes": f.size_bytes,
            "size_formatted": format_bytes(f.size_bytes),
            "suffix": f.suffix,
        }
        for f in inventory.files
    ]

    parse_errors_data = [
        {
            "file_path": err.file_path,
            "error": err.error,
            "line_number": err.line_number,
            "error_type": err.error_type,
        }
        for err in ast_result.errors
    ]

    update_progress(1.0, f"Successfully loaded {inventory.name}!")

    return LoadedRepository(
        name=inventory.name,
        source=source,
        source_type=source_type,
        root_path=str(target_path),
        commit_hash=inventory.commit_hash,
        branch=inventory.branch,
        is_git_repository=inventory.is_git_repository,
        file_count=inventory.file_count,
        files=files_data,
        code_unit_count=ast_result.unit_count,
        code_units=ast_result.code_units,
        parse_errors=parse_errors_data,
        graph_node_count=graph_nodes,
        graph_edge_count=graph_edges_count,
        graph_edges=edges_list,
        status="ready",
    )


__all__ = [
    "LoadedRepository",
    "format_bytes",
    "get_bundled_repositories",
    "load_repository_pipeline",
    "resolve_repository_source",
]
