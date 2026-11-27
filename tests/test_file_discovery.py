from pathlib import Path

import pytest

from repolens.ingestion.file_discovery import (
    EXCLUDED_DIRECTORIES,
    InvalidRepositoryPathError,
    PythonFileDiscovery,
    discover_python_files,
)


def create_repository_structure(root: Path) -> Path:
    """Create a sample repository with nested Python files."""
    (root / "src" / "module").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / ".venv" / "lib").mkdir(parents=True)
    (root / "__pycache__").mkdir()

    (root / "main.py").write_text(
        "print('Hello RepoLens')",
        encoding="utf-8",
    )

    (root / "src" / "app.py").write_text(
        "def app(): pass",
        encoding="utf-8",
    )

    (root / "src" / "module" / "helper.py").write_text(
        "def helper(): pass",
        encoding="utf-8",
    )

    (root / "tests" / "test_app.py").write_text(
        "def test_app(): pass",
        encoding="utf-8",
    )

    (root / ".venv" / "lib" / "ignored.py").write_text(
        "# Virtual environment file",
        encoding="utf-8",
    )

    (root / "__pycache__" / "cached.py").write_text(
        "# Cache file",
        encoding="utf-8",
    )

    (root / "README.md").write_text(
        "# Sample Repository",
        encoding="utf-8",
    )

    return root


def test_discover_python_files_recursively(tmp_path):
    repository = create_repository_structure(tmp_path / "sample-repo")

    result = PythonFileDiscovery().discover(repository)

    discovered_paths = {path.as_posix() for path in result.files}

    assert discovered_paths == {
        "main.py",
        "src/app.py",
        "src/module/helper.py",
        "tests/test_app.py",
    }

    assert result.file_count == 4


def test_discovered_paths_are_relative(tmp_path):
    repository = create_repository_structure(tmp_path / "sample-repo")

    result = discover_python_files(repository)

    for file_path in result.files:
        assert not file_path.is_absolute()
        assert ".." not in file_path.parts


def test_excluded_directories_are_not_scanned(tmp_path):
    repository = create_repository_structure(tmp_path / "sample-repo")

    result = PythonFileDiscovery().discover(repository)

    discovered_paths = [path.as_posix() for path in result.files]

    assert not any(path.startswith(".venv/") for path in discovered_paths)

    assert not any(path.startswith("__pycache__/") for path in discovered_paths)


@pytest.mark.parametrize(
    "directory_name",
    sorted(EXCLUDED_DIRECTORIES),
)
def test_all_excluded_directories_are_skipped(
    tmp_path,
    directory_name,
):
    repository = tmp_path / "sample-repo"
    excluded_directory = repository / directory_name
    excluded_directory.mkdir(parents=True)

    (excluded_directory / "ignored.py").write_text(
        "print('ignored')",
        encoding="utf-8",
    )

    result = PythonFileDiscovery().discover(repository)

    assert result.files == []


def test_empty_repository_returns_no_files(tmp_path):
    repository = tmp_path / "empty-repo"
    repository.mkdir()

    result = PythonFileDiscovery().discover(repository)

    assert result.files == []
    assert result.file_count == 0
    assert result.inaccessible_directories == []


def test_non_python_files_are_ignored(tmp_path):
    repository = tmp_path / "sample-repo"
    repository.mkdir()

    (repository / "README.md").write_text(
        "# Documentation",
        encoding="utf-8",
    )

    (repository / "config.json").write_text(
        "{}",
        encoding="utf-8",
    )

    (repository / "script.txt").write_text(
        "not Python",
        encoding="utf-8",
    )

    result = PythonFileDiscovery().discover(repository)

    assert result.files == []


def test_invalid_repository_path_is_rejected(tmp_path):
    discovery = PythonFileDiscovery()

    with pytest.raises(InvalidRepositoryPathError):
        discovery.discover(tmp_path / "does-not-exist")


def test_file_path_is_rejected_as_repository(tmp_path):
    file_path = tmp_path / "not-a-directory.py"
    file_path.write_text(
        "print('not a repository')",
        encoding="utf-8",
    )

    discovery = PythonFileDiscovery()

    with pytest.raises(InvalidRepositoryPathError):
        discovery.discover(file_path)


def test_inaccessible_directory_is_recorded(
    tmp_path,
    monkeypatch,
):
    repository = tmp_path / "sample-repo"
    repository.mkdir()

    inaccessible_path = repository / "restricted"

    def mock_walk(
        top,
        topdown=True,
        onerror=None,
        followlinks=False,
    ):
        if onerror:
            onerror(
                PermissionError(
                    13,
                    "Access denied",
                    str(inaccessible_path),
                )
            )

        yield str(top), [], []

    monkeypatch.setattr(
        "repolens.ingestion.file_discovery.os.walk",
        mock_walk,
    )

    result = PythonFileDiscovery().discover(repository)

    assert result.files == []
    assert inaccessible_path in result.inaccessible_directories


def test_discovered_files_are_sorted_deterministically(tmp_path):
    repository = tmp_path / "sample-repo"
    repository.mkdir()

    for filename in ["zebra.py", "alpha.py", "middle.py"]:
        (repository / filename).write_text(
            "pass",
            encoding="utf-8",
        )

    result = PythonFileDiscovery().discover(repository)

    assert result.files == [
        Path("alpha.py"),
        Path("middle.py"),
        Path("zebra.py"),
    ]
