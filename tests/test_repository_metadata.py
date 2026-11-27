from pathlib import Path

import pytest

from repolens.ingestion.file_discovery import (
    FileDiscoveryResult,
    InvalidRepositoryPathError,
)
from repolens.ingestion.repository_metadata import (
    RepositoryMetadataCollector,
    collect_repository_metadata,
)


def create_git_repository(path: Path) -> Path:
    """Create a temporary Git repository with one initial commit."""
    import subprocess

    path.mkdir(parents=True)

    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        ["git", "config", "user.name", "RepoLens Test"],
        cwd=path,
        check=True,
    )

    subprocess.run(
        ["git", "config", "user.email", "test@repolens.local"],
        cwd=path,
        check=True,
    )

    (path / "main.py").write_text(
        "print('RepoLens')\n",
        encoding="utf-8",
    )

    source_directory = path / "src"
    source_directory.mkdir()

    (source_directory / "helper.py").write_text(
        "def helper():\n    return True\n",
        encoding="utf-8",
    )

    (path / "README.md").write_text(
        "# Sample Repository\n",
        encoding="utf-8",
    )

    subprocess.run(
        ["git", "add", "."],
        cwd=path,
        check=True,
    )

    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )

    return path


def test_collect_git_repository_metadata(tmp_path):
    repository = create_git_repository(tmp_path / "sample-repo")

    inventory = RepositoryMetadataCollector().collect(repository)

    assert inventory.name == "sample-repo"
    assert inventory.is_git_repository is True
    assert inventory.commit_hash is not None
    assert len(inventory.commit_hash) >= 7
    assert inventory.branch == "main"
    assert inventory.source_type == "local"


def test_collect_python_file_inventory(tmp_path):
    repository = tmp_path / "sample-repo"
    repository.mkdir()

    (repository / "main.py").write_text(
        "print('main')\n",
        encoding="utf-8",
    )

    source_directory = repository / "src"
    source_directory.mkdir()

    (source_directory / "helper.py").write_text(
        "def helper():\n    pass\n",
        encoding="utf-8",
    )

    (repository / "README.md").write_text(
        "# Sample\n",
        encoding="utf-8",
    )

    inventory = RepositoryMetadataCollector().collect(repository)

    assert inventory.file_count == 2

    discovered_paths = {file.relative_path for file in inventory.files}

    assert discovered_paths == {
        "main.py",
        "src/helper.py",
    }


def test_collect_file_sizes(tmp_path):
    repository = tmp_path / "sample-repo"
    repository.mkdir()

    content_one = "print('one')\n"
    content_two = "print('two')\n"

    (repository / "one.py").write_bytes(content_one.encode("utf-8"))

    (repository / "two.py").write_bytes(content_two.encode("utf-8"))

    inventory = RepositoryMetadataCollector().collect(repository)

    expected_size = len(content_one.encode("utf-8")) + len(content_two.encode("utf-8"))

    assert inventory.total_size_bytes == expected_size

    for file in inventory.files:
        assert file.size_bytes is not None
        assert file.size_bytes > 0


def test_non_git_repository_is_supported(tmp_path):
    repository = tmp_path / "plain-folder"
    repository.mkdir()

    (repository / "main.py").write_text(
        "print('RepoLens')\n",
        encoding="utf-8",
    )

    inventory = RepositoryMetadataCollector().collect(repository)

    assert inventory.name == "plain-folder"
    assert inventory.is_git_repository is False
    assert inventory.commit_hash is None
    assert inventory.branch is None
    assert inventory.file_count == 1


@pytest.mark.parametrize(
    "source",
    [
        "https://github.com/owner/repository.git",
        "http://github.com/owner/repository.git",
        "ssh://git@github.com/owner/repository.git",
        "git://github.com/owner/repository.git",
        "git@github.com:owner/repository.git",
    ],
)
def test_remote_source_classification(tmp_path, source):
    repository = tmp_path / "sample-repo"
    repository.mkdir()

    (repository / "main.py").write_text(
        "print('RepoLens')\n",
        encoding="utf-8",
    )

    inventory = RepositoryMetadataCollector().collect(
        repository,
        source=source,
    )

    assert inventory.source == source
    assert inventory.source_type == "remote"


def test_local_source_classification(tmp_path):
    repository = tmp_path / "sample-repo"
    repository.mkdir()

    (repository / "main.py").write_text(
        "print('RepoLens')\n",
        encoding="utf-8",
    )

    inventory = RepositoryMetadataCollector().collect(repository)

    assert inventory.source_type == "local"
    assert inventory.source == str(repository.resolve())


def test_inventory_dictionary_serialization(tmp_path):
    repository = tmp_path / "sample-repo"
    repository.mkdir()

    (repository / "main.py").write_text(
        "print('RepoLens')\n",
        encoding="utf-8",
    )

    inventory = RepositoryMetadataCollector().collect(repository)

    result = inventory.to_dict()

    assert result["name"] == "sample-repo"
    assert result["source_type"] == "local"
    assert result["file_count"] == 1
    assert result["total_size_bytes"] > 0
    assert isinstance(result["files"], list)
    assert isinstance(result["inaccessible_directories"], list)

    assert result["files"][0]["relative_path"] == "main.py"
    assert result["files"][0]["suffix"] == ".py"


def test_empty_repository_inventory(tmp_path):
    repository = tmp_path / "empty-repo"
    repository.mkdir()

    inventory = RepositoryMetadataCollector().collect(repository)

    assert inventory.file_count == 0
    assert inventory.total_size_bytes == 0
    assert inventory.files == ()


def test_invalid_repository_path_is_rejected(tmp_path):
    collector = RepositoryMetadataCollector()

    with pytest.raises(InvalidRepositoryPathError):
        collector.collect(tmp_path / "does-not-exist")


def test_inaccessible_file_metadata_is_handled(
    tmp_path,
    monkeypatch,
):
    repository = tmp_path / "sample-repo"
    repository.mkdir()

    source_file = repository / "main.py"
    source_file.write_text(
        "print('RepoLens')",
        encoding="utf-8",
    )

    collector = RepositoryMetadataCollector()

    # Isolate the metadata collection stage from file discovery.
    monkeypatch.setattr(
        collector.file_discovery,
        "discover",
        lambda path: FileDiscoveryResult(
            files=[Path("main.py")],
            inaccessible_directories=[],
        ),
    )

    original_stat = Path.stat

    def mock_stat(path, *args, **kwargs):
        if path == source_file:
            raise PermissionError("Access denied")

        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", mock_stat)

    inventory = collector.collect(repository)

    assert inventory.file_count == 1
    assert inventory.files[0].relative_path == "main.py"
    assert inventory.files[0].size_bytes is None
    assert inventory.total_size_bytes == 0


def test_convenience_function_returns_inventory(tmp_path):
    repository = tmp_path / "sample-repo"
    repository.mkdir()

    (repository / "main.py").write_text(
        "print('RepoLens')\n",
        encoding="utf-8",
    )

    inventory = collect_repository_metadata(repository)

    assert inventory.name == "sample-repo"
    assert inventory.file_count == 1
    assert inventory.files[0].relative_path == "main.py"
