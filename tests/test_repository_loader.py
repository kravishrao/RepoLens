import subprocess
from pathlib import Path

import pytest

from repolens.ingestion.repository_loader import (
    InvalidRepositoryError,
    RepositoryCloneError,
    RepositoryLoader,
    load_repository,
)


def create_git_repository(path: Path) -> Path:
    """Create a temporary Git repository for testing."""
    path.mkdir(parents=True)

    subprocess.run(
        ["git", "init"],
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

    (path / "example.py").write_text(
        "def hello():\n    return 'Hello RepoLens'\n",
        encoding="utf-8",
    )

    subprocess.run(
        ["git", "add", "example.py"],
        cwd=path,
        check=True,
    )

    subprocess.run(
        ["git", "commit", "-m", "Initial test commit"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )

    return path


def test_load_valid_local_repository(tmp_path):
    repository = create_git_repository(tmp_path / "sample-repo")

    loader = RepositoryLoader()
    metadata = loader.load_local(repository)

    assert metadata.name == "sample-repo"
    assert metadata.local_path == str(repository.resolve())
    assert metadata.commit_hash
    assert len(metadata.commit_hash) >= 40
    assert metadata.is_git_repository is True

    # A local repository without a remote is still valid.
    assert metadata.remote_url is None


def test_load_repository_helper_with_local_path(tmp_path):
    repository = create_git_repository(tmp_path / "helper-repo")

    metadata = load_repository(str(repository))

    assert metadata.name == "helper-repo"
    assert metadata.is_git_repository is True


def test_invalid_local_repository_path(tmp_path):
    loader = RepositoryLoader()

    with pytest.raises(InvalidRepositoryError):
        loader.load_local(tmp_path / "does-not-exist")


def test_non_git_directory_is_rejected(tmp_path):
    ordinary_folder = tmp_path / "ordinary-folder"
    ordinary_folder.mkdir()

    loader = RepositoryLoader()

    with pytest.raises(InvalidRepositoryError):
        loader.load_local(ordinary_folder)


def test_clone_remote_repository(tmp_path):
    source = create_git_repository(tmp_path / "source-repo")

    workspace = tmp_path / "clones"
    loader = RepositoryLoader(workspace=workspace)

    metadata = loader.clone_remote(source.as_uri())

    assert metadata.name == "source-repo"
    assert Path(metadata.local_path).exists()
    assert metadata.commit_hash
    assert metadata.remote_url is not None


def test_invalid_remote_url():
    loader = RepositoryLoader()

    with pytest.raises(InvalidRepositoryError):
        loader.clone_remote("not-a-valid-url")


def test_existing_clone_destination_is_rejected(tmp_path):
    source = create_git_repository(tmp_path / "source-repo")

    workspace = tmp_path / "clones"
    loader = RepositoryLoader(workspace=workspace)

    repository_url = source.as_uri()

    loader.clone_remote(repository_url)

    with pytest.raises(RepositoryCloneError):
        loader.clone_remote(repository_url)


def test_load_repository_rejects_empty_source():
    with pytest.raises(InvalidRepositoryError):
        load_repository("")


@pytest.mark.parametrize(
    "repository_url",
    [
        "git@github.com:owner/repository.git",
        "ssh://git@github.com/owner/repository.git",
    ],
)
def test_valid_ssh_repository_urls(repository_url):
    RepositoryLoader._validate_remote_url(repository_url)


def test_ssh_url_with_password_is_rejected():
    loader = RepositoryLoader()

    with pytest.raises(InvalidRepositoryError):
        loader.clone_remote("ssh://git:password@github.com/owner/repository.git")


def test_https_url_with_credentials_is_rejected():
    loader = RepositoryLoader()

    with pytest.raises(InvalidRepositoryError):
        loader.clone_remote("https://user:password@github.com/owner/repository.git")
