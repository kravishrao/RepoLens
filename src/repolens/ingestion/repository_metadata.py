from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from repolens.ingestion.file_discovery import (
    PythonFileDiscovery,
)
from repolens.ingestion.repository_loader import (
    InvalidRepositoryError,
    RepositoryLoader,
)


def _is_remote_source(source: str | None) -> bool:
    """Determine whether the supplied source is a remote Git URL."""
    if source is None:
        return False

    return source.startswith(
        (
            "https://",
            "http://",
            "ssh://",
            "git://",
            "git@",
        )
    )


@dataclass(frozen=True)
class RepositoryFileMetadata:
    """Metadata describing a discovered Python source file."""

    relative_path: str
    size_bytes: int | None
    suffix: str


@dataclass(frozen=True)
class RepositoryInventory:
    """Structured metadata and source-file inventory for a repository."""

    name: str
    root_path: str
    source: str
    source_type: Literal["local", "remote"]

    commit_hash: str | None
    branch: str | None
    is_git_repository: bool

    files: tuple[RepositoryFileMetadata, ...]
    inaccessible_directories: tuple[str, ...]

    @property
    def file_count(self) -> int:
        """Return the number of discovered Python files."""
        return len(self.files)

    @property
    def total_size_bytes(self) -> int:
        """Return the total size of accessible files."""
        return sum(file.size_bytes or 0 for file in self.files)

    def to_dict(self) -> dict:
        """Convert the inventory into a JSON-friendly dictionary."""
        return {
            "name": self.name,
            "root_path": self.root_path,
            "source": self.source,
            "source_type": self.source_type,
            "commit_hash": self.commit_hash,
            "branch": self.branch,
            "is_git_repository": self.is_git_repository,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "files": [
                {
                    "relative_path": file.relative_path,
                    "size_bytes": file.size_bytes,
                    "suffix": file.suffix,
                }
                for file in self.files
            ],
            "inaccessible_directories": list(self.inaccessible_directories),
        }


class RepositoryMetadataCollector:
    """Collect repository metadata and build a Python file inventory."""

    def __init__(
        self,
        file_discovery: PythonFileDiscovery | None = None,
        repository_loader: RepositoryLoader | None = None,
    ) -> None:
        self.file_discovery = file_discovery or PythonFileDiscovery()
        self.repository_loader = repository_loader or RepositoryLoader()

    def collect(
        self,
        repository_path: str | Path,
        source: str | None = None,
    ) -> RepositoryInventory:
        """Collect repository information and discovered file metadata."""
        root = Path(repository_path).expanduser().resolve()

        # Discover files first. This also validates the repository path.
        discovery_result = self.file_discovery.discover(root)

        git_metadata = None

        # Git metadata is optional. Non-Git repositories remain supported.
        with suppress(InvalidRepositoryError):
            git_metadata = self.repository_loader.load_local(root)

        repository_source = source or str(root)

        source_type: Literal["local", "remote"] = (
            "remote" if _is_remote_source(source) else "local"
        )

        files = []

        for relative_path in discovery_result.files:
            absolute_path = root / relative_path

            try:
                size_bytes = absolute_path.stat().st_size
            except OSError:
                size_bytes = None

            files.append(
                RepositoryFileMetadata(
                    relative_path=relative_path.as_posix(),
                    size_bytes=size_bytes,
                    suffix=relative_path.suffix,
                )
            )

        inaccessible_directories = []

        for directory in discovery_result.inaccessible_directories:
            try:
                relative_directory = directory.relative_to(root)
                inaccessible_directories.append(relative_directory.as_posix())
            except ValueError:
                inaccessible_directories.append(str(directory))

        return RepositoryInventory(
            name=root.name,
            root_path=str(root),
            source=repository_source,
            source_type=source_type,
            commit_hash=(
                git_metadata.commit_hash if git_metadata is not None else None
            ),
            branch=(git_metadata.branch if git_metadata is not None else None),
            is_git_repository=git_metadata is not None,
            files=tuple(files),
            inaccessible_directories=tuple(
                sorted(
                    inaccessible_directories,
                    key=str.casefold,
                )
            ),
        )


def collect_repository_metadata(
    repository_path: str | Path,
    source: str | None = None,
) -> RepositoryInventory:
    """Convenience function for collecting repository inventory."""
    collector = RepositoryMetadataCollector()
    return collector.collect(repository_path, source=source)
