from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

EXCLUDED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "virtualenv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "node_modules",
    "build",
    "dist",
    "site-packages",
    ".idea",
    ".vscode",
    "htmlcov",
}


class FileDiscoveryError(Exception):
    """Base exception for source file discovery errors."""


class InvalidRepositoryPathError(FileDiscoveryError):
    """Raised when the repository path is invalid."""


@dataclass
class FileDiscoveryResult:
    """Store discovered Python files and filesystem warnings."""

    files: list[Path] = field(default_factory=list)
    inaccessible_directories: list[Path] = field(default_factory=list)

    @property
    def file_count(self) -> int:
        """Return the number of discovered Python files."""
        return len(self.files)


class PythonFileDiscovery:
    """Discover Python source files inside a repository."""

    def __init__(
        self,
        excluded_directories: set[str] | None = None,
    ) -> None:
        self.excluded_directories = (
            excluded_directories
            if excluded_directories is not None
            else EXCLUDED_DIRECTORIES
        )

    def discover(
        self,
        repository_path: str | Path,
    ) -> FileDiscoveryResult:
        """Recursively discover Python files in a repository."""
        root = Path(repository_path).expanduser().resolve()

        if not root.exists():
            raise InvalidRepositoryPathError(f"Repository path does not exist: {root}")

        if not root.is_dir():
            raise InvalidRepositoryPathError(
                f"Repository path is not a directory: {root}"
            )

        result = FileDiscoveryResult()

        def handle_walk_error(error: OSError) -> None:
            """Record inaccessible directories and continue scanning."""
            if error.filename:
                result.inaccessible_directories.append(Path(error.filename))

        for current_root, directories, filenames in os.walk(
            root,
            topdown=True,
            followlinks=False,
            onerror=handle_walk_error,
        ):
            current_path = Path(current_root)

            # Exclude irrelevant directories before descending into them.
            directories[:] = sorted(
                directory
                for directory in directories
                if directory not in self.excluded_directories
                and not (current_path / directory).is_symlink()
            )

            for filename in filenames:
                if not filename.endswith(".py"):
                    continue

                absolute_path = current_path / filename

                if absolute_path.is_symlink():
                    continue

                relative_path = absolute_path.relative_to(root)
                result.files.append(relative_path)

        result.files.sort(key=lambda path: path.as_posix().casefold())

        result.inaccessible_directories.sort(key=lambda path: str(path).casefold())

        return result


def discover_python_files(
    repository_path: str | Path,
) -> FileDiscoveryResult:
    """Convenience function for discovering repository Python files."""
    discovery = PythonFileDiscovery()
    return discovery.discover(repository_path)
