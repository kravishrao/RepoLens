from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse


class RepositoryError(Exception):
    """Base exception for repository loading errors."""


class InvalidRepositoryError(RepositoryError):
    """Raised when a repository path or URL is invalid."""


class RepositoryCloneError(RepositoryError):
    """Raised when cloning a remote repository fails."""


@dataclass(frozen=True)
class RepositoryMetadata:
    name: str
    local_path: str
    remote_url: str | None
    commit_hash: str
    branch: str
    is_git_repository: bool

    def to_dict(self) -> dict:
        """Convert repository metadata into a dictionary."""
        return asdict(self)


class RepositoryLoader:
    """Load local Git repositories or clone remote repositories."""

    def __init__(
        self,
        workspace: str | Path = "data/repositories",
        timeout: int = 120,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.timeout = timeout

    @staticmethod
    def _run_git(
        arguments: list[str],
        cwd: Path | None = None,
        timeout: int = 30,
    ) -> str:
        """Execute a Git command safely and return its output."""
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout,
                shell=False,
            )
            return result.stdout.strip()

        except FileNotFoundError as exc:
            raise RepositoryError(
                "Git is not installed or is not available in PATH."
            ) from exc

        except subprocess.TimeoutExpired as exc:
            raise RepositoryError(
                f"Git command timed out after {timeout} seconds."
            ) from exc

        except subprocess.CalledProcessError as exc:
            message = exc.stderr.strip() or "Unknown Git error."
            raise RepositoryError(message) from exc

    @staticmethod
    def _validate_remote_url(url: str) -> None:
        """Validate a remote repository URL."""
        if not isinstance(url, str) or not url.strip():
            raise InvalidRepositoryError("Repository URL cannot be empty.")

        parsed = urlparse(url.strip())

        # Support local file URLs for testing and local Git sources.
        if parsed.scheme == "file":
            if not parsed.path:
                raise InvalidRepositoryError("Invalid local repository URL.")
            return

        if parsed.scheme not in {"https", "http", "ssh", "git"}:
            raise InvalidRepositoryError("Unsupported repository URL scheme.")

        if not parsed.netloc:
            raise InvalidRepositoryError("Invalid repository URL.")

        if parsed.username or parsed.password:
            raise InvalidRepositoryError(
                "Credentials must not be embedded in repository URLs."
            )

    def _read_metadata(
        self,
        repository_path: Path,
    ) -> RepositoryMetadata:
        """Read metadata from an existing Git repository."""
        repository_path = repository_path.resolve()

        if not repository_path.is_dir():
            raise InvalidRepositoryError(
                f"Repository directory does not exist: {repository_path}"
            )

        try:
            self._run_git(
                ["rev-parse", "--is-inside-work-tree"],
                cwd=repository_path,
            )

            commit_hash = self._run_git(
                ["rev-parse", "HEAD"],
                cwd=repository_path,
            )

            branch = self._run_git(
                ["branch", "--show-current"],
                cwd=repository_path,
            )

        except RepositoryError as exc:
            raise InvalidRepositoryError(
                f"Path is not a valid Git repository: {repository_path}"
            ) from exc

        # A valid local repository does not necessarily have a remote.
        try:
            remote_url = self._run_git(
                ["config", "--get", "remote.origin.url"],
                cwd=repository_path,
            )
        except RepositoryError:
            remote_url = None

        return RepositoryMetadata(
            name=repository_path.name,
            local_path=str(repository_path),
            remote_url=remote_url or None,
            commit_hash=commit_hash,
            branch=branch or "detached",
            is_git_repository=True,
        )

    def load_local(
        self,
        repository_path: str | Path,
    ) -> RepositoryMetadata:
        """Load an existing local Git repository."""
        path = Path(repository_path).expanduser()

        if not path.exists() or not path.is_dir():
            raise InvalidRepositoryError(
                f"Repository path does not exist or is not a directory: {path}"
            )

        return self._read_metadata(path)

    def clone_remote(
        self,
        repository_url: str,
    ) -> RepositoryMetadata:
        """Clone a remote Git repository into the workspace."""
        self._validate_remote_url(repository_url)

        self.workspace.mkdir(
            parents=True,
            exist_ok=True,
        )

        parsed = urlparse(repository_url.strip())

        repository_name = Path(parsed.path.rstrip("/")).name

        repository_name = repository_name.removesuffix(".git")

        if not re.fullmatch(
            r"[A-Za-z0-9._-]+",
            repository_name,
        ):
            raise InvalidRepositoryError(
                "Could not determine a valid repository name from the URL."
            )

        destination = self.workspace / repository_name

        if destination.exists():
            raise RepositoryCloneError(f"Destination already exists: {destination}")

        try:
            self._run_git(
                [
                    "clone",
                    "--",
                    repository_url.strip(),
                    str(destination),
                ],
                timeout=self.timeout,
            )

        except RepositoryError as exc:
            raise RepositoryCloneError(f"Failed to clone repository: {exc}") from exc

        return self._read_metadata(destination)


def load_repository(
    source: str,
    workspace: str | Path = "data/repositories",
) -> RepositoryMetadata:
    """Load a local Git repository or clone a remote Git URL."""
    if not isinstance(source, str) or not source.strip():
        raise InvalidRepositoryError("Repository source cannot be empty.")

    source = source.strip()
    loader = RepositoryLoader(workspace=workspace)

    if source.startswith(("https://", "http://", "ssh://", "git://", "git@")):
        return loader.clone_remote(source)

    if source.startswith("file://"):
        return loader.clone_remote(source)

    return loader.load_local(source)
