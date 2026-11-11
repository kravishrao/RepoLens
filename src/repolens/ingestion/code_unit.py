from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

CodeElementType = Literal[
    "module",
    "class",
    "function",
    "method",
]


@dataclass(frozen=True)
class ImportReference:
    """An import statement discovered in a source file."""

    module: str


@dataclass(frozen=True)
class CallReference:
    """A statically discovered function or method call."""

    target: str


@dataclass(frozen=True)
class CodeUnit:
    """Standard representation of a code element in a repository."""

    id: str
    kind: CodeElementType
    name: str
    qualified_name: str
    file_path: str

    start_line: int
    end_line: int

    parent_id: str | None = None
    imports: tuple[ImportReference, ...] = field(default_factory=tuple)
    calls: tuple[CallReference, ...] = field(default_factory=tuple)

    # Keep metadata in its original position for positional compatibility.
    metadata: Mapping[str, object] = field(default_factory=dict)

    # Additional repository and source information.
    repository: str | None = None
    source_code: str | None = None
    docstring: str | None = None

    @property
    def unit_id(self) -> str:
        """Alias for the existing ID field."""
        return self.id

    @property
    def symbol(self) -> str:
        """Alias for the existing name field."""
        return self.name

    @property
    def type(self) -> CodeElementType:
        """Alias for the existing kind field."""
        return self.kind

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("CodeUnit ID cannot be empty.")

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("CodeUnit name cannot be empty.")

        if not isinstance(self.file_path, str) or not self.file_path.strip():
            raise ValueError("File path cannot be empty.")

        if not isinstance(self.kind, str) or self.kind not in {
            "module",
            "class",
            "function",
            "method",
        }:
            raise ValueError(f"Invalid CodeUnit kind: {self.kind!r}")

        if self.start_line < 1:
            raise ValueError("Start line must be >= 1.")

        if self.end_line < self.start_line:
            raise ValueError("End line cannot be before start line.")

        if self.repository is not None and (
            not isinstance(self.repository, str) or not self.repository.strip()
        ):
            raise ValueError("Repository identity cannot be empty.")

        if self.source_code is not None and not isinstance(self.source_code, str):
            raise TypeError("Source code must be a string or None.")

        if self.docstring is not None and not isinstance(self.docstring, str):
            raise TypeError("Docstring must be a string or None.")
