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
    """
    Standard representation of a code element.

    Produced by ingestion/extraction and consumed by the
    graph construction module.
    """

    id: str
    kind: CodeElementType
    name: str
    qualified_name: str
    file_path: str

    start_line: int
    end_line: int

    # Parent CodeUnit ID. None for repository-level modules.
    parent_id: str | None = None

    # Optional relationships discovered during extraction.
    imports: tuple[ImportReference, ...] = field(default_factory=tuple)

    calls: tuple[CallReference, ...] = field(default_factory=tuple)

    # Additional extractor-specific information.
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id.strip():
            raise ValueError("CodeUnit ID cannot be empty.")

        if not self.name.strip():
            raise ValueError("CodeUnit name cannot be empty.")

        if not self.file_path.strip():
            raise ValueError("File path cannot be empty.")

        if self.start_line < 1:
            raise ValueError("Start line must be >= 1.")

        if self.end_line < self.start_line:
            raise ValueError("End line cannot be before start line.")
