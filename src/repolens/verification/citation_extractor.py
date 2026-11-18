"""Extract repository source citations from generated answers."""

from __future__ import annotations

import re

from repolens.verification.citation_models import Citation

# Common source-code and query-language file extensions.
_SOURCE_EXTENSIONS = (
    "py",
    "js",
    "jsx",
    "ts",
    "tsx",
    "java",
    "c",
    "h",
    "cpp",
    "hpp",
    "cc",
    "hh",
    "cs",
    "go",
    "rs",
    "rb",
    "php",
    "swift",
    "kt",
    "kts",
    "sql",
    "scala",
    "sc",
    "sh",
    "bash",
    "lua",
    "r",
    "m",
    "mm",
    "dart",
    "vue",
    "svelte",
)

_EXTENSION_PATTERN = "|".join(_SOURCE_EXTENSIONS)

_FILE_PATTERN = rf"(?:[\w.@+-]+[\\/])*" rf"[\w.@+-]+\.(?:{_EXTENSION_PATTERN})"

_PATTERNS = (
    re.compile(
        rf"(?P<path>{_FILE_PATTERN})\s*:\s*" r"(?P<start>\d+)(?:\s*-\s*(?P<end>\d+))?",
        re.IGNORECASE,
    ),
    re.compile(
        rf"`(?P<path>{_FILE_PATTERN})`\s*"
        r"\(\s*lines?\s+(?P<start>\d+)"
        r"(?:\s*-\s*(?P<end>\d+))?\s*\)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^\s*File:\s*(?P<path>{_FILE_PATTERN})\s*$"
        r".*?^\s*Lines:\s*(?P<start>\d+)"
        r"\s*-\s*(?P<end>\d+)\s*$",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    ),
)

_SYMBOL_PATTERN = re.compile(
    r"^\s*\d+\.\s+(?P<symbol>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)",
    re.MULTILINE,
)

_INCOMPLETE_FILE_PATTERN = re.compile(
    rf"^\s*File:\s*(?P<path>{_FILE_PATTERN})\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_citations(answer: str) -> list[Citation]:
    """Extract repository file references and their line ranges."""
    if not isinstance(answer, str):
        raise TypeError("Answer must be a string.")

    citations: list[Citation] = []
    seen: set[tuple[str, int | None, int | None, str | None]] = set()
    symbols = list(_SYMBOL_PATTERN.finditer(answer))

    def find_symbol(position: int) -> str | None:
        preceding = [match for match in symbols if match.start() <= position]
        if not preceding:
            return None
        return preceding[-1].group("symbol")

    for pattern in _PATTERNS:
        for match in pattern.finditer(answer):
            path = match.group("path")
            start = int(match.group("start"))
            end_value = match.groupdict().get("end")
            end = int(end_value) if end_value is not None else start
            symbol = find_symbol(match.start())

            key = (path, start, end, symbol)

            if key in seen:
                continue

            seen.add(key)

            citations.append(
                Citation(
                    citation_id=f"citation-{len(citations) + 1}",
                    raw_text=match.group(0),
                    file_path=path,
                    start_line=start,
                    end_line=end,
                    symbol=symbol,
                )
            )

    for match in _INCOMPLETE_FILE_PATTERN.finditer(answer):
        path = match.group("path")
        symbol = find_symbol(match.start())

        if any(
            citation.file_path == path
            and citation.symbol == symbol
            and citation.start_line is not None
            for citation in citations
        ):
            continue

        key = (path, None, None, symbol)

        if key in seen:
            continue

        seen.add(key)

        citations.append(
            Citation(
                citation_id=f"citation-{len(citations) + 1}",
                raw_text=match.group(0),
                file_path=path,
                symbol=symbol,
            )
        )

    return citations


__all__ = ["extract_citations"]
