from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from repolens.ingestion.code_unit import CodeUnit

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_SPLIT_PATTERN = re.compile(r"[^a-zA-Z0-9]+")
_CAMEL_BOUNDARY = re.compile(
    r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[0-9])(?=[A-Z])"
)

# Common English and Python stop-words that carry no discriminating signal.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        # English
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "not",
        "no",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "if",
        "then",
        "than",
        "that",
        "this",
        "it",
        "its",
        "by",
        "from",
        "up",
        "out",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "each",
        "more",
        "also",
        "just",
        "should",
        "would",
        "could",
        "will",
        "can",
        "may",
        "might",
        # Python built-ins / keywords
        "def",
        "class",
        "return",
        "import",
        "pass",
        "none",
        "true",
        "false",
        "self",
        "cls",
        "init",
        "str",
        "int",
        "list",
        "dict",
        "set",
        "tuple",
        "type",
        "object",
        "super",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric chars, expand camelCase/PascalCase.

    Example
    -------
    >>> tokenize("loadRepository")
    ['load', 'repository']
    >>> tokenize("file_path_discovery")
    ['file', 'path', 'discovery']
    """
    # Expand camelCase / PascalCase before splitting on punctuation.
    text = _CAMEL_BOUNDARY.sub(" ", text)
    # Split on anything that is not a letter or digit.
    tokens = _SPLIT_PATTERN.split(text.lower())
    return [t for t in tokens if t and not t.isdigit()]


def _make_document(unit: CodeUnit) -> str:
    """Flatten a CodeUnit into a single text document for indexing."""
    parts: list[str] = [
        unit.name,
        unit.qualified_name,
        unit.kind,
        unit.file_path,
    ]

    # Imports add module-name signal.
    for imp in unit.imports:
        parts.append(imp.module)

    # Call targets add callee-name signal.
    for call in unit.calls:
        parts.append(call.target)

    # Any extra metadata stored as strings.
    for value in unit.metadata.values():
        if isinstance(value, str):
            parts.append(value)

    return " ".join(parts)


def preprocess(unit: CodeUnit) -> list[str]:
    """Return the token list that represents ``unit`` in the BM25 index."""
    raw_tokens = tokenize(_make_document(unit))
    return [t for t in raw_tokens if t not in _STOP_WORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalResult:
    """A single ranked result returned by :class:`BM25Retriever`."""

    code_unit: CodeUnit
    score: float
    rank: int

    # Convenience accessors — mirrors the most-used CodeUnit fields so callers
    # do not have to unpack `code_unit` themselves.
    @property
    def code_unit_id(self) -> str:
        return self.code_unit.id

    @property
    def name(self) -> str:
        return self.code_unit.name

    @property
    def qualified_name(self) -> str:
        return self.code_unit.qualified_name

    @property
    def kind(self) -> str:
        return self.code_unit.kind

    @property
    def file_path(self) -> str:
        return self.code_unit.file_path

    @property
    def start_line(self) -> int:
        return self.code_unit.start_line

    @property
    def end_line(self) -> int:
        return self.code_unit.end_line


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

#: Okapi BM25 free parameters (standard defaults).
_K1: float = 1.5  # term-frequency saturation
_B: float = 0.75  # length normalisation strength


@dataclass
class _IndexedDocument:
    """Internal storage for one indexed CodeUnit."""

    unit: CodeUnit
    tokens: list[str]
    term_freq: Counter[str] = field(default_factory=Counter)
    length: int = 0

    def __post_init__(self) -> None:
        self.term_freq = Counter(self.tokens)
        self.length = len(self.tokens)


class BM25Retriever:
    """
    Keyword retrieval over a collection of :class:`~repolens.ingestion.code_unit.CodeUnit`
    objects using the Okapi BM25 ranking function.

    BM25 scores each document (CodeUnit) for a given query by combining:

    * **TF saturation** - frequent query terms in a document increase the
      score, but with diminishing returns (controlled by *k1*).
    * **IDF weighting** - rare terms across the corpus are worth more than
      common ones.
    * **Length normalisation** - shorter documents are not unfairly
      penalised for having lower raw term counts (controlled by *b*).

    Usage
    -----
    ::

        retriever = BM25Retriever(code_units)
        results = retriever.query("repository loader clone", top_k=5)
        for r in results:
            print(r.rank, r.score, r.qualified_name, r.file_path)

    Parameters
    ----------
    code_units:
        Any iterable of :class:`~repolens.ingestion.code_unit.CodeUnit`
        objects.  Duplicates (by ``id``) are silently deduplicated.
    k1:
        BM25 term-frequency saturation parameter (default 1.5).
    b:
        BM25 length-normalisation parameter (default 0.75).
    """

    def __init__(
        self,
        code_units: Iterable[CodeUnit],
        *,
        k1: float = _K1,
        b: float = _B,
    ) -> None:
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"BM25 parameter b must be in [0, 1]; got {b}.")
        if k1 < 0.0:
            raise ValueError(f"BM25 parameter k1 must be >= 0; got {k1}.")

        self._k1 = k1
        self._b = b

        # Deduplicate by CodeUnit.id.
        seen_ids: set[str] = set()
        self._documents: list[_IndexedDocument] = []

        for unit in code_units:
            if not isinstance(unit, CodeUnit):
                raise TypeError(
                    f"Expected CodeUnit instances; got {type(unit).__name__}."
                )
            if unit.id in seen_ids:
                continue
            seen_ids.add(unit.id)
            self._documents.append(_IndexedDocument(unit=unit, tokens=preprocess(unit)))

        self._n = len(self._documents)
        self._avg_dl: float = (
            sum(d.length for d in self._documents) / self._n if self._n > 0 else 0.0
        )

        # Build inverted index: term -> list of (doc_index, term_freq)
        self._inverted: dict[str, list[tuple[int, int]]] = {}
        for idx, doc in enumerate(self._documents):
            for term, freq in doc.term_freq.items():
                self._inverted.setdefault(term, []).append((idx, freq))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def document_count(self) -> int:
        """Number of indexed CodeUnits (after deduplication)."""
        return self._n

    @property
    def is_empty(self) -> bool:
        """True when no CodeUnits have been indexed."""
        return self._n == 0

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """Return the top-*k* :class:`RetrievalResult` objects for *query_text*.

        Parameters
        ----------
        query_text:
            Free-form keyword query (e.g. ``"repository loader clone SSH"``).
        top_k:
            Maximum number of results to return.  When fewer documents score
            above *min_score* the returned list is shorter.
        min_score:
            Discard results whose BM25 score is at or below this threshold.
            Defaults to 0.0 (keep all positive-scoring results).

        Returns
        -------
        list[RetrievalResult]
            Results sorted by descending BM25 score.  The ``rank`` field starts
            at 1.  An empty list is returned for empty queries or when the
            index has no documents.
        """
        if not isinstance(query_text, str):
            raise TypeError(
                f"query_text must be a string; got {type(query_text).__name__}."
            )
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer; got {top_k!r}.")

        query_tokens = [
            t for t in tokenize(query_text) if t not in _STOP_WORDS and len(t) > 1
        ]

        if not query_tokens or self._n == 0:
            return []

        scores = self._score_all(query_tokens)

        # Sort descending, filter by min_score, take top_k.
        ranked = sorted(
            ((score, idx) for idx, score in enumerate(scores) if score > min_score),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        return [
            RetrievalResult(
                code_unit=self._documents[idx].unit,
                score=score,
                rank=rank,
            )
            for rank, (score, idx) in enumerate(ranked, start=1)
        ]

    # ------------------------------------------------------------------
    # BM25 scoring
    # ------------------------------------------------------------------

    def _idf(self, term: str) -> float:
        """Compute the IDF component for *term* (Robertson-Spärck Jones IDF)."""
        df = len(self._inverted.get(term, []))
        if df == 0:
            return 0.0
        return math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)

    def _score_all(self, query_tokens: Sequence[str]) -> list[float]:
        """Return a BM25 score for every document, indexed in insertion order."""
        scores = [0.0] * self._n

        for term in query_tokens:
            idf = self._idf(term)
            if idf <= 0.0:
                continue

            for doc_idx, tf in self._inverted.get(term, []):
                dl = self._documents[doc_idx].length
                # Okapi BM25 term score.
                numerator = tf * (self._k1 + 1.0)
                denominator = tf + self._k1 * (
                    1.0 - self._b + self._b * dl / max(self._avg_dl, 1.0)
                )
                scores[doc_idx] += idf * numerator / denominator

        return scores
