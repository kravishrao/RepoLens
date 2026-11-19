"""Fixed-size source file chunker for Naive RAG (Baseline B0).

Splits repository source files into fixed character/token windows with sliding
overlap, without relying on AST parsing or symbol boundaries.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from repolens.evaluation.models import FixedChunk
from repolens.retrieval.semantic_retriever import EmbeddingProvider


@dataclass(frozen=True)
class ChunkSearchResult:
    """Ranked search result for a fixed-size chunk."""

    chunk: FixedChunk
    score: float
    rank: int

    @property
    def file_path(self) -> str:
        return self.chunk.file_path

    @property
    def start_line(self) -> int:
        return self.chunk.start_line

    @property
    def end_line(self) -> int:
        return self.chunk.end_line

    @property
    def content(self) -> str:
        return self.chunk.content


class FixedSizeChunker:
    """Splits source files into fixed-size character windows with overlap."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        if (
            not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or chunk_size <= 0
        ):
            raise ValueError("chunk_size must be a positive integer.")

        if (
            not isinstance(chunk_overlap, int)
            or isinstance(chunk_overlap, bool)
            or chunk_overlap < 0
        ):
            raise ValueError("chunk_overlap must be a non-negative integer.")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self,
        text: str,
        file_path: str = "unknown.py",
    ) -> list[FixedChunk]:
        """Split a single file's text into fixed-size chunks with line tracking."""
        if not text:
            return []

        chunks: list[FixedChunk] = []
        step = self.chunk_size - self.chunk_overlap
        text_len = len(text)
        idx = 0

        # Precalculate newline indices for fast character-to-line translation
        newline_offsets = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]

        def char_to_line(char_pos: int) -> int:
            # 1-based line number via binary search or linear search
            import bisect

            return bisect.bisect_right(newline_offsets, char_pos)

        start_char = 0
        while start_char < text_len:
            end_char = min(start_char + self.chunk_size, text_len)
            chunk_content = text[start_char:end_char]

            start_line = char_to_line(start_char)
            end_line = char_to_line(max(start_char, end_char - 1))

            chunk = FixedChunk(
                chunk_id=f"{file_path}:chunk_{idx}",
                file_path=file_path,
                start_char=start_char,
                end_char=end_char,
                start_line=start_line,
                end_line=end_line,
                content=chunk_content,
                chunk_index=idx,
                metadata={
                    "file_path": file_path,
                    "char_count": len(chunk_content),
                    "line_count": max(1, end_line - start_line + 1),
                },
            )
            chunks.append(chunk)
            idx += 1

            if end_char >= text_len:
                break
            start_char += step

        return chunks

    def chunk_files(
        self,
        files: Sequence[tuple[str, str]],
    ) -> list[FixedChunk]:
        """Chunk a sequence of (file_path, content) pairs."""
        all_chunks: list[FixedChunk] = []
        for file_path, content in files:
            all_chunks.extend(self.chunk_text(content, file_path=file_path))
        return all_chunks

    def chunk_directory(
        self,
        repo_path: str | Path,
        max_files: int | None = None,
    ) -> list[FixedChunk]:
        """Discover and chunk all Python files in a directory."""
        path = Path(repo_path).resolve()
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Directory '{repo_path}' does not exist.")

        py_files = sorted(
            [
                p
                for p in path.rglob("*.py")
                if not any(
                    part.startswith((".", "__pycache__", "venv", ".venv"))
                    for part in p.parts
                )
            ]
        )
        if max_files is not None:
            py_files = py_files[:max_files]

        pairs: list[tuple[str, str]] = []
        for p in py_files:
            try:
                rel = str(p.relative_to(path)).replace("\\", "/")
                content = p.read_text(encoding="utf-8", errors="replace")
                pairs.append((rel, content))
            except Exception:
                continue

        return self.chunk_files(pairs)


class FixedChunkRetriever:
    """Vector similarity retriever over fixed-size chunks (Baseline B0)."""

    def __init__(
        self,
        chunks: Iterable[FixedChunk],
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.chunks = list(chunks)
        self.embedding_provider = embedding_provider
        self._vectors: list[list[float]] = []
        self._is_indexed = False

    def _index_chunks(self) -> None:
        if self._is_indexed:
            return
        if not self.chunks:
            self._is_indexed = True
            return

        texts = [
            f"File: {c.file_path}\nLines: {c.start_line}-{c.end_line}\n{c.content}"
            for c in self.chunks
        ]
        raw_vecs = self.embedding_provider.encode(texts)
        self._vectors = [[float(v) for v in vec] for vec in raw_vecs]
        self._is_indexed = True

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> list[ChunkSearchResult]:
        """Retrieve top_k chunks matching query text via cosine similarity."""
        if not query_text.strip() or not self.chunks:
            return []

        self._index_chunks()
        query_vecs = self.embedding_provider.encode([query_text])
        if not query_vecs:
            return []
        q_vec = [float(v) for v in query_vecs[0]]

        scores: list[tuple[FixedChunk, float]] = []
        for chunk, c_vec in zip(self.chunks, self._vectors, strict=True):
            sim = self._cosine_similarity(q_vec, c_vec)
            scores.append((chunk, sim))

        scores.sort(key=lambda item: item[1], reverse=True)
        return [
            ChunkSearchResult(chunk=c, score=score, rank=idx)
            for idx, (c, score) in enumerate(scores[:top_k], start=1)
        ]
