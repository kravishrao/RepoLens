from __future__ import annotations

from pathlib import Path

import pytest

from repolens.evaluation.chunker import (
    FixedChunkRetriever,
    FixedSizeChunker,
)
from repolens.evaluation.embeddings import DeterministicHashingEmbeddingProvider


def test_fixed_size_chunker_validation() -> None:
    """Verify FixedSizeChunker input validation."""
    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        FixedSizeChunker(chunk_size=0)
    with pytest.raises(
        ValueError, match="chunk_overlap must be a non-negative integer"
    ):
        FixedSizeChunker(chunk_size=100, chunk_overlap=-5)
    with pytest.raises(
        ValueError, match="chunk_overlap must be strictly less than chunk_size"
    ):
        FixedSizeChunker(chunk_size=100, chunk_overlap=100)


def test_fixed_size_chunker_empty_text() -> None:
    """Verify chunking empty text returns empty list."""
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    assert chunker.chunk_text("") == []


def test_fixed_size_chunker_lines_and_offsets() -> None:
    """Verify chunk boundaries and line number calculations."""
    text = (
        "def func_a():\n"
        "    x = 1\n"
        "    return x\n"
        "\n"
        "def func_b():\n"
        "    y = 2\n"
        "    return y\n"
    )
    chunker = FixedSizeChunker(chunk_size=40, chunk_overlap=10)
    chunks = chunker.chunk_text(text, file_path="test_module.py")

    assert len(chunks) > 1
    assert chunks[0].file_path == "test_module.py"
    assert chunks[0].start_line == 1
    assert chunks[0].end_line >= 1
    assert chunks[0].chunk_index == 0

    # Every chunk must have content of at most chunk_size
    for c in chunks:
        assert len(c.content) <= 40
        assert c.start_char < c.end_char


def test_fixed_size_chunker_chunk_files() -> None:
    """Verify chunking multiple files."""
    chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
    files = [
        ("a.py", "x = 1\ny = 2\n"),
        ("b.py", "class Foo:\n    pass\n"),
    ]
    chunks = chunker.chunk_files(files)
    assert len(chunks) >= 2
    paths = {c.file_path for c in chunks}
    assert paths == {"a.py", "b.py"}


def test_fixed_size_chunker_directory(tmp_path: Path) -> None:
    """Verify chunking files in a directory."""
    (tmp_path / "foo.py").write_text("def foo():\n    return 'bar'\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("Hello world", encoding="utf-8")

    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_directory(tmp_path)
    assert len(chunks) == 1
    assert chunks[0].file_path == "foo.py"
    assert "foo" in chunks[0].content


def test_fixed_size_chunker_directory_nonexistent() -> None:
    """Verify non-existent directory raises FileNotFoundError."""
    chunker = FixedSizeChunker()
    with pytest.raises(FileNotFoundError):
        chunker.chunk_directory("non/existent/path/for/repolens")


def test_fixed_chunk_retriever_query() -> None:
    """Verify FixedChunkRetriever indexes chunks and retrieves top matches."""
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
    text = (
        "def authenticate_user(username, password):\n"
        "    '''Verify user credentials.'''\n"
        "    return username == 'admin'\n\n"
        "def calculate_tax(amount):\n"
        "    '''Compute sales tax rate.'''\n"
        "    return amount * 0.18\n"
    )
    chunks = chunker.chunk_text(text, file_path="auth.py")
    provider = DeterministicHashingEmbeddingProvider(32)
    retriever = FixedChunkRetriever(chunks, embedding_provider=provider)

    results = retriever.query("authenticate credentials", top_k=2)
    assert len(results) >= 1
    top = results[0]
    assert "authenticate" in top.content or "credentials" in top.content
    assert top.rank == 1
    assert top.file_path == "auth.py"


def test_fixed_chunk_retriever_empty_query() -> None:
    """Verify empty query returns empty list."""
    provider = DeterministicHashingEmbeddingProvider(32)
    retriever = FixedChunkRetriever([], embedding_provider=provider)
    assert retriever.query("") == []
    assert retriever.query("   ") == []
