from __future__ import annotations

from pathlib import Path

from repolens.evaluation.benchmark_suite import (
    get_benchmark_queries,
    prepare_benchmark_context,
    resolve_repository_dataset_path,
)


def test_get_benchmark_queries_all() -> None:
    """Verify loading default benchmark query testbed."""
    queries = get_benchmark_queries()
    assert len(queries) >= 5
    repos = {q.repository.lower() for q in queries}
    assert "repolens" in repos
    assert "requests" in repos


def test_get_benchmark_queries_filter() -> None:
    """Verify filtering benchmark queries by repository and category."""
    rl_queries = get_benchmark_queries(repository_name="RepoLens")
    assert len(rl_queries) >= 3
    assert all(q.repository == "RepoLens" for q in rl_queries)

    symbol_queries = get_benchmark_queries(category="symbol_lookup")
    assert len(symbol_queries) >= 2
    assert all(q.category == "symbol_lookup" for q in symbol_queries)


def test_resolve_repository_dataset_path() -> None:
    """Verify resolving local dataset repositories or project root."""
    rl_path = resolve_repository_dataset_path("RepoLens")
    assert rl_path is not None
    assert rl_path.exists()

    requests_path = resolve_repository_dataset_path("requests")
    assert requests_path is not None
    assert requests_path.exists()


def test_prepare_benchmark_context_directory(tmp_path: Path) -> None:
    """Verify prepare_benchmark_context builds context from directory."""
    (tmp_path / "mod.py").write_text("def test_fn(): pass\n", encoding="utf-8")
    ctx = prepare_benchmark_context(tmp_path)
    assert ctx.root_path == tmp_path.resolve()
    assert len(ctx.code_units) >= 1
    assert any(u.name == "test_fn" for u in ctx.code_units)
