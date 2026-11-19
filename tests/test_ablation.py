from __future__ import annotations

import pytest

from repolens.evaluation.ablation import AblationStudy
from repolens.evaluation.baselines import RepositoryContext
from repolens.evaluation.embeddings import DeterministicHashingEmbeddingProvider
from repolens.evaluation.models import BenchmarkQuery
from repolens.ingestion.code_unit import CodeUnit


@pytest.fixture
def sample_context() -> RepositoryContext:
    units = [
        CodeUnit(
            id="core.py:execute",
            name="execute",
            qualified_name="core.execute",
            kind="function",
            file_path="core.py",
            start_line=1,
            end_line=12,
            docstring="Execute command workflow.",
            source_code="def execute(cmd):\n    return cmd.run()",
        ),
        CodeUnit(
            id="cmd.py:Command",
            name="Command",
            qualified_name="cmd.Command",
            kind="class",
            file_path="cmd.py",
            start_line=1,
            end_line=20,
            docstring="Command representation.",
            source_code="class Command:\n    def run(self):\n        pass",
        ),
    ]
    provider = DeterministicHashingEmbeddingProvider(32)
    return RepositoryContext.from_code_units(
        code_units=units,
        repository_name="AblationSample",
        embedding_provider=provider,
    )


def test_ablation_study_run_suite(sample_context: RepositoryContext) -> None:
    """Verify running full ablation suite generates comparative results."""
    study = AblationStudy(sample_context)
    queries = [
        BenchmarkQuery(
            query_id="q1",
            question="Where is execute defined?",
            repository="AblationSample",
            gold_symbols=["execute"],
            gold_files=["core.py"],
        )
    ]
    comparison = study.run_ablation_suite(queries)

    assert comparison.repository_name == "AblationSample"
    assert "B0: Naive RAG" in comparison.runs
    assert "B1: AST RAG" in comparison.runs
    assert "B2: Hybrid RAG" in comparison.runs
    assert "B3: Graph RAG" in comparison.runs
    assert "B4: Agentic RAG" in comparison.runs
    assert "B5: Full RepoLens" in comparison.runs


def test_ablation_study_compute_contributions(
    sample_context: RepositoryContext,
) -> None:
    """Verify computing marginal contributions across isolated components."""
    study = AblationStudy(sample_context)
    queries = ["Where is execute defined?"]
    comparison = study.run_ablation_suite(queries)
    contributions = study.compute_contributions(comparison)

    assert len(contributions) >= 3
    names = {c.component_name for c in contributions}
    assert "AST CodeUnit Chunking" in names
    assert "CodeGraph Context Expansion" in names
