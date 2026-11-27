"""Standardized Benchmark Datasets and Query Testbed.

Provides curated evaluation queries with verified gold symbols, target files,
and consistent testbed conditions across benchmark repositories.
"""

from __future__ import annotations

from pathlib import Path

from repolens.evaluation.baselines import RepositoryContext
from repolens.evaluation.embeddings import get_default_embedding_provider
from repolens.evaluation.models import BenchmarkQuery
from repolens.retrieval.semantic_retriever import EmbeddingProvider

# Project roots and bundled dataset paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "repositories"


BENCHMARK_QUERIES: list[BenchmarkQuery] = [
    # RepoLens Self-Evaluation Queries
    BenchmarkQuery(
        query_id="repolens_ast_parser",
        question="Where is parse_repository defined and how does it parse Python AST?",
        repository="RepoLens",
        gold_symbols=["parse_repository", "ASTParser"],
        gold_files=["ast_parser.py", "src/repolens/ingestion/ast_parser.py"],
        category="symbol_lookup",
        description="Verify locating top-level parser entry point and class definition.",
    ),
    BenchmarkQuery(
        query_id="repolens_hybrid_fusion",
        question="How does HybridRetriever combine BM25 and semantic search results?",
        repository="RepoLens",
        gold_symbols=[
            "HybridRetriever",
            "reciprocal_rank_fusion",
            "weighted_score_fusion",
        ],
        gold_files=[
            "hybrid_retriever.py",
            "src/repolens/retrieval/hybrid_retriever.py",
        ],
        category="architectural_component",
        description="Verify locating hybrid search fusion methods and class.",
    ),
    BenchmarkQuery(
        query_id="repolens_context_expander",
        question="Where is ContextExpander defined and how does it traverse CodeGraph relationships?",
        repository="RepoLens",
        gold_symbols=["ContextExpander", "expand"],
        gold_files=["context_expander.py", "src/repolens/graph/context_expander.py"],
        category="call_graph",
        description="Verify graph expansion component exploration.",
    ),
    BenchmarkQuery(
        query_id="repolens_citation_validator",
        question="Where is CitationValidator defined and how does it validate line ranges against source files?",
        repository="RepoLens",
        gold_symbols=["CitationValidator", "validate_citation", "validate_many"],
        gold_files=[
            "citation_validator.py",
            "src/repolens/verification/citation_validator.py",
        ],
        category="ground_truth_verification",
        description="Verify citation validator verification logic.",
    ),
    # Requests Benchmark Queries
    BenchmarkQuery(
        query_id="requests_session_dispatch",
        question="Where is Session defined and how does it manage authentication and requests?",
        repository="requests",
        gold_symbols=["Session", "SessionRedirectMixin"],
        gold_files=["sessions.py", "src/requests/sessions.py"],
        category="symbol_lookup",
        description="Locate core Session object in requests library.",
    ),
    BenchmarkQuery(
        query_id="requests_adapter_retries",
        question="Where is HTTPAdapter defined and how does it configure connection retries?",
        repository="requests",
        gold_symbols=["HTTPAdapter"],
        gold_files=["adapters.py", "src/requests/adapters.py"],
        category="architectural_component",
        description="Locate transport adapter class and mount points.",
    ),
    # Flask Benchmark Queries
    BenchmarkQuery(
        query_id="flask_app_initialization",
        question="Where is Flask class defined and how does it configure routing and wsgi app?",
        repository="flask",
        gold_symbols=["Flask", "App"],
        gold_files=["app.py", "sansio/app.py", "src/flask/app.py"],
        category="symbol_lookup",
        description="Locate Flask main application class.",
    ),
    BenchmarkQuery(
        query_id="flask_blueprint_registration",
        question="Where is Blueprint defined and how does it register routes and url prefixes?",
        repository="flask",
        gold_symbols=["Blueprint"],
        gold_files=["blueprints.py", "src/flask/blueprints.py"],
        category="call_graph",
        description="Locate Blueprint component in Flask.",
    ),
    # Click Benchmark Queries
    BenchmarkQuery(
        query_id="click_command_execution",
        question="Where is Command class defined and how does it parse arguments and invoke callbacks?",
        repository="click",
        gold_symbols=["Command", "BaseCommand"],
        gold_files=["core.py", "src/click/core.py"],
        category="symbol_lookup",
        description="Locate Click CLI Command core class.",
    ),
]


def get_benchmark_queries(
    repository_name: str | None = None,
    category: str | None = None,
) -> list[BenchmarkQuery]:
    """Retrieve benchmark queries filtered by repository name or category."""
    queries = BENCHMARK_QUERIES
    if repository_name:
        norm = repository_name.strip().lower()
        queries = [q for q in queries if q.repository.lower() == norm]
    if category:
        queries = [q for q in queries if q.category == category]
    return list(queries)


def resolve_repository_dataset_path(name: str) -> Path | None:
    """Find local snapshot path for a benchmark dataset repository."""
    norm = name.strip().lower()

    # Direct match in dataset/repositories/
    candidate = DATASET_ROOT / norm
    if candidate.exists() and candidate.is_dir():
        return candidate

    # Look inside project root
    if norm in ("repolens", "self"):
        return PROJECT_ROOT

    # Check if absolute path was supplied
    p = Path(name)
    if p.exists() and p.is_dir():
        return p

    return None


def prepare_benchmark_context(
    repo_name_or_path: str | Path,
    embedding_provider: EmbeddingProvider | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> RepositoryContext:
    """Build a standard RepositoryContext for consistent evaluation conditions."""
    provider = embedding_provider or get_default_embedding_provider()

    if isinstance(repo_name_or_path, Path) and repo_name_or_path.is_dir():
        return RepositoryContext.from_directory(
            repo_path=repo_name_or_path,
            embedding_provider=provider,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    resolved = resolve_repository_dataset_path(str(repo_name_or_path))
    if resolved is not None:
        return RepositoryContext.from_directory(
            repo_path=resolved,
            repository_name=str(repo_name_or_path),
            embedding_provider=provider,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # Fallback to RepoLens self directory if dataset repo not found
    return RepositoryContext.from_directory(
        repo_path=PROJECT_ROOT,
        repository_name=str(repo_name_or_path),
        embedding_provider=provider,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
