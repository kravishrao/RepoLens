"""Evaluation and Ablation Framework for RepoLens.

Provides standardized baseline pipelines (B0 to B5), component ablation runners,
benchmark query testbed, and reproducible experiment result logging.
"""

from __future__ import annotations

from repolens.evaluation.ablation import (
    AblationStudy,
    ComponentContribution,
)
from repolens.evaluation.baselines import (
    AgenticRAGPipeline,
    ASTStructureRAGPipeline,
    BaseBaselinePipeline,
    FullRepoLensPipeline,
    GraphRAGPipeline,
    HybridRAGPipeline,
    NaiveRAGPipeline,
    RepositoryContext,
    create_baseline_pipeline,
)
from repolens.evaluation.benchmark_suite import (
    BENCHMARK_QUERIES,
    get_benchmark_queries,
    prepare_benchmark_context,
    resolve_repository_dataset_path,
)
from repolens.evaluation.chunker import (
    ChunkSearchResult,
    FixedChunkRetriever,
    FixedSizeChunker,
)
from repolens.evaluation.documentation import (
    BASELINE_SPECIFICATIONS,
    generate_ablation_documentation,
    get_baseline_limitations,
)
from repolens.evaluation.embeddings import (
    DeterministicHashingEmbeddingProvider,
    get_default_embedding_provider,
)
from repolens.evaluation.models import (
    AblationComparison,
    AblationType,
    BaselineType,
    BenchmarkQuery,
    EvaluationMetrics,
    ExperimentConfig,
    ExperimentRunResult,
    FixedChunk,
    SingleQueryResult,
)
from repolens.evaluation.runner import (
    ExperimentRunner,
    get_git_commit_hash,
    get_system_provenance,
)

__all__ = [
    "BASELINE_SPECIFICATIONS",
    "BENCHMARK_QUERIES",
    "ASTStructureRAGPipeline",
    "AblationComparison",
    "AblationStudy",
    "AblationType",
    "AgenticRAGPipeline",
    "BaseBaselinePipeline",
    "BaselineType",
    "BenchmarkQuery",
    "ChunkSearchResult",
    "ComponentContribution",
    "DeterministicHashingEmbeddingProvider",
    "EvaluationMetrics",
    "ExperimentConfig",
    "ExperimentRunResult",
    "ExperimentRunner",
    "FixedChunk",
    "FixedChunkRetriever",
    "FixedSizeChunker",
    "FullRepoLensPipeline",
    "GraphRAGPipeline",
    "HybridRAGPipeline",
    "NaiveRAGPipeline",
    "RepositoryContext",
    "SingleQueryResult",
    "create_baseline_pipeline",
    "generate_ablation_documentation",
    "get_baseline_limitations",
    "get_benchmark_queries",
    "get_default_embedding_provider",
    "get_git_commit_hash",
    "get_system_provenance",
    "prepare_benchmark_context",
    "resolve_repository_dataset_path",
]
