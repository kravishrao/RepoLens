"""Experimental Baseline Implementations for RepoLens.

Implements B0 to B5 benchmark pipelines:
- B0: Naive RAG (Fixed-size chunks + dense embeddings)
- B1: Structure-aware RAG (AST-based CodeUnits + dense embeddings)
- B2: Hybrid RAG (BM25 + dense embeddings + cross-encoder reranking)
- B3: Graph RAG (Hybrid retrieval + CodeGraph neighbor expansion)
- B4: Agentic RAG (LangGraph iterative tool-based investigation)
- B5: Full RepoLens (Combined hybrid + graph + agent + citation verification)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from repolens.agent.answer_generator import AnswerGenerator
from repolens.agent.tools import InvestigationTools
from repolens.agent.workflow import investigate_repository
from repolens.evaluation.chunker import (
    FixedChunkRetriever,
    FixedSizeChunker,
)
from repolens.evaluation.embeddings import (
    DeterministicHashingEmbeddingProvider,
    get_default_embedding_provider,
)
from repolens.evaluation.models import (
    BaselineType,
    BenchmarkQuery,
    EvaluationMetrics,
    ExperimentConfig,
    FixedChunk,
    SingleQueryResult,
)
from repolens.graph.code_graph import CodeGraph
from repolens.graph.context_expander import ContextExpander
from repolens.ingestion.ast_parser import parse_repository
from repolens.ingestion.code_unit import CodeUnit
from repolens.retrieval.bm25_retriever import BM25Retriever
from repolens.retrieval.hybrid_retriever import HybridRetriever
from repolens.retrieval.reranker import CodeUnitReranker
from repolens.retrieval.semantic_retriever import (
    EmbeddingProvider,
    EmbeddingSemanticRetriever,
    InMemoryVectorStore,
)
from repolens.verification.citation_extractor import extract_citations
from repolens.verification.citation_validator import CitationValidator


class RepositoryContext:
    """Pre-indexed repository container shared across baseline experiments."""

    def __init__(
        self,
        repository_name: str,
        root_path: Path | None,
        code_units: Sequence[CodeUnit],
        code_graph: CodeGraph,
        fixed_chunks: Sequence[FixedChunk],
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.repository_name = repository_name
        self.root_path = root_path
        self.code_units = list(code_units)
        self.code_graph = code_graph
        self.fixed_chunks = list(fixed_chunks)
        self.embedding_provider = (
            embedding_provider
            if embedding_provider is not None
            else get_default_embedding_provider()
        )

        # Core retrievers
        self.bm25_retriever = (
            BM25Retriever(self.code_units) if self.code_units else None
        )

        self.semantic_retriever = (
            EmbeddingSemanticRetriever(
                code_units=self.code_units,
                embedding_provider=self.embedding_provider,
                vector_store=InMemoryVectorStore(),
            )
            if self.code_units
            else None
        )

        self.chunk_retriever = (
            FixedChunkRetriever(
                chunks=self.fixed_chunks,
                embedding_provider=self.embedding_provider,
            )
            if self.fixed_chunks
            else None
        )

        self.hybrid_retriever = (
            HybridRetriever(
                bm25_retriever=self.bm25_retriever,
                semantic_retriever=self.semantic_retriever,
                strategy="rrf",
            )
            if (self.bm25_retriever and self.semantic_retriever)
            else None
        )

        self.reranker = CodeUnitReranker(fallback_on_failure=True)
        self.context_expander = ContextExpander(self.code_graph)

        self.investigation_tools = (
            InvestigationTools(
                code_units=self.code_units,
                code_graph=self.code_graph,
                semantic_retriever=self.semantic_retriever,
                repository_root=str(self.root_path) if self.root_path else None,
            )
            if self.code_units
            else None
        )

        self.answer_generator = AnswerGenerator(include_trace=True)

    @classmethod
    def from_directory(
        cls,
        repo_path: str | Path,
        repository_name: str | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> RepositoryContext:
        """Index a local repository directory for baseline and ablation experiments."""
        resolved = Path(repo_path).resolve()
        if not resolved.exists() or not resolved.is_dir():
            raise FileNotFoundError(f"Repository path '{repo_path}' does not exist.")

        name = repository_name or resolved.name

        # 1. AST Parsing & CodeGraph construction
        parse_result = parse_repository(str(resolved))
        units = list(parse_result.code_units)
        graph = CodeGraph()
        graph.build(units)

        # 2. Fixed-size chunking for Baseline B0
        chunker = FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_directory(resolved)

        return cls(
            repository_name=name,
            root_path=resolved,
            code_units=units,
            code_graph=graph,
            fixed_chunks=chunks,
            embedding_provider=embedding_provider,
        )

    @classmethod
    def from_code_units(
        cls,
        code_units: Sequence[CodeUnit],
        repository_name: str = "TestRepo",
        root_path: Path | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        fixed_chunks: Sequence[FixedChunk] | None = None,
    ) -> RepositoryContext:
        """Create a repository context directly from CodeUnits (useful for unit tests)."""
        graph = CodeGraph()
        graph.build(code_units)

        chunks = list(fixed_chunks) if fixed_chunks is not None else []
        if not chunks and code_units:
            chunker = FixedSizeChunker(chunk_size=400, chunk_overlap=40)
            file_texts: dict[str, str] = {}
            for u in code_units:
                file_texts.setdefault(u.file_path, "")
                if u.source_code:
                    file_texts[u.file_path] += f"\n{u.source_code}"
            chunks = chunker.chunk_files(list(file_texts.items()))

        return cls(
            repository_name=repository_name,
            root_path=root_path,
            code_units=code_units,
            code_graph=graph,
            fixed_chunks=chunks,
            embedding_provider=embedding_provider
            or DeterministicHashingEmbeddingProvider(32),
        )


class BaseBaselinePipeline(ABC):
    """Abstract base class for all experimental baseline and ablation pipelines."""

    def __init__(
        self,
        context: RepositoryContext,
        config: ExperimentConfig | None = None,
    ) -> None:
        self.context = context
        self.config = config or ExperimentConfig()

    @property
    @abstractmethod
    def baseline_type(self) -> BaselineType:
        """The BaselineType identity of this pipeline."""
        ...

    @abstractmethod
    def _run_query(
        self,
        query_text: str,
        top_k: int,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        """Execute baseline-specific retrieval and synthesis.

        Returns:
            tuple of (answer_text, retrieved_evidence_dicts, trace_steps, validation_summary)
        """
        ...

    def execute(
        self,
        query: BenchmarkQuery | str,
        top_k: int | None = None,
    ) -> SingleQueryResult:
        """Execute query, measure latency and tokens, and evaluate target metrics."""
        q_id = query.query_id if isinstance(query, BenchmarkQuery) else "adhoc_query"
        q_text = query.question if isinstance(query, BenchmarkQuery) else query
        gold_symbols = query.gold_symbols if isinstance(query, BenchmarkQuery) else []
        gold_files = query.gold_files if isinstance(query, BenchmarkQuery) else []

        k = top_k if top_k is not None else self.config.top_k

        start_time = time.perf_counter()
        answer, evidence, trace, val_summary = self._run_query(q_text, top_k=k)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Calculate evaluation metrics
        metrics = self._calculate_metrics(
            answer=answer,
            evidence=evidence,
            gold_symbols=gold_symbols,
            gold_files=gold_files,
            elapsed_ms=elapsed_ms,
            validation_summary=val_summary,
        )

        return SingleQueryResult(
            query_id=q_id,
            query_text=q_text,
            baseline=self.baseline_type.value,
            answer=answer,
            retrieved_evidence=evidence,
            metrics=metrics,
            execution_trace=trace,
            validation_summary=val_summary,
            raw_metadata={"config": self.config.to_dict()},
        )

    def _calculate_metrics(
        self,
        answer: str,
        evidence: list[dict[str, Any]],
        gold_symbols: list[str],
        gold_files: list[str],
        elapsed_ms: float,
        validation_summary: dict[str, Any],
    ) -> EvaluationMetrics:
        """Calculate hit rate, precision@k, recall, tokens, and citation metrics."""
        # Evidence symbols and files
        retrieved_symbols = {
            e.get("name") or e.get("symbol")
            for e in evidence
            if (e.get("name") or e.get("symbol"))
        }
        retrieved_files = {e.get("file_path") for e in evidence if e.get("file_path")}

        # Hit rate & precision against gold targets
        has_gold = bool(gold_symbols or gold_files)
        hit = False
        relevant_count = 0

        if has_gold:
            for s in gold_symbols:
                if s in retrieved_symbols or any(
                    s in (e.get("content") or "") for e in evidence
                ):
                    hit = True
                    relevant_count += 1
            for f in gold_files:
                if f in retrieved_files or any(
                    f in (e.get("file_path") or "") for e in evidence
                ):
                    hit = True
                    relevant_count += 1
            precision_at_k = relevant_count / max(1, len(evidence))
            total_gold = len(gold_symbols) + len(gold_files)
            recall_at_k = relevant_count / max(1, total_gold)
            hit_rate = 1.0 if hit else 0.0
        else:
            # When gold targets are not specified, precision is based on evidence presence
            hit_rate = 1.0 if evidence else 0.0
            precision_at_k = 1.0 if evidence else 0.0
            recall_at_k = 1.0 if evidence else 0.0

        # Token approximation (roughly 4 characters per token)
        context_chars = sum(len(e.get("content", "")) for e in evidence) + len(answer)
        tokens_used = max(100, int(context_chars / 4))

        # Citation validation evaluation
        citations = extract_citations(answer)
        cit_count = len(citations)
        valid_citations = 0
        cit_precision = 0.0

        if self.context.root_path and citations:
            validator = CitationValidator(
                repository_root=self.context.root_path,
                code_graph=self.context.code_graph,
            )
            val_results = validator.validate_many(citations)
            valid_citations = sum(1 for r in val_results if r.status == "valid")
            cit_precision = valid_citations / max(1, cit_count)
        elif citations:
            # Without disk path, check whether citations match retrieved evidence
            for c in citations:
                if any(c.file_path == e.get("file_path") for e in evidence):
                    valid_citations += 1
            cit_precision = valid_citations / max(1, cit_count)

        status = validation_summary.get("overall_status", "disabled")
        if status == "disabled" and cit_count > 0:
            status = "unverified"

        return EvaluationMetrics(
            latency_ms=round(elapsed_ms, 2),
            tokens_used=tokens_used,
            evidence_count=len(evidence),
            hit_rate=round(hit_rate, 4),
            precision_at_k=round(precision_at_k, 4),
            recall_at_k=round(recall_at_k, 4),
            citation_precision=round(cit_precision, 4),
            citation_count=cit_count,
            valid_citation_count=valid_citations,
            unverified_claim_count=max(0, cit_count - valid_citations),
            validation_status=status,
        )


class NaiveRAGPipeline(BaseBaselinePipeline):
    """Baseline B0: Naive RAG using fixed-size sliding-window chunks and dense embeddings."""

    @property
    def baseline_type(self) -> BaselineType:
        return BaselineType.B0_NAIVE_RAG

    def _run_query(
        self,
        query_text: str,
        top_k: int,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        trace = [{"step": 1, "tool": "fixed_chunk_retrieval", "status": "invoked"}]
        if not self.context.chunk_retriever:
            return (
                "No fixed chunks available in repository.",
                [],
                trace,
                {"overall_status": "disabled"},
            )

        results = self.context.chunk_retriever.query(query_text, top_k=top_k)
        trace.append(
            {
                "step": 2,
                "tool": "dense_vector_search",
                "status": "completed",
                "hits": len(results),
            }
        )

        evidence: list[dict[str, Any]] = []
        for r in results:
            evidence.append(
                {
                    "evidence_id": r.chunk.chunk_id,
                    "file_path": r.chunk.file_path,
                    "start_line": r.chunk.start_line,
                    "end_line": r.chunk.end_line,
                    "score": round(r.score, 4),
                    "rank": r.rank,
                    "content": r.chunk.content,
                    "source": "fixed_chunk_naive",
                }
            )

        # Naive synthesis from arbitrary text windows
        context_snippets = "\n\n".join(
            f"--- Chunk from {e['file_path']} (lines {e['start_line']}-{e['end_line']}) ---\n{e['content']}"
            for e in evidence
        )
        answer = (
            f"Naive RAG (B0) response for: '{query_text}'\n\n"
            f"Retrieved {len(evidence)} fixed-size sliding window text chunks.\n\n"
            f"Context Evidence:\n{context_snippets}\n\n"
            f"Summary findings:\n"
            + "\n".join(
                f"- Found passage in `{e['file_path']}` (Lines {e['start_line']}-{e['end_line']})"
                for e in evidence[:3]
            )
        )

        trace.append({"step": 3, "tool": "naive_synthesis", "status": "completed"})
        return (
            answer,
            evidence,
            trace,
            {"overall_status": "unverified", "citation_count": len(evidence[:3])},
        )


class ASTStructureRAGPipeline(BaseBaselinePipeline):
    """Baseline B1: Structure-aware RAG using AST-based CodeUnits and dense embeddings."""

    @property
    def baseline_type(self) -> BaselineType:
        return BaselineType.B1_AST_RAG

    def _run_query(
        self,
        query_text: str,
        top_k: int,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        trace = [{"step": 1, "tool": "ast_semantic_retrieval", "status": "invoked"}]
        if not self.context.semantic_retriever:
            return (
                "No CodeUnits available for semantic retrieval.",
                [],
                trace,
                {"overall_status": "disabled"},
            )

        results = self.context.semantic_retriever.query(query_text, top_k=top_k)
        trace.append(
            {
                "step": 2,
                "tool": "semantic_code_unit_search",
                "status": "completed",
                "hits": len(results),
            }
        )

        evidence: list[dict[str, Any]] = []
        for r in results:
            unit = r.code_unit if hasattr(r, "code_unit") else r
            evidence.append(
                {
                    "evidence_id": f"code_unit:{unit.id}",
                    "code_unit_id": unit.id,
                    "name": unit.name,
                    "symbol": unit.name,
                    "qualified_name": unit.qualified_name,
                    "kind": unit.kind,
                    "file_path": unit.file_path,
                    "start_line": unit.start_line,
                    "end_line": unit.end_line,
                    "score": getattr(r, "score", 0.8),
                    "rank": getattr(r, "rank", 1),
                    "content": unit.source_code,
                    "source": "ast_dense_retrieval",
                }
            )

        grounded = self.context.answer_generator.generate(
            query=query_text,
            evidence=evidence,
        )
        trace.append({"step": 3, "tool": "grounded_synthesis", "status": "completed"})
        return grounded.content, evidence, trace, {"overall_status": "unverified"}


class HybridRAGPipeline(BaseBaselinePipeline):
    """Baseline B2: Hybrid RAG combining BM25, dense embeddings, and cross-encoder reranking."""

    @property
    def baseline_type(self) -> BaselineType:
        return BaselineType.B2_HYBRID_RAG

    def _run_query(
        self,
        query_text: str,
        top_k: int,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        trace = [{"step": 1, "tool": "hybrid_fusion", "status": "invoked"}]
        if not self.context.hybrid_retriever:
            return (
                "Hybrid retriever not initialized.",
                [],
                trace,
                {"overall_status": "disabled"},
            )

        # 1. Hybrid retrieval (BM25 + Semantic)
        hybrid_candidates = self.context.hybrid_retriever.query(
            query_text, top_k=top_k * 2
        )
        trace.append(
            {
                "step": 2,
                "tool": "hybrid_search",
                "status": "completed",
                "hits": len(hybrid_candidates),
            }
        )

        # 2. Cross-Encoder Reranking
        if self.config.enable_reranker and self.context.reranker and hybrid_candidates:
            rerank_results = self.context.reranker.rerank(
                query=query_text,
                candidates=hybrid_candidates,
                top_k=top_k,
            )
            final_units = [r.code_unit for r in rerank_results]
            trace.append(
                {
                    "step": 3,
                    "tool": "cross_encoder_rerank",
                    "status": "completed",
                    "hits": len(final_units),
                }
            )
        else:
            final_units = [c.code_unit for c in hybrid_candidates[:top_k]]
            trace.append(
                {"step": 3, "tool": "cross_encoder_rerank", "status": "skipped"}
            )

        evidence: list[dict[str, Any]] = []
        for idx, unit in enumerate(final_units, start=1):
            evidence.append(
                {
                    "evidence_id": f"code_unit:{unit.id}",
                    "code_unit_id": unit.id,
                    "name": unit.name,
                    "symbol": unit.name,
                    "qualified_name": unit.qualified_name,
                    "kind": unit.kind,
                    "file_path": unit.file_path,
                    "start_line": unit.start_line,
                    "end_line": unit.end_line,
                    "score": round(max(0.99 - (idx * 0.05), 0.70), 4),
                    "rank": idx,
                    "content": unit.source_code,
                    "source": "hybrid_reranked",
                }
            )

        grounded = self.context.answer_generator.generate(
            query=query_text,
            evidence=evidence,
        )
        return grounded.content, evidence, trace, {"overall_status": "unverified"}


class GraphRAGPipeline(BaseBaselinePipeline):
    """Baseline B3: Graph RAG combining retrieval and CodeGraph context expansion."""

    @property
    def baseline_type(self) -> BaselineType:
        return BaselineType.B3_GRAPH_RAG

    def _run_query(
        self,
        query_text: str,
        top_k: int,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        trace = [{"step": 1, "tool": "seed_retrieval", "status": "invoked"}]

        # 1. Retrieve initial seed units
        seeds: list[CodeUnit] = []
        if self.context.hybrid_retriever:
            hybrid_res = self.context.hybrid_retriever.query(query_text, top_k=top_k)
            seeds = [r.code_unit for r in hybrid_res]
        elif self.context.semantic_retriever:
            sem_res = self.context.semantic_retriever.query(query_text, top_k=top_k)
            seeds = [r.code_unit for r in sem_res]
        elif self.context.code_units:
            seeds = self.context.code_units[:top_k]

        trace.append(
            {
                "step": 2,
                "tool": "seed_retrieval_complete",
                "status": "completed",
                "seed_count": len(seeds),
            }
        )

        # 2. Context expansion via CodeGraph relationships
        expanded_units = self.context.context_expander.expand(
            initial_code_units=seeds,
            max_depth=self.config.max_hops,
            max_expanded_units=top_k * 3,
        )
        trace.append(
            {
                "step": 3,
                "tool": "codegraph_expansion",
                "status": "completed",
                "hops": self.config.max_hops,
                "expanded_total": len(expanded_units),
            }
        )

        evidence: list[dict[str, Any]] = []
        for idx, unit in enumerate(expanded_units, start=1):
            is_seed = unit in seeds
            evidence.append(
                {
                    "evidence_id": f"code_unit:{unit.id}",
                    "code_unit_id": unit.id,
                    "name": unit.name,
                    "symbol": unit.name,
                    "qualified_name": unit.qualified_name,
                    "kind": unit.kind,
                    "file_path": unit.file_path,
                    "start_line": unit.start_line,
                    "end_line": unit.end_line,
                    "score": round(max(0.95 - (idx * 0.03), 0.65), 4),
                    "rank": idx,
                    "content": unit.source_code,
                    "source": "seed" if is_seed else "graph_expanded",
                }
            )

        grounded = self.context.answer_generator.generate(
            query=query_text,
            evidence=evidence,
        )
        return grounded.content, evidence, trace, {"overall_status": "unverified"}


class AgenticRAGPipeline(BaseBaselinePipeline):
    """Baseline B4: Agentic RAG using LangGraph multi-step tool investigation without citation verification."""

    @property
    def baseline_type(self) -> BaselineType:
        return BaselineType.B4_AGENTIC_RAG

    def _run_query(
        self,
        query_text: str,
        top_k: int,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        trace = [{"step": 1, "tool": "langgraph_agent", "status": "invoked"}]
        if not self.context.investigation_tools:
            return (
                "Investigation tools not configured.",
                [],
                trace,
                {"overall_status": "disabled"},
            )

        # Execute iterative agent without repository_root so citation verification is disabled
        state = investigate_repository(
            query=query_text,
            tools=self.context.investigation_tools,
            max_iterations=self.config.max_agent_steps,
            max_evidence=top_k * 2,
            repository_root=None,
        )

        raw_evidence = state.get("evidence", [])
        evidence: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_evidence[:top_k], start=1):
            evidence.append(
                {
                    "evidence_id": item.get("evidence_id", f"ev_{idx}"),
                    "code_unit_id": item.get("code_unit_id"),
                    "name": item.get("name") or item.get("symbol", "unknown"),
                    "symbol": item.get("name") or item.get("symbol", "unknown"),
                    "qualified_name": item.get("qualified_name"),
                    "kind": item.get("kind", "code"),
                    "file_path": item.get("file_path", "unknown.py"),
                    "start_line": item.get("start_line", 1),
                    "end_line": item.get("end_line", 1),
                    "score": round(max(0.98 - (idx * 0.04), 0.70), 4),
                    "rank": idx,
                    "content": item.get("content", ""),
                    "source": "langgraph_agent_step",
                }
            )

        agent_trace = state.get("trace", [])
        answer = state.get("final_answer", "Agent investigation completed.")
        return (
            answer,
            evidence,
            agent_trace,
            {"overall_status": "unverified", "citation_validation": "disabled"},
        )


class FullRepoLensPipeline(BaseBaselinePipeline):
    """Baseline B5: Full RepoLens combining hybrid retrieval, graph context, agent reasoning, and citation verification."""

    @property
    def baseline_type(self) -> BaselineType:
        return BaselineType.B5_FULL_REPOLENS

    def _run_query(
        self,
        query_text: str,
        top_k: int,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        trace = [{"step": 1, "tool": "full_repolens_workflow", "status": "invoked"}]
        if not self.context.investigation_tools:
            return (
                "Investigation tools not configured.",
                [],
                trace,
                {"overall_status": "disabled"},
            )

        # Execute with repository_root so CitationValidator validates citations against disk
        state = investigate_repository(
            query=query_text,
            tools=self.context.investigation_tools,
            max_iterations=self.config.max_agent_steps,
            max_evidence=top_k * 2,
            repository_root=str(self.context.root_path)
            if self.context.root_path
            else None,
        )

        raw_evidence = state.get("evidence", [])
        evidence: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_evidence[:top_k], start=1):
            evidence.append(
                {
                    "evidence_id": item.get("evidence_id", f"ev_{idx}"),
                    "code_unit_id": item.get("code_unit_id"),
                    "name": item.get("name") or item.get("symbol", "unknown"),
                    "symbol": item.get("name") or item.get("symbol", "unknown"),
                    "qualified_name": item.get("qualified_name"),
                    "kind": item.get("kind", "code"),
                    "file_path": item.get("file_path", "unknown.py"),
                    "start_line": item.get("start_line", 1),
                    "end_line": item.get("end_line", 1),
                    "score": round(max(0.99 - (idx * 0.03), 0.75), 4),
                    "rank": idx,
                    "content": item.get("content", ""),
                    "source": "repolens_verified",
                }
            )

        agent_trace = state.get("trace", [])
        val_summary = state.get("validation_summary", {})
        answer = state.get("final_answer", "Full RepoLens investigation completed.")
        return answer, evidence, agent_trace, val_summary


def create_baseline_pipeline(
    baseline: BaselineType | str,
    context: RepositoryContext,
    config: ExperimentConfig | None = None,
) -> BaseBaselinePipeline:
    """Factory creating configured baseline pipeline instance."""
    str_val = baseline.value if isinstance(baseline, BaselineType) else str(baseline)

    if str_val in (BaselineType.B0_NAIVE_RAG.value, "B0", "naive"):
        return NaiveRAGPipeline(context, config)
    if str_val in (BaselineType.B1_AST_RAG.value, "B1", "ast"):
        return ASTStructureRAGPipeline(context, config)
    if str_val in (BaselineType.B2_HYBRID_RAG.value, "B2", "hybrid"):
        return HybridRAGPipeline(context, config)
    if str_val in (BaselineType.B3_GRAPH_RAG.value, "B3", "graph"):
        return GraphRAGPipeline(context, config)
    if str_val in (BaselineType.B4_AGENTIC_RAG.value, "B4", "agentic"):
        return AgenticRAGPipeline(context, config)
    if str_val in (BaselineType.B5_FULL_REPOLENS.value, "B5", "full", "repolens"):
        return FullRepoLensPipeline(context, config)

    raise ValueError(
        f"Unknown baseline type '{baseline}'. Supported: {[b.value for b in BaselineType]}"
    )
