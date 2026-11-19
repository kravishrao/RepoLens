from __future__ import annotations

import time
from typing import Any


def run_repolens_query(
    repo_name: str,
    query: str,
    baseline_mode: str = "B5: Full RepoLens",
    top_k: int = 5,
    max_hops: int = 2,
    enable_reranker: bool = True,
    enable_verification: bool = True,
    max_agent_steps: int = 8,
) -> dict[str, Any]:
    """Execute a RepoLens repository investigation query.

    When running against the demonstration environment, this adapter returns
    illustrative, evidence-grounded data modeled after the RepoLens AST,
    retrieval, graph, and citation pipeline.

    Args:
        repo_name: Name, path, or URL of the target repository.
        query: Natural language question about the repository.
        baseline_mode: Active evaluation pipeline configuration (B0 to B5).
        top_k: Number of retrieved CodeUnits.
        max_hops: Maximum graph traversal hops for context expansion.
        enable_reranker: Whether cross-encoder reranking is active.
        enable_verification: Whether citation and claim verification is active.
        max_agent_steps: Maximum reasoning steps permitted for the agent.

    Returns:
        dict[str, Any]: Formatted result dictionary containing metrics, answer,
            evidence units, execution trace, and graph relationships.
    """
    # Simulate processing time for UI responsiveness feel
    time.sleep(0.3)

    clean_query = query.strip()
    is_ast_question = "ast" in clean_query.lower() or "codeunit" in clean_query.lower()
    is_retrieval_question = (
        "retrieval" in clean_query.lower() or "index" in clean_query.lower()
    )
    is_trace_question = (
        "trace" in clean_query.lower() or "source-file" in clean_query.lower()
    )

    if is_ast_question:
        answer = (
            "AST CodeUnits in RepoLens are discovered by scanning Python source files "
            "and parsed using Python's native `ast` module. The `ASTParser` extracts top-level "
            "and nested functions, classes, and methods, storing them as immutable `CodeUnit` "
            "schemas with qualified symbol names, line spans, docstrings, and call/import references. "
            "Extraction errors are handled gracefully with recovery mechanisms for syntax anomalies."
        )
        evidence_units = [
            {
                "unit_id": "unit-ast-01",
                "repository": repo_name,
                "commit": "4ac0ca7",
                "file_path": "src/repolens/ingestion/ast_parser.py",
                "symbol": "ASTParser.parse_file",
                "type": "method",
                "start_line": 42,
                "end_line": 96,
                "parent": "ASTParser",
                "imports": [
                    "ast",
                    "pathlib.Path",
                    "repolens.ingestion.code_unit.CodeUnit",
                ],
                "calls": ["ast.parse", "_extract_symbols", "_build_code_unit"],
                "verified": True,
                "score": 0.94,
                "code": (
                    "def parse_file(self, file_path: Path) -> list[CodeUnit]:\n"
                    "    '''Parse a single Python file into discrete CodeUnits.'''\n"
                    "    source = file_path.read_text(encoding='utf-8')\n"
                    "    tree = ast.parse(source, filename=str(file_path))\n"
                    "    return self._extract_symbols(tree, file_path)"
                ),
            },
            {
                "unit_id": "unit-ast-02",
                "repository": repo_name,
                "commit": "4ac0ca7",
                "file_path": "src/repolens/ingestion/code_unit.py",
                "symbol": "CodeUnit",
                "type": "class",
                "start_line": 18,
                "end_line": 64,
                "parent": None,
                "imports": ["dataclasses.dataclass", "typing.Sequence"],
                "calls": [],
                "verified": True,
                "score": 0.89,
                "code": (
                    "@dataclass(frozen=True)\n"
                    "class CodeUnit:\n"
                    "    id: str\n"
                    "    kind: str  # 'function', 'class', 'method'\n"
                    "    name: str\n"
                    "    qualified_name: str\n"
                    "    file_path: str\n"
                    "    start_line: int\n"
                    "    end_line: int\n"
                    "    source_code: str"
                ),
            },
        ]
        graph_edges = [
            {"source": "ast_parser.py", "relation": "defines", "target": "ASTParser"},
            {
                "source": "ASTParser.parse_file",
                "relation": "calls",
                "target": "ast.parse",
            },
            {
                "source": "ASTParser.parse_file",
                "relation": "creates",
                "target": "CodeUnit",
            },
        ]
    elif is_retrieval_question:
        answer = (
            "The hybrid retrieval index in RepoLens is constructed in `HybridRetriever`, "
            "which fuses sparse BM25 lexical scores with dense SentenceTransformer vector "
            "embeddings indexed in FAISS. When cross-encoder reranking is enabled, candidates "
            "are re-scored using Reciprocal Rank Fusion (RRF) before context selection."
        )
        evidence_units = [
            {
                "unit_id": "unit-ret-01",
                "repository": repo_name,
                "commit": "4ac0ca7",
                "file_path": "src/repolens/retrieval/hybrid_retriever.py",
                "symbol": "HybridRetriever.search",
                "type": "method",
                "start_line": 35,
                "end_line": 78,
                "parent": "HybridRetriever",
                "imports": [
                    "repolens.retrieval.bm25_retriever",
                    "repolens.retrieval.semantic_retriever",
                ],
                "calls": [
                    "BM25Retriever.search",
                    "SemanticRetriever.search",
                    "_reciprocal_rank_fusion",
                ],
                "verified": True,
                "score": 0.96,
                "code": (
                    "def search(self, query: str, top_k: int = 10) -> list[HybridSearchResult]:\n"
                    "    '''Execute hybrid search combining BM25 and dense vector retrieval.'''\n"
                    "    bm25_hits = self.bm25_retriever.search(query, top_k=top_k * 2)\n"
                    "    semantic_hits = self.semantic_retriever.search(query, top_k=top_k * 2)\n"
                    "    return self._fuse_results(bm25_hits, semantic_hits, top_k=top_k)"
                ),
            },
            {
                "unit_id": "unit-ret-02",
                "repository": repo_name,
                "commit": "4ac0ca7",
                "file_path": "src/repolens/retrieval/reranker.py",
                "symbol": "CrossEncoderReranker.rerank",
                "type": "method",
                "start_line": 28,
                "end_line": 62,
                "parent": "CrossEncoderReranker",
                "imports": ["sentence_transformers.CrossEncoder"],
                "calls": ["CrossEncoder.predict"],
                "verified": True,
                "score": 0.91,
                "code": (
                    "def rerank(self, query: str, candidates: list[CodeUnit]) -> list[RankedResult]:\n"
                    "    '''Score (query, code) pairs with cross-encoder for precision reranking.'''\n"
                    "    pairs = [[query, unit.source_code] for unit in candidates]\n"
                    "    scores = self.model.predict(pairs)\n"
                    "    return sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)"
                ),
            },
        ]
        graph_edges = [
            {
                "source": "HybridRetriever",
                "relation": "uses",
                "target": "BM25Retriever",
            },
            {
                "source": "HybridRetriever",
                "relation": "uses",
                "target": "SemanticRetriever",
            },
            {
                "source": "HybridRetriever.search",
                "relation": "calls",
                "target": "CrossEncoderReranker.rerank",
            },
        ]
    elif is_trace_question:
        answer = (
            "The investigation starts in the LangGraph workflow (`build_investigation_workflow`), "
            "where the query analyzer determines search keywords and target symbols. Tools are dispatched "
            "to query the hybrid retriever and expand graph neighbors through `ContextExpander`. Finally, "
            "source files are read to populate exact citation spans verified against file contents."
        )
        evidence_units = [
            {
                "unit_id": "unit-agent-01",
                "repository": repo_name,
                "commit": "4ac0ca7",
                "file_path": "src/repolens/agent/workflow.py",
                "symbol": "build_investigation_workflow",
                "type": "function",
                "start_line": 85,
                "end_line": 140,
                "parent": None,
                "imports": ["langgraph.graph.StateGraph", "repolens.agent.nodes"],
                "calls": ["StateGraph", "add_node", "add_conditional_edges"],
                "verified": True,
                "score": 0.95,
                "code": (
                    "def build_investigation_workflow(tools: InvestigationTools) -> CompiledGraph:\n"
                    "    '''Construct the LangGraph state machine for repository investigation.'''\n"
                    "    builder = StateGraph(InvestigationState)\n"
                    "    builder.add_node('analyze_query', analyze_query_node)\n"
                    "    builder.add_node('retrieve_evidence', retrieve_evidence_node)\n"
                    "    builder.add_node('verify_citations', verify_citations_node)\n"
                    "    return builder.compile()"
                ),
            }
        ]
        graph_edges = [
            {
                "source": "workflow.py",
                "relation": "defines",
                "target": "build_investigation_workflow",
            },
            {
                "source": "build_investigation_workflow",
                "relation": "configures",
                "target": "InvestigationState",
            },
            {
                "source": "InvestigationState",
                "relation": "aggregates",
                "target": "EvidenceList",
            },
        ]
    else:
        answer = (
            f"Investigation for query: '{clean_query}'. RepoLens analyzed the repository "
            f"structure for '{repo_name}' using configuration '{baseline_mode}'. "
            f"Candidate CodeUnits were retrieved via hybrid lexical/semantic matching, "
            f"expanded across {max_hops} code graph traversal hops, and verified against source definitions."
        )
        evidence_units = [
            {
                "unit_id": "unit-gen-01",
                "repository": repo_name,
                "commit": "4ac0ca7",
                "file_path": "src/repolens/agent/workflow.py",
                "symbol": "investigate_repository",
                "type": "function",
                "start_line": 150,
                "end_line": 210,
                "parent": None,
                "imports": ["repolens.agent.tools.InvestigationTools"],
                "calls": ["workflow.invoke", "_validate_answer"],
                "verified": enable_verification,
                "score": 0.92,
                "code": (
                    "def investigate_repository(query: str, repo_path: Path) -> InvestigationResult:\n"
                    "    '''Run end-to-end repository investigation workflow.'''\n"
                    "    state = initial_state(query=query, repository_root=repo_path)\n"
                    "    final_state = workflow.invoke(state)\n"
                    "    return finalize_result(final_state)"
                ),
            }
        ]
        graph_edges = [
            {
                "source": "investigate_repository",
                "relation": "invokes",
                "target": "build_investigation_workflow",
            },
            {
                "source": "build_investigation_workflow",
                "relation": "reads",
                "target": "CodeGraph",
            },
        ]

    # Calculate realistic metrics based on parameters
    latency = 320 + (top_k * 18) + (max_hops * 65)
    tokens = 1450 + (len(evidence_units) * 380)

    return {
        "is_demo": True,
        "answer": answer,
        "metrics": {
            "citation_precision": "100%" if enable_verification else "87.5%",
            "evidence_coverage": "94.2%",
            "latency_ms": latency,
            "tokens_used": tokens,
            "cost_inr": round(tokens * 0.0008, 4),
        },
        "agent_trace": [
            {
                "step": 1,
                "tool": "query_analysis",
                "input": f"Query: {clean_query}",
                "status": "Success",
                "hits": 1,
            },
            {
                "step": 2,
                "tool": "hybrid_retrieval",
                "input": f"Top-K: {top_k}, Rerank: {enable_reranker}",
                "status": "Success",
                "hits": min(top_k, 5),
            },
            {
                "step": 3,
                "tool": "graph_context_expansion",
                "input": f"Max hops: {max_hops}",
                "status": "Success",
                "hits": len(graph_edges),
            },
            {
                "step": 4,
                "tool": "citation_verification",
                "input": f"Verification active: {enable_verification}",
                "status": "Verified" if enable_verification else "Skipped",
                "hits": len(evidence_units),
            },
        ],
        "evidence_units": evidence_units,
        "graph_edges": graph_edges,
        "configuration": {
            "repository": repo_name,
            "baseline_mode": baseline_mode,
            "top_k": top_k,
            "max_hops": max_hops,
            "reranker_enabled": enable_reranker,
            "verification_enabled": enable_verification,
            "max_agent_steps": max_agent_steps,
        },
    }
