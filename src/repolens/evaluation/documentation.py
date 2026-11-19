"""Documentation of Baseline and Ablation Experimental Configurations.

Documents the architectural mechanisms, comparative trade-offs, theoretical
foundations, and empirical limitations of each baseline (B0 to B5) and ablation
variant in RepoLens.
"""

from __future__ import annotations

from typing import Any

BASELINE_SPECIFICATIONS: dict[str, dict[str, Any]] = {
    "B0: Naive RAG": {
        "name": "Naive RAG",
        "code": "B0",
        "chunking": "Fixed-size character windows (500 chars, 50 char overlap)",
        "indexing": "Dense text embeddings over raw string chunks",
        "retrieval": "Dense vector cosine similarity top-k",
        "graph_expansion": False,
        "agentic_reasoning": False,
        "citation_verification": False,
        "strengths": [
            "Extremely fast to index (no AST parsing required).",
            "Universal across any programming or natural language file.",
            "Minimal compute requirements during indexing.",
        ],
        "limitations": [
            "Syntactic boundaries broken: functions, classes, and loops are arbitrarily sliced in half.",
            "Detached context: docstrings and comments are severed from the signatures they describe.",
            "Severe citation hallucination: line ranges refer to arbitrary chunk slices rather than valid AST symbols.",
            "No structural awareness: cannot discover callers, callees, or module hierarchy across file boundaries.",
        ],
    },
    "B1: AST RAG": {
        "name": "Structure-Aware AST RAG",
        "code": "B1",
        "chunking": "AST-based CodeUnits (functions, classes, methods, modules)",
        "indexing": "Dense text embeddings over structured CodeUnit representations",
        "retrieval": "Dense vector cosine similarity top-k",
        "graph_expansion": False,
        "agentic_reasoning": False,
        "citation_verification": False,
        "strengths": [
            "Preserves complete grammatical units and valid syntactic blocks.",
            "Retains metadata (qualified name, signature, line numbers, docstring).",
            "Significantly improves citation accuracy over naive sliding windows.",
        ],
        "limitations": [
            "Dense retrieval semantic gap: poor at exact keyword matching (variable names, snake_case identifiers, error strings).",
            "Limited to single-hop context: cannot explore related classes or caller/callee chains.",
            "No reranking or lexical fusion.",
        ],
    },
    "B2: Hybrid RAG": {
        "name": "Hybrid Lexical-Semantic RAG with Reranking",
        "code": "B2",
        "chunking": "AST-based CodeUnits",
        "indexing": "Dual inverted index (BM25) and dense vector embeddings",
        "retrieval": "Reciprocal Rank Fusion (RRF) combining BM25 + dense semantic search, followed by cross-encoder reranking",
        "graph_expansion": False,
        "agentic_reasoning": False,
        "citation_verification": False,
        "strengths": [
            "Combines keyword precision (BM25) with semantic generalization (embeddings).",
            "Cross-encoder evaluates deep joint query-code interaction, filtering out spurious lexical matches.",
            "High top-1 and top-3 retrieval precision for direct questions.",
        ],
        "limitations": [
            "Isolated CodeUnit view: cannot follow imports or call chains beyond the retrieved snippets.",
            "Unable to answer multi-hop repository architectural questions.",
        ],
    },
    "B3: Graph RAG": {
        "name": "Graph-Augmented RAG",
        "code": "B3",
        "chunking": "AST-based CodeUnits + CodeGraph construction",
        "indexing": "Hybrid index + NetworkX directed multigraph",
        "retrieval": "Seed hybrid retrieval followed by deterministic CodeGraph neighbor expansion (CALLS, IMPORTS, CONTAINS)",
        "graph_expansion": True,
        "agentic_reasoning": False,
        "citation_verification": False,
        "strengths": [
            "Automatically surfaces dependencies and caller/callee definitions across module boundaries.",
            "Significantly increases evidence coverage for multi-component questions.",
            "Eliminates blind spots in modular architectures.",
        ],
        "limitations": [
            "Indiscriminate neighbor expansion: may pull irrelevant connected nodes into the prompt context.",
            "Context inflation: increases token usage without goal-directed filtering.",
            "No iterative hypothesis testing or agent routing.",
        ],
    },
    "B4: Agentic RAG": {
        "name": "Agentic Tool-Based RAG",
        "code": "B4",
        "chunking": "AST CodeUnits + CodeGraph + Investigation Tools",
        "indexing": "Hybrid index + CodeGraph + Tool execution environment",
        "retrieval": "Iterative multi-step LangGraph reasoning loop with dynamic tool selection",
        "graph_expansion": True,
        "agentic_reasoning": True,
        "citation_verification": False,
        "strengths": [
            "Goal-directed multi-hop investigation: the agent dynamically inspects code, explores callers, and refines hypotheses.",
            "High recall for complex, exploratory, or diagnostic questions.",
            "Adaptive stopping criteria based on evidence sufficiency.",
        ],
        "limitations": [
            "Higher latency and LLM token consumption due to multi-iteration agent loops.",
            "Citation hallucination risk: the agent can still generate plausible-looking line numbers that do not match disk ground truth.",
        ],
    },
    "B5: Full RepoLens": {
        "name": "Full RepoLens (Grounded Repository Intelligence)",
        "code": "B5",
        "chunking": "AST CodeUnits + CodeGraph + Investigation Tools",
        "indexing": "Unified Hybrid + CodeGraph + Ground-Truth File Registry",
        "retrieval": "LangGraph multi-step investigation + Citation & Claim Ground-Truth Verification",
        "graph_expansion": True,
        "agentic_reasoning": True,
        "citation_verification": True,
        "strengths": [
            "Zero hallucinated source references: all line numbers and symbols are verified against actual repository files on disk.",
            "Provides mathematical guarantee that cited files exist and line ranges contain the claimed symbols.",
            "Highest end-to-end precision and evidence reliability.",
        ],
        "limitations": [
            "Requires read access to repository filesystem for ground-truth checks.",
            "Slight verification latency overhead compared to unverified agentic generation.",
        ],
    },
}


def get_baseline_limitations(baseline_name: str) -> list[str]:
    """Retrieve documented limitations for a specific baseline."""
    for key, spec in BASELINE_SPECIFICATIONS.items():
        if (
            key.lower() in baseline_name.lower()
            or spec["code"].lower() in baseline_name.lower()
        ):
            return spec["limitations"]
    return [
        "Experimental configuration limitations not individually cataloged.",
        "Ensure consistent dataset snapshot conditions when evaluating.",
    ]


def generate_ablation_documentation() -> str:
    """Generate complete markdown documentation for all baselines and ablations."""
    lines = [
        "# RepoLens Baseline & Ablation Experimental Architecture",
        "",
        "## 1. Overview",
        "",
        "RepoLens defines six standardized experimental pipeline configurations (B0 to B5) "
        "to evaluate the progressive contribution of AST structural parsing, hybrid retrieval, "
        "graph context expansion, agentic reasoning, and ground-truth citation verification.",
        "",
        "| Baseline | Architecture | Chunking | Retrieval Mode | Graph | Agent | Verification |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: |",
        "| **B0** | Naive RAG | Fixed 500-char | Dense only | ✗ | ✗ | ✗ |",
        "| **B1** | AST RAG | CodeUnits | Dense only | ✗ | ✗ | ✗ |",
        "| **B2** | Hybrid RAG | CodeUnits | BM25 + Dense + Rerank | ✗ | ✗ | ✗ |",
        "| **B3** | Graph RAG | CodeUnits | Hybrid + CodeGraph Expansion | ✓ | ✗ | ✗ |",
        "| **B4** | Agentic RAG | CodeUnits | Multi-step LangGraph tools | ✓ | ✓ | ✗ |",
        "| **B5** | Full RepoLens | CodeUnits | Hybrid + Graph + Agent + Verified | ✓ | ✓ | ✓ |",
        "",
        "## 2. Baseline Specifications & Limitations",
        "",
    ]

    for key, spec in BASELINE_SPECIFICATIONS.items():
        lines.extend(
            [
                f"### {key} — {spec['name']}",
                f"- **Chunking Strategy:** {spec['chunking']}",
                f"- **Indexing & Retrieval:** {spec['retrieval']}",
                f"- **CodeGraph Expansion:** {'Enabled' if spec['graph_expansion'] else 'Disabled'}",
                f"- **Agent Multi-Step Reasoning:** {'Enabled' if spec['agentic_reasoning'] else 'Disabled'}",
                f"- **Citation Verification:** {'Strict Ground Truth' if spec['citation_verification'] else 'Disabled'}",
                "",
                "**Key Strengths:**",
            ]
        )
        for s in spec["strengths"]:
            lines.append(f"- {s}")
        lines.append("")
        lines.append("**Inherent Limitations:**")
        for lim in spec["limitations"]:
            lines.append(f"- {lim}")
        lines.append("")

    lines.extend(
        [
            "## 3. Component Ablation Hypotheses",
            "",
            "1. **AST Chunking Contribution ($B_1 - B_0$):** Eliminates boundary splitting and preserves grammatical blocks, raising citation validity.",
            "2. **Lexical Fusion Contribution ($B_2 - B_1$):** BM25 bridges the semantic gap for exact symbol identifiers and variable names.",
            "3. **Cross-Encoder Contribution ($B_2 - B_{2\\text{no-rerank}}$):** Reorders candidates so top-k exhibits higher precision.",
            "4. **Graph Expansion Contribution ($B_3 - B_2$):** Traverses cross-module dependencies, increasing evidence coverage.",
            "5. **Agentic Reasoning Contribution ($B_4 - B_3$):** Enables multi-iteration hypothesis testing for exploratory queries.",
            "6. **Citation Verification Contribution ($B_5 - B_4$):** Enforces 100% verifiable citations, eliminating hallucinations.",
        ]
    )

    return "\n".join(lines)
