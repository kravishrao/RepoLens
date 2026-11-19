# Baseline and Ablation Experiments Documentation

## 1. Overview and Motivation

Repository-level code question answering and code intelligence require navigating complex, multi-file software projects with deep syntactic and architectural dependencies. Conventional Retrieval-Augmented Generation (RAG) approaches frequently struggle on codebases because:
1. **Arbitrary Text Chunking:** Slicing source files into fixed character windows breaks syntactic structures (function signatures, class hierarchies, indentation).
2. **Dense Semantic Mismatch:** Neural embeddings alone struggle with exact identifier names, error strings, and API contracts.
3. **Isolated Scope:** Standard retrieval cannot discover related caller/callee or import dependencies located in other modules.
4. **Hallucinated Citations:** Language models generate unverified file paths and line ranges that do not exist or do not contain the claimed symbols.

To evaluate RepoLens's architecture and measure the marginal contribution of each architectural component, RepoLens provides six progressive baseline configurations (**B0** through **B5**) and a systematic component ablation framework.

---

## 2. Experimental Baseline Configurations (B0 to B5)

### Summary Comparison Matrix

| Baseline | Name | Chunking Unit | Retrieval Mechanism | CodeGraph | Agent Loop | Citation Verification |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **B0** | Naive RAG | Fixed-size windows (500 chars, 50 overlap) | Dense embedding cosine similarity | ✗ | ✗ | ✗ |
| **B1** | AST RAG | AST CodeUnits (functions, classes, methods) | Dense embedding cosine similarity | ✗ | ✗ | ✗ |
| **B2** | Hybrid RAG | AST CodeUnits | BM25 + Dense Semantic (RRF) + Cross-Encoder | ✗ | ✗ | ✗ |
| **B3** | Graph RAG | AST CodeUnits | Seed Hybrid + CodeGraph Traversal (1-2 hops) | ✓ | ✗ | ✗ |
| **B4** | Agentic RAG | AST CodeUnits | LangGraph multi-step iterative tools | ✓ | ✓ | ✗ |
| **B5** | Full RepoLens | AST CodeUnits | Hybrid + Graph + Agent + Strict Verification | ✓ | ✓ | ✓ |

---

### Detailed Baseline Specifications

#### B0: Naive RAG
- **Description:** Traditional baseline splitting raw Python files into fixed-size sliding character windows (default 500 characters, 50 characters overlap) without AST parsing or symbol knowledge.
- **Indexing & Retrieval:** Chunks are embedded and indexed in an in-memory vector store. The top-$k$ chunks with the highest cosine similarity are retrieved.
- **Synthesis:** An answer is synthesized directly from the retrieved raw chunks.
- **Inherent Limitations:**
  - Functions, loops, and classes are arbitrarily cut across chunk boundaries.
  - Docstrings are detached from function signatures.
  - Sourced citations refer to arbitrary chunk line ranges rather than logical language constructs, leading to high citation failure rates.

#### B1: Structure-Aware AST RAG
- **Description:** Replaces arbitrary text slicing with AST-based `CodeUnit` extraction. Each chunk represents a discrete function, class, method, or module.
- **Indexing & Retrieval:** CodeUnit metadata, docstrings, imports, and source code are embedded into dense vectors.
- **Synthesis:** Synthesizes structured grounded answers from the top-$k$ retrieved CodeUnits.
- **Inherent Limitations:**
  - Dense retrieval struggles with exact lexical symbol matches (e.g., `_validate_token_v2`, `HTTP_404_NOT_FOUND`).
  - Lacks cross-module graph expansion; cannot trace caller/callee relationships.

#### B2: Hybrid RAG
- **Description:** Combines lexical search (BM25) and dense semantic search via Reciprocal Rank Fusion (RRF), followed by Cross-Encoder reranking.
- **Indexing & Retrieval:** Maintains dual indexes: inverted BM25 index and vector embeddings. Cross-encoder scores joint query-candidate pairs to rerank the top candidates.
- **Synthesis:** Synthesizes answer from the top reranked CodeUnits.
- **Inherent Limitations:**
  - Operates in isolation: cannot expand context across import statements or invocation trees to other files.

#### B3: Graph RAG
- **Description:** Augments hybrid retrieval with static code relationship traversal over the `CodeGraph`.
- **Indexing & Retrieval:** Retrieves seed CodeUnits via hybrid search, then expands the context along directed multigraph edges (`CALLS`, `IMPORTS`, `CONTAINS`) up to $N$ hops (default 2 hops).
- **Synthesis:** Synthesizes answer using both seed and structural neighbor CodeUnits.
- **Inherent Limitations:**
  - Indiscriminate expansion can pull in irrelevant connected nodes, inflating context token usage without goal-directed filtering.
  - Lacks iterative query reformulation.

#### B4: Agentic RAG
- **Description:** Replaces single-pass retrieval with an iterative LangGraph stateful agent. The agent analyzes the query, selects tools (`keyword_search`, `graph_neighbors`, `read_code_unit`), gathers evidence, and decides whether further investigation is necessary.
- **Indexing & Retrieval:** Dynamic multi-step tool execution.
- **Synthesis:** Final answer preparation combines multi-iteration accumulated evidence.
- **Inherent Limitations:**
  - Higher latency and token consumption from multiple agent reasoning steps.
  - Citation verification is disabled: citations in the generated answer are unverified and may contain hallucinations.

#### B5: Full RepoLens
- **Description:** Complete production pipeline coupling AST CodeUnits, hybrid retrieval, cross-encoder reranking, CodeGraph context expansion, LangGraph multi-step agent investigation, and ground-truth citation verification.
- **Verification Mechanism:** Every citation generated in the final answer is parsed and verified by `CitationValidator` against actual repository disk files:
  - Validates file existence on disk.
  - Validates line ranges within file bounds.
  - Validates symbol existence within the specified line range.
  - Validates CodeUnit alignment in the repository graph.
- **Inherent Limitations:**
  - Requires read access to repository source files on disk.
  - Slight verification latency overhead compared to unverified agent generation.

---

## 3. Component Ablation Methodology

The ablation suite systematically isolates individual components to measure their marginal impact on retrieval and verification metrics:

$$\Delta M = M(\text{Pipeline with Component}) - M(\text{Pipeline without Component})$$

### Evaluated Ablations

1. **AST Chunking Contribution ($B_1 - B_0$):**
   - Isolates the effect of structure-aware CodeUnit parsing vs. naive text window chunking.
   - Primary metric: Reduction in fragmented definitions and increase in citation validity.
2. **BM25 Lexical Fusion Contribution ($B_2 - B_1$):**
   - Isolates the impact of BM25 keyword matching alongside dense embeddings.
   - Primary metric: Hit rate for exact symbol identifiers and function names.
3. **Cross-Encoder Reranking Contribution ($B_2 - B_{2\text{-NoRerank}}$):**
   - Isolates cross-encoder scoring of top hybrid candidates.
   - Primary metric: Top-1 and top-3 retrieval precision.
4. **CodeGraph Context Expansion Contribution ($B_3 - B_2$):**
   - Isolates graph traversal along `CALLS`, `IMPORTS`, and `CONTAINS` edges.
   - Primary metric: Evidence coverage for multi-component architectural queries.
5. **Agentic Reasoning Contribution ($B_4 - B_3$):**
   - Isolates multi-step iterative tool routing against single-shot graph expansion.
   - Primary metric: Recall on multi-hop diagnostic questions.
6. **Ground-Truth Citation Verification Contribution ($B_5 - B_4$):**
   - Isolates source-level citation and claim validation.
   - Primary metric: 100% elimination of hallucinated line references and phantom file citations.

---

## 4. Reproducibility and Consistent Conditions

To ensure scientific reproducibility across evaluations:
- **Consistent Datasets:** Snapshots of five open-source repositories (`requests`, `flask`, `click`, `fastapi`, `pydantic`) pinned to specific commit hashes in `dataset/metadata/repositories.json`.
- **Predefined Ground-Truth Queries:** Curated queries in `repolens.evaluation.benchmark_suite` specifying expected target symbols, target files, and query categories.
- **Reproducible Artifacts:** Every execution of `ExperimentRunner` logs:
  - System provenance (OS, Python version, git commit hash, PID).
  - Exact hyperparameter configuration (`ExperimentConfig`).
  - Per-query latency, tokens, hit rate, and validation status.
  - Serialized JSON format exportable via `save_run_result()`.

---

## 5. Programmatic Usage Example

```python
from repolens.evaluation import (
    ExperimentRunner,
    ExperimentConfig,
    BaselineType,
    prepare_benchmark_context,
    get_benchmark_queries,
)

# 1. Prepare benchmark repository context
context = prepare_benchmark_context("RepoLens")

# 2. Configure baseline experiment (e.g. B5 Full RepoLens)
config = ExperimentConfig(
    baseline=BaselineType.B5_FULL_REPOLENS,
    repository_name="RepoLens",
    top_k=5,
    max_hops=2,
    enable_verification=True,
)

# 3. Execute experiment
runner = ExperimentRunner(context=context)
result = runner.run_experiment(config)

# 4. Inspect reproducible results
print(result.to_markdown_summary())
runner.save_run_result(result)

# 5. Run full comparative ablation suite across B0-B5
comparison = runner.run_all_baselines(repository_name="RepoLens")
print(comparison.to_markdown_table())
```
