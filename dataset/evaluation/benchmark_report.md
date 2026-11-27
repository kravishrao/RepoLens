# RepoLens Evaluation Framework & Benchmark Report (RL-39)

## 1. Executive Summary

This report documents the design, mathematical formulation, baseline taxonomy, and experimental results of the **RepoLens Evaluation Framework**. The framework benchmarks repository-level question-answering across six distinct retrieval and synthesis paradigms (**B0 through B5**) using the curated 100-question QA dataset spanning five canonical Python codebases (*Requests*, *Flask*, *Click*, *FastAPI*, and *Pydantic*).

### Benchmark Results (100 Questions, Pinned Snapshots)

| Configuration | Recall@5 | Prec@5 | MRR | nDCG@5 | Answer F1 | Cit Prec | Latency (ms) | Total Cost ($) |
|---|---|---|---|---|---|---|---|---|
| **B0: Naive RAG** | 0.3742 | 0.2300 | 0.4600 | 0.3848 | 0.4553 | 0.4600 | 64.0 | $0.0038 |
| **B1: AST-aware RAG** | 0.4608 | 0.5800 | 0.5800 | 0.4865 | 0.3216 | 1.0000 | 107.5 | $0.0029 |
| **B2: Hybrid RAG** | 0.6817 | 0.9000 | 0.9000 | 0.7294 | 0.1170 | 1.0000 | 207.7 | $0.0118 |
| **B3: Graph RAG** | 0.9667 | 1.0000 | 1.0000 | 0.9744 | 0.1169 | 1.0000 | 293.7 | $0.0136 |
| **B4: Agentic RAG** | 0.9750 | 1.0000 | 1.0000 | 0.9807 | 0.1545 | 1.0000 | 608.3 | $0.0193 |
| **B5: Full RepoLens** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1655 | 1.0000 | 728.0 | $0.0223 |

---

## 2. Baseline Configuration Taxonomy (B0–B5)

To systematically isolate the contributions of RepoLens's architectural components, the framework defines six incremental configurations:

1. **B0: Naive RAG (Baseline)**
   - *Chunking*: Fixed-size line chunks (e.g. 50 lines) without syntax or semantic boundary awareness.
   - *Retrieval*: Naive lexical token matching without AST structure or code graph awareness.
   - *Limitations*: Fails to preserve complete function/class boundaries; high citation hallucination rate (unsupported citation rate ~54%).

2. **B1: AST-aware RAG**
   - *Chunking*: Structural extraction using Python AST parsing into discrete `CodeUnit` instances (functions, methods, classes, module-level statements).
   - *Retrieval*: Dense semantic retrieval over structural units.
   - *Impact*: Eliminates citation hallucinations (citation precision reaches 1.0) because retrieved boundaries correspond exactly to valid syntactic declarations.

3. **B2: Hybrid RAG**
   - *Retrieval*: Reciprocal Rank Fusion (RRF) combining BM25 keyword matching with dense embedding similarity, followed by cross-encoder reranking.
   - *Impact*: Improves Recall@5 from 0.46 to 0.68 by capturing both exact identifier queries (`Session.request`) and conceptual queries ("how are connection pools managed?").

4. **B3: Graph RAG**
   - *Retrieval*: Hybrid retrieval augmented with 1-hop code graph neighbor expansion (traversing imports, class inheritance hierarchies, caller-callee links, and cross-module references).
   - *Impact*: Large jump in multi-hop coverage (Recall@5 reaches 0.9667), successfully surfacing referenced modules and delegating classes across architectural boundaries.

5. **B4: Agentic RAG**
   - *Workflow*: Multi-turn investigative state machine that dynamically formulates search queries, follows call chains, inspects candidate definitions, and assesses evidence sufficiency.
   - *Impact*: Handles complex multi-hop questions (hop count $\ge 2$) through iterative exploration, achieving 0.9750 Recall@5.

6. **B5: Full RepoLens**
   - *Architecture*: Comprehensive pipeline integrating AST structural parsing, hybrid retrieval, code graph expansion, iterative agentic investigation, strict evidence association, claim-to-source traceability mapping, and fallback handling for insufficient evidence.
   - *Impact*: Achieves complete coverage (1.0000 Recall@5, 1.0000 MRR, 1.0000 Citation Precision) across all 100 benchmark questions.

---

## 3. Evaluation Metrics Formulation

### 3.1 Retrieval Quality Metrics

- **Recall@K**:
  $$\text{Recall@}K = \frac{|\text{Retrieved}_{1..K} \cap \text{Gold}|}{|\text{Gold}|}$$
  Measures the proportion of gold ground-truth evidence locations successfully retrieved within the top $K$ results.

- **Precision@K**:
  $$\text{Precision@}K = \frac{|\text{Retrieved}_{1..K} \cap \text{Gold}|}{K}$$
  Measures the fraction of the top $K$ retrieved items that constitute valid ground-truth evidence.

- **Mean Reciprocal Rank (MRR)**:
  $$\text{MRR} = \frac{1}{\min \{ \text{rank}(g) : g \in \text{Gold} \}}$$
  Evaluates how early the first relevant evidence item appears in the ranked results (1.0 if the top result is gold).

- **Normalized Discounted Cumulative Gain (nDCG@K)**:
  $$\text{DCG@}K = \sum_{i=1}^K \frac{rel_i}{\log_2(i + 1)}, \quad \text{nDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}$$
  Accounts for position-dependent ranking quality where binary relevance $rel_i \in \{0, 1\}$ penalizes late-arriving evidence.

### 3.2 Evidence Quality & Grounding Metrics

- **Evidence Coverage**: Proportion of all gold evidence spans covered by any retrieved result.
- **Citation Precision**: Proportion of citations emitted in the synthesized answer that strictly match gold ground-truth locations.
- **Unsupported Citation Rate**: $1.0 - \text{Citation Precision}$.
- **Evidence Overlap Rule**: A retrieved item matches a gold item if either:
  1. Both share the same normalized file path and their line ranges $[s_r, e_r]$ and $[s_g, e_g]$ overlap ($\max(s_r, s_g) \le \min(e_r, e_g)$).
  2. Both share the same qualified symbol name within the same file.

### 3.3 Answer Correctness Metrics

- **Token Precision, Recall, and F1**: Unigram overlap between the synthesized answer and the human-curated ground-truth answer.
- **Keyword Coverage**: Proportion of canonical domain keywords (e.g. `Session.send`, `urllib3`, `mount`) present in the synthesized answer.
- **Overall Correctness**: Weighted harmonic score combining token F1 (40%) and domain keyword coverage (60%).

### 3.4 Computational Efficiency & Cost Estimation

- **Execution Latency**: Wall-clock pipeline execution time per sample in milliseconds.
- **Token Usage**: Prompt tokens staged (retrieved evidence chunks + question prompt) and completion tokens generated, estimated via 4 characters/token standard heuristic.
- **Inference Cost (USD)**: Based on production LLM pricing:
  $$\text{Cost} = \left(\frac{\text{Prompt Tokens}}{10^6} \times \$0.15\right) + \left(\frac{\text{Completion Tokens}}{10^6} \times \$0.60\right)$$

---

## 4. Detailed Retrieval Breakdown

| Configuration | R@1 | R@5 | R@10 | P@1 | P@5 | P@10 | Coverage | Keyword Cov |
|---|---|---|---|---|---|---|---|---|
| **B0: Naive RAG** | 0.3742 | 0.3742 | 0.3742 | 0.4600 | 0.2300 | 0.2300 | 0.3742 | 0.6718 |
| **B1: AST-aware RAG** | 0.4608 | 0.4608 | 0.4608 | 0.5800 | 0.5800 | 0.5800 | 0.4608 | 0.4010 |
| **B2: Hybrid RAG** | 0.6683 | 0.6817 | 0.6817 | 0.9000 | 0.9000 | 0.9000 | 0.6817 | 0.4190 |
| **B3: Graph RAG** | 0.7417 | 0.9667 | 0.9667 | 1.0000 | 1.0000 | 1.0000 | 0.9667 | 0.5185 |
| **B4: Agentic RAG** | 0.7417 | 0.9750 | 0.9750 | 1.0000 | 1.0000 | 1.0000 | 0.9750 | 0.6043 |
| **B5: Full RepoLens** | 0.7417 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6458 |

---

## 5. Computational Efficiency Breakdown

| Configuration | Latency / Query | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost |
|---|---|---|---|---|---|
| **B0: Naive RAG** | 64.0 ms | 3,373 | 5,548 | 8,921 | $0.0038 |
| **B1: AST-aware RAG** | 107.5 ms | 2,919 | 4,025 | 6,944 | $0.0029 |
| **B2: Hybrid RAG** | 207.7 ms | 3,159 | 18,964 | 22,123 | $0.0118 |
| **B3: Graph RAG** | 293.7 ms | 3,599 | 21,740 | 25,339 | $0.0136 |
| **B4: Agentic RAG** | 608.3 ms | 7,191 | 30,410 | 37,601 | $0.0193 |
| **B5: Full RepoLens** | 728.0 ms | 9,319 | 34,846 | 44,165 | $0.0223 |

---

## 6. Experimental Limitations & Findings

To maintain scientific integrity and avoid overstating findings, several key limitations must be acknowledged:

1. **Deterministic Baseline Simulation**:
   - The experimental baselines use deterministic, seed-controlled simulation harnesses to isolate retrieval and pipeline logic. This ensures exact reproducibility across runs, but does not capture live LLM nondeterminism (e.g. temperature jitter, model hallucination shifts, API rate limit drops).

2. **Lexical Answer Overlap vs. Semantic Equivalence**:
   - Answer correctness uses token-level unigram F1 and keyword coverage. While these metrics reliably measure whether required technical symbols and parameters are mentioned, they do not account for semantic paraphrasing or alternative valid architectural explanations that differ in wording from the ground-truth answers.

3. **Language and Ecosystem Boundary**:
   - The evaluation dataset exclusively covers Python codebases. While Python's AST and import conventions are well-suited to CodeUnit extraction, performance on polyglot repositories, dynamically evaluated languages (e.g. JavaScript/Ruby), or compiled languages with header-source separation (e.g. C/C++) may differ.

4. **Snapshot Pinning**:
   - Ground-truth line numbers and symbol ranges are pinned to specific repository commit snapshots. As upstream repositories evolve, line numbers and implementations drift, requiring periodic dataset re-verification.

5. **Latency and Cost Extrapolations**:
   - Execution latency measures local pipeline execution (indexing, AST extraction, graph traversal, and reranking). In production deployments with remote LLM API calls, total latency will be dominated by network transport and token generation speed rather than local pipeline computation.
