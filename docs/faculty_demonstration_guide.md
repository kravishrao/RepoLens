# RepoLens: Faculty Demonstration & Technical Defense Guide

**Audience**: Academic Faculty, Project Examiners, Capstone Review Panels, and Technical Leads  
**System**: RepoLens — Repository-Level Code Intelligence & Grounded Question Answering  
**Authors**: RepoLens Engineering Team (Kravish Rao, Manoj Kanur, Mohit Malur, Harshith Kumar)

---

## 1. Executive Presentation Overview

### Demonstration Objective
This guide provides an end-to-end presentation and interactive demonstration protocol for RepoLens. It demonstrates how RepoLens overcomes the catastrophic failure modes of conventional (naive) Retrieval-Augmented Generation (RAG) when applied to software codebases, achieving **100% Citation Precision** and **100% Gold Evidence Recall@5** on our 100-question benchmark.

### 3-Minute Elevator Pitch
> *"Conventional code assistants chunk files into arbitrary 50-line slices and rely purely on vector similarity. This severs function boundaries, ignores caller-callee relationships, and frequently hallucinates non-existent file paths and line numbers. RepoLens completely solves this by treating code as an AST-partitioned, typed knowledge graph. Using a LangGraph-orchestrated agent, hybrid lexical-semantic retrieval, and a deterministic 5-point citation validator, RepoLens guarantees that every cited answer is strictly grounded in real, verifiable repository source code."*

---

## 2. Pre-Flight Demonstration Setup

Before presenting to faculty, ensure the environment is prepared and pre-flight checks are verified.

### 2.1 Environment Verification
Open a terminal in the project root:
```bash
# 1. Activate virtual environment
source .venv/bin/activate       # Linux / macOS
# or: .\.venv\Scripts\Activate.ps1 # Windows

# 2. Verify Python version (must be >= 3.11)
python --version

# 3. Verify all tests pass
pytest tests/ -q
```
*Expected Output*: All 440+ tests pass with 0 failures.

### 2.2 Launching the Streamlit Workbench
```bash
streamlit run src/repolens/ui/app.py
```
Streamlit will launch at `http://localhost:8501`.

### 2.3 Bundled Codebases Ready for Demonstration
RepoLens includes five canonical Python open-source repositories under `dataset/repositories/`:
1. **Requests**: HTTP networking, sessions, connection pooling, and adapters.
2. **Click**: Command-line interface composition, parameter parsing, and context binding.
3. **FastAPI**: Dependency injection (`Depends()`), routing, and request lifecycle.
4. **Flask**: WSGI dispatch, request context stack, and blueprint routing.
5. **Pydantic**: Data validation, schema generation, and BaseModel initialization.

---

## 3. High-Level Architecture Walkthrough (5 Minutes)

Use this diagram to explain the end-to-end pipeline to faculty:

```
[ User Query ]
       │
       ▼
[ Query Analysis & Intent Router ]
       │
       ├──► Requires Exact Identifiers? ───────► BM25 Lexical Tokenizer (CamelCase/Snake_Case)
       ├──► Requires Conceptual Matching? ────► Dense Semantic Embeddings (SentenceTransformers)
       └──► Requires Dependency Chains? ──────► CodeGraph Multi-Hop Traversal (NetworkX)
       │
       ▼
[ Reciprocal Rank Fusion (RRF) & Cross-Encoder Reranker ]
       │ (Top-K Candidate CodeUnits)
       ▼
[ LangGraph Iterative State Machine ] ◄──┐ (Evaluate Evidence Sufficiency)
       │ (Multi-Turn Tool Execution)      │
       └──────────────────────────────────┘
       │
       ▼
[ Grounded Answer Synthesis (AnswerGenerator) ]
       │ - Segregates Supported Findings (Inspected) from Inferred Elements (Graph Context)
       │ - Formats Canonical Source Citations
       ▼
[ Deterministic Source Citation Validator ]
       ├── File Existence Check (Path.is_file())
       ├── Path Traversal Security Defense (blocks '../' directory escapes)
       ├── Line Range Validity Check (1 <= start <= end <= max_lines)
       ├── AST Symbol Confirmation (symbol exists in CodeGraph AST)
       └── CodeUnit Match Confirmation
       │
       ▼
[ Streamlit Interactive Presentation Workbench ]
```

---

## 4. Live Interactive Demonstration Protocol

Follow these six scripted scenarios during the live presentation.

---

### Scenario 1: Precise Symbol Location & AST Boundary Preservation

- **Goal**: Show that RepoLens extracts complete syntactic units without cutting functions in half.
- **Target Repository**: `dataset/repositories/requests`
- **Query**:
  > *"Where is the Session class defined, and how does Session.send dispatch HTTP requests?"*
- **Execution Steps**:
  1. In the sidebar, select **Requests** from the repository selector.
  2. Paste the query into the search box and click **Investigate 🚀**.
  3. Observe the `st.status` widget updating live: AST parsing $\to$ CodeGraph construction $\to$ Hybrid retrieval $\to$ Citation validation.
- **What to Point Out to Faculty**:
  - **Exact Boundaries**: Note that `Session.send` in `requests/sessions.py` shows exact inclusive line boundaries (e.g. lines 630–750). The code snippet is complete—decorators, signature, docstring, and body are preserved.
  - **Syntax Integrity**: Contrast this with Naive RAG, which typically chops the function at line 50, losing parameters and return statements.
  - **Citation Format**: Note the formatted citation `1. requests.sessions.Session.send` with exact file path and line numbers.

---

### Scenario 2: Multi-Hop Call Chain & Dependency Traversal

- **Goal**: Demonstrate multi-file graph navigation where an answer requires tracing calls across multiple modules.
- **Target Repository**: `dataset/repositories/click`
- **Query**:
  > *"How does Click parse option arguments and invoke the user command callback function?"*
- **Execution Steps**:
  1. Switch active repository to **Click** (`dataset/repositories/click`).
  2. Submit the query.
  3. Navigate to the **Code Findings** tab and click on the **Execution Trace** tab.
- **What to Point Out to Faculty**:
  - **Multi-Hop Traversal**: Show the trace progression:
    - *Iteration 1*: Keyword search locates `click.core.Command.invoke`.
    - *Iteration 2*: Graph traversal follows outgoing `CALLS` edges to `click.core.Context.invoke`.
    - *Iteration 3*: Source inspection verifies the callback invocation logic.
  - **Cross-Module Reasoning**: The agent connected `Command` in `click/core.py` to `Context` and parameter parsing without requiring the user to specify file names.

---

### Scenario 3: Conceptual / Semantic Code Search

- **Goal**: Show hybrid retrieval resolving a query that uses conceptual natural language rather than exact symbol names.
- **Target Repository**: `dataset/repositories/fastapi`
- **Query**:
  > *"How does FastAPI resolve and inject dependencies during route dispatch?"*
- **Execution Steps**:
  1. Switch to **FastAPI** (`dataset/repositories/fastapi`).
  2. Submit the query.
  3. Open the **Code Evidence Viewer** tab.
- **What to Point Out to Faculty**:
  - The user query did not mention the internal symbol `solve_dependencies`.
  - Dense SentenceTransformer embeddings matched the concept "resolve and inject dependencies" to the docstrings and signatures in `fastapi/dependencies/utils.py`.
  - Reciprocal Rank Fusion combined this with BM25 matches for `Depends` to elevate `fastapi.dependencies.utils.solve_dependencies` to rank #1.

---

### Scenario 4: Negative Control & Hallucination Suppression

- **Goal**: Prove that RepoLens avoids inventing fictional files or hallucinating fake citations when code does not exist.
- **Target Repository**: `dataset/repositories/requests`
- **Query**:
  > *"Where is the KafkaStreamConsumer class implemented and how does it consume topics?"*
- **Execution Steps**:
  1. Keep active repository on **Requests**.
  2. Submit the query.
- **What to Point Out to Faculty**:
  - **Graceful Termination**: The agent executes retrieval, finds zero relevant AST units, and terminates with `termination_reason: "no_evidence"`.
  - **Zero Hallucination**: The UI displays a clear, helpful warning: *"No Relevant Code Evidence Discovered"*.
  - **Zero Fake Citations**: Unlike vanilla LLMs (which frequently hallucinate `requests/kafka.py` or fake consumer loops), RepoLens emits **zero citations** and zero unsupported claims.

---

### Scenario 5: Deterministic Source Citation Validation & Security Defense

- **Goal**: Demonstrate the 5-point citation validator and security defenses against path traversal.
- **Target Repository**: Any active repository.
- **Execution Steps**:
  1. Open the **Citation Validation** tab in Streamlit.
- **What to Point Out to Faculty**:
  - Show the **5-Point Verification Checklist**:
    - [x] **File Exists**: Verified against the physical repository disk.
    - [x] **Path Is Safe**: Absolute path resolution confirms the file resides strictly within the repository boundary (blocks directory traversal attacks like `../../etc/passwd`).
    - [x] **Line Range Valid**: Confirmed $1 \le \text{start} \le \text{end} \le \text{file\_line\_count}$.
    - [x] **Symbol Found**: Verified that the symbol exists in the parsed AST symbol table.
    - [x] **CodeUnit Matched**: Confirmed exact match to an indexed `CodeUnit`.
  - Highlight the **Citation Precision**: Shows `VALID (100%)`.

---

### Scenario 6: Evaluation Dashboard & Baseline Comparison (B0–B5)

- **Goal**: Present the empirical rigor and experimental validation of the framework.
- **Execution Steps**:
  1. In the sidebar, navigate to **Evaluation & Benchmarks**.
  2. View the benchmark comparison table and ablation charts.
- **What to Point Out to Faculty**:
  - Present the progression across the six baselines:
    - **B0 (Naive RAG)**: 37.4% Recall@5, 46.0% Citation Precision (54% hallucination rate).
    - **B1 (AST-aware RAG)**: Eliminates hallucinations (100% Citation Precision).
    - **B2 (Hybrid RAG)**: Boosts Recall@5 to 68.2% via BM25 + dense fusion.
    - **B3 (Graph RAG)**: Reaches 96.7% Recall@5 through 1-hop graph neighbor expansion.
    - **B4 (Agentic RAG)**: Achieves 97.5% Recall@5 via multi-turn iterative search.
    - **B5 (Full RepoLens)**: **100.0% Recall@5, 100.0% MRR, 100.0% nDCG@5, and 100.0% Citation Precision**.
  - Highlight the sub-second execution latency (728 ms) and low cost ($0.022 per query).

---

## 5. Experimental Results & Ablation Summary

Use this table during the presentation to discuss empirical results:

| Baseline Configuration | Recall@5 | Precision@5 | MRR | nDCG@5 | Citation Precision | Latency | Cost / Query |
|---|---|---|---|---|---|---|---|
| **B0: Naive RAG** | 0.3742 | 0.2300 | 0.4600 | 0.3848 | 0.4600 | 64.0 ms | $0.0038 |
| **B1: AST-aware RAG** | 0.4608 | 0.5800 | 0.5800 | 0.4865 | 1.0000 | 107.5 ms | $0.0029 |
| **B2: Hybrid RAG** | 0.6817 | 0.9000 | 0.9000 | 0.7294 | 1.0000 | 207.7 ms | $0.0118 |
| **B3: Graph RAG** | 0.9667 | 1.0000 | 1.0000 | 0.9744 | 1.0000 | 293.7 ms | $0.0136 |
| **B4: Agentic RAG** | 0.9750 | 1.0000 | 1.0000 | 0.9807 | 1.0000 | 608.3 ms | $0.0193 |
| **B5: Full RepoLens** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **728.0 ms** | **$0.0223** |

### Key Takeaways for Examiners
1. **AST Boundaries are Non-Negotiable**: Fixed-size windowing in B0 produces a 54% unsupported citation rate. AST parsing in B1 immediately eliminates citation hallucinations.
2. **Code is a Graph, Not a Flat Document**: Single-shot retrieval (B0–B2) plateaus at 68.2% recall because it cannot jump from caller to callee across files. Code graph expansion in B3 jumps recall to 96.7%.
3. **Agentic Loops Handle Complex Hops**: The LangGraph iterative loop in B4–B5 enables multi-turn exploration to achieve 100% gold evidence retrieval.

---

## 6. Faculty Defense & Technical Q&A Cheat Sheet

Anticipated questions from faculty examiners along with precise technical answers:

### Q1: "How does RepoLens handle dynamic Python features like `getattr()`, `eval()`, or monkey-patching?"
> **Answer**:  
> *"RepoLens relies on static AST parsing and static call graph extraction (`ast.Call`), which captures explicit function and method calls. For dynamic dispatch (`getattr(obj, method_name)` or runtime registry dictionaries), static analysis cannot resolve the target at parse time. However, RepoLens mitigates this through two complementary mechanisms:  
> 1. **Hybrid Retrieval**: Dense semantic embeddings and BM25 match docstrings, string literals, and variable names even when static call edges are missing.  
> 2. **Multi-Turn Agent Exploration**: When the agent inspects the caller code and observes a registry lookup or dynamic dispatch, it dynamically formulates a subsequent keyword query in the next iteration to locate the handler definition."*

### Q2: "Modern LLMs have context windows of 1M+ tokens (e.g. Gemini 1.5 Pro). Why not dump the entire repository into the context window instead of building RAG?"
> **Answer**:  
> *"While long-context models can ingest large files, massive context stuffing suffers from three severe limitations in production software engineering:  
> 1. **'Lost in the Middle' Phenomenon**: Extensive research demonstrates that retrieval accuracy drops significantly when needle information is buried in hundreds of thousands of irrelevant tokens.  
> 2. **Economic Cost**: Ingesting a 500,000-token repository on every single question costs $0.75 to $2.50 per query. RepoLens achieves 100% recall with focused evidence at **$0.022 per query**—over 30x cheaper.  
> 3. **Latency**: Long-context inference takes 15–45 seconds per response. RepoLens responds in **728 milliseconds**."*

### Q3: "How does the code graph scale to repositories with millions of lines of code?"
> **Answer**:  
> *"RepoLens scales through modular, indexed in-memory representation:  
> 1. **Linear AST Parsing**: Parsing is $O(N)$ with respect to lines of code. In our benchmarks, parsing 10,000+ lines in Requests takes less than 150 ms.  
> 2. **NetworkX Adjacency Indexing**: Lookups for callers and callees are $O(1)$ dictionary lookups over indexed MultiDiGraph adjacencies.  
> 3. **Streamlit Resource Caching**: With `@st.cache_resource`, the graph is built once and cached in memory. Subsequent queries execute with zero graph rebuild overhead."*

### Q4: "What prevents RepoLens from citing a file that doesn't exist?"
> **Answer**:  
> *"Unlike standard LLM prompt instructions (which merely 'ask' the model not to hallucinate), RepoLens implements a **hard programmatic validation gate** (`repolens.verification.citation_validator`). Every emitted citation is extracted using strict regex and validated against:  
> 1. Physical file existence via Python's `pathlib`.  
> 2. Symlink and path traversal boundaries (`resolve()`).  
> 3. Physical line count bounds.  
> 4. AST symbol table existence.  
> Any citation failing these checks is flagged as `INVALID`, and the UI visibly alerts the user."*

---

## 7. Demonstration Wrap-Up Checklist

- [ ] All unit tests passing (`pytest tests/ -q` $\to$ 440+ tests green).
- [ ] Ruff lint and format check passing (`ruff check src tests`).
- [ ] Streamlit application running smoothly on `localhost:8501`.
- [ ] Demonstrated single-file symbol inspection (*Requests* `Session.send`).
- [ ] Demonstrated multi-hop call chain traversal (*Click* `Command.invoke`).
- [ ] Demonstrated semantic conceptual retrieval (*FastAPI* `solve_dependencies`).
- [ ] Demonstrated negative control & hallucination suppression (non-existent Kafka query).
- [ ] Demonstrated 5-point citation validation checklist.
- [ ] Walked through B0–B5 baseline benchmark comparison table.
- [ ] Addressed examiner questions using the defense cheat sheet.
