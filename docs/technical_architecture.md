# RepoLens Technical Architecture & Subsystem Specification

This document provides an in-depth technical specification of **RepoLens**, detailing the internal design, algorithms, data structures, and mathematical formulations of all core subsystems.

---

## 1. System Overview & Engineering Philosophy

RepoLens addresses the fundamental limitation of standard vector-based Retrieval-Augmented Generation (RAG) when applied to source code: **syntactic and relational blindness**.

Conventional RAG splits source text into arbitrary character or line windows (e.g. 50 lines). In contrast, RepoLens treats code as a structured, typed, multi-relational knowledge graph. The system guarantees:
1. **Syntactic Completeness**: No code element is severed mid-definition; boundaries correspond strictly to valid AST declarations.
2. **Relational Context**: CodeUnits are connected via directed edges representing caller-callee links, import statements, and module containment hierarchies.
3. **Multi-Hop Traversal**: An iterative LangGraph agent navigates dependencies across architectural boundaries.
4. **Verifiable Citations**: Answers cite exact symbols, file paths, and line spans that are deterministically validated against the underlying codebase before presentation.

---

## 2. Ingestion & File Discovery Subsystem (`repolens.ingestion`)

The ingestion pipeline transforms raw repository directories or Git URLs into structured AST parse trees.

```
[ Local Dir / Git URL ]
          │
          ▼
[ RepositoryLoader ] ──> Validates directory, git clone, branch checkout
          │
          ▼
[ FileDiscovery ]    ──> Scans for *.py, enforces .gitignore, skips caches
          │
          ▼
[ ASTParser ]        ──> ast.parse() with encoding/syntax fallback recovery
          │
          ▼
[ CodeUnit Extraction ] ──> Generates immutable CodeUnit instances
```

### 2.1 Repository Loader (`repository_loader.py`)
- Accepts either a local filesystem directory or a remote Git URL.
- Clones remote repositories to a temporary or cache directory using `git clone --depth 1`.
- Performs security validation: path traversal defense, symlink boundary checks, and existence verification.
- Raises typed exceptions: `InvalidRepositoryError`, `RepositoryCloneError`, and `RepositoryError`.

### 2.2 File Discovery (`file_discovery.py`)
- Recursively traverses repository paths to collect Python source files (`*.py`).
- Filters out non-source directories: `.venv`, `venv`, `__pycache__`, `.git`, `.tox`, `build`, `dist`, `site-packages`, `.mypy_cache`, `.pytest_cache`, and egg-info directories.
- Respects root `.gitignore` exclusion patterns.
- Returns deterministic, lexically sorted relative file paths.

### 2.3 AST Parser & Error Recovery (`ast_parser.py`)
- Uses Python's native `ast` module (`ast.parse()`).
- Implements two-tier error recovery:
  1. **Encoding Recovery**: Attempts UTF-8 decode; falls back to UTF-8 with `errors="replace"` and latin-1 if byte sequences are non-compliant.
  2. **Syntax Error Handling**: Catches `SyntaxError` on malformed Python files (e.g. Python 2 legacy syntax, incomplete code) without crashing the ingestion pipeline. Logs `ASTParseWarning` or partial module units.
- Calculates exact 1-indexed `start_line` and `end_line` for every AST node using `node.lineno` and `node.end_lineno`.

---

## 3. CodeUnit Schema & Representation

The core unit of code representation throughout RepoLens is the immutable `CodeUnit` dataclass (`src/repolens/ingestion/code_unit.py`):

```python
CodeElementType = Literal["module", "class", "function", "method"]

@dataclass(frozen=True)
class ImportReference:
    """An import statement discovered in a source file."""
    module: str

@dataclass(frozen=True)
class CallReference:
    """A statically discovered function or method call."""
    target: str

@dataclass(frozen=True)
class CodeUnit:
    """Standard representation of a code element in a repository."""
    id: str                                      # Unique deterministic identifier
    kind: CodeElementType                        # 'module', 'class', 'function', 'method'
    name: str                                    # Unqualified identifier name
    qualified_name: str                          # Fully qualified dot-path (e.g. pkg.mod.Class.func)
    file_path: str                               # Repository-relative file path
    start_line: int                              # 1-indexed inclusive start line
    end_line: int                                # 1-indexed inclusive end line
    parent_id: str | None = None                 # ID of enclosing class or module
    imports: tuple[ImportReference, ...] = ()    # Discovered static imports
    calls: tuple[CallReference, ...] = ()        # Static function/method calls
    metadata: Mapping[str, object] = field(default_factory=dict)
    repository: str | None = None                # Name of source repository
    source_code: str | None = None               # Extracted raw source text
    docstring: str | None = None                 # Extracted docstring
```

### Key Properties
- **Immutability**: `frozen=True` ensures CodeUnits are hashable, cacheable, and thread-safe.
- **Hierarchical Scoping**: `parent_id` enables walking up to enclosing classes or modules.
- **Static Dependencies**: `calls` and `imports` capture static call graphs and dependency graphs during the AST parse phase, before any graph execution.

---

## 4. Code Graph Representation & Traversal (`repolens.graph`)

RepoLens constructs a directed, typed multi-graph using **NetworkX** (`MultiDiGraph`).

### 4.1 Edge Types (`code_graph.py`)
1. **`CONTAINS`**:
   - Directed edge from parent container to enclosed element:
     - Module $\to$ Class
     - Class $\to$ Method
     - Module $\to$ Module-level Function
2. **`IMPORTS`**:
   - Directed edge from file/module to imported module or symbol.
3. **`CALLS`**:
   - Directed edge from caller function/method to callee function/method.
   - Built by resolving static call targets (`CallReference.target`) against the index of qualified names and short names.

### 4.2 Graph Queries (`graph_query.py`)
Provides indexed constant-time lookups:
- `get_callers(unit_id)`: Incoming `CALLS` edges.
- `get_callees(unit_id)`: Outgoing `CALLS` edges.
- `get_children(unit_id)`: Outgoing `CONTAINS` edges.
- `get_parent(unit_id)`: Incoming `CONTAINS` edge.
- `get_imports(unit_id)`: Outgoing `IMPORTS` edges.

### 4.3 Context Expansion (`context_expander.py`)
Expands outward from a seed set of CodeUnits to capture immediate architectural context:
- **1-Hop Expansion**: Gathers immediate callers, callees, and parent definitions.
- **Radial Expansion**: Extends up to $N$ hops with configurable degree decay, allowing the agent to follow execution paths (e.g. `Client.post` $\to$ `Client.request` $\to$ `Session.send` $\to$ `HTTPAdapter.send`).

---

## 5. Hybrid Retrieval & Reranking Subsystem (`repolens.retrieval`)

The retrieval pipeline combines exact lexical matching with dense semantic similarity, balanced by Reciprocal Rank Fusion and Cross-Encoder reranking.

```
                            [ User Query ]
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
      [ BM25 Lexical Retriever ]      [ Dense Semantic Retriever ]
      - Identifier-aware tokenization - SentenceTransformers (all-MiniLM-L6-v2)
      - BM25Okapi scoring             - FAISS IndexFlatIP (Cosine)
                 │                                 │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                   [ Reciprocal Rank Fusion (RRF) ]
                   - Combines ranks with k=60
                   - Configurable semantic weight
                                  │
                                  ▼
                   [ Cross-Encoder Reranker ]
                   - Evaluates query-code relevance
                   - Re-orders top-K candidates
```

### 5.1 BM25 Lexical Retrieval (`bm25_retriever.py`)
- Standard BM25 tokenizers fail on code identifiers like `authenticate_user_v2` or `HTTPConnectionPool`.
- RepoLens implements identifier-aware tokenization:
  - Splits camelCase (`HTTPConnection` $\to$ `http`, `connection`).
  - Splits snake_case (`parse_headers` $\to$ `parse`, `headers`).
  - Retains full compound tokens (`requests.sessions.Session`).
- Scores candidate CodeUnits using standard BM25Okapi term frequency and inverse document frequency saturation.

### 5.2 Dense Semantic Retrieval (`semantic_retriever.py`)
- Employs `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
- Generates searchable document representations per `CodeUnit`:
  $$\text{Document}(u) = \text{Kind: } u.\text{kind} + \text{" Name: "} + u.\text{qualified\_name} + \text{" Docstring: "} + u.\text{docstring} + \text{" Code: "} + u.\text{source\_code}$$
- Indexes vectors in memory using **FAISS** (`faiss.IndexFlatIP` with normalized vectors for cosine similarity).
- Handles empty queries and vector dimension mismatches gracefully.

### 5.3 Hybrid Fusion & Deduplication (`hybrid_retriever.py`)
- Integrates lexical and semantic rankings using Reciprocal Rank Fusion (RRF):
  $$\text{RRF Score}(u) = \frac{w_{\text{bm25}}}{k + \text{rank}_{\text{bm25}}(u)} + \frac{w_{\text{dense}}}{k + \text{rank}_{\text{dense}}(u)}$$
  Where $k = 60$ is the standard smoothing constant, and $w_{\text{bm25}} + w_{\text{dense}} = 1.0$.
- Deduplicates CodeUnits appearing in both retrieval lists while accumulating score signals.

### 5.4 Cross-Encoder Reranking (`reranker.py`)
- Re-evaluates top-$M$ candidates from hybrid retrieval ($M \approx 20$) against the query.
- Produces normalized relevance scores in $[0.0, 1.0]$.
- Discards candidates below a confidence threshold and outputs the top-$K$ candidates ($K \approx 5$).

---

## 6. Agentic Investigation Pipeline (`repolens.agent`)

The multi-turn investigation engine is orchestrated with **LangGraph** as a cyclic state machine.

### 6.1 State Schema (`InvestigationState`)
Defined in `src/repolens/agent/state.py` as an annotated `TypedDict`:

```python
class InvestigationState(TypedDict, total=False):
    query: str
    analyzed_query: str
    query_keywords: list[str]
    investigation_goals: list[str]
    requires_graph_traversal: bool
    requires_source_inspection: bool

    selected_tools: list[str]
    executed_tools: list[str]

    evidence: Annotated[list[dict[str, object]], merge_evidence]
    history: Annotated[list[dict[str, object]], merge_history]
    trace: Annotated[list[dict[str, object]], merge_trace]
    errors: Annotated[list[dict[str, str]], merge_errors]

    iteration: int
    max_iterations: int
    max_evidence: int
    min_evidence: int
    needs_more_investigation: bool
    is_complete: bool

    final_answer: str
    termination_reason: str

    citation_validation_results: list[dict[str, object]]
    claim_validation_results: list[dict[str, object]]
    validation_summary: dict[str, object]
```

### 6.2 Workflow Execution Graph
```
   START
     │
     ▼
[ query_analysis ]
     │
     ▼
┌─────────────────────────┐
│     tool_selection      │ ◄──────────────────────┐
└────────────┬────────────┘                        │
             │                                     │
             ▼                                     │
┌─────────────────────────┐                        │
│    tool_execution       │                        │
└────────────┬────────────┘                        │
             │                                     │
             ▼                                     │
┌─────────────────────────┐                        │ (Iteration < Max &&
│  investigation_control  │                        │  Needs More Evidence)
└────────────┬────────────┘                        │
             │                                     │
             ├──────────── [ route_after ] ────────┘
             │             (Termination reached)
             ▼
┌─────────────────────────┐
│   answer_preparation    │
└────────────┬────────────┘
             │
             ▼
            END
```

### 6.3 Toolset (`InvestigationTools`)
- `keyword_search(query, top_k)`: Executes BM25 retrieval.
- `semantic_search(query, top_k)`: Executes dense vector retrieval.
- `inspect_source(unit_id)`: Fetches full source text, docstring, and signature of a specific CodeUnit.
- `read_file(file_path, start_line, end_line)`: Reads arbitrary line ranges from physical files.
- `context_expansion(unit_id, hops)`: Traverses graph neighbors.
- `callers(unit_id)`: Lists all functions/methods calling target.
- `callees(unit_id)`: Lists all functions/methods called by target.

### 6.4 Termination Conditions
The loop terminates when any of the following occur:
1. `evidence_count >= max_evidence` (`termination_reason: "maximum_evidence"`).
2. `evidence_count >= min_evidence` and all required phases finished (`termination_reason: "sufficient_evidence"`).
3. Retrieval completed with 0 results (`termination_reason: "no_evidence"`).
4. `iteration >= max_iterations` (`termination_reason: "maximum_iterations"`).

---

## 7. Grounded Answer Generation & Citation Validation

### 7.1 Grounded Synthesis (`answer_generator.py`)
- `construct_prompt(query, evidence, errors)`: Constructs prompts containing:
  - Exact symbol declarations, file locations, and line ranges.
  - Inspected source code snippets.
  - Docstring context.
- **Evidence Confidence Separation**:
  - **Supported Findings**: Directly inspected via `inspect_source` or `read_file`.
  - **Inferred Findings**: Discovered via graph traversal without direct source verification.
- **Deterministic Synthesis Fallback**: When no external LLM API key is configured, RepoLens uses a local deterministic template engine that formats findings, lines, docstrings, and graph context into verified markdown without network calls.

### 7.2 Citation Extraction (`citation_extractor.py`)
Extracts citations using three regex patterns matching canonical citation grammar:
1. `File: <path>\n   Lines: <start>-<end>`
2. `<path>:<start>-<end>`
3. `` `<path>` (lines <start>-<end>) ``
Coupled with symbol extraction: `1. <qualified_symbol_name>`.

### 7.3 Source Citation Validator (`citation_validator.py`)
Validates every extracted citation against the codebase before presentation:
- **File Existence**: Verifies `Path(repo_root, file_path).is_file()`.
- **Path Traversal Security**: Ensures `resolve()` does not escape `repo_root` (blocks directory traversal attacks like `../../etc/passwd`).
- **Line Range Validity**: Checks $1 \le \text{start\_line} \le \text{end\_line} \le \text{total\_lines\_in\_file}$.
- **Symbol Confirmation**: Checks whether the cited symbol exists in the repository `CodeGraph` AST.
- **CodeUnit ID Match**: Confirms citation points to an indexed `CodeUnit`.
- Status emitted: `valid`, `invalid`, `incomplete`, `unverified`.

---

## 8. Streamlit Presentation Architecture (`repolens.ui`)

The UI is built with Streamlit and organized into modular layers:
- `src/repolens/ui/app.py`: Main application entrypoint with multi-page navigation.
- `src/repolens/ui/presentation.py`: Decoupled view-models (`PresentationData`, `EvidenceViewModel`, `CitationViewModel`, `TraceStepViewModel`, `ValidationSummaryViewModel`).
- `src/repolens/ui/components/`: Modular widgets (`render_answer_card`, `render_findings_section`, `render_evidence_viewer`, `render_citation_validation`, `render_investigation_trace`, `render_overview_metrics`).
- `src/repolens/ui/adapter.py`: Bridge between UI controls and the underlying evaluation / baseline runners (`run_repolens_query`).

### Performance Optimizations
- `@st.cache_resource`: Caches parsed ASTs and CodeGraph objects in memory per repository path. Subsequent queries on the same repository skip AST parsing entirely ($< 5\text{ ms}$).
- `st.session_state`: Preserves investigation outcomes across page switches and UI widget adjustments.
- `st.status`: Delivers progressive visual updates during long-running investigations without blocking UI interactivity.

---

## 9. Evaluation Framework Specification (`repolens.evaluation`)

The evaluation framework provides rigorous benchmarking across six incremental configurations:

| Baseline | Chunking | Retrieval | Graph Traversal | Agentic Loop | Citation Validation |
|---|---|---|---|---|---|
| **B0: Naive RAG** | Fixed 50-line window | Lexical token matching | None | None | None |
| **B1: AST-aware RAG** | AST `CodeUnit` | Dense semantic vector | None | None | AST boundary match |
| **B2: Hybrid RAG** | AST `CodeUnit` | BM25 + Dense (RRF) | None | None | AST boundary match |
| **B3: Graph RAG** | AST `CodeUnit` | Hybrid (RRF) | 1-hop expansion | None | AST + Graph match |
| **B4: Agentic RAG** | AST `CodeUnit` | Hybrid (RRF) | Multi-hop traversal | Multi-turn LangGraph | AST + Graph match |
| **B5: Full RepoLens** | AST `CodeUnit` | Hybrid + Reranker | Multi-hop traversal | Full StateGraph | Full 5-point validator |

### Metrics Formulation
- **Recall@K**: $\frac{|\text{Retrieved}_{1..K} \cap \text{Gold}|}{|\text{Gold}|}$
- **Precision@K**: $\frac{|\text{Retrieved}_{1..K} \cap \text{Gold}|}{K}$
- **MRR**: $\frac{1}{\min \{ \text{rank}(g) : g \in \text{Gold} \}}$
- **nDCG@K**: $\frac{\text{DCG@}K}{\text{IDCG@}K}$ where $\text{DCG@}K = \sum_{i=1}^K \frac{rel_i}{\log_2(i + 1)}$
- **Citation Precision**: $\frac{|\text{Valid Citations}|}{|\text{Total Citations Emitted}|}$
- **Token Cost**: $\left(\frac{\text{Prompt Tokens}}{10^6} \times \$0.15\right) + \left(\frac{\text{Completion Tokens}}{10^6} \times \$0.60\right)$

---

## 10. Summary of Architectural Guarantees

1. **Zero Hallucinated File Paths**: Every citation is verified against physical file existence.
2. **Zero Split Definitions**: AST parsing guarantees full function/class encapsulation.
3. **Cross-Module Relational Traceability**: Graph edges resolve imports and caller-callee hierarchies.
4. **Resilient Failure Handling**: Graceful fallback when evidence is absent, syntax is malformed, or external APIs are unavailable.
