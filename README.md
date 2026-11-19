# RepoLens

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Architecture: LangGraph](https://img.shields.io/badge/agent-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Interface: Streamlit](https://img.shields.io/badge/ui-Streamlit-red.svg)](https://streamlit.io/)

**RepoLens** is an open-source, repository-level code intelligence and question-answering framework. It combines **AST-driven syntax partitioning**, **typed code-graph relationships**, **hybrid lexical-semantic retrieval**, **multi-turn agentic investigation (LangGraph)**, and **deterministic source citation validation** to produce factual, hallucination-free answers about complex Python codebases.

---

## Table of Contents

- [1. Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
- [2. System Architecture](#2-system-architecture)
- [3. Key Features](#3-key-features)
- [4. Prerequisites & Requirements](#4-prerequisites--requirements)
- [5. Installation Guide](#5-installation-guide)
- [6. Environment Configuration](#6-environment-configuration)
- [7. Core Workflow & Subsystems](#7-core-workflow--subsystems)
  - [7.1 Ingestion & AST Extraction](#71-ingestion--ast-extraction)
  - [7.2 CodeUnit Schema](#72-codeunit-schema)
  - [7.3 Code Graph Representation](#73-code-graph-representation)
  - [7.4 Hybrid Retrieval & Reranking](#74-hybrid-retrieval--reranking)
  - [7.5 Agentic Investigation Pipeline](#75-agentic-investigation-pipeline)
  - [7.6 Grounded Answer Generation & Citation Validation](#76-grounded-answer-generation--citation-validation)
- [8. Python API Reference & Shared Interfaces](#8-python-api-reference--shared-interfaces)
- [9. Streamlit Web Application](#9-streamlit-web-application)
- [10. Example Repository Questions & Usage](#10-example-repository-questions--usage)
- [11. Experimental Benchmarks & Evaluation](#11-experimental-benchmarks--evaluation)
- [12. Testing & CI/CD Execution](#12-testing--cicd-execution)
- [13. Faculty Demonstration Guide](#13-faculty-demonstration-guide)
- [14. Repository Structure](#14-repository-structure)
- [15. Team & Contribution Workflow](#15-team--contribution-workflow)

---

## 1. Executive Summary & Problem Statement

### The Problem with Conventional (Naive) RAG
Standard Retrieval-Augmented Generation (RAG) splits text into fixed-size character or line chunks (e.g., 50–100 lines) and relies exclusively on dense vector similarity. When applied to software repositories, this approach suffers from four critical failure modes:
1. **Syntactic Boundary Blindness**: Arbitrary window chunking cuts functions, methods, and classes in half, severing decorators, docstrings, and signature contexts.
2. **Relational Blindness**: Dense embeddings cannot follow cross-file imports, class inheritance hierarchies, caller-callee chains, or interface implementations.
3. **Citation Hallucination**: Conventional LLMs routinely invent plausible-sounding file paths, non-existent symbol names, and fictional line spans (unsupported citation rate exceeds 50% in naive baselines).
4. **Single-Hop Information Bottleneck**: Complex questions require multi-file, multi-hop investigation across modules that cannot be retrieved in a single vector search step.

### The RepoLens Solution
RepoLens transforms source code into a structured, queryable knowledge substrate:
- **AST Structural Units (`CodeUnit`)**: Extracted via Python's `ast` module with error recovery for syntax irregularities.
- **Typed Multi-Directed Code Graph (`CodeGraph`)**: Encodes `CONTAINS`, `IMPORTS`, and `CALLS` edges across all codebase elements.
- **Hybrid Retrieval**: Merges BM25 lexical token matching with dense SentenceTransformer embeddings via Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking.
- **Multi-Turn Agentic Exploration**: A cyclic LangGraph state machine formulates queries, inspects candidate definitions, traverses graph neighbors, and evaluates evidence sufficiency.
- **Citation Validation**: Deterministic verification validating that every file exists, path traversals are safe, line ranges are bounded, and symbols exist in the repository AST before answering.

---

## 2. System Architecture

```
                                  [ User Query ]
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   LangGraph Agent State     │
                         │    (InvestigationState)     │
                         └──────────────┬──────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
     ┌───────────────────────┐                     ┌───────────────────────┐
     │   Hybrid Retrieval    │                     │  Code Graph Traversal │
     │  ┌─────────────────┐  │                     │  ┌─────────────────┐  │
     │  │  BM25 Lexical   │  │                     │  │ Caller / Callee │  │
     │  ├─────────────────┤  │                     │  ├─────────────────┤  │
     │  │ Dense Embedding │  │                     │  │ Module Hierarchy│  │
     │  ├─────────────────┤  │                     │  ├─────────────────┤  │
     │  │ RRF & Reranker  │  │                     │  │Context Expander │  │
     │  └─────────────────┘  │                     │  └─────────────────┘  │
     └───────────┬───────────┘                     └───────────┬───────────┘
                 │                                             │
                 └──────────────────────┬──────────────────────┘
                                        │ (Collected Evidence)
                                        ▼
                         ┌─────────────────────────────┐
                         │   Grounded Answer Generator │
                         │  (Supported vs. Inferred)   │
                         └──────────────┬──────────────┘
                                        │ (Draft Answer & Citations)
                                        ▼
                         ┌─────────────────────────────┐
                         │ Source Citation Validator   │
                         │  - Path Safety Verification │
                         │  - AST Symbol Confirmation  │
                         │  - Line Range Validation    │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │    Streamlit Presentation   │
                         │ (Answer, Evidence, Traces)  │
                         └─────────────────────────────┘
```

---

## 3. Key Features

- **Strict AST-Grounded Extraction**: Functions, classes, methods, and module statements are preserved intact with precise 1-indexed `start_line` and `end_line` spans.
- **Code Graph Exploration**: NetworkX-backed multi-directed graph capturing containment, import hierarchies, and static function/method invocations.
- **Reciprocal Rank Fusion (RRF)**: Combines exact keyword matches (e.g., `Session.send`) with semantic vector similarities, followed by cross-encoder scoring.
- **LangGraph Multi-Turn State Machine**: Dynamically selects between lexical search, semantic search, source inspection, and graph traversal across multiple iterations.
- **Separation of Evidence Confidence**: Clear visual and logical distinction between **Directly Inspected Evidence** (verified file contents) and **Inferred / Referenced Elements** (uninspected graph neighbors).
- **Hallucination-Proof Citation Engine**: Regex-driven citation extraction coupled with AST verification guarantees that answers cite only real code locations.
- **Interactive Multi-Page Streamlit UI**: Complete workbench with repository switching, graph exploration, interactive query workbench, citation validation checklists, and benchmark dashboards.
- **Standardized Evaluation Framework**: Reproducible 100-question QA benchmark suite measuring Recall@K, Precision@K, MRR, nDCG@K, Token F1, Citation Precision, latency, and token cost across six baseline configurations (B0 through B5).

---

## 4. Prerequisites & Requirements

### System Requirements
- **Python**: `>= 3.11` (tested extensively on Python 3.11, 3.12, and 3.13).
- **Operating Systems**: Linux (Ubuntu 22.04+), macOS (12.0+), Windows (10/11 x64).
- **Memory**: Minimum 4 GB RAM (8 GB recommended for large embedding models and graph caching).
- **Disk Space**: ~1.5 GB for dependencies, bundled repositories, and pre-trained embedding weights.

### Core Dependencies
- `langgraph >= 1.2, < 2.0`: State graph execution engine for agent orchestration.
- `networkx >= 3.2`: Directed multi-graph data structures and traversal algorithms.
- `streamlit >= 1.40`: Reactive web application and interactive presentation UI.
- `sentence-transformers >= 3.0`: Dense code embedding generation (`all-MiniLM-L6-v2`).
- `faiss-cpu >= 1.7.4`: Efficient vector similarity search and index querying.
- `numpy >= 1.24.0`: Vector mathematical operations and rank normalization.
- `pytest >= 8.0` & `ruff >= 0.9`: Test execution and strict linting/formatting.

---

## 5. Installation Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/kravishrao/RepoLens.git
cd RepoLens
```

### Step 2: Create and Activate a Virtual Environment

**On Linux / macOS:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install RepoLens

**Standard Installation (Core + UI):**
```bash
pip install --upgrade pip
pip install -e .
```

**Full Installation (with Semantic Retrieval & Development Tools):**
```bash
pip install -e ".[dev,semantic]"
```

**Or from `requirements.txt`:**
```bash
pip install -r requirements.txt
```

---

## 6. Environment Configuration

RepoLens operates completely offline by default using local AST parsing, local NetworkX graphs, BM25 retrieval, and local SentenceTransformer embeddings. No paid cloud API keys are required for core functionality.

To customize runtime options, copy the template `.env.example` file:
```bash
cp .env.example .env
```

### Configuration Variables (`.env.example`)

| Variable | Default Value | Description |
|---|---|---|
| `REPOLENS_CACHE_DIR` | `.cache/repolens` | Local directory for cached graphs and embeddings. |
| `REPOLENS_LOG_LEVEL` | `INFO` | Console logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `REPOLENS_DEFAULT_REPO` | `dataset/repositories/requests` | Repository loaded on Streamlit startup. |
| `REPOLENS_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Pre-trained SentenceTransformers model name. |
| `REPOLENS_RETRIEVAL_TOP_K` | `10` | Candidate CodeUnits retrieved per query. |
| `REPOLENS_SEMANTIC_WEIGHT` | `0.5` | Weight balance between semantic and BM25 scores (0.0 to 1.0). |
| `REPOLENS_MAX_ITERATIONS` | `4` | Maximum agent investigation loops. |
| `REPOLENS_MAX_EVIDENCE` | `20` | Maximum evidence records retained. |
| `STREAMLIT_SERVER_PORT` | `8501` | Local port for Streamlit web server. |

> [!IMPORTANT]
> **Security Note:** Never commit `.env` containing sensitive credentials to Git. The `.gitignore` file is pre-configured to ignore `.env`, virtual environments, and `.cache/` directories.

---

## 7. Core Workflow & Subsystems

### 7.1 Ingestion & AST Extraction
- Located in `src/repolens/ingestion/`.
- `repository_loader.py`: Validates local repository directories or clones remote Git repositories.
- `file_discovery.py`: Discovers all `.py` files while respecting `.gitignore`, skipping virtual environments (`.venv`), `__pycache__`, and binary artifacts.
- `ast_parser.py`: Parses Python code into AST trees using `ast.parse()`. Includes robust error recovery for encoding issues, syntax errors, and malformed files.
- `repository_metadata.py`: Computes inventory statistics (total files, lines of code, symbol density).

### 7.2 CodeUnit Schema
Every extracted symbol is modeled as a strongly-typed, immutable `CodeUnit` (`src/repolens/ingestion/code_unit.py`):
```python
@dataclass(frozen=True)
class CodeUnit:
    id: str  # Deterministic UUID or path-qualified ID
    kind: Literal["module", "class", "function", "method"]
    name: str  # Short symbol name (e.g., 'send')
    qualified_name: str  # Full module path (e.g., 'requests.sessions.Session.send')
    file_path: str  # Relative path within repository
    start_line: int  # 1-indexed start line
    end_line: int  # 1-indexed end line
    parent_id: str | None = None  # Parent class or module ID
    imports: tuple[ImportReference, ...] = ()  # Static imports discovered
    calls: tuple[CallReference, ...] = ()  # Outgoing function/method calls
    metadata: Mapping[str, object] = field(default_factory=dict)
    source_code: str | None = None  # Exact source snippet
    docstring: str | None = None  # Extracted docstring
```

### 7.3 Code Graph Representation
- Located in `src/repolens/graph/code_graph.py`.
- Constructs a directed `networkx.MultiDiGraph` with typed edges:
  - `CONTAINS`: Module $\to$ Class $\to$ Method, or Module $\to$ Function.
  - `IMPORTS`: Source file $\to$ imported module or symbol.
  - `CALLS`: Caller function/method $\to$ target callee symbol.
- `graph_query.py`: Provides fast lookups for callers, callees, class hierarchies, and member definitions.
- `context_expander.py`: Performs multi-hop radial expansion around seed CodeUnits to capture immediate context.

### 7.4 Hybrid Retrieval & Reranking
- Located in `src/repolens/retrieval/`.
- `bm25_retriever.py`: Lexical retrieval tokenizing identifiers using snake_case and camelCase splitting (`Session.send` $\to$ `["session", "send"]`). Computes BM25Okapi scores.
- `semantic_retriever.py`: Encodes docstrings, signatures, and code bodies with SentenceTransformers into FAISS L2/Cosine index.
- `hybrid_retriever.py`: Applies Reciprocal Rank Fusion (RRF) with configurable weighting:
  $$\text{RRF Score}(u) = \frac{w_{\text{bm25}}}{k + \text{rank}_{\text{bm25}}(u)} + \frac{w_{\text{dense}}}{k + \text{rank}_{\text{dense}}(u)}$$
- `reranker.py`: Cross-encoder scoring to reorder top candidates based on deep query-code relevance.

### 7.5 Agentic Investigation Pipeline
- Located in `src/repolens/agent/`.
- Orchestrated via **LangGraph** `StateGraph(InvestigationState)`:
  - **Tool Selection (`select_tools`)**: Evaluates query requirements (e.g. requires graph traversal, source inspection, or keyword search).
  - **Tool Execution (`execute_tools`)**: Dispatches tools (`keyword_search`, `semantic_search`, `inspect_source`, `context_expansion`, `callers`, `callees`).
  - **Tracing & Control (`route_after_investigation`)**: Appends tool metrics to `trace`, evaluates evidence sufficiency (`min_evidence`, `max_evidence`), and decides whether to continue or terminate.
  - **Answer Preparation (`prepare_answer`)**: Delegates to `AnswerGenerator` for grounded synthesis.

### 7.6 Grounded Answer Generation & Citation Validation
- `answer_generator.py`: Generates answers grounded strictly in retrieved evidence. Segregates **Supported Findings** (directly inspected code) from **Inferred Findings** (graph context). Avoids unsupported claims when evidence is missing.
- `citation_extractor.py`: Extracts citations conforming to RepoLens citation grammar (`1. <symbol>\n   File: <path>\n   Lines: <start>-<end>`).
- `citation_validator.py`: Validates citations against the physical filesystem and AST graph:
  - File existence verification.
  - Path traversal security checks (preventing `../` escaping repository root).
  - Valid line boundaries ($1 \le \text{start} \le \text{end} \le \text{file\_lines}$).
  - AST symbol confirmation.
  - `CodeUnit` ID match confirmation.

---

## 8. Python API Reference & Shared Interfaces

### End-to-End Investigation Example
```python
from repolens.agent.tools import InvestigationTools
from repolens.agent.workflow import investigate_repository
from repolens.graph.code_graph import CodeGraph
from repolens.ingestion.ast_parser import parse_repository

# 1. Parse repository source files into CodeUnits
parse_result = parse_repository("dataset/repositories/requests")
code_units = parse_result.code_units

# 2. Build the CodeGraph
code_graph = CodeGraph()
code_graph.build(code_units)

# 3. Initialize Investigation Tools
tools = InvestigationTools(
    code_units=code_units,
    code_graph=code_graph,
    repository_root="dataset/repositories/requests",
)

# 4. Execute iterative investigation
state = investigate_repository(
    query="Where is Session.send defined and what hooks are dispatched?",
    tools=tools,
    repository_root="dataset/repositories/requests",
    max_iterations=4,
    max_evidence=20,
)

# 5. Access grounded answer and validation outcomes
print("=== Grounded Answer ===")
print(state["final_answer"])

print("\n=== Citation Validation ===")
for result in state.get("citation_validation_results", []):
    print(f"Citation: {result['citation']['raw_text']} -> Status: {result['status']}")

print(f"\nExecution iterations: {state['iteration']}")
print(f"Evidence collected: {len(state['evidence'])}")
```

---

## 9. Streamlit Web Application

RepoLens includes a comprehensive Streamlit workbench for visual investigation, code graph exploration, and benchmark analysis.

### Launching the Application
```bash
streamlit run src/repolens/ui/app.py
```
Or with custom port and host:
```bash
streamlit run src/repolens/ui/app.py --server.port 8501 --server.address 0.0.0.0
```

### Application Workspaces
The application provides five distinct views accessible via the sidebar navigation:

1. **Overview Workspace**:
   - High-level system dashboard displaying indexing metrics, active repository status, and pipeline architecture.
2. **Repositories Workspace**:
   - Manage bundled Python repositories (*Requests*, *Flask*, *Click*, *FastAPI*, *Pydantic*), inspect file counts, total lines of code, and symbol density.
3. **Investigation Workspace**:
   - Interactive Q&A interface with real-time `st.status` progress notifications.
   - Grounded answer display with structured markdown.
   - Code evidence viewer with syntax-highlighted snippets (`st.code`) and docstrings.
   - Source citation validation badges and security checklist.
   - Iteration-by-iteration tool execution trace.
4. **Code Graph Explorer**:
   - Interactive visualizer for module containment, class hierarchies, and caller-callee chains.
5. **Evaluation & Benchmarks**:
   - Live benchmark suite comparing Baselines **B0 through B5**.
   - Side-by-side metric tables (Recall, Precision, MRR, nDCG, Citation Precision, Cost, Latency).

---

## 10. Example Repository Questions & Usage

The repository includes five bundled, real-world Python codebases under `dataset/repositories/`. Below are tested queries illustrating RepoLens's multi-hop reasoning capabilities:

### 1. *Requests* (`dataset/repositories/requests`)
- **Query**: `"Where is the Session class defined, and how does Session.request construct a PreparedRequest?"`
- **Primary Symbol**: `requests.sessions.Session.request`
- **File**: `requests/sessions.py` (lines 485–550)
- **Call Chain**: `Session.request` $\to$ `Session.prepare_request` $\to$ `PreparedRequest.prepare` $\to$ `requests.models.PreparedRequest`.

### 2. *Click* (`dataset/repositories/click`)
- **Query**: `"How does Click parse option arguments and invoke the command callback function?"`
- **Primary Symbol**: `click.core.Command.invoke`
- **File**: `src/click/core.py` (lines 1400–1450)
- **Call Chain**: `Command.main` $\to$ `Command.invoke` $\to$ `Context.invoke` $\to$ callback.

### 3. *FastAPI* (`dataset/repositories/fastapi`)
- **Query**: `"How are dependency injection functions declared using Depends() resolved during route execution?"`
- **Primary Symbol**: `fastapi.dependencies.utils.solve_dependencies`
- **File**: `fastapi/dependencies/utils.py` (lines 450–600)
- **Call Chain**: `Routing.get_request_handler` $\to$ `solve_dependencies` $\to$ parameter resolution.

### 4. *Flask* (`dataset/repositories/flask`)
- **Query**: `"Where is the request context pushed onto the context stack during wsgi application dispatch?"`
- **Primary Symbol**: `flask.ctx.RequestContext.push`
- **File**: `src/flask/ctx.py` (lines 280–360)
- **Context**: `Flask.wsgi_app` $\to$ `RequestContext.push` $\to$ `_request_ctx_stack`.

### 5. *Pydantic* (`dataset/repositories/pydantic`)
- **Query**: `"Where is BaseModel.__init__ defined and how does it trigger field validation?"`
- **Primary Symbol**: `pydantic.main.BaseModel.__init__`
- **File**: `pydantic/main.py`
- **Context**: Schema building, `__pydantic_validator__`, and field assignment.

---

## 11. Experimental Benchmarks & Evaluation

RepoLens was evaluated across **100 curated repository-level questions** using pinned commit snapshots of five major Python repositories.

### Baseline Comparison (100 Questions)

| Baseline | Recall@5 | Precision@5 | MRR | nDCG@5 | Citation Precision | Latency (ms) | Avg Cost ($) |
|---|---|---|---|---|---|---|---|
| **B0: Naive RAG** | 0.3742 | 0.2300 | 0.4600 | 0.3848 | 0.4600 | 64.0 ms | $0.0038 |
| **B1: AST-aware RAG** | 0.4608 | 0.5800 | 0.5800 | 0.4865 | 1.0000 | 107.5 ms | $0.0029 |
| **B2: Hybrid RAG** | 0.6817 | 0.9000 | 0.9000 | 0.7294 | 1.0000 | 207.7 ms | $0.0118 |
| **B3: Graph RAG** | 0.9667 | 1.0000 | 1.0000 | 0.9744 | 1.0000 | 293.7 ms | $0.0136 |
| **B4: Agentic RAG** | 0.9750 | 1.0000 | 1.0000 | 0.9807 | 1.0000 | 608.3 ms | $0.0193 |
| **B5: Full RepoLens** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 728.0 ms | $0.0223 |

### Key Findings
1. **Elimination of Citation Hallucinations**: Moving from Naive RAG (B0) to AST-aware retrieval (B1) improved Citation Precision from **0.4600 to 1.0000**, eliminating fabricated file lines.
2. **Multi-Hop Necessity**: Graph expansion (B3) and Agentic iteration (B4–B5) increased Recall@5 from **0.6817 to 1.0000**, confirming that static code graphs are essential for resolving cross-module dependencies.
3. **Cost-Efficiency**: Full RepoLens operates at an average cost of **$0.0223 per query** with sub-second retrieval latency (728 ms).

For the complete benchmark methodology, formulas, and ablation experiments, see [benchmark_report.md](dataset/evaluation/benchmark_report.md) and [research_paper.md](docs/research_paper.md).

---

## 12. Testing & CI/CD Execution

RepoLens enforces strict quality gates through automated tests and linting.

### Running Local Tests
Execute the comprehensive test suite with pytest:
```bash
# Run all unit and integration tests
pytest tests/ -v

# Run with concise output
pytest tests/ -q

# Run specific test modules
pytest tests/test_ui_components.py
pytest tests/test_agent_workflow.py
pytest tests/test_benchmark_suite.py
```

### Running Code Quality Checks
Format and lint check the codebase using Ruff:
```bash
# Check code formatting
ruff format --check src tests

# Run linter
ruff check src tests

# Auto-apply safe formatting and lint fixes
ruff format src tests
ruff check --fix src tests
```

### Continuous Integration (GitHub Actions)
Every pull request targeting `develop` triggers the automated CI workflow (`.github/workflows/ci.yml`):
- **Job 1: Code Quality Gate**: Runs `ruff format --check` and `ruff check`.
- **Job 2: Test Suite**: Runs all unit and integration tests across Linux environments on Python 3.11.

---

## 13. Faculty Demonstration Guide

For faculty evaluations, capstone defenses, and technical demonstrations, a dedicated guide is available:

👉 **[Faculty Demonstration Guide (`docs/faculty_demonstration_guide.md`)](docs/faculty_demonstration_guide.md)**

The guide contains:
- Complete live demonstration script (6 structured scenarios).
- Concrete questions, expected symbols, and verified answers.
- Architecture and theoretical defense talking points.
- Experimental result charts and benchmark comparisons.
- Answers to common technical and academic questions.

---

## 14. Repository Structure

```
RepoLens/
├── .github/
│   ├── pull_request_template.md    # Standard PR template with Jira tracking
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI workflow (lint + tests)
├── .env.example                    # Template environment variables
├── CONTRIBUTING.md                 # Team branch, commit, and review conventions
├── LICENSE                         # MIT License
├── pyproject.toml                  # Project metadata, tools, and dependencies
├── README.md                       # Main project documentation
├── requirements.txt                # Pinned production and test dependencies
├── dataset/
│   ├── evaluation/
│   │   ├── benchmark_report.json   # Full raw experimental results (B0-B5)
│   │   ├── benchmark_report.md     # Formatted benchmark evaluation report
│   │   ├── qa_dataset.json         # 100-question ground-truth evaluation dataset
│   │   └── qa_dataset.jsonl        # JSONL format of evaluation dataset
│   └── repositories/               # Bundled evaluation codebases
│       ├── click/
│       ├── fastapi/
│       ├── flask/
│       ├── pydantic/
│       └── requests/
├── docs/
│   ├── baseline_and_ablation_experiments.md # Detailed ablation analysis
│   ├── faculty_demonstration_guide.md       # Faculty evaluation & demo script
│   ├── research_paper.md                    # Academic research paper (Markdown)
│   ├── research_paper.tex                   # Academic research paper (LaTeX)
│   └── technical_architecture.md            # In-depth technical specification
├── src/
│   └── repolens/
│       ├── agent/                  # LangGraph multi-turn agentic workflow
│       │   ├── answer_generator.py # Grounded answer synthesis & prompt formatting
│       │   ├── nodes.py            # Workflow node functions & iteration control
│       │   ├── state.py            # InvestigationState TypedDict definition
│       │   ├── tools.py            # Investigation tools wrapper
│       │   └── workflow.py         # StateGraph assembly & routing logic
│       ├── evaluation/             # Evaluation framework & baseline runners
│       │   ├── ablation.py         # Component ablation runner
│       │   ├── baselines.py        # Baseline implementations (B0-B5)
│       │   ├── benchmark_suite.py  # 100-question benchmark runner
│       │   ├── metrics.py          # Recall, Precision, MRR, nDCG, F1 formulas
│       │   └── models.py           # Evaluation dataclasses & schemas
│       ├── graph/                  # Code graph data structures & traversal
│       │   ├── code_graph.py       # NetworkX MultiDiGraph representation
│       │   ├── context_expander.py # Multi-hop neighbor expansion
│       │   └── graph_query.py      # Callers, callees, and hierarchy queries
│       ├── ingestion/              # Repository loading & AST parsing
│       │   ├── ast_parser.py       # Syntax-tree parsing with error recovery
│       │   ├── code_unit.py        # CodeUnit, CallReference, ImportReference
│       │   ├── file_discovery.py   # Python source file scanner
│       │   ├── repository_loader.py# Local & Git repo loading
│       │   └── repository_metadata.py # Codebase inventory statistics
│       ├── retrieval/              # Lexical & semantic retrieval
│       │   ├── bm25_retriever.py   # Identifier-aware BM25Okapi retriever
│       │   ├── hybrid_retriever.py # Reciprocal Rank Fusion (RRF)
│       │   ├── reranker.py         # Cross-encoder relevance reranker
│       │   └── semantic_retriever.py# Dense SentenceTransformers + FAISS
│       ├── ui/                     # Interactive Streamlit application
│       │   ├── adapter.py          # Unified query runner adapter
│       │   ├── app.py              # Main Streamlit application entrypoint
│       │   ├── presentation.py     # Presentation view-models & extraction
│       │   ├── components/         # Reusable UI widgets & results cards
│       │   └── styles/             # CSS theme & typography styles
│       └── verification/           # Source citation & claim validation
│           ├── citation_extractor.py # Regex citation parser
│           ├── citation_models.py  # Citation, Issue, ValidationResult models
│           ├── citation_validator.py# Path, symbol, and line verification
│           └── claim_validator.py  # Lexical claim-to-evidence verification
└── tests/                          # 34 test files (440+ unit/integration tests)
```

---

## 15. Team & Contribution Workflow

RepoLens was developed collaboratively following standard agile engineering practices.

### Team Members
- **Kravish Rao** ([@kravishrao](https://github.com/kravishrao)) – Project Lead
- **Manoj Kanur** ([@Manoj-kanur](https://github.com/Manoj-kanur)) – Core Developer
- **Mohit Malur** ([@mohit2007malur-png](https://github.com/mohit2007malur-png)) – Core Developer
- **Harshith Kumar** ([@BmHarshithKumar](https://github.com/BmHarshithKumar)) – Core Developer

### Branch & Review Model
- **`develop`**: The primary integration branch. All feature branches merge into `develop`.
- **`main`**: Production releases only.
- **Branch naming convention**: `<prefix>/RL-XX-short-description` (e.g., `feature/RL-31-grounded-answer-generation`, `docs/RL-44-technical-documentation`).
- **Commit style**: Strict [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
- **Pull Request Rules**: Every PR requires at least 1 team code review approval and green CI status before merging. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
