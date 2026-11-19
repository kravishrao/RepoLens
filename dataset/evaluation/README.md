# RepoLens Question-Answering Evaluation Dataset

## 1. Overview & Purpose

The **RepoLens QA Evaluation Dataset** is a structured, benchmark-grade evaluation suite comprising **100 repository-level questions, expected answers, and ground-truth gold evidence locations** across five diverse open-source Python repositories.

This dataset is designed specifically to benchmark:
1. **Evidence Retrieval Quality**: Assessing whether RepoLens's hybrid lexical/semantic retrieval, code-graph neighbor expansion, and reranking pipeline accurately retrieve the exact source files and line spans containing relevant code units.
2. **Answer Correctness & Grounding**: Evaluating the fidelity of repository-grounded answer generation against human-verified expected answers, measuring hallucination resistance and citation accuracy.
3. **Multi-Hop & Multi-File Reasoning**: Testing the system's ability to navigate dependencies, callers, callees, and cross-module relationships across architectural boundaries.

---

## 2. Dataset Composition & Statistics

The dataset spans five canonical Python repositories with distinct software architectures:

| Repository | Domain | Pinned Commit | Production Files | Tracked Files | Total Questions |
|---|---|---|---|---|---|
| **Requests** | HTTP Client Library | `3c7baaa` | 19 | 130 | 20 |
| **Flask** | WSGI Web Framework | `49943b5` | 24 | 236 | 20 |
| **Click** | CLI Interface Framework | `8abb77a` | 17 | 179 | 20 |
| **FastAPI** | Modern ASGI Web API | `8f9f47d` | 52 | 3,239 | 20 |
| **Pydantic** | Data Validation & Settings | `57ffc4a` | 55 | 3,579 | 20 |
| **Total** | | | **167** | **7,363** | **100** |

### Question Distribution by Category

Each repository includes balanced representation across code structural dimensions:

- **Functions & Methods**: Implementation logic, parameter validation, docstring contracts, return types.
- **Classes & Inheritance**: Class hierarchies, abstract base classes, dataclasses, mixins, method overrides.
- **Modules & Packages**: Top-level exports (`__all__`), namespace organization, internal utilities (`_internal_utils`).
- **Imports & Namespaces**: Cross-module symbol imports, compatibility wrappers (`compat.py`), lazy imports.
- **Dependencies & Packaging**: `pyproject.toml` dependencies, minimum version constraints, optional dependencies.
- **Call Relationships & Workflows**: End-to-end request dispatch, hook systems, signal emission, dependency injection trees.

### Complexity & Scope Breakdown

| Dimension | Classification | Count | Percentage |
|---|---|---|---|
| **Scope** | Single-File | 59 | 59% |
| | Multi-File | 41 | 41% |
| **Complexity** | Single-Hop (Direct Lookup) | 52 | 52% |
| | Multi-Hop (Relational Traversal) | 48 | 48% |
| **Hop Depth** | 1 Hop | 52 | 52% |
| | 2 Hops | 43 | 43% |
| | 3 Hops | 5 | 5% |

---

## 3. Machine-Readable Schema Specification

The dataset is provided in two formats:
- [`qa_dataset.json`](./qa_dataset.json): Full formatted JSON structure.
- [`qa_dataset.jsonl`](./qa_dataset.jsonl): Line-delimited JSON for streaming and batch processing.

### Sample Record

```json
{
  "id": "REQ-01",
  "repository": "requests",
  "category": "call_graph",
  "scope": "multi_file",
  "complexity": "multi_hop",
  "hop_count": 3,
  "elements_involved": ["functions", "classes", "call_relationships"],
  "question": "When requests.get() is called, how is the request dispatched through Session and HTTPAdapter, and where is the connection retrieved?",
  "expected_answer": "requests.get() in requests.api invokes Session.request() with method='GET'. Session.request() creates a Request, calls Session.prepare_request() to produce a PreparedRequest, merges session environment settings, and delegates to Session.send(). Session.send() resolves the mounted HTTPAdapter (defaulting to HTTPAdapter for http/https) via Session.get_adapter(), which invokes HTTPAdapter.send(). Inside HTTPAdapter.send(), the connection is retrieved by calling self.get_connection(request.url, proxies) from the urllib3 ConnectionPool.",
  "gold_evidence": [
    {
      "repository": "requests",
      "file_path": "src/requests/api.py",
      "start_line": 64,
      "end_line": 77,
      "symbol": "get",
      "role": "api_entrypoint"
    },
    {
      "repository": "requests",
      "file_path": "src/requests/sessions.py",
      "start_line": 557,
      "end_line": 653,
      "symbol": "Session.request",
      "role": "session_dispatch"
    },
    {
      "repository": "requests",
      "file_path": "src/requests/sessions.py",
      "start_line": 752,
      "end_line": 829,
      "symbol": "Session.send",
      "role": "session_transport"
    },
    {
      "repository": "requests",
      "file_path": "src/requests/adapters.py",
      "start_line": 450,
      "end_line": 550,
      "symbol": "HTTPAdapter.send",
      "role": "transport_adapter"
    }
  ],
  "keywords": [
    "requests.get",
    "Session.request",
    "Session.send",
    "HTTPAdapter.send",
    "get_connection"
  ]
}
```

### Field Definitions

| Field | Type | Description |
|---|---|---|
| `id` | `string` | Unique identifier (e.g. `REQ-01`, `FLK-05`, `CLK-11`, `FAP-14`, `PYD-08`). |
| `repository` | `string` | Identifier matching repository snapshot (`requests`, `flask`, `click`, `fastapi`, `pydantic`). |
| `category` | `string` | Primary conceptual category: `function`, `class`, `module`, `import`, `dependency`, `call_graph`, `workflow`. |
| `scope` | `string` | `single_file` or `multi_file`. |
| `complexity` | `string` | `single_hop` or `multi_hop`. |
| `hop_count` | `integer` | Number of logical navigation steps required across symbols or files. |
| `elements_involved` | `list[string]` | Granular code unit elements (e.g. `["functions", "classes", "call_relationships"]`). |
| `question` | `string` | Natural language question targeting repository architecture, logic, or structure. |
| `expected_answer` | `string` | Human-curated ground-truth answer explaining the mechanism, contracts, and behavior. |
| `gold_evidence` | `list[object]` | Exact source locations providing ground-truth evidence. |
| `gold_evidence[].file_path` | `string` | Relative path within repository snapshot (using `/` separators). |
| `gold_evidence[].start_line` | `integer` | 1-indexed starting line number of the evidence block. |
| `gold_evidence[].end_line` | `integer` | 1-indexed ending line number of the evidence block. |
| `gold_evidence[].symbol` | `string` | Target function, class, method, or statement name. |
| `gold_evidence[].role` | `string` | Architectural role (e.g. `primary_evidence`, `call_site`, `entry_point`, `dependency_declaration`). |
| `keywords` | `list[string]` | Salient technical keywords and symbols for search queries. |

---

## 4. Methodology & Ground-Truth Verification

### 1. Repository Selection & Snapshot Pinning
Five mature, widely adopted Python repositories were chosen to ensure diversity across paradigms:
- **Imperative / Object-Oriented**: Requests (HTTP sessions, adapters).
- **WSGI Synchronous**: Flask (context locals, WSGI dispatch, blueprints).
- **Command-Line Parsing**: Click (composable command groups, argument processing).
- **ASGI Modern Asynchronous**: FastAPI (dependency injection, async route execution).
- **Type Validation & Metaclasses**: Pydantic (Rust/C core integration, schema validators, serialization).

Snapshots were archived at exact Git commits without history or external build artifacts.

### 2. Question Generation Guidelines
Questions were designed to simulate real-world developer inquiries:
- **Architectural Flow**: Tracing requests from entry points through middleware, routing, and handlers.
- **Contract Verification**: Parameter types, validation bounds, error handling, and default values.
- **Cross-Component Dependencies**: How modules import and share state, and how third-party libraries (`urllib3`, `werkzeug`, `pydantic-core`, `itsdangerous`) are configured.
- **Exclusion of Trivia**: Avoided trivial comments or boilerplate lines; focused on core functional behavior.

### 3. Automated Ground-Truth Verification
All 100 question annotations were verified using automated AST and disk-checking routines:
- Verified that every `file_path` exists on disk in `dataset/repositories/<repo>/`.
- Verified that `1 <= start_line <= end_line <= total_file_lines`.
- Verified that the target symbol definition or expected implementation exists within the specified line bounds.
- Guaranteed zero broken links, invalid paths, or drifted line numbers.

---

## 5. Usage in RepoLens Evaluation

The dataset can be loaded programmatically using RepoLens's evaluation module:

```python
from repolens.evaluation import load_qa_dataset, validate_qa_dataset

# Load evaluation dataset
dataset = load_qa_dataset()
print(f"Loaded {dataset.total_questions} evaluation samples.")

# Validate ground-truth files on disk
errors = validate_qa_dataset(dataset)
assert len(errors) == 0, f"Validation issues: {errors}"

# Filter questions by repository or complexity
multi_hop_questions = dataset.filter_by_complexity("multi_hop")
fastapi_questions = dataset.filter_by_repository("fastapi")
```

### Metrics Computed

When evaluating RepoLens baselines (B0 through B5), this dataset provides ground-truth for:
1. **Evidence Precision**: Fraction of retrieved CodeUnits overlapping gold evidence line ranges.
2. **Evidence Recall / Coverage**: Fraction of gold evidence locations successfully retrieved within Top-K.
3. **Mean Reciprocal Rank (MRR)**: Rank of the first gold evidence location in retrieved results.
4. **Citation Precision**: Proportion of generated citations that match gold evidence locations.
5. **Answer Groundedness**: Factual alignment between RepoLens synthesized answers and `expected_answer`.

### Reproducible Benchmark Runner

Run evaluations across baseline configurations B0–B5 using the evaluation runner CLI:

```bash
python -m repolens.evaluation.runner --configs B0 B1 B2 B3 B4 B5 --output-json dataset/evaluation/benchmark_report.json --output-markdown dataset/evaluation/benchmark_report.md
```

Full benchmark results, methodology, and limitations are documented in [`benchmark_report.md`](./benchmark_report.md).

---

## 6. Known Limitations

- **Fixed Snapshots**: The dataset is anchored to specific repository commit snapshots. Upstream repository evolution will not be reflected unless snapshots and line ranges are updated synchronously.
- **Language Scope**: Limited to Python codebases; polyglot projects or non-Python files (C-extensions, Rust modules) are not represented in gold annotations.
- **Answer Scope**: Expected answers summarize architectural mechanisms concisely; alternate valid phrasing exists and semantic embedding similarity or LLM-as-a-judge should be used for automated evaluation.
- **Static Coverage**: Does not evaluate dynamic runtime behaviors (e.g. monkeypatching, runtime reflection beyond standard inspection).
