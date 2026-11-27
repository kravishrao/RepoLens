"""Unit tests for evidence association, traceability, and insufficient evidence handling (RL-36)."""

from __future__ import annotations

from repolens.agent.answer_generator import (
    AnswerGenerator,
    CodeUnitAssociation,
    EvidenceStatus,
    SupportType,
    format_line_reference,
)
from repolens.agent.state import InvestigationState
from repolens.ingestion.code_unit import CallReference, CodeUnit, ImportReference
from repolens.ui.presentation import extract_presentation_data


def make_test_code_unit(
    *,
    unit_id: str,
    name: str,
    qualified_name: str | None = None,
    file_path: str = "src/module.py",
    start_line: int = 10,
    end_line: int = 25,
    kind: str = "function",
    source_code: str = "def func():\n    return 42",
    docstring: str = "Test docstring.",
    calls: list[str] | None = None,
    imports: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> CodeUnit:
    """Helper to create a test CodeUnit."""
    return CodeUnit(
        id=unit_id,
        kind=kind,
        name=name,
        qualified_name=qualified_name or f"module.{name}",
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        source_code=source_code,
        docstring=docstring,
        calls=tuple(CallReference(target=c) for c in (calls or [])),
        imports=tuple(ImportReference(module=i) for i in (imports or [])),
        metadata=metadata or {},
    )


# =============================================================================
# 1. Association & Line Reference Presentation Tests
# =============================================================================


def test_format_line_reference():
    """Verify human-readable line reference formatting."""
    assert format_line_reference(10, 20) == "Lines 10-20"
    assert format_line_reference(15, 15) == "Line 15"
    assert format_line_reference(10, None) == "Line 10+"
    assert format_line_reference(None, None) == "Lines unknown"


def test_associate_generated_answers_with_code_units():
    """Verify answers associate directly with supporting CodeUnits and metadata."""
    unit = make_test_code_unit(
        unit_id="unit-auth-101",
        name="verify_credentials",
        qualified_name="auth.service.verify_credentials",
        file_path="src/auth/service.py",
        start_line=50,
        end_line=75,
        kind="function",
        source_code="def verify_credentials(user, token):\n    return token.is_valid()",
        docstring="Verify incoming user authentication token.",
        calls=["token.is_valid"],
        imports=["hashlib"],
    )

    generator = AnswerGenerator()
    associations = generator.build_associations([unit])

    assert len(associations) == 1
    assoc = associations[0]
    assert assoc.code_unit_id == "unit-auth-101"
    assert assoc.name == "verify_credentials"
    assert assoc.qualified_name == "auth.service.verify_credentials"
    assert assoc.kind == "function"
    assert assoc.file_path == "src/auth/service.py"
    assert assoc.start_line == 50
    assert assoc.end_line == 75
    assert assoc.line_reference == "Lines 50-75"
    assert assoc.support_type == SupportType.DIRECT
    assert assoc.calls == ["token.is_valid"]
    assert assoc.imports == ["hashlib"]
    assert "def verify_credentials" in (assoc.content_snippet or "")


def test_present_relevant_file_paths_symbols_and_lines_in_answer():
    """Verify file paths, symbols, and line spans appear in generated answer content."""
    unit1 = make_test_code_unit(
        unit_id="u1",
        name="handle_request",
        qualified_name="server.handle_request",
        file_path="src/server.py",
        start_line=100,
        end_line=140,
        kind="function",
        source_code="def handle_request(req):\n    pass",
    )
    unit2 = make_test_code_unit(
        unit_id="u2",
        name="RequestHandler",
        qualified_name="server.RequestHandler",
        file_path="src/handler.py",
        start_line=1,
        end_line=50,
        kind="class",
        source_code="class RequestHandler:\n    pass",
    )

    generator = AnswerGenerator()
    answer = generator.generate("How are requests dispatched?", [unit1, unit2])

    assert "server.handle_request" in answer.content
    assert "File: src/server.py" in answer.content
    assert "Lines: 100-140" in answer.content

    assert "server.RequestHandler" in answer.content
    assert "File: src/handler.py" in answer.content
    assert "Lines: 1-50" in answer.content

    assert len(answer.associations) == 2
    symbols = {a.qualified_name for a in answer.associations}
    assert "server.handle_request" in symbols
    assert "server.RequestHandler" in symbols


# =============================================================================
# 2. Directly Supported vs. Inferred Explanations
# =============================================================================


def test_distinguish_directly_supported_from_inferred_findings():
    """Verify verified source inspections are separated from inferred graph references."""
    direct_unit = {
        "code_unit_id": "direct-1",
        "name": "authenticate",
        "qualified_name": "core.auth.authenticate",
        "kind": "function",
        "file_path": "src/core/auth.py",
        "start_line": 20,
        "end_line": 40,
        "source": "inspect_source",
        "content": "def authenticate(user):\n    return True",
    }
    inferred_unit = {
        "code_unit_id": "inferred-2",
        "name": "db_session",
        "qualified_name": "core.database.db_session",
        "kind": "function",
        "file_path": "src/core/database.py",
        "start_line": 80,
        "end_line": 95,
        "source": "context_expansion",
        "content": "",  # Uninspected source
        "metadata": {"calls": ["connect_db"]},
    }

    generator = AnswerGenerator()
    answer = generator.generate(
        "Explain authentication pipeline", [direct_unit, inferred_unit]
    )

    assert len(answer.supported_findings) == 1
    assert answer.supported_findings[0]["code_unit_id"] == "direct-1"

    assert len(answer.inferred_information) >= 1
    inferred_uids = [item.get("code_unit_id") for item in answer.inferred_information]
    assert "inferred-2" in inferred_uids

    # Check associations support types
    assoc_by_id = {a.code_unit_id: a for a in answer.associations}
    assert assoc_by_id["direct-1"].support_type == SupportType.DIRECT
    assert assoc_by_id["inferred-2"].support_type == SupportType.INFERRED

    assert "Inferred / Uncertain Information" in answer.content
    assert "implementation source code was not directly inspected" in answer.content


# =============================================================================
# 3. Insufficient Evidence Scenarios & Clear Explanations
# =============================================================================


def test_identify_insufficient_evidence_when_empty():
    """Verify completely empty evidence is flagged as insufficient with clear explanation."""
    generator = AnswerGenerator()
    answer = generator.generate("What algorithm does encrypt_data() use?", [])

    assert answer.evidence_status == EvidenceStatus.INSUFFICIENT
    assert answer.is_grounded is False
    assert answer.has_evidence is False
    assert answer.assessment is not None
    assert answer.assessment.is_sufficient is False
    assert answer.assessment.status == EvidenceStatus.INSUFFICIENT
    assert len(answer.assessment.missing_aspects) >= 1

    # Check clear response text
    assert (
        "No relevant code evidence was found for this repository question."
        in answer.content
    )
    assert "Evidence Assessment: Insufficient Evidence" in answer.content
    assert "Suggested Search Refinements:" in answer.content


def test_identify_insufficient_evidence_when_target_symbol_missing():
    """Verify partial, unrelated evidence is flagged as insufficient for specific query."""
    unrelated_unit = make_test_code_unit(
        unit_id="unrelated-1",
        name="render_html",
        qualified_name="ui.render_html",
        file_path="src/ui.py",
        source_code="def render_html(): pass",
    )

    generator = AnswerGenerator()
    # Query targets calculate_mortgage, but only render_html was retrieved
    answer = generator.generate(
        "What is the formula used in calculate_mortgage?", [unrelated_unit]
    )

    assert answer.evidence_status == EvidenceStatus.INSUFFICIENT
    assert answer.assessment is not None
    assert answer.assessment.is_sufficient is False
    assert any("calculate_mortgage" in m for m in answer.assessment.missing_aspects)
    assert "Repository Investigation: Insufficient Evidence" in answer.content
    assert "Suggested Search Refinements:" in answer.content


def test_identify_uncertain_evidence_when_graph_only():
    """Verify evidence lacking direct inspection is classified as UNCERTAIN."""
    graph_unit = {
        "code_unit_id": "graph-only-1",
        "name": "send_email",
        "qualified_name": "notify.send_email",
        "kind": "function",
        "file_path": "src/notify.py",
        "start_line": 10,
        "end_line": 30,
        "source": "graph_traversal",
        "content": "",
    }

    generator = AnswerGenerator()
    answer = generator.generate("How are notifications dispatched?", [graph_unit])

    assert answer.evidence_status == EvidenceStatus.UNCERTAIN
    assert answer.assessment is not None
    assert answer.assessment.is_uncertain is True
    assert answer.assessment.direct_findings_count == 0
    assert answer.assessment.inferred_findings_count == 1


# =============================================================================
# 4. Conflicting Evidence Identification
# =============================================================================


def test_identify_conflicting_evidence_different_signatures():
    """Verify conflicting definitions of the same symbol across different files are detected."""
    unit_v1 = {
        "code_unit_id": "v1-client",
        "name": "init_client",
        "qualified_name": "client.init_client",
        "kind": "function",
        "file_path": "src/v1/client.py",
        "start_line": 10,
        "end_line": 25,
        "source": "inspect_source",
        "content": "def init_client(api_key: str, timeout: int = 30):\n    return Client(api_key)",
    }
    unit_v2 = {
        "code_unit_id": "v2-client",
        "name": "init_client",
        "qualified_name": "client.init_client",
        "kind": "function",
        "file_path": "src/v2/client.py",
        "start_line": 40,
        "end_line": 65,
        "source": "inspect_source",
        "content": "def init_client(config: ClientConfig, strict: bool = True):\n    return ModernClient(config)",
    }

    generator = AnswerGenerator()
    answer = generator.generate("How is init_client configured?", [unit_v1, unit_v2])

    assert answer.evidence_status == EvidenceStatus.CONFLICTING
    assert answer.assessment is not None
    assert answer.assessment.is_conflicting is True
    assert len(answer.conflicting_evidence) >= 1

    conflict = answer.conflicting_evidence[0]
    assert conflict.symbol == "client.init_client"
    assert "Conflicting signatures" in conflict.description

    assert "Conflicting Evidence Identified" in answer.content
    assert "src/v1/client.py" in answer.content
    assert "src/v2/client.py" in answer.content


def test_identify_conflicting_evidence_kind_mismatch():
    """Verify same symbol defined as class and function is detected as a conflict."""
    unit_func = {
        "code_unit_id": "u-func",
        "name": "SessionManager",
        "qualified_name": "session.SessionManager",
        "kind": "function",
        "file_path": "src/compat.py",
        "start_line": 10,
        "end_line": 20,
        "source": "inspect_source",
        "content": "def SessionManager():\n    return _Manager()",
    }
    unit_class = {
        "code_unit_id": "u-class",
        "name": "SessionManager",
        "qualified_name": "session.SessionManager",
        "kind": "class",
        "file_path": "src/session.py",
        "start_line": 30,
        "end_line": 100,
        "source": "inspect_source",
        "content": "class SessionManager:\n    pass",
    }

    generator = AnswerGenerator()
    answer = generator.generate("What is SessionManager?", [unit_func, unit_class])

    assert answer.evidence_status == EvidenceStatus.CONFLICTING
    assert len(answer.conflicting_evidence) >= 1
    conflict = answer.conflicting_evidence[0]
    assert conflict.conflict_type == "symbol_kind_mismatch"


# =============================================================================
# 5. Traceability & Serialization Tests
# =============================================================================


def test_bidirectional_traceability_preservation():
    """Verify traceability map links symbol and file keys to CodeUnit IDs."""
    unit1 = make_test_code_unit(
        unit_id="unit-parse-01",
        name="parse_tokens",
        qualified_name="parser.parse_tokens",
        file_path="src/parser.py",
    )
    unit2 = make_test_code_unit(
        unit_id="unit-ast-02",
        name="build_ast",
        qualified_name="ast.build_ast",
        file_path="src/ast.py",
    )

    generator = AnswerGenerator()
    answer = generator.generate("How does parsing work?", [unit1, unit2])

    t_map = answer.traceability_map
    assert "parser.parse_tokens" in t_map
    assert "unit-parse-01" in t_map["parser.parse_tokens"]
    assert "parse_tokens" in t_map
    assert "src/parser.py" in t_map

    assert "ast.build_ast" in t_map
    assert "unit-ast-02" in t_map["ast.build_ast"]


def test_code_unit_association_serialization_roundtrip():
    """Verify CodeUnitAssociation serialization to and from dict."""
    assoc = CodeUnitAssociation(
        code_unit_id="unit-test-1",
        name="compute",
        qualified_name="math.compute",
        kind="function",
        file_path="src/math.py",
        start_line=12,
        end_line=24,
        line_reference="Lines 12-24",
        support_type=SupportType.DIRECT,
        source="inspect_source",
        content_snippet="def compute(): return 1",
        docstring="Compute value",
        calls=["add"],
        imports=["decimal"],
        relevance_score=0.95,
        metadata={"custom": "data"},
    )

    d = assoc.to_dict()
    reloaded = CodeUnitAssociation.from_dict(d)

    assert reloaded.code_unit_id == assoc.code_unit_id
    assert reloaded.name == assoc.name
    assert reloaded.qualified_name == assoc.qualified_name
    assert reloaded.support_type == SupportType.DIRECT
    assert reloaded.line_reference == "Lines 12-24"
    assert reloaded.relevance_score == 0.95
    assert reloaded.calls == ["add"]
    assert reloaded.metadata == {"custom": "data"}


# =============================================================================
# 6. Workflow State & Presentation Data Integration Tests
# =============================================================================


def test_generate_from_state_populates_evidence_fields():
    """Verify generate_from_state populates evidence_status and associations in state dict."""
    unit = make_test_code_unit(
        unit_id="unit-exec-1",
        name="execute",
        file_path="src/exec.py",
        source_code="def execute(): pass",
    )

    state: InvestigationState = {
        "query": "How is execution handled?",
        "evidence": [unit],
        "trace": [
            {"iteration": 1, "tool": "search", "status": "success", "duration_ms": 12.0}
        ],
        "errors": [],
    }

    generator = AnswerGenerator()
    result = generator.generate_from_state(state)

    assert "final_answer" in result
    assert result["is_complete"] is True
    assert result["evidence_status"] == "sufficient"
    assert "evidence_assessment" in result
    assert result["evidence_assessment"]["is_sufficient"] is True
    assert len(result["code_unit_associations"]) == 1
    assert "traceability_map" in result


def test_extract_presentation_data_with_evidence_status_and_conflicts():
    """Verify extract_presentation_data maps evidence status, conflicts, and reasons."""
    state = {
        "query": "Where is config parsed?",
        "final_answer": "Repository Investigation findings...",
        "is_complete": True,
        "evidence_status": "conflicting",
        "evidence_assessment": {
            "status": "conflicting",
            "is_sufficient": False,
            "is_conflicting": True,
            "missing_aspects": ["Branch mismatch in config definitions"],
        },
        "conflicting_evidence": [
            {
                "symbol": "parse_config",
                "conflict_type": "divergent_signature",
                "description": "Two conflicting signatures in config.py vs legacy.py",
                "conflicting_units": [],
            }
        ],
        "evidence": [
            {
                "code_unit_id": "cfg-1",
                "name": "parse_config",
                "file_path": "src/config.py",
                "source": "inspect_source",
                "content": "def parse_config(): pass",
            }
        ],
    }

    pres = extract_presentation_data(state)
    assert pres.evidence_status == "conflicting"
    assert pres.is_conflicting is True
    assert len(pres.conflicts) == 1
    assert pres.conflicts[0]["symbol"] == "parse_config"
    assert "Branch mismatch in config definitions" in pres.insufficient_reasons
