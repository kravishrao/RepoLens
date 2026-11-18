from __future__ import annotations

import pytest

from repolens.agent.answer_generator import (
    AnswerGenerationError,
    AnswerGenerator,
    GroundedAnswer,
    construct_prompt,
)
from repolens.agent.state import InvestigationState
from repolens.ingestion.code_unit import CallReference, CodeUnit, ImportReference
from repolens.verification.citation_extractor import extract_citations


def make_code_unit(
    *,
    unit_id: str,
    name: str,
    source_code: str = "",
    docstring: str = "",
    file_path: str = "src/module.py",
    start_line: int = 1,
    end_line: int = 10,
    kind: str = "function",
    calls: list[str] | None = None,
    imports: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> CodeUnit:
    return CodeUnit(
        id=unit_id,
        kind=kind,
        name=name,
        qualified_name=f"module.{name}",
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        source_code=source_code,
        docstring=docstring,
        calls=tuple(CallReference(target=c) for c in (calls or [])),
        imports=tuple(ImportReference(module=i) for i in (imports or [])),
        metadata=metadata or {},
    )


class MockGenerationProvider:
    """Mock generation provider returning fixed content."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class FailingGenerationProvider:
    """Mock provider simulating an LLM API or runtime failure."""

    def generate(self, prompt: str) -> str:
        raise RuntimeError("LLM API service unavailable / rate limit reached")


# =============================================================================
# Prompt Construction Tests
# =============================================================================


def test_construct_prompt_includes_query_and_code_units():
    unit = make_code_unit(
        unit_id="auth-1",
        name="authenticate_user",
        file_path="src/auth/service.py",
        start_line=20,
        end_line=45,
        kind="function",
        source_code="def authenticate_user(username, password):\n    return True",
        docstring="Authenticate user credentials against database.",
        calls=["verify_password", "get_user"],
        imports=["hashlib"],
    )

    prompt = construct_prompt(
        query="Where and how does authentication happen?",
        evidence=[unit],
    )

    assert "Where and how does authentication happen?" in prompt
    assert "Symbol: module.authenticate_user" in prompt
    assert "src/auth/service.py:20-45" in prompt
    assert "def authenticate_user(username, password):" in prompt
    assert "Docstring: Authenticate user credentials against database." in prompt
    assert "Calls: verify_password, get_user" in prompt
    assert "Imports: hashlib" in prompt
    assert "CRITICAL GROUNDING RULES" in prompt


def test_construct_prompt_rejects_empty_query():
    unit = make_code_unit(unit_id="1", name="fn")

    with pytest.raises(ValueError, match="Question cannot be empty"):
        construct_prompt("", [unit])

    with pytest.raises(ValueError, match="Question cannot be empty"):
        construct_prompt("   ", [unit])


def test_construct_prompt_includes_errors_when_present():
    unit = make_code_unit(unit_id="1", name="fn")
    errors = [{"tool": "keyword_search", "error": "Index unavailable"}]

    prompt = construct_prompt(
        query="find fn",
        evidence=[unit],
        errors=errors,
    )

    assert "Investigation Warnings / Errors Encountered:" in prompt
    assert "keyword_search: Index unavailable" in prompt


# =============================================================================
# Grounded Answer Generation Tests
# =============================================================================


def test_answer_generator_generates_grounded_findings():
    evidence_record = {
        "code_unit_id": "calc-1",
        "name": "calculate_total",
        "qualified_name": "billing.calculate_total",
        "kind": "function",
        "file_path": "src/billing/calc.py",
        "start_line": 15,
        "end_line": 30,
        "source": "keyword_search",
        "content": "def calculate_total(items):\n    return sum(items)",
        "metadata": {"docstring": "Compute order total from items list."},
    }

    generator = AnswerGenerator()
    answer: GroundedAnswer = generator.generate(
        query="How is order total calculated?",
        evidence=[evidence_record],
    )

    assert answer.is_grounded is True
    assert answer.has_evidence is True
    assert len(answer.supported_findings) == 1
    assert answer.supported_findings[0]["name"] == "billing.calculate_total"
    assert answer.supported_findings[0]["file_path"] == "src/billing/calc.py"
    assert answer.supported_findings[0]["start_line"] == 15
    assert answer.supported_findings[0]["end_line"] == 30

    content = answer.content
    assert "Repository Investigation" in content
    assert "Question: How is order total calculated?" in content
    assert "1. billing.calculate_total" in content
    assert "File: src/billing/calc.py" in content
    assert "Lines: 15-30" in content
    assert "def calculate_total(items):" in content
    assert "Docstring: Compute order total from items list." in content


def test_answer_citations_extractable_by_citation_extractor():
    unit = make_code_unit(
        unit_id="unit-1",
        name="process_payment",
        file_path="src/pay/processor.py",
        start_line=50,
        end_line=85,
        kind="function",
        source_code="def process_payment(amount):\n    pass",
    )

    generator = AnswerGenerator()
    answer = generator.generate("Find payment processor", [unit])

    citations = extract_citations(answer.content)
    assert len(citations) >= 1
    citation = citations[0]
    assert citation.file_path == "src/pay/processor.py"
    assert citation.start_line == 50
    assert citation.end_line == 85
    assert citation.symbol == "module.process_payment"


def test_distinguishes_supported_vs_inferred_information():
    # Unit 1 has direct inspected source code -> Supported
    unit_inspected = {
        "code_unit_id": "u1",
        "name": "login",
        "qualified_name": "auth.login",
        "kind": "function",
        "file_path": "src/auth.py",
        "start_line": 10,
        "end_line": 20,
        "source": "inspect_source",
        "content": "def login(): pass",
    }
    # Unit 2 was discovered via graph traversal without source inspection -> Inferred
    unit_graph = {
        "code_unit_id": "u2",
        "name": "session_store",
        "qualified_name": "auth.session_store",
        "kind": "class",
        "file_path": "src/session.py",
        "start_line": 5,
        "end_line": 35,
        "source": "graph_traversal",
        "content": "",
        "metadata": {"calls": ["redis_connect"]},
    }

    generator = AnswerGenerator()
    answer = generator.generate("How does login work?", [unit_inspected, unit_graph])

    assert len(answer.supported_findings) == 1
    assert answer.supported_findings[0]["code_unit_id"] == "u1"

    # Unit 2 and its calls should be in inferred information
    assert len(answer.inferred_information) >= 1
    inferred_ids = [item.get("code_unit_id") for item in answer.inferred_information]
    assert "u2" in inferred_ids

    content = answer.content
    assert "Inferred / Uncertain Information" in content
    assert "Inferred element: auth.session_store in src/session.py" in content
    assert "implementation source code was not directly inspected" in content
    assert "Evidence Limitations" in content


def test_avoids_unsupported_claims_when_evidence_is_empty():
    generator = AnswerGenerator()
    answer = generator.generate("How does authorization work?", [])

    assert answer.is_grounded is False
    assert answer.has_evidence is False
    assert len(answer.supported_findings) == 0
    assert (
        "No relevant code evidence was found for this repository question."
        in answer.content
    )
    assert "No repository evidence was retrieved" in answer.unsupported_notes[0]


def test_handles_empty_query_validation():
    generator = AnswerGenerator()
    with pytest.raises(ValueError, match="Question cannot be empty"):
        generator.generate("", [])

    with pytest.raises(ValueError, match="Question cannot be empty"):
        generator.generate("   ", [])

    with pytest.raises(ValueError, match="Question cannot be empty"):
        generator.generate("\t\n", [])


def test_handles_generation_provider_failures_gracefully():
    unit = make_code_unit(
        unit_id="1",
        name="helper",
        file_path="src/utils.py",
        start_line=1,
        end_line=10,
        source_code="def helper(): pass",
    )

    failing_provider = FailingGenerationProvider()
    generator = AnswerGenerator(provider=failing_provider)

    # In safe mode (default), catches provider failure and falls back to deterministic answer
    answer = generator.generate("Describe helper", [unit], safe=True)
    assert answer.is_grounded is True
    assert "Repository Investigation" in answer.content
    assert "module.helper" in answer.content
    assert "File: src/utils.py" in answer.content

    # In strict mode (safe=False), raises AnswerGenerationError
    with pytest.raises(AnswerGenerationError, match="Answer generation failed"):
        generator.generate("Describe helper", [unit], safe=False)


def test_custom_generation_provider_integration():
    unit = make_code_unit(unit_id="1", name="fn", file_path="src/fn.py")
    mock_provider = MockGenerationProvider(
        "1. module.fn\n   File: src/fn.py\n   Lines: 1-10\nCustom LLM analysis."
    )
    generator = AnswerGenerator(provider=mock_provider)

    answer = generator.generate("Analyze fn", [unit])
    assert (
        answer.content
        == "1. module.fn\n   File: src/fn.py\n   Lines: 1-10\nCustom LLM analysis."
    )
    assert len(mock_provider.prompts) == 1
    assert "Analyze fn" in mock_provider.prompts[0]


def test_generate_from_state_workflow_integration():
    state: InvestigationState = {
        "query": "Find calculate_total",
        "evidence": [
            {
                "code_unit_id": "unit-1",
                "name": "calculate_total",
                "qualified_name": "billing.calculate_total",
                "kind": "function",
                "file_path": "src/calc.py",
                "start_line": 10,
                "end_line": 20,
                "source": "keyword_search",
                "content": "def calculate_total(): pass",
            }
        ],
        "trace": [
            {
                "iteration": 1,
                "tool": "keyword_search",
                "status": "success",
                "result_count": 1,
                "retained_count": 1,
                "duration_ms": 12,
            }
        ],
        "errors": [],
    }

    generator = AnswerGenerator()
    result = generator.generate_from_state(state)

    assert result["is_complete"] is True
    assert result["needs_more_investigation"] is False
    assert result["termination_reason"] == "investigation_completed"
    assert "Execution Trace" in result["final_answer"]
    assert "billing.calculate_total" in result["final_answer"]
    assert "File: src/calc.py" in result["final_answer"]
