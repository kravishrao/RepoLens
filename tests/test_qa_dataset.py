"""Unit tests for RepoLens Question-Answering Evaluation Dataset."""

from __future__ import annotations

from pathlib import Path

from repolens.evaluation.qa_dataset import (
    GoldEvidenceLocation,
    QAEvaluationDataset,
    QuestionSample,
    load_qa_dataset,
    validate_qa_dataset,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = WORKSPACE_ROOT / "dataset" / "evaluation" / "qa_dataset.json"
DATASET_JSONL_PATH = WORKSPACE_ROOT / "dataset" / "evaluation" / "qa_dataset.jsonl"
REPOS_ROOT = WORKSPACE_ROOT / "dataset" / "repositories"


def test_qa_dataset_file_exists() -> None:
    """Verify that the evaluation dataset JSON and JSONL files exist."""
    assert DATASET_PATH.exists(), f"Missing dataset file: {DATASET_PATH}"
    assert DATASET_JSONL_PATH.exists(), (
        f"Missing dataset JSONL file: {DATASET_JSONL_PATH}"
    )


def test_load_qa_dataset() -> None:
    """Verify load_qa_dataset loads the full benchmark dataset with required properties."""
    dataset = load_qa_dataset(DATASET_PATH)

    # Acceptance criteria: approximately 80-100 repository-level questions
    assert dataset.total_questions >= 80
    assert dataset.total_questions == 100
    assert len(dataset.questions) == 100

    # Verify repository distribution across all 5 repos
    expected_repos = {"requests", "flask", "click", "fastapi", "pydantic"}
    actual_repos = {q.repository for q in dataset.questions}
    assert expected_repos == actual_repos

    for repo in expected_repos:
        repo_questions = dataset.filter_by_repository(repo)
        assert len(repo_questions) == 20, (
            f"Expected 20 questions for {repo}, got {len(repo_questions)}"
        )


def test_qa_dataset_structural_diversity() -> None:
    """Verify that dataset covers functions, classes, modules, imports, dependencies, and call relationships."""
    dataset = load_qa_dataset(DATASET_PATH)

    categories = {q.category for q in dataset.questions}
    assert "function" in categories
    assert "class" in categories
    assert "module" in categories
    assert "import" in categories
    assert "dependency" in categories
    assert "call_graph" in categories

    all_elements: set[str] = set()
    for q in dataset.questions:
        all_elements.update(q.elements_involved)

    required_elements = {
        "functions",
        "classes",
        "modules",
        "imports",
        "dependencies",
        "call_relationships",
    }
    assert required_elements.issubset(all_elements)


def test_qa_dataset_multi_file_and_multi_hop() -> None:
    """Verify dataset includes substantial multi-file and multi-hop understanding questions."""
    dataset = load_qa_dataset(DATASET_PATH)

    multi_file_questions = dataset.filter_by_scope("multi_file")
    single_file_questions = dataset.filter_by_scope("single_file")
    multi_hop_questions = dataset.filter_by_complexity("multi_hop")
    single_hop_questions = dataset.filter_by_complexity("single_hop")

    assert len(multi_file_questions) >= 30
    assert len(single_file_questions) >= 40
    assert len(multi_hop_questions) >= 30
    assert len(single_hop_questions) >= 40

    # Verify hop counts
    hop_counts = {q.hop_count for q in dataset.questions}
    assert 1 in hop_counts
    assert 2 in hop_counts
    assert 3 in hop_counts

    for q in multi_hop_questions:
        assert q.hop_count >= 2


def test_qa_dataset_ground_truth_integrity() -> None:
    """Verify that all gold evidence file paths and line ranges exist on disk with zero errors."""
    dataset = load_qa_dataset(DATASET_PATH)
    errors = validate_qa_dataset(dataset, REPOS_ROOT)

    assert errors == [], (
        f"Dataset ground truth validation failed with {len(errors)} issues:\n"
        + "\n".join(errors)
    )


def test_qa_dataset_query_and_filter_helpers() -> None:
    """Verify querying and filtering utility methods on QAEvaluationDataset."""
    dataset = load_qa_dataset(DATASET_PATH)

    q = dataset.get_question("REQ-01")
    assert q is not None
    assert q.id == "REQ-01"
    assert q.repository == "requests"
    assert q.category == "call_graph"
    assert len(q.gold_evidence) >= 3

    assert dataset.get_question("NON-EXISTENT-ID") is None

    # Test filtering by repository
    flask_qs = dataset.filter_by_repository("flask")
    assert all(item.repository == "flask" for item in flask_qs)

    # Test filtering by category
    class_qs = dataset.filter_by_category("class")
    assert all(item.category == "class" for item in class_qs)


def test_qa_dataset_roundtrip_serialization(tmp_path: Path) -> None:
    """Verify serialization to JSON and JSONL preserves complete dataset fidelity."""
    dataset = load_qa_dataset(DATASET_PATH)

    tmp_json = tmp_path / "test_out.json"
    tmp_jsonl = tmp_path / "test_out.jsonl"

    dataset.save_json(tmp_json)
    dataset.save_jsonl(tmp_jsonl)

    assert tmp_json.exists()
    assert tmp_jsonl.exists()

    reloaded = load_qa_dataset(tmp_json)
    assert reloaded.total_questions == dataset.total_questions
    assert reloaded.dataset_name == dataset.dataset_name
    assert reloaded.version == dataset.version

    # Check JSONL line count
    lines = tmp_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 100


def test_validate_qa_dataset_detects_malformed_entries(tmp_path: Path) -> None:
    """Verify validate_qa_dataset flags missing files, bad line ranges, and invalid fields."""
    bad_sample = QuestionSample(
        id="",  # Empty ID
        repository="requests",
        category="invalid_category",  # Bad category
        scope="single_file",
        complexity="single_hop",
        hop_count=0,  # Invalid hop count
        elements_involved=[],
        question="",  # Empty question
        expected_answer="",  # Empty answer
        gold_evidence=[
            GoldEvidenceLocation(
                repository="requests",
                file_path="nonexistent/file.py",  # Non-existent file
                start_line=100,
                end_line=50,  # start > end
            )
        ],
    )

    bad_dataset = QAEvaluationDataset(
        dataset_name="Bad Dataset",
        version="0.0.1",
        description="Testing validation error detection",
        repositories=["requests"],
        questions=[bad_sample],
    )

    errors = validate_qa_dataset(bad_dataset, REPOS_ROOT)
    assert len(errors) >= 5
    assert any("Empty ID" in e for e in errors)
    assert any("Empty question" in e for e in errors)
    assert any("Invalid category" in e for e in errors)
    assert any("File not found" in e for e in errors)
    assert any("end_line (50) < start_line (100)" in e for e in errors)
