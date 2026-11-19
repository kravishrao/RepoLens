"""Schema, view-models, and utilities for the RepoLens QA evaluation dataset."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_CATEGORIES = frozenset(
    {
        "function",
        "class",
        "module",
        "import",
        "dependency",
        "call_graph",
        "workflow",
    }
)

VALID_SCOPES = frozenset({"single_file", "multi_file"})
VALID_COMPLEXITIES = frozenset({"single_hop", "multi_hop"})


@dataclass(frozen=True)
class GoldEvidenceLocation:
    """Represents a ground-truth source code location supporting an answer."""

    repository: str
    file_path: str
    start_line: int
    end_line: int
    symbol: str | None = None
    role: str = "primary_evidence"

    def to_dict(self) -> dict[str, Any]:
        """Convert gold evidence location to dictionary representation."""
        data: dict[str, Any] = {
            "repository": self.repository,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "role": self.role,
        }
        if self.symbol is not None:
            data["symbol"] = self.symbol
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GoldEvidenceLocation:
        """Instantiate gold evidence location from dictionary."""
        return cls(
            repository=str(data["repository"]),
            file_path=str(data["file_path"]),
            start_line=int(data["start_line"]),
            end_line=int(data["end_line"]),
            symbol=str(data["symbol"]) if data.get("symbol") else None,
            role=str(data.get("role", "primary_evidence")),
        )


@dataclass(frozen=True)
class QuestionSample:
    """A structured question-answering evaluation sample."""

    id: str
    repository: str
    category: str
    scope: str
    complexity: str
    hop_count: int
    elements_involved: list[str]
    question: str
    expected_answer: str
    gold_evidence: list[GoldEvidenceLocation]
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert sample to machine-readable dictionary representation."""
        return {
            "id": self.id,
            "repository": self.repository,
            "category": self.category,
            "scope": self.scope,
            "complexity": self.complexity,
            "hop_count": self.hop_count,
            "elements_involved": list(self.elements_involved),
            "question": self.question,
            "expected_answer": self.expected_answer,
            "gold_evidence": [ev.to_dict() for ev in self.gold_evidence],
            "keywords": list(self.keywords),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QuestionSample:
        """Instantiate question sample from dictionary."""
        evidence_list = [
            GoldEvidenceLocation.from_dict(ev) for ev in data.get("gold_evidence", [])
        ]
        return cls(
            id=str(data["id"]),
            repository=str(data["repository"]),
            category=str(data["category"]),
            scope=str(data["scope"]),
            complexity=str(data["complexity"]),
            hop_count=int(data.get("hop_count", 1)),
            elements_involved=list(data.get("elements_involved", [])),
            question=str(data["question"]),
            expected_answer=str(data["expected_answer"]),
            gold_evidence=evidence_list,
            keywords=list(data.get("keywords", [])),
        )


@dataclass
class QAEvaluationDataset:
    """Collection of structured evaluation questions with metadata and queries."""

    dataset_name: str
    version: str
    description: str
    repositories: list[str]
    questions: list[QuestionSample] = field(default_factory=list)

    @property
    def total_questions(self) -> int:
        """Total number of question samples in the dataset."""
        return len(self.questions)

    def get_question(self, question_id: str) -> QuestionSample | None:
        """Look up a question sample by unique ID."""
        for q in self.questions:
            if q.id == question_id:
                return q
        return None

    def filter_by_repository(self, repository: str) -> list[QuestionSample]:
        """Return questions corresponding to a specific repository."""
        return [q for q in self.questions if q.repository == repository]

    def filter_by_category(self, category: str) -> list[QuestionSample]:
        """Return questions matching a given category."""
        return [q for q in self.questions if q.category == category]

    def filter_by_complexity(self, complexity: str) -> list[QuestionSample]:
        """Return questions matching complexity (single_hop or multi_hop)."""
        return [q for q in self.questions if q.complexity == complexity]

    def filter_by_scope(self, scope: str) -> list[QuestionSample]:
        """Return questions matching scope (single_file or multi_file)."""
        return [q for q in self.questions if q.scope == scope]

    def to_dict(self) -> dict[str, Any]:
        """Convert entire dataset to a JSON-serializable dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "version": self.version,
            "description": self.description,
            "repositories": list(self.repositories),
            "total_questions": self.total_questions,
            "questions": [q.to_dict() for q in self.questions],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QAEvaluationDataset:
        """Instantiate dataset from dictionary."""
        questions = [QuestionSample.from_dict(q) for q in data.get("questions", [])]
        return cls(
            dataset_name=str(
                data.get("dataset_name", "RepoLens QA Evaluation Dataset")
            ),
            version=str(data.get("version", "1.0.0")),
            description=str(data.get("description", "")),
            repositories=list(data.get("repositories", [])),
            questions=questions,
        )

    def save_json(self, file_path: Path | str) -> None:
        """Serialize dataset to indented JSON file."""
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            f.write("\n")

    def save_jsonl(self, file_path: Path | str) -> None:
        """Serialize question samples to line-delimited JSON (JSONL)."""
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            for q in self.questions:
                f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")


def load_qa_dataset(path: Path | str | None = None) -> QAEvaluationDataset:
    """Load QAEvaluationDataset from disk.

    Defaults to 'dataset/evaluation/qa_dataset.json' relative to workspace root.
    """
    if path is None:
        # Search relative to module location or current working directory
        candidates = [
            Path("dataset/evaluation/qa_dataset.json"),
            Path(__file__).resolve().parents[3]
            / "dataset"
            / "evaluation"
            / "qa_dataset.json",
        ]
        found: Path | None = None
        for cand in candidates:
            if cand.exists():
                found = cand
                break
        if found is None:
            raise FileNotFoundError(
                "Could not find default qa_dataset.json. Please specify path explicitly."
            )
        resolved_path = found
    else:
        resolved_path = Path(path)
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Evaluation dataset file not found at: {resolved_path}"
            )

    with open(resolved_path, encoding="utf-8") as f:
        data = json.load(f)
    return QAEvaluationDataset.from_dict(data)


def validate_qa_dataset(
    dataset: QAEvaluationDataset,
    repos_root: Path | str | None = None,
) -> list[str]:
    """Validate dataset consistency and verify all gold evidence lines exist on disk.

    Returns:
        list[str]: List of error messages. Empty list indicates 100% valid dataset.
    """
    errors: list[str] = []

    if repos_root is None:
        candidates = [
            Path("dataset/repositories"),
            Path(__file__).resolve().parents[3] / "dataset" / "repositories",
        ]
        repos_path = next(
            (c for c in candidates if c.exists()), Path("dataset/repositories")
        )
    else:
        repos_path = Path(repos_root)

    seen_ids: set[str] = set()

    for idx, q in enumerate(dataset.questions, start=1):
        # ID validation
        if not q.id or not q.id.strip():
            errors.append(f"Question #{idx}: Empty ID")
        elif q.id in seen_ids:
            errors.append(f"Question #{idx}: Duplicate ID '{q.id}'")
        else:
            seen_ids.add(q.id)

        # Question & Answer validation
        if not q.question or not q.question.strip():
            errors.append(f"[{q.id}]: Empty question text")
        if not q.expected_answer or not q.expected_answer.strip():
            errors.append(f"[{q.id}]: Empty expected_answer")

        # Category, scope, complexity
        if q.category not in VALID_CATEGORIES:
            errors.append(f"[{q.id}]: Invalid category '{q.category}'")
        if q.scope not in VALID_SCOPES:
            errors.append(f"[{q.id}]: Invalid scope '{q.scope}'")
        if q.complexity not in VALID_COMPLEXITIES:
            errors.append(f"[{q.id}]: Invalid complexity '{q.complexity}'")
        if q.hop_count < 1:
            errors.append(f"[{q.id}]: hop_count must be >= 1, got {q.hop_count}")

        # Gold evidence validation
        if not q.gold_evidence:
            errors.append(f"[{q.id}]: gold_evidence list is empty")
        else:
            for e_idx, ev in enumerate(q.gold_evidence, start=1):
                if ev.start_line < 1:
                    errors.append(
                        f"[{q.id}] Evidence #{e_idx}: start_line must be >= 1, got {ev.start_line}"
                    )
                if ev.end_line < ev.start_line:
                    errors.append(
                        f"[{q.id}] Evidence #{e_idx}: end_line ({ev.end_line}) < start_line ({ev.start_line})"
                    )

                # Disk verification if repository snapshot is available
                target_file = repos_path / ev.repository / ev.file_path
                if not target_file.exists():
                    errors.append(
                        f"[{q.id}] Evidence #{e_idx}: File not found: {target_file}"
                    )
                else:
                    try:
                        content = target_file.read_text(encoding="utf-8")
                        total_lines = len(content.splitlines())
                        if ev.end_line > total_lines:
                            errors.append(
                                f"[{q.id}] Evidence #{e_idx}: end_line {ev.end_line} exceeds file length {total_lines} in {target_file}"
                            )
                    except Exception as exc:
                        errors.append(
                            f"[{q.id}] Evidence #{e_idx}: Failed to read {target_file}: {exc}"
                        )

    return errors
