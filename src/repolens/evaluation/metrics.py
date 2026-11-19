"""Information retrieval, answer correctness, and efficiency metrics for RepoLens evaluation."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


def normalize_path(path: str | None) -> str:
    """Normalize file path for cross-platform matching."""
    if not path:
        return ""
    # Standardize forward slashes and lower-case across all OS platforms
    normalized = str(path).replace("\\", "/").lower().strip()
    # Strip leading "./" if present
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def ranges_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    """Determine whether two 1-indexed line spans intersect."""
    return max(start_a, start_b) <= min(end_a, end_b)


def evidence_matches(retrieved: Any, gold: Any) -> bool:
    """Check whether a retrieved code element matches a ground-truth gold evidence location."""
    # Extract file path
    ret_path = ""
    gold_path = ""
    if hasattr(retrieved, "file_path"):
        ret_path = retrieved.file_path or ""
    elif isinstance(retrieved, dict):
        ret_path = retrieved.get("file_path") or ""

    if hasattr(gold, "file_path"):
        gold_path = gold.file_path or ""
    elif isinstance(gold, dict):
        gold_path = gold.get("file_path") or ""

    norm_ret = normalize_path(ret_path)
    norm_gold = normalize_path(gold_path)

    # Check path ending (to handle relative paths e.g. 'src/requests/api.py' vs 'api.py')
    path_matches = (
        norm_ret == norm_gold
        or norm_ret.endswith(norm_gold)
        or norm_gold.endswith(norm_ret)
    )
    if not path_matches:
        return False

    # Extract lines
    ret_start = None
    ret_end = None
    if hasattr(retrieved, "start_line"):
        ret_start = retrieved.start_line
        ret_end = retrieved.end_line
    elif isinstance(retrieved, dict):
        ret_start = retrieved.get("start_line")
        ret_end = retrieved.get("end_line")

    gold_start = None
    gold_end = None
    if hasattr(gold, "start_line"):
        gold_start = gold.start_line
        gold_end = gold.end_line
    elif isinstance(gold, dict):
        gold_start = gold.get("start_line")
        gold_end = gold.get("end_line")

    # If both define line numbers, verify overlap
    if (
        isinstance(ret_start, int)
        and isinstance(ret_end, int)
        and isinstance(gold_start, int)
        and isinstance(gold_end, int)
        and ranges_overlap(ret_start, ret_end, gold_start, gold_end)
    ):
        return True

    # Check symbol match if lines don't overlap or are missing
    ret_sym = ""
    gold_sym = ""
    if hasattr(retrieved, "symbol"):
        ret_sym = retrieved.symbol or ""
    elif isinstance(retrieved, dict):
        ret_sym = (
            retrieved.get("symbol")
            or retrieved.get("qualified_name")
            or retrieved.get("name")
            or ""
        )

    if hasattr(gold, "symbol"):
        gold_sym = gold.symbol or ""
    elif isinstance(gold, dict):
        gold_sym = gold.get("symbol") or ""

    if ret_sym and gold_sym:
        norm_ret_sym = ret_sym.strip().lower()
        norm_gold_sym = gold_sym.strip().lower()
        if (
            norm_ret_sym == norm_gold_sym
            or norm_ret_sym.endswith(f".{norm_gold_sym}")
            or norm_gold_sym.endswith(f".{norm_ret_sym}")
        ):
            return True

    # If same file and either lines overlapped or lines were unconstrained
    return (ret_start is None or gold_start is None) and path_matches


# =============================================================================
# Retrieval Quality Metrics (Recall@K, Precision@K, MRR, nDCG)
# =============================================================================


def compute_recall_at_k(
    retrieved: Sequence[Any],
    gold: Sequence[Any],
    k: int = 5,
) -> float:
    """Compute Recall@K: fraction of gold evidence locations found in top-K retrieved items."""
    if not gold:
        return 1.0
    if not retrieved or k <= 0:
        return 0.0

    top_k = retrieved[:k]
    matched_gold_count = 0

    for g in gold:
        if any(evidence_matches(r, g) for r in top_k):
            matched_gold_count += 1

    return round(matched_gold_count / len(gold), 4)


def compute_precision_at_k(
    retrieved: Sequence[Any],
    gold: Sequence[Any],
    k: int = 5,
) -> float:
    """Compute Precision@K: fraction of top-K retrieved items that match gold evidence."""
    if not retrieved or k <= 0:
        return 0.0
    if not gold:
        return 0.0

    top_k = retrieved[:k]
    matched_retrieved_count = sum(
        1 for r in top_k if any(evidence_matches(r, g) for g in gold)
    )

    return round(matched_retrieved_count / len(top_k), 4)


def compute_mrr(
    retrieved: Sequence[Any],
    gold: Sequence[Any],
    k: int = 10,
) -> float:
    """Compute Mean Reciprocal Rank (MRR): 1 / rank of the first relevant retrieved item."""
    if not retrieved or not gold or k <= 0:
        return 0.0

    top_k = retrieved[:k]
    for rank, r in enumerate(top_k, start=1):
        if any(evidence_matches(r, g) for g in gold):
            return round(1.0 / rank, 4)

    return 0.0


def compute_ndcg_at_k(
    retrieved: Sequence[Any],
    gold: Sequence[Any],
    k: int = 5,
    graded: bool = False,
) -> float:
    """Compute Normalized Discounted Cumulative Gain (nDCG@K).

    When graded is True, primary_evidence has relevance 2, other roles have 1.
    When graded is False, binary relevance (1 or 0) is used.
    """
    if not retrieved or not gold or k <= 0:
        return 0.0

    top_k = retrieved[:k]

    def get_relevance(item: Any) -> float:
        for g in gold:
            if evidence_matches(item, g):
                if graded:
                    role = getattr(g, "role", None) or (
                        g.get("role") if isinstance(g, dict) else None
                    )
                    return 2.0 if role == "primary_evidence" else 1.0
                return 1.0
        return 0.0

    # Calculate DCG@K
    dcg = 0.0
    for idx, item in enumerate(top_k):
        rel = get_relevance(item)
        if rel > 0:
            dcg += (math.pow(2.0, rel) - 1.0) / math.log2(idx + 2)

    # Calculate Ideal DCG (IDCG@K)
    ideal_rels: list[float] = []
    for g in gold:
        if graded:
            role = getattr(g, "role", None) or (
                g.get("role") if isinstance(g, dict) else None
            )
            ideal_rels.append(2.0 if role == "primary_evidence" else 1.0)
        else:
            ideal_rels.append(1.0)

    ideal_rels.sort(reverse=True)
    ideal_top_k = ideal_rels[:k]

    idcg = 0.0
    for idx, rel in enumerate(ideal_top_k):
        if rel > 0:
            idcg += (math.pow(2.0, rel) - 1.0) / math.log2(idx + 2)

    if idcg <= 0.0:
        return 0.0

    return round(dcg / idcg, 4)


def compute_evidence_coverage(
    retrieved_or_cited: Sequence[Any],
    gold: Sequence[Any],
) -> float:
    """Compute proportion of gold evidence items covered by retrieved or cited items."""
    return compute_recall_at_k(retrieved_or_cited, gold, k=len(retrieved_or_cited))


def compute_citation_precision(
    citations: Sequence[Any],
    gold: Sequence[Any],
) -> float:
    """Compute proportion of generated citations that match true gold evidence locations."""
    if not citations:
        return 1.0  # Zero hallucinated citations
    if not gold:
        return 0.0

    valid_matches = sum(
        1 for c in citations if any(evidence_matches(c, g) for g in gold)
    )
    return round(valid_matches / len(citations), 4)


# =============================================================================
# Answer Correctness Evaluation
# =============================================================================


@dataclass(frozen=True)
class AnswerCorrectnessResult:
    """Detailed score result of comparing generated answer against expected answer."""

    token_precision: float
    token_recall: float
    token_f1: float
    keyword_coverage: float
    overall_correctness: float

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "token_precision": self.token_precision,
            "token_recall": self.token_recall,
            "token_f1": self.token_f1,
            "keyword_coverage": self.keyword_coverage,
            "overall_correctness": self.overall_correctness,
        }


def compute_answer_correctness(
    generated_answer: str,
    expected_answer: str,
    keywords: Sequence[str] | None = None,
) -> AnswerCorrectnessResult:
    """Evaluate factual alignment and lexical/semantic similarity against expected answer."""
    if not generated_answer or not generated_answer.strip():
        return AnswerCorrectnessResult(
            token_precision=0.0,
            token_recall=0.0,
            token_f1=0.0,
            keyword_coverage=0.0,
            overall_correctness=0.0,
        )

    if not expected_answer or not expected_answer.strip():
        return AnswerCorrectnessResult(
            token_precision=1.0,
            token_recall=1.0,
            token_f1=1.0,
            keyword_coverage=1.0,
            overall_correctness=1.0,
        )

    gen_tokens = _TOKEN_PATTERN.findall(generated_answer.lower())
    exp_tokens = _TOKEN_PATTERN.findall(expected_answer.lower())

    if not gen_tokens or not exp_tokens:
        return AnswerCorrectnessResult(
            token_precision=0.0,
            token_recall=0.0,
            token_f1=0.0,
            keyword_coverage=0.0,
            overall_correctness=0.0,
        )

    gen_counts = Counter(gen_tokens)
    exp_counts = Counter(exp_tokens)

    intersection = sum(
        min(gen_counts[t], exp_counts[t]) for t in gen_counts if t in exp_counts
    )

    precision = intersection / len(gen_tokens)
    recall = intersection / len(exp_tokens)
    f1 = (
        (2.0 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Keyword coverage
    kw_coverage = 1.0
    if keywords:
        found_count = 0
        gen_lower = generated_answer.lower()
        for kw in keywords:
            if kw.strip().lower() in gen_lower:
                found_count += 1
        kw_coverage = found_count / len(keywords)

    # Overall correctness score: balanced blend of Token F1 and Keyword Coverage
    overall = round(0.5 * f1 + 0.5 * kw_coverage, 4)

    return AnswerCorrectnessResult(
        token_precision=round(precision, 4),
        token_recall=round(recall, 4),
        token_f1=round(f1, 4),
        keyword_coverage=round(kw_coverage, 4),
        overall_correctness=overall,
    )


# =============================================================================
# Computational Efficiency & Cost Estimation
# =============================================================================


def estimate_token_count(text: str | None) -> int:
    """Estimate token count for a text string using standard ~4 characters per token heuristic."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4.0))


def estimate_inference_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_cost_per_million: float = 0.15,
    output_cost_per_million: float = 0.60,
) -> float:
    """Estimate USD inference cost given prompt and completion token counts."""
    input_cost = (prompt_tokens / 1_000_000.0) * input_cost_per_million
    output_cost = (completion_tokens / 1_000_000.0) * output_cost_per_million
    return round(input_cost + output_cost, 6)
