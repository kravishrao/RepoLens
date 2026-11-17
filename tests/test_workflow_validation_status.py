from types import SimpleNamespace

import pytest

from repolens.agent.workflow import _get_overall_validation_status


@pytest.mark.parametrize(
    ("citation_status", "claim_status", "expected"),
    [
        ("valid", "valid", "valid"),
        ("valid", "invalid", "invalid"),
        ("valid", "incomplete", "incomplete"),
        ("valid", "unverified", "unverified"),
        ("invalid", "valid", "invalid"),
    ],
)
def test_overall_status_combines_citation_and_claim_results(
    citation_status: str,
    claim_status: str,
    expected: str,
) -> None:
    citation_results = [SimpleNamespace(status=citation_status)]
    claim_results = [SimpleNamespace(status=claim_status)]

    result = _get_overall_validation_status(
        citation_results,
        claim_results,
    )

    assert result == expected


def test_overall_status_is_incomplete_when_no_citations_exist() -> None:
    claim_results = [SimpleNamespace(status="valid")]

    result = _get_overall_validation_status([], claim_results)

    assert result == "incomplete"
