from repolens.verification.claim_validator import ClaimValidator


def sample_evidence():
    return [
        {
            "evidence_id": "ev-001",
            "name": "authenticate_user",
            "qualified_name": "auth.authenticate_user",
            "kind": "function",
            "file_path": "src/auth.py",
            "content": (
                "The authenticate_user function validates user credentials "
                "and returns an authentication result."
            ),
            "metadata": {"docstring": "Validates user credentials."},
        },
        {
            "evidence_id": "ev-002",
            "name": "create_session",
            "qualified_name": "auth.create_session",
            "kind": "function",
            "file_path": "src/session.py",
            "content": "Creates a session for an authenticated user.",
            "metadata": {},
        },
    ]


def test_claim_with_matching_evidence_is_valid():
    validator = ClaimValidator(minimum_overlap=0.5)

    result = validator.validate(
        "The authenticate_user function validates user credentials.",
        sample_evidence(),
    )

    assert result.status == "valid"
    assert "ev-001" in result.supporting_evidence_ids
    assert any(issue.code == "lexical_support_only" for issue in result.issues)


def test_claim_with_no_evidence_is_unverified():
    validator = ClaimValidator()

    result = validator.validate(
        "The database uses encrypted passwords.",
        [],
    )

    assert result.status == "unverified"
    assert any(issue.code == "no_supporting_evidence" for issue in result.issues)


def test_missing_evidence_reference_is_invalid():
    validator = ClaimValidator()

    result = validator.validate(
        "The authentication function validates credentials.",
        sample_evidence(),
        supporting_evidence_ids=["missing-id"],
    )

    assert result.status == "invalid"
    assert any(issue.code == "missing_evidence_reference" for issue in result.issues)


def test_valid_evidence_reference_is_checked():
    validator = ClaimValidator()

    result = validator.validate(
        "The authenticate_user function validates user credentials.",
        sample_evidence(),
        supporting_evidence_ids=["ev-001"],
    )

    assert result.status == "valid"
    assert result.supporting_evidence_ids == ("ev-001",)


def test_weak_overlap_is_unverified():
    validator = ClaimValidator(minimum_overlap=0.8)

    result = validator.validate(
        "The payment processor encrypts financial transactions.",
        sample_evidence(),
    )

    assert result.status == "unverified"
    assert any(issue.code == "weak_evidence_overlap" for issue in result.issues)


def test_empty_claim_is_incomplete():
    validator = ClaimValidator()

    result = validator.validate("", sample_evidence())

    assert result.status == "incomplete"
    assert any(issue.code == "empty_claim" for issue in result.issues)


def test_claim_without_meaningful_terms_is_incomplete():
    validator = ClaimValidator()

    result = validator.validate("the and is", sample_evidence())

    assert result.status == "incomplete"


def test_validate_many_returns_one_result_per_claim():
    validator = ClaimValidator()

    results = validator.validate_many(
        [
            "The authenticate_user function validates user credentials.",
            "The database encrypts financial transactions.",
        ],
        sample_evidence(),
    )

    assert len(results) == 2
    assert results[0].status == "valid"
    assert results[1].status == "unverified"


def test_invalid_overlap_threshold_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        ClaimValidator(minimum_overlap=1.5)


def test_non_string_claim_is_rejected():
    import pytest

    validator = ClaimValidator()

    with pytest.raises(TypeError):
        validator.validate(None, sample_evidence())
