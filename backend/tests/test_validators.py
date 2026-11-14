from copy import deepcopy
from decimal import Decimal
from typing import Any

import pytest

from api_client import COLLECTION_STATUSES
from validators import (
    DEGREES,
    LOCATIONS,
    MANDATORY_FIELDS,
    analyze_candidate,
    compute_patch,
    is_candidate_complete,
    normalize_field,
    validate_field,
    validate_fields,
)

COMPLETE = {
    "full_name": "Test Candidate",
    "phone": "+919876543210",
    "email": "test@example.com",
    "college": "Test University",
    "degree": "B.Tech",
    "branch": "Computer Engineering",
    "graduation_year": 2027,
    "cgpa": 8.25,
    "preferred_location": "Remote",
    "available_from": "2026-11-20",
    "internship_duration_months": 6,
    "id": "test-id",
    "version": 1,
    "updated_at": "2026-11-14T12:00:00+05:30",
}


@pytest.mark.parametrize("field", MANDATORY_FIELDS)
def test_valid_mandatory_fields(field: str) -> None:
    assert validate_field(field, COMPLETE[field]) == {
        "valid": True,
        "value": COMPLETE[field],
        "error": None,
    }


@pytest.mark.parametrize("field", MANDATORY_FIELDS)
@pytest.mark.parametrize("value", [None, "", "  ", [], {}, True])
def test_missing_and_wrong_types(field: str, value: Any) -> None:
    assert not validate_field(field, value)["valid"]


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("degree", DEGREES),
        ("preferred_location", LOCATIONS),
        ("collection_status", COLLECTION_STATUSES),
        ("internship_duration_months", (2, 3, 6)),
        ("graduation_year", (2026, 2027, 2028)),
        ("attempt_count", (0, 50)),
    ],
)
def test_all_permitted_values(field: str, values: Any) -> None:
    for value in values:
        assert validate_field(field, value)["valid"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("degree", "BBA"),
        ("degree", "engineering"),
        ("preferred_location", "Mumbai or Remote"),
        ("collection_status", "done"),
        ("internship_duration_months", 4),
        ("internship_duration_months", 3.0),
        ("graduation_year", 2031),
        ("graduation_year", "2027"),
        ("graduation_year", "next year"),
        ("attempt_count", -1),
        ("attempt_count", 51),
        ("attempt_count", True),
        ("attempt_count", 1.0),
        ("full_name", "A"),
        ("full_name", "A" * 101),
        ("college", "A"),
        ("college", "A" * 151),
        ("notes", "x" * 4001),
        ("notes", None),
        ("notes", 12),
        ("phone", "919876543210"),
        ("phone", "+0123456789"),
        ("phone", "+1234567890123456"),
        ("phone", "+91 9876543210"),
    ],
)
def test_invalid_constraints(field: str, value: Any) -> None:
    result = validate_field(field, value)
    assert not result["valid"] and result["error"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("full_name", "Ab"),
        ("full_name", "a" * 100),
        ("college", "Ab"),
        ("college", "a" * 150),
        ("branch", "X"),
        ("branch", "x" * 200),
        ("notes", ""),
        ("notes", "x" * 4000),
        ("phone", "+123456789012345"),
    ],
)
def test_string_boundaries(field: str, value: Any) -> None:
    assert validate_field(field, value)["valid"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0.0),
        (10, 10.0),
        (8.125, 8.13),
        (Decimal("8.124"), 8.12),
        (9.999, 10.0),
        (Decimal("0.005"), 0.01),
    ],
)
def test_cgpa_rounding(value: Any, expected: float) -> None:
    assert validate_field("cgpa", value)["value"] == expected


@pytest.mark.parametrize(
    "value",
    [
        -0.01,
        10.001,
        85,
        "8.5",
        "3.8/4",
        "85%",
        float("nan"),
        float("inf"),
        float("-inf"),
        Decimal("NaN"),
        Decimal("Infinity"),
        True,
    ],
)
def test_invalid_cgpa(value: Any) -> None:
    assert not validate_field("cgpa", value)["valid"]


@pytest.mark.parametrize("value", ["2026-01-01", "2027-12-31", "2026-02-28"])
def test_valid_dates(value: str) -> None:
    assert validate_field("available_from", value)["valid"]


@pytest.mark.parametrize(
    "value",
    [
        "2026-02-29",
        "2027-02-29",
        "2028-02-29",
        "2024-02-29",
        "2025-12-31",
        "2028-01-01",
        "2026-04-31",
        "2026-13-01",
        "2026-00-01",
        "2026-01-00",
        "2026-1-01",
        "20260101",
        "01/02/2026",
        "2026-01-01T00:00:00",
        20260101,
    ],
)
def test_invalid_or_out_of_range_dates(value: Any) -> None:
    assert not validate_field("available_from", value)["valid"]


@pytest.mark.parametrize(
    "value",
    [
        "2026-11-20T15:30:00+05:30",
        "2026-11-20T15:30:00Z",
        "2024-02-29T00:00:00-08:00",
        "2026-11-20T15:30:00.123456+00:00",
    ],
)
def test_timezone_datetimes(value: str) -> None:
    assert validate_field("last_contacted_at", value)["valid"]


@pytest.mark.parametrize(
    "value",
    [
        "2026-11-20T15:30:00",
        "2026-11-20",
        "2026-02-29T15:30:00Z",
        "2026-11-20T25:30:00Z",
        "2026-11-20T15:30:00+24:00",
        "2026-11-20T15:30:00+05:99",
        None,
        123,
    ],
)
def test_invalid_datetimes(value: Any) -> None:
    assert not validate_field("last_contacted_at", value)["valid"]


@pytest.mark.parametrize("value", ["a+b@example.co.in", "a.b@example.com", "O'Neil@example.com"])
def test_valid_emails(value: str) -> None:
    assert validate_field("email", value)["valid"]


@pytest.mark.parametrize(
    "value",
    [
        "abc@",
        "@example.com",
        "a@@example.com",
        "a b@example.com",
        "a..b@example.com",
        ".a@example.com",
        "a.@example.com",
        "a@-example.com",
        "a@example-.com",
        "a@localhost",
        "a@example..com",
        "a@exam_ple.com",
        "a@example.com\nBcc:other@example.com",
        "Name <a@example.com>",
        "a" * 65 + "@example.com",
    ],
)
def test_invalid_emails(value: str) -> None:
    assert not validate_field("email", value)["valid"]


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("degree", " btech ", "B.Tech"),
        ("degree", "b. e.", "B.E."),
        ("preferred_location", " Bangalore ", "Bengaluru"),
        ("preferred_location", "no   preference", "No preference"),
        ("full_name", " Test   Candidate ", "Test Candidate"),
        ("email", " Test@EXAMPLE.COM ", "Test@example.com"),
        ("college", "  Test College  ", "Test College"),
        ("skills", [" Python ", "SQL"], ["Python", "SQL"]),
        ("notes", "  keep spacing\n", "  keep spacing\n"),
    ],
)
def test_safe_normalization(field: str, value: Any, expected: Any) -> None:
    assert normalize_field(field, value) == expected
    assert validate_field(field, value)["value"] == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("degree", "engineering"),
        ("cgpa", "85%"),
        ("graduation_year", "next year"),
        ("preferred_location", "Mumbai/Bangalore"),
    ],
)
def test_ambiguous_values_not_invented(field: str, value: Any) -> None:
    assert normalize_field(field, value) == value
    assert not validate_field(field, value)["valid"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("linkedin_url", "https://www.linkedin.com/in/test-person/"),
        ("linkedin_url", "https://linkedin.com"),
        ("github_url", "https://github.com/test-person"),
        ("github_url", "http://www.github.com/test-person/"),
    ],
)
def test_valid_urls(field: str, value: str) -> None:
    assert validate_field(field, value)["valid"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("linkedin_url", "https://linkedin.com.evil.test/in/name"),
        ("linkedin_url", "https://evil.test/linkedin.com"),
        ("linkedin_url", "https://user@linkedin.com/in/name"),
        ("linkedin_url", "ftp://linkedin.com/in/name"),
        ("linkedin_url", "https://linkedin.com:bad/in/name"),
        ("linkedin_url", "https://linkedin.com/in/has space"),
        ("linkedin_url", "https://[broken"),
        ("github_url", "https://github.com/person/repository"),
        ("github_url", "https://github.com/"),
        ("github_url", "github.com/person"),
        ("github_url", "https://github.com/-person"),
        ("github_url", "https://github.com/person--name"),
        ("github_url", "https://github.com/%2fperson"),
    ],
)
def test_invalid_urls(field: str, value: str) -> None:
    assert not validate_field(field, value)["valid"]


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ([], True),
        (["x"] * 30, True),
        (["x"] * 31, False),
        (["Python", 3], False),
        ("Python, SQL", False),
        (("Python",), False),
        (None, False),
    ],
)
def test_skills(value: Any, valid: bool) -> None:
    assert validate_field("skills", value)["valid"] is valid


@pytest.mark.parametrize("field", ["unknown", "id", "version", "updated_at"])
def test_unknown_and_metadata_fields(field: str) -> None:
    assert not validate_field(field, "anything")["valid"]


def test_partial_validation_reports_each_error() -> None:
    result = validate_fields({"degree": "btech", "cgpa": 85, "email": "abc@"})
    assert not result["valid"]
    assert result["data"] == {"degree": "B.Tech"}
    assert set(result["errors"]) == {"cgpa", "email"}
    assert validate_fields({}) == {"valid": True, "data": {}, "errors": {}}


def test_complete_ignores_optional_and_tracking_errors() -> None:
    record = {**COMPLETE, "github_url": "bad", "skills": [3], "attempt_count": 90}
    analysis = analyze_candidate(record)
    assert analysis["complete"] and is_candidate_complete(record)
    assert analysis["status"] == "apparently_complete"
    assert set(analysis["other_errors"]) == {"github_url", "skills", "attempt_count"}


def test_mixed_missing_and_invalid_existing_data() -> None:
    record = {**COMPLETE, "email": "abc@", "cgpa": 85, "college": None, "branch": " "}
    original = deepcopy(record)
    analysis = analyze_candidate(record)
    assert set(analysis["missing"]) == {"college", "branch"}
    assert set(analysis["invalid"]) == {"email", "cgpa"}
    assert set(analysis["clarification"]) == {"email", "cgpa"}
    assert not analysis["complete"]
    assert record == original


def test_empty_record_missing_all_mandatory_fields() -> None:
    analysis = analyze_candidate({})
    assert set(analysis["missing"]) == set(MANDATORY_FIELDS)
    assert analysis["needs_human"] == ["phone"]
    assert not analysis["invalid"] and not is_candidate_complete({})


def test_existing_alias_needs_confirmation_without_mutation() -> None:
    record = {**COMPLETE, "preferred_location": "Bangalore"}
    analysis = analyze_candidate(record)
    assert analysis["normalization_suggestions"] == {"preferred_location": "Bengaluru"}
    assert "preferred_location" in analysis["invalid"]
    assert "preferred_location" in analysis["clarification"]
    assert not analysis["complete"]
    assert record["preferred_location"] == "Bangalore"
    assert compute_patch(record, {"notes": "Ask about location"}) == {
        "notes": "Ask about location",
    }


def test_invalid_read_only_phone_needs_human() -> None:
    assert analyze_candidate({**COMPLETE, "phone": "bad"})["needs_human"] == ["phone"]


def test_minimal_patch_preserves_existing_values() -> None:
    original = deepcopy(COMPLETE)
    assert compute_patch(COMPLETE, {"email": "test@example.com", "degree": "btech"}) == {}
    assert compute_patch(COMPLETE, {"degree": "btech", "cgpa": 9.5}) == {"cgpa": 9.5}
    assert compute_patch(COMPLETE, {"collection_status": "completed"}) == {
        "collection_status": "completed",
    }
    assert COMPLETE == original


@pytest.mark.parametrize("field", ["phone", "id", "version", "updated_at", "unknown"])
def test_read_only_updates_rejected(field: str) -> None:
    assert not validate_fields({field: COMPLETE.get(field)})["valid"]
    with pytest.raises(ValueError):
        compute_patch(COMPLETE, {field: COMPLETE.get(field)})


def test_invalid_patch_is_atomic() -> None:
    with pytest.raises(ValueError):
        compute_patch(COMPLETE, {"notes": "valid", "cgpa": 85})
    with pytest.raises(ValueError):
        compute_patch({}, {"collection_status": "completed"})
    assert compute_patch({}, {"collection_status": "in_progress", "attempt_count": 1}) == {
        "collection_status": "in_progress",
        "attempt_count": 1,
    }


def test_boolean_not_preserved_as_number() -> None:
    assert compute_patch({"cgpa": True}, {"cgpa": 1}) == {"cgpa": 1.0}


def test_minute_precision_datetime() -> None:
    assert validate_field("last_contacted_at", "2026-11-20T15:30+05:30")["valid"]


def test_existing_completed_status_does_not_bypass_validation() -> None:
    with pytest.raises(ValueError):
        compute_patch({"collection_status": "completed"}, {"collection_status": "completed"})


@pytest.mark.parametrize("field", MANDATORY_FIELDS)
def test_every_mandatory_field_blocks_completion_when_absent(field: str) -> None:
    record = {key: value for key, value in COMPLETE.items() if key != field}
    assert not is_candidate_complete(record)
    assert analyze_candidate(record)["missing"] == [field]


def test_completion_with_full_validated_updates() -> None:
    updates = {
        field: value
        for field, value in COMPLETE.items()
        if field not in {"id", "phone", "version", "updated_at"}
    }
    patch = compute_patch(
        {"phone": COMPLETE["phone"]}, {**updates, "collection_status": "completed"}
    )
    assert is_candidate_complete({"phone": COMPLETE["phone"], **patch})
