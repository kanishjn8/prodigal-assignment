import re
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, TypedDict
from urllib.parse import urlsplit

from api_client import COLLECTION_STATUSES, WRITABLE_FIELDS

MANDATORY_FIELDS = (
    "full_name",
    "phone",
    "email",
    "college",
    "degree",
    "branch",
    "graduation_year",
    "cgpa",
    "preferred_location",
    "available_from",
    "internship_duration_months",
)
DEGREES = ("B.Tech", "B.E.", "B.Sc", "BCA", "MBA Tech", "MCA", "M.Tech", "Other")
LOCATIONS = ("Mumbai", "Bengaluru", "Remote", "No preference")
ENUMS = {
    "degree": DEGREES,
    "preferred_location": LOCATIONS,
    "collection_status": tuple(COLLECTION_STATUSES),
}


class FieldResult(TypedDict):
    valid: bool
    value: Any
    error: str | None


class FieldsResult(TypedDict):
    valid: bool
    data: dict[str, Any]
    errors: dict[str, str]


def normalize_field(field_name: str, value: Any) -> Any:
    """Normalize unambiguous input only; this does not establish validity."""
    if isinstance(value, str) and field_name != "notes":
        value = value.strip()
    if field_name in ENUMS and isinstance(value, str):
        compact = re.sub(r"[.\s]", "", value).casefold()
        aliases = {re.sub(r"[.\s]", "", item).casefold(): item for item in ENUMS[field_name]}
        if field_name == "preferred_location":
            aliases["bangalore"] = "Bengaluru"
        return aliases.get(compact, value)
    if field_name in {"full_name", "college", "branch"} and isinstance(value, str):
        return " ".join(value.split())
    if field_name == "email" and isinstance(value, str) and value.count("@") == 1:
        local, domain = value.rsplit("@", 1)
        return f"{local}@{domain.lower()}"
    if field_name == "skills" and isinstance(value, list):
        return [item.strip() if isinstance(item, str) else item for item in value]
    return value


def _valid_email(value: str) -> bool:
    # Conservative dot-atom syntax; unusual quoted or international addresses need clarification.
    if len(value) > 254 or value.count("@") != 1:
        return False
    local, domain = value.rsplit("@", 1)
    atom = r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"
    if len(local) > 64 or not re.fullmatch(rf"{atom}(?:\.{atom})*", local):
        return False
    labels = domain.split(".")
    return (
        len(labels) >= 2
        and all(
            re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
            for label in labels
        )
        and not labels[-1].isdigit()
    )


def _valid_url(field_name: str, value: str) -> bool:
    if any(char.isspace() or ord(char) < 32 for char in value) or "\\" in value:
        return False
    try:
        parsed = urlsplit(value)
        domain = "linkedin.com" if field_name == "linkedin_url" else "github.com"
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {domain, f"www.{domain}"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
        ):
            return False
        if field_name == "github_url":
            username = parsed.path.removeprefix("/").removesuffix("/")
            return bool(
                re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", username)
                and "--" not in username
            )
        return True
    except ValueError:
        return False


def _validate_value(field_name: str, value: Any) -> Any:
    if field_name == "cgpa":
        if type(value) not in {int, float, Decimal}:
            raise ValueError("Use a number on the 10-point scale; clarify other scales.")
        number = Decimal(str(value))
        if not number.is_finite() or not 0 <= number <= 10:
            raise ValueError("CGPA must be finite and between 0 and 10; clarify other scales.")
        return float(number.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    if field_name in {"graduation_year", "internship_duration_months", "attempt_count"}:
        allowed = {
            "graduation_year": range(2026, 2029),
            "internship_duration_months": (2, 3, 6),
            "attempt_count": range(51),
        }
        if type(value) is not int or value not in allowed[field_name]:
            descriptions = {
                "graduation_year": "Graduation year must be 2026, 2027, or 2028.",
                "internship_duration_months": "Internship duration must be 2, 3, or 6 months.",
                "attempt_count": "Attempt count must be an integer from 0 through 50.",
            }
            raise ValueError(descriptions[field_name])
        return value
    if field_name == "skills":
        if (
            not isinstance(value, list)
            or len(value) > 30
            or any(not isinstance(item, str) for item in value)
        ):
            raise ValueError("Skills must be an array of at most 30 strings.")
        return value
    if not isinstance(value, str):
        raise ValueError("Expected a string.")
    if field_name == "notes":
        if len(value) > 4000:
            raise ValueError("Notes must be at most 4000 characters.")
        return value
    if not value:
        raise ValueError("A nonempty value is required; omit unknown optional fields.")
    if field_name in {"full_name", "college"}:
        maximum = 100 if field_name == "full_name" else 150
        if not 2 <= len(value) <= maximum:
            raise ValueError(f"Expected 2 to {maximum} characters.")
    elif field_name == "phone":
        if not re.fullmatch(r"\+[1-9][0-9]{1,14}", value):
            raise ValueError("Expected an E.164 phone number; this field is read-only.")
    elif field_name == "email":
        if not _valid_email(value):
            raise ValueError("Expected a valid email address; unusual syntax needs clarification.")
    elif field_name in ENUMS:
        if value not in ENUMS[field_name]:
            raise ValueError(f"Expected one of: {', '.join(ENUMS[field_name])}.")
    elif field_name == "available_from":
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            raise ValueError("Expected a calendar date in YYYY-MM-DD format.")
        parsed_date = date.fromisoformat(value)
        if not date(2026, 1, 1) <= parsed_date <= date(2027, 12, 31):
            raise ValueError("Date must be between 2026-01-01 and 2027-12-31 inclusive.")
    elif field_name == "last_contacted_at":
        if not re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}"
            r"(?::[0-9]{2}(?:\.[0-9]+)?)?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])",
            value,
        ):
            raise ValueError("Expected an ISO 8601 datetime with an explicit timezone offset.")
        datetime.fromisoformat(value)
    elif field_name in {"linkedin_url", "github_url"}:
        if not _valid_url(field_name, value):
            raise ValueError("Expected a valid URL on the exact service domain (GitHub: profile).")
    return value


def validate_field(field_name: str, value: Any) -> FieldResult:
    """Validate a single value, including read-only phone for record inspection."""
    if field_name not in WRITABLE_FIELDS | {"phone"}:
        return {"valid": False, "value": None, "error": "Unknown or unsupported field."}
    try:
        normalized = _validate_value(field_name, normalize_field(field_name, value))
    except ValueError as exc:
        return {"valid": False, "value": None, "error": str(exc)}
    return {"valid": True, "value": normalized, "error": None}


def validate_fields(proposed_data: dict[str, Any]) -> FieldsResult:
    """Validate a partial update; phone is inspectable but never writable."""
    if not isinstance(proposed_data, dict):
        raise ValueError("Proposed data must be a dictionary.")
    data, errors = {}, {}
    for field, value in proposed_data.items():
        if field not in WRITABLE_FIELDS:
            errors[field] = "Unknown or read-only field."
            continue
        result = validate_field(field, value)
        if result["valid"]:
            data[field] = result["value"]
        else:
            errors[field] = result["error"]
    return {"valid": not errors, "data": data, "errors": errors}


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def analyze_candidate(record: dict[str, Any]) -> dict[str, Any]:
    """Inspect without mutating historical data or treating normalization as confirmation."""
    missing, invalid, clarification, suggestions = [], {}, {}, {}
    for field in MANDATORY_FIELDS:
        value = record.get(field)
        if _missing(value):
            missing.append(field)
            continue
        result = validate_field(field, value)
        if not result["valid"]:
            invalid[field] = result["error"]
            clarification[field] = result["error"]
        elif result["value"] != value:
            clarification[field] = "Confirm the normalized value before changing existing data."
            suggestions[field] = result["value"]
            if field in ENUMS and value not in ENUMS[field]:
                invalid[field] = "Stored value is not a canonical enum value."
    other_errors = {}
    for field in WRITABLE_FIELDS - set(MANDATORY_FIELDS):
        if field in record and not _missing(record[field]):
            result = validate_field(field, record[field])
            if not result["valid"]:
                other_errors[field] = result["error"]
    complete = not missing and not invalid and not clarification
    return {
        "complete": complete,
        "status": "apparently_complete" if complete else "needs_information",
        "missing": missing,
        "invalid": invalid,
        "clarification": clarification,
        "normalization_suggestions": suggestions,
        "other_errors": other_errors,
        "needs_human": ["phone"] if "phone" in missing or "phone" in clarification else [],
    }


def is_candidate_complete(record: dict[str, Any]) -> bool:
    return analyze_candidate(record)["complete"]


def compute_patch(
    existing_record: dict[str, Any],
    validated_updates: dict[str, Any],
) -> dict[str, Any]:
    """Revalidate explicit, confirmed updates; never derive updates from existing data."""
    result = validate_fields(validated_updates)
    if not result["valid"]:
        raise ValueError(f"Invalid updates: {result['errors']}")
    patch = {}
    for field, value in result["data"].items():
        existing = existing_record.get(field)
        # Python considers True == 1; do not preserve an invalid boolean as a numeric value.
        existing_valid = validate_field(field, existing)["valid"]
        if field not in existing_record or not existing_valid or existing != value:
            patch[field] = value
    if result["data"].get("collection_status") == "completed" and not is_candidate_complete(
        {**existing_record, **patch}
    ):
        raise ValueError("Cannot mark an incomplete candidate as completed.")
    return patch
