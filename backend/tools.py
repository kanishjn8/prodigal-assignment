import json
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from api_client import WRITABLE_FIELDS, APIError, VersionConflict
from decision_log import RecordDecision, ignore_decision
from outcomes import outcome_notes
from validators import (
    MANDATORY_FIELDS,
    analyze_candidate,
    compute_patch,
    is_candidate_complete,
    validate_fields,
)

TRACKING_FIELDS = {"collection_status", "attempt_count", "last_contacted_at", "notes"}
PROFILE_FIELDS = WRITABLE_FIELDS - TRACKING_FIELDS
CONFIRMATION_FIELDS = {"graduation_year", "available_from"}
PROFILE_PROPERTIES = {key: {"type": "string"} for key in sorted(PROFILE_FIELDS)}
PROFILE_PROPERTIES.update(
    {
        "cgpa": {"type": "number", "description": "JSON number on a 10-point scale, not a string."},
        "graduation_year": {"type": "integer"},
        "internship_duration_months": {"type": "integer"},
        "skills": {"type": "array", "items": {"type": "string"}},
    }
)
PROFILE_SCHEMA = {"type": "object", "properties": PROFILE_PROPERTIES, "additionalProperties": False}


class CandidateAPI(Protocol):
    def get_candidate(self, candidate_id: str) -> dict[str, Any]: ...

    def update_candidate(
        self,
        candidate_id: str,
        changes: dict[str, Any],
        version: int,
    ) -> dict[str, Any]: ...


@dataclass
class AgentSession:
    candidate_id: str
    original_candidate: dict[str, Any]
    latest_candidate: dict[str, Any]
    version: int
    missing_fields: list[str] = field(default_factory=list)
    invalid_fields: dict[str, str] = field(default_factory=dict)
    identity: str = "unconfirmed"
    confirmed_values: dict[str, Any] = field(default_factory=dict)
    validated_values: dict[str, Any] = field(default_factory=dict)
    confirmed_by_candidate: dict[str, Any] = field(default_factory=dict)
    validation_errors: dict[str, str] = field(default_factory=dict)
    refused_fields: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    candidate_messages: list[str] = field(default_factory=list)
    pending_clarification: dict[str, str] = field(default_factory=dict)
    conflict_fields: set[str] = field(default_factory=set)
    outcome: str | None = None
    ended: bool = False
    outcome_persisted: bool = False
    write_blocked: bool = False
    contacted_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    contact_recorded: bool = False
    last_error: str | None = None
    requested_outcome: dict[str, Any] | None = None
    disputed_fields: set[str] = field(default_factory=set)

    def refresh(self, record: dict[str, Any]) -> None:
        if record.get("id") != self.candidate_id:
            raise ValueError("API response does not match the session candidate.")
        self.latest_candidate = deepcopy(record)
        self.version = record["version"]
        analysis = analyze_candidate(record)
        self.missing_fields = analysis["missing"]
        self.invalid_fields = analysis["invalid"]
        self.pending_clarification = analysis["clarification"]


def _schema(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS = [
    _schema("inspect_candidate_state", "Inspect current gaps, valid values and session state.", {}),
    _schema(
        "validate_candidate_information",
        "Validate candidate-provided values without saving. Supply exact short quotes for each "
        "value from a candidate reply after identity confirmation.",
        {
            "information": PROFILE_SCHEMA,
            "evidence": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Same field names mapped to exact candidate quotes.",
            },
        },
    ),
    _schema(
        "save_candidate_information",
        "Save previously validated values; never change correct "
        "existing values. Check success before saying saved.",
        {
            "updates": PROFILE_SCHEMA,
        },
    ),
    _schema(
        "finish_collection",
        "Persist an outcome and end. completed requires valid mandatory "
        "data. Other outcomes need context. Refusal requires the declined field names.",
        {
            "collection_status": {
                "type": "string",
                "enum": ["completed", "in_progress", "no_answer", "refused", "needs_human"],
            },
            "notes": {"type": "string"},
            "refused_fields": {"type": "array", "items": {"type": "string"}},
        },
    ),
]


def _error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": code, "message": message}


class CandidateTools:
    def __init__(
        self,
        session: AgentSession,
        api: CandidateAPI,
        on_progress: Callable[[str], None] | None = None,
        on_decision: RecordDecision = ignore_decision,
    ) -> None:
        self.session = session
        self.api = api
        self.on_progress = on_progress or (lambda _: None)
        self.on_decision = on_decision

    def execute(self, name: str, arguments: Any) -> dict[str, Any]:
        result = self._execute(name, arguments)
        self.on_decision(
            "tool_result",
            tool=name
            if isinstance(name, str)
            and name in {schema["function"]["name"] for schema in TOOL_SCHEMAS}
            else "unknown",
            ok=result["ok"],
            error=result.get("error"),
        )
        return result

    def _execute(self, name: str, arguments: Any) -> dict[str, Any]:
        registry = {
            "inspect_candidate_state": self.inspect_candidate_state,
            "validate_candidate_information": self.validate_candidate_information,
            "save_candidate_information": self.save_candidate_information,
            "finish_collection": self.finish_collection,
        }
        if not isinstance(name, str) or name not in registry:
            return _error("unknown_tool", "Use only the registered tools.")
        if self.session.ended:
            return _error("session_ended", "The session has ended; no further tools may run.")
        try:
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Arguments must be a JSON object.")
            return registry[name](**arguments)
        except (TypeError, ValueError):
            return _error("invalid_arguments", "Arguments did not match the tool schema or rules.")
        except Exception:
            self.session.last_error = "tool_failure"
            if name in {"save_candidate_information", "finish_collection"}:
                self.session.write_blocked = True
            return _error("tool_failure", "Tool failed unexpectedly; request human review.")

    def inspect_candidate_state(self) -> dict[str, Any]:
        self.on_progress("Checking what’s still needed")
        session = self.session
        if session.identity != "confirmed":
            return {
                "ok": True,
                "identity": session.identity,
                "full_name": session.latest_candidate.get("full_name"),
            }
        return {
            "ok": True,
            "instruction": "Collect only missing or invalid fields and resolve clarification. "
            "Do not reconfirm existing valid fields.",
            "identity": session.identity,
            "version": session.version,
            "record": {
                key: value
                for key, value in session.latest_candidate.items()
                if key in PROFILE_FIELDS
            },
            "missing": session.missing_fields,
            "invalid": session.invalid_fields,
            "clarification": session.pending_clarification,
            "validated": session.validated_values,
            "errors": session.validation_errors,
            "conflict_fields": sorted(session.conflict_fields),
            "write_blocked": session.write_blocked,
        }

    def validate_candidate_information(
        self,
        information: dict[str, Any],
        evidence: dict[str, str],
    ) -> dict[str, Any]:
        session = self.session
        if session.ended:
            return _error("session_ended", "The session has ended.")
        if session.identity != "confirmed":
            return _error("identity_required", "Confirm identity by name before collecting data.")
        if not isinstance(information, dict) or not isinstance(evidence, dict):
            return _error("invalid_arguments", "Information and evidence must be objects.")
        if not information or not information.keys() <= PROFILE_FIELDS:
            return _error(
                "unsupported_fields", "Only writable candidate profile fields are allowed."
            )
        self.on_progress("Validating your answer")
        result = validate_fields(information)
        refused = set((session.requested_outcome or {}).get("refused_fields", []))
        for key in refused & information.keys():
            result["data"].pop(key, None)
            result["errors"][key] = "Candidate declined this field; do not collect or save it."
        for key in {"cgpa", "graduation_year", "internship_duration_months"} & information.keys():
            value = information[key]
            if isinstance(value, str) and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value):
                result["errors"][key] = (
                    "Extraction type error: retry this value as an unquoted JSON number "
                    "using the same evidence. Do not ask the candidate to clarify the scale "
                    "just because you supplied a string."
                )
        for key in information:
            session.validated_values.pop(key, None)
        for key in list(result["data"]):
            quote = evidence.get(key)
            if not self._has_evidence(key, information[key], quote):
                result["errors"][key] = "Quote the exact value from a confirmed candidate reply."
                del result["data"][key]
        session.validation_errors.update(result["errors"])
        self.on_decision(
            "validation_result", accepted=sorted(result["data"]), rejected=sorted(result["errors"])
        )
        for key in result["data"]:
            session.validation_errors.pop(key, None)
        session.validated_values.update(deepcopy(result["data"]))
        return {"ok": not result["errors"], "data": result["data"], "errors": result["errors"]}

    def _has_evidence(self, key: str, value: Any, quote: Any) -> bool:
        if (
            not isinstance(quote, str)
            or not quote.strip()
            or not self.session.candidate_messages
            or quote not in self.session.candidate_messages[-1]
        ):
            return False
        # Exact value quotes prevent the model from supplying unrelated invented values.
        source: Any = quote
        if key in {"cgpa", "graduation_year", "internship_duration_months"}:
            message = self.session.candidate_messages[-1]
            if re.search(
                r"\b(?:around|about|roughly|approximately|maybe)\s+" + re.escape(quote),
                message,
                re.I,
            ):
                return False
            if not re.search(r"(?<![\w.+-])" + re.escape(quote) + r"(?![\w.])", message):
                return False
            if key == "cgpa" and re.search(
                re.escape(quote) + r"\s*(?:%|percent|(?:/|out of)\s*(?!10\b)[0-9])",
                message,
                re.I,
            ):
                return False
            try:
                source = json.loads(quote)
            except ValueError:
                return False
        if key == "skills":
            return (
                isinstance(value, list)
                and bool(value)
                and all(isinstance(item, str) and item in quote for item in value)
            )
        source_result = validate_fields({key: source})
        value_result = validate_fields({key: value})
        return source_result["valid"] and source_result["data"] == value_result["data"]

    def save_candidate_information(self, updates: dict[str, Any]) -> dict[str, Any]:
        session = self.session
        if session.ended:
            return _error("session_ended", "The session has ended.")
        if session.identity != "confirmed":
            return _error("identity_required", "Confirm identity before saving candidate data.")
        if not isinstance(updates, dict) or not updates or not updates.keys() <= PROFILE_FIELDS:
            return _error("unsupported_fields", "Supply writable candidate profile fields only.")
        validated = validate_fields(updates)
        if updates.keys() & set((session.requested_outcome or {}).get("refused_fields", [])):
            return _error("refused_field", "Respect the candidate's refusal.")
        if not validated["valid"]:
            return {"ok": False, "error": "validation", "errors": validated["errors"]}
        for key, value in validated["data"].items():
            if key not in session.validated_values or session.validated_values[key] != value:
                return _error(
                    "validation_required", "Validate quoted candidate input before saving."
                )
            if key in CONFIRMATION_FIELDS and session.confirmed_by_candidate.get(key) != value:
                return _error(
                    "confirmation_required", "Read back the value and wait for confirmation."
                )
            if key in session.conflict_fields and session.latest_candidate.get(key) != value:
                return _error(
                    "conflict", "A concurrent edit changed this field; needs human review."
                )
            existing = session.latest_candidate.get(key)
            old_result = validate_fields({key: existing})
            if old_result["valid"] and existing != value and old_result["data"][key] != value:
                session.disputed_fields.add(key)
                self.on_decision("update_blocked", field=key, reason="existing_valid_value")
                session.validated_values.pop(key, None)
                return _error(
                    "existing_value", "Keep existing valid data; escalate disputed values."
                )
        patch = compute_patch(session.latest_candidate, validated["data"])
        if not patch:
            for key in updates:
                session.validated_values.pop(key, None)
            return {"ok": True, "saved": False, "changed_fields": []}
        result = self._write(
            {**patch, "collection_status": "in_progress", **self._contact_tracking()}
        )
        if result["ok"]:
            session.confirmed_values.update(deepcopy(validated["data"]))
            for key in updates:
                session.validated_values.pop(key, None)
        return result

    def _contact_tracking(self) -> dict[str, Any]:
        if self.session.contact_recorded:
            return {}
        count = self.session.latest_candidate.get("attempt_count", 0)
        if type(count) is not int or not 0 <= count <= 50:
            raise ValueError("Invalid existing attempt_count; needs human review.")
        return {"attempt_count": min(count + 1, 50), "last_contacted_at": self.session.contacted_at}

    def _write(self, changes: dict[str, Any]) -> dict[str, Any]:
        session = self.session
        if session.ended:
            return _error("session_ended", "The session has ended.")
        if changes.keys() & PROFILE_FIELDS and session.identity != "confirmed":
            return _error("identity_required", "Identity must be confirmed before profile writes.")
        if session.write_blocked:
            return _error(
                "write_blocked", "A previous write failed; reconcile outside this session."
            )
        patch = compute_patch(session.latest_candidate, changes)
        if not patch:
            return {"ok": True, "saved": False, "changed_fields": []}
        try:
            self.on_decision("patch_requested", fields=sorted(patch), version=session.version)
            self.on_progress("Updating the candidate database")
            updated = self.api.update_candidate(session.candidate_id, patch, session.version)
            session.refresh(updated)
        except KeyboardInterrupt:
            # An interrupted write may have reached the server; never retry it with a fresh key.
            session.write_blocked = True
            session.last_error = "interrupted_write"
            raise
        except VersionConflict as exc:
            self.on_decision("patch_conflict", refreshed=exc.latest_candidate is not None)
            session.validated_values.clear()
            if exc.latest_candidate is not None:
                latest = exc.latest_candidate
                session.conflict_fields.update(
                    key
                    for key in PROFILE_FIELDS
                    if latest.get(key) != session.latest_candidate.get(key)
                )
                session.disputed_fields.update(session.conflict_fields & changes.keys())
                session.refresh(latest)
            else:
                session.write_blocked = True
            return {
                "ok": False,
                "error": "conflict",
                "version": session.version,
                "message": "No write confirmed. Reinspect; do not repeat the stale update.",
                "conflict_fields": sorted(session.conflict_fields),
            }
        except APIError as exc:
            self.on_decision("patch_failed", reason=exc.kind, status_code=exc.status_code)
            session.write_blocked = True
            session.last_error = exc.kind
            return {
                "ok": False,
                "error": exc.kind,
                "saved": False,
                "message": "Save was not confirmed. Stop writes and request human review.",
            }
        except (TypeError, ValueError, KeyError):
            session.write_blocked = True
            return _error(
                "malformed_response", "Write result could not be verified; request review."
            )
        if "last_contacted_at" in patch:
            session.contact_recorded = True
        self.on_decision("patch_saved", fields=sorted(patch), version=session.version)
        return {
            "ok": True,
            "saved": True,
            "version": session.version,
            "changed_fields": list(patch),
            "updated": {key: updated.get(key) for key in patch},
        }

    def ready_to_complete(self) -> bool:
        return (
            is_candidate_complete(self.session.latest_candidate)
            and not self.session.disputed_fields
            and not set(self.session.validation_errors) & set(MANDATORY_FIELDS)
        )

    def finish_collection(
        self,
        collection_status: str,
        notes: str,
        refused_fields: list[str],
    ) -> dict[str, Any]:
        session = self.session
        if session.ended:
            return _error("session_ended", "The session has ended.")
        if collection_status not in {
            "completed",
            "in_progress",
            "no_answer",
            "refused",
            "needs_human",
        }:
            return _error("invalid_outcome", "Choose a supported terminal collection outcome.")
        if (
            not isinstance(notes, str)
            or not isinstance(refused_fields, list)
            or any(not isinstance(key, str) or key not in PROFILE_FIELDS for key in refused_fields)
        ):
            return _error("invalid_arguments", "Supply notes and valid refused field names.")
        if (
            session.requested_outcome
            and collection_status != session.requested_outcome["collection_status"]
        ):
            return _error("requested_outcome", "Respect the candidate's requested exit outcome.")
        if session.disputed_fields and collection_status != "needs_human":
            return _error("dispute", "Disputed valid information requires human review.")
        if collection_status == "completed":
            if session.identity != "confirmed":
                return _error("incomplete", "Identity and all mandatory fields must be valid.")
        elif not notes.strip():
            return _error("context_required", "Describe what happened and what remains.")
        if collection_status == "refused" and (
            session.identity != "confirmed" or not refused_fields
        ):
            return _error("refusal_required", "Confirm identity and identify refused fields.")
        if refused_fields and collection_status != "refused":
            return _error("invalid_outcome", "Refused fields require the refused outcome.")
        # Retain validated progress even if the model finishes before issuing its save call.
        pending = {
            key: value
            for key, value in session.validated_values.items()
            if key not in refused_fields
            and key not in session.disputed_fields
            and (key not in CONFIRMATION_FIELDS or session.confirmed_by_candidate.get(key) == value)
        }
        if pending and session.identity == "confirmed":
            saved = self.save_candidate_information(pending)
            if not saved["ok"]:
                return saved
        if collection_status == "no_answer" and session.confirmed_values:
            collection_status = "in_progress"
        if collection_status == "completed" and not self.ready_to_complete():
            return _error("incomplete", "Resolve all mandatory data and validation issues first.")
        outstanding = sorted(
            set(session.missing_fields)
            | set(session.invalid_fields)
            | set(session.pending_clarification)
            | (set(session.validation_errors) & set(MANDATORY_FIELDS))
            | session.disputed_fields
        )
        context = notes.strip()
        if outstanding:
            context += "\nOutstanding: " + ", ".join(outstanding)
        if refused_fields:
            context += "\nRefused: " + ", ".join(refused_fields)
        if session.requested_outcome:
            context += "\n" + session.requested_outcome["notes"]
        old_notes = session.latest_candidate.get("notes")
        if old_notes is None:
            old_notes = ""
        if not isinstance(old_notes, str):
            return _error("invalid_existing_notes", "Existing notes require human review.")
        combined_notes = outcome_notes(old_notes, context)
        result = self._write(
            {
                "collection_status": collection_status,
                "notes": combined_notes,
                **self._contact_tracking(),
            }
        )
        if result["ok"]:
            session.refused_fields = list(refused_fields)
            session.outcome = collection_status
            session.outcome_persisted = True
            session.ended = True
            self.on_decision("outcome_saved", status=collection_status)
        return result
