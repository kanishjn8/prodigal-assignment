import json
from copy import deepcopy
from datetime import datetime
from typing import Any

import httpx
import pytest
from test_agent import RECORD, FakeAPI, FakeLLM, calls, confirmed_agent, text, validate_email
from test_cli import setup_cli  # noqa: F401

import main
from agent import CandidateAgent
from api_client import APIError, CandidateAPIClient
from outcomes import requested_exit
from validators import is_candidate_complete


def finish(agent: CandidateAgent, status: str, notes: str = "Test outcome") -> dict[str, Any]:
    return agent.tools.execute(
        "finish_collection",
        {
            "collection_status": status,
            "notes": notes,
            "refused_fields": [],
        },
    )


@pytest.mark.parametrize("exit_input", ["/no-answer", "/quit", EOFError()])
def test_no_useful_reply_records_no_answer(setup_cli, exit_input: Any) -> None:  # noqa: F811
    api, llm = setup_cli(inputs=[exit_input])
    assert main.main(["--candidate-id", "c-test"]) == 0
    assert api.record["collection_status"] == "no_answer"
    assert api.record["attempt_count"] == 1
    assert datetime.fromisoformat(api.record["last_contacted_at"]).utcoffset() is not None
    assert "Outstanding: email" in api.record["notes"]
    assert api.record["email"] is None and not llm.requests


def test_quit_after_partial_save_preserves_progress(setup_cli) -> None:  # noqa: F811
    api, _ = setup_cli(
        record={**RECORD, "cgpa": None},
        inputs=["yes", "test@example.com", "/quit"],
        responses=[
            calls(("inspect_candidate_state", {})),
            text(),
            calls(
                validate_email(),
                ("save_candidate_information", {"updates": {"email": "test@example.com"}}),
            ),
            text("What is your official CGPA?"),
        ],
    )
    assert main.main(["--candidate-id", "c-test"]) == 0
    assert api.record["collection_status"] == "in_progress"
    assert api.record["email"] == "test@example.com"
    assert api.record["cgpa"] is None
    assert "cgpa" in api.record["notes"] and api.record["attempt_count"] == 1


def test_continue_later_preserves_requested_time_and_unsaved_validated_data() -> None:
    agent = confirmed_agent(
        calls(validate_email()),
        calls(
            (
                "finish_collection",
                {
                    "collection_status": "in_progress",
                    "notes": "Continue later",
                    "refused_fields": [],
                },
            )
        ),
        api=FakeAPI({**RECORD, "cgpa": None}),
    )
    response = agent.respond("My email is test@example.com. Can we continue tomorrow at 6 PM?")
    assert agent.session.outcome == "in_progress"
    assert agent.session.latest_candidate["email"] == "test@example.com"
    assert "tomorrow at 6 PM" in agent.session.latest_candidate["notes"]
    assert "Outstanding: cgpa" in agent.session.latest_candidate["notes"]
    assert "saved" in response
    assert agent.session.latest_candidate["attempt_count"] == 1


def test_refusal_stops_without_pressuring_and_keeps_other_valid_data() -> None:
    agent = confirmed_agent(
        calls(validate_email()),
        calls(
            (
                "finish_collection",
                {
                    "collection_status": "refused",
                    "notes": "CGPA declined",
                    "refused_fields": ["cgpa"],
                },
            )
        ),
        api=FakeAPI({**RECORD, "cgpa": None}),
    )
    agent.respond("I won't share my CGPA but my email is test@example.com")
    assert agent.session.outcome == "refused" and agent.session.ended
    assert agent.session.latest_candidate["email"] == "test@example.com"
    assert agent.session.latest_candidate["cgpa"] is None
    assert "Refused: cgpa" in agent.session.latest_candidate["notes"]
    assert agent.respond("anything else") is None


@pytest.mark.parametrize(
    "message",
    [
        "This is not my record",
        "I want to talk to a human",
        "I'm upset",
        "This is frustrating",
    ],
)
def test_human_requests_stop_without_model_collection(message: str) -> None:
    agent = confirmed_agent()
    before = len(agent.llm.requests)
    agent.respond(message)
    assert agent.session.outcome == "needs_human" and agent.session.outcome_persisted
    assert len(agent.llm.requests) == before
    assert not agent.session.confirmed_values


def test_unverified_deferral_and_wrong_identity() -> None:
    for message, status in [
        ("Can we continue later tomorrow?", "in_progress"),
        ("This belongs to someone else's application", "needs_human"),
    ]:
        api, llm = FakeAPI(), FakeLLM()
        agent = CandidateAgent("c-test", api, llm=llm)
        agent.start()
        agent.respond(message)
        assert agent.session.outcome == status and not llm.requests
        assert "email" not in api.writes[0]["changes"]


@pytest.mark.parametrize(
    ("field", "value", "quote"),
    [
        ("cgpa", 85, "85"),
        ("email", "abc@", "abc@"),
        ("degree", "BBA", "BBA"),
        ("internship_duration_months", 4, "4"),
        ("available_from", "2026-02-30", "2026-02-30"),
        ("graduation_year", 2031, "2031"),
    ],
)
def test_invalid_answers_cannot_be_written(field: str, value: Any, quote: str) -> None:
    agent = confirmed_agent()
    agent.session.candidate_messages.append(quote)
    result = agent.tools.execute(
        "validate_candidate_information",
        {
            "information": {field: value},
            "evidence": {field: quote},
        },
    )
    assert not result["ok"] and field in result["errors"]
    assert not agent.tools.execute("save_candidate_information", {"updates": {field: value}})["ok"]
    assert not agent.tools.api.writes


@pytest.mark.parametrize(
    ("field", "value", "message", "quote"),
    [
        ("cgpa", 8, "Around 8", "8"),
        ("cgpa", 8, "Approximately 8", "8"),
        ("graduation_year", 2027, "Next year", "Next year"),
        ("available_from", "2026-12-01", "Sometime in December", "December"),
        ("preferred_location", "Mumbai", "Near home", "Near home"),
    ],
)
def test_ambiguous_answers_need_clarification(
    field: str, value: Any, message: str, quote: str
) -> None:
    agent = confirmed_agent()
    agent.session.candidate_messages.append(message)
    result = agent.tools.execute(
        "validate_candidate_information",
        {
            "information": {field: value},
            "evidence": {field: quote},
        },
    )
    assert not result["ok"] and not agent.session.validated_values


def test_mixed_valid_and_invalid_answers_preserve_only_valid_progress() -> None:
    agent = confirmed_agent(api=FakeAPI({**RECORD, "cgpa": None}))
    agent.session.candidate_messages.append("test@example.com CGPA 85")
    agent.tools.execute(
        "validate_candidate_information",
        {
            "information": {"email": "test@example.com", "cgpa": 85},
            "evidence": {"email": "test@example.com", "cgpa": "85"},
        },
    )
    assert agent.stop()["ok"]
    assert agent.session.outcome == "in_progress"
    assert agent.session.latest_candidate["email"] == "test@example.com"
    assert agent.session.latest_candidate["cgpa"] is None


def test_new_contradiction_blocks_completion_and_keeps_old_value() -> None:
    agent = confirmed_agent()
    agent.session.candidate_messages.append("test@example.com Different College")
    agent.tools.execute(*validate_email())
    agent.tools.save_candidate_information({"email": "test@example.com"})
    agent.tools.validate_candidate_information(
        {"college": "Different College"}, {"college": "Different College"}
    )
    assert not agent.tools.save_candidate_information({"college": "Different College"})["ok"]
    assert not finish(agent, "completed")["ok"]
    assert finish(agent, "needs_human")["ok"]
    assert agent.session.latest_candidate["college"] == RECORD["college"]


def test_completion_waits_for_unresolved_validation_errors() -> None:
    agent = confirmed_agent()
    agent.session.candidate_messages.append("test@example.com CGPA 85")
    agent.tools.validate_candidate_information(
        {"email": "test@example.com", "cgpa": 85}, {"email": "test@example.com", "cgpa": "85"}
    )
    agent.tools.save_candidate_information({"email": "test@example.com"})
    assert is_candidate_complete(agent.session.latest_candidate)
    assert not finish(agent, "completed")["ok"]
    agent.session.candidate_messages.append("My CGPA is 8.25")
    agent.tools.validate_candidate_information({"cgpa": 8.25}, {"cgpa": "8.25"})
    assert finish(agent, "completed")["ok"]


def test_invalid_existing_values_identified_without_overwriting() -> None:
    api = FakeAPI({**RECORD, "cgpa": 85, "graduation_year": 2031})
    agent = CandidateAgent("c-test", api, llm=FakeLLM())
    assert set(agent.session.invalid_fields) == {"cgpa", "graduation_year"}
    assert not api.writes


def test_complete_record_gets_only_status_correction() -> None:
    api, llm = FakeAPI({**RECORD, "email": "test@example.com"}), FakeLLM()
    agent = CandidateAgent("c-test", api, llm=llm)
    assert agent.start() is None and not llm.requests
    assert api.writes == [
        {"id": "c-test", "version": 1, "changes": {"collection_status": "completed"}}
    ]
    assert agent.session.outcome_persisted
    assert api.record["attempt_count"] == 0


@pytest.mark.parametrize("count", [0, 49, 50])
def test_attempt_count_increment_once_and_cap(count: int) -> None:
    api = FakeAPI({**RECORD, "attempt_count": count, "cgpa": None})
    agent = confirmed_agent(api=api)
    agent.session.candidate_messages.append("test@example.com")
    agent.tools.execute(*validate_email())
    agent.tools.save_candidate_information({"email": "test@example.com"})
    assert agent.stop()["ok"]
    assert api.record["attempt_count"] == min(50, count + 1)
    assert sum("attempt_count" in write["changes"] for write in api.writes) <= 1
    previous_writes = len(api.writes)
    assert not agent.tools.finish_collection("needs_human", "retry", [])["ok"]
    assert not agent.tools.save_candidate_information({"email": "other@example.com"})["ok"]
    assert len(api.writes) == previous_writes


def test_bounded_notes_keep_recent_outcome_and_followup_time() -> None:
    agent = confirmed_agent(
        text("We can continue later."), api=FakeAPI({**RECORD, "notes": "x" * 4000})
    )
    agent.respond("Can we continue tomorrow at 5 PM?")
    notes = agent.session.latest_candidate["notes"]
    assert len(notes) <= 4000
    assert "Earlier notes shortened" in notes and "tomorrow at 5 PM" in notes
    assert "Outstanding: email" in notes


def test_final_status_write_failure_never_claims_saved() -> None:
    api = FakeAPI()
    api.error = APIError("timeout", "sensitive error")
    agent = confirmed_agent(api=api)
    response = agent.respond("I want a human")
    assert agent.session.ended and not agent.session.outcome_persisted
    assert "not confirmed" in response
    assert "sensitive" not in response
    assert len(api.writes) == 1


def test_tool_failure_is_structured_and_not_success() -> None:
    agent = confirmed_agent()
    for name, arguments in [
        ("unknown", {}),
        ("finish_collection", {"notes": []}),
        ("save_candidate_information", "bad json"),
    ]:
        assert not agent.tools.execute(name, arguments)["ok"]
    assert not agent.session.ended and not agent.tools.api.writes


def test_http_conflict_refetches_and_blocks_newer_value() -> None:
    requests = []
    latest = {**RECORD, "email": "newer@example.com", "version": 2}

    def handler(request):
        requests.append(request)
        if request.method == "PATCH":
            return httpx.Response(409)
        return httpx.Response(200, json=RECORD if len(requests) == 1 else latest)

    with CandidateAPIClient(api_key="test", transport=httpx.MockTransport(handler)) as api:
        agent = confirmed_agent(api=api)
        agent.session.candidate_messages.append("test@example.com")
        agent.tools.execute(*validate_email())
        assert (
            agent.tools.save_candidate_information({"email": "test@example.com"})["error"]
            == "conflict"
        )
        assert agent.session.latest_candidate == latest and not agent.session.validated_values
        agent.tools.execute(*validate_email())
        assert not agent.tools.save_candidate_information({"email": "test@example.com"})["ok"]
    assert [request.method for request in requests] == ["GET", "PATCH", "GET"]


@pytest.mark.parametrize("always_fail", [False, True])
def test_transport_retry_reuses_key_body_and_contact_count(monkeypatch, always_fail: bool) -> None:
    monkeypatch.setattr("api_client.sleep", lambda _: None)
    record, writes = deepcopy(RECORD), []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=record)
        writes.append(request)
        if len(writes) == 1:
            record.update(json.loads(request.content))
            record["version"] += 1
        if len(writes) == 1 or always_fail:
            raise httpx.ReadTimeout("response lost")
        return httpx.Response(200, json=record)

    with CandidateAPIClient(api_key="test", transport=httpx.MockTransport(handler)) as api:
        agent = CandidateAgent("c-test", api, llm=FakeLLM())
        agent.start()
        result = agent.stop()
        assert result["ok"] is not always_fail
        assert agent.session.outcome_persisted is not always_fail
    assert len(writes) == 2
    assert writes[0].content == writes[1].content
    assert writes[0].headers["Idempotency-Key"] == writes[1].headers["Idempotency-Key"]
    assert record["attempt_count"] == 1


def test_availability_sentence_is_not_a_deferral() -> None:
    assert requested_exit("I am available tomorrow") is None


def test_extended_identity_denial_stops_without_questions() -> None:
    agent = CandidateAgent("c-test", FakeAPI(), llm=FakeLLM())
    agent.start()
    agent.respond("No, I am not Test Candidate. You have the wrong person.")
    assert agent.session.identity == "denied"
    assert agent.session.outcome == "needs_human" and not agent.llm.requests


def test_refused_field_cannot_be_collected_from_quoted_refusal() -> None:
    agent = confirmed_agent()
    agent.session.candidate_messages.append("I won't share email test@example.com")
    agent.session.requested_outcome = requested_exit(agent.session.candidate_messages[-1])
    assert not agent.tools.execute(*validate_email())["ok"]
    assert not agent.tools.save_candidate_information({"email": "test@example.com"})["ok"]
    assert not agent.tools.api.writes


def test_unexpected_tool_error_is_safe(monkeypatch) -> None:
    agent = confirmed_agent()

    def fail(**kwargs):
        raise RuntimeError("sensitive provider details")

    monkeypatch.setattr(agent.tools, "save_candidate_information", fail)
    result = agent.tools.execute(
        "save_candidate_information", {"updates": {"email": "test@example.com"}}
    )
    assert result["error"] == "tool_failure"
    assert agent.session.write_blocked
    assert "sensitive" not in str(result)


def test_transient_server_retry_uses_same_write_identity(monkeypatch) -> None:
    monkeypatch.setattr("api_client.sleep", lambda _: None)
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={**RECORD, "version": 2, "notes": "retry"})

    with CandidateAPIClient(api_key="test", transport=httpx.MockTransport(handler)) as api:
        assert api.update_candidate("c-test", {"notes": "retry"}, 1)["version"] == 2
    assert len(requests) == 2
    assert requests[0].content == requests[1].content
    assert requests[0].headers["Idempotency-Key"] == requests[1].headers["Idempotency-Key"]


def test_complete_record_status_failure_is_not_success() -> None:
    api = FakeAPI({**RECORD, "email": "test@example.com"})
    api.error = APIError("timeout", "unknown")
    agent = CandidateAgent("c-test", api, llm=FakeLLM())
    assert agent.session.ended and agent.session.outcome == "needs_human"
    assert not agent.session.outcome_persisted and agent.start() is None
    assert api.writes[0]["changes"] == {"collection_status": "completed"}
