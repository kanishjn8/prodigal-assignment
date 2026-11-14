from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
from test_agent import RECORD, FakeAPI, FakeLLM, calls, text, validate_email

import main
from api_client import APIError
from tools import TOOL_SCHEMAS


class CLIAPI(FakeAPI):
    def __init__(self, record: dict[str, Any] | None = None) -> None:
        super().__init__(record)
        self.get_count = 0
        self.read_error: APIError | None = None
        self.verification_mismatch = False

    def __enter__(self) -> "CLIAPI":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def get_candidate(self, candidate_id: str) -> dict[str, Any]:
        self.get_count += 1
        if self.read_error:
            raise self.read_error
        result = super().get_candidate(candidate_id)
        if self.verification_mismatch and self.get_count > 1:
            result["collection_status"] = "pending"
        return result


class CLILLM(FakeLLM):
    def chat(self, **kwargs: Any) -> Any:
        if self.responses and isinstance(self.responses[0], KeyboardInterrupt):
            raise self.responses.pop(0)
        return super().chat(**kwargs)

    def list(self) -> Any:
        return SimpleNamespace(models=[SimpleNamespace(model=main.config.OLLAMA_MODEL)])


@pytest.fixture
def setup_cli(monkeypatch):
    def setup(inputs=(), responses=(), record=None):
        api, llm = CLIAPI(record), CLILLM(*responses)
        monkeypatch.setattr(main, "CandidateAPIClient", lambda: api)
        monkeypatch.setattr(main, "Client", lambda **kwargs: llm)
        replies = iter(inputs)

        def read(prompt):
            print(prompt, end="")
            item = next(replies, EOFError())
            if isinstance(item, BaseException):
                raise item
            return item

        monkeypatch.setattr("builtins.input", read)
        return api, llm

    return setup


def test_required_candidate_id(capsys) -> None:
    with pytest.raises(SystemExit) as caught:
        main.main([])
    assert caught.value.code == 2
    assert "--candidate-id" in capsys.readouterr().err


@pytest.mark.parametrize("candidate_id", ["", " ", "../schema", "c001?x=1", "a/b"])
def test_invalid_candidate_id(candidate_id: str) -> None:
    with pytest.raises(SystemExit) as caught:
        main.main(["--candidate-id", candidate_id])
    assert caught.value.code == 2


@pytest.mark.parametrize("kind", ["authentication", "missing_api_key", "not_found", "timeout"])
def test_api_startup_failure(setup_cli, capsys, kind: str) -> None:
    api, llm = setup_cli()
    api.read_error = APIError(kind, "secret-key-must-not-print")
    assert main.main(["--candidate-id", "c-test"]) == 1
    output = capsys.readouterr().out
    assert "secret-key" not in output and "Agent:" not in output
    assert not api.writes and not llm.requests


def test_ollama_connection_failure(setup_cli, monkeypatch, capsys) -> None:
    api, llm = setup_cli()

    def fail():
        raise ConnectionError("secret must not print")

    monkeypatch.setattr(llm, "list", fail)
    assert main.main(["--candidate-id", "c-test"]) == 1
    assert "Start Ollama" in capsys.readouterr().out
    assert not api.writes


def test_missing_model(setup_cli, monkeypatch, capsys) -> None:
    api, llm = setup_cli()
    monkeypatch.setattr(llm, "list", lambda: SimpleNamespace(models=[]))
    assert main.main(["--candidate-id", "c-test"]) == 1
    assert "ollama pull" in capsys.readouterr().out
    assert not api.writes


def test_complete_candidate_does_not_chat(setup_cli, capsys) -> None:
    api, llm = setup_cli(record={**RECORD, "email": "test@example.com"})
    assert main.main(["--candidate-id", "c-test"]) == 0
    output = capsys.readouterr().out
    assert "No chat" in output and "Agent:" not in output
    assert not llm.requests and not api.writes and api.get_count == 1


def test_identity_greeting_and_quit(setup_cli, capsys) -> None:
    api, llm = setup_cli(inputs=["/quit"])
    assert main.main(["--candidate-id", "c-test"]) == 0
    output = capsys.readouterr().out
    assert "Agent: Hi! I'm ACME Corp" in output
    assert "Am I speaking with Test Candidate?" in output
    assert "You:" in output and "no_answer (verified in database)" in output
    assert not llm.requests and api.get_count == 2


def test_multi_turn_and_normal_completion(setup_cli, capsys) -> None:
    api, llm = setup_cli(
        inputs=["yes", "What do you need?", "My email is test@example.com"],
        responses=[
            calls(("inspect_candidate_state", {})),
            text(),
            text("Only your email is missing."),
            calls(
                validate_email(),
                ("save_candidate_information", {"updates": {"email": "test@example.com"}}),
            ),
            calls(
                (
                    "finish_collection",
                    {"collection_status": "completed", "notes": "", "refused_fields": []},
                )
            ),
        ],
    )
    original = deepcopy(api.record)
    assert main.main(["--candidate-id", "c-test"]) == 0
    output = capsys.readouterr().out
    assert "Only your email is missing" in output
    assert "completed (verified in database)" in output
    assert api.record["email"] == "test@example.com"
    assert api.record["college"] == original["college"]
    assert api.get_count == 2 and len(api.writes) == 2
    user_messages = [
        message["content"] for message in llm.requests[-1]["messages"] if message["role"] == "user"
    ]
    assert "What do you need?" in user_messages and "My email is test@example.com" in user_messages


@pytest.mark.parametrize("ending", [EOFError(), KeyboardInterrupt(), "/quit"])
def test_terminal_exits_record_outcome(setup_cli, capsys, ending: Any) -> None:
    api, _ = setup_cli(inputs=[ending])
    assert main.main(["--candidate-id", "c-test"]) == 0
    assert api.record["collection_status"] == "no_answer"
    assert api.record["attempt_count"] == 1
    assert "Traceback" not in capsys.readouterr().out


def test_help_blank_and_unknown_command_do_not_reach_model(setup_cli, capsys) -> None:
    api, llm = setup_cli(inputs=["/help", "", "/unknown", "/quit"])
    assert main.main(["--candidate-id", "c-test"]) == 0
    output = capsys.readouterr().out
    assert "Unknown command" in output and "Please enter a reply" in output
    assert not llm.requests and len(api.writes) == 1


def test_failed_exit_save_is_not_success(setup_cli, capsys) -> None:
    api, _ = setup_cli(inputs=["/quit"])
    api.error = APIError("timeout", "secret")
    assert main.main(["--candidate-id", "c-test"]) == 1
    output = capsys.readouterr().out
    assert "could not be saved" in output and "secret" not in output


def test_verification_detects_concurrent_change(setup_cli, capsys) -> None:
    api, _ = setup_cli(inputs=["/quit"])
    api.verification_mismatch = True
    assert main.main(["--candidate-id", "c-test"]) == 1
    assert "mismatch or concurrent change" in capsys.readouterr().out


def test_interrupt_during_inference_records_exit(setup_cli) -> None:
    api, _ = setup_cli(inputs=["yes"], responses=[KeyboardInterrupt()])
    assert main.main(["--candidate-id", "c-test"]) == 0
    assert api.record["collection_status"] == "no_answer"


def test_interrupt_during_write_does_not_retry(setup_cli, monkeypatch, capsys) -> None:
    api, _ = setup_cli(inputs=["/quit"])
    attempts = []

    def interrupted(*args):
        attempts.append(args)
        raise KeyboardInterrupt()

    monkeypatch.setattr(api, "update_candidate", interrupted)
    assert main.main(["--candidate-id", "c-test"]) == 1
    assert len(attempts) == 1
    assert "could not be saved" in capsys.readouterr().out


def test_numeric_extraction_schema_and_retry(setup_cli, capsys) -> None:
    for index, argument in [(1, "information"), (2, "updates")]:
        properties = TOOL_SCHEMAS[index]["function"]["parameters"]["properties"][argument][
            "properties"
        ]
        assert properties["cgpa"]["type"] == "number"
        assert properties["graduation_year"]["type"] == "integer"
    api, llm = setup_cli(
        record={**RECORD, "email": "test@example.com", "cgpa": None},
        inputs=["yes", "My CGPA is 8.5 on a 10-point scale"],
        responses=[
            calls(("inspect_candidate_state", {})),
            text("What is your CGPA?"),
            calls(
                (
                    "validate_candidate_information",
                    {"information": {"cgpa": "8.5"}, "evidence": {"cgpa": "8.5"}},
                )
            ),
            calls(
                (
                    "validate_candidate_information",
                    {"information": {"cgpa": 8.5}, "evidence": {"cgpa": "8.5"}},
                )
            ),
            calls(("save_candidate_information", {"updates": {"cgpa": 8.5}})),
            calls(
                (
                    "finish_collection",
                    {"collection_status": "completed", "notes": "", "refused_fields": []},
                )
            ),
        ],
    )
    assert main.main(["--candidate-id", "c-test"]) == 0
    assert api.record["cgpa"] == 8.5
    assert len(api.writes) == 2
    assert "Extraction type error" in llm.requests[3]["messages"][-1]["content"]
    assert "completed (verified in database)" in capsys.readouterr().out


def test_model_cannot_suggest_arbitrary_percentage_conversion(setup_cli, capsys) -> None:
    setup_cli(
        record={**RECORD, "cgpa": None},
        inputs=["yes", "/quit"],
        responses=[
            calls(("inspect_candidate_state", {})),
            text("If you scored 85%, you would enter 8.5."),
        ],
    )
    assert main.main(["--candidate-id", "c-test"]) == 0
    output = capsys.readouterr().out
    assert "If you scored 85%" not in output
    assert "official CGPA" in output
