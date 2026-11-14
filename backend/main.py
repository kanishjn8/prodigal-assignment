import argparse
import re
from collections.abc import Sequence
from contextlib import ExitStack

import httpx
from ollama import Client, ResponseError

import config
import terminal_ui as ui
from agent import CandidateAgent
from api_client import WRITABLE_FIELDS, APIError, CandidateAPIClient
from decision_log import ignore_decision, open_decision_log
from validators import is_candidate_complete

HELP_TEXT = (
    "/help: show commands. /quit: end the chat. /no-answer: simulate no further response. "
    "EOF or Ctrl+C also exits. Validated progress is retained."
)


def candidate_id_argument(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError(
            "Use a nonempty candidate ID with letters, digits, _ or -."
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ACME Corp internship application chat")
    parser.add_argument(
        "--candidate-id",
        required=True,
        type=candidate_id_argument,
        help="Candidate record ID supplied by the evaluator (for example c001)",
    )
    parser.epilog = HELP_TEXT
    parser.add_argument(
        "--decision-log",
        action="store_true",
        help="Write a redacted decision log under runs/ and print its path",
    )
    return parser


def chat(agent: CandidateAgent) -> None:
    print(ui.styled("  /help · /quit · /no-answer · Ctrl+C to stop", "2"), flush=True)
    greeting = agent.start()
    if greeting:
        ui.agent_message(greeting)
    try:
        while not agent.session.ended:
            reply = input(ui.input_prompt()).strip()
            if reply == "/help":
                print(HELP_TEXT, flush=True)
                continue
            if reply in {"/quit", "/no-answer"}:
                with ui.busy("Saving conversation outcome"):
                    agent.stop(notes=f"Candidate ended the chat with {reply}.")
                break
            if reply.startswith("/") and not (
                agent.awaiting_confirmation and reply in {"/accept", "/edit"}
            ):
                print("Unknown command. Use /help or /quit.", flush=True)
                continue
            if not reply:
                print("Please enter a reply, or use /quit to exit.", flush=True)
                continue
            with ui.busy("Reviewing your reply") as progress:
                agent.set_progress(progress)
                try:
                    response = agent.respond(reply)
                finally:
                    agent.set_progress(None)
            if response:
                ui.agent_message(response)
    except (EOFError, KeyboardInterrupt):
        print("\nEnding the conversation...", flush=True)
        with ui.busy("Saving conversation outcome"):
            agent.stop(notes="Candidate left the terminal chat (EOF or Ctrl+C).")


def verify_outcome(agent: CandidateAgent, api: CandidateAPIClient) -> int:
    session = agent.session
    if not session.outcome_persisted:
        agent.on_decision("outcome_verification", ok=False, reason="unconfirmed_write")
        print(
            "Outcome could not be saved reliably. Human review is required; do not retry blindly."
        )
        return 1
    with ui.busy("Verifying saved changes"):
        record = api.get_candidate(session.candidate_id)
    changed = {
        key
        for key in WRITABLE_FIELDS
        if session.latest_candidate.get(key) != session.original_candidate.get(key)
    } | {"collection_status"}
    if (
        record["version"] < session.version
        or any(record.get(key) != session.latest_candidate.get(key) for key in changed)
        or (session.outcome == "completed" and not is_candidate_complete(record))
    ):
        agent.on_decision("outcome_verification", ok=False, reason="record_mismatch")
        print(
            "Database verification found a mismatch or concurrent change. Human review is required."
        )
        return 1
    agent.on_decision("outcome_verification", ok=True, status=session.outcome)
    print(ui.styled(f"\nOutcome: {session.outcome} (verified in database).", "1;32"), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ui.banner()
    agent = None
    try:
        with ExitStack() as resources:
            record = ignore_decision
            if args.decision_log:
                try:
                    path, record = resources.enter_context(
                        open_decision_log(config.PROJECT_ROOT / "runs")
                    )
                    print(f"Decision log: {path}", flush=True)
                except OSError:
                    print("Could not open the decision log. Continuing the chat.")
            api = resources.enter_context(CandidateAPIClient())
            llm = Client(host=config.OLLAMA_BASE_URL, timeout=120.0)
            # The startup GET checks reachability and authentication without a redundant API call.
            with ui.busy("Checking candidate record"):
                agent = CandidateAgent(args.candidate_id, api, llm=llm, on_decision=record)
            if agent.session.ended:
                if agent.session.outcome == "completed":
                    print("All mandatory information is valid. No chat needed.")
                return verify_outcome(agent, api)
            with ui.busy("Connecting to local Ollama"):
                available = {entry.model for entry in llm.list().models}
            if config.OLLAMA_MODEL not in available:
                print(f"Local model is missing. Install it with: ollama pull {config.OLLAMA_MODEL}")
                return 1
            try:
                chat(agent)
            except Exception:
                # Do not print raw exception strings: provider errors can contain request data.
                print(
                    "The chat encountered a technical error. Recording a request for human review."
                )
                agent.stop("needs_human", "Unexpected terminal chat error; review saved progress.")
            return verify_outcome(agent, api)
    except APIError as exc:
        explanations = {
            "missing_api_key": "Set CANDIDATE_API_KEY in backend/.env or the repository root .env.",
            "authentication": "Candidate API authentication failed. Check your local API key.",
            "not_found": "Candidate record was not found. Check the supplied candidate ID.",
            "timeout": "Candidate API timed out. Check connectivity before retrying.",
            "connection": "Could not connect to the candidate API.",
            "rate_limit": "Candidate API rate limit reached. Wait before trying again.",
        }
        print(
            explanations.get(exc.kind, "Candidate API request failed; human review may be needed.")
        )
        if agent is not None and agent.started:
            print(
                "Final database verification was not confirmed. "
                "Do not assume the last change failed."
            )
        return 1
    except (ConnectionError, httpx.HTTPError, ResponseError):
        print("Could not access local Ollama. Start Ollama and check OLLAMA_BASE_URL.")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Any in-flight operation needs verification before another attempt.")
        return 130
    except Exception:
        print("Startup or outcome handling failed. Check configuration; no success is assumed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
