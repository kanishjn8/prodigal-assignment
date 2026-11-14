import argparse
import re
from collections.abc import Sequence

import httpx
from ollama import Client, ResponseError

import config
from agent import CandidateAgent
from api_client import WRITABLE_FIELDS, APIError, CandidateAPIClient
from validators import is_candidate_complete

HELP_TEXT = (
    "/help: show commands. /quit: end the chat and record the outcome. EOF or Ctrl+C also exits."
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
    return parser


def chat(agent: CandidateAgent) -> None:
    print(HELP_TEXT, flush=True)
    greeting = agent.start()
    if greeting:
        print(f"Agent: {greeting}", flush=True)
    try:
        while not agent.session.ended:
            reply = input("You: ").strip()
            if reply == "/help":
                print(HELP_TEXT, flush=True)
                continue
            if reply == "/quit":
                agent.stop(notes="Candidate ended the chat with /quit.")
                break
            if reply.startswith("/"):
                print("Unknown command. Use /help or /quit.", flush=True)
                continue
            if not reply:
                print("Please enter a reply, or use /quit to exit.", flush=True)
                continue
            response = agent.respond(reply)
            if response:
                print(f"Agent: {response}", flush=True)
    except (EOFError, KeyboardInterrupt):
        print("\nEnding the conversation...", flush=True)
        agent.stop(notes="Candidate left the terminal chat (EOF or Ctrl+C).")


def verify_outcome(agent: CandidateAgent, api: CandidateAPIClient) -> int:
    session = agent.session
    if not session.outcome_persisted:
        print(
            "Outcome could not be saved reliably. Human review is required; do not retry blindly."
        )
        return 1
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
        print(
            "Database verification found a mismatch or concurrent change. Human review is required."
        )
        return 1
    print(f"Outcome: {session.outcome} (verified in database).", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    agent = None
    try:
        with CandidateAPIClient() as api:
            print("Checking candidate record...", flush=True)
            llm = Client(host=config.OLLAMA_BASE_URL, timeout=120.0)
            # The startup GET checks reachability and authentication without a redundant API call.
            agent = CandidateAgent(args.candidate_id, api, llm=llm)
            if agent.session.ended:
                print(
                    "All mandatory information is already valid. "
                    "No chat or database changes needed."
                )
                return 0
            print("Checking local Ollama model...", flush=True)
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
            "missing_api_key": "Set CANDIDATE_API_KEY in your local .env file.",
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
