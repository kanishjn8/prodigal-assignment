# Candidate information collection agent

ACME Corp has incomplete internship applications. This terminal agent loads one
candidate's record, checks which required values are missing or invalid, confirms
the candidate's name, and collects the remaining information in conversation.
Ollama runs inference locally, so the chat does not require a hosted model or
send prompts to a cloud inference provider. Candidate records are read and
updated through the configured HTTP API.

## Setup and run

Prerequisites are uv, Python 3.12 (which uv can provision), and a running Ollama
instance. From the repository root:

```sh
uv sync --locked
cp -n .env.example .env
ollama pull llama3.1:8b
uv run python backend/main.py --candidate-id c002 --decision-log
```

Before the last command, set `CANDIDATE_API_KEY` in `.env` or `backend/.env`.
Environment variables take precedence, then `backend/.env`, then the root
`.env`. Both files are ignored by Git. Set `OLLAMA_MODEL` to the name and tag of
another installed tool-capable model to try it. The default is `llama3.1:8b`,
with Ollama at `http://localhost:11434` unless `OLLAMA_BASE_URL` is set. Supply
any candidate ID from your workspace. The ID goes on the command line and the
candidate confirms their name in chat.

## Stack and architecture

The project uses Python 3.12, uv, the Ollama Python SDK, httpx, python-dotenv,
deterministic Python validators, pytest and Ruff.

```text
main.py + terminal_ui.py
  -> api_client.py          GET one record by ID
  -> validators.py         assess missing and invalid fields
  -> agent.py               name gate and in-memory conversation
     <-> local Ollama      native tool calls and follow-up questions
     -> tools.py           evidence, validation, confirmation and outcome checks
        -> api_client.py   versioned PATCH
  -> main.py               re-fetch to verify the outcome
```

The model chooses what to ask and can extract several answers in one reply.
Python executes allowlisted tools and returns their results to the model. Simple
answers to focused questions can be handled directly. Session state holds the
original and latest records, version, identity, conversation history, staged
values, validation errors and outcome. Python records completion once all
required fields are valid and any confirmations or disputes are resolved.
Already-complete records skip chat and only correct the status if necessary.

| Tool | Purpose |
| --- | --- |
| `inspect_candidate_state` | Check what is missing or invalid and see validated progress. |
| `validate_candidate_information` | Check extracted values against the schema and quoted candidate evidence. |
| `save_candidate_information` | Write validated, changed fields after identity and applicable read-back confirmation. |
| `finish_collection` | Record a permitted outcome and notes, including fields still outstanding. |

## Conversation decisions

- **No reply or early exit.** Use `/no-answer` to simulate silence, or `/quit`, EOF
  or Ctrl+C to leave. Without useful progress this records `no_answer`. With
  progress it retains valid values and records `in_progress`. The terminal does
  not run an idle timer.
- **Continue later or refuse.** Clear deferrals record `in_progress` with the
  requested time if supplied. Refusals record `refused` and the declined fields.
  The agent does not schedule follow-up messages.
- **Ambiguous or invalid answers.** The agent asks for clarification. Recognized
  invalid single-field replies get a focused retry. Graduation year and start
  date are read back before saving. Use `/accept` or `/edit` at that prompt.
- **Existing data and identity.** Nonempty invalid fields still need attention.
  The agent checks identity before writing profile data. A mismatch, disputed
  valid value, or explicit request for human help can end in `needs_human`.
  Technical failures can also require review.

## API safety and logs

The API stores values as sent. Python validates the shape, field rules and
evidence before building a minimal PATCH. Candidate ID, phone and server
metadata are read-only. Each write uses the current version in `If-Match` and a
UUID idempotency key. A bounded retry reuses the same body and key after a
transport failure or transient gateway error. A conflict refreshes the record
and clears pending validation. Disputed values are preserved. If the result
of a write is uncertain, further writes are blocked in that session.

The agent records the status, notes, contact timestamp and attempt count when
applicable. The CLI verifies persisted outcomes by reading the record again.
With `--decision-log`, the CLI prints a local JSON Lines path under `runs/`.
Follow it with `tail -f <path>` in another terminal. It includes field names,
tool outcomes, blocked writes and status without raw candidate replies, record
values or credentials. Generated decision logs are ignored by Git.

## Tests and sample runs

```sh
uv run pytest -q
uv run ruff check .
uv run python backend/main.py --help
```

The default suite uses mocked APIs and model replies to cover validation,
confirmation, completion, deferral, refusal, no reply, conflicts, retries and
write failures. The redacted samples under [runs/](runs/) use local Ollama and
the live private test API:

- [c002-completion.json](runs/c002-completion.json) records a rejected five-month
  duration, a valid correction, and verified completion.
- [c004-deferral.json](runs/c004-deferral.json) records a request to continue
  later and a verified `in_progress` outcome.

To deliberately restore all records in the private test workspace, run
`uv run python backend/scripts/reset_workspace.py`. This overwrites collected
test data for every candidate in that workspace.

## Current limits

The CLI supports one session at a time. Ollama can misinterpret free-form or
mixed replies, repeat questions, or miss a required tool call. The deterministic
checks protect writes when that happens. The read-back commands provide a
fallback if the model cannot interpret a free-form confirmation. Alternative
Ollama models and live concurrency or provider failures were not exhaustively
tested. Session history is in memory and disappears when the process exits.

Future work includes more reliable structured extraction, a broader
conversation evaluation set, persistent resumption, stronger recovery and
observability, and integration with a production messaging channel.
