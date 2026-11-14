# Candidate information collection agent

An autonomous local-LLM chat agent for completing internship applications, as specified in
[ASSIGNMENT.md](ASSIGNMENT.md). **Phase 1 only:** configuration and real Ollama inference/tool-call
verification are implemented. The candidate API, conversation loop, validation and outcome tracking
are not implemented yet.

Prerequisites: uv, Python 3.12 (uv can provision it), and a running local Ollama server
at `http://localhost:11434`. The required default model is `llama3.1:8b`; install it with
`ollama pull llama3.1:8b` if absent. Start Ollama's desktop app or `ollama serve` first.
No cloud LLM or candidate API key is needed for this phase.

## Setup and verification

From the repository root:

```sh
uv sync --locked
cp .env.example .env
uv run python scripts/smoke_ollama.py
uv run ruff check .
uv run pytest
```

Edit `.env` to change configuration. Existing environment variables take precedence. Never commit
API keys. `.env`, virtual environments, caches and common secret-file extensions are ignored.

SDK reference: [official Ollama Python library](https://github.com/ollama/ollama-python).
