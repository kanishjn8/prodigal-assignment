import re
import sys
from pathlib import Path
from time import perf_counter

import httpx
from ollama import Client, Message, ResponseError

# Direct script execution puts scripts/, rather than the repository root, on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL  # noqa: E402


def add_numbers(a: int, b: int) -> int:
    """Add two integers and return their sum."""
    return a + b


def execute_tool_call(call: Message.ToolCall) -> int:
    if call.function.name != "add_numbers":
        raise ValueError(f"Unexpected tool: {call.function.name}")
    arguments = call.function.arguments
    if set(arguments) != {"a", "b"} or any(type(v) is not int for v in arguments.values()):
        raise ValueError("add_numbers requires exactly two integer arguments: a and b")
    if arguments != {"a": 17, "b": 25}:
        raise ValueError("Tool arguments did not match the requested calculation")
    return add_numbers(**arguments)


def run_smoke_test(client: Client, model: str) -> None:
    started = perf_counter()
    available = {entry.model for entry in client.list().models}
    print(f"PASS: connected to Ollama ({perf_counter() - started:.2f}s)", flush=True)
    if model not in available:
        raise ValueError(f"Model {model!r} is not installed. Run: ollama pull {model}")
    print(f"PASS: model {model} is installed", flush=True)

    started = perf_counter()
    reply = client.chat(
        model=model,
        messages=[{"role": "user", "content": "Say hello in one short sentence."}],
        stream=False,
        options={"temperature": 0, "num_predict": 128},
    )
    if not reply.message.content or not reply.message.content.strip():
        raise ValueError("Simple chat returned no text")
    print(
        f"PASS: simple chat ({perf_counter() - started:.2f}s): {reply.message.content}", flush=True
    )

    messages = [
        Message(role="system", content="Use the provided tool for arithmetic. Do not guess."),
        Message(role="user", content="Call add_numbers with a=17 and b=25. What is the sum?"),
    ]
    started = perf_counter()
    response = client.chat(
        model=model,
        messages=messages,
        tools=[add_numbers],
        stream=False,
        options={"temperature": 0, "num_predict": 256},
    )
    calls = response.message.tool_calls
    if not calls:
        raise ValueError("FAIL: no structured tool_calls; prose about a tool does not count")
    if len(calls) != 1:
        raise ValueError(f"Expected one tool call, received {len(calls)}")
    messages.append(response.message)
    result = execute_tool_call(calls[0])
    print(f"Structured tool call: {calls[0].model_dump_json()}", flush=True)
    print(f"Python tool result: {result}", flush=True)
    messages.append(Message(role="tool", tool_name="add_numbers", content=str(result)))
    print(f"PASS: native tool invocation ({perf_counter() - started:.2f}s)", flush=True)

    started = perf_counter()
    final = client.chat(
        model=model,
        messages=messages,
        stream=False,
        options={"temperature": 0, "num_predict": 128},
    )
    if final.message.tool_calls or not re.search(r"\b42\b", final.message.content or ""):
        raise ValueError("Final response did not report the expected tool result (42)")
    print(
        f"PASS: final answer ({perf_counter() - started:.2f}s): {final.message.content}", flush=True
    )


def main() -> int:
    try:
        run_smoke_test(Client(host=OLLAMA_BASE_URL, timeout=120.0), OLLAMA_MODEL)
    except (ConnectionError, httpx.HTTPError, ResponseError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Smoke test interrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
