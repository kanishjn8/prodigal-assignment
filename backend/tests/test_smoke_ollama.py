import pytest
from ollama import Message

from scripts.smoke_ollama import execute_tool_call


def test_expected_tool_call() -> None:
    call = Message.ToolCall(function={"name": "add_numbers", "arguments": {"a": 17, "b": 25}})
    assert execute_tool_call(call) == 42


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("unknown_tool", {"a": 17, "b": 25}),
        ("add_numbers", {"a": "17", "b": 25}),
        ("add_numbers", {"a": True, "b": 25}),
        ("add_numbers", {"a": 17}),
        ("add_numbers", {"a": 17, "b": 25, "extra": 0}),
        ("add_numbers", {"a": 18, "b": 25}),
    ],
)
def test_rejects_unexpected_tool_calls(name: str, arguments: dict[str, object]) -> None:
    call = Message.ToolCall(function={"name": name, "arguments": arguments})
    with pytest.raises(ValueError):
        execute_tool_call(call)
