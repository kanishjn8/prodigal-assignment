import io
import threading

import pytest

import terminal_ui as ui


def test_redirected_output_has_readable_status_without_escapes(capsys) -> None:
    with ui.busy("Thinking"):
        pass
    ui.agent_message("What is your email address?")
    output = capsys.readouterr().out
    assert "Thinking..." in output
    assert "Agent: What is your email address?" in output
    assert "\033" not in output


def test_spinner_cleans_up_when_interrupted(monkeypatch) -> None:
    stream = io.StringIO()
    monkeypatch.setattr(ui.sys, "stdout", stream)
    monkeypatch.setattr(ui, "interactive", lambda: True)
    with pytest.raises(KeyboardInterrupt), ui.busy("Thinking"):
        raise KeyboardInterrupt
    assert "Thinking" in stream.getvalue()
    assert stream.getvalue().endswith("\r\033[2K")
    assert not any(thread.name == "cli-status" for thread in threading.enumerate())


def test_no_color_is_respected(monkeypatch) -> None:
    monkeypatch.setattr(ui, "interactive", lambda: True)
    monkeypatch.setenv("NO_COLOR", "1")
    assert ui.styled("Agent", "1;36") == "Agent"
