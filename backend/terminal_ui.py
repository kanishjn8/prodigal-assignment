import os
import shutil
import sys
import textwrap
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import Event, Thread
from time import monotonic


def interactive() -> bool:
    return sys.stdout.isatty() and os.getenv("TERM") != "dumb"


def styled(text: str, code: str) -> str:
    if not interactive() or "NO_COLOR" in os.environ:
        return text
    return f"\033[{code}m{text}\033[0m"


def banner() -> None:
    print()
    print(styled("  ACME CORP  /  INTERNSHIPS", "1;36"))
    print(styled("  Let's complete your application.", "2"))
    print(styled("  " + "─" * min(54, max(10, shutil.get_terminal_size().columns - 4)), "2"))
    print()


def agent_message(text: str) -> None:
    print()
    if not interactive():
        print(f"Agent: {text}", flush=True)
        return
    width = max(20, min(88, shutil.get_terminal_size().columns - 4))
    print(styled("  Agent", "1;36"))
    for paragraph in text.splitlines():
        print(textwrap.fill(paragraph, width=width, initial_indent="  ", subsequent_indent="  "))
    print(flush=True)


def input_prompt() -> str:
    return styled("  You: ", "1;32") if interactive() else "You: "


@contextmanager
def busy(label: str) -> Iterator[Callable[[str], None]]:
    """Animate only the display; keep API calls on the main thread for Ctrl+C handling."""
    if not interactive():
        print(f"{label}...", flush=True)

        def report(next_label: str) -> None:
            nonlocal label
            if next_label != label:
                label = next_label
                print(f"{label}...", flush=True)

        yield report
        return
    stream = sys.stdout
    stopped = Event()
    started = monotonic()

    def draw(frame: int) -> None:
        elapsed = int(monotonic() - started)
        status = f"  {'⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'[frame % 10]} {label}…  {elapsed}s"
        stream.write("\r\033[2K" + styled(status, "2;36"))
        stream.flush()

    def animate() -> None:
        frame = 1
        while not stopped.wait(0.12):
            draw(frame)
            frame += 1

    def report(next_label: str) -> None:
        nonlocal label
        label = next_label

    draw(0)
    worker = Thread(target=animate, name="cli-status", daemon=True)
    worker.start()
    try:
        yield report
    finally:
        stopped.set()
        worker.join()
        stream.write("\r\033[2K")
        stream.flush()
