import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol


class RecordDecision(Protocol):
    def __call__(self, event: str, /, **details: Any) -> None: ...


def ignore_decision(event: str, /, **details: Any) -> None:
    pass


@contextmanager
def open_decision_log(directory: Path) -> Iterator[tuple[Path, RecordDecision]]:
    directory.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=directory,
        prefix="decision-",
        suffix=".jsonl",
        delete=False,
    ) as stream:
        failed = False

        def emit(event: str, /, **details: Any) -> None:
            nonlocal failed
            if failed:
                return
            try:
                row = {"time": datetime.now(UTC).isoformat(), "event": event, **details}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
            except OSError:
                failed = True
                print("Could not continue writing the decision log.", file=sys.stderr)

        yield Path(stream.name), emit
