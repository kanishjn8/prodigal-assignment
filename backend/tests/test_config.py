import runpy
from pathlib import Path
from shutil import copyfile

import pytest

import config


@pytest.mark.parametrize(
    ("exported_key", "backend_key", "expected"),
    [
        (None, "backend-test-key", "backend-test-key"),
        (None, None, "root-test-key"),
        ("exported-test-key", "backend-test-key", "exported-test-key"),
    ],
)
def test_candidate_api_key_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exported_key: str | None,
    backend_key: str | None,
    expected: str,
) -> None:
    backend = tmp_path / "backend"
    backend.mkdir()
    module = backend / "config.py"
    copyfile(config.__file__, module)
    (tmp_path / ".env").write_text("CANDIDATE_API_KEY=root-test-key\n")
    if backend_key is not None:
        (backend / ".env").write_text(f"CANDIDATE_API_KEY={backend_key}\n")
    monkeypatch.delenv("CANDIDATE_API_KEY", raising=False)
    if exported_key is not None:
        monkeypatch.setenv("CANDIDATE_API_KEY", exported_key)
    assert runpy.run_path(str(module))["CANDIDATE_API_KEY"] == expected
