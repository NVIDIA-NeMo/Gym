# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess
from pathlib import Path

import pytest
import setup_scoring


def test_install_is_locked_and_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "uv.lock").write_text("lock-one")
    monkeypatch.setattr(setup_scoring, "PROJECT", project)
    monkeypatch.delenv("LONGMT_SCORING_PYTHON", raising=False)
    calls = []

    def run(argv: list[str], **kwargs: object) -> None:
        calls.append(argv)
        target = Path(kwargs["env"]["UV_PROJECT_ENVIRONMENT"]) / "bin/python"
        target.parent.mkdir(parents=True)
        target.touch()

    monkeypatch.setattr(setup_scoring.subprocess, "run", run)
    monkeypatch.setattr(setup_scoring, "validate_python", lambda _: None)
    first = setup_scoring.ensure_scoring_python(tmp_path / "cache")
    assert setup_scoring.ensure_scoring_python(tmp_path / "cache") == first
    assert len(calls) == 1 and "--locked" in calls[0] and "3.12" in calls[0]
    (project / "uv.lock").write_text("lock-two")
    assert setup_scoring.ensure_scoring_python(tmp_path / "cache") != first


def test_failed_install_never_marks_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LONGMT_SCORING_PYTHON", raising=False)
    monkeypatch.setattr(setup_scoring, "PROJECT", tmp_path)
    (tmp_path / "uv.lock").write_text("lock")

    def fail(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, "uv")

    monkeypatch.setattr(setup_scoring.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        setup_scoring.ensure_scoring_python(tmp_path / "cache")
    assert not list((tmp_path / "cache").glob("*/ready"))


def test_override_is_validated_without_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "python"
    monkeypatch.setenv("LONGMT_SCORING_PYTHON", str(target))
    validated = []
    monkeypatch.setattr(setup_scoring, "validate_python", validated.append)
    assert setup_scoring.ensure_scoring_python(tmp_path / "cache") == target
    assert validated == [target] and not (tmp_path / "cache").exists()
