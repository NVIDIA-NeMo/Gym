# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path

import pytest

from resources_servers.swe_rebench import log_parsers


pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is required")


@pytest.fixture
def upstream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "upstream"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "symbolic-ref", "HEAD", "refs/heads/main"], cwd=repo, check=True)
    parser = repo / "lib" / "agent" / "log_parsers.py"
    parser.parent.mkdir(parents=True)
    for name in ("pinned", "new_main"):
        parser.write_text(f"NAME_TO_PARSER = {{'{name}': lambda output: {{}}}}\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-m",
                name,
            ],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        if name == "pinned":
            pin = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
            subprocess.run(["git", "tag", "pinned", pin], cwd=repo, check=True)
            monkeypatch.setattr(log_parsers, "UPSTREAM_COMMIT", pin)
    monkeypatch.setattr(log_parsers, "UPSTREAM_REPO", repo.as_uri())
    monkeypatch.setattr(log_parsers, "_MODULE", None)
    return tmp_path / "cache"


def test_load_uses_pin_and_ignores_old_main_cache(upstream: Path) -> None:
    stale = upstream / "SWE-rebench-V2" / "lib" / "agent" / "log_parsers.py"
    stale.parent.mkdir(parents=True)
    stale.write_text("NAME_TO_PARSER = {'stale': None}\n")

    module = log_parsers.load_parsers(upstream)

    assert set(module.NAME_TO_PARSER) == {"pinned"}
    destination = upstream / f"SWE-rebench-V2-{log_parsers.UPSTREAM_COMMIT}"
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=destination, text=True).strip() == (
        log_parsers.UPSTREAM_COMMIT
    )
    assert "stale" in stale.read_text()


@pytest.mark.parametrize("fail_on", ["fetch", "checkout"])
def test_failed_pin_is_not_published_and_can_retry(
    upstream: Path, monkeypatch: pytest.MonkeyPatch, fail_on: str
) -> None:
    real_run = subprocess.run

    def fail(args, **kwargs):
        if args[1] == fail_on:
            raise subprocess.CalledProcessError(128, args)
        return real_run(args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(log_parsers.subprocess, "run", fail)
        with pytest.raises(subprocess.CalledProcessError):
            log_parsers.load_parsers(upstream)
    assert list(upstream.iterdir()) == []
    assert set(log_parsers.load_parsers(upstream).NAME_TO_PARSER) == {"pinned"}


def test_other_workers_cannot_see_initial_main_checkout(upstream: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_run = subprocess.run
    destination = upstream / f"SWE-rebench-V2-{log_parsers.UPSTREAM_COMMIT}"
    observed = []

    def inspect(args, **kwargs):
        result = real_run(args, **kwargs)
        observed.append(args[1])
        assert not destination.exists()
        return result

    monkeypatch.setattr(log_parsers.subprocess, "run", inspect)
    assert set(log_parsers.load_parsers(upstream).NAME_TO_PARSER) == {"pinned"}
    assert observed == ["clone", "fetch", "checkout"]
    assert destination.exists()
