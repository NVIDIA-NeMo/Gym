# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenHands runtime install: repeatable attempts and retries (offline: local bash, fake sandbox).

Run with the agent venv python:
  .venv/bin/python -m pytest responses_api_agents/gdpval_sandboxed_agent/tests/test_sandbox_runtime.py
"""

import hashlib
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from responses_api_agents.gdpval_sandboxed_agent import sandbox_runtime
from responses_api_agents.gdpval_sandboxed_agent.sandbox_runtime import install_command, install_runtime

# The staged install.sh's checks, with PREFIX taken from the environment so the test can relocate it.
FAKE_INSTALL_SH = r"""#!/usr/bin/env bash
set -euo pipefail
ARCHIVE="$2"
[[ -r "$ARCHIVE" && -r "$ARCHIVE.sha256" ]] || { echo "Missing staged OpenHands runtime: $ARCHIVE" >&2; exit 1; }
read -r EXPECTED _ < "$ARCHIVE.sha256"
ACTUAL="$(sha256sum "$ARCHIVE")"
[[ "${ACTUAL%% *}" == "$EXPECTED" ]] || { echo 'Runtime archive checksum mismatch' >&2; exit 1; }
if [[ -e "$PREFIX" ]]; then
  [[ -f "$PREFIX/.installed-sha256" && "$(cat "$PREFIX/.installed-sha256")" == "$EXPECTED" ]] || {
    echo "Existing $PREFIX does not match this runtime. Use a fresh sandbox." >&2
    exit 1
  }
else
  tar -xzf "$ARCHIVE" -C "$(dirname "$PREFIX")"
  printf '%s\n' "$EXPECTED" > "$PREFIX/.installed-sha256"
fi
echo 'OpenHands staged runtime ready'
"""


@pytest.fixture
def staged(tmp_path):
    """A staged mount (install.sh, runtime.tar.gz, .sha256) and an empty install root."""
    payload = tmp_path / "build" / "gdpval-openhands"
    (payload / "venv" / "bin").mkdir(parents=True)
    (payload / "venv" / "bin" / "python").write_text("#!/bin/sh\n")
    mount = tmp_path / "mount"
    mount.mkdir()
    archive = mount / "runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload, arcname="gdpval-openhands")
    (mount / "runtime.tar.gz.sha256").write_text(f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  runtime.tar.gz\n")
    script = mount / "install.sh"
    script.write_text(FAKE_INSTALL_SH)
    root = tmp_path / "opt"
    root.mkdir()
    return SimpleNamespace(script=script, archive=archive, prefix=root / "gdpval-openhands", local=tmp_path / "local")


def _run(staged, timeout_s=900):
    command = install_command(str(staged.script), str(staged.archive), timeout_s=timeout_s,
                              prefix=str(staged.prefix), local_dir=str(staged.local))
    return subprocess.run(["bash", "-c", command], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "PREFIX": str(staged.prefix)})


def test_clean_install(staged):
    result = _run(staged)
    assert result.returncode == 0, result.stdout
    assert "OpenHands staged runtime ready" in result.stdout
    assert (staged.prefix / ".installed-sha256").is_file()


def test_partial_install_is_removed_before_the_attempt(staged):
    """A killed attempt leaves files but no marker; installing on top of it must still succeed."""
    (staged.prefix / "venv").mkdir(parents=True)
    (staged.prefix / "venv" / "half-written").write_text("x")
    result = _run(staged)
    assert result.returncode == 0, result.stdout
    assert "removing partial install" in result.stdout
    assert not (staged.prefix / "venv" / "half-written").exists()
    assert (staged.prefix / "venv" / "bin" / "python").is_file()


def test_finished_install_is_kept(staged):
    assert _run(staged).returncode == 0
    (staged.prefix / "keep-me").write_text("x")
    result = _run(staged)
    assert result.returncode == 0, result.stdout
    assert "removing partial install" not in result.stdout
    assert (staged.prefix / "keep-me").exists()


def test_installer_errors_reach_stdout_and_the_local_copy_is_removed(staged):
    staged.archive.write_bytes(b"truncated")
    result = _run(staged)
    assert result.returncode == 1
    assert "Runtime archive checksum mismatch" in result.stdout
    assert not staged.local.exists()


def test_attempt_is_killed_before_the_exec_deadline():
    assert "timeout -k 10 870 bash -c" in install_command("/s/install.sh", "/s/runtime.tar.gz", timeout_s=900)
    assert "timeout -k 10 60 bash -c" in install_command("/s/install.sh", "/s/runtime.tar.gz", timeout_s=30)


def test_paths_must_be_absolute():
    with pytest.raises(ValueError):
        install_command("install.sh", "/s/runtime.tar.gz")


class FakeSandbox:
    def __init__(self, results):
        self.results = list(results)
        self.commands = []

    async def exec(self, command, timeout_s):
        self.commands.append(command)
        return self.results.pop(0)


def _res(code, stdout=None, stderr=None):
    return SimpleNamespace(return_code=code, stdout=stdout, stderr=stderr)


@pytest.fixture
def no_sleep(monkeypatch):
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(sandbox_runtime.asyncio, "sleep", fake_sleep)
    return slept


async def test_retry_succeeds_after_a_failed_attempt(no_sleep):
    sandbox = FakeSandbox([_res(2, "tar: Unexpected EOF in archive", "exit status 2"), _res(0, "ready")])
    await install_runtime(sandbox, script_path="/s/install.sh", archive_path="/s/runtime.tar.gz", timeout_s=900)
    assert len(sandbox.commands) == 2
    assert sandbox.commands[0] == sandbox.commands[1]
    assert 15 <= no_sleep[0] <= 22.5


async def test_final_error_carries_the_installer_output(no_sleep):
    failures = [_res(1, "Existing /opt/gdpval-openhands does not match this runtime.", "exit status 1")] * 4
    sandbox = FakeSandbox(failures + [_res(0, "listing")])
    with pytest.raises(RuntimeError) as err:
        await install_runtime(sandbox, script_path="/s/install.sh", archive_path="/s/runtime.tar.gz", timeout_s=900)
    assert "does not match this runtime" in str(err.value)
    assert "listing" in str(err.value)
    assert len(no_sleep) == 3
