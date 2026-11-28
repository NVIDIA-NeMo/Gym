# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A replayed prefix command that times out fails the seed, unless the row counts the timeout as a mismatch."""

import subprocess
import time

import pytest

from nemo_gym.sandbox.providers import SandboxExecResult
from resources_servers.agentic_if.episode.opencode_replay import _workspace_diff, restore_workspace
from resources_servers.agentic_if.episode.runtime import BoundSandbox
from resources_servers.agentic_if.episode.workspace import workspace_base
from resources_servers.agentic_if.task_variants.runtime import ContinuationSpec, RuntimeSpec
from resources_servers.agentic_if.tests.support import LocalTransport, exported


class DeadlineTransport(LocalTransport):
    """Runs commands on this host and reports one killed at its deadline as OpenSandbox does (exit 124, timeout)."""

    async def exec(self, command=None, *, timeout_s=None, **kwargs):
        started = time.monotonic()
        result = await super().exec(command, timeout_s=timeout_s, **kwargs)
        if result.return_code == 124 and timeout_s is not None and time.monotonic() - started >= timeout_s:
            return SandboxExecResult(result.stdout, result.stderr, 124, error_type="timeout")
        return result


class UserSwitchTransport(LocalTransport):
    """Refuses every replayed prefix command the way a provider reports a user it cannot switch to."""

    async def exec(self, command=None, **kwargs):
        if command.startswith('"${SHELL:-/bin/sh}" -c'):
            return SandboxExecResult(None, "cannot switch user", 125, error_type="user_switch_unavailable")
        return await super().exec(command, **kwargs)


def bash_part(index: int, command: str, timeout_ms: int) -> dict:
    """A completed bash call that printed nothing when it was recorded."""
    state = {
        "status": "completed",
        "input": {"command": command, "timeout": timeout_ms},
        "output": "",
        "metadata": {},
        "time": {"end": 3},
    }
    return {"type": "tool", "tool": "bash", "callID": f"call-{index}", "state": state}


async def restore(tmp_path, transport, **options):
    """Restore a prefix whose second command outlives its 2 s deadline; the third runs after it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-q"], ["config", "user.email", "t@example.invalid"], ["config", "user.name", "T"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    (repo / "file").write_text("before")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    sandbox = BoundSandbox(transport, user=None, workdir=str(repo), timeout_s=60)
    base = await workspace_base(sandbox)
    (repo / "file").write_text("recorded")
    recorded = await _workspace_diff(sandbox, base=base)
    (repo / "file").write_text("before")
    prefix = exported("root")
    prefix["info"]["directory"] = str(repo)
    prefix["messages"][1]["parts"] = [
        bash_part(0, "printf one > ../one", 120_000),
        bash_part(1, "printf started > ../started; sleep 5; printf late > ../late", 2_000),
        bash_part(2, "printf three > ../three", 120_000),
    ]
    continuation = ContinuationSpec(
        mode="restore",
        prefix_export=prefix,
        workspace_diff=recorded,
        source_task_sha256="a" * 64,
        source_tools_sha256="b" * 64,
        **options,
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    spec = RuntimeSpec(harness="opencode", task_backend="swe")
    return await restore_workspace(sandbox, spec, continuation, base=base, staging=str(staging))


@pytest.mark.asyncio
async def test_a_replayed_timeout_fails_the_restore_by_default(tmp_path):
    with pytest.raises(RuntimeError, match="prefix shell replay failed at the sandbox transport"):
        await restore(tmp_path, DeadlineTransport())
    assert (tmp_path / "started").exists() and not (tmp_path / "three").exists()


@pytest.mark.asyncio
async def test_a_row_can_count_a_replayed_timeout_as_a_mismatch(tmp_path):
    receipt = await restore(tmp_path, DeadlineTransport(), replay_timeouts="mismatch")
    assert receipt == {
        "mode": "restore",
        "replayed_commands": 3,
        "command_output_mismatch_indices": [1],
        "workspace_matches_diff": True,
        "command_timeout_indices": [1],
    }
    # The timed-out command keeps what it did before its deadline, the next one still runs, and the files are the
    # recorded ones.
    assert (tmp_path / "started").exists() and not (tmp_path / "late").exists()
    assert (tmp_path / "three").read_text() == "three"
    assert (tmp_path / "repo" / "file").read_text() == "recorded"


@pytest.mark.asyncio
async def test_other_transport_errors_still_fail_the_restore(tmp_path):
    with pytest.raises(RuntimeError, match="prefix shell replay failed at the sandbox transport"):
        await restore(tmp_path, UserSwitchTransport(), replay_timeouts="mismatch")


def test_the_option_is_sealed_only_when_set():
    fields = {
        "mode": "restore",
        "prefix_export": exported("root"),
        "workspace_diff": "",
        "source_task_sha256": "a" * 64,
        "source_tools_sha256": "b" * 64,
    }
    assert list(ContinuationSpec(**fields).model_dump(mode="json")) == [
        "mode",
        "prefix_export",
        "prefix_events",
        "workspace_diff",
        "source_task_sha256",
        "source_tools_sha256",
    ]
    dumped = ContinuationSpec(**fields, replay_timeouts="mismatch").model_dump(mode="json")
    assert dumped["replay_timeouts"] == "mismatch"
    assert ContinuationSpec.model_validate(dumped).replay_timeouts == "mismatch"
