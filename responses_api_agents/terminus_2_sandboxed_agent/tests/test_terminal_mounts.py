# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from responses_api_agents.terminus_2_sandboxed_agent.app import NeMoGymLLM, NeMoGymTerminus2
from responses_api_agents.terminus_2_sandboxed_agent.terminal import TerminusTmuxSession
from responses_api_agents.terminus_2_sandboxed_agent.terminal_mounts import private_terminal_bootstrap


def make_agent(tmp_path: Path, *, hidden_mounts: list[str] | None = None) -> NeMoGymTerminus2:
    return NeMoGymTerminus2(
        logs_dir=tmp_path,
        model_name="test",
        llm=MagicMock(spec=NeMoGymLLM),
        dump_trajectory=False,
        record_terminal_session=False,
        terminal_hidden_mounts=hidden_mounts,
    )


def make_environment(tmp_path: Path, execute: Any) -> SimpleNamespace:
    return SimpleNamespace(exec=execute, default_user=None, trial_paths=SimpleNamespace(agent_dir=tmp_path))


@pytest.mark.asyncio
async def test_default_setup_starts_tmux_session_without_mount_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = AsyncMock()
    monkeypatch.setattr(TerminusTmuxSession, "start", start)
    environment = make_environment(tmp_path, AsyncMock())

    await make_agent(tmp_path).setup(environment)

    start.assert_awaited_once()
    environment.exec.assert_not_awaited()


@pytest.mark.asyncio
async def test_private_setup_brackets_session_start_with_bootstrap_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    events = []

    async def execute(command, **kwargs):
        events.append((command, kwargs))
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def start(self):
        events.append("session start")

    monkeypatch.setattr(TerminusTmuxSession, "start", start)
    mounts = ["/mnt/s3-data", "/mnt/.s3-gate"]
    await make_agent(tmp_path, hidden_mounts=mounts).setup(make_environment(tmp_path, execute))

    assert events == [
        (private_terminal_bootstrap(mounts), {"user": "root", "timeout_sec": 25}),
        "session start",
        ("tmux kill-session -t gym-internal-mount-bootstrap", {"user": "root"}),
    ]


@pytest.mark.asyncio
async def test_bootstrap_failure_stops_setup_without_using_unisolated_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    start = AsyncMock()
    monkeypatch.setattr(TerminusTmuxSession, "start", start)
    environment = make_environment(
        tmp_path, AsyncMock(return_value=SimpleNamespace(return_code=1, stdout="", stderr="unshare: not permitted"))
    )

    with pytest.raises(RuntimeError, match="unshare: not permitted"):
        await make_agent(tmp_path, hidden_mounts=["/mnt/s3-data"]).setup(environment)

    start.assert_not_awaited()
    assert environment.exec.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("setup_error", [RuntimeError("session start failed"), asyncio.CancelledError()])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_setup_failure_removes_bootstrap_and_preserves_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setup_error: BaseException, cleanup_fails: bool
) -> None:
    monkeypatch.setattr(TerminusTmuxSession, "start", AsyncMock(side_effect=setup_error))
    environment = make_environment(
        tmp_path,
        AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout="", stderr=""),
                SimpleNamespace(return_code=int(cleanup_fails), stdout="", stderr="cleanup error"),
            ]
        ),
    )
    with pytest.raises(type(setup_error)) as error:
        await make_agent(tmp_path, hidden_mounts=["/mnt/s3-data"]).setup(environment)

    assert error.value is setup_error
    assert environment.exec.await_count == 2
    environment.exec.assert_awaited_with("tmux kill-session -t gym-internal-mount-bootstrap", user="root")


@pytest.mark.asyncio
async def test_cleanup_failure_after_successful_setup_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(TerminusTmuxSession, "start", AsyncMock())
    environment = make_environment(
        tmp_path,
        AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout="", stderr=""),
                SimpleNamespace(return_code=1, stdout="", stderr="cannot remove bootstrap"),
            ]
        ),
    )
    with pytest.raises(RuntimeError, match="cannot remove bootstrap"):
        await make_agent(tmp_path, hidden_mounts=["/mnt/s3-data"]).setup(environment)


@pytest.mark.parametrize(
    "path", ["", "relative/mount", "/", "//", "/mnt/../etc", "/mnt//data", "/mnt/data/", "/mnt/\0"]
)
def test_rejects_ambiguous_or_root_mount_paths(path: str) -> None:
    with pytest.raises(ValueError, match="terminal_hidden_mounts"):
        private_terminal_bootstrap([path])


def test_mount_paths_are_passed_as_literal_shell_arguments() -> None:
    mounts = ["/mnt/space and 'quote'", "/mnt/$(touch SHOULD_NOT_EXECUTE)"]
    command = shlex.split(private_terminal_bootstrap(mounts))
    assert command[:5] == ["timeout", "--kill-after=2", "20", "bash", "-c"]
    assert command[6:] == ["bash", *mounts]
