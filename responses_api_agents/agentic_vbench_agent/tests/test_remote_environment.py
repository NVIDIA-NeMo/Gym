# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from harbor.environments.base import ExecResult
from harbor.environments.docker.docker import DockerEnvironment

from responses_api_agents.agentic_vbench_agent.remote_environment import RemoteEnvironment


@pytest.mark.parametrize(
    "detail",
    [
        "netavark (exit code 1): open container netns: No such file or directory",
        "removing container from runtime: /bundle/crun delete --force abc failed: exit status 1",
    ],
)
def test_teardown_retries_missing_runtime_state(monkeypatch: pytest.MonkeyPatch, detail: str) -> None:
    environment = object.__new__(RemoteEnvironment)
    environment.logger = Mock()
    result = ExecResult(return_code=0, stdout="container removed")
    run = AsyncMock(side_effect=[RuntimeError(detail), result])
    sleep = AsyncMock()
    monkeypatch.setattr(DockerEnvironment, "_run_docker_compose_command", run)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    command = ["down", "--rmi", "all", "--volumes", "--remove-orphans"]

    assert asyncio.run(environment._run_docker_compose_command(command, timeout_sec=30)) is result

    assert run.await_count == 2
    for call in run.await_args_list:
        assert call.args == (command,)
        assert call.kwargs == {"check": True, "timeout_sec": 30}
    sleep.assert_awaited_once_with(1)


@pytest.mark.parametrize(
    ("command", "detail", "expected_calls"),
    [
        (["down"], "netavark: No such file or directory", 3),
        (["down"], "permission denied", 1),
        (["exec", "main", "false"], "netavark: No such file or directory", 1),
    ],
)
def test_teardown_keeps_failures_visible_and_does_not_retry_tools(
    monkeypatch: pytest.MonkeyPatch, command: list[str], detail: str, expected_calls: int
) -> None:
    environment = object.__new__(RemoteEnvironment)
    environment.logger = Mock()
    failure = RuntimeError(detail)
    run = AsyncMock(side_effect=failure)
    sleep = AsyncMock()
    monkeypatch.setattr(DockerEnvironment, "_run_docker_compose_command", run)
    monkeypatch.setattr(asyncio, "sleep", sleep)

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(environment._run_docker_compose_command(command))

    assert caught.value is failure
    assert run.await_count == expected_calls
    assert sleep.await_count == expected_calls - 1


def test_unchecked_compose_result_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = object.__new__(RemoteEnvironment)
    environment.logger = Mock()
    result = ExecResult(return_code=1, stdout="netavark: No such file or directory")
    run = AsyncMock(return_value=result)
    monkeypatch.setattr(DockerEnvironment, "_run_docker_compose_command", run)

    assert asyncio.run(environment._run_docker_compose_command(["down"], check=False)) is result
    run.assert_awaited_once_with(["down"], check=False, timeout_sec=None)


@pytest.mark.parametrize(
    ("persistent", "provided", "expected"),
    [
        ({}, None, {"TAR_OPTIONS": "--no-same-owner"}),
        ({"BASE": "kept"}, {"CALL": "kept"}, {"BASE": "kept", "CALL": "kept", "TAR_OPTIONS": "--no-same-owner"}),
        (
            {"TAR_OPTIONS": "--no-same-owner --warning=no-timestamp"},
            None,
            {"TAR_OPTIONS": "--no-same-owner --warning=no-timestamp"},
        ),
        ({"TAR_OPTIONS": "--no-same-owner"}, {"TAR_OPTIONS": ""}, {"TAR_OPTIONS": ""}),
    ],
)
def test_exec_uses_rootless_tar_default_and_preserves_explicit_options(
    monkeypatch: pytest.MonkeyPatch,
    persistent: dict[str, str],
    provided: dict[str, str] | None,
    expected: dict[str, str],
) -> None:
    environment = object.__new__(RemoteEnvironment)
    environment._persistent_env = dict(persistent)
    provided_before = dict(provided) if provided is not None else None
    result = ExecResult(return_code=127, stdout="installer failed")
    run = AsyncMock(return_value=result)
    monkeypatch.setattr(DockerEnvironment, "exec", run)

    assert (
        asyncio.run(environment.exec("tar -xf node.tar.xz", cwd="/tmp", env=provided, timeout_sec=60, user="root"))
        is result
    )

    run.assert_awaited_once_with("tar -xf node.tar.xz", cwd="/tmp", env=expected, timeout_sec=60, user="root")
    assert environment._persistent_env == persistent
    assert provided == provided_before
