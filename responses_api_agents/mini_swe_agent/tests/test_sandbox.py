# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The session's sandbox side: the locked runtime, install and reuse, staging, and what is collected."""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from packaging.requirements import Requirement

from nemo_gym.agents import AgentSeedSessionRequest, SandboxSession
from nemo_gym.agents.utils import sandbox_python
from nemo_gym.core import EpisodeId, TaskId
from responses_api_agents.mini_swe_agent import sandbox as sandbox_module
from responses_api_agents.mini_swe_agent.sandbox import MINI_SWE_RUNTIME, MINI_SWE_VERSION, MiniSWESession


RUNTIME_DIR = Path(sandbox_module.__file__).with_name("runtime")


def result(return_code: int = 0, stdout: str = "", error_type: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(return_code=return_code, stdout=stdout, stderr="", error_type=error_type)


@pytest.fixture
def state() -> MiniSWESession:
    box = SimpleNamespace(sandbox_id="box", exec=AsyncMock(return_value=result()), upload=AsyncMock())
    return MiniSWESession(
        request=AgentSeedSessionRequest(
            agent_session_id="session",
            episode_id=EpisodeId(rollout_id="rollout"),
            task_id=TaskId(taskset="test", task_id="task"),
        ),
        provider_name="sandbox",
        sandbox_id="box",
        session=SandboxSession(
            sandbox=box, workdir="/app", session_dir="/tmp/nemo-gym-mini-swe-sessions/s", harness="mini-SWE"
        ),
    )


def test_the_runtime_is_the_pinned_harness_locked_independently_of_this_server(monkeypatch) -> None:
    requirements = {Requirement(line).name: str(Requirement(line).specifier) for line in MINI_SWE_RUNTIME.requirements}
    assert requirements == {"mini-swe-agent": f"=={MINI_SWE_VERSION}", "litellm": "==1.106.0", "mcp": "==2.2.0"}
    pins = {Requirement(line).name: Requirement(line) for line in MINI_SWE_RUNTIME.pins}
    # Every transitive dependency is pinned exactly, so an episode never resolves a different closure.
    assert all(str(pin.specifier).startswith("==") for pin in pins.values())
    assert {"litellm", "mcp", "openai", "pydantic", "httpx2", "tenacity"} <= set(pins)
    assert str(pins["mini-swe-agent"].specifier) == f"=={MINI_SWE_VERSION}"
    # mini-SWE's batch benchmark runners need datasets; an agent session never imports it.
    assert MINI_SWE_RUNTIME.overrides == ("datasets; sys_platform == 'never'",)
    # The SWE-Gym fork /run uses lives in this server's environment and never shapes the sandbox runtime.
    directory = MINI_SWE_RUNTIME.directory
    monkeypatch.setattr(sandbox_python, "_agent_environment_pins", lambda: ("litellm==1.74.15.post1",))
    assert MINI_SWE_RUNTIME.pins == sandbox_module._requirement_lines(RUNTIME_DIR / "requirements.txt")
    assert dataclasses.replace(MINI_SWE_RUNTIME).directory == directory


def test_the_lock_was_compiled_from_the_requirements_and_overrides() -> None:
    header = (RUNTIME_DIR / "requirements.txt").read_text().splitlines()[1]
    assert "uv pip compile requirements.in --override overrides.txt --universal" in header


async def test_install_reuses_a_runtime_that_passes_its_check_and_stages_the_harness_files(state, monkeypatch) -> None:
    monkeypatch.setattr(type(MINI_SWE_RUNTIME), "is_installed", AsyncMock(return_value=True))
    install = AsyncMock()
    monkeypatch.setattr(type(MINI_SWE_RUNTIME), "install", install)

    await state.install(timeout=60)

    install.assert_not_awaited()
    uploads = [call.args[1] for call in state.session.sandbox.upload.await_args_list]
    assert uploads == [
        "/tmp/nemo-gym-mini-swe-sessions/s/sandbox_runner.py",
        "/tmp/nemo-gym-mini-swe-sessions/s/mcp_cli.py",
    ]


async def test_install_builds_the_runtime_when_none_passes_its_check(state, monkeypatch) -> None:
    monkeypatch.setattr(type(MINI_SWE_RUNTIME), "is_installed", AsyncMock(return_value=False))
    install = AsyncMock()
    monkeypatch.setattr(type(MINI_SWE_RUNTIME), "install", install)

    await state.install(timeout=60)

    install.assert_awaited_once_with(state.session.sandbox, timeout_s=60)


async def test_a_failed_runtime_install_fails_the_seed_before_staging(state, monkeypatch) -> None:
    monkeypatch.setattr(type(MINI_SWE_RUNTIME), "is_installed", AsyncMock(return_value=False))
    monkeypatch.setattr(
        type(MINI_SWE_RUNTIME), "install", AsyncMock(side_effect=RuntimeError("no wheel for glibc 2.17"))
    )

    with pytest.raises(RuntimeError, match="glibc"):
        await state.install(timeout=60)
    state.session.sandbox.upload.assert_not_awaited()


async def test_a_session_directory_that_cannot_be_made_fails_the_seed(state) -> None:
    state.session.sandbox.exec.return_value = result(return_code=1)

    with pytest.raises(RuntimeError, match="session directory"):
        await state.install(timeout=60)


def test_the_import_check_verifies_the_pinned_version_and_the_clients_the_run_uses() -> None:
    check = MINI_SWE_RUNTIME.import_check
    assert f"minisweagent.__version__ == {MINI_SWE_VERSION!r}" in check
    assert "LitellmResponseModel" in check and "streamable_http_client" in check


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [("", {}), ('{"mcp_servers": [], "skills_dir": "/skills"}', {"mcp_servers": [], "skills_dir": "/skills"})],
)
async def test_the_task_context_is_read_when_present(state, stdout, expected) -> None:
    state.session.sandbox.exec.return_value = result(stdout=stdout)

    assert await state.read_task_context() == expected


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (result(error_type="timeout"), "could not read"),
        (result(return_code=2), "could not read"),
        (result(stdout="{oops"), "not JSON"),
        (result(stdout="[1]"), "not a JSON object"),
    ],
)
async def test_an_unreadable_task_context_fails_instead_of_dropping_the_tasks_tools(state, response, match) -> None:
    state.session.sandbox.exec.return_value = response

    with pytest.raises(RuntimeError, match=match):
        await state.read_task_context()


async def test_staging_writes_the_input_and_mcp_servers_and_runs_the_runner_with_the_private_python(
    state, monkeypatch
) -> None:
    uploads = {}

    async def upload_text(sandbox, *, path, text):
        uploads[path] = text

    monkeypatch.setattr(sandbox_module, "upload_text", upload_text)
    state.mcp_servers = [{"name": "task_tools", "url": "http://x/mcp", "headers": {"Authorization": "Bearer t"}}]

    command = await state.stage_activation({"task": "t"})

    directory = state.session.session_dir
    assert json.loads(uploads[f"{directory}/mcp_servers.json"])[0]["headers"] == {"Authorization": "Bearer t"}
    assert json.loads(uploads[f"{directory}/input.json"]) == {"task": "t"}
    # -E -s: the task's PYTHON* settings and user site-packages never reach the harness.
    assert command.argv == [
        MINI_SWE_RUNTIME.python,
        "-E",
        "-s",
        f"{directory}/sandbox_runner.py",
        f"{directory}/input.json",
        f"{directory}/output.json",
    ]
    assert not command.stdlib_runner


async def test_staging_without_mcp_servers_writes_no_server_file(state, monkeypatch) -> None:
    uploads = {}

    async def upload_text(sandbox, *, path, text):
        uploads[path] = text

    monkeypatch.setattr(sandbox_module, "upload_text", upload_text)

    await state.stage_activation({"task": "t"})

    assert list(uploads) == [f"{state.session.session_dir}/input.json"]


async def test_collect_copies_what_the_run_left_and_tolerates_what_it_did_not(state, monkeypatch) -> None:
    directory = state.session.session_dir
    files = {
        f"{directory}/trajectory.json": json.dumps({"info": {"exit_status": "Submitted"}, "messages": []}),
        f"{directory}/mcp_calls.jsonl": json.dumps({"server": "s", "tool": "t", "output": "{}"}) + "\n\n",
    }

    async def exec_(command, **kwargs):
        if command.startswith("test -f "):
            path = command.removeprefix("test -f ").strip("'")
            return result(return_code=0 if path in files else 1)
        if command.startswith("tail -c"):
            return result(stdout="mcp started" if "mcp.log" in command else "supervisor log")
        return result()

    async def read_text(sandbox, *, path):
        return files[path]

    state.session.sandbox.exec = AsyncMock(side_effect=exec_)
    monkeypatch.setattr(sandbox_module, "read_text", read_text)

    artifacts = await state.collect()

    assert artifacts.trajectory["info"]["exit_status"] == "Submitted"
    # The runner was killed before writing its account of the run.
    assert artifacts.output is None and state.runtime_info is None
    assert artifacts.mcp_calls == [{"server": "s", "tool": "t", "output": "{}"}]
    assert artifacts.log == "[output.log]\nsupervisor log\n[mcp.log]\nmcp started"


async def test_collect_fails_when_the_sandbox_cannot_say_whether_a_file_exists(state) -> None:
    state.session.sandbox.exec.return_value = result(error_type="timeout")

    with pytest.raises(RuntimeError, match="Could not check"):
        await state.collect()


async def test_collect_reads_the_runners_process_identity(state, monkeypatch) -> None:
    directory = state.session.session_dir
    output = {"runtime": {"hostname": "box", "pid": 42, "python": "/rt/python/bin/python3"}}

    async def read_text(sandbox, *, path):
        if path == f"{directory}/output.json":
            return json.dumps(output)
        raise AssertionError(path)

    async def exec_(command, **kwargs):
        return result(return_code=0 if command.endswith("output.json'") or command.endswith("output.json") else 1)

    state.session.sandbox.exec = AsyncMock(side_effect=exec_)
    monkeypatch.setattr(sandbox_module, "read_text", read_text)

    artifacts = await state.collect()

    assert artifacts.output == output
    assert (state.runtime_info.hostname, state.runtime_info.pid) == ("box", 42)
