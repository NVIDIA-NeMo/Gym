# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The session's sandbox side: the locked runtime, install and reuse, staging, and what is collected."""

import dataclasses
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from packaging.requirements import Requirement

from nemo_gym.agents import AgentSeedRejectedError, AgentSeedSessionRequest, SandboxSession
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


def test_the_lock_pins_every_direct_requirement_at_its_requested_version() -> None:
    pins = {Requirement(line).name: Requirement(line).specifier for line in MINI_SWE_RUNTIME.pins}
    # A requirements.in bump without regenerating requirements.txt would make the host build unsatisfiable at seed.
    for line in MINI_SWE_RUNTIME.requirements:
        requirement = Requirement(line)
        assert pins.get(requirement.name) == requirement.specifier, requirement


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


def sandbox_platform(monkeypatch, *, libc: str = "gnu", glibc_minor: int | None = 35) -> None:
    platform = sandbox_python.SandboxPlatform(arch="x86_64", libc=libc, glibc_minor=glibc_minor)
    monkeypatch.setattr(sandbox_module, "probe_sandbox_platform", AsyncMock(return_value=platform))


async def test_install_builds_the_runtime_when_none_passes_its_check(state, monkeypatch) -> None:
    sandbox_platform(monkeypatch)
    monkeypatch.setattr(type(MINI_SWE_RUNTIME), "is_installed", AsyncMock(return_value=False))
    install = AsyncMock()
    monkeypatch.setattr(type(MINI_SWE_RUNTIME), "install", install)

    await state.install(timeout=60)

    install.assert_awaited_once_with(state.session.sandbox, timeout_s=60)


async def test_a_failed_runtime_install_fails_the_seed_before_staging(state, monkeypatch) -> None:
    sandbox_platform(monkeypatch)
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
    # mini-SWE makes its config directory at import: in the check's own HOME, never in the task's XDG_CONFIG_HOME.
    assert "MSWEA_GLOBAL_CONFIG_DIR=os.path.join(os.environ['HOME'], 'mini-swe-agent')" in check


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [("", {}), ('{"mcp_servers": [], "skills_dir": "/skills"}', {"mcp_servers": [], "skills_dir": "/skills"})],
)
async def test_the_task_context_is_read_when_present(state, stdout, expected) -> None:
    state.session.sandbox.exec.return_value = result(stdout=stdout)

    assert await state.read_task_context() == expected


@pytest.mark.parametrize(
    ("response", "error", "match"),
    [
        (result(error_type="timeout"), RuntimeError, "could not read"),
        (result(return_code=2), RuntimeError, "could not read"),
        # The task's own data is wrong: the seed is refused instead of retried.
        (result(stdout="{oops"), AgentSeedRejectedError, "not JSON"),
        (result(stdout="[1]"), AgentSeedRejectedError, "not a JSON object"),
    ],
)
async def test_an_unreadable_task_context_fails_instead_of_dropping_the_tasks_tools(
    state, response, error, match
) -> None:
    state.session.sandbox.exec.return_value = response

    with pytest.raises(error, match=match):
        await state.read_task_context()


async def test_the_task_context_read_tells_an_absent_file_from_one_it_cannot_read(
    state, monkeypatch, tmp_path
) -> None:
    async def shell(command, **kwargs):
        ran = subprocess.run(["bash", "-c", command], capture_output=True, text=True, check=False)
        return result(return_code=ran.returncode, stdout=ran.stdout)

    state.session.sandbox.exec = AsyncMock(side_effect=shell)
    path = tmp_path / "task.json"
    monkeypatch.setattr(sandbox_module, "TASK_CONTEXT_FILE", str(path))

    assert await state.read_task_context() == {}
    path.write_text('{"skills_dir": "/skills"}')
    assert await state.read_task_context() == {"skills_dir": "/skills"}
    # A path that exists but cannot be read (a directory: unreadable even for root) must not pass for an absent file.
    path.unlink()
    path.mkdir()
    with pytest.raises(RuntimeError, match="could not read"):
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
    # The runtime's interpreter, linked under a name without "python": a task's `pkill -9 -f python` must not match
    # the supervisor, whose command line carries this argv.
    interpreter = f"{directory}/bin/mini-swe"
    [link] = [call.args[0] for call in state.session.sandbox.exec.await_args_list if " ln -sfn " in call.args[0]]
    assert link.endswith(f"ln -sfn {MINI_SWE_RUNTIME.python} {interpreter}")
    assert not any("python" in word for word in command.argv)
    # -E -s: the task's PYTHON* settings and user site-packages never reach the harness.
    assert command.argv == [
        interpreter,
        "-E",
        "-s",
        f"{directory}/sandbox_runner.py",
        f"{directory}/input.json",
        f"{directory}/output.json",
    ]


async def test_staging_without_mcp_servers_writes_no_server_file(state, monkeypatch) -> None:
    uploads = {}

    async def upload_text(sandbox, *, path, text):
        uploads[path] = text

    monkeypatch.setattr(sandbox_module, "upload_text", upload_text)

    await state.stage_activation({"task": "t"})

    assert list(uploads) == [f"{state.session.session_dir}/input.json"]


async def test_collect_copies_what_the_run_left_and_tolerates_what_it_did_not(state, monkeypatch) -> None:
    state.mcp_servers = [{"name": "s"}]
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
    assert artifacts.invalid_mcp_calls == 0
    assert artifacts.log == "[output.log]\nsupervisor log\n[mcp.log]\nmcp started"


def stage_mcp_calls(state: MiniSWESession, monkeypatch, lines: list[str]) -> None:
    """Leave ``mcp_calls.jsonl`` with these lines in the session directory, and no other run file."""
    path = f"{state.session.session_dir}/mcp_calls.jsonl"

    async def exec_(command, **kwargs):
        if command.startswith("test -f "):
            return result(return_code=0 if command.removeprefix("test -f ").strip("'") == path else 1)
        return result()

    async def read_text(sandbox, *, path):
        return "".join(line + "\n" for line in lines)

    state.session.sandbox.exec = AsyncMock(side_effect=exec_)
    monkeypatch.setattr(sandbox_module, "read_text", read_text)


def mcp_record(server: str = "task_tools", tool_call_id: str | None = "call_1") -> str:
    return json.dumps(
        {"tool_call_id": tool_call_id, "server": server, "tool": "get_weather", "arguments": {}, "output": "{}"}
    )


async def test_a_run_without_mcp_servers_reports_no_mcp_calls_whatever_its_commands_wrote(state, monkeypatch) -> None:
    # The model's commands can write the file too; a run with no MCP servers made no MCP call.
    stage_mcp_calls(state, monkeypatch, [mcp_record()])

    artifacts = await state.collect()

    assert (artifacts.mcp_calls, artifacts.invalid_mcp_calls) == ([], 0)


async def test_collect_keeps_every_mcp_call_record(state, monkeypatch) -> None:
    state.mcp_servers = [{"name": "task_tools"}]
    stage_mcp_calls(
        state,
        monkeypatch,
        [
            mcp_record(),
            # A call from a process the model left running in the background, outside any bash call.
            mcp_record(tool_call_id=None),
            # The CLI records a call to a server the run was not granted as a failed call; the model made it.
            mcp_record(server="example_single_tool_call"),
        ],
    )

    artifacts = await state.collect()

    assert [(call["server"], call["tool_call_id"]) for call in artifacts.mcp_calls] == [
        ("task_tools", "call_1"),
        ("task_tools", None),
        ("example_single_tool_call", "call_1"),
    ]
    assert artifacts.invalid_mcp_calls == 0


async def test_malformed_mcp_records_are_counted_without_failing_the_run(state, monkeypatch) -> None:
    state.mcp_servers = [{"name": "task_tools"}]
    # A malformed line, a line that is not an object, and a last line cut short by a killed process.
    stage_mcp_calls(state, monkeypatch, [mcp_record(), "{oops", "[1]", mcp_record()[:20]])

    artifacts = await state.collect()

    assert [call["tool_call_id"] for call in artifacts.mcp_calls] == ["call_1"]
    assert artifacts.invalid_mcp_calls == 3


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


@pytest.mark.parametrize("runtime", [None, {"hostname": "box", "pid": "7"}, {"hostname": "box", "pid": 7, "extra": 1}])
async def test_malformed_runner_diagnostics_are_dropped_without_failing_the_run(state, monkeypatch, runtime) -> None:
    directory = state.session.session_dir
    output = {"runtime": runtime, "exit_status": "Submitted", "error": None}

    async def read_text(sandbox, *, path):
        assert path == f"{directory}/output.json"
        return json.dumps(output)

    async def exec_(command, **kwargs):
        return result(return_code=0 if "output.json" in command else 1)

    state.session.sandbox.exec = AsyncMock(side_effect=exec_)
    monkeypatch.setattr(sandbox_module, "read_text", read_text)

    artifacts = await state.collect()

    assert artifacts.output == output
    assert state.runtime_info is None


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
@pytest.mark.parametrize("ensure_ascii", [False, True])
async def test_mcp_jsonl_preserves_unicode_record_content(state, monkeypatch, separator, ensure_ascii):
    state.mcp_servers = [{"name": "task_tools"}]
    record = {
        "server": "task_tools",
        "tool": "echo",
        "tool_call_id": "call_1",
        "arguments": {"text": f"left{separator}right"},
        "output": f"left{separator}right",
    }
    stage_mcp_calls(state, monkeypatch, [json.dumps(record, ensure_ascii=ensure_ascii) + "\r", ""])

    artifacts = await state.collect()

    assert artifacts.mcp_calls == [record]
    assert artifacts.invalid_mcp_calls == 0


async def test_an_mcp_record_file_too_large_to_read_is_reported_as_unread(state, monkeypatch) -> None:
    state.mcp_servers = [{"name": "task_tools"}]
    read = AsyncMock()

    async def exec_(command, **kwargs):
        if command.startswith("wc -c"):
            return result(stdout=f"{sandbox_module.MCP_CALLS_MAX_BYTES + 1}\n")
        return result(return_code=1 if command.startswith("test -f ") else 0)

    state.session.sandbox.exec = AsyncMock(side_effect=exec_)
    monkeypatch.setattr(sandbox_module, "read_text", read)

    assert await state._read_mcp_calls(f"{state.session.session_dir}/mcp_calls.jsonl") == ([], 0, True)
    read.assert_not_awaited()


async def test_an_mcp_call_keeps_a_bounded_output(state, monkeypatch) -> None:
    state.mcp_servers = [{"name": "task_tools"}]
    record = json.loads(mcp_record())
    record["output"] = "x" * (sandbox_module.MCP_CALL_OUTPUT_MAX_CHARS + 10)
    stage_mcp_calls(state, monkeypatch, [json.dumps(record)])

    artifacts = await state.collect()

    [call] = artifacts.mcp_calls
    assert call["output"].startswith("x" * sandbox_module.MCP_CALL_OUTPUT_MAX_CHARS)
    assert call["output"].endswith("[10 characters omitted by NeMo Gym]")
    assert not artifacts.mcp_calls_unread


@pytest.mark.parametrize(("libc", "glibc_minor"), [("gnu", 27), ("gnu", 17)])
async def test_a_sandbox_too_old_for_the_runtime_is_a_final_refusal(state, monkeypatch, libc, glibc_minor) -> None:
    sandbox_platform(monkeypatch, libc=libc, glibc_minor=glibc_minor)
    monkeypatch.setattr(type(MINI_SWE_RUNTIME), "is_installed", AsyncMock(return_value=False))
    install = AsyncMock()
    monkeypatch.setattr(type(MINI_SWE_RUNTIME), "install", install)

    # No attempt can change the image, so the seed is refused once instead of failing every attempt with a 500.
    with pytest.raises(AgentSeedRejectedError, match=f"glibc 2.{glibc_minor}"):
        await state.install(timeout=60)
    install.assert_not_awaited()


@pytest.mark.parametrize(("libc", "glibc_minor"), [("gnu", 28), ("gnu", None), ("musl", None)])
async def test_a_sandbox_that_can_run_the_runtime_installs_it(state, monkeypatch, libc, glibc_minor) -> None:
    sandbox_platform(monkeypatch, libc=libc, glibc_minor=glibc_minor)
    monkeypatch.setattr(type(MINI_SWE_RUNTIME), "is_installed", AsyncMock(return_value=False))
    install = AsyncMock()
    monkeypatch.setattr(type(MINI_SWE_RUNTIME), "install", install)

    await state.install(timeout=60)

    install.assert_awaited_once()


async def test_staging_fails_when_the_interpreter_link_cannot_be_made(state, monkeypatch) -> None:
    monkeypatch.setattr(sandbox_module, "upload_text", AsyncMock())
    state.session.sandbox.exec.return_value = result(return_code=1)

    with pytest.raises(RuntimeError, match="link the mini-SWE interpreter"):
        await state.stage_activation({"task": "t"})
