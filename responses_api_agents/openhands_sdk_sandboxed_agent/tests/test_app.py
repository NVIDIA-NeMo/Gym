# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from pytest import MonkeyPatch

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.core import AGENT_ROUTE_FAILED, NgFailureEvent
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.openhands_sdk_sandboxed_agent import app as app_module
from responses_api_agents.openhands_sdk_sandboxed_agent.app import (
    OpenHandsSDKSandboxedAgent,
    OpenHandsSDKSandboxedAgentConfig,
)


class _Request:
    """The parts of a Request that run() reads and rebinds: `run` sets `_cookies`, `responses` reads `cookies`."""

    def __init__(self) -> None:
        self.session = {SESSION_ID_KEY: "session"}
        self.state = SimpleNamespace()
        self._cookies: dict = {}

    @property
    def cookies(self) -> dict:
        return self._cookies


def _agent() -> OpenHandsSDKSandboxedAgent:
    config = OpenHandsSDKSandboxedAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        model_server=ModelServerRef(type="responses_api_models", name="model"),
    )
    return OpenHandsSDKSandboxedAgent(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))


def _exec(return_code: int = 0, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(return_code=return_code, stdout=stdout, stderr=stderr, error_type=None)


@pytest.fixture
def world(tmp_path, monkeypatch: MonkeyPatch):
    """An agent whose seed returns a sandbox handle and whose sandbox is a mock; /verify calls are recorded."""
    agent = _agent()
    sandbox = AsyncMock()
    sandbox.exec.return_value = _exec(0, "OPENHANDS_INSTALLED")
    monkeypatch.setattr(app_module, "__file__", str(tmp_path / "app.py"))  # results/ goes under tmp_path
    monkeypatch.setattr(OpenHandsSDKSandboxedAgent, "_connect_sandbox", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(OpenHandsSDKSandboxedAgent, "rollout_id_from_run", lambda self, body: "rollout")
    monkeypatch.setattr(
        OpenHandsSDKSandboxedAgent, "resolve_model_base_url", lambda self, name, rollout_id: "http://model/v1"
    )
    monkeypatch.setattr(app_module, "raise_for_status", AsyncMock())
    seed = AsyncMock()
    seed.cookies = {}
    seed.json.return_value = {"sandbox_handle": "sandbox-1"}
    posts: list[str] = []

    async def post(*, url_path: str, **kwargs):
        posts.append(url_path)
        if url_path == "/seed_session":
            return seed
        raise AssertionError(f"unexpected call to {url_path}")

    agent.server_client.post = post
    request = _Request()
    body = MagicMock()
    body.model_dump.return_value = {}
    body.responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "user", "content": "build it"}]
    )
    return SimpleNamespace(agent=agent, sandbox=sandbox, posts=posts, request=request, body=body)


async def test_install_failure_is_a_failed_setup_and_is_never_verified(world) -> None:
    world.sandbox.exec.return_value = _exec(1, "", "no network")

    with pytest.raises(NgFailureEvent) as caught:
        await world.agent.run(world.request, world.body)

    assert caught.value.failure_kind == AGENT_ROUTE_FAILED
    assert caught.value.operation == "openhands.setup"
    assert "install failed" in caught.value.failure_reason
    assert world.posts == ["/seed_session"]  # no /verify: an untouched workspace must not score as a 0
    world.sandbox.stop.assert_awaited_once()
    assert world.agent._sandbox_id_to_sandbox == {}


@pytest.mark.parametrize(
    "install",
    [
        _exec(0, "pip finished but no marker"),  # no OPENHANDS_INSTALLED marker
        SimpleNamespace(return_code=0, stdout="OPENHANDS_INSTALLED", stderr="", error_type="sandbox"),
        SimpleNamespace(return_code=125, stdout="", stderr="", error_type="timeout"),
    ],
)
async def test_every_kind_of_install_failure_is_a_failed_setup(world, install) -> None:
    world.sandbox.exec.return_value = install

    with pytest.raises(NgFailureEvent) as caught:
        await world.agent.run(world.request, world.body)

    assert caught.value.operation == "openhands.setup"
    assert world.posts == ["/seed_session"]
    world.sandbox.stop.assert_awaited_once()


PARTIAL_RESULT = {
    "messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "build it"},
        {"role": "assistant", "content": "half done"},
    ],
    "partial": True,
    "finished": False,
}


def runner_sandbox(
    world,
    tmp_path,
    *,
    launch=None,
    confirm=None,
    poll="RUNNING",
    quiesce="QUIESCED",
    result=None,
    foreground=None,
    pid=None,
):
    """Script the sandbox like the real one: install ok, a detached launch that prints `launched`, a pid file, polls,
    and the stop-the-runner command. `events` records the order of the calls that matter."""
    events: list[str] = []
    launches: list[dict] = []
    commands: list[str] = []
    polls = list(poll) if isinstance(poll, (list, tuple)) else [poll]

    async def exec_(command: str, **kwargs):
        commands.append(command)
        if "OPENHANDS_INSTALLED" in command:
            return _exec(0, "OPENHANDS_INSTALLED")
        if "setsid -w" in command:  # the foreground run, in a session of its own
            launches.append(kwargs)
            events.append("foreground")
            if isinstance(foreground, BaseException):
                raise foreground
            return foreground or _exec(0, "OpenHands run finished")
        if command.startswith("cat ") and command.endswith("/pid"):
            events.append("pid")
            return pid or _exec(0, "4242")
        if "setsid -f" in command:
            launches.append(kwargs)
            events.append("launch")
            if isinstance(launch, BaseException):
                raise launch
            return launch or _exec(0, "launched")
        if "seq 1 60" in command:
            events.append("confirm")
            return confirm or _exec(0, "4242")
        if "elif [ -d /proc/" in command:
            item = polls.pop(0) if len(polls) > 1 else polls[0]
            return _exec(0, item) if isinstance(item, str) else item
        if "kill -TERM" in command or "pkill" in command:
            events.append("quiesce")
            if isinstance(quiesce, BaseException):
                raise quiesce
            return _exec(0, quiesce) if isinstance(quiesce, str) else quiesce
        return _exec(0)

    async def download(remote: str, local):
        if remote.endswith("result.json"):
            events.append("download-result")
            if result is not None:
                Path(local).write_text(json.dumps(result))
            return
        events.append("download-log")
        Path(local).write_text("OpenHands run log")

    world.sandbox.exec.side_effect = exec_
    world.sandbox.download = AsyncMock(side_effect=download)
    world.agent._sandbox_id_to_sandbox["session"] = world.sandbox
    world.request._cookies = {"sandbox_id": "session"}
    return SimpleNamespace(events=events, launches=launches, commands=commands)


async def call_responses(world, body=None):
    return await world.agent.responses(world.request, body or world.body.responses_create_params)


def request_json(tmp_path) -> dict:
    return json.loads((tmp_path / "results" / "session" / "request.json").read_text())


async def test_a_confirmed_runner_that_hits_its_deadline_is_stopped_then_graded(
    world, tmp_path, monkeypatch: MonkeyPatch
) -> None:
    """A runner that started and ran past its budget is quiesced before anything is collected, then graded."""
    monkeypatch.setattr(world.agent.config, "sandbox_timeout", 0.05)
    monkeypatch.setattr(world.agent.config, "poll_interval_s", 0)
    script = runner_sandbox(world, tmp_path, poll="RUNNING", result=PARTIAL_RESULT)

    response = await call_responses(world)

    assert response.output  # the partial transcript is kept
    run_result = world.agent._sandbox_id_to_run_result["session"]
    assert "sandbox_timeout" in run_result["openhands_poll_error"]
    assert run_result["openhands_partial"] is True
    # confirmed started, then stopped and confirmed gone, and only then is the result collected
    assert script.events.index("confirm") < script.events.index("quiesce") < script.events.index("download-result")
    assert any("pid=4242" in command and "kill -TERM -- -$pid" in command for command in script.commands)


async def test_a_runner_that_exited_on_its_own_is_not_killed(world, tmp_path) -> None:
    script = runner_sandbox(world, tmp_path, poll="0", result={**PARTIAL_RESULT, "partial": False, "finished": True})

    await call_responses(world)

    assert "quiesce" not in script.events
    assert world.agent._sandbox_id_to_run_result["session"]["openhands_finished"] is True


@pytest.mark.parametrize(
    "launch",
    [
        RuntimeError("provider gone"),
        SimpleNamespace(return_code=0, stdout="launched", stderr="", error_type="sandbox"),
        SimpleNamespace(return_code=1, stdout="", stderr="setsid: not found", error_type=None),
    ],
)
async def test_a_runner_launch_failure_is_ungraded(world, tmp_path, launch) -> None:
    runner_sandbox(world, tmp_path, launch=launch)

    with pytest.raises(NgFailureEvent) as caught:
        await world.agent.run(world.request, world.body)

    assert caught.value.failure_kind == AGENT_ROUTE_FAILED
    assert caught.value.operation == "openhands.execute"
    assert world.posts == ["/seed_session"]  # never verified
    world.sandbox.stop.assert_awaited_once()


async def test_an_unconfirmed_launch_is_ungraded(world, tmp_path) -> None:
    """`launched` is printed even when the runner never started: only the pid file confirms it."""
    runner_sandbox(world, tmp_path, confirm=_exec(1, ""))

    with pytest.raises(NgFailureEvent) as caught:
        await world.agent.run(world.request, world.body)

    assert caught.value.operation == "openhands.execute"
    assert "did not start" in caught.value.failure_reason
    assert world.posts == ["/seed_session"]
    world.sandbox.stop.assert_awaited_once()


STOP_FAILURES = [
    "ALIVE",  # the marker says something is still running
    "",  # no QUIESCED marker at all
    SimpleNamespace(return_code=125, stdout="QUIESCED", stderr="", error_type="sandbox"),  # provider error
    SimpleNamespace(return_code=1, stdout="QUIESCED", stderr="", error_type=None),
    TimeoutError("the sandbox cannot be reached"),
]


@pytest.mark.parametrize("quiesce", STOP_FAILURES)
async def test_a_runner_that_cannot_be_confirmed_stopped_is_ungraded(
    world, tmp_path, monkeypatch: MonkeyPatch, quiesce
) -> None:
    monkeypatch.setattr(world.agent.config, "sandbox_timeout", 0.05)
    monkeypatch.setattr(world.agent.config, "poll_interval_s", 0)
    script = runner_sandbox(world, tmp_path, poll="RUNNING", quiesce=quiesce, result=PARTIAL_RESULT)

    with pytest.raises(NgFailureEvent) as caught:
        await world.agent.run(world.request, world.body)

    assert caught.value.operation == "openhands.execute"
    assert world.posts == ["/seed_session"]  # skipped /verify
    assert "download-result" not in script.events  # nothing collected from a runner that may still be writing
    world.sandbox.stop.assert_awaited_once()  # and the sandbox is still torn down


@pytest.mark.parametrize("quiesce", STOP_FAILURES)
async def test_a_foreground_runner_that_cannot_be_confirmed_stopped_is_ungraded(
    world, tmp_path, monkeypatch: MonkeyPatch, quiesce
) -> None:
    monkeypatch.setattr(world.agent.config, "background_exec", False)
    timed_out = SimpleNamespace(return_code=124, stdout="partial log", stderr="", error_type="timeout")
    script = runner_sandbox(world, tmp_path, foreground=timed_out, quiesce=quiesce, result=PARTIAL_RESULT)

    with pytest.raises(NgFailureEvent) as caught:
        await world.agent.run(world.request, world.body)

    assert caught.value.operation == "openhands.execute"
    assert world.posts == ["/seed_session"]
    assert "download-result" not in script.events
    world.sandbox.stop.assert_awaited_once()


@pytest.mark.parametrize(
    "outcome",
    [
        TimeoutError("exec lost"),
        SimpleNamespace(return_code=1, stdout="", stderr="", error_type="sandbox"),
    ],
)
async def test_a_foreground_exec_failure_is_ungraded(world, tmp_path, monkeypatch: MonkeyPatch, outcome) -> None:
    monkeypatch.setattr(world.agent.config, "background_exec", False)
    runner_sandbox(world, tmp_path)

    async def exec_(command: str, **kwargs):
        if "OPENHANDS_INSTALLED" in command:
            return _exec(0, "OPENHANDS_INSTALLED")
        if "runner.py" in command:
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return _exec(0)

    world.sandbox.exec.side_effect = exec_

    with pytest.raises(NgFailureEvent) as caught:
        await world.agent.run(world.request, world.body)

    assert caught.value.operation == "openhands.execute"
    assert world.posts == ["/seed_session"]


async def test_a_foreground_run_runs_in_its_own_session_and_a_timeout_stops_it_before_grading(
    world, tmp_path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(world.agent.config, "background_exec", False)
    timed_out = SimpleNamespace(return_code=124, stdout="partial log", stderr="", error_type="timeout")
    script = runner_sandbox(world, tmp_path, foreground=timed_out, result=PARTIAL_RESULT)

    response = await call_responses(world)

    assert response.output  # confirmed stopped, so the partial work is graded
    assert "sandbox_timeout" in world.agent._sandbox_id_to_run_result["session"]["openhands_poll_error"]
    # started in a session of its own whose pid is recorded first, so the whole tree can be found and stopped
    [foreground_command] = [command for command in script.commands if "setsid -w" in command]
    assert foreground_command.startswith("setsid -w bash -c") and "/pid;" in foreground_command
    # the stop is confirmed before anything is collected
    assert script.events.index("pid") < script.events.index("quiesce") < script.events.index("download-result")
    assert any("pid=4242" in command and "kill -TERM -- -$pid" in command for command in script.commands)


async def test_a_foreground_run_that_finished_is_not_stopped(world, tmp_path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(world.agent.config, "background_exec", False)
    script = runner_sandbox(world, tmp_path, result={**PARTIAL_RESULT, "partial": False, "finished": True})

    await call_responses(world)

    assert "quiesce" not in script.events
    assert world.agent._sandbox_id_to_run_result["session"]["openhands_finished"] is True


async def test_a_foreground_run_that_never_wrote_its_pid_is_ungraded(
    world, tmp_path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(world.agent.config, "background_exec", False)
    runner_sandbox(world, tmp_path, pid=_exec(1, ""), result=PARTIAL_RESULT)

    with pytest.raises(NgFailureEvent) as caught:
        await world.agent.run(world.request, world.body)

    assert caught.value.operation == "openhands.execute"
    assert "did not start" in caught.value.failure_reason
    assert world.posts == ["/seed_session"]


async def test_a_child_that_keeps_writing_stops_writing_before_the_workspace_is_collected(
    world, tmp_path, monkeypatch: MonkeyPatch
) -> None:
    """The real supervision, with real processes: the foreground runner starts a child that keeps writing; when the
    exec times out the tree keeps running, and the stop must end the child before the result is downloaded."""
    shim_bin = tmp_path / "shim-bin"
    shim_bin.mkdir()
    for name in ("pkill", "pgrep"):  # never touch (or be fooled by) this machine's own tmux
        (shim_bin / name).write_text(
            f'#!/bin/bash\nfor a in "$@"; do [ "$a" = tmux ] && exit 1; done\nexec /usr/bin/{name} "$@"\n'
        )
        (shim_bin / name).chmod(0o755)
    writes = tmp_path / "writes"
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin" / "python"
    python.write_text(
        f"#!/bin/bash\n(while true; do echo x >> {writes}; sleep 0.05; done) &\nwhile true; do sleep 1; done\n"
    )
    python.chmod(0o755)
    monkeypatch.setattr(world.agent.config, "background_exec", False)
    monkeypatch.setattr(world.agent.config, "venv_dir", str(venv))
    monkeypatch.setattr(world.agent.config, "sandbox_timeout", 1.0)
    world.agent._sandbox_id_to_sandbox["session"] = world.sandbox
    world.agent._sandbox_id_to_workdir["session"] = str(tmp_path)
    world.request._cookies = {"sandbox_id": "session"}
    order: list[str] = []
    sizes: dict[str, int] = {}

    async def local_exec(command: str, cwd=None, timeout_s=None, **kwargs):
        if "OPENHANDS_INSTALLED" in command:
            return _exec(0, "OPENHANDS_INSTALLED")
        # Output goes to files, not pipes: the runner's orphaned child inherits them and would keep a pipe open, and
        # asyncio would then never report the killed wrapper as finished.
        out_path, err_path = tmp_path / "exec.out", tmp_path / "exec.err"
        with out_path.open("wb") as out_file, err_path.open("wb") as err_file:
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                cwd=cwd or "/",
                env={"PATH": f"{shim_bin}:/usr/bin:/bin"},
                stdout=out_file,
                stderr=err_file,
            )
            try:
                await asyncio.wait_for(process.wait(), timeout_s)
            except asyncio.TimeoutError:
                process.kill()  # the provider gives up on the exec; the tree it started keeps running
                await process.wait()
                order.append("timeout")
                return SimpleNamespace(return_code=124, stdout="", stderr="", error_type="timeout")
        if "kill -TERM" in command:
            order.append("stopped")
        return SimpleNamespace(
            return_code=process.returncode,
            stdout=out_path.read_text(),
            stderr=err_path.read_text(),
            error_type=None,
        )

    async def download(remote: str, local):
        if remote.endswith("result.json"):
            order.append("download-result")
            sizes["at_download"] = writes.stat().st_size if writes.exists() else 0
            Path(local).write_text(json.dumps(PARTIAL_RESULT))

    world.sandbox.exec.side_effect = local_exec
    world.sandbox.download = AsyncMock(side_effect=download)

    try:
        response = await call_responses(world)
        await asyncio.sleep(0.4)
        assert order[:3] == ["timeout", "stopped", "download-result"]
        assert sizes["at_download"] > 0  # the child really was writing while the runner ran
        assert writes.stat().st_size == sizes["at_download"]  # and wrote nothing after the stop
        assert response.output  # graded on the partial work
    finally:
        import subprocess

        subprocess.run(["pkill", "-KILL", "-f", str(venv / "bin" / "python")], check=False)


async def test_the_seeded_workdir_is_where_the_runner_launches(world, tmp_path) -> None:
    script = runner_sandbox(world, tmp_path, poll="0", result=PARTIAL_RESULT)
    world.agent._sandbox_id_to_workdir["session"] = "/workspace/nested"

    await call_responses(world)

    assert request_json(tmp_path)["workdir"] == "/workspace/nested"
    assert [launch["cwd"] for launch in script.launches] == ["/workspace/nested"]


async def test_the_default_workdir_is_unchanged(world, tmp_path) -> None:
    script = runner_sandbox(world, tmp_path, poll="0", result=PARTIAL_RESULT)

    await call_responses(world)

    assert request_json(tmp_path)["workdir"] == "/workspace"
    assert [launch["cwd"] for launch in script.launches] == ["/workspace"]


async def test_the_workdir_from_the_seed_reaches_the_runner(world, tmp_path, monkeypatch: MonkeyPatch) -> None:
    runner_sandbox(world, tmp_path, poll="0", result=PARTIAL_RESULT)
    seen: dict = {}

    async def responses(self, request, body):
        seen["workdir"] = self._sandbox_id_to_workdir[request.session[SESSION_ID_KEY]]
        raise RuntimeError("stop here")

    monkeypatch.setattr(OpenHandsSDKSandboxedAgent, "responses", responses)
    seed = await world.agent.server_client.post(url_path="/seed_session")
    seed.json.return_value = {"sandbox_handle": "sandbox-1", "workdir": "/workspace/nested"}

    with pytest.raises(RuntimeError):
        await world.agent.run(world.request, world.body)

    assert seen["workdir"] == "/workspace/nested"
    assert world.agent._sandbox_id_to_workdir == {}


async def test_a_partial_snapshot_without_metrics_reports_no_usage(world, tmp_path) -> None:
    runner_sandbox(world, tmp_path, poll="0", result={**PARTIAL_RESULT, "metrics": None})

    response = await call_responses(world)

    assert response.usage is None  # not a fabricated zero
    assert response.output


async def test_metrics_become_usage(world, tmp_path) -> None:
    metrics = {"prompt_tokens": 10, "completion_tokens": 5, "cache_read_tokens": 2, "reasoning_tokens": 1}
    runner_sandbox(world, tmp_path, poll="0", result={**PARTIAL_RESULT, "metrics": metrics})

    response = await call_responses(world)

    assert (response.usage.input_tokens, response.usage.output_tokens, response.usage.total_tokens) == (10, 5, 15)


@pytest.mark.parametrize(
    ("requested", "configured", "expected"),
    [("high", "low", "high"), (None, "low", "low"), (None, None, None), ("medium", None, "medium")],
)
async def test_the_requests_reasoning_effort_wins_over_the_configured_one(
    world, tmp_path, monkeypatch: MonkeyPatch, requested, configured, expected
) -> None:
    monkeypatch.setattr(world.agent.config, "reasoning_effort", configured)
    runner_sandbox(world, tmp_path, poll="0", result=PARTIAL_RESULT)
    fields = {"reasoning": {"effort": requested}} if requested else {}
    body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "build it"}], **fields)

    await call_responses(world, body)

    assert request_json(tmp_path)["reasoning_effort"] == expected


async def test_teardown_finishes_through_repeated_cancellation(world) -> None:
    stopped = asyncio.Event()
    release = asyncio.Event()

    async def slow_stop() -> None:
        await release.wait()
        stopped.set()

    world.sandbox.stop.side_effect = slow_stop
    world.agent._sandbox_id_to_sandbox["session"] = world.sandbox
    world.agent._sandbox_id_to_workdir["session"] = "/workspace"

    task = asyncio.ensure_future(world.agent._teardown("session", world.sandbox))
    for _ in range(3):  # the request is cancelled again and again while the stop is in flight
        await asyncio.sleep(0)
        task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stopped.is_set()  # the stop ran to completion anyway
    assert world.agent._sandbox_id_to_sandbox == {} and world.agent._sandbox_id_to_workdir == {}


async def test_a_failed_stop_keeps_the_handle_for_a_later_retry(world) -> None:
    world.sandbox.stop.side_effect = [RuntimeError("provider down"), None]
    world.agent._sandbox_id_to_sandbox["session"] = world.sandbox

    await world.agent._teardown("session", world.sandbox)  # does not raise: the original error must not be replaced

    assert world.agent._sandbox_id_to_sandbox == {"session": world.sandbox}  # still reachable

    await world.agent.shutdown()  # a later close or the shutdown retries

    assert world.agent._sandbox_id_to_sandbox == {}
    assert world.sandbox.stop.await_count == 2


async def test_install_exec_error_is_also_a_failed_setup(world) -> None:
    world.sandbox.exec.side_effect = TimeoutError("install timed out")

    with pytest.raises(NgFailureEvent) as caught:
        await world.agent.run(world.request, world.body)

    assert caught.value.operation == "openhands.setup"
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert world.posts == ["/seed_session"]
    world.sandbox.stop.assert_awaited_once()


@pytest.mark.parametrize("error", [RuntimeError("verify failed"), asyncio.CancelledError()])
async def test_sandbox_is_stopped_when_verification_raises_or_is_cancelled(
    world, monkeypatch: MonkeyPatch, error: BaseException
) -> None:
    monkeypatch.setattr(
        OpenHandsSDKSandboxedAgent, "responses", AsyncMock(return_value=MagicMock(model_dump=lambda **kw: {}))
    )
    monkeypatch.setattr(
        app_module,
        "OpenHandsSDKSandboxedAgentVerifyRequest",
        MagicMock(model_validate=MagicMock(return_value=MagicMock(model_dump=lambda: {}))),
    )
    seed_post = world.agent.server_client.post

    async def post(*, url_path: str, **kwargs):
        if url_path == "/verify":
            world.posts.append(url_path)
            raise error
        return await seed_post(url_path=url_path, **kwargs)

    world.agent.server_client.post = post
    world.agent._sandbox_id_to_run_result["session"] = {"left": "over"}

    with pytest.raises(type(error)):
        await world.agent.run(world.request, world.body)

    assert world.posts == ["/seed_session", "/verify"]
    world.sandbox.stop.assert_awaited_once()
    assert world.agent._sandbox_id_to_sandbox == {}
    assert world.agent._sandbox_id_to_run_result == {}


@pytest.mark.parametrize(
    "bad_poll",
    [
        SimpleNamespace(return_code=125, stdout="", stderr="", error_type="sandbox"),  # a structured provider error
        SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None),  # successful, but says nothing
        SimpleNamespace(return_code=0, stdout="garbage", stderr="", error_type=None),
        SimpleNamespace(return_code=1, stdout="0", stderr="", error_type=None),  # an exit code on a failed poll
    ],
)
async def test_a_poll_that_proves_nothing_is_not_a_natural_exit(
    world, tmp_path, monkeypatch: MonkeyPatch, bad_poll
) -> None:
    monkeypatch.setattr(world.agent.config, "poll_interval_s", 0)
    monkeypatch.setattr(world.agent.config, "max_consecutive_poll_errors", 3)
    script = runner_sandbox(world, tmp_path, poll=bad_poll, result=PARTIAL_RESULT)

    response = await call_responses(world)

    # the bounded recovery gave up, so the runner is stopped (it was never seen to exit) before it is graded
    assert "quiesce" in script.events
    assert script.events.index("quiesce") < script.events.index("download-result")
    assert world.agent._sandbox_id_to_run_result["session"]["openhands_poll_error"]
    assert response.output


async def test_a_poll_error_does_not_reset_the_error_budget(world, tmp_path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(world.agent.config, "poll_interval_s", 0)
    monkeypatch.setattr(world.agent.config, "max_consecutive_poll_errors", 3)
    provider_error = SimpleNamespace(return_code=125, stdout="", stderr="", error_type="sandbox")
    # error, RUNNING (a real answer resets the count), error, error, error: only three in a row end the wait
    script = runner_sandbox(
        world,
        tmp_path,
        poll=[provider_error, "RUNNING", provider_error, provider_error, provider_error],
        result=PARTIAL_RESULT,
    )

    await call_responses(world)

    assert "quiesce" in script.events
    assert "poll failed" in world.agent._sandbox_id_to_run_result["session"]["openhands_poll_error"]


async def test_a_nonzero_exit_code_is_a_natural_exit(world, tmp_path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(world.agent.config, "poll_interval_s", 0)
    script = runner_sandbox(world, tmp_path, poll=["RUNNING", "RUNNING", "3"], result=PARTIAL_RESULT)

    response = await call_responses(world)

    assert "quiesce" not in script.events
    assert response.output  # an explicit exit code, zero or not, ends the run and is graded


# --- unsupported request fields (item 7) ----------------------------------------------------------


async def test_request_instructions_are_refused_before_any_sandbox_work(world, tmp_path) -> None:
    script = runner_sandbox(world, tmp_path, poll="0", result=PARTIAL_RESULT)
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "user", "content": "build it"}], instructions="Be terse."
    )

    with pytest.raises(HTTPException) as caught:
        await call_responses(world, body)

    assert caught.value.status_code == 422 and "instructions" in caught.value.detail
    assert script.commands == []  # nothing was set up or launched


async def test_run_refuses_instructions_before_it_seeds_a_sandbox(world, tmp_path) -> None:
    runner_sandbox(world, tmp_path, poll="0", result=PARTIAL_RESULT)
    world.body.responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "user", "content": "build it"}], instructions="Be terse."
    )

    with pytest.raises(HTTPException) as caught:
        await world.agent.run(world.request, world.body)

    assert caught.value.status_code == 422
    assert world.posts == []  # not even /seed_session


@pytest.mark.parametrize(("user_prompt", "expected"), [("fixed prompt", "fixed prompt"), (None, "build it")])
async def test_requests_without_instructions_keep_their_prompt(
    world, tmp_path, monkeypatch: MonkeyPatch, user_prompt, expected
) -> None:
    monkeypatch.setattr(world.agent.config, "user_prompt", user_prompt)
    runner_sandbox(world, tmp_path, poll="0", result=PARTIAL_RESULT)

    await call_responses(world)

    assert request_json(tmp_path)["prompt"] == expected


# --- usage details (item 8) ------------------------------------------------------------------------


async def test_totals_without_details_leave_the_details_unknown(world, tmp_path) -> None:
    metrics = {"prompt_tokens": 10, "completion_tokens": 5}
    runner_sandbox(world, tmp_path, poll="0", result={**PARTIAL_RESULT, "metrics": metrics})

    response = await call_responses(world)

    assert response.usage.total_tokens == 15
    assert response.usage.input_tokens_details.cached_tokens is None
    assert response.usage.output_tokens_details.reasoning_tokens is None


@pytest.mark.parametrize(("cached", "reasoning"), [(0, 2), (1, 0), (2, 1)])
async def test_reported_details_pass_through_unchanged(world, tmp_path, cached, reasoning) -> None:
    metrics = {"prompt_tokens": 3, "completion_tokens": 4, "cache_read_tokens": cached, "reasoning_tokens": reasoning}
    runner_sandbox(world, tmp_path, poll="0", result={**PARTIAL_RESULT, "metrics": metrics})

    response = await call_responses(world)

    assert response.usage.input_tokens_details.cached_tokens == cached
    assert response.usage.output_tokens_details.reasoning_tokens == reasoning


@pytest.mark.parametrize("module", ["runner", "runner_v2"])
def test_runners_write_unreported_usage_details_as_null(module: str) -> None:
    import importlib

    runner = importlib.import_module(f"responses_api_agents.openhands_sdk_sandboxed_agent.{module}")
    metrics = SimpleNamespace(accumulated_cost=0.5, token_usages=[1, 2])

    unreported = json.loads(
        json.dumps(runner._usage_metrics(metrics, SimpleNamespace(prompt_tokens=3, completion_tokens=4)))
    )
    reported = runner._usage_metrics(
        metrics, SimpleNamespace(prompt_tokens=3, completion_tokens=4, cache_read_tokens=0, reasoning_tokens=2)
    )

    assert unreported["cache_read_tokens"] is None and unreported["reasoning_tokens"] is None
    assert unreported["prompt_tokens"] == 3 and unreported["n_llm_calls"] == 2
    assert (reported["cache_read_tokens"], reported["reasoning_tokens"]) == (0, 2)  # an explicit zero stays a zero
