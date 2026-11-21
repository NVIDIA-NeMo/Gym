# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
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


def runner_sandbox(world, tmp_path, *, launch=None, confirm=None, poll="RUNNING", quiesce="QUIESCED", result=None):
    """Script the sandbox like the real one: install ok, a detached launch that prints `launched`, a pid file, polls,
    and the stop-the-runner command. `events` records the order of the calls that matter."""
    events: list[str] = []
    launches: list[dict] = []
    commands: list[str] = []

    async def exec_(command: str, **kwargs):
        commands.append(command)
        if "OPENHANDS_INSTALLED" in command:
            return _exec(0, "OPENHANDS_INSTALLED")
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
            return _exec(0, poll)
        if "kill -TERM" in command or "pkill" in command:
            events.append("quiesce")
            return _exec(0, quiesce)
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


async def test_a_runner_that_cannot_be_stopped_is_ungraded(world, tmp_path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(world.agent.config, "sandbox_timeout", 0.05)
    monkeypatch.setattr(world.agent.config, "poll_interval_s", 0)
    runner_sandbox(world, tmp_path, poll="RUNNING", quiesce="ALIVE", result=PARTIAL_RESULT)

    with pytest.raises(NgFailureEvent) as caught:
        await world.agent.run(world.request, world.body)

    assert "still running" in caught.value.failure_reason
    assert world.posts == ["/seed_session"]


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


async def test_a_foreground_run_that_timed_out_is_graded_on_its_partial_work(
    world, tmp_path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(world.agent.config, "background_exec", False)
    runner_sandbox(world, tmp_path, result=PARTIAL_RESULT)

    async def exec_(command: str, **kwargs):
        if "OPENHANDS_INSTALLED" in command:
            return _exec(0, "OPENHANDS_INSTALLED")
        if "runner.py" in command and "pkill" not in command:
            return SimpleNamespace(return_code=124, stdout="partial log", stderr="", error_type="timeout")
        return _exec(0)

    world.sandbox.exec.side_effect = exec_

    response = await call_responses(world)

    assert response.output
    assert "sandbox_timeout" in world.agent._sandbox_id_to_run_result["session"]["openhands_poll_error"]


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
