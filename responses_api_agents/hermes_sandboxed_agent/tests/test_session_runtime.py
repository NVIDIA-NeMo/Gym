# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import shlex
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from anyio import CancelScope

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.hermes_sandboxed_agent import app, session_runtime
from responses_api_agents.hermes_sandboxed_agent.app import (
    HermesSandboxedAgent,
    HermesSandboxedAgentConfig,
    HermesSandboxedRunRequest,
    trajectory_response,
)


@pytest.fixture
def agent(tmp_path):
    return HermesSandboxedAgent(
        config=HermesSandboxedAgentConfig(
            name="hermes",
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            model="policy",
            resources_server={"type": "resources_servers", "name": "swe"},
            model_server={"type": "responses_api_models", "name": "policy"},
            runtime_python=None,
            context_length=32768,
            max_tokens=1024,
            compression_enabled=False,
            results_dir=str(tmp_path),
            enabled_toolsets=["terminal"],
        ),
        server_client=MagicMock(spec=ServerClient),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("installed", [False, True])
async def test_managed_runtime_checks_and_reuses_exact_pin(monkeypatch, installed):
    runtime = session_runtime._runtime()
    assert runtime.requirements[0] == (
        "hermes-agent[mcp] @ https://github.com/cmunley1/hermes-agent/archive/"
        "26bb847a88493342ca1b194e0455b479073ae21d.tar.gz"
    )
    fake = SimpleNamespace(is_installed=AsyncMock(return_value=installed), install=AsyncMock(), python=runtime.python)
    monkeypatch.setattr(session_runtime, "_runtime", lambda: fake)
    sandbox = SimpleNamespace(upload=AsyncMock())
    assert await session_runtime.prepare_runtime(sandbox, remote="/private/run", timeout=120) == runtime.python
    fake.is_installed.assert_awaited_once_with(sandbox)
    assert fake.install.await_count == (0 if installed else 1)
    if not installed:
        fake.install.assert_awaited_once_with(sandbox, timeout_s=120)
    assert [call.args[0].name for call in sandbox.upload.await_args_list] == [
        "sandbox_runner.py",
        "sandbox_observer.py",
        "model_kwargs.py",
    ]
    assert all(call.args[0].parent.name == "hermes_agent" for call in sandbox.upload.await_args_list)


@pytest.mark.asyncio
async def test_run_uses_existing_shim_with_matched_controls_and_task_cwd(agent, monkeypatch):
    sandbox = SimpleNamespace(
        upload=AsyncMock(),
        exec=AsyncMock(
            side_effect=[
                SandboxExecResult(return_code=0, stdout="/repo with spaces\n", stderr=""),
                SandboxExecResult(return_code=0, stdout="", stderr=""),
                SandboxExecResult(return_code=0, stdout="", stderr=""),
            ]
        ),
    )
    result = {
        "completed": True,
        "messages": [
            {"role": "user", "content": "fix"},
            {"role": "assistant", "reasoning": "inspect the failure", "content": "fixed"},
        ],
        "prompt_tokens": 20,
        "completion_tokens": 8,
        "reasoning_tokens": 3,
    }

    async def download(remote, local):
        if remote.endswith("output.tail.log"):
            local.write_text("")
        elif remote.endswith("cleanup.json"):
            local.write_text(json.dumps({"cleanup_confirmed": True, "return_code": 0}))
        else:
            local.write_text(json.dumps({"result": result}))

    sandbox.download = download
    monkeypatch.setattr(app, "prepare_runtime", AsyncMock(return_value="/private/python/bin/python3"))
    monkeypatch.setattr(HermesSandboxedAgent, "resolve_model_base_url", lambda *args: "http://policy/rollout/v1")
    response, _ = await agent._run_in_sandbox(
        sandbox, NeMoGymResponseCreateParamsNonStreaming(input="fix", instructions="Preserve the API"), "rollout"
    )
    request = json.loads(sandbox.upload.await_args_list[0].args[0].read_text())
    assert request["model_base_url"] == "http://policy/rollout/v1"
    assert request["max_tokens"] == 1024 and request["temperature"] == 1.0
    assert request["chat_template_kwargs_enabled"] is True
    assert request["system_message"] == "Preserve the API"
    config = yaml.safe_load(request["config_yaml"])
    assert config["model"]["context_length"] == 32768
    assert config["compression"]["enabled"] is False
    assert config["terminal"]["cwd"] == "/repo with spaces"
    launch = sandbox.exec.await_args_list[1]
    assert launch.kwargs["cwd"] == "/repo with spaces"
    assert "HOME" not in launch.kwargs["env"]
    args = shlex.split(launch.args[0])
    assert args[args.index("--") + 1 : args.index("--") + 4] == ["/private/python/bin/python3", "-E", "-s"]
    assert any(arg.endswith("/sandbox_runner.py") for arg in args)
    assert response.status == "completed"
    assert response.output[0].summary[0].text == "inspect the failure"
    assert response.usage.input_tokens == 20 and response.usage.output_tokens == 8


@pytest.mark.asyncio
async def test_managed_install_failure_closes_resources_before_returning_unscored(agent, monkeypatch):
    seeded = SimpleNamespace(cookies={"session": "seeded"}, data={"sandbox_descriptor": {"sandbox_id": "box"}})
    agent.server_client.post = AsyncMock(side_effect=[seeded, SimpleNamespace()])
    provider = SimpleNamespace(aclose=AsyncMock())
    sandbox = SimpleNamespace(
        upload=AsyncMock(),
        exec=AsyncMock(return_value=SandboxExecResult(return_code=0, stdout="/repo", stderr="")),
        disconnect=AsyncMock(side_effect=provider.aclose),
    )
    monkeypatch.setattr(app, "create_provider", lambda config: provider)
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {"sandbox": {"fake": {}}})
    monkeypatch.setattr(app.AsyncSandbox, "connect", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())
    monkeypatch.setattr(app, "get_response_json", AsyncMock(side_effect=lambda value: value.data))
    monkeypatch.setattr(app, "prepare_runtime", AsyncMock(side_effect=RuntimeError("installation failed")))
    monkeypatch.setattr(HermesSandboxedAgent, "resolve_model_base_url", lambda *args: "http://policy/v1")
    body = HermesSandboxedRunRequest(responses_create_params={"input": "fix"})
    # Setup failure does not launch the supervisor or make the patch eligible for verification.
    result = await agent.run(SimpleNamespace(cookies={}), body)
    assert result.reward is None and result.response is None
    assert result.model_dump()["_ng_failure_class"] == "agent_run_error"
    assert [call.kwargs["url_path"] for call in agent.server_client.post.await_args_list] == [
        "/seed_session",
        "/close_session",
    ]
    assert agent.server_client.post.await_args_list[-1].kwargs["cookies"] == {"session": "seeded"}
    provider.aclose.assert_awaited_once()
    sandbox.upload.assert_not_awaited()


@pytest.mark.parametrize(
    "native",
    [
        {"completed": True},
        {"completed": True, "prompt_tokens": 20},
    ],
)
def test_managed_unknown_usage_stays_unknown(native):
    result = session_runtime.trajectory_result({"result": native}, n_input=1)
    response = trajectory_response(result, NeMoGymResponseCreateParamsNonStreaming(input="fix"), "policy")
    assert response.usage is None


@pytest.mark.parametrize(
    ("native", "budget"),
    [
        ({"stop_reason": "max_iterations", "completed": True}, True),
        ({"stop_reason": "wall_time", "interrupted": True}, True),
        ({"failed": True, "error": "First response truncated due to output length limit"}, True),
        ({"partial": True, "error": "Response truncated due to output length limit"}, True),
        ({"failed": True, "error": "Connection error"}, False),
        ({"partial": True, "error": "invalid tool output"}, False),
    ],
)
def test_managed_explicit_limits_do_not_hide_other_failures(native, budget):
    result = session_runtime.trajectory_result({"result": native}, n_input=1)
    assert result.get("budget_exhausted", False) is budget
    if budget:
        assert result["completed"] is False and result["failed"] is False
    else:
        assert result["error"] == native["error"]


@pytest.mark.parametrize("output", [{"error": "cannot import Hermes"}, {}, {"result": []}])
def test_managed_bad_output_is_not_a_completed_trajectory(output):
    with pytest.raises((ValueError, RuntimeError)):
        session_runtime.trajectory_result(output, n_input=1)


@pytest.fixture
def run_setup(agent, monkeypatch):
    body = HermesSandboxedRunRequest(responses_create_params={"input": "fix"})
    response = trajectory_response({"completed": True}, body.responses_create_params, "policy")
    provider = SimpleNamespace(aclose=AsyncMock())
    sandbox = SimpleNamespace(
        stop=AsyncMock(side_effect=PermissionError("operate lease cannot stop")),
        disconnect=AsyncMock(side_effect=provider.aclose),
    )
    replies = {
        "/seed_session": SimpleNamespace(
            status=200,
            cookies={"session": "seeded"},
            data={"sandbox_descriptor": {"sandbox_id": "task", "scope": "operate"}},
        ),
        "/verify": SimpleNamespace(
            status=200,
            cookies={"session": "verified"},
            data=body.model_dump() | {"reward": 1.0, "response": response.model_dump(), "mask_sample": False},
        ),
        "/close_session": SimpleNamespace(status=200),
    }

    async def post(*, url_path, **kwargs):
        return replies[url_path]

    async def check_status(reply):
        if reply.status >= 400:
            raise RuntimeError(f"HTTP {reply.status}")

    agent.server_client.post = AsyncMock(side_effect=post)
    connect = AsyncMock(return_value=sandbox)
    execute = AsyncMock(return_value=(response, {}))
    monkeypatch.setattr(app, "create_provider", lambda config: provider)
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {"sandbox": {"fake": {}}})
    monkeypatch.setattr(app.AsyncSandbox, "connect", connect)
    monkeypatch.setattr(app, "raise_for_status", check_status)
    monkeypatch.setattr(app, "get_response_json", AsyncMock(side_effect=lambda value: value.data))
    monkeypatch.setattr(HermesSandboxedAgent, "_run_in_sandbox", execute)
    return SimpleNamespace(
        body=body,
        response=response,
        provider=provider,
        sandbox=sandbox,
        replies=replies,
        post=post,
        connect=connect,
        execute=execute,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "incomplete", "failed"])
@pytest.mark.parametrize(("completed", "masked"), [(False, True), (False, False), (True, True)])
async def test_managed_verifier_result_reaches_rl_without_no_result_marker(
    agent, run_setup, status, completed, masked
):
    setup = run_setup
    setup.response.status = status
    expected = setup.replies["/verify"].data
    expected.update(
        reward=0.0,
        response=setup.response.model_dump(),
        evaluation_completed=completed,
        mask_sample=masked,
        failure_kind="verifier_error",
        failure_reason="execution interrupted",
    )
    result = await agent.run(SimpleNamespace(cookies={"caller": "original"}), setup.body)
    wire = result.model_dump(mode="json")
    assert wire["reward"] == 0.0 and wire["verifier_reward"] == 0.0
    assert wire["response"] == expected["response"]
    assert wire["evaluation_completed"] is completed and wire["mask_sample"] is masked
    assert wire["failure_kind"] == "verifier_error" and wire["failure_reason"] == "execution interrupted"
    assert "_ng_failure_class" not in wire and "_ng_failure_message" not in wire
    assert agent.server_client.post.await_args.kwargs["cookies"] == {"caller": "original", "session": "verified"}
    setup.sandbox.disconnect.assert_awaited_once()
    setup.sandbox.stop.assert_not_awaited()
    setup.provider.aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["seed", "connect", "verify", "unmasked_response"])
async def test_managed_no_verifier_result_or_unmasked_harness_failure_stays_unscored(agent, run_setup, failure):
    setup = run_setup
    if failure == "connect":
        setup.connect.side_effect = RuntimeError("connect failed")
    elif failure == "unmasked_response":
        setup.response.status = "failed"
    else:

        async def post(*, url_path, **kwargs):
            if url_path == ("/seed_session" if failure == "seed" else "/verify"):
                raise RuntimeError(f"{failure} failed")
            return await setup.post(url_path=url_path, **kwargs)

        agent.server_client.post.side_effect = post
    wire = (await agent.run(SimpleNamespace(cookies={}), setup.body)).model_dump(mode="json")
    assert "reward" not in wire and "response" not in wire
    assert wire["_ng_failure_class"] == "agent_run_error"
    assert agent.server_client.post.await_args.kwargs["url_path"] == "/close_session"
    setup.sandbox.stop.assert_not_awaited()
    assert setup.sandbox.disconnect.await_count == int(failure not in {"seed", "connect"})
    assert setup.provider.aclose.await_count == int(failure != "seed")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["close", "http_500", "http_404", "timeout", "disconnect"])
async def test_managed_cleanup_failure_surfaces_after_verification_and_disconnects(agent, run_setup, failure):
    setup = run_setup
    agent.config.cleanup_timeout = 0.02

    async def post(*, url_path, **kwargs):
        if url_path == "/close_session":
            if failure == "close":
                raise RuntimeError("owner close failed")
            if failure == "timeout":
                await asyncio.Event().wait()
            if failure.startswith("http_"):
                return SimpleNamespace(status=int(failure.removeprefix("http_")))
        return await setup.post(url_path=url_path, **kwargs)

    agent.server_client.post.side_effect = post
    if failure == "disconnect":
        setup.sandbox.disconnect.side_effect = RuntimeError("disconnect failed")
    with pytest.raises((RuntimeError, TimeoutError)):
        await asyncio.wait_for(agent.run(SimpleNamespace(cookies={}), setup.body), timeout=1)
    assert any(call.kwargs["url_path"] == "/verify" for call in agent.server_client.post.await_args_list)
    setup.sandbox.disconnect.assert_awaited_once()
    setup.sandbox.stop.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 405])
@pytest.mark.parametrize("permits_stop", [False, True])
async def test_mounted_runtime_fallback_is_limited_to_absent_close_endpoint(agent, run_setup, status, permits_stop):
    setup = run_setup
    agent.config.runtime_python = "/opt/hermes/bin/hermes-python"
    setup.replies["/close_session"].status = status
    if permits_stop:
        setup.sandbox.stop.side_effect = None
        assert (await agent.run(SimpleNamespace(cookies={}), setup.body)).reward == 1
    else:
        with pytest.raises(PermissionError, match="operate lease"):
            await agent.run(SimpleNamespace(cookies={}), setup.body)
    setup.sandbox.stop.assert_awaited_once()
    setup.sandbox.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_owner_close_survives_anyio_request_cancellation(agent, run_setup):
    setup = run_setup

    async def post(*, url_path, **kwargs):
        await asyncio.sleep(0)
        return await setup.post(url_path=url_path, **kwargs)

    agent.server_client.post.side_effect = post
    with CancelScope() as scope:
        scope.cancel()
        await agent._close_run(setup.sandbox, setup.provider, cookies={"session": "seeded"})
    setup.sandbox.disconnect.assert_awaited_once()
    setup.sandbox.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_cancellation_waits_for_owner_close_and_disconnect(agent, run_setup):
    setup = run_setup
    running, closing, allow_close = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def execute(*args):
        running.set()
        await asyncio.Event().wait()

    async def post(*, url_path, **kwargs):
        if url_path == "/close_session":
            closing.set()
            await allow_close.wait()
        return await setup.post(url_path=url_path, **kwargs)

    setup.execute.side_effect = execute
    agent.server_client.post.side_effect = post
    task = asyncio.create_task(agent.run(SimpleNamespace(cookies={}), setup.body))
    try:
        await asyncio.wait_for(running.wait(), timeout=1)
        task.cancel()
        await asyncio.wait_for(closing.wait(), timeout=1)
        task.cancel()  # A repeated caller cancellation cannot abandon owner cleanup.
        await asyncio.sleep(0)
        assert not task.done()
        allow_close.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
    finally:
        allow_close.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    setup.sandbox.disconnect.assert_awaited_once()
    setup.sandbox.stop.assert_not_awaited()
    setup.provider.aclose.assert_awaited_once()
    assert [call.kwargs["url_path"] for call in agent.server_client.post.await_args_list] == [
        "/seed_session",
        "/close_session",
    ]
