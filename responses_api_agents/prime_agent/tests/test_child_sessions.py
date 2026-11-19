# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from nemo_gym.agents import AgentSeedSessionRequest, check_agent_seed_rejected
from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.prime_agent.app import PrimeAgent
from responses_api_agents.prime_agent.child_app import PrimeChildAgent, PrimeChildConfig
from responses_api_agents.prime_agent.sandbox import PrimeSession


def agent(**overrides):
    config = PrimeChildConfig(
        name="prime_child",
        host="127.0.0.1",
        port=8200,
        entrypoint="child_app.py",
        model="test",
        model_server={"type": "responses_api_models", "name": "model"},
        **overrides,
    )
    return PrimeChildAgent(config=config, server_client=MagicMock(spec=ServerClient))


def seed(**overrides):
    return AgentSeedSessionRequest(
        agent_session_id="session-one",
        episode_id=EpisodeId.from_capture_key("0-0"),
        task_id={"taskset": "test", "task_id": "one"},
        **overrides,
    )


def child_state(*, request=None, session=None, **kwargs):
    if session is None:
        session = AsyncMock()
    if isinstance(session, (MagicMock, AsyncMock)):
        session.cleanup_confirmed = True
        session.expired = False
        session.expire_reason = None
    kwargs.setdefault("provider_name", "sandbox")
    kwargs.setdefault("sandbox_id", "task-sandbox")
    kwargs.setdefault("runtime", "/runtime")
    return PrimeSession(request=request or seed(), session=session, **kwargs)


def preflight_result():
    return SandboxExecResult(
        return_code=0, stdout=json.dumps({"sdk": "verified", "skill": "agent-message"}), stderr=""
    )


def response():
    from nemo_gym.core import NeMoGymResponse

    return NeMoGymResponse(
        id="response",
        created_at=0,
        object="response",
        model="test",
        output=[],
        tools=[],
        tool_choice="auto",
        parallel_tool_calls=False,
        status="completed",
    )


def test_no_unused_host_cli_install():
    with patch("responses_api_agents.prime_agent.app.ensure_prime_agent") as installer:
        agent()
    installer.assert_not_called()


def test_unsupported_seed_returns_terminal_422():
    assert "task sandbox" in check_agent_seed_rejected(agent().setup_webserver(), seed())


@pytest.mark.parametrize("phase", ["seeding", "active"])
def test_child_uses_inherited_rollout_stop_lifecycle(phase):
    instance = agent()
    session = AsyncMock()
    session.expire.return_value = True
    state = child_state(request=seed(), runtime="/runtime", session=session) if phase == "active" else None
    record = SimpleNamespace(state=state, stop_requested=False, stop_reason=None)
    with patch.object(PrimeChildAgent, "_open_rollout_record", return_value=record):
        with TestClient(instance.setup_webserver()) as client:
            response = client.post("/rollouts/0-0/stop", json={"reason": "operator stop"})
    assert response.status_code == 200 and response.json()["stopped"] is True
    if phase == "seeding":
        assert record.stop_requested is True and record.stop_reason == "operator stop"
        session.expire.assert_not_awaited()
    else:
        session.expire.assert_awaited_once()
        assert session.expire.await_args.kwargs["reason"] == "operator stop"


@pytest.mark.parametrize(
    "change",
    [
        {"temperature": 0.5},
        {"max_output_tokens": 256},
        {"top_p": 0.95},
        {"input": [{"role": "assistant", "content": "old"}, {"role": "user", "content": "new"}]},
        {"model": "wrong"},
    ],
)
def test_unsupported_inputs_and_controls_rejected(change):
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": "Hello", **change})
    with pytest.raises(HTTPException) as caught:
        agent()._session_conversation(body)
    assert caught.value.status_code == 422


@pytest.mark.asyncio
async def test_identical_activation_retries_join_and_changed_input_is_rejected():
    instance = agent()
    state = child_state(request=seed(), runtime="/runtime", session=MagicMock(closing=False))
    instance._agent_session_id_from_request = MagicMock(return_value="session-one")
    instance._require_agent_session = MagicMock(return_value=state)
    instance._activate_session = AsyncMock(return_value=response())
    request = Request({"type": "http", "path_params": {"rollout_id": "0-0"}})
    body = NeMoGymResponseCreateParamsNonStreaming(input="Hello")
    first, second = await asyncio.gather(instance.responses(request, body), instance.responses(request, body))
    assert first == second
    instance._activate_session.assert_awaited_once()
    with pytest.raises(HTTPException, match="one activation"):
        await instance.responses(request, body.model_copy(update={"input": "changed"}))


@pytest.mark.asyncio
async def test_rollout_binding_prevents_execution():
    instance = agent()
    instance._agent_session_id_from_request = MagicMock(return_value="session-one")
    instance._require_agent_session = MagicMock(
        return_value=child_state(request=seed(), runtime="/runtime", session=MagicMock())
    )
    with pytest.raises(HTTPException, match="rollout"):
        await instance.responses(
            Request({"type": "http", "path_params": {"rollout_id": "other"}}),
            NeMoGymResponseCreateParamsNonStreaming(input="Hello"),
        )


@pytest.mark.asyncio
async def test_close_disconnects_and_returns_explicit_observation_gaps():
    instance = agent()
    state = child_state(request=seed(), runtime="/runtime", session=AsyncMock())
    closed = await instance._close_agent_session_state(state)
    state.session.close.assert_awaited_once()
    assert closed.agent_observations.gaps
    assert closed.resources_cookies is None


@pytest.mark.asyncio
async def test_lost_http_waiter_does_not_cancel_shared_activation():
    instance = agent()
    state = child_state(request=seed(), runtime="/runtime", session=MagicMock(closing=False))
    instance._agent_session_id_from_request = MagicMock(return_value="session-one")
    instance._require_agent_session = MagicMock(return_value=state)
    started, finish = asyncio.Event(), asyncio.Event()

    async def execute(*args):
        started.set()
        await asyncio.wait_for(finish.wait(), timeout=3)
        return response()

    instance._activate_session = AsyncMock(side_effect=execute)
    request = Request({"type": "http", "path_params": {"rollout_id": "0-0"}})
    body = NeMoGymResponseCreateParamsNonStreaming(input="Hello")
    waiter = asyncio.create_task(instance.responses(request, body))
    await asyncio.wait_for(started.wait(), timeout=3)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not state.task.cancelled()
    finish.set()
    assert await instance.responses(request, body) == response()


@pytest.fixture(autouse=True)
def global_configuration():
    with patch("responses_api_agents.prime_agent.app.get_global_config_dict", return_value={}):
        yield


def borrowed_seed(**access_overrides):
    return seed(
        sandbox_access={
            "connection": {"provider_config_ref": "sandbox", "descriptor": {"id": "owned-by-resources"}},
            "workdir": "/workspace/repo",
            **access_overrides,
        }
    )


@pytest.mark.parametrize(
    "overrides", [{"prime_agent_version": "0.8.0"}, {"num_workers": 2}, {"capture_directory": "relative"}]
)
def test_invalid_child_configuration_fails_before_setup(overrides):
    with pytest.raises(ValueError):
        agent(**overrides)


@pytest.mark.asyncio
async def test_task_user_override_is_refused_by_the_base_lifecycle():
    with pytest.raises(HTTPException, match="user override"):
        await agent()._seed_agent_session_state(borrowed_seed(user="task-user"))


@pytest.mark.asyncio
async def test_required_external_tool_is_refused():
    instance = agent()
    with patch.object(PrimeChildAgent, "effective_tool_accesses", return_value=[MagicMock(required=True)]):
        with pytest.raises(HTTPException, match="tool grants"):
            await instance._seed_agent_session_state(borrowed_seed())


@pytest.mark.asyncio
async def test_model_server_is_required():
    instance = agent()
    instance.config.model_server = None
    with pytest.raises(HTTPException, match="model_server"):
        await instance._seed_agent_session_state(borrowed_seed())


@pytest.mark.asyncio
async def test_failed_connection_releases_provider():
    instance = agent()
    provider = AsyncMock()
    with (
        patch("responses_api_agents.prime_agent.app.create_provider", return_value=provider),
        patch("responses_api_agents.prime_agent.app.resolve_provider_config", return_value={}),
        patch(
            "responses_api_agents.prime_agent.app.AsyncSandbox.connect",
            AsyncMock(side_effect=RuntimeError("connection failed")),
        ),
    ):
        with pytest.raises(RuntimeError, match="connection failed"):
            await instance._seed_agent_session_state(borrowed_seed())
    provider.aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "install", "cleanup"])
async def test_seed_reuses_prime_install_and_failure_cleanup(failure):
    from nemo_gym.agents import AgentSessionSetupError

    instance = agent()
    sandbox = AsyncMock()
    provider = MagicMock(name="provider")
    provider.name = "sandbox"
    sandbox.sandbox_id = "borrowed-task"
    install = AsyncMock(side_effect=RuntimeError("install failed") if failure else None)
    close = AsyncMock(side_effect=RuntimeError("cleanup pending") if failure == "cleanup" else None)
    with (
        patch("responses_api_agents.prime_agent.app.create_provider", return_value=provider),
        patch("responses_api_agents.prime_agent.app.resolve_provider_config", return_value={}),
        patch("responses_api_agents.prime_agent.app.AsyncSandbox.connect", AsyncMock(return_value=sandbox)),
        patch.object(PrimeSession, "install", install),
        patch.object(PrimeSession, "close", close),
    ):
        if failure:
            with pytest.raises(AgentSessionSetupError if failure == "cleanup" else RuntimeError):
                await instance._seed_agent_session_state(borrowed_seed())
            close.assert_awaited_once_with(timeout=instance.config.session_close_timeout_seconds)
        else:
            state = await instance._seed_agent_session_state(borrowed_seed())
            assert isinstance(state, PrimeSession)
            assert not state.session.owns_sandbox
            assert state.session.workdir == "/workspace/repo"
            install.assert_awaited_once_with(timeout=instance.config.sandbox_install_timeout_seconds)
            sandbox.upload.assert_not_awaited()
            close.assert_not_awaited()


def test_shared_prime_activation_and_stop_hooks_are_inherited():
    assert PrimeChildAgent._session_response is PrimeAgent._session_response
    assert PrimeChildAgent._rollout_sandbox_session is PrimeAgent._rollout_sandbox_session


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id,closing", [(None, False), ("session-one", True)])
async def test_unseeded_or_closing_activation_is_rejected(session_id, closing):
    instance = agent()
    instance._agent_session_id_from_request = MagicMock(return_value=session_id)
    instance._require_agent_session = MagicMock(
        return_value=child_state(request=seed(), runtime="/runtime", session=MagicMock(closing=closing))
    )
    with pytest.raises(HTTPException):
        await instance.responses(
            Request({"type": "http", "path_params": {"rollout_id": "0-0"}}),
            NeMoGymResponseCreateParamsNonStreaming(input="Hello"),
        )


@pytest.mark.asyncio
async def test_close_cancels_outstanding_activation_after_remote_cleanup():
    instance = agent()
    task = asyncio.create_task(asyncio.sleep(60))
    state = child_state(request=seed(), runtime="/runtime", session=AsyncMock(), task=task)
    await instance._close_agent_session_state(state)
    assert task.cancelled()


@pytest.mark.asyncio
async def test_base_close_preserves_observations_and_borrowed_ownership():
    from nemo_gym.trajectory import AgentObservationBundle

    session = AsyncMock()
    original = AgentObservationBundle(source="prime_child")
    state = child_state(request=borrowed_seed(), session=session, observations=original)
    reply = await agent()._close_agent_session_state(state)
    assert reply.agent_observations == original
    assert reply.cleanup_confirmed is True
    session.close.assert_awaited_once()
    session.sandbox.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_child_uploads_only_the_gym_provider_without_host_credentials(tmp_path):
    from nemo_gym.agents import SandboxSession

    extra_secret = "dummy-extra-provider-secret"
    configured = {
        "providers": {
            "unused-provider": {"baseUrl": "https://unused.invalid/v1", "apiKey": extra_secret},
            "nemo": {"apiKey": "dummy-host-nemo-secret"},
        },
        "hostOnlyOption": extra_secret,
    }
    instance = agent(models_config=configured, artifact_directory=str(tmp_path / "captures"))
    original = copy.deepcopy(instance.config.models_config)
    uploads = {}

    async def capture_upload(source, destination):
        uploads[destination] = Path(source).read_text()

    sandbox = AsyncMock()
    sandbox.exec.return_value = preflight_result()
    sandbox.upload.side_effect = capture_upload
    session = SandboxSession(
        sandbox=sandbox, session_dir="/tmp/agent-session", workdir="/workspace/repo", harness="test"
    )
    state = child_state(
        request=borrowed_seed(
            env={
                "CUSTOM": "task-value",
                "PYTHONPATH": "/task-python",
                "OPENAI_API_KEY": "dummy-task-key",
                "PRIME_AGENT_KERNEL_PYTHON": "/wrong-python",
            }
        ),
        runtime="/runtime",
        session=session,
    )

    async def execute(**kwargs):
        await kwargs["stage_activation"]()
        return {"termination": "completed", "events": [], "assistantTurns": 0}

    session.execute = AsyncMock(side_effect=execute)
    endpoint = "http://model.invalid/ng-rollout/0-0/v1"
    with patch.object(PrimeChildAgent, "resolve_model_base_url", return_value=endpoint) as resolve:
        await instance._activate_session(state, NeMoGymResponseCreateParamsNonStreaming(input="Inspect task"))

    assert set(uploads) == {
        "/tmp/agent-session/child_runtime.mjs",
        "/tmp/agent-session/input.json",
        "/tmp/agent-session/agent/models.json",
        "/tmp/agent-session/params.sh",
        "/tmp/agent-session/stdout.jsonl",
    }
    assert all(extra_secret not in text and "dummy-host-nemo-secret" not in text for text in uploads.values())
    preflight = sandbox.exec.await_args
    assert "--check-runtime /runtime" in preflight.args[0]
    assert preflight.kwargs["env"]["PRIME_AGENT_KERNEL_PYTHON"] == "/runtime/kernel/bin/python3"
    assert preflight.kwargs["env"]["CUSTOM"] == "task-value"
    assert "OPENAI_API_KEY" not in preflight.kwargs["env"]
    assert preflight.kwargs["env"]["PYTHONPATH"] == (
        "/runtime/lib/node_modules/prime-agent/dist/skills/agent-message/src:/task-python:/workspace/repo"
    )
    uploaded = uploads["/tmp/agent-session/input.json"]
    assert extra_secret not in uploaded and "dummy-host-nemo-secret" not in uploaded
    models = json.loads(uploaded)["models"]
    assert set(models) == {"providers"}
    assert set(models["providers"]) == {"nemo"}
    nemo = models["providers"]["nemo"]
    assert nemo["baseUrl"] == endpoint
    assert nemo["apiKey"] == "EMPTY"
    assert nemo["models"][0]["id"] == "test"
    assert "unused-provider" not in uploaded
    assert json.loads(uploads["/tmp/agent-session/agent/models.json"]) == models
    resolve.assert_called_once_with("model", "0-0")
    assert instance.config.models_config == configured == original


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["complete", "no_usage", "model_error", "turn_limit", "bad_capture"])
async def test_execution_preserves_output_usage_and_error_distinctions(tmp_path, kind):
    import json

    from nemo_gym.agents import SandboxSession

    instance = agent(artifact_directory=str(tmp_path / "captures"))
    sandbox = AsyncMock()
    sandbox.exec.return_value = preflight_result()
    session = SandboxSession(
        sandbox=sandbox, session_dir="/tmp/agent-session", workdir="/workspace/repo", harness="test"
    )
    state = child_state(request=seed(), runtime="/runtime", session=session)
    message = {
        "role": "assistant",
        "stopReason": "stop",
        "content": [{"type": "text", "text": "Observed task findings"}],
        "usage": {"input": 1, "output": 2, "cacheRead": 3, "cacheWrite": 4},
    }
    if kind == "no_usage":
        message.pop("usage")
    if kind == "model_error":
        message.update(stopReason="error", errorMessage="Request timed out")
    events = [{"type": "message_end", "message": message}]
    if kind == "turn_limit":
        events.append(
            {"type": "message_end", "message": {"role": "assistant", "stopReason": "aborted", "content": []}}
        )
    captured = {
        "version": 2 if kind == "bad_capture" else 1,
        "termination": "turn_limit" if kind == "turn_limit" else "completed",
        "events": events,
        "deliveries": [],
        "assistantTurns": 1,
    }

    async def execute(**kwargs):
        command = await kwargs["stage_activation"]()
        assert command.argv[0] == "bash"
        return await kwargs["collect"]()

    session.execute = AsyncMock(side_effect=execute)
    instance._build_models_config = MagicMock(return_value={"providers": {"nemo": {}}})
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "system", "content": "System"}, {"role": "user", "content": "Inspect task"}]
    )

    async def read_artifact(sandbox, *, path):
        if path.endswith("effective-session.json"):
            return json.dumps({"systemPrompt": "Native SDK system prompt"})
        return json.dumps(captured)

    with patch("responses_api_agents.prime_agent.child_app.read_text", AsyncMock(side_effect=read_artifact)):
        if kind == "bad_capture":
            with pytest.raises(RuntimeError):
                await instance._activate_session(state, body)
        else:
            result = await instance._activate_session(state, body)
            assert state.observations.records[0].conversation[0].content == "Native SDK system prompt"
            if kind == "model_error":
                assert result.status == "failed"
                assert "Request timed out" in result.error.message
                assert result.metadata["ng_agent_outcome"]
                return
            assert result.output[0].content[0].text == "Observed task findings"
            assert (result.usage is None) == (kind == "no_usage")
            if result.usage:
                assert result.usage.input_tokens == 8
                assert result.usage.output_tokens == 2
                assert result.usage.total_tokens == 10
                assert result.usage.input_tokens_details.cached_tokens == 3
            assert result.status == ("incomplete" if kind == "turn_limit" else "completed")
    instance._build_models_config.assert_called_once_with("0-0")


@pytest.mark.asyncio
async def test_runtime_capability_failure_prevents_harness_staging(tmp_path):
    from nemo_gym.agents import SandboxSession

    instance = agent(artifact_directory=str(tmp_path / "captures"))
    sandbox = AsyncMock()
    sandbox.exec.return_value = SandboxExecResult(return_code=1, stdout="", stderr="agent-message unavailable")
    session = SandboxSession(sandbox=sandbox, session_dir="/tmp/session", workdir="/repo", harness="test")
    session.cleanup = {"return_code": None, "timed_out": False, "cleanup_confirmed": True, "error": None}
    session.close = AsyncMock()
    state = child_state(session=session)
    instance._build_models_config = MagicMock(return_value={"providers": {"nemo": {}}})

    async def execute(**kwargs):
        return await kwargs["stage_activation"]()

    session.execute = AsyncMock(side_effect=execute)
    with patch.object(PrimeSession, "stage", AsyncMock()) as stage:
        with pytest.raises(RuntimeError, match="capability check failed.*agent-message unavailable"):
            await instance._activate_session(state, NeMoGymResponseCreateParamsNonStreaming(input="Inspect task"))
    stage.assert_not_awaited()
    instance.server_client.post.assert_not_called()
    assert not (tmp_path / "captures").exists()
    closed = await instance._close_agent_session_state(state)
    session.close.assert_awaited_once()
    assert closed.cleanup_confirmed is True
    assert closed.agent_observations == state.observations
    observation = next(record for record in closed.agent_observations.records if record.kind == "sandbox")
    assert observation.outcome == "failed" and observation.error_type == "RuntimeError"
    assert observation.exit_code is None and observation.teardown_confirmed is None


@pytest.mark.asyncio
async def test_stop_before_launch_returns_incomplete_without_reading_sdk_capture(tmp_path):
    from nemo_gym.agents import ROLLOUT_STOP_METADATA_KEY, SandboxSession

    instance = agent(artifact_directory=str(tmp_path / "captures"))
    sandbox = AsyncMock()
    session = SandboxSession(sandbox=sandbox, session_dir="/tmp/session", workdir="/repo", harness="test")
    session._expire_requested = True
    session.expire_reason = "operator stop"
    session.cleanup = {"return_code": None, "timed_out": True, "cleanup_confirmed": True, "error": None}
    state = child_state(session=session)
    instance._build_models_config = MagicMock(return_value={"providers": {"nemo": {}}})

    async def execute(**kwargs):
        return await kwargs["collect"]()

    session.execute = AsyncMock(side_effect=execute)
    with patch("responses_api_agents.prime_agent.child_app.read_text", AsyncMock()) as read:
        result = await instance._activate_session(state, NeMoGymResponseCreateParamsNonStreaming(input="Inspect task"))
    assert result.status == "incomplete" and result.output == [] and result.usage is None
    assert json.loads(result.metadata[ROLLOUT_STOP_METADATA_KEY]) == {"reason": "operator stop"}
    read.assert_not_awaited()
    sandbox.upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_child_legacy_run_is_not_exposed():
    with pytest.raises(HTTPException, match="Environment Server"):
        await agent().run(MagicMock(), {})


@pytest.mark.asyncio
@pytest.mark.parametrize("delivered,dispose_error", [(True, False), (True, True), (False, False)])
@pytest.mark.parametrize("provider_error", [None, "503 Service Unavailable", "Connection error."])
async def test_sdk_error_capture_survives_export_close_and_shared_grading(
    tmp_path, delivered, dispose_error, provider_error
):
    import json
    import shutil

    if shutil.which("node") is None:
        pytest.skip("Node is required for the actual SDK shim")
    from pathlib import Path

    import yaml

    from nemo_gym.agents import AgentOutcomePolicyConfig, SandboxSession
    from resources_servers.partner_if.app import VerifyRequest
    from resources_servers.partner_if.tests.test_prime_evidence import (
        lend_for_test,
        sandbox_for_test,
    )
    from resources_servers.partner_if.tests.test_prime_evidence import (
        seed as resource_seed,
    )
    from resources_servers.partner_if.tests.test_prime_evidence import (
        server as resource_server,
    )
    from responses_api_agents.prime_agent.tests.test_child_runtime import sdk_error_capture
    from tests.helpers.single_agent_turn import _Client, _environment_server, _request, _Response

    captured = sdk_error_capture(
        tmp_path, delivered=delivered, dispose_error=dispose_error, provider_error=provider_error
    )
    assert captured["termination"] == "error"
    instance = agent(artifact_directory=str(tmp_path / "captures"))
    from nemo_gym.sandbox import SandboxExecResult

    sandbox = AsyncMock()
    sandbox.exec.return_value = SandboxExecResult(return_code=0, stdout="", stderr="")
    session = SandboxSession(sandbox=sandbox, session_dir="/tmp/session", workdir="/repo", harness="test")
    state = child_state(request=seed(), runtime="/runtime", session=session)

    async def execute(**kwargs):
        return await kwargs["collect"]()

    session.execute = AsyncMock(side_effect=execute)
    instance._build_models_config = MagicMock(return_value={"providers": {"nemo": {}}})

    async def read(sandbox, *, path):
        if path.endswith("capture.json"):
            return json.dumps(captured)
        raise FileNotFoundError(path)  # Optional diagnostics must not erase the capture.

    with patch("responses_api_agents.prime_agent.child_app.read_text", AsyncMock(side_effect=read)):
        reply = await instance._activate_session(state, NeMoGymResponseCreateParamsNonStreaming(input="Send findings"))
    assert reply.status == "failed" and reply.error is not None
    assert state.observations.records[0].status == "failed"
    assert list((tmp_path / "captures").glob("*/capture.json"))
    assert state.observations.gaps
    closed = await instance._close_agent_session_state(state)
    assert closed.agent_observations.records

    resources = resource_server()
    contract = {
        "task_id": "synthetic",
        "origin": "synthetic",
        "source": "authored test",
        "rubric_version": "v1",
        "audit_signoff": True,
        "data_kind": "synthetic_smoke",
        "evidence_mode": "live_agent",
        "rubrics": [
            {
                "id": "send",
                "constraint": "Deliver findings",
                "verifier": {
                    "id": "text_contains_v1",
                    "params": {"source": "artifact:parent_reports", "literal": "Task-specific findings"},
                },
            }
        ],
    }
    body = VerifyRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "Send findings"}]},
            "response": reply.model_dump(),
            "verifier_metadata": {"partner_if": contract},
        }
    )
    environment, client = _environment_server()
    pairing = yaml.safe_load((Path(__file__).resolve().parents[3] / "benchmarks/partner_if/config.yaml").read_text())
    policy = pairing["partner_if_prime_environment_server"]["environment_servers"]["single_agent_turn_legacy"][
        "agent_outcome_policy"
    ]
    environment.config.agent_outcome_policy = AgentOutcomePolicyConfig.model_validate(policy)
    request = _request()
    request.task.task_input.responses_create_params = body.responses_create_params
    request.task.task_input.task_data = {"verifier_metadata": body.verifier_metadata.model_dump(mode="json")}
    client.responses[2] = _Response(reply.model_dump(mode="json"))
    client.responses[3] = _Response(closed.model_dump(mode="json"))
    original_post = _Client.post
    session_id = None

    async def post(self, server_name, url_path, **kwargs):
        nonlocal session_id
        result = await original_post(self, server_name, url_path, **kwargs)
        if url_path == "/seed_session":
            session_id = kwargs["json"]["resources_session_id"]
            seeded = resource_seed().model_copy(
                update={"resources_session_id": session_id, "task_data": request.task.task_input.task_data}
            )
            await lend_for_test(resources, seeded, sandbox_for_test())
        if url_path == "/verify":
            verified = await resources.verify(
                Request({"type": "http", "session": {"session_id": session_id}}),
                VerifyRequest.model_validate(kwargs["json"]),
            )
            return _Response(verified.model_dump(mode="json"))
        return result

    with (
        patch.object(_Client, "post", post),
        patch("resources_servers.partner_if.prime_app.read_text", AsyncMock(return_value=json.dumps(captured))),
    ):
        result = await environment.run_request(request)
    assert result.failure is None
    assert sum(path == "/verify" for _, path, _ in client.calls) == 1
    graded = result.result
    assert graded.mask_sample is (not delivered)
    assert graded.evaluation_completed is delivered
    if delivered:
        assert graded.reward == 1


def test_text_blocks_are_preserved_and_multimodal_inputs_are_refused():
    instance = agent()
    instance._session_conversation(
        NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}]
        )
    )
    with pytest.raises(HTTPException, match="text input"):
        instance._session_conversation(
            NeMoGymResponseCreateParamsNonStreaming(
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": "https://example.org/image.png", "detail": "auto"}
                        ],
                    }
                ]
            )
        )
