# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.episode_types import EpisodeId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from nemo_gym.testing.agent_session_conformance import check_agent_seed_rejected
from responses_api_agents.prime_agent.child_app import ChildSession, PrimeChildAgent, PrimeChildConfig


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


def response():
    from nemo_gym.openai_utils import NeMoGymResponse

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
        agent().validate_input(body)
    assert caught.value.status_code == 422


@pytest.mark.asyncio
async def test_identical_activation_retries_join_and_changed_input_is_rejected():
    instance = agent()
    state = ChildSession(request=seed(), runtime="/runtime", session=MagicMock(closing=False))
    instance._agent_session_id_from_request = MagicMock(return_value="session-one")
    instance._require_agent_session = MagicMock(return_value=state)
    instance._execute_child = AsyncMock(return_value=response())
    request = Request({"type": "http", "path_params": {"rollout_id": "0-0"}})
    body = NeMoGymResponseCreateParamsNonStreaming(input="Hello")
    first, second = await asyncio.gather(instance.responses(request, body), instance.responses(request, body))
    assert first == second
    instance._execute_child.assert_awaited_once()
    with pytest.raises(HTTPException, match="one activation"):
        await instance.responses(request, body.model_copy(update={"input": "changed"}))


@pytest.mark.asyncio
async def test_rollout_binding_prevents_execution():
    instance = agent()
    instance._agent_session_id_from_request = MagicMock(return_value="session-one")
    instance._require_agent_session = MagicMock(
        return_value=ChildSession(request=seed(), runtime="/runtime", session=MagicMock())
    )
    with pytest.raises(HTTPException, match="rollout"):
        await instance.responses(
            Request({"type": "http", "path_params": {"rollout_id": "other"}}),
            NeMoGymResponseCreateParamsNonStreaming(input="Hello"),
        )


@pytest.mark.asyncio
async def test_close_disconnects_and_returns_explicit_observation_gaps():
    instance = agent()
    state = ChildSession(request=seed(), runtime="/runtime", session=AsyncMock())
    closed = await instance._close_agent_session_state(state)
    state.session.close.assert_awaited_once()
    assert closed.agent_observations.gaps
    assert closed.resources_cookies is None


@pytest.mark.asyncio
async def test_lost_http_waiter_does_not_cancel_shared_activation():
    instance = agent()
    state = ChildSession(request=seed(), runtime="/runtime", session=MagicMock(closing=False))
    instance._agent_session_id_from_request = MagicMock(return_value="session-one")
    instance._require_agent_session = MagicMock(return_value=state)
    started, finish = asyncio.Event(), asyncio.Event()

    async def execute(*args):
        started.set()
        await asyncio.wait_for(finish.wait(), timeout=3)
        return response()

    instance._execute_child = AsyncMock(side_effect=execute)
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
    with patch("responses_api_agents.prime_agent.child_app.get_global_config_dict", return_value={}):
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
@pytest.mark.parametrize("access", [{"user": "task-user"}, {"env": {"CUSTOM": "value"}}])
async def test_task_user_and_env_overrides_are_refused(access):
    with pytest.raises(HTTPException, match="environment overrides"):
        await agent()._seed_agent_session_state(borrowed_seed(**access))


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
    with pytest.raises(HTTPException, match="Model Server"):
        await instance._seed_agent_session_state(borrowed_seed())


@pytest.mark.asyncio
async def test_failed_connection_releases_provider():
    instance = agent()
    provider = AsyncMock()
    with (
        patch("responses_api_agents.prime_agent.child_app.create_provider", return_value=provider),
        patch("responses_api_agents.prime_agent.child_app.resolve_provider_config", return_value={}),
        patch(
            "responses_api_agents.prime_agent.child_app.AsyncSandbox.connect",
            AsyncMock(side_effect=RuntimeError("connection failed")),
        ),
    ):
        with pytest.raises(RuntimeError, match="connection failed"):
            await instance._seed_agent_session_state(borrowed_seed())
    provider.aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "mkdir", "checksum", "upload", "install", "cleanup"])
async def test_seed_verifies_archives_and_cleans_failed_installations(failure):
    import hashlib
    from types import SimpleNamespace

    from nemo_gym.base_responses_api_agent import AgentSessionSetupError

    instance = agent()
    sandbox = AsyncMock()
    good = SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")
    bad = SimpleNamespace(return_code=1, error_type=None, stdout="installer diagnostics", stderr="failed")
    sandbox.exec.side_effect = [
        bad if failure == "mkdir" else good,
        bad if failure in {"install", "cleanup"} else good,
    ]
    if failure == "upload":
        sandbox.upload.side_effect = RuntimeError("upload failed")
    response = MagicMock()
    response.read = AsyncMock(return_value=b"verified archive")
    expected = hashlib.sha256(b"verified archive").hexdigest() if failure != "checksum" else "0" * 64
    close = AsyncMock(side_effect=RuntimeError("cleanup pending") if failure == "cleanup" else None)
    with (
        patch("responses_api_agents.prime_agent.child_app.create_provider"),
        patch("responses_api_agents.prime_agent.child_app.resolve_provider_config", return_value={}),
        patch("responses_api_agents.prime_agent.child_app.AsyncSandbox.connect", AsyncMock(return_value=sandbox)),
        patch("responses_api_agents.prime_agent.child_app.ARCHIVES", {"prime-agent": expected}),
        patch("responses_api_agents.prime_agent.child_app.http_request", AsyncMock(return_value=response)),
        patch("responses_api_agents.prime_agent.child_app.SandboxSession.close", close),
    ):
        if failure:
            with pytest.raises(AgentSessionSetupError if failure == "cleanup" else RuntimeError):
                await instance._seed_agent_session_state(borrowed_seed())
            close.assert_awaited_once()
        else:
            state = await instance._seed_agent_session_state(borrowed_seed())
            assert not state.session.owns_sandbox
            assert state.session.workdir == "/workspace/repo"
            assert sandbox.upload.await_count == 3
            close.assert_not_awaited()
        if failure != "mkdir":
            response.release.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id,closing", [(None, False), ("session-one", True)])
async def test_unseeded_or_closing_activation_is_rejected(session_id, closing):
    instance = agent()
    instance._agent_session_id_from_request = MagicMock(return_value=session_id)
    instance._require_agent_session = MagicMock(
        return_value=ChildSession(request=seed(), runtime="/runtime", session=MagicMock(closing=closing))
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
    state = ChildSession(request=seed(), runtime="/runtime", session=AsyncMock(), task=task)
    await instance._close_agent_session_state(state)
    assert task.cancelled()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["complete", "no_usage", "model_error", "turn_limit", "bad_capture"])
async def test_execution_preserves_output_usage_and_error_distinctions(tmp_path, kind):
    import json

    from nemo_gym.agent_utils.sandbox_session import SandboxSession

    instance = agent(artifact_directory=str(tmp_path / "captures"))
    sandbox = AsyncMock()
    session = SandboxSession(
        sandbox=sandbox, session_dir="/tmp/agent-session", workdir="/workspace/repo", harness="test"
    )
    state = ChildSession(request=seed(), runtime="/runtime", session=session)
    message = {
        "role": "assistant",
        "stopReason": "stop",
        "content": [{"type": "text", "text": "Observed task findings"}],
        "usage": {"input": 1, "output": 2},
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
        assert command.argv[0] == "env"
        return await kwargs["collect"]()

    session.execute = AsyncMock(side_effect=execute)
    instance._build_models_config = MagicMock(return_value={})
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "system", "content": "System"}, {"role": "user", "content": "Inspect task"}]
    )
    with patch("responses_api_agents.prime_agent.child_app.read_text", AsyncMock(return_value=json.dumps(captured))):
        if kind == "bad_capture":
            with pytest.raises(RuntimeError):
                await instance._execute_child(state, body)
        else:
            result = await instance._execute_child(state, body)
            if kind == "model_error":
                assert result.status == "failed"
                assert "Request timed out" in result.error.message
                assert result.metadata["ng_agent_outcome"]
                return
            assert result.output[0].content[0].text == "Observed task findings"
            assert (result.usage is None) == (kind == "no_usage")
            if result.usage:
                assert result.usage.total_tokens == 3
            assert result.status == ("incomplete" if kind == "turn_limit" else "completed")
    instance._build_models_config.assert_called_once_with("0-0")


@pytest.mark.asyncio
async def test_child_legacy_run_is_not_exposed():
    with pytest.raises(HTTPException, match="Environment Server"):
        await agent().run(MagicMock(), {})


def test_text_blocks_are_preserved_and_multimodal_inputs_are_refused():
    instance = agent()
    instance.validate_input(
        NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}]
        )
    )
    with pytest.raises(HTTPException, match="text input only"):
        instance.validate_input(
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
