# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from http.cookies import SimpleCookie

import orjson
import pytest
from aiohttp import ClientConnectionError
from omegaconf import OmegaConf
from pydantic import ConfigDict

from environment_servers.single_agent.app import (
    SingleAgentEnvironmentServer,
    SingleAgentEnvironmentServerConfig,
    _is_retryable_dependency_error,
)
from nemo_gym.base_resources_server import ResourcesVerifyResponse
from nemo_gym.config_types import AgentServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId, MaterializedTask, TaskId
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from nemo_gym.single_agent_episode_types import (
    SingleAgentEpisodeRequest,
    SingleAgentEpisodeResponse,
    SingleAgentTaskInput,
)


class _Response:
    ok = True

    def __init__(self, body: dict, *, cookies: dict[str, str] | None = None) -> None:
        self.body = orjson.dumps(body)
        self.cookies = SimpleCookie(cookies or {})

    async def read(self) -> bytes:
        return self.body


def _agent_response() -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "id": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Patch applied", "annotations": []}],
                "prompt_token_ids": [1, 2],
                "generation_token_ids": [3, 4],
                "generation_log_probs": [-0.1, -0.2],
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )


class _Client(ServerClient):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    calls: list[tuple[str, str, dict]]
    responses: list[_Response | Exception]

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Response:
        self.calls.append((server_name, url_path, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def _resolve_base_url(self, server_name: str) -> str:
        return f"http://{server_name}:8000"


def _environment(
    *, token_capture: bool = False, resources_tool_transports: tuple[str, ...] = ()
) -> tuple[SingleAgentEnvironmentServer, _Client]:
    global_config = OmegaConf.create(
        {
            "resources": {"resources_servers": {"test": {"host": "resources", "port": 8000, "entrypoint": "app.py"}}},
            "agent": {
                "responses_api_agents": {
                    "test": {
                        "host": "agent",
                        "port": 8001,
                        "entrypoint": "app.py",
                        "token_id_capture": token_capture,
                    }
                }
            },
            "token_id_capture": {"enabled": token_capture},
        }
    )
    response = _agent_response()
    client = _Client(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=global_config,
        calls=[],
        responses=[
            _Response(
                {
                    "resources_session_id": "resources-session",
                    "sandbox_access": {
                        "connection": {
                            "kind": "direct",
                            "provider_config_ref": "sandbox-runtime",
                            "descriptor": {"sandbox_id": "task-sandbox"},
                        },
                        "workdir": "/app",
                    },
                },
                cookies={"session": "resources-seeded"},
            ),
            _Response({"agent_session_id": "agent-session"}, cookies={"session": "agent-seeded"}),
            _Response(response.model_dump(mode="json"), cookies={"session": "agent-activated"}),
            _Response(
                {
                    "agent_session_id": "agent-session",
                    "resources_cookies": {"session": "resources-updated"},
                }
            ),
            _Response(
                {
                    "responses_create_params": {"input": "task"},
                    "response": response.model_dump(mode="json"),
                    "reward": 1.0,
                    "benchmark_field": "preserved",
                }
            ),
            _Response({"resources_session_id": "resources-session"}),
        ],
    )
    config = SingleAgentEnvironmentServerConfig(
        name="environment",
        host="environment",
        port=8002,
        entrypoint="app.py",
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        agent_server=AgentServerRef(type="responses_api_agents", name="agent"),
        resources_tool_transports=list(resources_tool_transports),
        default_episode_timeout_seconds=10,
        cleanup_timeout_seconds=10,
    )
    return SingleAgentEnvironmentServer(config=config, server_client=client), client


def _request() -> SingleAgentEpisodeRequest:
    return SingleAgentEpisodeRequest(
        episode_id=EpisodeId(rollout_id="rollout", attempt=2),
        task=MaterializedTask(
            task_id=TaskId(taskset="source", task_id="task"),
            task_input=SingleAgentTaskInput(
                responses_create_params={"input": "task"},
                task_data={"instance_id": "task"},
            ),
        ),
    )


async def test_single_agent_protocol_and_direct_tool_access() -> None:
    environment, client = _environment(resources_tool_transports=("direct_http",))
    result = await environment.run_request(_request())

    assert result.result is not None
    assert result.result.verification.reward == 1.0
    assert result.result.verification.model_dump()["benchmark_field"] == "preserved"
    assert [path for _, path, _ in client.calls] == [
        "/seed_session",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/v1/responses",
        "/v1/agent_sessions/close",
        "/verify",
        "/close_session",
    ]
    create_body = client.calls[1][2]["json"]
    assert create_body.task_id == TaskId(taskset="source", task_id="task")
    assert len(create_body.tool_accesses) == 1
    access = create_body.tool_accesses[0]
    assert access.kind == "direct_http"
    assert access.name == "resources.direct_http"
    assert access.required is True
    assert str(access.base_url) == "http://resources:8000/"
    assert access.cookies == {"session": "resources-seeded"}
    assert create_body.sandbox_access.connection.provider_config_ref == "sandbox-runtime"
    assert create_body.sandbox_access.connection.descriptor == {"sandbox_id": "task-sandbox"}
    assert create_body.sandbox_access.workdir == "/app"
    assert client.calls[2][2]["cookies"] == {"session": "agent-seeded"}
    assert client.calls[3][2]["cookies"] == {"session": "agent-activated"}
    assert client.calls[4][2]["cookies"] == {"session": "resources-updated"}
    assert client.calls[5][2]["cookies"] == {"session": "resources-updated"}
    assert client.calls[3][2]["json"].episode_id == _request().episode_id
    assert client.calls[5][2]["json"].episode_id == _request().episode_id


async def test_verifier_only_episode_does_not_expose_resource_tools() -> None:
    environment, client = _environment()
    result = await environment.run_request(_request())

    assert result.result.verification.reward == 1.0
    assert client.calls[1][2]["json"].tool_accesses == []
    assert client.calls[1][2]["json"].sandbox_access is not None


async def test_token_capture_keeps_prefixed_twin_route() -> None:
    environment, client = _environment(token_capture=True)
    result = await environment.run_request(_request())
    assert client.calls[2][1] == "/ng-rollout/rollout-a2/training-token-capture/v1/responses"
    verified_response = client.calls[4][2]["json"].verification_input.response
    assert verified_response == _agent_response()
    assert result.result.verification.response == verified_response


@pytest.mark.parametrize("reward", [0.0, 1.0])
@pytest.mark.parametrize("mask_sample", [False, True])
async def test_failed_agent_response_still_reaches_verification(reward: float, mask_sample: bool) -> None:
    environment, client = _environment()
    failed_response = _agent_response().model_dump(mode="json")
    failed_response.update(
        status="failed",
        error={"code": "server_error", "message": "Model generated invalid tool call: finish"},
        metadata={"partial": "true", "turns": "26"},
    )
    client.responses[2] = _Response(failed_response, cookies={"session": "agent-activated"})
    verification = orjson.loads(client.responses[4].body)
    verification.update(response=failed_response, reward=reward, mask_sample=mask_sample)
    client.responses[4] = _Response(verification)

    result = await environment.run_request(_request())

    assert result.failure is None
    assert [path for _, path, _ in client.calls][-3:] == [
        "/v1/agent_sessions/close",
        "/verify",
        "/close_session",
    ]
    forwarded = client.calls[4][2]["json"].verification_input.response
    assert forwarded == NeMoGymResponse.model_validate(failed_response)
    assert client.calls[4][2]["cookies"] == {"session": "resources-updated"}
    assert result.result.verification.response == forwarded
    assert result.result.verification.reward == reward
    assert result.result.verification.mask_sample is mask_sample
    assert result.result.verification.response.status == "failed"


@pytest.mark.parametrize("mask_sample", [False, True])
@pytest.mark.parametrize("result_path", ["native", "flat-adapter"])
async def test_results_preserve_verification_and_observations(mask_sample: bool, result_path: str) -> None:
    adapter_module = None
    if result_path == "flat-adapter":
        adapter_module = pytest.importorskip(
            "environment_servers.single_agent_legacy.app", reason="Flat-row adapter is a separate upstream PR"
        )
    environment, client = _environment()
    observations = {
        "source": "hermes",
        "records": [
            {
                "kind": "agent_invocation",
                "invocation_id": "root",
                "status": "completed",
                "model_calls": [{"model_call_id": "model-call-1"}],
            }
        ],
    }
    close_body = orjson.loads(client.responses[3].body)
    close_body["agent_observations"] = observations
    client.responses[3] = _Response(close_body)
    verification_body = orjson.loads(client.responses[4].body)
    verification_body.update(mask_sample=mask_sample, grader_output={"resolved": True})
    client.responses[4] = _Response(verification_body)

    result = await environment.run_request(_request())
    result = SingleAgentEpisodeResponse.model_validate_json(result.model_dump_json())
    assert result.result.verification.mask_sample is mask_sample
    assert result.result.verification.reward == 1.0
    assert result.result.verification.model_extra["benchmark_field"] == "preserved"
    assert result.result.verification.model_extra["grader_output"] == {"resolved": True}
    assert result.result.verification.response == _agent_response()
    assert result.result.agent_observations == AgentObservationBundle.model_validate(observations)
    if adapter_module is None:
        return
    adapter = adapter_module.SingleAgentLegacyEnvironmentServer(config=environment.config, server_client=client)
    legacy = adapter._legacy_result(result)
    assert legacy["mask_sample"] is mask_sample
    assert legacy["reward"] == 1.0
    assert legacy["benchmark_field"] == "preserved"
    assert legacy["grader_output"] == {"resolved": True}
    assert legacy["response"] == _agent_response().model_dump(mode="json")
    assert legacy["ng_agent_observations"] == AgentObservationBundle.model_validate(observations).model_dump(
        mode="json"
    )


@pytest.mark.parametrize("stage", ["agent", "verification"])
@pytest.mark.parametrize("error", [ClientConnectionError("connection lost"), ValueError("invalid payload")])
async def test_dependency_failure_closes_sessions(stage: str, error: Exception) -> None:
    environment, client = _environment()
    responses = client.responses
    if stage == "agent":
        client.responses = [*responses[:2], error, responses[3], responses[5]]
    else:
        client.responses = [*responses[:4], error, responses[5]]

    result = await environment.run_request(_request())

    assert result.result is None
    assert result.failure.stage == stage
    assert result.failure.message == str(error)
    assert result.failure.terminal is isinstance(error, ValueError)
    assert result.failure.partial_response == (_agent_response() if stage == "verification" else None)
    paths = [path for _, path, _ in client.calls]
    assert paths == [
        "/seed_session",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/v1/responses",
        "/v1/agent_sessions/close",
        *(["/verify"] if stage == "verification" else []),
        "/close_session",
    ]
    assert not client.responses


@pytest.mark.parametrize("retry_fails", [False, True], ids=["retry-recovers", "retry-also-fails"])
async def test_agent_close_failure_prevents_verification_and_unwind_retries(*, retry_fails: bool) -> None:
    environment, client = _environment()
    responses = client.responses
    retry_response = RuntimeError("agent close still fails") if retry_fails else responses[3]
    client.responses = [*responses[:3], TimeoutError("agent close timed out"), retry_response, responses[5]]

    result = await environment.run_request(_request())

    assert result.result is None
    assert result.failure.stage == "cleanup"
    assert result.failure.terminal is True
    assert result.failure.partial_response == _agent_response()
    assert [path for _, path, _ in client.calls] == [
        "/seed_session",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/v1/responses",
        "/v1/agent_sessions/close",
        "/v1/agent_sessions/close",
        "/close_session",
    ]
    assert not client.responses


async def test_admission_timeout_is_retryable_without_starting_sessions() -> None:
    environment, client = _environment()
    environment = SingleAgentEnvironmentServer(
        config=environment.config.model_copy(update={"max_concurrent_episodes": 1, "queue_timeout_seconds": 0.01}),
        server_client=client,
    )
    async with environment._admission:
        result = await environment.run_request(_request())

    assert result.result is None
    assert result.failure.message == "Episode admission timed out"
    assert result.failure.terminal is False
    assert client.calls == []


@pytest.mark.parametrize(("reward", "evaluation_completed"), [(0.0, True), (1.0, True), (0.0, False)])
@pytest.mark.parametrize("mask_sample", [False, True])
@pytest.mark.parametrize("retry_fails", [False, True], ids=["retry-recovers", "retry-also-fails"])
async def test_resources_close_failure_preserves_verification(
    reward: float,
    evaluation_completed: bool,
    mask_sample: bool,
    retry_fails: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    environment, client = _environment()
    observations = AgentObservationBundle.model_validate(
        {
            "source": "hermes",
            "records": [
                {"kind": "agent_invocation", "invocation_id": "root", "status": "completed"},
            ],
        }
    )
    close_body = orjson.loads(client.responses[3].body)
    close_body["agent_observations"] = observations.model_dump(mode="json")
    client.responses[3] = _Response(close_body)
    verification = orjson.loads(client.responses[4].body)
    verification.update(reward=reward, mask_sample=mask_sample, evaluation_completed=evaluation_completed)
    client.responses[4] = _Response(verification)
    responses = client.responses
    retry_response = RuntimeError("resources close still fails") if retry_fails else responses[5]
    client.responses = [*responses[:5], RuntimeError("resources close failed"), retry_response]

    result = await environment.run_request(_request())
    result = SingleAgentEpisodeResponse.model_validate_json(result.model_dump_json())

    assert result.failure is None
    assert result.episode_id == _request().episode_id
    assert result.task_id == _request().task.task_id
    assert result.result.verification == ResourcesVerifyResponse.model_validate(verification)
    assert result.result.agent_observations == observations
    assert [path for _, path, _ in client.calls][-3:] == ["/verify", "/close_session", "/close_session"]
    assert client.calls[-1][2]["cookies"] == {"session": "resources-updated"}
    assert "resources close failed" in caplog.text
    if retry_fails:
        assert "resources close still fails" in caplog.text
    assert not client.responses


async def test_resources_close_timeout_preserves_verification(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    environment, client = _environment()
    environment.config.cleanup_timeout_seconds = 0.02
    close_cancelled = asyncio.Event()
    original_post = _Client.post

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Response:
        if url_path == "/close_session":
            self.calls.append((server_name, url_path, kwargs))
            try:
                await asyncio.Event().wait()
            finally:
                close_cancelled.set()
        return await original_post(self, server_name, url_path, **kwargs)

    monkeypatch.setattr(_Client, "post", post)
    result = await asyncio.wait_for(environment.run_request(_request()), timeout=1)

    assert result.failure is None
    assert result.result.verification.reward == 1.0
    assert result.result.verification.response == _agent_response()
    assert close_cancelled.is_set()
    assert "Episode cleanup timed out" in caplog.text
    assert [path for _, path, _ in client.calls][-2:] == ["/verify", "/close_session"]


async def test_post_verification_cleanup_uses_cleanup_not_episode_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment, client = _environment()
    environment.config.default_episode_timeout_seconds = 0.05
    environment.config.cleanup_timeout_seconds = 1
    original_post = _Client.post

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Response:
        if url_path == "/close_session":
            await asyncio.sleep(0.1)
        return await original_post(self, server_name, url_path, **kwargs)

    monkeypatch.setattr(_Client, "post", post)
    result = await asyncio.wait_for(environment.run_request(_request()), timeout=1)

    assert result.failure is None
    assert result.result.verification.reward == 1.0
    assert [path for _, path, _ in client.calls][-2:] == ["/verify", "/close_session"]
    assert not client.responses


@pytest.mark.parametrize("cancel", [False, True], ids=["episode-timeout", "caller-cancellation"])
async def test_interrupted_activation_closes_agent_before_resources(
    monkeypatch: pytest.MonkeyPatch, *, cancel: bool
) -> None:
    environment, client = _environment()
    environment.config.default_episode_timeout_seconds = 0.05
    responses = client.responses
    client.responses = [*responses[:2], responses[3], responses[5]]
    activation_started = asyncio.Event()
    original_post = _Client.post

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Response:
        if url_path.endswith("/v1/responses"):
            self.calls.append((server_name, url_path, kwargs))
            activation_started.set()
            await asyncio.Event().wait()
            raise AssertionError("Activation must be interrupted")
        return await original_post(self, server_name, url_path, **kwargs)

    monkeypatch.setattr(_Client, "post", post)
    run_task = asyncio.create_task(environment.run_request(_request()))
    await asyncio.wait_for(activation_started.wait(), timeout=1)
    if cancel:
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
    else:
        result = await asyncio.wait_for(run_task, timeout=1)
        assert result.result is None
        assert result.failure.message == "Episode timed out"
        assert result.failure.terminal is False

    assert [path for _, path, _ in client.calls] == [
        "/seed_session",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/v1/responses",
        "/v1/agent_sessions/close",
        "/close_session",
    ]
    assert not client.responses


async def test_legacy_compatibility_is_a_separate_environment_deployment() -> None:
    adapter_module = pytest.importorskip(
        "environment_servers.single_agent_legacy.app", reason="Flat-row adapter is a separate upstream PR"
    )
    environment, client = _environment()
    adapter = adapter_module.SingleAgentLegacyEnvironmentServer(config=environment.config, server_client=client)
    result = await adapter.run_legacy(
        {
            "_ng_task_index": 3,
            "_ng_rollout_index": 2,
            "_ng_attempt_index": 1,
            "instance_id": "task",
            "benchmark_field": "input",
            "responses_create_params": {"input": "task"},
        }
    )

    assert result["reward"] == 1.0
    assert result["benchmark_field"] == "preserved"
    assert result["agent_ref"] == {"name": "agent"}


def test_dependency_failure_messages_are_bounded() -> None:
    environment, _ = _environment()
    error = environment._failure(stage="agent", message="x" * 3000, terminal=False)
    assert len(error.failure.message) == 2000


def test_retry_requires_a_transient_dependency_error() -> None:
    assert _is_retryable_dependency_error(TimeoutError()) is True
    assert _is_retryable_dependency_error(ValueError("invalid response")) is False
