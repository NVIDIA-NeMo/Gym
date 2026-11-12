# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preserve Atom context rejection and transport failures through the worker and environment."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError
from nooa.events import Summary
from nooa.runtime.actor import ActorRuntime, _is_context_window_error
from nooa_atom import gym as atom_gym

from nemo_gym.agents import AGENT_OUTCOME_METADATA_KEY
from nemo_gym.client import ServerClient
from nemo_gym.core import (
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.nooa_agent import app as agent_module
from responses_api_agents.nooa_agent import runner as runner_module
from responses_api_agents.nooa_agent import sandbox_entrypoint as entrypoint
from responses_api_agents.nooa_agent.app import NOOAAgent
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig
from responses_api_agents.nooa_agent.tests.test_config import agent_config
from responses_api_agents.nooa_agent.tests.test_environment import _environment, _paths, _request
from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response
from responses_api_agents.nooa_agent.tests.test_sandbox_runner import MemorySandbox, cleanup_receipt


@pytest.mark.asyncio
@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("kind", ["context", "transport"])
async def test_atom_worker_preserves_context_rejection_and_transport_failure(
    tmp_path, monkeypatch, warm, kind
) -> None:
    settings = tmp_path / "settings.yaml"
    settings.write_text("atom:\n  summarization:\n    policy: none\n  context_error_policy: raise\n")
    workspace = tmp_path / "task"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("NEMO_OO_SETTINGS", str(settings))
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user-settings"))
    monkeypatch.setattr("nooa.agent._auto_tracing_attempted", True)
    agent_objects = []
    original_factory = atom_gym.create_session_agent

    def factory(*args, **kwargs):
        agent = original_factory(*args, **kwargs)
        assert agent.context_error_policy == "raise"
        assert agent.get_summarization_status()["policy"] == "none"
        agent_objects.append(agent)
        return agent

    monkeypatch.setattr(atom_gym, "create_session_agent", factory)
    histories_at_close = []
    original_close = atom_gym.SessionRegistry.close_all

    async def close_registry(registry):
        histories_at_close.extend([list(agent.event_manager.all_events()) for agent in agent_objects])
        return await original_close(registry)

    monkeypatch.setattr(atom_gym.SessionRegistry, "close_all", close_registry)
    archive = MagicMock(side_effect=AssertionError("no context-error archival is allowed"))
    monkeypatch.setattr(ActorRuntime, "_archive_on_context_error", archive)
    request_info = SimpleNamespace(
        url="http://policy/v1/responses", method="POST", headers={}, real_url="http://policy/v1/responses"
    )
    original_error = ClientResponseError(
        request_info=request_info,
        history=(),
        status=400,
        message="Bad Request",
        headers={"x-request-id": "controlled-overflow"},
    )
    body = {
        "error": {
            "code": "context_length_exceeded",
            "message": "This model's maximum context length is 32768 tokens. However, your request has 34000 input tokens.",
        }
    }

    class Rejected(FakeHTTPResponse):
        ok = False
        status = 400

        def raise_for_status(self):
            raise original_error

    calls = []
    reasoning = NeMoGymResponseReasoningItem(
        id="reasoning-prior",
        summary=[{"type": "summary_text", "text": "Scripted reasoning fixture; no model generation claimed."}],
    )

    async def post(self, **kwargs):
        calls.append(kwargs)
        assert kwargs["json"].max_output_tokens == 1024
        if warm and len(calls) == 1:
            output = NeMoGymResponseFunctionToolCall(
                id="f-prior",
                call_id="call-prior",
                name="python_cell",
                arguments=json.dumps({"code": "print('preserved prior turn')"}),
            )
            return FakeHTTPResponse(model_response(reasoning, output, response_id="prior-served-response"))
        if kind == "transport":
            raise ConnectionError("controlled provider connection failure")
        return Rejected(body)

    monkeypatch.setattr(entrypoint._ModelClient, "post", post)
    original_policy = runner_module.policy_termination
    classifications = []

    def policy(error, **kwargs):
        result = original_policy(error, **kwargs)
        classifications.append((error, kwargs.get("model_error"), result))
        return result

    monkeypatch.setattr(runner_module, "policy_termination", policy)
    invocation = NOOAInvocationConfig(
        agent_class="nooa_atom.gym:AtomHarness",
        invocation_adapter="responses_api_agents.nooa_agent.atom_adapter:invoke_atom",
        execution_mode="sandboxed",
        init_kwargs={
            "turn_method": "handle_batch",
            "workdir": str(workspace),
            "sessions_dir": str(tmp_path / "sessions"),
        },
    )
    model_client = MagicMock(spec=ServerClient)
    model_client.global_config_dict = {"policy": {"responses_api_models": {"model": {"host": "policy", "port": 8000}}}}
    model_client._build_server_base_url.return_value = "http://policy:8000"
    instance = NOOAAgent(
        config=agent_config(nooa=invocation, settings_file=settings, context_window=32768, max_policy_calls=4),
        server_client=model_client,
    )
    sandbox = MemorySandbox()
    monkeypatch.setattr(agent_module.AsyncSandbox, "connect", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(agent_module, "create_provider", MagicMock())
    monkeypatch.setattr(agent_module, "resolve_provider_config", MagicMock())
    monkeypatch.setattr(agent_module, "prepare_nooa_runtime", AsyncMock(return_value="/unused/worker-python"))
    artifacts = []

    # Exercise the actual worker entrypoint and SDK in this process; the memory
    # sandbox replaces transport and cleanup receipts, not the agent execution.
    async def execute(command, **kwargs):
        if "/process_supervisor.sh" in command and "--timeout" in command:
            [input_path] = [path for path in sandbox.files if path.endswith("/input.json")]
            directory = input_path.removesuffix("/input.json")
            assert f"env NEMO_OO_SETTINGS={directory}/settings.yaml" in command
            assert sandbox.files[directory + "/settings.yaml"] == settings.read_text()
            payload = entrypoint.SandboxInput.model_validate_json(sandbox.files[input_path])
            artifact = await entrypoint.execute(payload)
            artifacts.append(artifact)
            sandbox.files[directory + "/result.json"] = artifact.model_dump_json()
            sandbox.files[directory + "/cleanup.json"] = json.dumps(cleanup_receipt())
        return SandboxExecResult(return_code=0, stdout="", stderr="")

    sandbox.exec.side_effect = execute
    environment, client = _environment(instance)
    existing_post = client.post.side_effect

    async def route(**kwargs):
        response = await existing_post(**kwargs)
        if kwargs["url_path"] == "/seed_session":
            response.body["sandbox_access"] = {
                "connection": {"provider_config_ref": "fixture", "descriptor": {"id": "borrowed"}},
                "workdir": str(workspace),
            }
        if kwargs["url_path"] == "/verify":
            response.body.update(reward=0.0, evaluation_completed=True, mask_sample=False)
        return response

    client.post.side_effect = route
    request = _request()
    request.task.task_input.responses_create_params.max_output_tokens = 1024
    episode = await environment.run_request(request)
    assert len(artifacts) == 1
    artifact = artifacts[0]
    archive.assert_not_called()
    assert len(agent_objects) == 1
    assert len(histories_at_close) == 1
    assert not any(isinstance(event, Summary) for event in histories_at_close[0])
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()
    assert _paths(client)[-1] == "/close_session"
    if kind == "context":
        assert len(calls) == 1 + int(warm)
        error, model_error, classification = classifications[-1]
        assert error is model_error is original_error
        assert _is_context_window_error(original_error)
        assert original_error.response_content == json.dumps(body).encode()
        assert original_error.headers["x-request-id"] == "controlled-overflow"
        if warm:
            replay = calls[-1]["json"].model_dump(mode="json", exclude_none=True)["input"]
            assert reasoning.model_dump(mode="json", exclude_none=True) in replay
        assert artifact.error is None
        assert artifact.termination_reason == "model_rejected" and artifact.model_status_code == 400
        assert episode.failure is None
        assert episode.result.response.status == "failed"
        assert json.loads(episode.result.response.metadata[AGENT_OUTCOME_METADATA_KEY]) == {
            "failure_kind": "model_request_error",
            "status_code": 400,
        }
        assert _paths(client)[-3:] == ["/v1/agent_sessions/close", "/verify", "/close_session"]
    else:
        assert artifact.error is not None and artifact.error.kind == "transient"
        assert artifact.termination_reason == "infrastructure_error"
        assert episode.result is None and episode.failure is not None
        assert "/verify" not in _paths(client)
