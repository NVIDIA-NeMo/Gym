# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from nemo_gym.base_responses_api_agent import RUN_TOKEN_HEADER
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.sandbox_agent.app import (
    SANDBOX_SETUP_METADATA_KEY,
    SandboxAgent,
    SandboxAgentConfig,
    SandboxAgentRunRequest,
    SandboxWorkspaceSetup,
    archive_workspace,
)


def _config(**kwargs) -> SandboxAgentConfig:
    base = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="sbx",
        resources_server=ResourcesServerRef(type="resources_servers", name="rs"),
        sandbox_provider={"opensandbox": {}},
    )
    base.update(kwargs)
    return SandboxAgentConfig(**base)


def _make_agent(**cfg_kwargs) -> SandboxAgent:
    # skip provider creation and gym tar build (both side effects) during construction
    with (
        patch("responses_api_agents.sandbox_agent.app.create_provider", return_value=MagicMock()),
        patch.object(SandboxAgent, "_build_gym_tar", return_value=None),
    ):
        return SandboxAgent(config=_config(**cfg_kwargs), server_client=MagicMock(spec=ServerClient))


def test_config_defaults():
    cfg = _config()
    assert cfg.mode == "agent_only_runner"
    assert cfg.model_transport == "direct"
    assert cfg.sandbox_image == "python:3.12-slim"
    assert cfg.sandbox_python == "python3"


def test_endpoint_bridge_requires_agent_only_runner_and_valid_port():
    with pytest.raises(ValueError, match="only supported in agent_only_runner"):
        _config(mode="gym_runner", model_transport="endpoint_bridge")
    with pytest.raises(ValueError, match="between 1 and 65535"):
        _config(model_bridge_port=0)


def test_gym_runner_config_and_script():
    agent = _make_agent(
        mode="gym_runner",
        nested_config_paths=["a.yaml", "b.yaml"],
        nested_agent_name="nested_math",
        nested_agent_port=12345,
    )
    script, runner_config, cmd = agent._runner()
    assert runner_config["config_paths"] == ["a.yaml", "b.yaml"]
    assert runner_config["agent_name"] == "nested_math"
    assert runner_config["agent_port"] == 12345
    assert "ng_collect_rollouts" in script
    compile(script, "<gym_runner>", "exec")


def test_gym_runner_skips_gym_tar():
    with (
        patch("responses_api_agents.sandbox_agent.app.create_provider", return_value=MagicMock()),
        patch.object(SandboxAgent, "_build_gym_tar", return_value="/tmp/fake.tar.gz") as tar,
    ):
        nested = SandboxAgent(config=_config(mode="gym_runner"), server_client=MagicMock(spec=ServerClient))
        assert nested._gym_tar is None
        tar.assert_not_called()


def test_runner_config_carries_agent_symbols():
    agent = _make_agent(
        agent_module="responses_api_agents.opencode_agent.app",
        agent_class="OpenCodeAgent",
        agent_config_class="OpenCodeAgentConfig",
        sandbox_python="/deps/bin/python3",
    )
    script, runner_config, cmd = agent._runner()
    assert runner_config["agent_module"] == "responses_api_agents.opencode_agent.app"
    assert runner_config["agent_class"] == "OpenCodeAgent"
    assert runner_config["agent_config_class"] == "OpenCodeAgentConfig"
    assert "runner_config.json" in script
    compile(script, "<agent_runner>", "exec")
    assert cmd == "/deps/bin/python3 /work/runner.py"


def test_agent_config_for_runner_normalizes_opencode_limits_without_mutating_source():
    config = {
        "opencode_config": {
            "provider": {
                "nvinf": {
                    "models": {
                        "policy_model": {
                            "limit": {"context": "80000", "output": "8192"},
                        }
                    }
                }
            }
        }
    }

    normalized = SandboxAgent._agent_config_for_runner(config)

    normalized_limits = normalized["opencode_config"]["provider"]["nvinf"]["models"]["policy_model"]["limit"]
    source_limits = config["opencode_config"]["provider"]["nvinf"]["models"]["policy_model"]["limit"]
    assert normalized_limits == {"context": 80000, "output": 8192}
    assert source_limits == {"context": "80000", "output": "8192"}


def test_sandbox_model_url_rewrites_host_to_ip_and_preserves_port():
    agent = _make_agent(model_server={"type": "responses_api_models", "name": "policy_model"})
    with (
        patch.object(SandboxAgent, "harness_base_url", return_value="http://model-host:8000/runs/run-1") as resolve,
        patch("responses_api_agents.sandbox_agent.app.socket.gethostbyname", return_value="10.1.2.3"),
    ):
        request = MagicMock()
        url = agent._sandbox_model_url(request)
    assert url == "http://10.1.2.3:8000/runs/run-1"
    resolve.assert_called_once_with(request)


def test_sandbox_model_url_strips_v1_after_run_prefix():
    agent = _make_agent(model_server={"type": "responses_api_models", "name": "policy_model"})
    with (
        patch.object(SandboxAgent, "harness_base_url", return_value="http://model-host:8000/runs/run-1/v1"),
        patch("responses_api_agents.sandbox_agent.app.socket.gethostbyname", return_value="10.1.2.3"),
    ):
        url = agent._sandbox_model_url(MagicMock())
    assert url == "http://10.1.2.3:8000/runs/run-1"


def test_sandbox_model_url_falls_back_to_hostname_on_dns_failure():
    agent = _make_agent(model_server={"type": "responses_api_models", "name": "policy_model"})
    with (
        patch.object(SandboxAgent, "harness_base_url", return_value="http://model-host:8000/runs/run-1"),
        patch("responses_api_agents.sandbox_agent.app.socket.gethostbyname", side_effect=OSError),
    ):
        url = agent._sandbox_model_url(MagicMock())
    assert url == "http://model-host:8000/runs/run-1"


def test_endpoint_bridge_model_url_preserves_loopback_target():
    agent = _make_agent(
        model_server={"type": "responses_api_models", "name": "policy_model"},
        model_transport="endpoint_bridge",
    )
    with patch.object(
        SandboxAgent,
        "harness_base_url",
        return_value="http://127.0.0.1:8000/runs/run-1/v1",
    ):
        url = agent._sandbox_model_url(MagicMock())
    assert url == "http://127.0.0.1:8000/runs/run-1"


def test_runner_model_url_uses_local_bridge_without_changing_target():
    agent = _make_agent(model_transport="endpoint_bridge", model_bridge_port=19090)
    target = "http://10.1.2.3:8000/runs/run-1"
    assert agent._runner_model_url(target) == "http://127.0.0.1:19090"


async def test_forward_model_request_preserves_run_path_headers_body_and_sse():
    request_body = b'{"messages":[{"role":"user","content":"test"}]}'
    sse_body = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
    model_client = SimpleNamespace(
        request=AsyncMock(
            return_value=SimpleNamespace(
                status_code=200,
                headers={
                    "content-type": "text/event-stream",
                    "content-encoding": "gzip",
                    "content-length": "999",
                    "x-request-id": "request-1",
                },
                content=sse_body,
            )
        )
    )
    pending = {
        "method": "POST",
        "path": "/v1/chat/completions?stream=true",
        "headers": {
            "authorization": "Bearer gym",
            "host": "127.0.0.1:18080",
            "content-length": str(len(request_body)),
            "x-nemo-gym-test": "preserved",
        },
        "body_b64": base64.b64encode(request_body).decode(),
    }

    reply = await SandboxAgent._forward_model_request(
        pending,
        "http://10.66.25.19:17589/runs/run-token-1",
        model_client,
    )

    model_client.request.assert_awaited_once_with(
        "POST",
        "http://10.66.25.19:17589/runs/run-token-1/v1/chat/completions?stream=true",
        headers={
            "authorization": "Bearer gym",
            "x-nemo-gym-test": "preserved",
        },
        content=request_body,
    )
    assert reply["status"] == 200
    assert reply["headers"] == {
        "content-type": "text/event-stream",
        "x-request-id": "request-1",
    }
    assert base64.b64decode(reply["body_b64"]) == sse_body


async def test_forward_model_request_rejects_bridge_control_path():
    with pytest.raises(RuntimeError, match="invalid model bridge request path"):
        await SandboxAgent._forward_model_request(
            {"path": "/bridge/next"},
            "http://model/runs/token",
            SimpleNamespace(request=AsyncMock()),
        )


async def test_start_model_bridge_uses_public_sandbox_endpoint():
    agent = _make_agent(model_transport="endpoint_bridge", model_bridge_port=19090)
    endpoint = SimpleNamespace(endpoint="https://sandbox.example/19090", headers={"x-auth": "secret"})
    get_endpoint = AsyncMock(return_value=endpoint)
    handle = SimpleNamespace(raw=SimpleNamespace(get_endpoint=get_endpoint))
    agent._provider.exec = AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="", stderr=""))

    result = await agent._start_model_bridge(handle)

    assert result is endpoint
    get_endpoint.assert_awaited_once_with(19090)
    command = agent._provider.exec.await_args.args[1]
    assert "model_bridge_server.py" in command
    assert "127.0.0.1:19090/health" in command


def test_model_bridge_endpoint_url_supports_current_and_legacy_sdk_shapes():
    agent = _make_agent()
    assert (
        agent._model_bridge_endpoint_url(SimpleNamespace(endpoint="sandbox.example/current/"))
        == "http://sandbox.example/current"
    )
    assert (
        agent._model_bridge_endpoint_url(SimpleNamespace(endpoint="https://sandbox.example/current/"))
        == "https://sandbox.example/current"
    )
    assert (
        agent._model_bridge_endpoint_url(SimpleNamespace(url="https://sandbox.example/legacy/"))
        == "https://sandbox.example/legacy"
    )
    with pytest.raises(RuntimeError, match="missing its URL"):
        agent._model_bridge_endpoint_url(SimpleNamespace())


def test_model_bridge_endpoint_url_uses_configured_https_protocol():
    agent = _make_agent(sandbox_provider={"opensandbox": {"connection": {"protocol": "https"}}})
    assert (
        agent._model_bridge_endpoint_url(SimpleNamespace(endpoint="sandbox.example/current/"))
        == "https://sandbox.example/current"
    )


def test_model_bridge_endpoint_url_rejects_unsupported_protocol():
    agent = _make_agent(sandbox_provider={"opensandbox": {"connection": {"protocol": "ftp"}}})
    with pytest.raises(RuntimeError, match="unsupported sandbox endpoint protocol"):
        agent._model_bridge_endpoint_url(SimpleNamespace(endpoint="sandbox.example/current/"))


def test_log_runner_result_keeps_stderr_from_completed_runner(caplog):
    with caplog.at_level("INFO"):
        SandboxAgent._log_runner_result(
            SimpleNamespace(
                return_code=0,
                stdout="RUNNER_DONE\n",
                stderr="OpenCode produced no assistant message",
            )
        )
    assert "OpenCode produced no assistant message" in caplog.text
    assert "runner incomplete" not in caplog.text


def test_log_runner_result_warns_on_unsuccessful_runner(caplog):
    with caplog.at_level("WARNING"):
        SandboxAgent._log_runner_result(SimpleNamespace(return_code=2, stdout="", stderr="invalid OpenCode option"))
    assert "runner incomplete (exit 2)" in caplog.text
    assert "invalid OpenCode option" in caplog.text


def test_gym_tar_built_on_init():
    with (
        patch("responses_api_agents.sandbox_agent.app.create_provider", return_value=MagicMock()),
        patch.object(SandboxAgent, "_build_gym_tar", return_value="/tmp/fake.tar.gz"),
    ):
        agent = SandboxAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        assert agent._gym_tar == "/tmp/fake.tar.gz"


def test_gym_source_prebuilt_path_and_url():
    with (
        patch("responses_api_agents.sandbox_agent.app.create_provider", return_value=MagicMock()),
        patch.object(SandboxAgent, "_build_gym_tar") as build,
    ):
        prebuilt = SandboxAgent(
            config=_config(gym_source="/tmp/prebuilt.tar.gz"), server_client=MagicMock(spec=ServerClient)
        )
        assert str(prebuilt._gym_tar) == "/tmp/prebuilt.tar.gz"
        assert prebuilt._gym_source_url is None
        remote = SandboxAgent(
            config=_config(gym_source="https://example.com/gym.tar.gz"), server_client=MagicMock(spec=ServerClient)
        )
        assert remote._gym_tar is None
        assert remote._gym_source_url == "https://example.com/gym.tar.gz"
        build.assert_not_called()


class _FakeHttpResponse:
    def __init__(self, payload: dict):
        self.payload = payload
        self.cookies = {}
        self.ok = True

    async def read(self):
        return json.dumps(self.payload).encode()


def _agent_response(text: str = "done", *, with_tokens: bool = False) -> dict:
    output = {
        "id": "message-1",
        "content": [{"annotations": [], "text": text, "type": "output_text"}],
        "role": "assistant",
        "status": "completed",
        "type": "message",
    }
    if with_tokens:
        output |= {
            "prompt_token_ids": [10, 11],
            "generation_token_ids": [12],
            "generation_log_probs": [-0.1],
        }
    return {
        "id": "response-1",
        "created_at": 0.0,
        "model": "policy_model",
        "object": "response",
        "output": [output],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 1,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 2,
        },
    }


def test_workspace_setup_rejects_escaping_artifact_path():
    with pytest.raises(ValueError, match="must be relative"):
        SandboxWorkspaceSetup(artifact_paths=["../secret"])


def test_body_from_seed_replaces_prompt_and_carries_private_setup():
    body = SandboxAgentRunRequest(responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="original"))
    seeded = SandboxAgent._body_from_seed(
        body,
        {
            "responses_create_params": {
                "input": [{"role": "user", "content": "seeded"}],
                "metadata": {"workdir": "/workspace"},
            },
            "sandbox_setup": {
                "workspace_path": "/shared/task",
                "workdir": "/workspace",
                "artifact_paths": ["analysis.py"],
            },
        },
    )
    assert seeded.responses_create_params.input[0].content == "seeded"
    encoded = seeded.responses_create_params.metadata[SANDBOX_SETUP_METADATA_KEY]
    assert json.loads(encoded)["workspace_path"] == "/shared/task"


def test_archive_workspace_preserves_relative_layout(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "data").mkdir()
    (workspace / "data" / "values.csv").write_text("x\n1\n")
    destination = tmp_path / "workspace.tar.gz"
    archive_workspace(workspace, destination)
    with tarfile.open(destination, "r:gz") as archive:
        assert "data/values.csv" in archive.getnames()


async def test_download_artifacts_enforces_allowlist_and_size():
    agent = _make_agent()

    async def download_file(handle, remote, local):
        del handle
        content = "print(1)" if remote.endswith("analysis.py") else "x" * 20
        Path(local).write_text(content)

    agent._provider.download_file = AsyncMock(side_effect=download_file)
    setup = SandboxWorkspaceSetup(
        workdir="/workspace",
        artifact_paths=["analysis.py", "analysis.ipynb"],
        max_artifact_bytes=10,
    )
    artifacts = await agent._download_artifacts(MagicMock(), setup)
    assert artifacts == {"analysis.py": "print(1)"}


def test_verify_seed_context_keeps_only_verifier_fields():
    context = SandboxAgent._verify_seed_context(
        {
            "env_id": "env-1",
            "workspace": "/shared/private",
            "verify_context": {"rubric": "r"},
        }
    )
    assert context == {"env_id": "env-1", "rubric": "r"}


async def test_run_uses_run_token_and_verifies_reconstructed_trajectory():
    agent = _make_agent(model_server={"type": "responses_api_models", "name": "policy_model"})
    harness_response = _agent_response("harness transcript")
    reconstructed_response = NeMoGymResponse.model_validate(_agent_response("exact trajectory", with_tokens=True))
    calls = []

    async def post(server_name, url_path, json=None, cookies=None, headers=None):
        calls.append((server_name, url_path, json, cookies, headers))
        if url_path == "/seed_session":
            return _FakeHttpResponse(
                {
                    "env_id": "env-1",
                    "responses_create_params": {"input": [{"role": "user", "content": "seeded"}]},
                }
            )
        if url_path == "/v1/responses":
            return _FakeHttpResponse(harness_response)
        if url_path == "/verify":
            [verified] = json["response"]["output"]
            assert verified["content"][0]["text"] == "exact trajectory"
            assert verified["prompt_token_ids"] == [10, 11]
            assert json["seed_session"] == {"env_id": "env-1"}
            return _FakeHttpResponse(
                {
                    "responses_create_params": {"input": [{"role": "user", "content": "seeded"}]},
                    "response": json["response"],
                    "reward": 1.0,
                }
            )
        raise AssertionError(f"unexpected path: {url_path}")

    agent.server_client.post = AsyncMock(side_effect=post)
    request = MagicMock(cookies={})
    body = SandboxAgentRunRequest(responses_create_params={"input": "original"})
    with (
        patch(
            "responses_api_agents.sandbox_agent.app.uuid4",
            return_value=SimpleNamespace(hex="run-token-1"),
        ),
        patch.object(
            SandboxAgent,
            "get_monotonic_trajectory",
            new_callable=AsyncMock,
            return_value=reconstructed_response.output,
        ) as get_trajectory,
    ):
        result = await agent.run(request, body)

    assert result.reward == 1.0
    assert [url_path for _, url_path, _, _, _ in calls] == ["/seed_session", "/v1/responses", "/verify"]
    assert calls[1][4] == {RUN_TOKEN_HEADER: "run-token-1"}
    get_trajectory.assert_awaited_once_with("policy_model", "run-token-1")


async def test_run_retries_an_empty_constructed_trajectory():
    agent = _make_agent(
        model_server={"type": "responses_api_models", "name": "policy_model"},
        empty_trajectory_retries=1,
    )
    harness_response = _agent_response("harness transcript")
    reconstructed_response = NeMoGymResponse.model_validate(_agent_response("exact trajectory", with_tokens=True))
    calls = []

    async def post(server_name, url_path, json=None, cookies=None, headers=None):
        calls.append((server_name, url_path, json, cookies, headers))
        if url_path == "/seed_session":
            return _FakeHttpResponse(
                {
                    "env_id": "env-retry",
                    "responses_create_params": {"input": [{"role": "user", "content": "seeded"}]},
                }
            )
        if url_path == "/v1/responses":
            return _FakeHttpResponse(harness_response)
        if url_path == "/verify":
            [verified] = json["response"]["output"]
            assert verified["content"][0]["text"] == "exact trajectory"
            return _FakeHttpResponse(
                {
                    "responses_create_params": {"input": [{"role": "user", "content": "seeded"}]},
                    "response": json["response"],
                    "reward": 1.0,
                }
            )
        raise AssertionError(f"unexpected path: {url_path}")

    agent.server_client.post = AsyncMock(side_effect=post)
    request = MagicMock(cookies={})
    body = SandboxAgentRunRequest(responses_create_params={"input": "original"})
    with (
        patch(
            "responses_api_agents.sandbox_agent.app.uuid4",
            side_effect=[SimpleNamespace(hex="run-token-1"), SimpleNamespace(hex="run-token-2")],
        ),
        patch.object(
            SandboxAgent,
            "get_monotonic_trajectory",
            new_callable=AsyncMock,
            side_effect=[[], reconstructed_response.output],
        ) as get_trajectory,
    ):
        result = await agent.run(request, body)

    assert result.reward == 1.0
    assert [url_path for _, url_path, _, _, _ in calls] == [
        "/seed_session",
        "/v1/responses",
        "/v1/responses",
        "/verify",
    ]
    assert calls[1][4] == {RUN_TOKEN_HEADER: "run-token-1"}
    assert calls[2][4] == {RUN_TOKEN_HEADER: "run-token-2"}
    assert get_trajectory.await_args_list == [
        call("policy_model", "run-token-1"),
        call("policy_model", "run-token-2"),
    ]
