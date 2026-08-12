# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.terminalbench.app import (
    TerminalBenchResourcesServer,
    TerminalBenchResourcesServerConfig,
    TerminalBenchSeedSessionRequest,
    TerminalBenchVerifyRequest,
)


def _server() -> TerminalBenchResourcesServer:
    config = TerminalBenchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="terminalbench",
        sandbox_provider="sandbox",
        evaluation_timeout=300,
    )
    return TerminalBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _params(task_dir) -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "user", "content": "solve"}],
        metadata={
            "instance_id": "terminalbench::task",
            "task_name": "task",
            "docker_image": "task-image:latest",
            "task_dir": str(task_dir),
            "workdir": "/workspace",
            "verifier_timeout_sec": "60",
            "cpus": "4",
            "memory_mb": "8192",
            "storage_mb": "12288",
            "gpus": "0",
        },
    )


def test_task_resources_override_server_defaults(tmp_path) -> None:
    server = _server()
    with (
        patch("resources_servers.terminalbench.app.get_global_config_dict", return_value={}),
        patch("resources_servers.terminalbench.app.resolve_provider_metadata", return_value={}),
    ):
        spec = server._sandbox_spec(dict(_params(tmp_path).metadata))
    assert spec.image == "task-image:latest"
    assert spec.workdir == "/workspace"
    assert spec.resources.cpu == 4
    assert spec.resources.memory_mib == 8192
    assert spec.resources.disk_gib == 12


async def test_seed_session_returns_serialized_handle(tmp_path) -> None:
    server = _server()
    sandbox = AsyncMock()
    sandbox.serialize.return_value = {"sandbox_id": "box", "workdir": "/workspace"}
    request = MagicMock(session={SESSION_ID_KEY: "session"})
    body = TerminalBenchSeedSessionRequest(responses_create_params=_params(tmp_path))

    with (
        patch("resources_servers.terminalbench.app.get_global_config_dict", return_value={}),
        patch("resources_servers.terminalbench.app.resolve_provider_config", return_value={"test": {}}),
        patch("resources_servers.terminalbench.app.resolve_provider_metadata", return_value={}),
        patch("resources_servers.terminalbench.app.AsyncSandbox", return_value=sandbox),
    ):
        response = await server.seed_session(request, body)

    assert response.sandbox_handle["sandbox_id"] == "box"
    assert server._sandboxes["session"] is sandbox
    sandbox.start.assert_awaited_once()


async def test_verify_stages_hidden_tests_and_reads_reward(tmp_path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
    server = _server()
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [
        MagicMock(return_code=0, stderr=""),
        MagicMock(return_code=0, error_type=None),
        MagicMock(stdout="all tests passed\n"),
        MagicMock(stdout="1\n"),
    ]
    server._sandboxes["session"] = sandbox
    server._metadata["session"] = dict(_params(tmp_path).metadata)
    request = MagicMock(session={SESSION_ID_KEY: "session"})
    body = TerminalBenchVerifyRequest(
        responses_create_params=_params(tmp_path),
        response=NeMoGymResponse(
            id="response",
            created_at=0,
            model="model",
            object="response",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        ),
    )

    result = await server.verify(request, body)

    assert result.reward == 1.0
    assert result.resolved is True
    assert result.test_stdout == "all tests passed\n"
    assert result.mask_sample is False
    sandbox.upload.assert_awaited_once()
    sandbox.stop.assert_awaited_once()
