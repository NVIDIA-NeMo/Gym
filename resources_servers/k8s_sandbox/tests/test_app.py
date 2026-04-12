# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import (
    ExecuteCodeRequest,
    ExecuteCodeResponse,
    K8sSandboxConfig,
    K8sSandboxResourcesServer,
    K8sSandboxVerifyRequest,
)
from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient

_CALL_COUNTER = 0


def _make_server() -> K8sSandboxResourcesServer:
    config = K8sSandboxConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="k8s_sandbox",
        job_namespace="default",
        job_image="python:3.12-slim",
        execution_timeout=30,
    )
    return K8sSandboxResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_response(function_call_outputs: list[str]) -> NeMoGymResponse:
    global _CALL_COUNTER
    output_items = []
    for i, raw in enumerate(function_call_outputs):
        _CALL_COUNTER += 1
        output_items.append(NeMoGymFunctionCallOutput(call_id=f"call_{_CALL_COUNTER}_{i}", output=raw))

    return NeMoGymResponse(
        id="resp_test",
        created_at=0,
        model="test-model",
        object="response",
        output=output_items,
        status="completed",
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def _make_verify_request(response: NeMoGymResponse, expected_output: str) -> K8sSandboxVerifyRequest:
    return K8sSandboxVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="What is 2^10?"),
        response=response,
        expected_output=expected_output,
    )


class TestK8sSandboxResourcesServer:
    def test_sanity(self) -> None:
        server = _make_server()
        assert server.config.job_namespace == "default"
        assert server.config.job_image == "python:3.12-slim"
        assert server._runner is not None

    @pytest.mark.asyncio
    async def test_execute_code_success(self) -> None:
        server = _make_server()

        with patch.object(server._runner, "run_job", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (0, "1024\n", "")
            result = await server.execute_code(ExecuteCodeRequest(code="print(2**10)"))

        assert isinstance(result, ExecuteCodeResponse)
        assert result.exit_code == 0
        assert "1024" in result.stdout
        assert result.stderr == ""

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["image"] == "python:3.12-slim"
        assert call_kwargs["env"]["__CODE"] == "print(2**10)"
        assert call_kwargs["job_name"].startswith("sandbox-")

    @pytest.mark.asyncio
    async def test_execute_code_timeout(self) -> None:
        server = _make_server()

        with patch.object(server._runner, "run_job", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (-1, "", "timeout: job 'sandbox-abc' did not complete within 30s")
            result = await server.execute_code(ExecuteCodeRequest(code="import time; time.sleep(999)"))

        assert result.exit_code == -1
        assert "timeout" in result.stderr

    @pytest.mark.asyncio
    async def test_verify_matched(self) -> None:
        server = _make_server()
        response = _make_response([json.dumps({"exit_code": 0, "stdout": "1024\n", "stderr": ""})])
        result = await server.verify(_make_verify_request(response, "1024"))

        assert result.reward == 1.0
        assert result.matched is True
        assert result.actual_output == "1024"

    @pytest.mark.asyncio
    async def test_verify_not_matched(self) -> None:
        server = _make_server()
        response = _make_response([json.dumps({"exit_code": 0, "stdout": "wrong\n", "stderr": ""})])
        result = await server.verify(_make_verify_request(response, "1024"))

        assert result.reward == 0.0
        assert result.matched is False

    @pytest.mark.asyncio
    async def test_verify_no_tool_calls(self) -> None:
        server = _make_server()
        response = _make_response([])
        result = await server.verify(_make_verify_request(response, "1024"))

        assert result.reward == 0.0
        assert result.actual_output is None

    @pytest.mark.asyncio
    async def test_verify_uses_last_tool_output(self) -> None:
        server = _make_server()
        response = _make_response([
            json.dumps({"exit_code": 0, "stdout": "wrong\n", "stderr": ""}),
            json.dumps({"exit_code": 0, "stdout": "1024\n", "stderr": ""}),
        ])
        result = await server.verify(_make_verify_request(response, "1024"))

        assert result.reward == 1.0
        assert result.matched is True
