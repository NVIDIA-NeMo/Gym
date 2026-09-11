# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the first NOOA capability slice running through Gym."""

from __future__ import annotations

import hashlib
import json
from http.cookies import SimpleCookie
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from fastapi import Response
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseFunctionToolCallForTraining
from nemo_gym.server_utils import ServerClient
from resources_servers.nooa_capability.app import (
    NOOACapabilityResourcesServer,
    NOOACapabilityResourcesServerConfig,
    NOOACapabilityVerifyRequest,
)
from responses_api_agents.nooa_agent.app import NOOAAgent, NOOAAgentRunRequest
from responses_api_agents.nooa_agent.capabilities.calculate import CalculateSingleAgent
from responses_api_agents.nooa_agent.config import NOOAAgentConfig


PACKAGE_DIR = Path(__file__).parents[1]
CONFIG_PATH = PACKAGE_DIR / "configs" / "nooa_calculate_capability.yaml"
DATA_PATH = PACKAGE_DIR / "data" / "capability_calculate.jsonl"
MANIFEST_RELATIVE_PATH = "responses_api_agents/nooa_agent/capabilities/source_manifest.json"
MANIFEST_PATH = PACKAGE_DIR / "capabilities" / "source_manifest.json"


class FakeHTTPResponse:
    ok = True
    status = 200

    def __init__(self, payload: dict[str, Any], *, cookie: tuple[str, str] | None = None) -> None:
        self._payload = json.dumps(payload).encode()
        self.content = self
        self.cookies = SimpleCookie()
        if cookie is not None:
            self.cookies[cookie[0]] = cookie[1]

    async def read(self) -> bytes:
        return self._payload


def _load_rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in DATA_PATH.read_text().splitlines() if line.strip()]


def _agent_config() -> NOOAAgentConfig:
    document = yaml.safe_load(CONFIG_PATH.read_text())
    values = document["nooa_calculate_capability"]["responses_api_agents"]["nooa_agent"]
    return NOOAAgentConfig.model_validate(
        {"name": "nooa_calculate_capability", "host": "127.0.0.1", "port": 9000, **values}
    )


def _policy_response(result: int | float | str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id=f"calculate-{result}",
        created_at=0,
        model="policy",
        object="response",
        output=[
            NeMoGymResponseFunctionToolCallForTraining(
                id=f"fc-{result}",
                call_id=f"return-{result}",
                name="return_result",
                arguments=json.dumps({"result": result}),
                prompt_token_ids=[1, 2],
                generation_token_ids=[3],
                generation_log_probs=[-0.1],
            )
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def capability_server_client(
    *, policy_result: int | float | str
) -> tuple[ServerClient, list[tuple[str, str]], list[Any]]:
    calls: list[tuple[str, str]] = []
    policy_requests: list[Any] = []
    verifier = NOOACapabilityResourcesServer(
        config=NOOACapabilityResourcesServerConfig(
            host="127.0.0.1", port=9001, entrypoint="app.py", name="nooa_capability"
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    client = ServerClient.model_construct(head_server_config=MagicMock(), global_config_dict={})

    async def post(*, server_name: str, url_path: str, json: Any, **kwargs: Any) -> FakeHTTPResponse:
        calls.append((server_name, url_path))
        if server_name == "policy_model":
            policy_requests.append(json)
            return FakeHTTPResponse(_policy_response(policy_result).model_dump(mode="json"))
        if url_path == "/seed_session":
            return FakeHTTPResponse({}, cookie=("resource_session", "seeded"))
        if url_path == "/verify":
            request = NOOACapabilityVerifyRequest.model_validate(json)
            verified = await verifier.verify(request)
            return FakeHTTPResponse(verified.model_dump(mode="json"))
        raise AssertionError(f"unexpected call: {server_name=} {url_path=}")

    object.__setattr__(client, "post", post)
    return client, calls, policy_requests


def test_shipped_calculate_capability_matches_immutable_source_manifest() -> None:
    config = _agent_config()
    rows = _load_rows()
    manifest = json.loads(MANIFEST_PATH.read_text())
    canonical = [
        {
            "args": [],
            "kwargs": {"a": row["a"], "b": row["b"], "calculation": row["calculation"]},
            "expected": row["expected_result"],
        }
        for row in rows
    ]

    assert config.nooa.agent_class.endswith(":CalculateSingleAgent")
    assert config.nooa.entrypoint == "calculate"
    assert set(config.nooa.arguments) == {"a", "b", "calculation"}
    assert manifest["source_commit"] == "54e8fa23778ec384dc5813cee61b8f814276e05b"
    assert (
        manifest["agent_sha256"]
        == hashlib.sha256((PACKAGE_DIR / "capabilities" / "calculate.py").read_bytes()).hexdigest()
    )
    assert manifest["cases"] == canonical
    assert [row["expected_result"] for row in rows] == [7, 56]
    assert all(row["capability_metadata"]["source_manifest"] == MANIFEST_RELATIVE_PATH for row in rows)
    assert all(row["responses_create_params"]["tools"] == [] for row in rows)


def test_capability_config_resolves_with_gym_components() -> None:
    initial = OmegaConf.create(
        {
            "config_paths": [
                str(CONFIG_PATH),
                "resources_servers/nooa_capability/configs/nooa_capability.yaml",
                "responses_api_models/openai_model/configs/openai_model.yaml",
            ],
            "policy_base_url": "https://example.invalid/v1",
            "policy_api_key": "test-key",
            "policy_model_name": "test-model",
        }
    )
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
            hide_secrets=True,
        )
    )

    agent = resolved.nooa_calculate_capability.responses_api_agents.nooa_agent
    assert agent.nooa.entrypoint == "calculate"
    assert agent.resources_server.name == "nooa_capability"
    assert resolved.policy_model.responses_api_models.openai_model.openai_model == "test-model"


@pytest.mark.asyncio
@pytest.mark.parametrize("row", _load_rows(), ids=lambda row: row["id"])
async def test_calculate_capability_runs_through_gym_and_verifies(row: dict[str, Any]) -> None:
    client, calls, policy_requests = capability_server_client(policy_result=row["expected_result"])
    agent = NOOAAgent(config=_agent_config(), server_client=client)
    body = NOOAAgentRunRequest.model_validate(row)
    request = SimpleNamespace(cookies={}, path_params={}, url=SimpleNamespace(path="/run"))

    result = await agent.run(request, Response(), body)

    assert result.reward == 1.0
    assert result.expected_result == row["expected_result"]
    assert result.actual_result == row["expected_result"]
    assert result.output_correct is True
    assert result.ng_agent_observations is not None
    rendered_request = policy_requests[0].model_dump_json()
    assert str(row["a"]) in rendered_request
    assert str(row["b"]) in rendered_request
    assert row["calculation"] in rendered_request
    assert "expected_result" not in rendered_request
    assert "capability_metadata" not in rendered_request
    assert calls == [
        ("nooa_capability", "/seed_session"),
        ("policy_model", "/v1/responses"),
        ("nooa_capability", "/verify"),
    ]


@pytest.mark.asyncio
async def test_calculate_capability_verifier_rejects_a_wrong_result() -> None:
    row = _load_rows()[0]
    client, _, _ = capability_server_client(policy_result=999)
    agent = NOOAAgent(config=_agent_config(), server_client=client)
    body = NOOAAgentRunRequest.model_validate(row)
    request = SimpleNamespace(cookies={}, path_params={}, url=SimpleNamespace(path="/run"))

    result = await agent.run(request, Response(), body)

    assert result.reward == 0.0
    assert result.expected_result == 7
    assert result.actual_result == 999
    assert result.output_correct is False


@pytest.mark.asyncio
@pytest.mark.parametrize("row", _load_rows(), ids=lambda row: f"parity-{row['id']}")
async def test_native_and_gym_calculate_results_have_exact_score_parity(row: dict[str, Any]) -> None:
    expected = row["expected_result"]
    native_llm = FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                raw_response=None,
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"native-return-{row['id']}",
                        name="return_result",
                        arguments=json.dumps({"result": expected}),
                    )
                ],
                finish_reason="tool_calls",
                assistant_message={"role": "assistant", "content": ""},
            )
        ]
    )
    native_result = await CalculateSingleAgent(llm=native_llm).calculate(
        a=row["a"], b=row["b"], calculation=row["calculation"]
    )

    client, _, _ = capability_server_client(policy_result=native_result)
    gym_agent = NOOAAgent(config=_agent_config(), server_client=client)
    gym_result = await gym_agent.run(
        SimpleNamespace(cookies={}, path_params={}, url=SimpleNamespace(path="/run")),
        Response(),
        NOOAAgentRunRequest.model_validate(row),
    )

    assert native_result == expected
    assert gym_result.actual_result == native_result
    assert gym_result.output_correct is True
    assert gym_result.reward == 1.0
