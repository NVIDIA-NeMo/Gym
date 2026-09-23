# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from http.cookies import SimpleCookie
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import BaseModel

from benchmarks.swebench.pro.materialize_single_agent_tasks import materialize_row
from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
)
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig, get_first_server_config_dict
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from nemo_gym.server_utils import ServerClient
from responses_api_agents.opencode_agent.app import OpenCodeAgentConfig
from responses_api_agents.opencode_agent.tests.test_native_sessions import seed, setup  # noqa: F401


class _Response:
    def __init__(self, payload: dict, cookies: dict[str, str] | None = None) -> None:
        self.payload = payload
        self.cookies = SimpleCookie(cookies or {})
        self.ok = True
        self.status = 200

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _resolved_config(path: str):
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [path],
                    "policy_model": {"responses_api_models": {"dummy_model": {"entrypoint": "app.py"}}},
                }
            ),
        )
    )


@pytest.mark.parametrize(
    "config_path",
    [
        "benchmarks/swebench/pro/opencode_native.yaml",
        "responses_api_agents/opencode_agent/configs/opencode_agent_swebench_pro_native.yaml",
    ],
)
async def test_native_recipe_routes_collector_through_environment_and_responses(
    setup, monkeypatch: pytest.MonkeyPatch, config_path: str
):
    agent, sandbox = setup
    monkeypatch.chdir(Path(__file__).resolve().parents[3])
    config = _resolved_config(config_path)
    agent_name = "swebench_pro_opencode_agent"
    resources_name = "swebench_pro_opencode_resources_server"
    environment_name = config.environment_server_routes["swebench_pro:smoke"]
    assert environment_name == "swebench_pro_opencode"
    agent.config = OpenCodeAgentConfig.model_validate(
        OmegaConf.to_container(get_first_server_config_dict(config, agent_name), resolve=True) | {"name": agent_name}
    )
    assert agent.config.execution_mode == "sandbox"
    agent.server_client.global_config_dict = config
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = config
    client._resolve_base_url.side_effect = lambda name: f"http://{name}:8000"
    environment = SingleAgentTurnEnvironmentServer(
        config=SingleAgentTurnEnvironmentServerConfig.model_validate(
            OmegaConf.to_container(get_first_server_config_dict(config, environment_name), resolve=True)
            | {"name": environment_name}
        ),
        server_client=client,
    )
    calls = []
    seed_id = None
    responses_body = None
    with (
        TestClient(agent.setup_webserver()) as agent_http,
        TestClient(environment.setup_webserver()) as environment_http,
    ):

        async def dispatch(*, server_name, url_path, json, cookies=None):
            nonlocal seed_id, responses_body
            body = json.model_dump(mode="json") if isinstance(json, BaseModel) else json
            calls.append((server_name, url_path, body))
            if server_name == environment_name:
                assert url_path == "/run"
                response = await asyncio.to_thread(environment_http.post, url_path, json=body)
                assert response.status_code == 200, response.text
                return _Response(response.json())
            if server_name == agent_name:
                assert url_path != "/run", "Native collection must not call agent /run"
                agent_http.cookies.clear()
                agent_http.cookies.update(cookies or {})
                response = await asyncio.to_thread(agent_http.post, url_path, json=body)
                assert response.status_code == 200, response.text
                if url_path == "/v1/agent_sessions":
                    seed_id = body["agent_session_id"]
                    assert response.json()["agent_session_id"] == seed_id
                elif url_path.endswith("/v1/responses"):
                    responses_body = response.json()
                elif url_path == "/v1/agent_sessions/close":
                    assert body["agent_session_id"] == seed_id
                return _Response(response.json(), dict(response.cookies))
            assert server_name == resources_name
            if url_path == "/seed_session":
                assert body["task_data"]["patch"] == "grader-only"
                return _Response(
                    {
                        "resources_session_id": body["resources_session_id"],
                        "sandbox_access": seed().sandbox_access.model_dump(mode="json"),
                    },
                    {"session": "resources-cookie"},
                )
            assert cookies == {"session": "resources-cookie"}
            if url_path == "/verify":
                sandbox.disconnect.assert_awaited_once()
                assert body["verification_input"]["response"] == responses_body
                return _Response({**body["verification_input"], "reward": 1.0})
            assert url_path == "/close_session"
            return _Response({"resources_session_id": body["resources_session_id"]})

        client.post = AsyncMock(side_effect=dispatch)
        monkeypatch.setattr(RolloutCollectionHelper, "setup_server_client", lambda *args, **kwargs: client)
        materialized = materialize_row(
            {"instance_id": "case-1", "responses_create_params": {"input": "Fix the code"}, "patch": "grader-only"},
            taskset="swebench_pro:smoke",
        )
        collection_config = RolloutCollectionConfig(
            input_jsonl_fpath="unused.jsonl",
            output_jsonl_fpath="unused-output.jsonl",
            environment_routing_mode=config.environment_routing_mode,
            environment_server_routes=dict(config.environment_server_routes),
            num_repeats=1,
        )
        rows = RolloutCollectionHelper._preprocess_raw_rows(
            [(0, json.dumps(materialized), materialized)], collection_config
        )
        _, result = await next(RolloutCollectionHelper().run_examples(rows))
    assert result["failure"] is None, result
    assert result["result"]["verification"]["reward"] == 1.0
    assert [path for _, path, _ in calls] == [
        "/run",
        "/seed_session",
        "/v1/agent_sessions",
        "/ng-rollout/0-0/v1/responses",
        "/v1/agent_sessions/close",
        "/verify",
        "/close_session",
    ]
    assert calls[0][0] == environment_name
    assert calls[3][2]["input"] == "Fix the code"
    assert "grader-only" not in json.dumps(calls[3][2])
    assert responses_body["output"][-1]["content"][0]["text"] == "Fixed"
    assert responses_body["usage"]["total_tokens"] == 16
    assert not agent._native_sessions
    sandbox.stop.assert_not_awaited()
