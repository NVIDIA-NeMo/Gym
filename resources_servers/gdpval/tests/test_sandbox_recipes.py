# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
)
from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.global_config import GlobalConfigDictParser
from nemo_gym.rollout_collection import (
    SharedRolloutCollectionConfig,
    _environment_server_for_config_row,
    _episode_record,
)
from nemo_gym.server_utils import ServerClient
from nemo_gym.single_agent_turn_types import SingleAgentTurnRequest
from resources_servers.gdpval.app import GDPValResourcesServer, GDPValVerifyRequest, GDPValVerifyResponse
from resources_servers.gdpval.comparison import task_attempted
from resources_servers.gdpval.task_data import main, prepare_row
from resources_servers.gdpval.tests.test_app import response, row
from resources_servers.gdpval.tests.test_app import sandbox_server as sandbox_server


@pytest.mark.parametrize("harness", ["pi", "hermes", "opencode", "openclaw"])
def test_sandbox_recipe_runs_prepared_task_through_verification(tmp_path, monkeypatch, sandbox_server, harness):
    recipe = Path(__file__).parents[1] / "configs" / f"gdpval_{harness}.yaml"
    _, configs = GlobalConfigDictParser().load_extra_config_paths([str(recipe)])
    config = OmegaConf.merge(*configs)
    routing = SharedRolloutCollectionConfig(
        output_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
        environment_routing_mode=config.environment_routing_mode,
        environment_server_routes=OmegaConf.to_container(config.environment_server_routes),
    )
    source = tmp_path / "source.jsonl"
    output = tmp_path / "prepared.jsonl"
    original = json.dumps(row()) + "\n"
    source.write_text(original)
    monkeypatch.setattr(
        sys, "argv", ["task_data", "--input", str(source), "--output", str(output), "--taskset", "gdpval"]
    )
    main()
    assert source.read_text() == original
    prepared = json.loads(output.read_text())
    destination = _environment_server_for_config_row(prepared, routing)
    environment_config = config[destination].environment_servers.single_agent_turn
    if harness in {"opencode", "openclaw"}:
        assert config[f"{harness}_agent"].responses_api_agents[f"{harness}_agent"].max_output_tokens == 32768
    # A stale flat input must fail before an HTTP request is dispatched.
    with pytest.raises(ValueError, match="taskset routing requires"):
        _environment_server_for_config_row(prepare_row(row()), routing)

    resources, sandbox, resource_request = sandbox_server
    calls = []

    async def grade(self, body):
        assert body.task_id == "task-1"
        assert body.rubric_pretty == "PRIVATE RUBRIC"
        assert task_attempted(body.deliverables_dir)
        assert (Path(body.deliverables_dir) / "report.csv").read_bytes() == b"name,value\na,3\n"
        return GDPValVerifyResponse(**body.model_dump(), reward=0.75)

    monkeypatch.setattr(GDPValResourcesServer, "_grade_deliverables", grade)

    async def post(server_name, url_path, **kwargs):
        calls.append(url_path)
        body = kwargs["json"]
        if hasattr(body, "model_dump"):
            body = body.model_dump(mode="json")
        if url_path == "/seed_session":
            value = await resources.seed_session(resource_request, ResourcesSeedSessionRequest.model_validate(body))
            payload = value.model_dump(mode="json")
        elif url_path == "/v1/agent_sessions":
            assert body["sandbox_access"]["workdir"] == "/workspace"
            payload = {"agent_session_id": body["agent_session_id"]}
        elif url_path.endswith("/v1/responses"):
            assert "PRIVATE RUBRIC" not in json.dumps(body)
            assert "/workspace/output" in json.dumps(body)
            payload = response().model_dump(mode="json")
        elif url_path == "/v1/agent_sessions/close":
            payload = {"agent_session_id": body["agent_session_id"]}
        elif url_path == "/verify":
            value = await resources.verify(GDPValVerifyRequest.model_validate(body), request=resource_request)
            payload = value.model_dump(mode="json")
        elif url_path == "/close_session":
            value = await resources.close_resources_session(
                resource_request, ResourcesCloseSessionRequest.model_validate(body)
            )
            payload = value.model_dump(mode="json")
        else:
            raise AssertionError(url_path)
        return SimpleNamespace(
            ok=True,
            cookies={"session": SimpleNamespace(value="session-cookie")},
            read=AsyncMock(return_value=json.dumps(payload).encode()),
        )

    client = MagicMock(spec=ServerClient)
    client.global_config_dict = config
    client._resolve_base_url.return_value = "http://resources:8000"
    client.post = AsyncMock(side_effect=post)
    environment = SingleAgentTurnEnvironmentServer(
        config=SingleAgentTurnEnvironmentServerConfig(
            **OmegaConf.to_container(environment_config), name=destination, host="127.0.0.1", port=8001
        ),
        server_client=client,
    )
    body = SingleAgentTurnRequest(episode_id={"rollout_id": "sandbox-recipe"}, task=prepared).model_dump(mode="json")
    with TestClient(environment.setup_webserver()) as http:
        reply = http.post("/run", json=body)
    assert reply.status_code == 200, reply.text
    assert reply.json()["failure"] is None, reply.text
    record = _episode_record(reply.json())
    assert record["reward"] == 0.75
    assert record["task_id"] == "task-1"
    assert record["_ng_task_id"] == {"taskset": "gdpval", "task_id": "task-1"}
    assert calls.index("/v1/agent_sessions/close") < calls.index("/verify") < calls.index("/close_session")
    assert not resources._sessions
    sandbox.stop.assert_awaited_once()


def test_prepare_command_preserves_flat_output_by_default(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "flat.jsonl"
    source.write_text(json.dumps(row()) + "\n")
    monkeypatch.setattr(sys, "argv", ["task_data", "--input", str(source), "--output", str(output)])
    main()
    assert json.loads(output.read_text()) == prepare_row(row())
    with pytest.raises(FileExistsError):
        main()
