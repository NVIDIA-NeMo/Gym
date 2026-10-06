# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import runpy
from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import ConfigDict

import nemo_gym.server_utils
from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
)
from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
from nemo_gym.config_types import AgentServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId, MaterializedTask, TaskId
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import BaseServerConfig, ServerClient, SimpleServer
from nemo_gym.single_agent_turn_types import SingleAgentTurnRequest, SingleAgentTurnTaskInput


class _Cookie:
    value = "cookie-value"


def test_legacy_module_exports_app_for_multi_worker_import(monkeypatch: pytest.MonkeyPatch) -> None:
    worker_app = object()
    monkeypatch.setattr(nemo_gym.server_utils, "is_nemo_gym_fastapi_entrypoint", lambda _: True)
    monkeypatch.setattr(SimpleServer, "run_webserver", classmethod(lambda _: worker_app))

    namespace = runpy.run_path(
        str(Path(__file__).parents[1] / "app.py"),
        run_name="single_agent_turn_legacy.worker_test",
    )

    assert namespace["app"] is worker_app


class _Response:
    ok = True
    cookies = {"session": _Cookie()}

    def __init__(self, body: dict) -> None:
        self.body = orjson.dumps(body)

    async def read(self) -> bytes:
        return self.body


def _agent_response() -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )


class _Client(ServerClient):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    calls: list[tuple[str, str, dict]]
    responses: list[_Response]

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Response:
        self.calls.append((server_name, url_path, kwargs))
        response = self.responses.pop(0)
        payload = orjson.loads(response.body)
        body = kwargs.get("json")
        if url_path == "/seed_session" and "resources_session_id" in body:
            payload["resources_session_id"] = body["resources_session_id"]
        elif url_path == "/v1/agent_sessions":
            payload["agent_session_id"] = body["agent_session_id"]
        elif url_path == "/v1/agent_sessions/close":
            payload["agent_session_id"] = body["agent_session_id"]
        elif url_path == "/close_session":
            payload["resources_session_id"] = body["resources_session_id"]
        return _Response(payload)

    def _resolve_base_url(self, server_name: str) -> str:
        return f"http://{server_name}:8000"


def _environment_server() -> tuple[SingleAgentTurnEnvironmentServer, _Client]:
    global_config = OmegaConf.create(
        {
            "resources": {"resources_servers": {"test": {"host": "resources", "port": 8000, "entrypoint": "app.py"}}},
            "agent": {"responses_api_agents": {"test": {"host": "agent", "port": 8001, "entrypoint": "app.py"}}},
        }
    )
    response = _agent_response()
    client = _Client(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=global_config,
        calls=[],
        responses=[
            _Response({"resources_session_id": "resources-session"}),
            _Response({"agent_session_id": "agent-session"}),
            _Response(response.model_dump(mode="json")),
            _Response(
                {
                    "agent_session_id": "agent-session",
                    "resources_cookies": {"session": "updated-cookie"},
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
    config = SingleAgentTurnEnvironmentServerConfig(
        name="environment",
        host="environment",
        port=8002,
        entrypoint="app.py",
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        agent_server=AgentServerRef(type="responses_api_agents", name="agent"),
        default_episode_timeout_seconds=10,
        cleanup_timeout_seconds=10,
    )
    return SingleAgentTurnEnvironmentServer(config=config, server_client=client), client


async def test_legacy_compatibility_is_a_separate_environment_deployment() -> None:
    environment_server, client = _environment_server()
    adapter = SingleAgentTurnLegacyEnvironmentServer(config=environment_server.config, server_client=client)
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
    assert "verification" not in result
    assert "ng_agent_observations" not in result


def test_legacy_adapter_forwards_aggregate_metrics_to_resources() -> None:
    environment_server, client = _environment_server()
    client.responses = [_Response({"agent_metrics": {"mean/reward": 0.5}})]
    adapter = SingleAgentTurnLegacyEnvironmentServer(config=environment_server.config, server_client=client)

    response = TestClient(adapter.setup_webserver()).post(
        "/aggregate_metrics",
        json={"verify_responses": [{"_ng_task_index": 0, "reward": 0.5}]},
    )

    assert response.status_code == 200
    assert response.json()["agent_metrics"] == {"mean/reward": 0.5}
    assert [(server, path) for server, path, _ in client.calls] == [("resources", "/aggregate_metrics")]


def test_legacy_adapter_preserves_task_identity_and_data():
    row = {
        "problem_id": "problem-1",
        "instance_id": "instance-1",
        "responses_create_params": {"input": "fix it", "temperature": 0.4},
        "run_script": "verifier\nscript\n",
        "verifier_metadata": {"answer": "expected"},
        "agent_ref": {"name": "old-agent"},
        "skills_ref": "old-skills",
        "task_source": "old-source",
        "_ng_task_index": 7,
        "_ng_rollout_index": 2,
        "_ng_attempt_index": 1,
    }
    row.update(task_source="resources", agent_ref={"name": "agent"})
    adapter = SingleAgentTurnLegacyEnvironmentServer(
        config=SingleAgentTurnEnvironmentServerConfig(
            name="environment",
            host="localhost",
            port=1,
            entrypoint="app.py",
            cleanup_timeout_seconds=10,
            resources_server={"type": "resources_servers", "name": "resources"},
            agent_server={"type": "responses_api_agents", "name": "agent"},
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    request = adapter._episode_request_from_row(row)
    assert (request.task.task_id.taskset, request.task.task_id.task_id) == ("resources", "problem-1")
    assert request.task.task_input.task_data == {
        "problem_id": "problem-1",
        "instance_id": "instance-1",
        "run_script": "verifier\nscript\n",
        "verifier_metadata": {"answer": "expected"},
    }
    assert request.episode_id.attempt == 1


@pytest.mark.parametrize("task_id_field", ["task_id", "problem_id", "instance_id"])
def test_explicit_identity_does_not_require_collector_indexes(task_id_field: str) -> None:
    environment_server, client = _environment_server()
    adapter = SingleAgentTurnLegacyEnvironmentServer(config=environment_server.config, server_client=client)

    response = TestClient(adapter.setup_webserver()).post(
        "/run",
        json={
            task_id_field: "task",
            "_ng_rollout_id": "explicit-rollout",
            "_ng_attempt_index": 2,
            "responses_create_params": {"input": "task"},
        },
    )

    assert response.status_code == 200
    assert response.json()["reward"] == 1.0
    seed_body = client.calls[0][2]["json"]
    assert seed_body["task_id"] == {"taskset": "resources", "task_id": "task"}
    assert seed_body["episode_id"] == {"rollout_id": "explicit-rollout", "attempt": 2}
    assert client.calls[2][1] == "/ng-rollout/explicit-rollout-a2/v1/responses"


@pytest.mark.parametrize("task_fields, expected_task_id", [({}, "3"), ({"task_id": 0}, "0")])
def test_task_identity_preserves_index_fallback_and_zero(task_fields: dict[str, int], expected_task_id: str) -> None:
    environment_server, client = _environment_server()
    adapter = SingleAgentTurnLegacyEnvironmentServer(config=environment_server.config, server_client=client)

    request = adapter._episode_request_from_row(
        {
            **task_fields,
            "_ng_task_index": 3,
            "_ng_rollout_index": 2,
            "responses_create_params": {"input": "task"},
        }
    )

    assert request.task.task_id.task_id == expected_task_id
    assert request.episode_id == EpisodeId(rollout_id="3-2", attempt=0)


@pytest.mark.parametrize("task_source", ["resources", "agent"])
def test_task_source_may_name_the_bound_resources_server_or_agent(task_source: str) -> None:
    # Collation stamps task_source with the instance that declares the dataset, usually the agent.
    environment_server, client = _environment_server()
    adapter = SingleAgentTurnLegacyEnvironmentServer(config=environment_server.config, server_client=client)

    request = adapter._episode_request_from_row(
        {
            "task_source": task_source,
            "_ng_task_index": 0,
            "_ng_rollout_index": 0,
            "responses_create_params": {"input": "task"},
        }
    )

    assert request.task.task_id.taskset == task_source
    assert "task_source" not in request.task.task_input.task_data


def test_task_source_naming_another_instance_is_rejected() -> None:
    environment_server, client = _environment_server()
    adapter = SingleAgentTurnLegacyEnvironmentServer(config=environment_server.config, server_client=client)

    with pytest.raises(ValueError, match="names none of this Environment Server's instances"):
        adapter._episode_request_from_row(
            {
                "task_source": "other_agent",
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "responses_create_params": {"input": "task"},
            }
        )


async def test_flat_rows_and_episode_requests_project_the_same_result() -> None:
    legacy_environment, legacy_client = _environment_server()
    episode_environment, episode_client = _environment_server()
    legacy_adapter = SingleAgentTurnLegacyEnvironmentServer(
        config=legacy_environment.config,
        server_client=legacy_client,
    )
    episode_adapter = SingleAgentTurnLegacyEnvironmentServer(
        config=episode_environment.config,
        server_client=episode_client,
    )
    flat_row = {
        "_ng_task_index": 3,
        "_ng_rollout_index": 2,
        "_ng_attempt_index": 1,
        "instance_id": "task",
        "benchmark_field": "input",
        "responses_create_params": {"input": "task"},
    }
    episode_request = SingleAgentTurnRequest(
        episode_id=EpisodeId(rollout_id="3-2", attempt=1),
        task=MaterializedTask(
            task_id=TaskId(taskset="resources", task_id="task"),
            task_input=SingleAgentTurnTaskInput(
                responses_create_params={"input": "task"},
                task_data={"instance_id": "task", "benchmark_field": "input"},
            ),
        ),
    )

    legacy_result = await legacy_adapter.run_legacy(flat_row)
    episode_result = await episode_adapter.run_legacy(episode_request.model_dump(mode="json"))

    assert episode_result == legacy_result


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("failure_kind", [None, "judge_failed"])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("terminal", [False, True])
async def test_protocol_failure_metadata_survives_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy: bool,
    failure_kind: str | None,
    partial: bool,
    terminal: bool,
) -> None:
    from unittest.mock import AsyncMock

    import nemo_gym.rollout_collection as collection
    from nemo_gym.episode_types import EpisodeFailure
    from nemo_gym.rollout_outcomes import RolloutFailure
    from nemo_gym.rollout_records import logical_rollout_id
    from nemo_gym.rollout_store import RolloutStore
    from nemo_gym.single_agent_turn_types import SingleAgentTurnFailure, SingleAgentTurnResponse
    from tests.unit_tests.test_rollout_collection import FakeResponse, install_fake_server_client

    environment, client = _environment_server()
    if legacy:
        environment = SingleAgentTurnLegacyEnvironmentServer(config=environment.config, server_client=client)
    answer = _agent_response() if partial else None

    async def failed_episode(
        self: SingleAgentTurnEnvironmentServer, request: SingleAgentTurnRequest
    ) -> SingleAgentTurnResponse:
        return SingleAgentTurnResponse(
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            failure=SingleAgentTurnFailure(
                failure_reason="Judge unavailable",
                failure_kind=failure_kind,
                stage="verification",
                terminal=terminal,
                partial_response=answer,
            ),
        )

    monkeypatch.setattr(SingleAgentTurnEnvironmentServer, "run_request", failed_episode)
    with TestClient(environment.setup_webserver()) as http:

        async def post(**kwargs):
            response = http.post("/run", json=kwargs["json"])
            assert response.status_code == 200
            return FakeResponse(response.status_code, response.json())

        collector = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        collector.global_config_dict = OmegaConf.create(
            {
                "agent": {"responses_api_agents": {"test": {}}},
                "environment": {
                    "environment_servers": {
                        "single_agent_turn_legacy" if legacy else "single_agent_turn": {
                            "agent_server": {"name": "agent"},
                            "resources_server": {"name": "resources"},
                        }
                    }
                },
            }
        )
        collector.global_config_dict["resources"] = {"resources_servers": {"example": {}}}
        monkeypatch.setattr(collection, "get_global_config_dict", lambda: {})
        source = tmp_path / "input.jsonl"
        row = (
            {
                "agent_ref": {"name": "agent"},
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "responses_create_params": {"input": "task"},
            }
            if legacy
            else {
                "task_id": {"taskset": "resources", "task_id": "task"},
                "task_input": {"responses_create_params": {"input": "task"}, "task_data": {}},
            }
        )
        source.write_bytes(orjson.dumps(row) + b"\n")
        output = tmp_path / "rollouts.jsonl"
        with pytest.raises(RuntimeError, match="has no score to report"):
            await collection.RolloutCollectionHelper().run_from_config(
                collection.RolloutCollectionConfig(
                    input_jsonl_fpath=str(source),
                    output_jsonl_fpath=str(output),
                    environment_server_routes={"resources": "environment"} if not legacy else {},
                    route_failures_to_sidecar=True,
                    disable_aggregation=True,
                    disable_health_check=True,
                    require_complete=False,
                )
            )

    store = RolloutStore.read(output)
    [saved] = store.failures()
    expected_kind = failure_kind or "environment_server_failed"
    assert saved["_ng_failure_class"] == expected_kind
    assert saved.get("_ng_failure_terminal", False) is terminal
    assert "reward" not in saved and "response" not in saved
    record = RolloutFailure.model_validate(saved["_ng_failure_record"])
    assert record.run_id == store.manifest.run_id
    assert record.episode_id.rollout_id == logical_rollout_id(saved)
    assert (record.source, record.delivery) == ("environment", "delivered")
    assert type(record.failure) is EpisodeFailure
    assert record.failure.failure_kind == failure_kind
    assert record.failure.failure_reason == "Judge unavailable"
    assert record.failure.stage == "verification" and record.failure.terminal is terminal
    assert "partial_response" not in record.model_dump()["failure"]
    assert RolloutFailure.model_validate_json(record.model_dump_json()) == record
    if answer is None:
        assert "_ng_failure_partial_response" not in saved
    else:
        assert saved["_ng_failure_partial_response"] == answer.model_dump(mode="json")
    assert store.coverage()["failed"] == 1 and store.coverage()["measured"] == 0
    assert store.selected("success") == []
