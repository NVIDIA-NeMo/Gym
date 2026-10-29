# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The episode adapter preserves the existing environment-server boundary."""

import asyncio
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from multidict import CIMultiDict
from omegaconf import OmegaConf
from pydantic import ValidationError
from starlette.requests import Request

from environment_servers.legacy_agent.tests.test_app import _Client, _Upstream
from nemo_gym.config import AggregateMetricsRequest, BaseServerConfig


METRICS_SERVER = {"type": "resources_servers", "name": "partner_if"}


def server(upstream=None):
    module = importlib.import_module("responses_api_agents.partner_if_dispatcher.app")
    harnesses = ("text", "tools", "harbor", "swe")
    routes = {f"task-{h}": {"type": "environment_servers", "name": f"env-{h}"} for h in harnesses}
    client = _Client(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create(
            {
                f"env-{h}": {
                    "environment_servers": {
                        "legacy_agent": {"agent_server": {"type": "responses_api_agents", "name": f"agent-{h}"}}
                    }
                }
                for h in harnesses
            }
        ),
        calls=[],
        responses=[upstream or _Upstream(b'{"reward": 1}')],
    )
    config = module.PartnerIFDispatcherConfig(
        entrypoint="app.py",
        name="dispatcher",
        host="localhost",
        port=8100,
        task_routes=routes,
        resources_server=METRICS_SERVER,
    )
    return module.PartnerIFDispatcher(config=config, server_client=client), client


@pytest.mark.parametrize("harness", ["text", "tools", "harbor", "swe"])
def test_routes_the_complete_episode_without_rewriting_native_fields(harness):
    actor = {
        "responses_create_params": {"input": "original"},
        "verifier_metadata": {"partner_if": {"task_id": f"task-{harness}"}},
        "_ng_task_index": 7,
        "_ng_rollout_index": 0,
        "_ng_rollout_id": "capture-7",
        "workdir": "/repo",
        "test_patch": "private tests",
        "task_source": "dispatcher",
        "agent_ref": {"type": "responses_api_agents", "name": "dispatcher"},
    }
    result = {**actor, "reward": 1, "grading_result": {"original": True}}
    raw = json.dumps(actor).encode()
    expected = json.dumps(result).encode()
    adapter, downstream = server(
        _Upstream(expected, headers=CIMultiDict([("set-cookie", "one=1"), ("set-cookie", "two=2")]))
    )
    with TestClient(adapter.setup_webserver()) as client:
        response = client.post(
            "/run",
            content=raw,
            headers={"content-type": "application/json", "cookie": "session=original", "x-trace": "trace-7"},
        )
    assert response.content == expected
    assert response.headers.get_list("set-cookie") == ["one=1", "two=2"]
    [(destination, path, kwargs)] = downstream.calls
    assert (destination, path) == (f"env-{harness}", "/run")
    forwarded = json.loads(kwargs["data"])
    assert forwarded == {**actor, "agent_ref": {"name": f"agent-{harness}"}, "task_source": f"agent-{harness}"}
    assert kwargs["cookies"] == {"session": "original"}
    assert kwargs["headers"]["x-trace"] == "trace-7"
    assert "host" not in kwargs["headers"] and "content-length" not in kwargs["headers"]

    # Exercise the actual session boundary, whose host-route validation rejects
    # an unchanged dispatcher agent_ref/task_source before any model call.
    if harness in {"text", "tools"}:
        from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
        from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer

        environment = SingleAgentTurnLegacyEnvironmentServer(
            config=SingleAgentTurnEnvironmentServerConfig(
                name=f"env-{harness}",
                entrypoint="app.py",
                host="localhost",
                port=8111,
                cleanup_timeout_seconds=180,
                agent_server={"type": "responses_api_agents", "name": f"agent-{harness}"},
                resources_server={"type": "resources_servers", "name": "resources"},
            ),
            server_client=downstream,
        )
        request = environment._episode_request_from_row(forwarded)
        assert request.task.task_input.responses_create_params.input == "original"


def test_tool_executions_in_the_agent_reply_reach_the_agent_observations():
    from nemo_gym.harness_capabilities.checker import inspect_record
    from nemo_gym.rollout_collection import _attach_trajectory_record

    module = importlib.import_module("responses_api_agents.partner_if_dispatcher.app")
    invocation = {
        "kind": "agent_invocation",
        "invocation_id": "root",
        "conversation": [
            {"type": "function_call", "call_id": "call-1", "name": "search", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call-1", "output": "found"},
        ],
    }
    execution = {
        "kind": "tool_call",
        "invocation_id": "root",
        "tool_call_id": "call-1",
        "tool_name": "search",
        "timing_source": "executor",
        "status": "completed",
    }
    result = {
        "reward": 1,
        "ng_agent_observations": {"source": "simple_agent", "records": [invocation]},
        "response": {
            "_ng_trajectory": {
                "task_id": "unscoped",
                "rollout_id": "unscoped",
                "invocations": [invocation],
                "tool_calls": [execution | {"output": "found"}],
            }
        },
    }
    adapter, _ = server(_Upstream(json.dumps(result).encode()))
    with TestClient(adapter.setup_webserver()) as client:
        response = client.post(
            "/run", json={"verifier_metadata": {"partner_if": {"task_id": "task-tools"}}, "task_source": "dispatcher"}
        )
    row = response.json()
    [_, added] = row["ng_agent_observations"]["records"]
    assert {key: added[key] for key in execution} == execution and added.get("output") is None
    # Each execution is added once, and collection joins its output from the conversation.
    assert module._with_reply_tool_observations(response.content) == response.content
    _attach_trajectory_record({"_ng_task_index": 0, "_ng_rollout_index": 0}, row)
    assert [tool["output"] for tool in row["ng_trajectory"]["tool_calls"]] == ["found"]
    assert inspect_record(row)["evidence"]["TE-5"]["verdict"] == "fulfilled"


@pytest.mark.parametrize(
    "payload", [{}, [], {"verifier_metadata": {}}, {"verifier_metadata": {"partner_if": {"task_id": "unknown"}}}]
)
def test_missing_or_unknown_task_never_starts_an_environment(payload):
    adapter, downstream = server()
    with TestClient(adapter.setup_webserver()) as client:
        response = client.post("/run", json=payload)
    assert response.status_code in (400, 422)
    assert downstream.calls == []


def test_request_routing_fields_cannot_change_the_host_selected_route():
    adapter, downstream = server()
    with TestClient(adapter.setup_webserver()) as client:
        response = client.post(
            "/run",
            json={
                "verifier_metadata": {"partner_if": {"task_id": "task-text"}},
                "agent_ref": {"type": "responses_api_agents", "name": "untrusted"},
            },
        )
    assert response.status_code == 422
    assert downstream.calls == []


@pytest.mark.parametrize(
    "status,payload",
    [(503, b'{"detail":"backend unavailable"}'), (200, b'{"reward":0,"mask_sample":true,"failure_kind":"infra"}')],
)
def test_preserves_downstream_failure_status_and_receipt_without_retry(status, payload):
    adapter, downstream = server(_Upstream(payload, status=status))
    with TestClient(adapter.setup_webserver()) as client:
        response = client.post("/run", json={"verifier_metadata": {"partner_if": {"task_id": "task-text"}}})
    assert response.status_code == status and response.content == payload
    assert len(downstream.calls) == 1


async def test_cancellation_is_not_converted_into_an_if_failure(monkeypatch):
    adapter, downstream = server()

    async def cancel(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(type(downstream), "post", cancel)
    raw = b'{"verifier_metadata":{"partner_if":{"task_id":"task-text"}}}'

    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    request = Request({"type": "http", "headers": []}, receive)
    with pytest.raises(asyncio.CancelledError):
        await adapter.run(request)


def test_self_routing_is_rejected_at_configuration_time():
    module = importlib.import_module("responses_api_agents.partner_if_dispatcher.app")
    with pytest.raises(ValueError, match="self"):
        module.PartnerIFDispatcherConfig(
            entrypoint="app.py",
            name="dispatcher",
            host="localhost",
            port=8100,
            task_routes={"task": {"type": "environment_servers", "name": "dispatcher"}},
            resources_server=METRICS_SERVER,
        )


@pytest.mark.parametrize("target", ["missing", "loop"])
def test_missing_environments_and_indirect_self_routes_rejected(target):
    adapter, client = server()
    route = client.global_config_dict["env-text"].environment_servers.legacy_agent
    if target == "loop":
        route.agent_server.name = "dispatcher"
    else:
        del client.global_config_dict["env-text"]
    with pytest.raises(ValueError, match="environment|self"):
        type(adapter)(config=adapter.config, server_client=client)


def test_metrics_resources_server_is_required_at_configuration_time():
    module = importlib.import_module("responses_api_agents.partner_if_dispatcher.app")
    with pytest.raises(ValidationError, match="resources_server"):
        module.PartnerIFDispatcherConfig(
            entrypoint="app.py",
            name="dispatcher",
            host="localhost",
            port=8100,
            task_routes={"task-text": {"type": "environment_servers", "name": "env-text"}},
        )


def test_aggregate_metrics_are_computed_by_the_benchmark_resources_server():
    rows = [
        {"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 1.0, "if_result": {"passed": 2, "total": 2}},
        {"_ng_task_index": 1, "_ng_rollout_index": 0, "reward": 0.0, "if_result": {"passed": 1, "total": 2}},
    ]
    computed = {
        "group_level_metrics": [{"_ng_task_index": 0}, {"_ng_task_index": 1}],
        "agent_metrics": {"pass@1/all_if_pass": 50.0, "pass@1/if_partial": 75.0},
        "key_metrics": {"pass@1[avg-of-1]/all_if_pass": 50.0},
    }
    adapter, downstream = server(_Upstream(json.dumps(computed).encode()))
    with TestClient(adapter.setup_webserver()) as client:
        response = client.post("/aggregate_metrics", json={"verify_responses": rows})
    assert response.status_code == 200
    result = response.json()
    assert result["agent_metrics"] == computed["agent_metrics"]
    assert result["key_metrics"] == computed["key_metrics"]
    assert result["group_level_metrics"] == computed["group_level_metrics"]
    # The rows go unchanged to the scorer; no environment or agent is contacted.
    [(destination, path, kwargs)] = downstream.calls
    assert (destination, path) == ("partner_if", "/aggregate_metrics")
    assert kwargs["json"] == AggregateMetricsRequest(verify_responses=rows)


async def test_metrics_server_failure_is_raised_not_replaced_by_empty_metrics():
    class Unavailable(_Upstream):
        ok = False

        def __init__(self):
            super().__init__(b'{"detail":"metrics unavailable"}', status=503)
            self.content = SimpleNamespace(read=AsyncMock(return_value=self.body))

        def raise_for_status(self):
            raise RuntimeError("resources server returned 503")

    adapter, downstream = server(Unavailable())
    with pytest.raises(RuntimeError, match="503"):
        await adapter.aggregate_metrics(AggregateMetricsRequest(verify_responses=[{"reward": 1.0}]))
    assert [(destination, path) for destination, path, _ in downstream.calls] == [("partner_if", "/aggregate_metrics")]


def test_example_config_runs_each_partner_if_example_in_its_own_environment():
    from nemo_gym import PARENT_DIR
    from nemo_gym.client import ServerClient
    from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig, get_first_server_config_dict

    module = importlib.import_module("responses_api_agents.partner_if_dispatcher.app")
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {"config_paths": ["responses_api_agents/partner_if_dispatcher/configs/partner_if_dispatcher.yaml"]},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    block = get_first_server_config_dict(config, "partner_if_dispatcher")
    settings = OmegaConf.to_container(block, resolve=True) | {"name": "partner_if_dispatcher"}
    # Construction checks every route: one environment server that binds one execution agent.
    dispatcher = module.PartnerIFDispatcher(
        config=module.PartnerIFDispatcherConfig.model_validate(settings),
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=11000), global_config_dict=config
        ),
    )
    assert dispatcher._execution_agents == {
        "partner_if_environment_server": "partner_if_simple_agent",
        "partner_if_mock_tools_environment_server": "partner_if_mock_tools_simple_agent",
    }
    # The example is partner_if's two examples, and each task goes to its own environment.
    examples = PARENT_DIR / "resources_servers/partner_if/data"
    rows = [
        json.loads(line) for line in (PARENT_DIR / settings["datasets"][0]["jsonl_fpath"]).read_text().splitlines()
    ]
    assert rows == [
        json.loads(line)
        for name in ("example.jsonl", "mock_tools_example.jsonl")
        for line in (examples / name).read_text().splitlines()
    ]
    environments = {
        "text_response": "partner_if_environment_server",
        "live_agent": "partner_if_mock_tools_environment_server",
    }
    contracts = [row["verifier_metadata"]["partner_if"] for row in rows]
    assert {task: route["name"] for task, route in settings["task_routes"].items()} == {
        contract["task_id"]: environments[contract.get("evidence_mode", "text_response")] for contract in contracts
    }
    assert settings["resources_server"]["name"] == "partner_if"
