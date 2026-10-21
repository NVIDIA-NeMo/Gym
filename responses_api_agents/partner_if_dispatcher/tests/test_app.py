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
from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.server_utils import BaseServerConfig


METRICS_SERVER = {"type": "resources_servers", "name": "partner_if"}


def server(upstream=None):
    module = importlib.import_module("responses_api_agents.partner_if_dispatcher.app")
    profiles = ("text", "tools", "markdown", "swe")
    routes = {f"task-{p}": {"type": "environment_servers", "name": f"env-{p}"} for p in profiles}
    client = _Client(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create(
            {
                f"env-{p}": {
                    "environment_servers": {
                        "legacy_agent": {"agent_server": {"type": "responses_api_agents", "name": f"agent-{p}"}}
                    }
                }
                for p in profiles
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


@pytest.mark.parametrize("profile", ["text", "tools", "markdown", "swe"])
def test_routes_the_complete_episode_without_rewriting_native_fields(profile):
    actor = {
        "responses_create_params": {"input": "original"},
        "verifier_metadata": {"partner_if": {"task_id": f"task-{profile}"}},
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
    assert (destination, path) == (f"env-{profile}", "/run")
    forwarded = json.loads(kwargs["data"])
    assert forwarded == {**actor, "agent_ref": {"name": f"agent-{profile}"}, "task_source": f"agent-{profile}"}
    assert kwargs["cookies"] == {"session": "original"}
    assert kwargs["headers"]["x-trace"] == "trace-7"
    assert "host" not in kwargs["headers"] and "content-length" not in kwargs["headers"]

    # Exercise the actual session boundary, whose host-route validation rejects
    # an unchanged dispatcher agent_ref/task_source before any model call.
    if profile in {"text", "tools"}:
        from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
        from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer

        environment = SingleAgentTurnLegacyEnvironmentServer(
            config=SingleAgentTurnEnvironmentServerConfig(
                name=f"env-{profile}",
                entrypoint="app.py",
                host="localhost",
                port=8111,
                cleanup_timeout_seconds=180,
                agent_server={"type": "responses_api_agents", "name": f"agent-{profile}"},
                resources_server={"type": "resources_servers", "name": "resources"},
            ),
            server_client=downstream,
        )
        request = environment._episode_request_from_row(forwarded)
        assert request.task.task_input.responses_create_params.input == "original"


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
