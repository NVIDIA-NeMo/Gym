# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The episode adapter preserves the existing environment-server boundary."""

import asyncio
import copy
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
from nemo_gym.core import FailureBoundary, FailureReport, FailureReportRow, NgFailureEvent, failure_report_for
from nemo_gym.server_runtime import SingleAgentTurnRequest, SingleAgentTurnResponse
from tests.helpers.single_agent_turn import _agent_response


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
                tool_servers=[],
            ),
            server_client=downstream,
        )
        request = environment._episode_request_from_row(forwarded)
        assert request.task.task_input.responses_create_params.input == "original"


def test_tool_executions_in_the_agent_reply_reach_the_agent_observations():
    from nemo_gym.rollouts.harness_capabilities import inspect_record
    from nemo_gym.rollouts.trajectory_builder import _attach_trajectory_record

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
    from nemo_gym.config import (
        GlobalConfigDictParser,
        GlobalConfigDictParserConfig,
        get_first_server_config_dict,
    )

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


def direct_server(upstream):
    adapter, downstream = server(upstream)
    environment = downstream.global_config_dict["env-text"].environment_servers
    settings = OmegaConf.to_container(environment.legacy_agent, resolve=True)
    settings["resources_server"] = METRICS_SERVER
    del environment["legacy_agent"]
    environment["single_agent_turn"] = settings
    return type(adapter)(config=adapter.config, server_client=downstream), downstream


def direct_row():
    from benchmarks.partner_if.tests.fixture_data import TASK

    row = copy.deepcopy(TASK)
    row["verifier_metadata"]["partner_if"]["task_id"] = "task-text"
    return row | {
        "_ng_task_index": 7,
        "_ng_rollout_index": 2,
        "_ng_attempt_index": 3,
        "_ng_rollout_id": "capture-7",
        "agent_ref": {"type": "responses_api_agents", "name": "dispatcher"},
        "task_source": "dispatcher",
    }


def direct_reply(row, *, rollout_id="capture-7", attempt=3, taskset="agent-text", task_id="7"):
    return {
        "episode_id": {"rollout_id": rollout_id, "attempt": attempt},
        "task_id": {"taskset": taskset, "task_id": task_id},
        "result": {
            "responses_create_params": row["responses_create_params"],
            "response": _agent_response().model_dump(mode="json"),
            "reward": 1.0,
            "grading_result": {"original": True},
        },
    }


@pytest.mark.parametrize(
    "identity", ["positional", "explicit", "attempt-suffix", "task-id", "problem-id", "default-taskset"]
)
def test_direct_episode_request_preserves_frozen_actor_identity_sampling_and_agent_context(identity):
    from benchmarks.partner_if.tests.fixture_data import PARAMS
    from responses_api_agents.ngc_recorded_context.replay import actor_identity

    row = direct_row()
    row["_ng_model_sampling"] = {"temperature": 0.7, "top_p": 0.95, "max_output_tokens": 8192}
    row["_ng_skip_token_capture"] = True
    row["agent_context"] = {
        "agent": {"values": {"unchanged": "context"}, "artifacts": {"skill": {"path": "skills/example"}}}
    }
    expected = {"rollout_id": "capture-7", "attempt": 3, "taskset": "agent-text", "task_id": "7"}
    if identity == "positional":
        row.pop("_ng_rollout_id")
        expected["rollout_id"] = "7-2"
    elif identity == "attempt-suffix":
        row["_ng_rollout_id"] = "capture-7-a3"
        row.pop("_ng_attempt_index")
    elif identity == "task-id":
        row["task_id"] = "caller-task"
        expected["task_id"] = "caller-task"
    elif identity == "problem-id":
        row.update(problem_id="caller-problem", instance_id="other-instance")
        expected["task_id"] = "caller-problem"
    elif identity == "default-taskset":
        row.pop("task_source")
        expected["taskset"] = "partner_if"
    before = copy.deepcopy(row)
    envelope = direct_reply(row, **expected)
    adapter, downstream = direct_server(_Upstream(json.dumps(envelope).encode()))
    with TestClient(adapter.setup_webserver()) as client:
        response = client.post("/run", json=row)
    assert response.status_code == 200
    assert row == before
    [(destination, path, kwargs)] = downstream.calls
    assert (destination, path) == ("env-text", "/run")
    wire = json.loads(kwargs["data"])
    # This is the actual direct endpoint's strict request schema, including its typed nested actor parameters.
    episode = SingleAgentTurnRequest.model_validate(wire)
    assert episode.episode_id.model_dump() == {"rollout_id": expected["rollout_id"], "attempt": expected["attempt"]}
    assert episode.task.task_id.model_dump() == {"taskset": expected["taskset"], "task_id": expected["task_id"]}
    # Gym may omit completed message status; every actor-visible field and tool remains frozen.
    assert actor_identity(wire["task"]["task_input"]["responses_create_params"]) == actor_identity(PARAMS)
    assert "parallel_tool_calls" not in wire["task"]["task_input"]["responses_create_params"]
    assert wire["model_sampling"] == row["_ng_model_sampling"] and wire["skip_token_capture"] is True
    assert wire["task"]["task_input"]["agent_context"] == row["agent_context"]
    assert episode.task.task_input.task_data == {
        key: value
        for key, value in row.items()
        if not key.startswith("_ng_")
        and key not in {"responses_create_params", "agent_ref", "task_source", "agent_context"}
    }
    assert response.json()["agent_ref"] == {"name": "agent-text"}
    assert response.json()["reward"] == 1 and response.json()["grading_result"] == {"original": True}


@pytest.mark.parametrize(
    "invalid",
    [
        {"agent_ref": {"name": "other"}},
        {"task_source": "other"},
        {"_ng_attempt_index": True},
        {"_ng_attempt_index": -1},
        {"_ng_attempt_index": "3"},
        {"_ng_rollout_id": "unsafe/capture"},
        {"_ng_rollout_id": "capture-7-a3"},
        {"_ng_model_sampling": {"top_p": 0.0}},
        {"agent_context": {"other-role": {"values": {"tag": "wrong"}}}},
    ],
)
def test_invalid_direct_episode_input_is_rejected_before_an_environment_is_called(invalid):
    adapter, downstream = direct_server(_Upstream(b"unused"))
    with TestClient(adapter.setup_webserver()) as client:
        response = client.post("/run", json=direct_row() | invalid)
    assert response.status_code == 422
    assert downstream.calls == []


@pytest.mark.parametrize("fault", ["json", "shape", "result", "ambiguous", "episode", "attempt", "task", "taskset"])
def test_malformed_or_mismatched_direct_replies_are_bad_gateway_without_a_flat_success(fault):
    row = direct_row()
    envelope = direct_reply(row)
    if fault == "shape":
        envelope = {"reward": 1.0}
    elif fault == "result":
        envelope["result"].pop("response")
    elif fault == "ambiguous":
        envelope["failure"] = {"failure_reason": "conflicting outcome", "terminal": False}
    elif fault == "episode":
        envelope["episode_id"]["rollout_id"] = "other-capture"
    elif fault == "attempt":
        envelope["episode_id"]["attempt"] = 4
    elif fault == "task":
        envelope["task_id"]["task_id"] = "other-task"
    elif fault == "taskset":
        envelope["task_id"]["taskset"] = "other-taskset"
    raw = b"not JSON" if fault == "json" else json.dumps(envelope).encode()
    adapter, downstream = direct_server(_Upstream(raw))
    with TestClient(adapter.setup_webserver()) as client:
        response = client.post("/run", json=row)
    assert response.status_code == 502
    assert response.json() == {"detail": "Invalid single-agent episode response"}
    assert len(downstream.calls) == 1


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("failure_kind", [None, "provider_unavailable"])
def test_direct_failure_keeps_terminality_failure_report_and_all_partial_evidence(terminal, failure_kind):
    row = direct_row()
    envelope = direct_reply(row)
    invocation = {"kind": "agent_invocation", "invocation_id": "owned", "status": "failed"}
    observations = {
        "source": "controlled-fixture",
        "records": [invocation],
        "gaps": [{"code": "known-gap", "invocation_id": "owned"}],
    }
    trajectory = {"task_id": "7", "rollout_id": "capture-7-a3", "invocations": [invocation]}
    verifiers = [{"invocation_id": "judge", "status": "failed", "failure_kind": "judge_failed"}]
    event = NgFailureEvent(
        boundary=FailureBoundary.ENVIRONMENT,
        failure_kind="environment_server_failed",
        failure_reason="fixture failure",
    )
    report = failure_report_for(event).model_copy(update={"omitted": True})
    envelope.pop("result")
    envelope["failure"] = {
        "failure_reason": "fixture failure",
        "terminal": terminal,
        "failure_kind": failure_kind,
        "stage": "agent",
        "partial_response": _agent_response().model_dump(mode="json"),
        "agent_observations": observations,
        "trajectory": trajectory,
        "verifier_invocations": verifiers,
    }
    envelope["failure_report"] = report.model_dump(mode="json")
    expected = SingleAgentTurnResponse.model_validate(envelope)
    adapter, downstream = direct_server(_Upstream(json.dumps(envelope).encode()))
    with TestClient(adapter.setup_webserver()) as client:
        response = client.post("/run", json=row)
    assert response.status_code == 200
    result = response.json()
    assert result["_ng_failure_class"] == (failure_kind or "environment_server_failed")
    assert result["_ng_failure_kind"] == failure_kind
    assert result["_ng_failure_terminal"] is terminal
    assert result["_ng_failure_message"] == "fixture failure" and result["_ng_failure_stage"] == "agent"
    assert result["_ng_failure_partial_response"] == expected.failure.partial_response.model_dump(mode="json")
    assert result["ng_agent_observations"] == expected.failure.agent_observations.model_dump(mode="json")
    assert result["ng_trajectory"] == expected.failure.trajectory.model_dump(mode="json")
    assert result["ng_verifier_invocations"] == [
        item.model_dump(mode="json") for item in expected.failure.verifier_invocations
    ]
    assert FailureReportRow.report(result) == report
    assert result["agent_ref"] == {"name": "agent-text"}
    assert "reward" not in result and len(downstream.calls) == 1


@pytest.mark.parametrize("own_selection", [False, True])
def test_direct_success_retains_its_grade_selection_and_cleanup_failure_events(own_selection):
    row = direct_row()
    envelope = direct_reply(row)
    judged = failure_report_for(
        NgFailureEvent(boundary=FailureBoundary.RESOURCES_SERVER, failure_kind="judge_failed", failure_reason="judge")
    )
    cleanup = failure_report_for(
        NgFailureEvent(
            boundary=FailureBoundary.ENVIRONMENT, failure_kind="environment_server_failed", failure_reason="cleanup"
        )
    )
    if own_selection:
        envelope["result"].update(FailureReportRow.fields(judged))
    envelope["failure_report"] = FailureReport(events=cleanup.events).model_dump(mode="json")
    adapter, _ = direct_server(_Upstream(json.dumps(envelope).encode()))
    with TestClient(adapter.setup_webserver()) as client:
        response = client.post("/run", json=row)
    result = response.json()
    assert result["reward"] == 1 and result["grading_result"] == {"original": True}
    report = FailureReportRow.report(result)
    assert set(cleanup.events) <= report.events.keys()
    assert report.failure_id == (judged.failure_id if own_selection else None)


@pytest.mark.parametrize("status", [401, 422, 503])
def test_direct_http_failures_and_repeated_headers_are_opaque_without_retry(status):
    body = b'{"detail":"original provider failure"}'
    headers = CIMultiDict([("set-cookie", "one=1"), ("set-cookie", "two=2"), ("x-failure", "preserved")])
    adapter, downstream = direct_server(_Upstream(body, status=status, headers=headers))
    with TestClient(adapter.setup_webserver()) as client:
        response = client.post("/run", json=direct_row(), headers={"cookie": "session=original", "x-trace": "trace"})
    assert response.status_code == status and response.content == body
    assert response.headers.get_list("set-cookie") == ["one=1", "two=2"]
    assert response.headers["x-failure"] == "preserved"
    assert response.headers["content-length"] == str(len(body))
    [(destination, path, kwargs)] = downstream.calls
    assert (destination, path) == ("env-text", "/run")
    assert kwargs["cookies"] == {"session": "original"} and kwargs["headers"]["x-trace"] == "trace"
    assert not {"cookie", "host", "content-length"} & kwargs["headers"].keys()


def test_direct_environment_requires_a_host_selected_resources_binding():
    adapter, downstream = direct_server(_Upstream(b"unused"))
    del downstream.global_config_dict["env-text"].environment_servers.single_agent_turn["resources_server"]
    with pytest.raises(ValueError, match="bind one resources server"):
        type(adapter)(config=adapter.config, server_client=downstream)


def test_direct_success_keeps_authoritative_scoped_evidence_and_native_extras():
    row = direct_row()
    envelope = direct_reply(row)
    scoped = {"kind": "agent_invocation", "invocation_id": "session:root", "status": "completed"}
    envelope["result"].update(
        ng_agent_observations={"source": "session", "records": [scoped]},
        ng_trajectory={"task_id": "7", "rollout_id": "capture-7-a3", "invocations": [scoped]},
        replay_diagnostics={"termination": "ordering_passed", "native_fact": "unchanged"},
    )
    # The response's private per-activation source is not the canonical, episode-wide owner namespace.
    envelope["result"]["response"]["_ng_trajectory"] = {
        "task_id": "7",
        "rollout_id": "capture-7-a3",
        "invocations": [{"invocation_id": "root"}],
        "tool_calls": [{"invocation_id": "root", "tool_call_id": "raw-tool", "output": "source only"}],
    }
    expected = SingleAgentTurnResponse.model_validate(envelope).result.model_dump(mode="json")
    headers = CIMultiDict([("set-cookie", "one=1"), ("set-cookie", "two=2"), ("x-native", "preserved")])
    adapter, _ = direct_server(_Upstream(json.dumps(envelope).encode(), headers=headers))
    with TestClient(adapter.setup_webserver()) as client:
        response = client.post("/run", json=row)
    assert response.status_code == 200
    assert response.json() == expected | {"agent_ref": {"name": "agent-text"}}
    assert response.headers.get_list("set-cookie") == ["one=1", "two=2"]
    assert response.headers["x-native"] == "preserved"
    assert response.headers["content-length"] == str(len(response.content))


def test_direct_row_without_capture_identity_is_rejected_before_dispatch():
    row = direct_row()
    for key in ("_ng_rollout_id", "_ng_task_index", "_ng_rollout_index"):
        row.pop(key)
    adapter, downstream = direct_server(_Upstream(b"unused"))
    with TestClient(adapter.setup_webserver()) as client:
        response = client.post("/run", json=row)
    assert response.status_code == 422 and downstream.calls == []


async def test_direct_cancellation_propagates_without_a_flat_failure(monkeypatch):
    adapter, downstream = direct_server(_Upstream(b"unused"))

    async def cancel(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(type(downstream), "post", cancel)
    raw = json.dumps(direct_row()).encode()

    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    with pytest.raises(asyncio.CancelledError):
        await adapter.run(Request({"type": "http", "headers": []}, receive))
