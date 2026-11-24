# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""partner_if's Environment Server runs the text tasks itself and relays every other task to its routed server."""

import asyncio
import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from multidict import CIMultiDict
from omegaconf import OmegaConf
from pydantic import ValidationError
from starlette.requests import Request

from environment_servers.partner_if.app import (
    PartnerIFEnvironmentServer,
    PartnerIFEnvironmentServerConfig,
    _with_reply_tool_observations,
)
from environment_servers.single_agent_turn_legacy.app import (
    SingleAgentTurnLegacyEnvironmentServer,
    SingleAgentTurnLegacyEnvironmentServerConfig,
)
from nemo_gym.config import AgentServerRef, AggregateMetricsRequest, BaseServerConfig, ResourcesServerRef
from tests.helpers.single_agent_turn import _agent_response, _Response
from tests.helpers.single_agent_turn import _Client as _SessionClient


FAMILIES = ("tools", "harbor", "swe")


class _Upstream:
    """A routed server's /run reply."""

    ok = True

    def __init__(self, body: bytes, *, status: int = 200, headers: CIMultiDict | None = None) -> None:
        self.body = body
        self.status = status
        self.headers = headers or CIMultiDict()

    async def read(self) -> bytes:
        return self.body


class _Client(_SessionClient):
    """The scripted session calls of an episode run here, and the replies of the routed servers' /run."""

    relays: list[_Upstream] = []

    async def post(self, server_name: str, url_path: str, **kwargs):
        if url_path == "/run":
            self.calls.append((server_name, url_path, kwargs))
            return self.relays.pop(0)
        return await super().post(server_name, url_path, **kwargs)


def _session_responses(reward: float = 1.0) -> list[_Response]:
    response = _agent_response().model_dump(mode="json")
    return [
        _Response({"resources_session_id": "resources-session"}),
        _Response({"agent_session_id": "agent-session"}),
        _Response(response),
        _Response({"agent_session_id": "agent-session"}),
        _Response({"responses_create_params": {"input": "task"}, "response": response, "reward": reward}),
        _Response({"resources_session_id": "resources-session"}),
    ]


def _global_config() -> dict:
    config = {
        "resources": {
            "resources_servers": {"partner_if": {"host": "resources", "port": 8000, "entrypoint": "app.py"}}
        },
        "agent": {"responses_api_agents": {"simple_agent": {"host": "agent", "port": 8001, "entrypoint": "app.py"}}},
    }
    for family in FAMILIES:
        config[f"env-{family}"] = {
            "environment_servers": {
                "single_agent_turn_legacy": {
                    "agent_server": {"type": "responses_api_agents", "name": f"agent-{family}"}
                }
            }
        }
    return config


def _config(**routes: str) -> PartnerIFEnvironmentServerConfig:
    return PartnerIFEnvironmentServerConfig(
        name="partner_if_environment",
        entrypoint="app.py",
        host="localhost",
        port=8100,
        cleanup_timeout_seconds=10,
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        agent_server=AgentServerRef(type="responses_api_agents", name="agent"),
        resources_tool_transports=[],
        task_routes={
            "task-text": {"type": "environment_servers", "name": "partner_if_environment"},
            **{f"task-{family}": {"type": "environment_servers", "name": f"env-{family}"} for family in FAMILIES},
            **{task: {"type": "environment_servers", "name": name} for task, name in routes.items()},
        },
    )


def server(*relays: _Upstream, sessions: list[_Response] | None = None, global_config: dict | None = None):
    client = _Client(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create(global_config or _global_config()),
        calls=[],
        responses=list(sessions or []),
        relays=list(relays or [_Upstream(b'{"reward": 1}')]),
    )
    return PartnerIFEnvironmentServer(config=_config(), server_client=client), client


def _row(task: str, **extra) -> dict:
    return {
        "responses_create_params": {"input": "original"},
        "verifier_metadata": {"partner_if": {"task_id": task}},
        "_ng_task_index": 7,
        "_ng_rollout_index": 0,
        "task_source": "agent",
        "agent_ref": {"type": "responses_api_agents", "name": "agent"},
        **extra,
    }


def test_text_task_runs_its_episode_here():
    environment, client = server(sessions=_session_responses(reward=1.0))
    with TestClient(environment.setup_webserver()) as http:
        response = http.post("/run", json=_row("task-text"))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["reward"] == 1.0
    assert result["agent_ref"] == {"name": "agent"}
    # The single-agent-turn protocol ran here, against this server's own pairing; nothing was relayed.
    assert [(name, path) for name, path, _ in client.calls] == [
        ("resources", "/seed_session"),
        ("agent", "/v1/agent_sessions"),
        ("agent", "/ng-rollout/7-0/v1/responses"),
        ("agent", "/v1/agent_sessions/close"),
        ("resources", "/verify"),
        ("resources", "/close_session"),
    ]
    # The Resources Server grades the row's own request, as an agent's /run would send it.
    verify = client.calls[4][2]["json"]
    assert verify["responses_create_params"]["input"] == "original"
    assert verify["verifier_metadata"] == {"partner_if": {"task_id": "task-text"}}


@pytest.mark.parametrize("family", FAMILIES)
def test_routes_other_tasks_to_their_server_without_rewriting_row_fields(family):
    actor = _row(f"task-{family}", _ng_rollout_id="capture-7", workdir="/repo", test_patch="private tests")
    result = {**actor, "reward": 1, "grading_result": {"original": True}}
    expected = json.dumps(result).encode()
    environment, client = server(
        _Upstream(expected, headers=CIMultiDict([("set-cookie", "one=1"), ("set-cookie", "two=2")]))
    )
    with TestClient(environment.setup_webserver()) as http:
        response = http.post(
            "/run",
            content=json.dumps(actor).encode(),
            headers={"content-type": "application/json", "cookie": "session=original", "x-trace": "trace-7"},
        )
    assert response.content == expected
    assert response.headers.get_list("set-cookie") == ["one=1", "two=2"]
    [(destination, path, kwargs)] = client.calls
    assert (destination, path) == (f"env-{family}", "/run")
    forwarded = json.loads(kwargs["data"])
    assert forwarded == {**actor, "agent_ref": {"name": f"agent-{family}"}, "task_source": f"agent-{family}"}
    assert kwargs["cookies"] == {"session": "original"}
    assert kwargs["headers"]["x-trace"] == "trace-7"
    assert "host" not in kwargs["headers"] and "content-length" not in kwargs["headers"]

    # The routed server accepts the relayed row: its host-route validation rejects an unadapted
    # agent_ref/task_source before any model call.
    routed_server = SingleAgentTurnLegacyEnvironmentServer(
        config=SingleAgentTurnLegacyEnvironmentServerConfig(
            name=f"env-{family}",
            entrypoint="app.py",
            host="localhost",
            port=8111,
            agent_server={"type": "responses_api_agents", "name": f"agent-{family}"},
            resources_server={"type": "resources_servers", "name": f"resources-{family}"},
            tool_servers=[],
        ),
        server_client=client,
    )
    request = routed_server._episode_request_from_row(forwarded)
    assert request.task.task_input.responses_create_params.input == "original"
    with pytest.raises(ValueError, match="task_source"):
        routed_server._episode_request_from_row(actor)


def test_tool_executions_in_the_agent_reply_reach_the_agent_observations():
    from nemo_gym.rollouts.harness_capabilities import inspect_record
    from nemo_gym.rollouts.trajectory_builder import _attach_trajectory_record

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
    environment, _ = server(_Upstream(json.dumps(result).encode()))
    with TestClient(environment.setup_webserver()) as http:
        response = http.post("/run", json=_row("task-tools"))
    row = response.json()
    [_, added] = row["ng_agent_observations"]["records"]
    assert {key: added[key] for key in execution} == execution and added.get("output") is None
    # Each execution is added once, and collection joins its output from the conversation.
    assert _with_reply_tool_observations(response.content) == response.content
    _attach_trajectory_record({"_ng_task_index": 0, "_ng_rollout_index": 0}, row)
    assert [tool["output"] for tool in row["ng_trajectory"]["tool_calls"]] == ["found"]
    assert inspect_record(row)["evidence"]["TE-5"]["verdict"] == "fulfilled"


@pytest.mark.parametrize(
    "payload", [{}, [], {"verifier_metadata": {}}, {"verifier_metadata": {"partner_if": {"task_id": "unknown"}}}]
)
def test_missing_or_unknown_task_never_starts_an_episode(payload):
    environment, client = server()
    with TestClient(environment.setup_webserver()) as http:
        response = http.post("/run", json=payload)
    assert response.status_code == 422
    assert client.calls == []


@pytest.mark.parametrize("task", ["task-text", "task-tools"])
@pytest.mark.parametrize(
    "fields",
    [
        {"agent_ref": {"type": "responses_api_agents", "name": "untrusted"}},
        {"task_source": "untrusted"},
    ],
)
def test_request_routing_fields_cannot_change_the_route(task, fields):
    environment, client = server()
    with TestClient(environment.setup_webserver()) as http:
        response = http.post("/run", json=_row(task, **fields))
    assert response.status_code == 422
    assert client.calls == []


@pytest.mark.parametrize(
    "status,payload",
    [(503, b'{"detail":"backend unavailable"}'), (200, b'{"reward":0,"mask_sample":true,"failure_kind":"infra"}')],
)
def test_preserves_routed_failure_status_and_receipt_without_retry(status, payload):
    environment, client = server(_Upstream(payload, status=status))
    with TestClient(environment.setup_webserver()) as http:
        response = http.post("/run", json=_row("task-swe"))
    assert response.status_code == status and response.content == payload
    assert len(client.calls) == 1


async def test_cancellation_is_not_converted_into_an_if_failure(monkeypatch):
    environment, client = server()

    async def cancel(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(type(client), "post", cancel)
    raw = json.dumps(_row("task-tools")).encode()

    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    request = Request({"type": "http", "headers": []}, receive)
    with pytest.raises(asyncio.CancelledError):
        await environment.run_task(request)


def test_empty_task_ids_are_rejected_at_configuration_time():
    with pytest.raises(ValidationError, match="Task IDs"):
        _config(**{" ": "env-tools"})


@pytest.mark.parametrize("target", ["missing", "two_implementations", "no_agent", "null_agent", "typed", "partner_if"])
def test_routes_must_name_one_flat_row_environment_with_one_agent(target):
    config = _global_config()
    if target == "missing":
        del config["env-tools"]
    elif target == "two_implementations":
        config["env-tools"]["environment_servers"]["legacy_agent"] = {"agent_server": {"name": "other"}}
    elif target == "no_agent":
        config["env-tools"]["environment_servers"]["single_agent_turn_legacy"] = {}
    elif target == "null_agent":
        config["env-tools"]["environment_servers"]["single_agent_turn_legacy"] = {"agent_server": None}
    else:
        # Neither a typed single_agent_turn server nor another partner_if server takes the flat row as it is.
        implementation = "single_agent_turn" if target == "typed" else "partner_if"
        config["env-tools"]["environment_servers"] = {implementation: {"agent_server": {"name": "agent-tools"}}}
    with pytest.raises(ValueError, match="env-tools"):
        server(global_config=config)


def test_aggregate_metrics_are_computed_by_the_benchmark_resources_server_for_every_task():
    rows = [
        {"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 1.0, "if_result": {"passed": 2, "total": 2}},
        {"_ng_task_index": 1, "_ng_rollout_index": 0, "reward": 0.0, "if_result": {"passed": 1, "total": 2}},
    ]
    computed = {
        "group_level_metrics": [{"_ng_task_index": 0}, {"_ng_task_index": 1}],
        "agent_metrics": {"pass@1/all_if_pass": 50.0, "pass@1/if_partial": 75.0},
        "key_metrics": {"pass@1[avg-of-1]/all_if_pass": 50.0},
    }
    environment, client = server(sessions=[_Response(computed)])
    with TestClient(environment.setup_webserver()) as http:
        response = http.post("/aggregate_metrics", json={"verify_responses": rows})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["agent_metrics"] == computed["agent_metrics"]
    assert result["key_metrics"] == computed["key_metrics"]
    # The rows go unchanged to the scorer; no environment or agent is contacted.
    [(destination, path, kwargs)] = client.calls
    assert (destination, path) == ("resources", "/aggregate_metrics")
    assert kwargs["json"] == AggregateMetricsRequest(verify_responses=rows).model_dump(exclude_unset=True)


async def test_metrics_server_failure_is_raised_not_replaced_by_empty_metrics():
    class Unavailable(_Response):
        ok = False

        def __init__(self, body: dict, headers: dict | None = None) -> None:
            super().__init__(body, headers)
            self.content = SimpleNamespace(read=AsyncMock(return_value=self.body))

        def raise_for_status(self):
            raise RuntimeError("resources server returned 503")

    environment, client = server(sessions=[Unavailable({"detail": "metrics unavailable"})])
    with pytest.raises(RuntimeError, match="503"):
        await environment.aggregate_metrics(AggregateMetricsRequest(verify_responses=[{"reward": 1.0}]))
    assert [(name, path) for name, path, _ in client.calls] == [("resources", "/aggregate_metrics")]


def test_example_config_runs_each_partner_if_example_in_its_own_environment():
    from nemo_gym import PARENT_DIR
    from nemo_gym.client import ServerClient
    from nemo_gym.config import (
        GlobalConfigDictParser,
        GlobalConfigDictParserConfig,
        get_first_server_config_dict,
    )

    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {"config_paths": ["environment_servers/partner_if/configs/partner_if.yaml"]},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    # This server replaces the text example's own server, so the text agent has one environment server.
    assert config.get("partner_if_environment_server") is None
    block = get_first_server_config_dict(config, "partner_if_examples_environment_server")
    settings = OmegaConf.to_container(block, resolve=True) | {"name": "partner_if_examples_environment_server"}
    # Construction checks every route: one environment server that binds one execution agent.
    environment = PartnerIFEnvironmentServer(
        config=PartnerIFEnvironmentServerConfig.model_validate(settings),
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=11000), global_config_dict=config
        ),
    )
    assert environment._execution_agents == {
        "partner_if_mock_tools_environment_server": "partner_if_mock_tools_simple_agent",
    }
    # The example is partner_if's two examples, and each task goes to its own environment.
    examples = PARENT_DIR / "resources_servers/partner_if/data"
    agent = get_first_server_config_dict(config, "partner_if_simple_agent")
    rows = [json.loads(line) for line in (PARENT_DIR / agent.datasets[0].jsonl_fpath).read_text().splitlines()]
    assert rows == [
        json.loads(line)
        for name in ("example.jsonl", "mock_tools_example.jsonl")
        for line in (examples / name).read_text().splitlines()
    ]
    environments = {
        "text_response": "partner_if_examples_environment_server",
        "live_agent": "partner_if_mock_tools_environment_server",
    }
    contracts = [row["verifier_metadata"]["partner_if"] for row in rows]
    assert {task: route["name"] for task, route in settings["task_routes"].items()} == {
        contract["task_id"]: environments[contract.get("evidence_mode", "text_response")] for contract in contracts
    }
    assert settings["resources_server"]["name"] == "partner_if"


def recorded_row_and_reply():
    from benchmarks.partner_if.tests.fixture_data import TASK
    from resources_servers.partner_if.grading_contract import GradingContract
    from resources_servers.partner_if.receipt import masked_fields
    from resources_servers.partner_if.recorded_context_app import (
        recorded_context_result_sha256,
        recorded_context_task_data,
    )

    row = copy.deepcopy(TASK)
    row["verifier_metadata"]["partner_if"]["task_id"] = "task-recorded"
    task_data = recorded_context_task_data(row) | {"recorded_context_run_binding": {}}
    contract = GradingContract.model_validate(row["verifier_metadata"]["partner_if"])
    result = task_data | {
        "responses_create_params": row["responses_create_params"],
        "response": _agent_response().model_dump(mode="json"),
        **masked_fields(contract, "partner_if:recorded_context_fixture_gap", "synthetic gap"),
        "evaluation_completed": False,
        "recorded_context": {
            "task_data_sha256": hashlib.sha256(
                json.dumps(task_data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            ).hexdigest()
        },
    }
    result["recorded_context_result_sha256"] = recorded_context_result_sha256(result)
    return row, {
        "reward": 0,
        "mask_sample": True,
        "terminated": False,
        "truncated": True,
        "info": {"partner_if_recorded_context_result": result},
    }


def recorded_server(reply):
    config = _global_config()
    config["env-recorded"] = {
        "environment_servers": {"legacy_agent": {"agent_server": {"type": "responses_api_agents", "name": "recorder"}}}
    }
    config["recorder"] = {
        "responses_api_agents": {
            "gymnasium_agent": {"resources_server": {"type": "resources_servers", "name": "recorded-resources"}}
        }
    }
    config["recorded-resources"] = {"resources_servers": {"partner_if": {"entrypoint": "recorded_context_app.py"}}}
    client = _Client(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create(config),
        calls=[],
        responses=[],
        relays=[_Upstream(json.dumps(reply).encode())],
    )
    environment = PartnerIFEnvironmentServer(config=_config(**{"task-recorded": "env-recorded"}), server_client=client)
    assert environment._recorded_context_routes == {"env-recorded"}
    return environment, client


@pytest.mark.parametrize("own_selection", [False, True])
def test_recorded_context_normalization_keeps_masking_failure_evidence_and_cleanup_warning(own_selection):
    from nemo_gym.core import FailureBoundary, FailureReportRow, NgFailureEvent, failure_report_for

    row, reply = recorded_row_and_reply()
    report = failure_report_for(
        NgFailureEvent(
            boundary=FailureBoundary.ENVIRONMENT, failure_kind="environment_server_failed", failure_reason="cleanup"
        )
    )
    selected = None
    if own_selection:
        from resources_servers.partner_if.recorded_context_app import recorded_context_result_sha256

        selected = failure_report_for(
            NgFailureEvent(
                boundary=FailureBoundary.RESOURCES_SERVER, failure_kind="judge_failed", failure_reason="judge"
            )
        )
        resources_report = reply["info"]["partner_if_recorded_context_result"]
        resources_report.update(FailureReportRow.fields(selected))
        resources_report["recorded_context_result_sha256"] = recorded_context_result_sha256(resources_report)
    reply.update(FailureReportRow.fields(report))
    reply["info"]["cleanup_warning"] = {"operation": "close", "message": "unavailable"}
    environment, client = recorded_server(reply)
    with TestClient(environment.setup_webserver()) as http:
        response = http.post("/run", json=row)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["mask_sample"] is True and result["evaluation_completed"] is False
    assert result["reward"] == 0 and result["failure_kind"] == "partner_if:recorded_context_fixture_gap"
    assert result["recorded_context"]["cleanup_warning"]["operation"] == "close"
    combined = FailureReportRow.report(result)
    assert set(report.events) <= combined.events.keys()
    assert combined.failure_id == (selected.failure_id if selected else report.failure_id)
    if selected:
        assert set(selected.events) <= combined.events.keys()
    assert result["agent_ref"] == {"name": "recorder"}
    assert [(name, path) for name, path, _ in client.calls] == [("env-recorded", "/run")]


@pytest.mark.parametrize(
    "fault", ["checksum", "actor", "task", "grade", "completion", "terminal", "envelope", "model_only"]
)
def test_invalid_recorded_context_reports_cannot_become_flat_successes(fault):
    from resources_servers.partner_if.recorded_context_app import recorded_context_result_sha256

    row, reply = recorded_row_and_reply()
    result = reply["info"]["partner_if_recorded_context_result"]
    if fault == "actor":
        result["responses_create_params"] = {"input": "another actor"}
    elif fault == "task":
        result["task_name"] = "different task"
    elif fault == "grade":
        result.pop("if_result")
    elif fault == "completion":
        result["evaluation_completed"] = True
    elif fault == "terminal":
        reply.update(terminated=False, truncated=False)
    elif fault == "envelope":
        reply["reward"] = 1
    elif fault == "model_only":
        reply["response"] = {"partner_if_recorded_context_result": result}
        reply["info"] = {}
    # A matching checksum does not excuse an identity/grade mismatch.
    result["recorded_context_result_sha256"] = (
        recorded_context_result_sha256(result) if fault != "checksum" else "0" * 64
    )
    environment, client = recorded_server(reply)
    with TestClient(environment.setup_webserver()) as http:
        response = http.post("/run", json=row)
    assert response.status_code == 502
    assert response.json() == {"detail": "Invalid recorded-context Resources result"}
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "identity",
    [
        {"_ng_rollout_id": "new-capture"},
        {"_ng_rollout_id": "same-capture", "_ng_attempt_index": 1},
        {"task_index": 27, "rollout_index": 4},
    ],
)
def test_prior_recorded_grade_cannot_be_rebound_to_a_new_capture_or_attempt(identity):
    row, prior_reply = recorded_row_and_reply()
    environment, client = recorded_server(prior_reply)
    with TestClient(environment.setup_webserver()) as http:
        response = http.post("/run", json=row | identity)
    assert response.status_code == 502
    forwarded = json.loads(client.calls[0][2]["data"])
    assert forwarded["recorded_context_run_binding"] == identity
    assert forwarded["responses_create_params"] == row["responses_create_params"]
