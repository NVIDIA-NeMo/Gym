# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise wire protocols, independent witnesses, and fail-closed reporting."""

import copy
import dataclasses
import json
import os
import subprocess
import sys

import psutil
import pytest
from fastapi.testclient import TestClient
from scripts.harness_conformance.episode import HARNESSES, _config, _supports_agent_sessions
from scripts.harness_conformance.provider import SUMMARY_PROMPTS, Probe, is_compaction_request
from scripts.harness_conformance.runner import _fingerprint, inspect_episode, main, run_process, run_suite
from scripts.harness_conformance.scenarios import SCENARIOS, Scenario

from nemo_gym.agent_outcome import AGENT_OUTCOME_METADATA_KEY
from nemo_gym.harness_capabilities.cli import json_rows
from nemo_gym.observability.model_calls import build_model_call_record
from nemo_gym.server_utils import ServerClient
from tests.unit_tests.harness_capabilities.synthetic import evidence_and_witness


SCENARIO = {s.name: s for s in SCENARIOS}
# The fixture is tested independently of its later catalog/CI registration.
SCENARIO["context_compaction"] = Scenario("context_compaction", "Compact shell history", tier="P1", compaction=True)
TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
}

FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}}},
            "required": ["reason", "paths"],
        },
    },
}


def test_catalog_covers_p0_without_exposing_expectations():
    assert set().union(*(s.evidence for s in SCENARIOS)) == {f"TE-{i}" for i in range(1, 10)}
    for scenario in SCENARIOS:
        assert set(scenario.task()) == {"task_id", "responses_create_params"}
    assert main(["--list-scenarios"]) == 0


@pytest.mark.parametrize("use_agent_sessions", [False, True])
@pytest.mark.parametrize("harness", HARNESSES)
def test_launch_uses_local_endpoints_and_isolated_workspaces(harness, use_agent_sessions, tmp_path):
    config = _config(
        harness,
        tmp_path,
        {"policy_model": 10001, "probe_resources": 10002, "probe_agent": 10003, "probe_environment": 10004},
        12,
        use_agent_sessions=use_agent_sessions,
    )
    from omegaconf import OmegaConf

    from nemo_gym.rollout_collection import _environment_server_for_agent, _environment_servers_by_agent

    assert (
        _environment_server_for_agent("probe_agent", _environment_servers_by_agent(OmegaConf.create(config)))
        == "probe_environment"
    )
    agent = config["probe_agent"]["responses_api_agents"][f"{harness}_agent"]
    assert agent["model_server"] == {"type": "responses_api_models", "name": "policy_model"}
    assert config["model_call_capture_dir"] == str(tmp_path / "capture")
    environment_type = "single_agent_turn_legacy" if use_agent_sessions else "legacy_agent"
    assert set(config["probe_environment"]["environment_servers"]) == {environment_type}
    if use_agent_sessions:
        from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig

        SingleAgentTurnEnvironmentServerConfig.model_validate(
            config["probe_environment"]["environment_servers"][environment_type]
        )
        assert "resources_server" not in agent
        if harness == "simple":
            assert "sandbox_provider" not in agent
            assert "probe_sandbox" not in config
            assert config["probe_environment"]["environment_servers"][environment_type][
                "resources_tool_transports"
            ] == ["direct_http"]
        else:
            assert agent["sandbox_config"]["workdir"] == str(tmp_path / "workspace")
            assert config[agent["sandbox_provider"]] == {"local": {"concurrency": 1}}
        if harness == "codex":
            assert "cwd" not in agent and "extra_config" not in agent
        if harness == "opencode":
            assert "agent" not in agent["opencode_config"]
    # A CI job installs only its own adapter's dependencies, so only that adapter can be imported there.
    if os.environ.get("HARNESS") == harness:
        import importlib

        module = importlib.import_module(f"responses_api_agents.{harness}_agent.app")
        getattr(module, HARNESSES[harness] + "Config").model_validate(agent)


def test_session_detection_distinguishes_base_stubs_and_inherited_implementations():
    from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent

    class LegacyAgent(SimpleResponsesAPIAgent):
        pass

    class SeedOnlyAgent(LegacyAgent):
        async def _seed_agent_session_state(self, body):
            raise RuntimeError("setup failed")

    class SessionAgent(SeedOnlyAgent):
        async def _close_agent_session_state(self, state):
            raise RuntimeError("cleanup failed")

    class InheritedSessionAgent(SessionAgent):
        pass

    class PublicSessionAgent(LegacyAgent):
        async def seed_agent_session(self, body, request):
            raise RuntimeError("setup failed")

        async def close_agent_session(self, body, request):
            raise RuntimeError("cleanup failed")

    assert not _supports_agent_sessions(LegacyAgent)
    assert not _supports_agent_sessions(SeedOnlyAgent)
    assert _supports_agent_sessions(SessionAgent)
    assert _supports_agent_sessions(InheritedSessionAgent)
    assert _supports_agent_sessions(PublicSessionAgent)


def test_session_runtime_does_not_require_host_cli(monkeypatch):
    from scripts.harness_conformance import episode

    monkeypatch.setattr(episode.shutil, "which", lambda _: None)
    assert episode.check_runtime("pi", use_agent_sessions=True) == {"installation": "agent_session"}
    with pytest.raises(RuntimeError, match="install the pi runtime"):
        episode.check_runtime("pi")


@pytest.mark.parametrize(
    "interface", ["legacy", "session", "session_setup_error", "session_response_error", "session_close_error"]
)
async def test_episode_selects_interface_and_collects_through_environment(tmp_path, monkeypatch, interface):
    from types import SimpleNamespace

    from fastapi import HTTPException, Request
    from scripts.harness_conformance import episode

    import nemo_gym.global_config
    import nemo_gym.server_utils
    from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
    from nemo_gym.base_responses_api_agent import (
        AgentCloseSessionResponse,
        AgentSessionState,
        BaseResponsesAPIAgentConfig,
        SimpleResponsesAPIAgent,
    )
    from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.rollout_observability import AgentObservationBundle

    calls = []
    response = NeMoGymResponse(
        id="response",
        created_at=0,
        model="conformance-model",
        object="response",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "id": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": "CONFORMANCE_DONE", "annotations": []}],
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )

    class LegacyAgent(SimpleResponsesAPIAgent):
        async def responses(self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
            raise AssertionError("legacy responses should be called by the agent's run")

        async def run(self, request: Request, body: BaseRunRequest) -> BaseVerifyResponse:
            calls.append("run")
            return BaseVerifyResponse(**body.model_dump(), response=response, reward=1.0)

    class SessionAgent(LegacyAgent):
        async def _seed_agent_session_state(self, body):
            calls.append("seed")
            assert body.sandbox_access is None
            if interface == "session_setup_error":
                raise HTTPException(422, "controlled setup failure")
            return AgentSessionState(request=body)

        async def responses(self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
            state = self._require_agent_session(self._agent_session_id_from_request(request))
            assert state.request.episode_id.capture_key in request.url.path
            calls.append("responses")
            if interface == "session_response_error":
                raise HTTPException(422, "controlled activation failure")
            return response

        async def _close_agent_session_state(self, state):
            calls.append("close")
            if interface == "session_close_error":
                raise HTTPException(422, "controlled cleanup failure")
            return AgentCloseSessionResponse(
                agent_session_id=state.request.agent_session_id,
                agent_observations=AgentObservationBundle(source="test"),
            )

    module = SimpleNamespace(
        TestAgent=LegacyAgent if interface == "legacy" else SessionAgent, TestAgentConfig=BaseResponsesAPIAgentConfig
    )
    monkeypatch.setitem(episode.HARNESSES, "codex", "TestAgent")
    monkeypatch.setattr(episode, "importlib", SimpleNamespace(import_module=lambda _: module))
    monkeypatch.setattr(episode, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(episode, "check_runtime", lambda *args, **kwargs: {})
    monkeypatch.setattr(nemo_gym.server_utils, "_GLOBAL_AIOHTTP_CLIENT", None)
    monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)
    monkeypatch.setenv("NEMO_GYM_CONFIG_DICT", "{}")
    if interface.endswith("error"):
        with pytest.raises(RuntimeError, match="None of the 1 dispatched rollouts produced a result"):
            await episode.run_episode(harness="codex", scenario_name="tool_success", directory=tmp_path, timeout=10)
    else:
        await episode.run_episode(harness="codex", scenario_name="tool_success", directory=tmp_path, timeout=10)

    runtime = json.loads((tmp_path / "runtime.json").read_text())
    witness = json.loads((tmp_path / "witness.json").read_text())
    if interface == "legacy":
        assert calls == ["run"]
        assert runtime["agent_interface"] == "legacy_run"
    else:
        assert runtime["agent_interface"] == "agent_sessions"
        assert witness["seeded"] == 1 and witness["closed"] == 1
        if interface == "session_setup_error":
            assert calls == ["seed"]
        elif interface == "session_close_error":
            assert calls[:2] == ["seed", "responses"]
            assert calls[2:] and set(calls[2:]) == {"close"}
        else:
            assert calls == ["seed", "responses", "close"]
    if interface.endswith("error"):
        assert witness["verifications"] == []
        failures = list(json_rows(tmp_path / "rollouts_failures.jsonl"))
        assert len(failures) == 1
        assert failures[0][1]["_ng_failure_stage"] == ("cleanup" if interface == "session_close_error" else "agent")
        assert not list(json_rows(tmp_path / "rollouts.jsonl"))
    else:
        [(_, rollout)] = list(json_rows(tmp_path / "rollouts.jsonl"))
        assert rollout["reward"] == 1.0
        assert rollout["response"]["id"] == response.id
        if interface == "session":
            assert rollout["ng_agent_observations"]["source"] == "test"
            assert witness["verifications"] == [{"reward": 1.0, "answer_seen": True}]


@pytest.mark.parametrize("scenario_name", ["tool_success", "retry_429"])
def test_probe_retains_configured_assistant_message_identity(tmp_path, scenario_name):
    from omegaconf import OmegaConf

    config = _config(
        "opencode",
        tmp_path,
        {"policy_model": 10001, "probe_resources": 10002, "probe_agent": 10003, "probe_environment": 10004},
        12,
    )
    server_client = ServerClient(head_server_config=config["head_server"], global_config_dict=OmegaConf.create(config))
    header = server_client.assistant_message_header("policy_model")
    assert header is not None
    probe = Probe(SCENARIO[scenario_name], tmp_path)
    with TestClient(probe.model_app(assistant_message_header=header)) as client:
        response = client.post(
            "/ng-rollout/0-0/v1/chat/completions",
            json={"model": "conformance-model", "messages": [], "tools": [TOOL]},
            headers={"x-session-id": "session", header.decode(): "assistant-message"},
        )
    assert response.status_code == (429 if scenario_name == "retry_429" else 200)
    [(_, capture)] = list(json_rows(tmp_path / "capture/0-0.capture.jsonl"))
    call = build_model_call_record(capture, call_index=0)
    assert call.client_session_id == "session"
    assert call.client_assistant_message_id == "assistant-message"


@pytest.mark.parametrize("scenario_name", ["tool_success", "tool_failure", "usage_omitted", "retry_429", "retry_500"])
def test_live_chat_endpoint_executes_script_and_captures_every_attempt(tmp_path, scenario_name):
    scenario = SCENARIO[scenario_name]
    probe = Probe(scenario, tmp_path)
    messages = [{"role": "user", "content": "Run checks"}]
    body = {"model": "conformance-model", "messages": messages, "tools": [TOOL]}
    with TestClient(probe.model_app()) as client:
        for status in scenario.http_errors:
            response = client.post("/ng-rollout/0-0/v1/chat/completions", json=body)
            assert response.status_code == status
        for index in range(2):
            response = client.post("/ng-rollout/0-0/v1/chat/completions", json=body)
            assert response.status_code == 200
            message = response.json()["choices"][0]["message"]
            call = message["tool_calls"][0]
            execution = subprocess.run(
                json.loads(call["function"]["arguments"])["command"],
                shell=True,
                capture_output=True,
                text=True,
                check=False,
            )
            assert execution.returncode == (scenario.tool_exit_code if index == 0 else 0)
            messages.extend([message, {"role": "tool", "tool_call_id": call["id"], "content": execution.stdout}])
        response = client.post("/ng-rollout/0-0/v1/chat/completions", json=body)
        assert response.json()["choices"][0]["message"]["content"] == "CONFORMANCE_DONE"
    assert probe.finished and not probe.violations
    assert all(t["executed"] and t["result_seen"] for t in probe.tool_calls)
    assert all(t["outputs"] == [t["token"]] for t in probe.tool_calls)
    captures = [
        build_model_call_record(row, call_index=i).model_dump()
        for i, row in json_rows(tmp_path / "capture/0-0.capture.jsonl")
    ]
    assert len(captures) == len(scenario.http_errors) + 3
    assert len({c["model_call_id"] for c in captures}) == len(captures)
    assert [c["status_code"] for c in captures] == [*scenario.http_errors, 200, 200, 200]
    if not scenario.usage:
        assert all(c.get("tokens_in") is None and c.get("tokens_out") is None for c in captures)
    else:
        assert captures[-1]["tokens_in"] == 20 + len(scenario.http_errors) + 2
        assert captures[-1]["tokens_reasoning"] == 2
        assert captures[-1]["cached_tokens"] == 3


def test_a_harness_that_finishes_on_a_tool_gets_a_finish_call(tmp_path):
    probe = Probe(SCENARIO["tool_success"], tmp_path)
    code_exec = {
        "type": "function",
        "function": {
            "name": "code_exec",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
        },
    }
    messages = [{"role": "user", "content": "Run checks"}]
    body = {"model": "conformance-model", "messages": messages, "tools": [code_exec, FINISH_TOOL]}
    with TestClient(probe.model_app()) as client:
        for _ in range(2):
            message = client.post("/ng-rollout/0-0/v1/chat/completions", json=body).json()["choices"][0]["message"]
            call = message["tool_calls"][0]
            assert call["function"]["name"] == "code_exec"
            command = json.loads(call["function"]["arguments"])["cmd"]
            execution = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
            messages.extend([message, {"role": "tool", "tool_call_id": call["id"], "content": execution.stdout}])
        choice = client.post("/ng-rollout/0-0/v1/chat/completions", json=body).json()["choices"][0]
    (call,) = choice["message"]["tool_calls"]
    assert (choice["finish_reason"], choice["message"]["content"]) == ("tool_calls", None)
    assert call["function"]["name"] == "finish"
    assert json.loads(call["function"]["arguments"]) == {"reason": "CONFORMANCE_DONE", "paths": []}
    assert probe.finished and not probe.violations
    assert probe.finish_call["id"] == call["id"]


@pytest.mark.parametrize("compaction", [True, False])
def test_compaction_probe_keeps_summary_separate_from_policy_replies(tmp_path, compaction):
    scenario = dataclasses.replace(SCENARIO["context_compaction"], compaction=compaction, tool_steps=2)
    probe = Probe(scenario, tmp_path)
    messages = [{"role": "user", "content": "Run checks"}]
    body = {"model": "conformance-model", "messages": messages, "tools": [TOOL]}
    with TestClient(probe.model_app()) as client:
        for index in range(2):
            response = client.post("/ng-rollout/0-0/v1/chat/completions", json=body)
            assert response.status_code == 200
            message = response.json()["choices"][0]["message"]
            call = message["tool_calls"][0]
            execution = subprocess.run(
                json.loads(call["function"]["arguments"])["command"],
                shell=True,
                capture_output=True,
                text=True,
                check=True,
            )
            messages.extend([message, {"role": "tool", "tool_call_id": call["id"], "content": execution.stdout}])
            if compaction and index == 0:
                helper = client.post(
                    "/ng-rollout/0-0/v1/chat/completions",
                    json={**body, "messages": [{"role": "user", "content": SUMMARY_PROMPTS[0]}]},
                ).json()
                assert helper["choices"][0]["message"]["content"] == probe.compaction_summary
                assert helper["usage"]["total_tokens"] == 511
                assert not probe.finished and len(probe.tool_calls) == 1
                messages.append(helper["choices"][0]["message"])
        final = client.post("/ng-rollout/0-0/v1/chat/completions", json=body)
    assert final.json()["choices"][0]["message"]["content"] == "CONFORMANCE_DONE"
    assert probe.finished and not probe.violations
    assert all(t["executed"] and t["result_seen"] for t in probe.tool_calls)
    witness = json.loads((tmp_path / "witness.json").read_text())
    assert len(witness["attempts"]) == 3 + int(compaction)
    assert len(witness["compaction"]["response_ids"]) == int(compaction)
    assert sum(a["response"]["usage"]["total_tokens"] for a in witness["attempts"]) == (27738 if compaction else 81027)


@pytest.mark.parametrize("prompt", SUMMARY_PROMPTS)
@pytest.mark.parametrize("tools", [[], [TOOL]])
def test_summary_recognition_requires_an_explicit_last_user_instruction(prompt, tools):
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}], "tools": tools}
    assert is_compaction_request(body)
    body["messages"].append({"role": "tool", "content": prompt})
    assert not is_compaction_request(body)
    assert not is_compaction_request({"messages": [{"role": "user", "content": "Run checks"}], "tools": []})


def test_summary_does_not_waive_policy_tool_result_requirement(tmp_path):
    probe = Probe(SCENARIO["context_compaction"], tmp_path)
    body = {"model": "conformance-model", "messages": [{"role": "user", "content": "Run checks"}], "tools": [TOOL]}
    with TestClient(probe.model_app()) as client:
        assert client.post("/v1/chat/completions", json=body).status_code == 200
        helper = {**body, "messages": [{"role": "user", "content": SUMMARY_PROMPTS[0]}]}
        assert client.post("/v1/chat/completions", json=helper).status_code == 200
        assert client.post("/v1/chat/completions", json=body).status_code == 409
    assert len(probe.compaction_response_ids) == 1


def test_compaction_probe_rejects_unexpected_summary_request(tmp_path):
    probe = Probe(SCENARIO["context_compaction"], tmp_path)
    with TestClient(probe.model_app()) as client:
        response = client.post("/v1/chat/completions", json={"model": "conformance-model", "messages": []})
    assert response.status_code == 409
    assert probe.violations and not probe.compaction_response_ids


def test_a_finish_tool_with_unfillable_arguments_falls_back_to_a_text_answer(tmp_path):
    probe = Probe(dataclasses.replace(SCENARIO["tool_success"], tool_steps=0), tmp_path)
    finish = copy.deepcopy(FINISH_TOOL)
    finish["function"]["parameters"]["properties"]["paths"] = {"type": "integer"}
    body = {"model": "conformance-model", "messages": [], "tools": [finish]}
    with TestClient(probe.model_app()) as client:
        message = client.post("/v1/chat/completions", json=body).json()["choices"][0]["message"]
    assert message["content"] == "CONFORMANCE_DONE" and "tool_calls" not in message
    assert probe.finished and probe.finish_call is None


@pytest.mark.parametrize("dialect", ["chat/completions", "responses"])
def test_streaming_tool_roundtrip_and_unknown_usage(tmp_path, dialect):
    probe = Probe(SCENARIO["usage_omitted"], tmp_path)
    body = {"model": "conformance-model", "stream": True}
    if dialect == "responses":
        body.update(
            input=[{"role": "user", "content": "Run checks"}],
            tools=[{"type": "namespace", "name": "functions", "tools": [{"type": "function", **TOOL["function"]}]}],
        )
    else:
        body.update(messages=[{"role": "user", "content": "Run checks"}], tools=[TOOL])
    with TestClient(probe.model_app()) as client:
        response = client.post(f"/ng-rollout/0-0/v1/{dialect}", json=body)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
    if dialect == "responses":
        assert events[-1]["type"] == "response.completed"
        call = events[-1]["response"]["output"][0]
        assert call["namespace"] == "functions" and call["name"] == "bash"
    else:
        assert response.text.endswith("data: [DONE]\n\n")
        assert any(e["choices"] and e["choices"][0]["delta"].get("tool_calls") for e in events)
    (capture,) = [
        build_model_call_record(row, call_index=i).model_dump()
        for i, row in json_rows(tmp_path / "capture/0-0.capture.jsonl")
    ]
    assert capture["tokens_in"] is None
    assert capture["tokens_out"] is None


def test_tool_request_without_execution_is_not_exercised(tmp_path):
    probe = Probe(SCENARIO["tool_success"], tmp_path)
    body = {"model": "probe", "messages": [], "tools": [TOOL]}
    with TestClient(probe.model_app()) as client:
        assert client.post("/v1/chat/completions", json=body).status_code == 200
        assert client.post("/v1/chat/completions", json=body).status_code == 409
    assert probe.violations and not probe.finished


def test_terminal_failure_does_not_turn_into_a_success_on_retry(tmp_path):
    probe = Probe(SCENARIO["model_error"], tmp_path)
    with TestClient(probe.model_app()) as client:
        for _ in range(3):
            response = client.post("/v1/responses", json={"model": "probe", "input": []})
            assert response.status_code == 400
            assert "id" not in response.json()
    assert len(probe.attempts) == 3 and not probe.finished


@pytest.mark.parametrize("name,reward", [("tool_success", 1.0), ("verifier_failure", 0.0)])
def test_verifier_records_actual_final_answer(tmp_path, name, reward):
    probe = Probe(SCENARIO[name], tmp_path)
    body = {
        "response": {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "CONFORMANCE_DONE"}],
                }
            ]
        }
    }
    with TestClient(probe.resources_app()) as client:
        client.post("/seed_session", json={})
        response = client.post("/ng-rollout/0-0/verify", json=body)
    assert response.json()["reward"] == reward
    assert response.json()["evaluation_completed"] is True
    assert response.json()["mask_sample"] is False
    assert probe.verifications == [{"reward": reward, "answer_seen": True}]
    assert probe.seeded == 1


@pytest.mark.parametrize("name,exercised", [("model_error", True), ("tool_success", False)])
def test_only_a_terminal_error_episode_is_inspected_from_its_failure_record(tmp_path, name, exercised):
    (tmp_path / "rollouts.jsonl").write_text("")
    (tmp_path / "rollouts_failures.jsonl").write_text(
        json.dumps({"_ng_task_index": 0, "_ng_rollout_index": 0, "_ng_failure_class": "environment_server_failed"})
        + "\n"
    )

    result = inspect_episode(SCENARIO[name], tmp_path, {"returncode": 1, "timed_out": False})

    assert result["exercised"] is exercised
    assert result["delivery"] == "failure_record"


def test_verifier_accepts_the_answer_in_the_finish_call(tmp_path):
    probe = Probe(SCENARIO["tool_success"], tmp_path)
    probe.finish_call = {
        "id": "call-finish",
        "name": "finish",
        "arguments": {"reason": "CONFORMANCE_DONE", "paths": []},
    }
    arguments = json.dumps({"reason": "CONFORMANCE_DONE", "paths": []})
    body = {
        "response": {
            "output": [{"type": "function_call", "call_id": "call-finish", "name": "finish", "arguments": arguments}]
        }
    }
    with TestClient(probe.resources_app()) as client:
        client.post("/seed_session", json={})
        response = client.post("/ng-rollout/0-0/verify", json=body)
    assert response.json()["reward"] == 1.0
    assert probe.verifications == [{"reward": 1.0, "answer_seen": True}]


@pytest.fixture
def retained_episode(tmp_path):
    # Contract fixture only: used to prove a green artifact cannot hide a missing live probe.
    record, witness = evidence_and_witness()
    record["response"]["usage"] = {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
    calls = [
        {**c, "status_code": c["response_metadata"]["status_code"]} for c in record["ng_trajectory"]["model_calls"]
    ]
    (tmp_path / "rollouts.jsonl").write_text(json.dumps(record) + "\n")
    (tmp_path / "witness.json").write_text(json.dumps(witness))
    (tmp_path / "capture").mkdir()
    (tmp_path / "capture/0-0.capture.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    return tmp_path, witness


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "missing_response",
        "completed",
        "missing_error",
        "lost_detail",
        "missing_metadata",
        "missing_outcome",
        "malformed_outcome",
        "unknown_kind",
        "unregistered_kind",
        "wrong_status",
        "invalid_status",
    ],
)
def test_terminal_model_error_must_be_reported_to_environment(retained_episode, mutation):
    directory, witness = retained_episode
    witness["attempts"] = [
        {"status_code": 400, "request": {}, "response": {"error": {"message": "controlled model failure"}}}
    ]
    response = {"status": "failed", "error": {"code": "server_error", "message": "HTTP 400: controlled model failure"}}
    response["metadata"] = {
        AGENT_OUTCOME_METADATA_KEY: json.dumps({"failure_kind": "model_request_error", "status_code": 400})
    }
    witness["agent_responses"] = [response]
    if mutation == "missing_response":
        witness["agent_responses"] = []
    elif mutation == "completed":
        response["status"] = "completed"
    elif mutation == "missing_error":
        response["error"] = None
    elif mutation == "lost_detail":
        response["error"]["message"] = "unknown failure"
    elif mutation == "missing_metadata":
        del response["metadata"]
    elif mutation == "missing_outcome":
        response["metadata"] = {}
    elif mutation is not None:
        response["metadata"][AGENT_OUTCOME_METADATA_KEY] = {
            "malformed_outcome": "{",
            "unknown_kind": json.dumps({"failure_kind": "unknown"}),
            "unregistered_kind": json.dumps({"failure_kind": "invented"}),
            "wrong_status": json.dumps({"failure_kind": "model_request_error", "status_code": 503}),
            "invalid_status": json.dumps({"failure_kind": "model_request_error", "status_code": "400"}),
        }[mutation]
    (directory / "witness.json").write_text(json.dumps(witness))
    result = inspect_episode(SCENARIO["model_error"], directory, {"returncode": 0, "timed_out": False})
    check = next(c for c in result["checks"] if c["id"] == "model.terminal_error")
    assert check["status"] == ("pass" if mutation is None else "fail")


@pytest.mark.parametrize("graded,masked,reward", [(True, False, 0.0), (True, False, 1.0), (False, True, 0.0)])
def test_terminal_error_does_not_prescribe_grading_policy(retained_episode, graded, masked, reward):
    directory, witness = retained_episode
    bundle = directory / "rollouts.jsonl"
    record = json.loads(bundle.read_text())
    record.update(evaluation_completed=graded, mask_sample=masked, reward=reward)
    witness["verifications"] = [{"reward": reward, "answer_seen": False}] if graded else []
    bundle.write_text(json.dumps(record) + "\n")
    (directory / "witness.json").write_text(json.dumps(witness))
    result = inspect_episode(SCENARIO["model_error"], directory, {"returncode": 0, "timed_out": False})
    checks = {c["id"]: c for c in result["checks"]}
    for check_id in (
        "verifier.answer",
        "verifier.outcome",
        "verifier.saved_reward",
        "verifier.saved_outcome",
        "evaluation.evaluation_completed",
    ):
        assert checks[check_id]["status"] == "not_applicable"


def test_behavior_failure_does_not_erase_artifact_success(retained_episode):
    directory, witness = retained_episode
    result = inspect_episode(SCENARIO["verifier_failure"], directory, {"returncode": 0, "timed_out": False})
    assert result["verdict"] == "fulfilled", result["issues"]
    witness["attempts"].append(copy.deepcopy(witness["attempts"][0]))
    (directory / "witness.json").write_text(json.dumps(witness))
    result = inspect_episode(SCENARIO["verifier_failure"], directory, {"returncode": 0, "timed_out": False})
    assert result["verdict"] == "not_fulfilled"
    assert result["behavioral_status"] == "fail"
    assert all(v["verdict"] == "fulfilled" for v in result["evidence"].values())
    assert "canonical" in " ".join(result["issues"])


def test_execution_status_does_not_erase_retained_results(retained_episode):
    directory, _ = retained_episode
    result = inspect_episode(SCENARIO["verifier_failure"], directory, {"returncode": 1, "timed_out": True})
    assert result["verdict"] == "fulfilled" and result["behavioral_status"] == "pass"
    assert result["execution"] == {"returncode": 1, "timed_out": True}
    assert not any(c["id"].startswith("execution") for c in result["checks"])


def test_health_failure_blocks_p0_without_erasing_artifact_or_behavior_passes(retained_episode):
    directory, _ = retained_episode
    bundle = directory / "rollouts.jsonl"
    record = json.loads(bundle.read_text())
    record["response"]["usage"]["input_tokens"] = 999
    bundle.write_text(json.dumps(record) + "\n")
    result = inspect_episode(SCENARIO["verifier_failure"], directory, {"returncode": 0, "timed_out": False})
    assert result["verdict"] == "not_fulfilled" and result["behavioral_status"] == "pass"
    assert all(row["verdict"] == "fulfilled" for row in result["evidence"].values())
    failed = [c["id"] for c in result["checks"] if c["status"] == "fail"]
    assert failed == ["health.rollout_token_count_mismatch"]
    assert (directory / result["health_report"]).is_file()


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "extra", "arguments", "name", "output", "status", "owner"]
)
def test_tool_witness_rejects_lost_or_changed_evidence(retained_episode, mutation):
    directory, _ = retained_episode
    bundle = directory / "rollouts.jsonl"
    record = json.loads(bundle.read_text())
    trajectory = record["ng_trajectory"]
    tools = trajectory["tool_calls"]
    invocations = [trajectory["invocations"][0], record["ng_agent_observations"]["records"][0]]
    if mutation == "missing":
        tools.pop()
        record["ng_agent_observations"]["records"].pop()
        for invocation in invocations:
            invocation["conversation"] = [i for i in invocation["conversation"] if i.get("call_id") != "tool-2"]
    elif mutation in {"duplicate", "extra"}:
        tool = copy.deepcopy(tools[0])
        if mutation == "extra":
            tool["tool_call_id"] = "unwitnessed-tool"
        tools.append(tool)
    elif mutation in {"arguments", "name"}:
        for invocation in invocations:
            invocation["conversation"][1][mutation] = '{"key":"changed"}' if mutation == "arguments" else "other_tool"
        if mutation == "name":
            tools[0]["tool_name"] = "other_tool"
    elif mutation == "output":
        tools[0]["output"] = "changed"
        for invocation in invocations:
            invocation["conversation"][2]["output"] = "changed"
    elif mutation == "status":
        tools[0]["status"] = "failed"
    else:
        tools[0]["invocation_id"] = "other-invocation"
    bundle.write_text(json.dumps(record) + "\n")
    result = inspect_episode(SCENARIO["verifier_failure"], directory, {"returncode": 0, "timed_out": False})
    assert result["verdict"] == "not_fulfilled" and result["behavioral_status"] == "fail"
    assert any("retained tool" in issue for issue in result["issues"])
    assert not any("retained model attempts differ" in issue for issue in result["issues"])
    if mutation in {"missing", "arguments", "name", "output", "status"}:
        # These artifacts remain internally consistent; the independent witness
        # must expose the lost/changed evidence even when TE-5 alone passes.
        assert result["evidence"]["TE-5"]["artifact_verdict"] == "fulfilled"


@pytest.mark.parametrize("surface", ["trajectory", "observations"])
def test_tool_witness_uses_only_designated_surface(retained_episode, surface):
    directory, _ = retained_episode
    bundle = directory / "rollouts.jsonl"
    record = json.loads(bundle.read_text())
    if surface == "trajectory":
        del record["ng_agent_observations"]
    else:
        del record["ng_trajectory"]["tool_calls"]
        del record["ng_trajectory"]["invocations"]
    bundle.write_text(json.dumps(record) + "\n")
    result = inspect_episode(SCENARIO["verifier_failure"], directory, {"returncode": 0, "timed_out": False})
    assert bool(result["issues"]) == (surface == "observations")
    assert result["verdict"] == ("fulfilled" if surface == "trajectory" else "not_fulfilled")


@pytest.mark.parametrize("failed", [True, False])
def test_tool_witness_requires_prescribed_failure_status(retained_episode, failed):
    directory, witness = retained_episode
    witness["tool_calls"][0]["exit_code"] = 7
    (directory / "witness.json").write_text(json.dumps(witness))
    bundle = directory / "rollouts.jsonl"
    record = json.loads(bundle.read_text())
    if failed:
        record["ng_trajectory"]["tool_calls"][0]["status"] = "failed"
    bundle.write_text(json.dumps(record) + "\n")
    result = inspect_episode(SCENARIO["verifier_failure"], directory, {"returncode": 0, "timed_out": False})
    assert (result["behavioral_status"] == "pass") == failed
    assert ("retained tool status differs from the independent tool witness" in result["issues"]) != failed


def test_missing_runtime_is_recorded_and_no_stale_output_reused(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    output = tmp_path / "run"
    summary, code = run_suite(harnesses=["pi", "opencode"], scenarios=(SCENARIOS[0],), output=output, timeout=1)
    assert code == 2 and not summary["full_suite"]
    for row in summary["harnesses"].values():
        assert row["evidence"]["TE-1"] == {"required": 1, "observed": 0, "passed": 0}
    assert (output / "conformance_summary.json").is_file()
    before = (output / "suite.json").read_bytes()
    manifest = json.loads(before)
    assert {"behavior.py", "checker.py", "checks.py"} <= manifest["checker_sources"].keys()
    assert "behavior.py" not in manifest["runner_sources"]
    assert main(["--harness", "pi", "--output", str(output)]) == 2
    assert (output / "suite.json").read_bytes() == before


def test_timeout_reaps_descendant_in_another_process_group(tmp_path):
    child_code = "import time; time.sleep(300)"
    worker = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}],start_new_session=True); "
        f"Path({str(tmp_path / 'pid')!r}).write_text(str(p.pid)); time.sleep(300)"
    )
    result = run_process([sys.executable, "-c", worker], directory=tmp_path, timeout=1)
    assert result["timed_out"]
    pid = int((tmp_path / "pid").read_text())
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_invalid_timeout_rejected_before_launch(tmp_path, timeout):
    with pytest.raises(SystemExit) as exc:
        main(["--output", str(tmp_path / "absent"), "--timeout", timeout])
    assert exc.value.code == 2
    assert not (tmp_path / "absent").exists()


def test_witness_detects_changed_payload_even_when_attempt_identity_matches(retained_episode):
    directory, witness = retained_episode
    witness["attempts"][0]["response"]["usage"]["total_tokens"] += 1
    (directory / "witness.json").write_text(json.dumps(witness))
    result = inspect_episode(SCENARIO["verifier_failure"], directory, {"returncode": 0, "timed_out": False})
    assert result["verdict"] == "not_fulfilled"
    assert "canonical" in " ".join(result["issues"])


def test_checker_error_does_not_prevent_remaining_episodes(tmp_path, monkeypatch):
    from scripts.harness_conformance import runner

    inspected = []
    monkeypatch.setattr(runner, "run_process", lambda *args, **kwargs: {"returncode": 0, "timed_out": False})

    def fail_inspection(scenario, directory, execution):
        inspected.append(scenario.name)
        raise ValueError("truncated rollout")

    monkeypatch.setattr(runner, "inspect_episode", fail_inspection)
    summary, code = run_suite(harnesses=["pi"], scenarios=SCENARIOS[:2], output=tmp_path / "run", timeout=1)
    assert code == 2
    assert inspected == ["tool_success", "tool_failure"]
    assert summary["harnesses"]["pi"]["evidence"]["TE-1"] == {"required": 2, "observed": 0, "passed": 0}


@pytest.mark.parametrize(
    "failed_scenario,gate_tier,expected_code",
    [("context_compaction", "P0", 0), ("context_compaction", "P1", 2), ("tool_success", "P0", 2)],
)
def test_checker_error_preserves_independent_ci_measurements(
    tmp_path, monkeypatch, failed_scenario, gate_tier, expected_code
):
    from scripts.harness_conformance import ci_table, runner

    scenarios = tuple(s for s in SCENARIOS if s.name != "context_compaction") + (SCENARIO["context_compaction"],)
    monkeypatch.setattr(runner, "SCENARIOS", scenarios)
    monkeypatch.setattr(ci_table, "SCENARIOS", scenarios)
    monkeypatch.setattr(runner, "run_process", lambda *args, **kwargs: {"returncode": 0, "timed_out": False})

    def inspect(scenario, directory, execution):
        if scenario.name == failed_scenario:
            raise ValueError("truncated rollout")
        return {
            "scenario": scenario.name,
            "tier": scenario.tier,
            "exercised": True,
            "verdict": "fulfilled",
            "checks": [],
            "evidence": {},
        }

    monkeypatch.setattr(runner, "inspect_episode", inspect)
    artifacts = tmp_path / "artifacts"
    probes = artifacts / "pi" / "probes"
    summary, code = run_suite(harnesses=["pi"], scenarios=scenarios, output=probes, timeout=1, gate_tier=gate_tier)
    assert code == expected_code
    result = next(s for s in summary["harnesses"]["pi"]["scenarios"] if s["scenario"] == failed_scenario)
    assert result["checks"] == [] and result["checker_error"] == "ValueError"
    commit = "a" * 40
    (probes / "ci-result.json").write_text(
        json.dumps({"harness": "pi", "sha": commit, "status": "passed" if code == 0 else "unavailable"})
    )
    output = tmp_path / "table.mdx"
    ci_table.render_ci(artifacts=artifacts, commit=commit, output=output, run_url="https://example.com/run")
    expected = f"Passed | {len(scenarios) - 1}/{len(scenarios)} | Failed" if code == 0 else "Unavailable | —"
    assert f"`pi` | {expected}" in output.read_text()


def test_interrupted_run_does_not_publish_completed_summary(tmp_path, monkeypatch):
    from scripts.harness_conformance import runner

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "run_process", interrupt)
    output = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        run_suite(harnesses=["pi"], scenarios=SCENARIOS[:1], output=output, timeout=1)
    assert not (output / "conformance_summary.json").exists()


@pytest.mark.parametrize("gate_tier", ["P0", "P1"])
def test_compaction_runs_for_every_harness_and_reports_its_own_priority(retained_episode, monkeypatch, gate_tier):
    from scripts.harness_conformance import runner

    source, _ = retained_episode
    files = {
        name: (source / name).read_bytes() for name in ("rollouts.jsonl", "witness.json", "capture/0-0.capture.jsonl")
    }
    launched = []

    def run(command, *, directory, timeout):
        launched.append((directory.parent.name, directory.name))
        for name, content in files.items():
            destination = directory / name
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(content)
        return {"returncode": 0, "timed_out": False}

    monkeypatch.setattr(runner, "run_process", run)
    # Valid existing evidence deliberately lacks a compaction event/witness.
    scenarios = (
        SCENARIO["verifier_failure"],
        dataclasses.replace(SCENARIO["context_compaction"], expected_reward=0.0, tool_steps=2),
    )
    summary, code = run_suite(
        harnesses=list(HARNESSES), scenarios=scenarios, output=source / "run", timeout=1, gate_tier=gate_tier
    )
    assert set(launched) == {(h, s.name) for h in HARNESSES for s in scenarios}
    assert code == (0 if gate_tier == "P0" else 1)
    for row in summary["harnesses"].values():
        assert row["tier_verdicts"] == {"P0": "fulfilled", "P1": "not_fulfilled"}
        result = row["scenarios"][1]
        assert result["p0_verdict"] == "fulfilled" and result["verdict"] == "not_fulfilled"
        assert next(c for c in result["checks"] if c["id"] == "compaction.event")["status"] == "fail"


def test_compaction_only_run_cannot_pass_an_empty_p0_gate(tmp_path):
    with pytest.raises(ValueError, match="--gate-tier P1"):
        run_suite(
            harnesses=["opencode"], scenarios=(SCENARIO["context_compaction"],), output=tmp_path / "run", timeout=1
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "collection,field,value",
    [("turns", "resolved", "wrong"), ("model_calls", "response_metadata", False), ("tool_calls", "tool_call_id", [])],
)
def test_malformed_evidence_keeps_other_check_results(retained_episode, collection, field, value):
    directory, _ = retained_episode
    path = directory / "rollouts.jsonl"
    record = json.loads(path.read_text())
    record["ng_trajectory"][collection][0][field] = value
    path.write_text(json.dumps(record) + "\n")
    result = inspect_episode(SCENARIO["verifier_failure"], directory, {"returncode": 0, "timed_out": False})
    checks = {c["id"]: c for c in result["checks"]}
    assert any(c["status"] == "fail" for c in checks.values())
    assert checks["tokens.prompt_tokens"]["status"] == "pass"


def test_missing_witness_does_not_fail_retained_evidence(retained_episode):
    directory, _ = retained_episode
    (directory / "witness.json").unlink()
    result = inspect_episode(SCENARIO["verifier_failure"], directory, {"returncode": 0, "timed_out": False})
    assert all(c["status"] in {"fail", "not_applicable"} for c in result["checks"] if c["kind"] == "behavioral")
    assert all(v["verdict"] == "fulfilled" for v in result["evidence"].values())


@pytest.mark.parametrize("tier,verdict", [("P0", "not_fulfilled"), ("P1", "fulfilled"), ("P2", "fulfilled")])
def test_scenario_gate_uses_individual_priority(retained_episode, monkeypatch, tier, verdict):
    from dataclasses import replace

    from scripts.harness_conformance import runner

    directory, _ = retained_episode
    original = runner.inspect_behavior

    def with_failing_check(*args, **kwargs):
        checks = original(*args, **kwargs)
        # No TE label: this rule's priority alone determines whether it blocks P0.
        checks.append(
            {
                "id": "independent.test",
                "kind": "behavioral",
                "tier": tier,
                "status": "fail",
                "evidence": (),
                "locations": [],
                "reasons": ["mismatch"],
                "blocked_by": [],
            }
        )
        return checks

    monkeypatch.setattr(runner, "inspect_behavior", with_failing_check)
    scenario = replace(SCENARIO["verifier_failure"], evidence=())
    result = inspect_episode(scenario, directory, {"returncode": 0, "timed_out": False})
    assert result["verdict"] == verdict
    assert result["behavioral_status"] == "fail"
    assert result["evidence"] == {}
    assert all(c["status"] in {"pass", "not_applicable"} for c in result["checks"] if c["id"] != "independent.test")


def test_step_attribution_failure_blocks_scenario_p0(retained_episode):
    directory, _ = retained_episode
    bundle = directory / "rollouts.jsonl"
    record = json.loads(bundle.read_text())
    for turn in record["ng_trajectory"]["turns"]:
        turn["model_calls"] = []
    bundle.write_text(json.dumps(record) + "\n")
    result = inspect_episode(SCENARIO["verifier_failure"], directory, {"returncode": 0, "timed_out": False})
    assert result["evidence"]["TE-8"]["verdict"] == "fulfilled"
    assert result["evidence"]["TE-9"]["verdict"] == "not_fulfilled"
    assert result["verdict"] == "not_fulfilled"


@pytest.mark.parametrize("delivery", ["missing", "failure_record"])
def test_missing_rollout_fails_required_checks_and_preserves_exclusions(tmp_path, delivery):
    if delivery == "failure_record":
        (tmp_path / "rollouts_failures.jsonl").write_text('{"failure": "model error"}\n')
    result = inspect_episode(SCENARIO["model_error"], tmp_path, {"returncode": 1, "timed_out": False})
    checks = {c["id"]: c for c in result["checks"]}
    assert checks["calls.outcome"]["status"] == "fail"
    # A failure record is inspected, so the evidence it lacks blocks the check rather than its absence.
    reason = "required input is unavailable" if delivery == "missing" else "blocked by model_calls.present"
    assert checks["calls.outcome"]["reasons"][0].startswith(reason)
    assert checks["tool_calls.present"]["status"] == "not_applicable"
    assert checks["evaluation.reward"]["status"] == "not_applicable"
    assert checks["model.terminal_error"]["status"] == "fail"
    assert checks["turns.present"]["status"] == "not_applicable"
    assert checks["steps.attempt_accounting"]["status"] == "not_applicable"
    assert result["evidence"]["TE-1"]["verdict"] == "not_fulfilled"
    assert result["behavioral_status"] == "fail"
    assert result["delivery"] == delivery
    assert result["execution"]["returncode"] == 1


@pytest.mark.parametrize("scenario_name", ["model_error", "retry_429", "retry_500"])
def test_step_scope_is_declared_by_scenario_not_missing_turns(retained_episode, scenario_name):
    directory, _ = retained_episode
    bundle = directory / "rollouts.jsonl"
    record = json.loads(bundle.read_text())
    record["ng_trajectory"]["turns"] = []
    bundle.write_text(json.dumps(record) + "\n")
    result = inspect_episode(SCENARIO[scenario_name], directory, {"returncode": 0, "timed_out": False})
    checks = {c["id"]: c for c in result["checks"]}
    expected = "not_applicable" if scenario_name == "model_error" else "fail"
    for check_id in ("turns.present", "steps.number", "steps.call_owner", "steps.attempt_accounting"):
        assert checks[check_id]["status"] == expected
    for check_id in ("calls.identity", "calls.timing", "calls.outcome", "ownership.call_owner"):
        assert checks[check_id]["status"] != "not_applicable"


def test_chat_fingerprint_ignores_creation_clock():
    request = {"messages": [{"role": "user", "content": "hello"}]}
    response = {"id": "r", "object": "chat.completion", "created": 123, "choices": []}
    without_clock = {key: value for key, value in response.items() if key != "created"}
    assert _fingerprint(request, 200, response) == _fingerprint(request, 200, without_clock)


def test_missing_judge_reference_fails_p1_without_blocking_p0(retained_episode):
    from dataclasses import replace

    directory, _ = retained_episode
    scenario = replace(SCENARIO["verifier_failure"], judge_verifier=True)
    result = inspect_episode(scenario, directory, {"returncode": 0, "timed_out": False})
    check = next(c for c in result["checks"] if c["id"] == "verifier.judge_reference")
    assert check["tier"] == "P1"
    assert check["status"] == "fail"
    assert result["behavioral_status"] == "fail"
    assert result["verdict"] == "fulfilled"
    assert all(c["status"] in {"pass", "not_applicable"} for c in result["checks"] if c["tier"] == "P0")
