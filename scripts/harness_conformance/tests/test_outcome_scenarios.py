# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scripted endings, harness termination and outcome expectations of the outcome-ladder scenarios."""

import ast
import asyncio
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.harness_conformance.episode import DEADLINE_KEYS, HARNESSES, _config
from scripts.harness_conformance.provider import CONTEXT_OVERFLOW_ERROR, Probe, harness_kill_command
from scripts.harness_conformance.registry import LACKS
from scripts.harness_conformance.runner import inspect_episode, run_suite
from scripts.harness_conformance.scenarios import SCENARIOS, scenarios_for

from nemo_gym.agents import AGENT_OUTCOME_METADATA_KEY
from nemo_gym.core import NeMoGymChatCompletion, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.models import VLLMConverter
from nemo_gym.models.token_capture import CONTEXT_LENGTH_EXCEEDED_ERROR_CODE
from nemo_gym.rollouts.harness_capabilities import json_rows


ROOT = Path(__file__).resolve().parents[3]
SCENARIO = {s.name: s for s in SCENARIOS}
TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
}
RESPONSES_TOOL = {"type": "function", **TOOL["function"]}


def _chat_tool_step(client: TestClient, messages: list[dict]) -> None:
    """Take the scripted shell call, run it and return its result, as a harness would."""
    message = client.post("/v1/chat/completions", json={"messages": messages, "tools": [TOOL]}).json()
    call = message["choices"][0]["message"]["tool_calls"][0]
    command = json.loads(call["function"]["arguments"])["command"]
    output = subprocess.run(command, shell=True, capture_output=True, text=True, check=True).stdout
    messages += [message["choices"][0]["message"], {"role": "tool", "tool_call_id": call["id"], "content": output}]


def _responses_tool_step(client: TestClient, items: list[dict]) -> None:
    output = client.post("/v1/responses", json={"input": items, "tools": [RESPONSES_TOOL]}).json()["output"]
    call = output[0]
    command = json.loads(call["arguments"])["command"]
    result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True).stdout
    items += [call, {"type": "function_call_output", "call_id": call["call_id"], "output": result}]


@pytest.mark.parametrize("dialect", ["chat", "responses"])
def test_context_overflow_refuses_every_request_after_the_tool_step(tmp_path, dialect):
    probe = Probe(SCENARIO["context_overflow"], tmp_path)
    history = [{"role": "user", "content": "Run checks"}]
    with TestClient(probe.model_app()) as client:
        if dialect == "chat":
            _chat_tool_step(client, history)
            path, body = "/v1/chat/completions", {"messages": history, "tools": [TOOL]}
        else:
            _responses_tool_step(client, history)
            path, body = "/v1/responses", {"input": history, "tools": [RESPONSES_TOOL]}
        for _ in range(2):  # A harness that compacts and retries meets the same refusal.
            response = client.post(path, json=body)
            assert response.status_code == 400
            assert response.json() == CONTEXT_OVERFLOW_ERROR
            assert "retry-after" not in response.headers
    assert CONTEXT_OVERFLOW_ERROR["error"]["code"] == CONTEXT_LENGTH_EXCEEDED_ERROR_CODE
    assert [a["status_code"] for a in probe.attempts] == [200, 400, 400]
    assert probe.tool_calls[0]["executed"] and probe.tool_calls[0]["result_seen"]
    assert not probe.violations and not probe.finished


def test_length_stop_is_vllm_models_empty_completion_in_both_dialects(tmp_path):
    probe = Probe(SCENARIO["context_overflow_length"], tmp_path)
    messages = [{"role": "user", "content": "Run checks"}]
    with TestClient(probe.model_app()) as client:
        _chat_tool_step(client, messages)
        chat = client.post("/v1/chat/completions", json={"model": "m", "messages": messages, "tools": [TOOL]}).json()
        responses = client.post("/v1/responses", json={"model": "m", "input": messages}).json()
    choice = chat["choices"][0]
    assert (choice["finish_reason"], choice["message"]["content"]) == ("length", None)
    assert "tool_calls" not in choice["message"]
    # vllm_model's Responses reply is its converter's output for that same chat completion.
    converted = VLLMConverter(return_token_id_information=False).chat_completion_to_response(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(model="m", input=[]),
        chat_completion=NeMoGymChatCompletion.model_validate(chat),
    )
    expected = converted.model_dump(mode="json")
    assert responses["status"] == expected["status"] == "incomplete"
    assert responses["incomplete_details"] == expected["incomplete_details"] == {"reason": "max_output_tokens"}
    assert responses["usage"] == expected["usage"]
    assert [part["text"] for item in responses["output"] for part in item["content"]] == [""]
    assert [part["text"] for item in expected["output"] for part in item["content"]] == [""]
    assert responses["metadata"] == {"nemo_gym_context_window_exceeded": "true"}
    assert not probe.violations and not probe.finished


async def test_a_held_request_ends_when_the_harness_disconnects(tmp_path):
    probe = Probe(SCENARIO["deadline_before_reply"], tmp_path)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(probe.model_app(), log_level="critical"))
        serving = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            while not server.started:
                await asyncio.sleep(0.01)
            async with aiohttp.ClientSession() as session:
                with pytest.raises(TimeoutError):
                    async with asyncio.timeout(0.5):
                        await session.post(
                            f"http://127.0.0.1:{port}/ng-rollout/0-0/v1/responses", json={"input": [], "tools": []}
                        )
            # The provider notices the closed connection and stops holding the request.
            async with asyncio.timeout(5):
                while not (tmp_path / "capture/0-0.capture.jsonl").is_file():
                    await asyncio.sleep(0.05)
        finally:
            server.should_exit = True
            await serving
    [attempt] = probe.attempts
    assert (attempt["status_code"], attempt["response"]) == (None, None)
    [(_, capture)] = list(json_rows(tmp_path / "capture/0-0.capture.jsonl"))
    assert (capture["status_code"], capture["error_category"]) == (None, "cancelled")


def test_kill_command_without_a_supervised_harness_records_its_call_and_kills_nothing(tmp_path):
    marker = tmp_path / "marker.txt"
    env = {key: value for key, value in os.environ.items() if key != "NEMO_GYM_SUPERVISOR"}
    result = subprocess.run(["sh", "-c", harness_kill_command("token", marker)], env=env, check=False, timeout=30)
    assert result.returncode == 97
    assert marker.read_text() == "token"


@pytest.mark.skipif(sys.platform != "linux", reason="the command reads /proc")
def test_kill_command_ends_the_supervised_harness_and_spares_the_supervisor(tmp_path):
    marker = tmp_path / "marker.txt"
    # As the process supervisor runs a harness: the supervisor is marked detached, the harness (its child) carries the
    # identity, starts a helper, and runs the tool call in a grandchild shell.
    harness = f"sleep 60 & sh -c {shlex.quote(harness_kill_command('token', marker))}; sleep 60"
    supervisor = subprocess.Popen(
        ["sh", "-c", f"NEMO_GYM_SUPERVISOR=probe.1 sh -c {shlex.quote(harness)} & echo $!; wait"],
        env={**os.environ, "NEMO_GYM_SUPERVISOR": "detached"},
        stdout=subprocess.PIPE,
        text=True,
    )
    harness_pid = int(supervisor.stdout.readline())
    supervisor.wait(timeout=30)
    # The supervisor was not killed: it saw its harness end and exited normally.
    assert supervisor.returncode == 0
    assert marker.read_text() == "token"
    assert not Path(f"/proc/{harness_pid}").exists()

    def marked() -> list[str]:
        found = []
        for environ in Path("/proc").glob("[0-9]*/environ"):
            try:
                if b"NEMO_GYM_SUPERVISOR=probe.1" in environ.read_bytes().split(b"\0"):
                    found.append(environ.parent.name)
            except OSError:
                continue
        return found

    # The helper and the tool shell end too, the tool shell last.
    deadline = time.monotonic() + 10
    while marked() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marked() == []


def test_a_request_after_the_killing_tool_call_is_a_protocol_violation(tmp_path):
    probe = Probe(SCENARIO["harness_no_result"], tmp_path)
    with TestClient(probe.model_app()) as client:
        message = client.post("/v1/chat/completions", json={"messages": [], "tools": [TOOL]}).json()
        command = json.loads(message["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["command"]
        # Unsupervised, the command records its call and exits 97 instead of ending a harness.
        subprocess.run(command, shell=True, check=False, timeout=30, env={"PATH": os.environ["PATH"]})
        probe.save()
        assert probe.tool_calls[0]["executed"] is True
        assert client.post("/v1/chat/completions", json={"messages": [], "tools": [TOOL]}).status_code == 409
    assert probe.violations == ["unexpected model request after the scripted tool call ended the harness"]


def test_agent_replies_are_witnessed_for_the_responses_route_only(tmp_path):
    app = FastAPI()

    @app.post("/ng-rollout/{rollout_id}/v1/responses")
    async def responses(rollout_id: str) -> dict:
        raise RuntimeError("unhandled")

    @app.post("/v1/agent_sessions/seed")
    async def seed() -> dict:
        return {}

    probe = Probe(SCENARIO["harness_no_result"], tmp_path)
    with TestClient(probe.observe_agent(app), raise_server_exceptions=False) as client:
        assert client.post("/v1/agent_sessions/seed").status_code == 200
        assert client.post("/ng-rollout/0-0/v1/responses").status_code == 500
    assert [reply["http_status"] for reply in probe.agent_replies] == [500]
    assert json.loads((tmp_path / "witness.json").read_text())["agent_replies"] == probe.agent_replies


def _config_fields(harness: str) -> dict[str, str]:
    """The annotated fields of an adapter's config class, read from its source without importing it."""
    fields = {}
    adapter = ROOT / f"responses_api_agents/{harness}_agent"
    for source in (adapter / "app.py", adapter / "config.py"):
        if not source.is_file():
            continue
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.ClassDef) and node.name == f"{HARNESSES[harness]}Config":
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        fields[item.target.id] = ast.unparse(item.annotation)
    return fields


@pytest.mark.parametrize("harness", HARNESSES)
def test_each_harness_has_a_run_deadline_key_or_skips_the_deadline_scenarios(harness):
    if harness not in DEADLINE_KEYS:
        assert "run_deadline" in LACKS[harness]
        return
    key = DEADLINE_KEYS[harness]
    annotation = _config_fields(harness)[key]
    assert annotation == ("int" if key == "timeout" else "float")


@pytest.mark.parametrize("harness", sorted(DEADLINE_KEYS))
@pytest.mark.parametrize("run_deadline", [None, 20])
def test_the_scenario_deadline_reaches_the_adapters_deadline_key(tmp_path, harness, run_deadline):
    ports = {"policy_model": 1, "probe_resources": 2, "probe_agent": 3, "probe_environment": 4}
    config = _config(harness, tmp_path, ports, 90.5, use_agent_sessions=True, run_deadline=run_deadline)
    agent = config["probe_agent"]["responses_api_agents"][f"{harness}_agent"]
    environment = config["probe_environment"]["environment_servers"]["single_agent_turn_legacy"]
    if run_deadline is not None:
        assert agent[DEADLINE_KEYS[harness]] == 20
    else:
        # The probe timeout, rounded up for an integer key.
        assert agent[DEADLINE_KEYS[harness]] == (91 if DEADLINE_KEYS[harness] == "timeout" else 90.5)
    assert environment["default_episode_timeout_seconds"] == 90.5


def test_harnesses_without_a_capability_skip_only_the_scenarios_that_need_it(tmp_path):
    skipped = {s.name for s in SCENARIOS} - {s.name for s in scenarios_for("simple")}
    assert skipped == {"deadline_before_reply", "deadline_after_reply", "harness_no_result"}
    assert scenarios_for("pi") == SCENARIOS
    selection = (SCENARIO["deadline_before_reply"],)
    with pytest.raises(ValueError, match="simple runs none of the selected scenarios"):
        run_suite(harnesses=["simple"], scenarios=selection, output=tmp_path / "run", timeout=10)


@pytest.mark.parametrize("scenario_name, raises", [("model_401", False), ("tool_success", True)])
async def test_an_ungraded_run_is_a_measurement_only_where_the_scenario_allows_it(
    tmp_path, monkeypatch, scenario_name, raises
):
    from types import SimpleNamespace

    from fastapi import Request
    from scripts.harness_conformance import episode

    import nemo_gym.config.global_config
    from nemo_gym.agents import (
        AgentCloseSessionResponse,
        AgentSessionState,
        BaseResponsesAPIAgentConfig,
        SimpleResponsesAPIAgent,
    )
    from nemo_gym.core import NeMoGymResponse
    from nemo_gym.rollouts import IncompleteEvaluationError
    from nemo_gym.trajectory import AgentObservationBundle

    class RejectedAgent(SimpleResponsesAPIAgent):
        """Reports the outcome ladder's response to a model that rejected its first request."""

        async def _seed_agent_session_state(self, body):
            return AgentSessionState(request=body)

        async def responses(self, request: Request, body: dict) -> NeMoGymResponse:
            return NeMoGymResponse(
                id="response",
                created_at=0,
                model="conformance-model",
                object="response",
                output=[],
                tool_choice="auto",
                parallel_tool_calls=True,
                tools=[],
                status="failed",
                error={"code": "server_error", "message": "HTTP 401: controlled model failure"},
                metadata={AGENT_OUTCOME_METADATA_KEY: json.dumps({"failure_kind": "model_request_rejected"})},
            )

        async def _close_agent_session_state(self, state):
            return AgentCloseSessionResponse(
                agent_session_id=state.request.agent_session_id,
                agent_observations=AgentObservationBundle(source="test"),
            )

    module = SimpleNamespace(TestAgent=RejectedAgent, TestAgentConfig=BaseResponsesAPIAgentConfig)
    monkeypatch.setitem(episode.HARNESSES, "codex", "TestAgent")
    monkeypatch.setattr(episode, "importlib", SimpleNamespace(import_module=lambda _: module))
    monkeypatch.setattr(episode, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(episode, "check_runtime", lambda *args, **kwargs: {})
    monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", None)
    monkeypatch.setattr(nemo_gym.config.global_config, "_GLOBAL_CONFIG_DICT", None)
    monkeypatch.setenv("NEMO_GYM_CONFIG_DICT", "{}")
    run = episode.run_episode(harness="codex", scenario_name=scenario_name, directory=tmp_path, timeout=10)
    if raises:
        with pytest.raises(IncompleteEvaluationError):
            await run
    else:
        await run
    # The Environment Server failed the episode (model_request_rejected is not retried) and kept a failure row.
    [(_, row)] = list(json_rows(tmp_path / "rollouts_failures.jsonl"))
    assert row["_ng_failure_terminal"] is True
    witness = json.loads((tmp_path / "witness.json").read_text())
    assert [reply["http_status"] for reply in witness["agent_replies"]] == [200]
    assert [response["status"] for response in witness["agent_responses"]] == ["failed"]


def _episode(directory: Path, *, response: dict, failure_row: dict | None, http_status: int = 200) -> None:
    witness = {
        "seeded": 1,
        "violations": [],
        "verifications": [],
        "attempts": [{"request": {"messages": []}, "response": {}, "status_code": 200, "received_at": 1.0}],
        "tool_calls": [],
        "agent_replies": [{"http_status": http_status, "replied_at": 2.0}],
        "agent_responses": [response] if http_status == 200 else [],
    }
    (directory / "witness.json").write_text(json.dumps(witness))
    (directory / "rollouts.jsonl").write_text("")
    if failure_row is not None:
        row = {"_ng_task_index": 0, "_ng_rollout_index": 0, "_ng_failure_class": "environment_server_failed"}
        (directory / "rollouts_failures.jsonl").write_text(json.dumps(row | failure_row) + "\n")


@pytest.mark.parametrize(
    "outcome, terminal, http_status, failed",
    [
        ({"failure_kind": "agent_run_error"}, False, 200, set()),
        # The flag the Environment Server stamped disagrees with the retried outcome.
        ({"failure_kind": "agent_run_error"}, True, 200, {"outcome.disposition"}),
        ({"failure_kind": "unknown"}, False, 200, {"outcome.ladder", "outcome.disposition"}),
        (None, False, 502, {"outcome.http_ok", "outcome.ladder", "outcome.disposition"}),
    ],
)
def test_no_result_episode_is_judged_from_its_failure_row(tmp_path, outcome, terminal, http_status, failed):
    response = {"status": "failed", "metadata": {AGENT_OUTCOME_METADATA_KEY: json.dumps(outcome)}}
    _episode(tmp_path, response=response, failure_row={"_ng_failure_terminal": terminal}, http_status=http_status)
    result = inspect_episode(SCENARIO["harness_no_result"], tmp_path, {"returncode": 0, "timed_out": False})
    checks = {c["id"]: c["status"] for c in result["checks"]}
    assert result["delivery"] == "failure_record"
    outcome_ids = ("outcome.http_ok", "outcome.ladder", "outcome.disposition")
    assert {key for key in outcome_ids if checks[key] == "fail"} == failed
    # A killed harness returns no tool result and no answer, and a retried run is never verified.
    for key in ("tools.executed", "model.finished", "verifier.outcome", "evaluation.reward"):
        assert checks[key] == "not_applicable", key
