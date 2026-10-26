# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from nemo_gym.agent_outcome import AGENT_OUTCOME_METADATA_KEY
from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from nemo_gym.testing.agent_session_conformance import check_agent_seed_rejected
from responses_api_agents.copilot_agent.app import (
    CopilotAgent,
    CopilotAgentConfig,
    parse_copilot_events,
    parse_copilot_usage,
)


RUNTIME = {
    "binary": "/tmp/nemo-gym-copilot-1.0.103/copilot",
    "dist_dir": "/tmp/nemo-gym-copilot-1.0.103/home/.cache/copilot/pkg/linux-x64/1.0.103",
    "package": "copilot-linux-x64",
    "version": "1.0.103",
    "integrity": "sha512-test",
}


def seed(*, session_id: str = "copilot-session", tool_accesses=None, sandbox_access=True) -> AgentSeedSessionRequest:
    body = {
        "agent_session_id": session_id,
        "episode_id": EpisodeId(rollout_id="copilot-smoke", attempt=2).model_dump(),
        "task_id": TaskId(taskset="terminal_bench_2_1", task_id="task").model_dump(),
        "sandbox_access": (
            {
                "connection": {
                    "kind": "direct",
                    "provider_config_ref": "sandbox",
                    "descriptor": {"sandbox_id": "resources-owned"},
                },
                "workdir": "/app",
            }
            if sandbox_access
            else None
        ),
        "tool_accesses": tool_accesses or [],
    }
    return AgentSeedSessionRequest.model_validate(body)


def copilot_events(*, error=None, result=True) -> list[dict]:
    """Event shapes recorded from Copilot 1.0.103 against a scripted endpoint."""
    events = [
        {
            "type": "assistant.message",
            "data": {
                "content": "",
                "toolRequests": [{"toolCallId": "call-1", "name": "bash", "arguments": {"command": "pwd"}}],
                "apiCallId": "chatcmpl-1",
            },
            "timestamp": "2026-10-23T03:13:02.230Z",
        },
        {"type": "assistant.reasoning", "data": {"content": "Inspect the repository"}},
        {
            "type": "tool.execution_start",
            "data": {"toolCallId": "call-1", "toolName": "bash", "arguments": {"command": "pwd"}},
            "timestamp": "2026-10-23T03:13:02.232Z",
        },
        {
            "type": "tool.execution_complete",
            "data": {"toolCallId": "call-1", "success": True, "result": {"content": "/app\n<exit code 0>"}},
            "timestamp": "2026-10-23T03:13:02.291Z",
        },
    ]
    if error is not None:
        events.append({"type": "session.error", "data": error})
    else:
        events.append(
            {"type": "assistant.message", "data": {"content": "Fixed", "toolRequests": [], "apiCallId": "chatcmpl-2"}}
        )
    if result:
        events.append({"type": "result", "sessionId": "copilot-run-1", "exitCode": 1 if error else 0})
    return events


def chat_span(response_id, *, usage=(10, 3), status=0):
    attributes = {"gen_ai.operation.name": "chat", "gen_ai.response.id": response_id}
    if usage is not None:
        attributes |= {"gen_ai.usage.input_tokens": usage[0], "gen_ai.usage.output_tokens": usage[1]}
    return json.dumps({"type": "span", "name": "chat m", "attributes": attributes, "status": {"code": status}})


def otel(*spans) -> str:
    tool = json.dumps({"type": "span", "attributes": {"gen_ai.operation.name": "execute_tool"}, "status": {"code": 0}})
    return "\n".join([tool, *spans]) + "\n"


class Sandbox:
    """Provider double: records uploads, answers the installer, and plays one supervised run."""

    def __init__(self):
        self.files = {}
        self.result = {"return_code": 0, "timed_out": False, "cleanup_confirmed": True, "error": None}
        self.events = copilot_events()
        self.otel = otel(chat_span("chatcmpl-1", usage=(10, 3)), chat_span("chatcmpl-2", usage=(5, 2)))
        self.exec = AsyncMock(side_effect=self.execute)
        self.stop = AsyncMock()
        self.disconnect = AsyncMock()
        self.launch = AsyncMock(side_effect=self.run)
        self.payload = None

    async def execute(self, command, **kwargs):
        if "--receipt" in command and "process_supervisor.py" in command:
            return await self.launch(command, **kwargs)
        if "install_copilot_runtime.py" in command:
            return SimpleNamespace(return_code=0, error_type=None, stdout=json.dumps(RUNTIME) + "\n", stderr="")
        if "ln -s stop " in command:
            self.result["timed_out"] = True
        return SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")

    async def upload(self, source, destination):
        self.files[destination] = Path(source).read_text()

    async def download(self, source, destination):
        Path(destination).write_text(self.files[source])

    async def run(self, command, **kwargs):
        payload_path = next(path for path in self.files if path.endswith("/input.json"))
        self.payload = json.loads(self.files[payload_path])
        directory = self.payload["directory"]
        assert kwargs["cwd"] == "/app"
        self.files[f"{directory}/cleanup.json"] = json.dumps(self.result)
        self.files[f"{directory}/runtime.json"] = json.dumps({"hostname": "task-container", "pid": 123})
        self.files[f"{directory}/events.jsonl"] = "\n".join(
            json.dumps([float(i), event]) for i, event in enumerate(self.events)
        )
        if self.otel is not None:
            self.files[f"{directory}/otel.jsonl"] = self.otel
        return SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")


@pytest.fixture
def setup():
    sandbox = Sandbox()
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = OmegaConf.create(
        {"policy": {"responses_api_models": {"openai_model": {"host": "model.example", "port": 9000}}}}
    )
    client._build_server_base_url.return_value = "http://model.example:9000"
    config = CopilotAgentConfig(
        name="copilot",
        host="localhost",
        port=8001,
        entrypoint="app.py",
        num_workers=1,
        model_server={"type": "responses_api_models", "name": "policy"},
        model="test-model",
        session_close_timeout_seconds=1,
    )
    module = "responses_api_agents.copilot_agent.app"
    with (
        patch(f"{module}.resolve_provider_config"),
        patch(
            f"{module}.get_global_config_dict",
            return_value={"sandbox": {"local": {}, "default_metadata": {"provider": "test"}}},
        ),
        patch(f"{module}.create_provider"),
        patch(f"{module}.AsyncSandbox.connect", AsyncMock(return_value=sandbox)),
    ):
        yield CopilotAgent(config=config, server_client=client), sandbox


def run_session(agent, body=None):
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        session_id = created.json()["agent_session_id"]
        result = client.post("/ng-rollout/copilot-smoke-a2/v1/responses", json=body or {"input": "Fix the code"})
        closed = client.post(
            "/v1/agent_sessions/close",
            json={"agent_session_id": session_id, "episode_id": seed().episode_id.model_dump()},
        )
        return result, closed


def test_parse_events_orders_reasoning_before_the_message_it_preceded():
    transcript = parse_copilot_events([(float(i), e) for i, e in enumerate(copilot_events())], "inv")
    assert [item.type for item in transcript.output] == [
        "reasoning",
        "function_call",
        "function_call_output",
        "message",
    ]
    call, output = transcript.output[1], transcript.output[2]
    assert (call.call_id, call.name, json.loads(call.arguments)) == ("call-1", "bash", {"command": "pwd"})
    assert (output.call_id, output.output) == ("call-1", "/app\n<exit code 0>")
    tool = transcript.tools["call-1"]
    assert tool.status == "completed" and tool.tool_name == "bash"
    assert tool.duration_ms == pytest.approx(59, abs=1)
    assert transcript.response_ids == ["chatcmpl-1", "chatcmpl-2"]
    assert transcript.session_id == "copilot-run-1" and transcript.error is None and transcript.gaps == []


def test_parse_events_keeps_unfinished_and_failed_tools():
    events = copilot_events()[:3] + [
        {"type": "tool.execution_start", "data": {"toolCallId": "call-2", "toolName": "view"}},
        {
            "type": "tool.execution_complete",
            "data": {"toolCallId": "call-1", "success": False, "error": {"message": "denied"}},
        },
    ]
    transcript = parse_copilot_events([(0.0, e) for e in events], "inv")
    assert transcript.tools["call-1"].status == "failed"
    assert transcript.output[-1].output == "denied"
    assert transcript.tools["call-2"].status == "incomplete"
    assert {gap.code for gap in transcript.gaps} == {"tool_timing_unavailable"}


def test_nonzero_shell_exit_is_a_failed_tool_call():
    # Copilot 1.0.103 reports success for a shell command that exits 7 (seen in the tool_failure probe).
    events = copilot_events()
    events[3]["data"] |= {"success": True, "shellExecution": {"exitCode": 7}}
    tool = parse_copilot_events([(0.0, e) for e in events], "inv").tools["call-1"]
    assert (tool.status, tool.error_type) == ("failed", "exit_code_7")
    events[3]["data"]["shellExecution"] = {"exitCode": 0}
    assert parse_copilot_events([(0.0, e) for e in events], "inv").tools["call-1"].status == "completed"


def test_usage_sums_successful_calls_and_keeps_missing_counts_unknown():
    usage = parse_copilot_usage(otel(chat_span("a", usage=(10, 3)), chat_span("b", usage=(5, 2))))
    assert (usage.input_tokens, usage.output_tokens, usage.response_ids) == (15, 5, ["a", "b"])
    assert usage.cached_tokens is None and usage.reasoning_tokens is None
    # A retried 429 attempt has an error span and no tokens to count.
    retried = parse_copilot_usage(otel(chat_span("x", usage=None, status=2), chat_span("a", usage=(10, 3))))
    assert (retried.input_tokens, retried.response_ids) == (10, ["a"])
    omitted = parse_copilot_usage(otel(chat_span("a", usage=None)))
    assert omitted.input_tokens is None and omitted.output_tokens is None
    assert parse_copilot_usage(None) is None


@pytest.mark.parametrize(
    "native,expected",
    [
        ({"statusCode": 400, "message": "400 bad request"}, ("model_request_error", 400)),
        ({"statusCode": 429}, ("model_request_error", 429)),
        ({"message": "Could not connect to local model provider at http://x/v1."}, ("model_transport_error", None)),
        ({"message": "something else"}, ("unknown", None)),
    ],
)
def test_classify_outcome(setup, native, expected):
    agent, _ = setup
    outcome = agent.classify_outcome(native)
    assert (outcome.failure_kind, outcome.status_code) == expected


@pytest.mark.parametrize(
    "body,detail",
    [
        ({"input": "x", "instructions": "be brief"}, "system-prompt"),
        ({"input": [{"role": "system", "content": "s"}, {"role": "user", "content": "x"}]}, "one text user prompt"),
        ({"input": "x", "max_output_tokens": 10}, "total max_output_tokens"),
        ({"input": "x", "temperature": 0.5}, "temperature"),
        ({"input": "x", "model": "other"}, "configured model"),
    ],
)
def test_unsupported_request_controls_are_rejected(setup, body, detail):
    agent, _ = setup
    with pytest.raises(HTTPException) as error:
        agent._validate_request(NeMoGymResponseCreateParamsNonStreaming.model_validate(body))
    assert error.value.status_code == 422 and detail in error.value.detail


def test_text_parts_and_timeout_metadata_are_accepted(setup):
    agent, _ = setup
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Fix "}, {"type": "input_text", "text": "it"}],
                }
            ],
            "metadata": {"agent_timeout_sec": "30"},
        }
    )
    assert agent._validate_request(body)[1] == "Fix it"


def test_session_runs_copilot_in_borrowed_sandbox(setup):
    agent, sandbox = setup
    result, closed = run_session(agent)
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["status"] == "completed"
    assert [item["type"] for item in body["output"]] == [
        "reasoning",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert body["output"][-1]["content"][0]["text"] == "Fixed"
    assert (body["usage"]["input_tokens"], body["usage"]["output_tokens"], body["usage"]["total_tokens"]) == (
        15,
        5,
        20,
    )
    assert body["metadata"]["harness_execution"] == "sandbox"
    assert body["metadata"]["copilot_version"] == "1.0.103"
    command, env = sandbox.payload["command"], sandbox.payload["env"]
    assert command[0] == RUNTIME["binary"] and command[1:3] == ["-p", "Fix the code"]
    for flag in ("--allow-all-tools", "--no-ask-user", "--disable-builtin-mcps", "--no-auto-update"):
        assert flag in command
    assert command[command.index("--stream") + 1] == "off"
    assert command[command.index("--output-format") + 1] == "json"
    assert env["COPILOT_PROVIDER_BASE_URL"] == "http://model.example:9000/ng-rollout/copilot-smoke-a2/v1"
    assert env["COPILOT_OFFLINE"] == "true" and env["COPILOT_MODEL"] == "test-model"
    # Capture assigns failed attempts, which have no response ID, to the invocation through this header.
    assert env["COPILOT_PROVIDER_HEADERS"] == "x-session-id: copilot-smoke-a2"
    assert env["COPILOT_CLI_DIST_DIR"] == RUNTIME["dist_dir"]
    assert env["COPILOT_HOME"].startswith(sandbox.payload["directory"])
    # Tool shells keep the task environment.
    assert not ({"HOME", "PATH"} & set(env)) and not any(key.startswith("XDG_") for key in env)
    assert "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS" not in env
    install = next(c for c in sandbox.exec.await_args_list if "install_copilot_runtime.py" in c.args[0])
    assert install.args[0].endswith("/tmp/nemo-gym-copilot-1.0.103 1.0.103 https://registry.npmjs.org")
    assert closed.status_code == 200, closed.text
    observations = closed.json()["agent_observations"]
    assert observations["source"] == "copilot"
    invocation, tool = observations["records"]
    assert invocation["status"] == "completed" and invocation["harness_session_id"] == "copilot-run-1"
    assert [call["response_id"] for call in invocation["model_calls"]] == ["chatcmpl-1", "chatcmpl-2"]
    assert tool["tool_call_id"] == "call-1" and tool["timing_source"] == "harness"
    assert "partial_model_usage_unavailable" not in {gap["code"] for gap in observations["gaps"]}
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


def test_configured_limits_reach_copilot(setup):
    agent, sandbox = setup
    agent.config.context_window = 32768
    agent.config.max_output_tokens = 4096
    agent.config.reasoning_effort = "low"
    result, _ = run_session(agent)
    assert result.status_code == 200, result.text
    env, command = sandbox.payload["env"], sandbox.payload["command"]
    assert (env["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"], env["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"]) == ("32768", "4096")
    assert command[command.index("--reasoning-effort") + 1] == "low"


def test_omitted_provider_usage_stays_unknown(setup):
    agent, sandbox = setup
    sandbox.otel = otel(chat_span("chatcmpl-1", usage=None), chat_span("chatcmpl-2", usage=(5, 2)))
    result, closed = run_session(agent)
    assert result.json()["usage"] is None
    assert "partial_model_usage_unavailable" in {gap["code"] for gap in closed.json()["agent_observations"]["gaps"]}


def test_reported_provider_error_is_a_classified_failed_response(setup):
    agent, sandbox = setup
    sandbox.events = copilot_events(error={"errorType": "query", "message": "400 rejected", "statusCode": 400})
    sandbox.otel = otel(chat_span("rejected", usage=None, status=2))
    sandbox.result["return_code"] = 1
    result, closed = run_session(agent)
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["status"] == "failed" and body["error"]["message"] == "400 rejected"
    assert body["usage"]["total_tokens"] == 0
    outcome = json.loads(body["metadata"][AGENT_OUTCOME_METADATA_KEY])
    assert outcome == {"failure_kind": "model_request_error", "status_code": 400}
    assert closed.json()["agent_observations"]["records"][0]["status"] == "failed"


def test_unexplained_nonzero_exit_is_an_execution_error(setup):
    agent, sandbox = setup
    sandbox.result["return_code"] = 3
    result, closed = run_session(agent)
    assert result.status_code == 500
    assert closed.status_code == 200


def test_missing_result_event_is_an_execution_error(setup):
    agent, sandbox = setup
    sandbox.events = copilot_events(result=False)
    result, _ = run_session(agent)
    assert result.status_code == 500


def test_timeout_returns_gradable_partial_work(setup):
    agent, sandbox = setup
    sandbox.result |= {"timed_out": True, "return_code": None}
    sandbox.events = copilot_events(result=False)
    result, closed = run_session(agent)
    assert result.status_code == 200, result.text
    assert result.json()["status"] == "incomplete"
    gaps = {gap["code"] for gap in closed.json()["agent_observations"]["gaps"]}
    assert "worker_exit_code_unavailable" in gaps


def test_sessionless_calls_and_compatibility_run_are_rejected(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/responses", json={"input": "x"}).status_code == 422
        response = client.post("/run", json={"responses_create_params": {"input": "x"}})
        assert response.status_code == 422 and "Environment Server" in response.json()["detail"]
    sandbox.exec.assert_not_awaited()


@pytest.mark.parametrize("option", ["no-sandbox", "required-tool", "no-model", "unpinned"])
def test_unsupported_seed_is_refused_before_connection(setup, option):
    agent, sandbox = setup
    body = seed()
    if option == "no-sandbox":
        body = seed(sandbox_access=False)
    elif option == "required-tool":
        body = seed(tool_accesses=[{"kind": "direct_http", "name": "tools", "base_url": "http://r", "required": True}])
    elif option == "no-model":
        agent.config.model_server = None
    else:
        agent.config.copilot_version = "latest"
    check_agent_seed_rejected(agent.setup_webserver(), body)
    sandbox.exec.assert_not_awaited()


async def test_install_failure_keeps_installer_output(setup):
    agent, sandbox = setup
    ok = SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")
    failed = SimpleNamespace(return_code=1, error_type=None, stdout="glibc 2.17 too old", stderr="exit status 1")
    sandbox.exec.side_effect = [ok, failed, ok, ok, ok]
    with pytest.raises(RuntimeError) as error:
        await agent._seed_agent_session_state(seed())
    assert "glibc 2.17 too old" in str(error.value) and "exit status 1" in str(error.value)
    sandbox.disconnect.assert_awaited_once()
    sandbox.launch.assert_not_awaited()


def test_changed_activation_is_rejected_and_identical_retry_joins(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).status_code == 200
        first = client.post("/ng-rollout/copilot-smoke-a2/v1/responses", json={"input": "Fix the code"})
        again = client.post("/ng-rollout/copilot-smoke-a2/v1/responses", json={"input": "Fix the code"})
        changed = client.post("/ng-rollout/copilot-smoke-a2/v1/responses", json={"input": "Other"})
        wrong_route = client.post("/ng-rollout/other-a0/v1/responses", json={"input": "Fix the code"})
    assert first.status_code == again.status_code == 200
    assert first.json()["id"] == again.json()["id"]
    assert changed.status_code == 409 and wrong_route.status_code == 409
    assert sandbox.launch.await_count == 1


def test_owned_sandbox_is_created_and_stopped(setup):
    agent, sandbox = setup
    agent.config.sandbox_provider = "sandbox"
    sandbox.start = AsyncMock()
    module = "responses_api_agents.copilot_agent.app"
    with patch(f"{module}.AsyncSandbox", return_value=sandbox):
        with TestClient(agent.setup_webserver()) as client:
            created = client.post("/v1/agent_sessions", json=seed(sandbox_access=False).model_dump(mode="json"))
            assert created.status_code == 200, created.text
            session_id = created.json()["agent_session_id"]
            assert client.post("/ng-rollout/copilot-smoke-a2/v1/responses", json={"input": "x"}).status_code == 200
            closed = client.post(
                "/v1/agent_sessions/close",
                json={"agent_session_id": session_id, "episode_id": seed().episode_id.model_dump()},
            )
            assert closed.status_code == 200
    sandbox.start.assert_awaited_once()
    spec = sandbox.start.await_args.args[0]
    assert spec.ttl_s == agent.config.sandbox_install_timeout_seconds + agent.config.timeout + 600
    assert spec.metadata == {"provider": "test", "harness": "copilot", "agent_session_id": seed().agent_session_id}
    sandbox.stop.assert_awaited_once()


def test_interrupted_activation_keeps_observations(setup):
    agent, sandbox = setup
    started = asyncio.Event()

    async def hang(command, **kwargs):
        started.set()
        await asyncio.Event().wait()

    sandbox.launch.side_effect = hang

    async def scenario():
        state = await agent._seed_agent_session_state(seed())
        body = NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": "Fix"})
        task = asyncio.create_task(agent._sandbox_response(state, body, prompt="Fix"))
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return state

    state = asyncio.run(scenario())
    codes = {gap.code for gap in state.observations.gaps}
    assert "agent_activation_interrupted" in codes
    assert state.observations.records[0].status == "incomplete"


@pytest.mark.parametrize("spans", ["", "not-json", otel()])
def test_absent_chat_spans_do_not_invent_zero_usage(spans):
    assert parse_copilot_usage(spans) is None


def test_closed_session_rejects_activation_and_close_replays(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).status_code == 200
        payload = {"agent_session_id": seed().agent_session_id, "episode_id": seed().episode_id.model_dump()}
        first = client.post("/v1/agent_sessions/close", json=payload)
        again = client.post("/v1/agent_sessions/close", json=payload)
        stale = client.post("/ng-rollout/copilot-smoke-a2/v1/responses", json={"input": "Fix"})
    assert first.status_code == again.status_code == 200
    assert first.json() == again.json()
    assert stale.status_code == 409
    assert "resources_cookies" not in first.json() or first.json()["resources_cookies"] is None
    assert first.json()["agent_observations"]["gaps"]
    sandbox.launch.assert_not_awaited()


def test_rejected_calls_have_zero_completed_response_tokens():
    usage = parse_copilot_usage(otel(chat_span("failed", usage=None, status=2)))
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens, usage.cached_tokens, usage.reasoning_tokens) == (0, 0, 0, 0)
    assert usage.response_ids == []
