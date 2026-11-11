# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
import shlex
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientConnectionError, ClientResponseError
from fastapi import HTTPException
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    HARNESS_RUNNER_FILE,
    HARNESS_RUNNER_SCRIPT,
    AgentSeedSessionRequest,
    check_agent_seed_rejected,
)
from nemo_gym.client import ServerClient
from nemo_gym.core import (
    EpisodeId,
    FailureHeader,
    NeMoGymResponseCreateParamsNonStreaming,
    NgFailureEvent,
    RequestFailureLog,
    TaskId,
    failure_report_for,
)
from responses_api_agents.copilot_agent.app import (
    CopilotAgent,
    CopilotAgentConfig,
    copilot_package,
    copilot_package_metadata,
    parse_copilot_events,
    parse_copilot_usage,
)
from responses_api_agents.copilot_agent.sandbox import CopilotArtifacts


MODULE = "responses_api_agents.copilot_agent.app"
TARBALL = "https://registry.npmjs.org/@github/copilot-linux-x64/-/copilot-linux-x64-1.0.103.tgz"
SHA512 = bytes(range(64)).hex()
RUNTIME = {
    "binary": "/tmp/nemo-gym-copilot-1.0.103/copilot",
    "dist_dir": "/tmp/nemo-gym-copilot-1.0.103/home/.cache/copilot/pkg/linux-x64/1.0.103",
    "package": "copilot-linux-x64",
    "version": "1.0.103",
    "sha512": SHA512,
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


def staged(params: str) -> dict:
    """The activation as harness_runner.sh sees it: params.sh sourced by bash."""
    names = ("cwd", "stdin", "stdout", "stderr", "events")
    script = 'source "$1"; printf "%s\\0" ' + " ".join(f'"${name}"' for name in names)
    script += ' "${#command[@]}" "${command[@]}" "${env[@]}"'
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "params.sh"
        path.write_text(params)
        values = subprocess.run(
            ["bash", "-c", script, "bash", str(path)], capture_output=True, text=True, check=True
        ).stdout.split("\0")[:-1]
    count = int(values[len(names)])
    command = values[len(names) + 1 : len(names) + 1 + count]
    env = dict(item.split("=", 1) for item in values[len(names) + 1 + count :])
    return {**dict(zip(names, values)), "command": command, "env": env}


def ok(stdout="") -> SimpleNamespace:
    return SimpleNamespace(return_code=0, error_type=None, stdout=stdout, stderr="")


class Sandbox:
    """Provider double: records uploads, answers setup commands, and plays one supervised run."""

    def __init__(self):
        self.files = {}
        self.sandbox_id = "resources-owned"
        self.result = {"return_code": 0, "timed_out": False, "cleanup_confirmed": True, "error": None}
        self.events = copilot_events()
        self.otel = otel(chat_span("chatcmpl-1", usage=(10, 3)), chat_span("chatcmpl-2", usage=(5, 2)))
        self.compaction_state = False
        self.platform = "Linux\nx86_64\nglibc 2.31\n"
        self.stderr = ""
        self.exec = AsyncMock(side_effect=self.execute)
        self.stop = AsyncMock()
        self.disconnect = AsyncMock()
        self.launch = AsyncMock(side_effect=self.run)
        # The Agent Server's registry lookup, patched in by the fixture.
        self.metadata = AsyncMock(return_value=(TARBALL, SHA512))
        self.payload = None

    async def execute(self, command, **kwargs):
        if command.startswith("ls -1 -- ") and "/copilot-home/session-state" in command:
            if self.compaction_state is None:
                return SimpleNamespace(return_code=2, error_type=None, stdout="", stderr="No such file or directory")
            path = shlex.split(command)[3].replace("*", "copilot-run-1")
            kind = "session.compaction_complete" if self.compaction_state else "assistant.message"
            self.files[path] = "\n".join(json.dumps({"type": event}) for event in ("session.start", kind)) + "\n"
            return ok(path + "\n")
        if "--receipt" in command and "process_supervisor" in command:
            return await self.launch(command, **kwargs)
        if command.startswith("uname -s;"):
            return ok(self.platform)
        if "install_copilot_runtime.sh" in command:
            return ok("Downloading the package\n" + json.dumps(RUNTIME) + "\n")
        if "ln -s stop " in command:
            self.result["timed_out"] = True
        return ok()

    async def upload(self, source, destination):
        self.files[destination] = Path(source).read_text()

    async def download(self, source, destination):
        Path(destination).write_text(self.files[source])

    async def run(self, command, **kwargs):
        directory = next(path.rsplit("/", 1)[0] for path in self.files if path.endswith("/params.sh"))
        self.payload = {"directory": directory, **staged(self.files[f"{directory}/params.sh"])}
        assert kwargs["cwd"] == "/app"
        # The shared bash runner starts Copilot, so the task image needs no Python.
        # The supervisor's argv follows "--"; its output redirection follows the argv.
        harness = shlex.split(command.split(" -- ", 1)[1])[:-2]
        assert harness == ["bash", f"{directory}/{HARNESS_RUNNER_FILE}", f"{directory}/params.sh"]
        assert "gym-runner" not in command
        self.files[f"{directory}/cleanup.json"] = json.dumps(self.result)
        self.files[f"{directory}/runtime.json"] = json.dumps({"hostname": "task-container", "pid": 123})
        self.files[f"{directory}/events.jsonl"] = "\n".join(
            json.dumps([float(i), event]) for i, event in enumerate(self.events)
        )
        self.files[f"{directory}/stderr.log"] = self.stderr
        if self.otel is not None:
            self.files[f"{directory}/otel.jsonl"] = self.otel
        return SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")


def agent_config(**overrides) -> CopilotAgentConfig:
    return CopilotAgentConfig(
        name="copilot",
        host="localhost",
        port=8001,
        entrypoint="app.py",
        num_workers=1,
        model_server={"type": "responses_api_models", "name": "policy"},
        model="test-model",
        session_close_timeout_seconds=1,
        **overrides,
    )


@pytest.fixture
def setup():
    sandbox = Sandbox()
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = OmegaConf.create(
        {"policy": {"responses_api_models": {"openai_model": {"host": "model.example", "port": 9000}}}}
    )
    client._build_server_base_url.return_value = "http://model.example:9000"
    with (
        patch(f"{MODULE}.resolve_provider_config"),
        patch(
            f"{MODULE}.get_global_config_dict",
            return_value={"sandbox": {"local": {}, "default_metadata": {"provider": "test"}}},
        ),
        patch(f"{MODULE}.create_provider"),
        patch(f"{MODULE}.AsyncSandbox.connect", AsyncMock(return_value=sandbox)),
        patch(f"{MODULE}.copilot_package_metadata", sandbox.metadata),
    ):
        yield CopilotAgent(config=agent_config(), server_client=client), sandbox


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


@pytest.mark.parametrize("finished", [True, False])
def test_later_turn_reasoning_stays_after_the_previous_tool_result(finished):
    events = copilot_events()[:4] + [{"type": "assistant.reasoning", "data": {"content": "Next turn"}}]
    if finished:
        events += copilot_events()[4:]
    transcript = parse_copilot_events([(0.0, event) for event in events], "inv")
    assert [item.type for item in transcript.output] == [
        "reasoning",
        "function_call",
        "function_call_output",
        "reasoning",
        *(["message"] if finished else []),
    ]
    assert transcript.output[3].summary[0].text == "Next turn"


@pytest.mark.parametrize("marker", ["agentId", "data.agentId", "data.parentToolCallId"])
def test_child_events_keep_references_without_changing_root_transcript(marker):
    child_events = copilot_events()
    for event in child_events:
        if marker == "agentId":
            event["agentId"] = "child"
        else:
            event.setdefault("data", {})[marker.split(".")[1]] = "child"
    child_events[0]["data"]["apiCallId"] = "child-response"
    events = [
        {"type": "assistant.message", "data": {"content": "Root answer", "apiCallId": "root-response"}},
        *child_events,
        {"type": "assistant.reasoning", "data": {"content": "Root reasoning"}},
        {"type": "result", "sessionId": "root-session", "exitCode": 0},
    ]
    transcript = parse_copilot_events([(0.0, event) for event in events], "inv")
    assert [item.type for item in transcript.output] == ["reasoning", "message"]
    assert transcript.output[0].summary[0].text == "Root reasoning"
    assert transcript.output[1].content[0].text == "Root answer"
    assert transcript.tools == {} and transcript.session_id == "root-session"
    assert transcript.response_ids == ["root-response", "child-response", "chatcmpl-2"]


def test_explicit_turn_start_detaches_reasoning_from_previous_message():
    events = [
        {"type": "assistant.message", "data": {"content": "Previous"}},
        {"type": "assistant.turn_start"},
        {"type": "assistant.reasoning", "data": {"content": "New turn"}},
    ]
    transcript = parse_copilot_events([(0.0, event) for event in events], "inv")
    assert [item.type for item in transcript.output] == ["message", "reasoning"]


@pytest.mark.parametrize(
    "error_event,fails",
    [
        ({"agentId": "sub-1", "data": {"errorType": "query", "message": "503", "statusCode": 503}}, False),
        ({"data": {"errorType": "query", "message": "503", "parentToolCallId": "call-9"}}, False),
        ({"data": {"errorType": "hook", "message": "hook failed"}}, False),
        # A later root assistant message shows that Copilot recovered from the failed call.
        ({"data": {"errorType": "model_call", "message": "fetch failed"}}, False),
        ({"data": {"errorType": "query", "message": "400 rejected", "statusCode": 400}}, True),
    ],
    ids=["subagent", "subagent-tool", "hook", "recovered-model-call", "root-query"],
)
def test_only_root_errors_that_end_the_run_fail_it(error_event, fails):
    events = copilot_events()
    events.insert(4, {"type": "session.error", **error_event})
    transcript = parse_copilot_events([(0.0, e) for e in events], "inv")
    assert (transcript.error is not None) is fails
    assert transcript.output[-1].content[0].text == "Fixed"


def test_unrecovered_model_call_error_fails_the_run():
    events = copilot_events(result=False)[:4] + [
        {"type": "session.error", "data": {"errorType": "model_call", "message": "fetch failed"}},
        {"type": "result", "sessionId": "copilot-run-1", "exitCode": 0},
    ]
    assert parse_copilot_events([(0.0, e) for e in events], "inv").error["message"] == "fetch failed"


def test_subagent_error_keeps_a_finished_run_gradable(setup):
    agent, sandbox = setup
    sandbox.events = copilot_events()
    sandbox.events.insert(
        4, {"type": "session.error", "agentId": "sub-1", "data": {"errorType": "query", "statusCode": 503}}
    )
    result, _ = run_session(agent)
    body = result.json()
    assert body["status"] == "completed" and body["error"] is None
    assert AGENT_OUTCOME_METADATA_KEY not in body["metadata"]


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
    sandbox.disconnect.side_effect = lambda: setattr(sandbox, "sandbox_id", None)
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
    command, env, directory = sandbox.payload["command"], sandbox.payload["env"], sandbox.payload["directory"]
    assert command[0] == RUNTIME["binary"] and command[1] == "--prompt=Fix the code"
    for flag in ("--allow-all-tools", "--no-ask-user", "--disable-builtin-mcps", "--no-auto-update"):
        assert flag in command
    # Usage comes from the OpenTelemetry spans; Copilot's own usage file is not written.
    assert "--usage-output-file" not in command
    assert sandbox.payload["cwd"] == "/app" and sandbox.payload["stdin"] == "/dev/null"
    assert sandbox.payload["events"] == f"{directory}/events.jsonl"
    assert command[command.index("--stream") + 1] == "off"
    assert command[command.index("--output-format") + 1] == "json"
    assert env["COPILOT_PROVIDER_BASE_URL"] == "http://model.example:9000/ng-rollout/copilot-smoke-a2/v1"
    assert env["COPILOT_OFFLINE"] == "true" and env["COPILOT_MODEL"] == "test-model"
    # Capture assigns failed attempts, which have no response ID, to the invocation through this header.
    header_name, session_header = env["COPILOT_PROVIDER_HEADERS"].split(": ")
    assert header_name == "x-session-id"
    assert env["COPILOT_CLI_DIST_DIR"] == RUNTIME["dist_dir"]
    assert env["COPILOT_HOME"].startswith(sandbox.payload["directory"])
    # Tool shells keep the task environment.
    assert not ({"HOME", "PATH"} & set(env)) and not any(key.startswith("XDG_") for key in env)
    assert "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS" not in env
    # The Agent Server reads the package's digest from the registry; the sandbox installer checks the download.
    sandbox.metadata.assert_awaited_once_with("https://registry.npmjs.org", "copilot-linux-x64", "1.0.103")
    install = next(c for c in sandbox.exec.await_args_list if "install_copilot_runtime.sh" in c.args[0])
    assert shlex.split(install.args[0]) == [
        "bash",
        f"{directory}/install_copilot_runtime.sh",
        "/tmp/nemo-gym-copilot-1.0.103",
        "1.0.103",
        "copilot-linux-x64",
        TARBALL,
        SHA512,
    ]
    assert install.kwargs["cwd"] == "/"
    assert sandbox.files[f"{directory}/{HARNESS_RUNNER_FILE}"] == HARNESS_RUNNER_SCRIPT.read_text()
    # A CLI harness needs only bash in the task image.
    assert not any("python" in call.args[0] for call in sandbox.exec.await_args_list)
    assert closed.status_code == 200, closed.text
    observations = closed.json()["agent_observations"]
    assert observations["source"] == "copilot"
    invocation, tool, sandbox_record = observations["records"]
    assert sandbox_record["sandbox_id"] == tool["sandbox_id"] == "resources-owned"
    assert sandbox_record["outcome"] == "completed" and sandbox_record["exit_code"] == 0
    assert sandbox_record["provider"] == "sandbox"
    assert invocation["invocation_id"] == tool["invocation_id"] == session_header
    assert invocation["status"] == "completed" and invocation["harness_session_id"] == "copilot-run-1"
    assert [call["response_id"] for call in invocation["model_calls"]] == ["chatcmpl-1", "chatcmpl-2"]
    assert tool["tool_call_id"] == "call-1" and tool["timing_source"] == "harness"
    assert "partial_model_usage_unavailable" not in {gap["code"] for gap in observations["gaps"]}
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


def test_provider_secrets_cannot_mask_task_text(setup):
    # Copilot masks its provider key and header values in every event it writes. With the capture key "1-0" as the
    # header value, a timestamp "2026-11-03" was recorded as "2026-1******3", and tool-call IDs containing "2-0" were
    # rewritten. Each value must be one that model text, tool output and IDs cannot contain.
    agent, sandbox = setup
    result, closed = run_session(agent)
    assert result.status_code == 200, result.text
    env = sandbox.payload["env"]
    session_header = env["COPILOT_PROVIDER_HEADERS"].removeprefix("x-session-id: ")
    api_key = env["COPILOT_PROVIDER_API_KEY"]
    assert session_header.startswith("copilot-") and len(session_header.removeprefix("copilot-")) == 32
    assert api_key.startswith("unused-") and len(api_key.removeprefix("unused-")) == 32
    # The capture key stays in the rollout-prefixed URL, which Copilot does not mask.
    assert "/ng-rollout/copilot-smoke-a2/" in env["COPILOT_PROVIDER_BASE_URL"]
    assert closed.json()["agent_observations"]["records"][0]["invocation_id"] == session_header


def test_leading_dash_prompt_is_passed_as_one_option_value(setup):
    agent, sandbox = setup
    prompt = '- You are given a PyTorch state dictionary.\nPreserve "quotes", =, and ü.'
    result, closed = run_session(agent, {"input": prompt})
    assert result.status_code == 200 and closed.status_code == 200
    assert sandbox.payload["command"][1] == f"--prompt={prompt}"
    assert "-p" not in sandbox.payload["command"]


def test_configured_limits_reach_copilot(setup):
    agent, sandbox = setup
    agent.config.context_window = 32768
    agent.config.max_output_tokens = 4096
    agent.config.reasoning_effort = "low"
    result, _ = run_session(agent)
    assert result.status_code == 200, result.text
    env, command = sandbox.payload["env"], sandbox.payload["command"]
    # The context window holds the reply too; Copilot's limit is for the prompt alone.
    assert (env["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"], env["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"]) == ("28672", "4096")
    assert command[command.index("--reasoning-effort") + 1] == "low"


@pytest.mark.parametrize("max_output_tokens", [8192, 9000])
def test_output_limit_must_leave_room_for_the_prompt(max_output_tokens):
    with pytest.raises(ValidationError, match="max_output_tokens must be smaller than context_window"):
        agent_config(context_window=8192, max_output_tokens=max_output_tokens)
    assert agent_config(context_window=8192, max_output_tokens=8191).max_output_tokens == 8191


@pytest.mark.parametrize(
    "prompt", ["Fix\0it", "x" * (131072 - len("--prompt=")), "ü" * 65532], ids=["nul", "too-long", "too-long-utf8"]
)
def test_prompt_that_cannot_be_one_argument_is_rejected(setup, prompt):
    agent, _ = setup
    with pytest.raises(HTTPException) as error:
        agent._validate_request(NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": prompt}))
    assert error.value.status_code == 422 and "one command-line argument" in error.value.detail
    longest = "x" * (131072 - len("--prompt=") - 1)
    assert agent._validate_request(NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": longest}))[1]


def test_omitted_provider_usage_stays_unknown(setup):
    agent, sandbox = setup
    sandbox.otel = otel(chat_span("chatcmpl-1", usage=None), chat_span("chatcmpl-2", usage=(5, 2)))
    result, closed = run_session(agent)
    assert result.json()["usage"] is None
    assert "partial_model_usage_unavailable" in {gap["code"] for gap in closed.json()["agent_observations"]["gaps"]}


def test_span_without_an_event_still_counts(setup):
    agent, sandbox = setup
    # Background calls have chat spans but no assistant event; coverage needs only every event's span.
    sandbox.otel = otel(
        chat_span("chatcmpl-1", usage=(10, 3)), chat_span("chatcmpl-bg", usage=(4, 1)), chat_span("chatcmpl-2")
    )
    result, closed = run_session(agent)
    assert (result.json()["usage"]["input_tokens"], result.json()["usage"]["output_tokens"]) == (24, 7)
    records = closed.json()["agent_observations"]["records"]
    assert [call["response_id"] for call in records[0]["model_calls"]] == ["chatcmpl-1", "chatcmpl-bg", "chatcmpl-2"]


def test_missing_span_for_a_reported_call_keeps_usage_unknown(setup):
    agent, sandbox = setup
    # Events name chatcmpl-2, but its span never reached the file exporter.
    sandbox.otel = otel(chat_span("chatcmpl-1", usage=(10, 3)))
    result, closed = run_session(agent)
    assert result.json()["status"] == "completed" and result.json()["usage"] is None
    observations = closed.json()["agent_observations"]
    assert [call["response_id"] for call in observations["records"][0]["model_calls"]] == ["chatcmpl-1", "chatcmpl-2"]
    gaps = {gap["code"]: gap["detail"] for gap in observations["gaps"]}
    assert "chatcmpl-2" in gaps["model_usage_spans_missing"]
    assert "partial_model_usage_unavailable" not in gaps


@pytest.mark.parametrize("compacted", [False, True])
def test_compaction_does_not_publish_partial_token_totals(setup, compacted):
    agent, sandbox = setup
    span = json.loads(chat_span("chatcmpl-2", usage=(5, 2)))
    # Real Copilot 1.0.103 marks the next chat span but omits the compaction call's span.
    span["attributes"]["gen_ai.conversation.compacted"] = compacted
    sandbox.otel = otel(chat_span("chatcmpl-1", usage=(10, 3)), json.dumps(span))
    result, closed = run_session(agent)
    assert result.json()["status"] == "completed"
    observations = closed.json()["agent_observations"]
    assert [call["response_id"] for call in observations["records"][0]["model_calls"]] == ["chatcmpl-1", "chatcmpl-2"]
    if compacted:
        assert result.json()["usage"] is None
        assert "compaction_model_usage_unavailable" in {gap["code"] for gap in observations["gaps"]}
    else:
        assert result.json()["usage"]["total_tokens"] == 20
        assert "compaction_model_usage_unavailable" not in {gap["code"] for gap in observations["gaps"]}


@pytest.mark.parametrize("compaction_state", [False, True, None])
def test_late_or_unobserved_compaction_does_not_publish_partial_totals(setup, compaction_state):
    agent, sandbox = setup
    # A background compaction can finish after the final ordinary chat: no next-chat marker exists.
    sandbox.compaction_state = compaction_state
    result, closed = run_session(agent)
    assert result.json()["status"] == "completed"
    gaps = {gap["code"] for gap in closed.json()["agent_observations"]["gaps"]}
    if compaction_state is False:
        assert result.json()["usage"]["total_tokens"] == 20
    else:
        assert result.json()["usage"] is None
        assert ("compaction_model_usage_unavailable" if compaction_state else "compaction_state_unavailable") in gaps
        # Every chat span reported usage; only compaction makes the aggregate unknown.
        assert "partial_model_usage_unavailable" not in gaps


def test_reported_provider_error_is_a_classified_failed_response(setup):
    agent, sandbox = setup
    sandbox.events = copilot_events(error={"errorType": "query", "message": "400 rejected", "statusCode": 400})
    # The first call succeeded; the rejected attempt adds no completed-response tokens.
    sandbox.otel = otel(chat_span("chatcmpl-1", usage=(10, 3)), chat_span("rejected", usage=None, status=2))
    sandbox.result["return_code"] = 1
    result, closed = run_session(agent)
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["status"] == "failed" and body["error"]["message"] == "400 rejected"
    assert body["usage"]["total_tokens"] == 13
    record = closed.json()["agent_observations"]["records"][-1]
    assert (record["outcome"], record["error_type"], record["exit_code"]) == ("failed", "copilot_error", 1)
    outcome = json.loads(body["metadata"][AGENT_OUTCOME_METADATA_KEY])
    assert outcome == {"failure_kind": "model_request_error", "status_code": 400}
    assert closed.json()["agent_observations"]["records"][0]["status"] == "failed"


def test_unexplained_nonzero_exit_is_an_execution_error(setup):
    agent, sandbox = setup
    sandbox.result["return_code"] = 3
    result, closed = run_session(agent)
    assert result.status_code == 500
    assert closed.status_code == 200


@pytest.mark.parametrize("stderr", ["native startup failed", "x" * 5000 + "specific failure"], ids=["short", "long"])
def test_native_startup_error_preserves_bounded_stderr(setup, stderr):
    agent, sandbox = setup
    sandbox.events = []
    sandbox.otel = None
    sandbox.stderr = stderr
    sandbox.result["return_code"] = 1
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200
        with pytest.raises(NgFailureEvent, match="agent.responses") as error:
            client.post("/ng-rollout/copilot-smoke-a2/v1/responses", json={"input": "Fix the code"})
        # The agent route reports the adapter's error as the failure's cause and replies with it.
        cause = error.value.__cause__
        assert isinstance(cause, RuntimeError) and "Copilot exited with code 1" in str(cause)
        assert stderr[-4000:] in str(cause)
        assert len(str(cause)) < 4100
        closed = client.post(
            "/v1/agent_sessions/close",
            json={
                "agent_session_id": created.json()["agent_session_id"],
                "episode_id": seed().episode_id.model_dump(),
            },
        )
    assert closed.status_code == 200
    records = closed.json()["agent_observations"]["records"]
    assert records[0]["status"] == "failed" and records[0]["model_calls"] == []


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
    record = closed.json()["agent_observations"]["records"][-1]
    assert record["outcome"] == "timeout" and record["sandbox_id"] == "resources-owned"
    gaps = {gap["code"] for gap in closed.json()["agent_observations"]["gaps"]}
    assert "worker_exit_code_unavailable" in gaps


def test_sessionless_calls_and_compatibility_run_are_rejected(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/responses", json={"input": "x"}).status_code == 422
        response = client.post("/run", json={"responses_create_params": {"input": "x"}})
        assert response.status_code == 422 and "Environment Server" in response.json()["detail"]
    sandbox.exec.assert_not_awaited()


@pytest.mark.parametrize(
    "option,reason",
    [
        ("no-sandbox", "sandbox_access or a configured sandbox_provider"),
        ("relative-workdir", "workdir must be absolute"),
        ("required-tool", "required grants: tools"),
        ("no-model", "sandbox-reachable Gym model_server"),
        ("unpinned", "exact copilot_version"),
    ],
)
def test_unsupported_seed_is_refused_before_connection(setup, option, reason):
    agent, sandbox = setup
    body = seed()
    if option == "no-sandbox":
        body = seed(sandbox_access=False)
    elif option == "relative-workdir":
        body.sandbox_access.workdir = "app"
    elif option == "required-tool":
        body = seed(tool_accesses=[{"kind": "direct_http", "name": "tools", "base_url": "http://r", "required": True}])
    elif option == "no-model":
        agent.config.model_server = None
    else:
        agent.config.copilot_version = "latest"
    assert reason in check_agent_seed_rejected(agent.setup_webserver(), body)
    sandbox.exec.assert_not_awaited()


def test_optional_tool_grant_is_skipped(setup):
    agent, sandbox = setup
    body = seed(tool_accesses=[{"kind": "direct_http", "name": "tools", "base_url": "http://r", "required": False}])
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
    assert created.status_code == 200, created.text


@pytest.mark.parametrize(
    "probe,package",
    [
        ("Linux\nx86_64\nglibc 2.31\n", "copilot-linux-x64"),
        ("Linux\naarch64\nglibc 2.28\n", "copilot-linux-arm64"),
        ("Linux\nx86_64\nmusl\n", "copilot-linuxmusl-x64"),
        ("Linux\naarch64\nmusl\n", "copilot-linuxmusl-arm64"),
    ],
)
def test_platform_probe_selects_the_package(probe, package):
    assert copilot_package(probe) == package


@pytest.mark.parametrize(
    "probe,reason",
    [
        ("Darwin\narm64\nunknown\n", "Linux only"),
        ("Linux\nppc64le\nglibc 2.31\n", "x86_64 and aarch64"),
        ("Linux\nx86_64\nglibc 2.17\n", "glibc 2.28 or newer"),
    ],
)
def test_unsupported_platform_is_refused_at_seed(setup, probe, reason):
    agent, sandbox = setup
    sandbox.platform = probe
    assert reason in check_agent_seed_rejected(agent.setup_webserver(), seed())
    sandbox.metadata.assert_not_awaited()
    assert not any("install_copilot_runtime.sh" in call.args[0] for call in sandbox.exec.await_args_list)
    sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("probe", ["Linux\nx86_64\nunknown\n", "Linux\nx86_64\n"])
def test_unidentified_platform_is_an_error_to_retry(probe):
    # Not AgentSeedRejectedError: an Environment Server retries a 500 with a fresh sandbox.
    with pytest.raises(RuntimeError):
        copilot_package(probe)


class Reply:
    """An aiohttp response double for the registry."""

    def __init__(self, body, status=200):
        self.body, self.status = body, status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise ClientResponseError(MagicMock(), (), status=self.status)

    async def json(self, content_type="application/json"):
        return self.body


def registry(monkeypatch, reply) -> tuple[list[tuple[str, str]], dict]:
    calls, cache = [], {}

    async def request(method, url, **kwargs):
        calls.append((method, url))
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(f"{MODULE}.http_request", request)
    monkeypatch.setattr(f"{MODULE}._PACKAGE_METADATA", cache)
    return calls, cache


async def test_package_metadata_is_read_once_on_the_host(monkeypatch):
    integrity = "sha512-" + base64.b64encode(bytes(range(64))).decode()
    calls, _ = registry(monkeypatch, Reply({"dist": {"tarball": TARBALL, "integrity": integrity}}))
    for _ in range(2):
        assert await copilot_package_metadata("https://registry.npmjs.org/", "copilot-linux-x64", "1.0.103") == (
            TARBALL,
            SHA512,
        )
    assert calls == [("GET", "https://registry.npmjs.org/@github/copilot-linux-x64/1.0.103")]


@pytest.mark.parametrize(
    "reply",
    [
        ClientConnectionError("connection refused"),
        Reply({"error": "Unavailable"}, status=503),
        Reply({"name": "@github/copilot-linux-x64"}),
        Reply({"dist": {"tarball": TARBALL, "integrity": "sha1-AAAA"}}),
        Reply({"dist": {"tarball": TARBALL, "integrity": "sha512-not base64"}}),
    ],
    ids=["transport", "status", "no-dist", "sha1", "malformed"],
)
async def test_package_metadata_failure_is_an_error_to_retry(monkeypatch, reply):
    _, cache = registry(monkeypatch, reply)
    with pytest.raises(RuntimeError, match="@github/copilot-linux-x64"):
        await copilot_package_metadata("https://registry.npmjs.org", "copilot-linux-x64", "1.0.103")
    # A later seed tries again.
    assert cache == {}


def test_unreachable_registry_fails_the_seed_for_retry(setup, monkeypatch):
    agent, sandbox = setup
    registry(monkeypatch, ClientConnectionError("connection refused"))
    sandbox.metadata.side_effect = copilot_package_metadata
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
    assert created.status_code == 500
    sandbox.disconnect.assert_awaited_once()
    sandbox.launch.assert_not_awaited()


async def test_install_failure_keeps_installer_output(setup):
    agent, sandbox = setup
    original = sandbox.execute

    async def execute(command, **kwargs):
        if "install_copilot_runtime.sh" in command:
            return SimpleNamespace(return_code=1, error_type=None, stdout="curl: (6) no host", stderr="exit status 1")
        return await original(command, **kwargs)

    sandbox.exec.side_effect = execute
    with pytest.raises(RuntimeError) as error:
        await agent._seed_agent_session_state(seed())
    assert "curl: (6) no host" in str(error.value) and "exit status 1" in str(error.value)
    sandbox.disconnect.assert_awaited_once()
    sandbox.launch.assert_not_awaited()


@pytest.mark.parametrize("stdout", [None, "", "\n", "Downloading\nnot a record\n"])
async def test_installer_without_a_runtime_record_fails_setup(setup, stdout):
    agent, sandbox = setup
    original = sandbox.execute

    async def execute(command, **kwargs):
        if "install_copilot_runtime.sh" in command:
            return SimpleNamespace(return_code=0, error_type=None, stdout=stdout, stderr="installer diagnostics")
        return await original(command, **kwargs)

    sandbox.exec.side_effect = execute
    with pytest.raises(RuntimeError, match="Copilot sandbox installation reported"):
        await agent._seed_agent_session_state(seed())
    sandbox.disconnect.assert_awaited_once()


async def test_setup_is_bounded_by_the_install_budget(setup):
    agent, sandbox = setup
    agent.config.sandbox_install_timeout_seconds = 0.2
    original = sandbox.execute

    async def execute(command, **kwargs):
        if "install_copilot_runtime.sh" in command:
            await asyncio.Event().wait()
        return await original(command, **kwargs)

    sandbox.exec.side_effect = execute
    # The provider's own exec timeout is not the only bound: the whole setup shares the budget.
    seeding = asyncio.create_task(agent._seed_agent_session_state(seed()))
    done, _ = await asyncio.wait({seeding}, timeout=10)
    if not done:
        seeding.cancel()
    assert done and isinstance(seeding.exception(), TimeoutError)
    sandbox.disconnect.assert_awaited_once()


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
    sandbox.stop.side_effect = lambda: setattr(sandbox, "sandbox_id", None)
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
    assert sandbox.sandbox_id is None
    assert closed.json()["agent_observations"]["records"][-1]["sandbox_id"] == "resources-owned"


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


@pytest.mark.parametrize("mode", ["missing-span", "missing-count", "no-artifacts"])
def test_interruption_retains_the_same_evidence_gaps_as_a_response(setup, mode):
    agent, _ = setup

    async def scenario():
        state = await agent._seed_agent_session_state(seed())
        body = NeMoGymResponseCreateParamsNonStreaming(input="Fix")
        artifacts = CopilotArtifacts(
            events="\n".join(json.dumps([0.0, event]) for event in copilot_events()),
            otel=otel(chat_span("chatcmpl-1"), chat_span("chatcmpl-2", usage=None)),
            compaction_state=False,
        )
        if mode == "missing-span":
            artifacts.otel = otel(chat_span("chatcmpl-1"))
        elif mode == "no-artifacts":
            artifacts = CopilotArtifacts(events="")
        state.session.cleanup = {"return_code": None, "timed_out": True, "cleanup_confirmed": True, "error": None}
        agent._response(state, body, artifacts, [], timeout=1)
        normal = {gap.code for gap in state.observations.gaps}

        async def interrupted(**kwargs):
            state.session.artifacts = None if mode == "no-artifacts" else artifacts
            raise asyncio.CancelledError

        state.execute = interrupted
        with pytest.raises(asyncio.CancelledError):
            await agent._sandbox_response(state, body, prompt="Fix")
        return normal, {gap.code for gap in state.observations.gaps}

    normal, interrupted = asyncio.run(scenario())
    assert interrupted == normal | {"agent_activation_interrupted"}
    assert {
        "cached_token_usage_unavailable",
        "reasoning_token_usage_unavailable",
        "runtime_info_unavailable",
        "worker_exit_code_unavailable",
    } <= interrupted
    assert {
        "missing-span": "model_usage_spans_missing",
        "missing-count": "partial_model_usage_unavailable",
        "no-artifacts": "agent_transcript_unavailable",
    }[mode] in interrupted


def test_recovered_native_error_is_forwarded_on_success_retry_and_close(setup):
    agent, sandbox = setup
    sandbox.events.insert(4, {"type": "session.error", "data": {"errorType": "model_call", "message": "fetch failed"}})
    app = agent.setup_webserver()
    agent.setup_exception_middleware(app)
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).status_code == 200
        reports = []
        for _ in range(2):
            response = client.post("/ng-rollout/copilot-smoke-a2/v1/responses", json={"input": "Fix"})
            assert response.status_code == 200
            assert response.json()["status"] == "completed" and response.json()["error"] is None
            assert AGENT_OUTCOME_METADATA_KEY not in response.json()["metadata"]
            reports.append(FailureHeader.decode(response.headers))
        closed = client.post(
            "/v1/agent_sessions/close",
            json={"agent_session_id": seed().agent_session_id, "episode_id": seed().episode_id.model_dump()},
        )
        assert closed.status_code == 200
        reports.append(FailureHeader.decode(closed.headers))
    assert reports[0] is not None and all(report == reports[0] for report in reports)
    assert reports[0].failure_id is None
    (event,) = reports[0].events.values()
    assert (event.boundary, event.operation, event.failure_reason) == ("harness", "session.error", "fetch failed")
    assert event.failure_kind == "copilot:session_error" and event.http_status is None


@pytest.mark.parametrize("typed_cleanup", [True, False])
def test_connect_failure_survives_a_second_cleanup_failure(setup, typed_cleanup):
    agent, _ = setup
    original = NgFailureEvent(
        boundary="sandbox",
        failure_kind="provider_unavailable",
        failure_reason="connect failed",
        operation="connect",
        stage="seed",
        event_id="original-connect-event",
    )
    cleanup = (
        NgFailureEvent(
            boundary="sandbox",
            failure_kind="session_release_failed",
            failure_reason="close failed",
            operation="aclose",
            stage="cleanup",
            event_id="cleanup-event",
        )
        if typed_cleanup
        else RuntimeError("close failed")
    )
    provider = SimpleNamespace(aclose=AsyncMock(side_effect=cleanup))

    async def scenario():
        with (
            patch(f"{MODULE}.create_provider", return_value=provider),
            patch(f"{MODULE}.AsyncSandbox.connect", AsyncMock(side_effect=original)),
            RequestFailureLog.capture() as log,
        ):
            with pytest.raises(NgFailureEvent) as caught:
                await agent._seed_agent_session_state(seed())
            assert caught.value is original
            report = failure_report_for(caught.value)
            event = report.events[report.failure_id]
            assert (event.event_id, event.stage, event.operation) == ("original-connect-event", "seed", "connect")
            (cleanup_event,) = log.report.events.values()
            assert cleanup_event.stage == "cleanup" and cleanup_event.operation == "aclose"
            if typed_cleanup:
                assert cleanup_event.event_id == "cleanup-event"

    asyncio.run(scenario())
    provider.aclose.assert_awaited_once()


@pytest.mark.parametrize("spans", ["", "not-json", otel()])
def test_absent_chat_spans_do_not_invent_zero_usage(spans):
    assert parse_copilot_usage(spans) is None


@pytest.mark.parametrize("unreadable", ["[1, 2]", chat_span("chatcmpl-bg", usage=(500, 50))[:-20]])
def test_unreadable_telemetry_makes_usage_unknown(unreadable):
    usage = parse_copilot_usage(otel(chat_span("a", usage=(10, 3)), unreadable))
    assert usage.telemetry_malformed and usage.response_ids == ["a"]
    assert (usage.input_tokens, usage.output_tokens, usage.cached_tokens, usage.reasoning_tokens) == (None,) * 4


def test_unreadable_telemetry_keeps_session_usage_unknown(setup):
    agent, sandbox = setup
    # A span cut off mid-write may be a background call with no event; summing the rest would publish a subtotal.
    background = chat_span("chatcmpl-bg", usage=(500, 50))[:-20]
    sandbox.otel = otel(chat_span("chatcmpl-1", usage=(10, 3)), chat_span("chatcmpl-2", usage=(5, 2)), background)
    result, closed = run_session(agent)
    assert result.json()["status"] == "completed" and result.json()["usage"] is None
    observations = closed.json()["agent_observations"]
    assert [call["response_id"] for call in observations["records"][0]["model_calls"]] == ["chatcmpl-1", "chatcmpl-2"]
    gaps = {gap["code"] for gap in observations["gaps"]}
    assert "model_usage_telemetry_malformed" in gaps and "partial_model_usage_unavailable" not in gaps


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


@pytest.mark.parametrize("provider_config", [{"unknown_provider": {}}, {"local": 17}])
@pytest.mark.parametrize("borrowed", [True, False])
def test_invalid_provider_configuration_is_a_final_seed_refusal(setup, provider_config, borrowed):
    from nemo_gym.sandbox import create_provider

    agent, _ = setup
    agent.config.sandbox_provider = "sandbox"
    with (
        patch(f"{MODULE}.resolve_provider_config", return_value=provider_config),
        patch(f"{MODULE}.create_provider", create_provider),
        patch(f"{MODULE}.AsyncSandbox.connect", AsyncMock()) as connect,
    ):
        reason = check_agent_seed_rejected(agent.setup_webserver(), seed(sandbox_access=borrowed))
    assert "provider" in reason.lower()
    connect.assert_not_awaited()
    assert not agent._session_records


def test_unpublished_registry_version_is_a_final_seed_refusal(setup, monkeypatch):
    agent, sandbox = setup
    registry(monkeypatch, Reply({"error": "Not found"}, status=404))
    sandbox.metadata.side_effect = copilot_package_metadata

    reason = check_agent_seed_rejected(agent.setup_webserver(), seed())

    assert "is not published" in reason
    sandbox.disconnect.assert_awaited_once()
    sandbox.launch.assert_not_awaited()
    assert not agent._session_records


@pytest.mark.parametrize("settings", [{"metadata": 17}, {"unknown_setting": True}])
def test_invalid_owned_sandbox_spec_is_refused_before_provider_allocation(setup, settings):
    agent, _ = setup
    agent.config.sandbox_provider = "sandbox"
    agent.config.sandbox_config = settings
    with patch(f"{MODULE}.create_provider") as create:
        assert check_agent_seed_rejected(agent.setup_webserver(), seed(sandbox_access=False))
    create.assert_not_called()
    assert not agent._session_records


@pytest.mark.parametrize("allow_web_fetch", [False, True])
def test_web_fetch_is_denied_unless_explicitly_enabled(setup, allow_web_fetch):
    agent, sandbox = setup
    agent.config.allow_web_fetch = allow_web_fetch
    result, _ = run_session(agent)
    assert result.status_code == 200
    command = sandbox.payload["command"]
    assert ("--deny-tool" in command) is not allow_web_fetch
    if not allow_web_fetch:
        assert command[command.index("--deny-tool") + 1] == "web_fetch"


@pytest.mark.parametrize("field", ["copilot_binary", "copilot_binary_sha256", "copilot_dist_dir"])
def test_preinstalled_runtime_requires_all_pins(field):
    values = {"copilot_binary": "/opt/copilot", "copilot_binary_sha256": "a" * 64, "copilot_dist_dir": "/opt/dist"}
    del values[field]
    with pytest.raises(ValidationError, match="together"):
        agent_config(**values)


@pytest.mark.parametrize("bad_digest,wrong_version", [(False, False), (True, False), (False, True)])
def test_preinstalled_runtime_checks_bytes_without_registry_or_installer(setup, tmp_path, bad_digest, wrong_version):
    import hashlib

    agent, sandbox = setup
    binary = tmp_path / "copilot"
    marker = tmp_path / "executed"
    version = "1.0.104" if wrong_version else "1.0.103"
    binary.write_text(f"#!/bin/sh\ntouch {shlex.quote(str(marker))}\nprintf '{version}\\n'\n")
    binary.chmod(0o755)
    agent.config.copilot_binary = str(binary)
    agent.config.copilot_binary_sha256 = "0" * 64 if bad_digest else hashlib.sha256(binary.read_bytes()).hexdigest()
    agent.config.copilot_dist_dir = str(tmp_path)
    original = sandbox.execute

    async def execute(command, **kwargs):
        if "sha256sum -c" in command:
            completed = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
            return SimpleNamespace(
                return_code=completed.returncode, error_type=None, stdout=completed.stdout, stderr=completed.stderr
            )
        return await original(command, **kwargs)

    sandbox.exec.side_effect = execute
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        response = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert response.status_code == (500 if bad_digest or wrong_version else 200), response.text
    assert marker.exists() is not bad_digest
    sandbox.metadata.assert_not_awaited()
    assert not any(path.endswith("install_copilot_runtime.sh") for path in sandbox.files)
    if bad_digest or wrong_version:
        sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("owned", [False, True])
def test_restricted_network_requires_owned_preinstalled_runtime(setup, owned):
    agent, sandbox = setup
    agent.config.network_access = "model_only"
    agent.config.sandbox_provider = "sandbox"
    reason = check_agent_seed_rejected(agent.setup_webserver(), seed(sandbox_access=not owned))
    assert ("preinstalled" if owned else "agent-owned") in reason
    sandbox.exec.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("network_access", ["model_only", "model_and_tools"])
async def test_owned_network_policy_is_applied_before_start(setup, network_access):
    agent, sandbox = setup
    agent.config.network_access = network_access
    agent.config.sandbox_provider = "sandbox"
    agent.config.copilot_binary = "/opt/copilot"
    agent.config.copilot_binary_sha256 = "a" * 64
    agent.config.copilot_dist_dir = "/opt/dist"
    provider = SimpleNamespace(name="opensandbox", aclose=AsyncMock())
    with (
        patch(f"{MODULE}.create_provider", return_value=provider),
        patch(f"{MODULE}.AsyncSandbox", return_value=sandbox),
        patch.object(agent, "_install", AsyncMock()) as install,
    ):
        await agent._seed_agent_session_state(seed(sandbox_access=False))
    spec = install.await_args.args[1]
    assert spec.provider_options["network_policy"] == {
        "defaultAction": "deny",
        "egress": [{"action": "allow", "target": "model.example"}],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["unsupported", "conflict", "loopback"])
async def test_network_policy_refusal_closes_provider_before_allocation(setup, failure):
    agent, sandbox = setup
    agent.config.network_access = "model_only"
    agent.config.sandbox_provider = "sandbox"
    agent.config.copilot_binary = "/opt/copilot"
    agent.config.copilot_binary_sha256 = "a" * 64
    agent.config.copilot_dist_dir = "/opt/dist"
    provider = SimpleNamespace(name="docker" if failure == "unsupported" else "opensandbox", aclose=AsyncMock())
    if failure == "conflict":
        agent.config.sandbox_config = {"provider_options": {"network_policy": {"defaultAction": "deny"}}}
    if failure == "loopback":
        agent.server_client._build_server_base_url.return_value = "http://127.0.0.1:9000"
    with (
        patch(f"{MODULE}.create_provider", return_value=provider),
        patch(f"{MODULE}.AsyncSandbox") as allocate,
        pytest.raises(ValueError),
    ):
        await agent._seed_agent_session_state(seed(sandbox_access=False))
    allocate.assert_not_called()
    provider.aclose.assert_awaited_once()
