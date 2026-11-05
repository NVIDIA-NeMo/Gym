# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent sessions end to end: a real crew, a scripted Model Server and a borrowed local sandbox."""

import asyncio
import copy
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.cookies import SimpleCookie
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from nemo_gym.agents import AgentCloseSessionRequest, AgentSeedSessionRequest, check_agent_seed_rejected
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import EpisodeId, NeMoGymResponseCreateParamsNonStreaming, TaskId
from nemo_gym.sandbox import AsyncSandbox, DirectSandboxConnection, SandboxAccess, SandboxSpec
from nemo_gym.server_runtime import DirectHTTPToolAccess
from responses_api_agents.crewai_agent import app, sandbox_shell
from responses_api_agents.crewai_agent.app import CrewAIAgent, CrewAIAgentConfig
from responses_api_agents.crewai_agent.sandbox_shell import ShellRunnerError


_EPISODE = EpisodeId(rollout_id="rollout", attempt=0)
_TASK = "Write hello to out.txt. Keep the {braces} as they are."


class _Reply:
    """A Model Server response as the agent reads it from ``ServerClient.post``."""

    def __init__(self, body: dict[str, Any] | None = None, *, status: int = 200, cookies: dict | None = None) -> None:
        self.status = status
        self.ok = status < 400
        self._body = json.dumps(body or {"error": "unavailable"}).encode()
        self.content = SimpleNamespace(read=self.read)
        self.cookies = SimpleCookie(cookies or {})

    async def read(self) -> bytes:
        return self._body


def _completion(content: str | None = None, *calls: tuple[str, dict], id: str, usage: bool = True) -> _Reply:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {"id": f"{id}-call-{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(calls)
        ]
    body: dict[str, Any] = {"id": id, "choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}
    if usage:
        body["usage"] = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    return _Reply(body)


class _Model:
    """Scripted Model Server replies; a callable reply is awaited when its request arrives."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    async def post(self, *, server_name: str, url_path: str, json: dict, cookies=None, headers=None) -> _Reply:
        self.requests.append(
            {
                "server": server_name,
                "path": url_path,
                "json": copy.deepcopy(json),
                "headers": headers,
                # The agent passes one dict that it keeps updating: record what this call carried.
                "cookies": dict(cookies or {}),
            }
        )
        reply = self.replies.pop(0)
        return await reply() if callable(reply) else reply


def _agent(model: _Model, **config: Any) -> CrewAIAgent:
    server_client = MagicMock(spec=ServerClient)
    server_client.post = model.post
    # A crew that hangs fails its test in a minute, not at the default 30.
    config.setdefault("timeout", 60)
    return CrewAIAgent(
        config=CrewAIAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="crewai_agent",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            **config,
        ),
        server_client=server_client,
    )


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    """The borrowed sandbox's task directory, served by the local provider."""
    root = tmp_path / "task"
    root.mkdir()
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {"sandbox": {"local": {}}})
    monkeypatch.setattr(sandbox_shell, "SESSION_ROOT", str(tmp_path / "sessions"))

    # The local provider cannot be reattached, so connecting starts one in the task directory.
    async def connect(descriptor, *, provider):
        box = AsyncSandbox(provider)
        await box.start(SandboxSpec(workdir=str(root)))
        box._connected = True
        return box

    monkeypatch.setattr(AsyncSandbox, "connect", connect)
    return root


def _sandbox_access(root: Path, **fields: Any) -> SandboxAccess:
    return SandboxAccess(
        connection=DirectSandboxConnection(provider_config_ref="sandbox", descriptor={"id": "box"}),
        workdir=str(root),
        **fields,
    )


def _seed_body(**fields: Any) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="agent-session", episode_id=_EPISODE, task_id=TaskId(taskset="tb", task_id="t"), **fields
    )


def _seed(client: TestClient, **fields: Any):
    return client.post("/v1/agent_sessions", json=_seed_body(**fields).model_dump(mode="json"))


def _respond(client: TestClient, body: dict | None = None, rollout: str = _EPISODE.capture_key):
    return client.post(
        f"/ng-rollout/{rollout}/v1/responses", json=body or {"input": [{"role": "user", "content": _TASK}]}
    )


def _close(client: TestClient):
    body = AgentCloseSessionRequest(agent_session_id="agent-session", episode_id=_EPISODE)
    return client.post("/v1/agent_sessions/close", json=body.model_dump(mode="json"))


def test_crewai_sends_nothing_to_crewai_services():
    assert os.environ["CREWAI_DISABLE_TELEMETRY"] == "true"
    assert os.environ["CREWAI_TRACING_ENABLED"] == "false"
    assert os.environ["CREWAI_DISABLE_VERSION_CHECK"] == "true"


def test_an_episode_runs_one_crew_turn_with_its_shell_in_the_borrowed_sandbox(root, tmp_path):
    model = _Model(
        _completion(None, ("shell", {"command": "echo hello > out.txt; cat out.txt"}), id="chatcmpl-1"),
        _completion("Wrote hello; the {braces} stay.", id="chatcmpl-2"),
    )
    agent = _agent(model)
    with TestClient(agent.setup_webserver()) as client:
        seeded = _seed(client, sandbox_access=_sandbox_access(root))
        body = {"input": [{"role": "user", "content": _TASK}], "temperature": 0.0, "max_output_tokens": 256}
        response = _respond(client, body)
        closed = _close(client)

    assert seeded.status_code == 200
    assert response.status_code == 200, response.text
    assert (root / "out.txt").read_text() == "hello\n"
    first, second = model.requests
    assert first["path"] == f"/ng-rollout/{_EPISODE.capture_key}/v1/chat/completions"
    assert first["headers"] == {"x-session-id": _EPISODE.capture_key}
    assert (first["json"]["temperature"], first["json"]["max_tokens"]) == (0.0, 256)
    assert [tool["function"]["name"] for tool in first["json"]["tools"]] == ["shell"]
    assert _TASK in first["json"]["messages"][-1]["content"]
    # CrewAI's own message keys (cache_breakpoint) never reach the Model Server.
    assert all(
        set(message) <= {"role", "content", "tool_calls", "tool_call_id"} for message in second["json"]["messages"]
    )
    result = second["json"]["messages"][-1]
    assert result["role"] == "tool" and result["content"] == "hello\n\n[exit code: 0]"

    output = response.json()["output"]
    assert [item["type"] for item in output] == ["function_call", "function_call_output", "message"]
    assert output[0]["call_id"] == output[1]["call_id"] == "chatcmpl-1-call-0"
    assert output[2]["content"][0]["text"] == "Wrote hello; the {braces} stay."
    usage = response.json()["usage"]
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (20, 10, 30)

    assert closed.status_code == 200, closed.text
    closed = closed.json()
    assert closed["cleanup_confirmed"] is True
    invocation, tool, sandbox = closed["agent_observations"]["records"]
    assert (invocation["status"], invocation["stop_reason"]) == ("completed", None)
    assert [ref["response_id"] for ref in invocation["model_calls"]] == ["chatcmpl-1", "chatcmpl-2"]
    assert (tool["tool_call_id"], tool["tool_name"], tool["status"]) == ("chatcmpl-1-call-0", "shell", "completed")
    # The agent used a lent sandbox: it observed it, and left its teardown to the owner.
    assert (sandbox["kind"], sandbox["role"], sandbox["provider"]) == ("sandbox", "agent", "local")
    assert (sandbox["outcome"], sandbox["error_type"], sandbox["teardown_confirmed"]) == ("completed", None, None)
    assert {gap["code"] for gap in closed["agent_observations"]["gaps"]} == {
        "cached_token_usage_unavailable",
        "reasoning_token_usage_unavailable",
    }
    # Close removed the agent's files and left the task directory to its owner.
    assert list((tmp_path / "sessions").iterdir()) == []
    assert sorted(path.name for path in root.iterdir()) == ["out.txt"]


def test_a_failed_command_is_a_failed_tool_call_the_model_sees(root):
    model = _Model(
        _completion(None, ("shell", {"command": "echo missing >&2; exit 2"}), id="chatcmpl-1"),
        _completion("It failed.", id="chatcmpl-2"),
    )
    with TestClient(_agent(model).setup_webserver()) as client:
        _seed(client, sandbox_access=_sandbox_access(root))
        assert _respond(client).status_code == 200
        closed = _close(client).json()

    assert model.requests[1]["json"]["messages"][-1]["content"] == "missing\n\n[exit code: 2]"
    tool = closed["agent_observations"]["records"][1]
    assert (tool["status"], tool["error_type"]) == ("failed", "exit_code_2")


def test_an_errexit_from_the_tasks_bash_env_does_not_cost_a_command_its_status(root, tmp_path):
    bash_env = tmp_path / "bash_env.sh"
    bash_env.write_text("set -e\n")
    model = _Model(
        _completion(None, ("shell", {"command": "false"}), id="chatcmpl-1"),
        _completion("It failed.", id="chatcmpl-2"),
    )
    with TestClient(_agent(model).setup_webserver()) as client:
        _seed(client, sandbox_access=_sandbox_access(root, env={"BASH_ENV": str(bash_env)}))
        response = _respond(client)
        closed = _close(client).json()

    assert response.status_code == 200, response.text
    assert model.requests[1]["json"]["messages"][-1]["content"] == "[exit code: 1]"
    tool = closed["agent_observations"]["records"][1]
    assert (tool["status"], tool["error_type"]) == ("failed", "exit_code_1")


def test_a_large_tool_result_reaches_the_model_whole(root):
    # Without a context window CrewAI assumes 4,096 tokens, where its compaction would move this result to a file.
    model = _Model(
        _completion(None, ("shell", {"command": "head -c 10000 /dev/zero | tr '\\0' x"}), id="chatcmpl-1"),
        _completion("Done.", id="chatcmpl-2"),
    )
    with TestClient(_agent(model).setup_webserver()) as client:
        _seed(client, sandbox_access=_sandbox_access(root))
        assert _respond(client).status_code == 200

    second = model.requests[1]["json"]
    assert second["messages"][-1]["content"] == "x" * 10000 + "\n[exit code: 0]"
    # No tool that would read a moved result is offered.
    assert [tool["function"]["name"] for tool in second["tools"]] == ["shell"]


def test_each_tool_result_is_recorded_once_in_call_order(root):
    model = _Model(
        _completion(None, ("shell", {"command": "echo one"}), id="chatcmpl-1"),
        _completion(None, ("shell", {"command": "echo two"}), id="chatcmpl-2"),
        _completion("Done.", id="chatcmpl-3"),
    )
    with TestClient(_agent(model).setup_webserver()) as client:
        _seed(client, sandbox_access=_sandbox_access(root))
        response = _respond(client)
        closed = _close(client).json()

    # Each later model call carries every earlier tool result again; the transcript holds each one once.
    output = response.json()["output"]
    assert [(item["type"], item.get("call_id"), item.get("output")) for item in output] == [
        ("function_call", "chatcmpl-1-call-0", None),
        ("function_call_output", "chatcmpl-1-call-0", "one\n\n[exit code: 0]"),
        ("function_call", "chatcmpl-2-call-0", None),
        ("function_call_output", "chatcmpl-2-call-0", "two\n\n[exit code: 0]"),
        ("message", None, None),
    ]
    tools = [record for record in closed["agent_observations"]["records"] if record["kind"] == "tool_call"]
    assert [tool["tool_call_id"] for tool in tools] == ["chatcmpl-1-call-0", "chatcmpl-2-call-0"]


def test_a_call_crewai_refused_does_not_take_the_execution_of_a_later_call(root):
    model = _Model(
        # CrewAI answers a call with invalid arguments itself, without running the tool.
        _completion(None, ("shell", {"cmd": "echo bad"}), id="chatcmpl-1"),
        _completion(None, ("shell", {"command": "echo good"}), id="chatcmpl-2"),
        _completion("Done.", id="chatcmpl-3"),
    )
    with TestClient(_agent(model).setup_webserver()) as client:
        _seed(client, sandbox_access=_sandbox_access(root))
        assert _respond(client).status_code == 200
        closed = _close(client).json()

    tools = [record for record in closed["agent_observations"]["records"] if record["kind"] == "tool_call"]
    assert [(tool["tool_call_id"], tool["status"]) for tool in tools] == [("chatcmpl-2-call-0", "completed")]


def test_an_activation_runs_once_and_a_retry_gets_the_same_response(root):
    model = _Model(_completion("Done.", id="chatcmpl-1"))
    with TestClient(_agent(model).setup_webserver()) as client:
        _seed(client, sandbox_access=_sandbox_access(root))
        first = _respond(client)
        again = _respond(client)
        different = _respond(client, {"input": [{"role": "user", "content": "Something else."}]})

    assert first.status_code == again.status_code == 200
    assert first.json()["id"] == again.json()["id"]
    assert len(model.requests) == 1
    assert different.status_code == 409


async def test_a_waiter_that_goes_away_leaves_the_shared_run_to_finish(root):
    release = asyncio.Event()

    async def held() -> _Reply:
        await release.wait()
        return _completion("Done.", id="chatcmpl-1")

    model = _Model(held)
    agent = _agent(model)
    session: dict[str, Any] = {}
    await agent.seed_agent_session(
        SimpleNamespace(session=session, cookies={}), _seed_body(sandbox_access=_sandbox_access(root))
    )
    request = SimpleNamespace(
        session=session,
        cookies={},
        path_params={"rollout_id": _EPISODE.capture_key},
        url=SimpleNamespace(path=f"/ng-rollout/{_EPISODE.capture_key}/v1/responses"),
    )
    body = NeMoGymResponseCreateParamsNonStreaming(input=_TASK)

    first = asyncio.create_task(agent.responses(request, body))
    async with asyncio.timeout(15):
        while not model.requests:
            await asyncio.sleep(0.05)
    # The HTTP client went away: its handler is cancelled, the crew's run is not.
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()
    again = await asyncio.wait_for(agent.responses(request, body), 15)

    assert again.status == "completed" and len(model.requests) == 1
    close = AgentCloseSessionRequest(agent_session_id="agent-session", episode_id=_EPISODE)
    closed = await asyncio.wait_for(agent.close_agent_session(request, close), 15)
    assert closed.agent_observations.records[0].status == "completed"


def test_activations_outside_the_seeded_session_are_refused(root):
    with TestClient(_agent(_Model()).setup_webserver()) as client:
        assert _respond(client).status_code == 409
        _seed(client, sandbox_access=_sandbox_access(root))
        assert _respond(client, rollout="another-rollout").status_code == 409
        assert _close(client).status_code == 200
        assert _respond(client).status_code == 409
    with TestClient(_agent(_Model()).setup_webserver()) as client:
        assert client.post("/run", json={"responses_create_params": {"input": "hi"}}).status_code == 409


@pytest.mark.parametrize(
    "body, reason",
    [
        (
            {"input": "hi", "tools": [{"type": "function", "name": "f", "parameters": {}, "strict": False}]},
            "only through an MCP grant",
        ),
        ({"input": "hi", "reasoning": {"effort": "high"}}, "request field reasoning"),
        (
            {
                "input": [
                    {"role": "user", "content": [{"type": "input_image", "image_url": "data:,", "detail": "auto"}]}
                ]
            },
            "only text",
        ),
        ({"input": [{"role": "system", "content": "Be brief."}]}, "needs a user message"),
    ],
)
def test_a_request_the_crew_cannot_apply_is_refused_before_it_binds(root, body, reason):
    model = _Model(_completion("Done.", id="chatcmpl-1"))
    with TestClient(_agent(model).setup_webserver()) as client:
        _seed(client, sandbox_access=_sandbox_access(root))
        refused = _respond(client, body)
        accepted = _respond(client)

    assert refused.status_code == 422 and reason in refused.json()["detail"]
    assert accepted.status_code == 200


def test_system_messages_and_instructions_join_the_agents_system_prompt(root):
    model = _Model(_completion("Done.", id="chatcmpl-1"))
    body = {
        "instructions": "Answer in French.",
        "input": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": _TASK}],
    }
    with TestClient(_agent(model).setup_webserver()) as client:
        _seed(client)
        assert _respond(client, body).status_code == 200

    system, user = model.requests[0]["json"]["messages"]
    assert "Answer in French." in system["content"] and "Be brief." in system["content"]
    assert _TASK in user["content"] and "Be brief." not in user["content"]
    # Without a sandbox or grants the crew has no tools.
    assert "tools" not in model.requests[0]["json"]


def test_parallel_tool_calls_and_top_p_reach_the_model_with_the_tools(root):
    model = _Model(_completion("Done.", id="chatcmpl-1"))
    body = {"input": [{"role": "user", "content": _TASK}], "parallel_tool_calls": False, "top_p": 0.5}
    with TestClient(_agent(model).setup_webserver()) as client:
        _seed(client, sandbox_access=_sandbox_access(root))
        assert _respond(client, body).status_code == 200

    sent = model.requests[0]["json"]
    assert (sent["parallel_tool_calls"], sent["top_p"], len(sent["tools"])) == (False, 0.5, 1)


@pytest.mark.parametrize(
    "seed, reason",
    [
        (
            {
                "tool_accesses": [
                    DirectHTTPToolAccess(name="resources.direct_http", required=True, base_url="http://resources:1")
                ]
            },
            "only MCP tool grants",
        ),
        ({"sandbox_access": _sandbox_access(Path("task"))}, "absolute sandbox workdir"),
        ({"sandbox_access": _sandbox_access(Path("/task"), user="agent")}, "SandboxAccess user"),
    ],
)
def test_a_seed_the_crew_cannot_serve_is_refused(seed, reason):
    detail = check_agent_seed_rejected(_agent(_Model()).setup_webserver(), _seed_body(**seed))

    assert reason in detail


def test_an_optional_direct_http_grant_is_skipped(root):
    grant = DirectHTTPToolAccess(name="resources.direct_http", required=False, base_url="http://resources:1")
    with TestClient(_agent(_Model()).setup_webserver()) as client:
        assert _seed(client, tool_accesses=[grant]).status_code == 200


def test_close_before_an_activation_reports_the_missing_evidence_and_releases_the_sandbox(root, tmp_path):
    with TestClient(_agent(_Model()).setup_webserver()) as client:
        _seed(client, sandbox_access=_sandbox_access(root))
        closed = _close(client)

    assert closed.status_code == 200
    bundle = closed.json()["agent_observations"]
    assert [(record["kind"], record["outcome"]) for record in bundle["records"]] == [("sandbox", "cancelled")]
    assert [gap["code"] for gap in bundle["gaps"]] == ["observation_capture_failed"]
    assert list((tmp_path / "sessions").iterdir()) == []


# Killing a process group is reliable on Linux; macOS's bash 3.2 sometimes leaves a subshell of a killed group running.
@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-group kills")
def test_close_stops_a_running_crew_and_kills_its_command(root, tmp_path):
    model = _Model(_completion(None, ("shell", {"command": "sleep 3; touch late.txt"}), id="chatcmpl-1"))
    activation: dict[str, Any] = {}
    with TestClient(_agent(model).setup_webserver(), raise_server_exceptions=False) as client:
        _seed(client, sandbox_access=_sandbox_access(root))
        thread = threading.Thread(target=lambda: activation.update(response=_respond(client)))
        thread.start()
        deadline = time.monotonic() + 15
        while not list((tmp_path / "sessions").glob("*/*.pid")):
            assert time.monotonic() < deadline, "the command never started"
            time.sleep(0.05)
        closed = _close(client)
        thread.join(15)

    assert not thread.is_alive()
    assert closed.status_code == 200, closed.text
    bundle = closed.json()["agent_observations"]
    assert bundle["records"][0]["status"] == "incomplete"
    assert bundle["records"][-1]["outcome"] == "cancelled"
    assert "agent_activation_interrupted" in {gap["code"] for gap in bundle["gaps"]}
    time.sleep(3.5)
    assert not (root / "late.txt").exists()
    # The crew made no model call after close.
    assert len(model.requests) == 1


def test_a_crew_that_runs_out_of_time_answers_incomplete(root):
    async def slow() -> _Reply:
        await asyncio.sleep(30)
        raise AssertionError("the deadline should have cancelled this call")

    model = _Model(_completion(None, ("shell", {"command": "echo partial > partial.txt"}), id="chatcmpl-1"), slow)
    with TestClient(_agent(model, timeout=3).setup_webserver()) as client:
        _seed(client, sandbox_access=_sandbox_access(root))
        started = time.monotonic()
        response = _respond(client)
        closed = _close(client).json()

    assert time.monotonic() - started < 15
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "incomplete"
    assert response.json()["metadata"]["crewai_stop_reason"] == "timeout"
    assert (root / "partial.txt").read_text() == "partial\n"
    assert "agent_run_timeout" in {gap["code"] for gap in closed["agent_observations"]["gaps"]}
    invocation = closed["agent_observations"]["records"][0]
    assert (invocation["status"], invocation["stop_reason"]) == ("incomplete", "timeout")
    assert closed["agent_observations"]["records"][-1]["outcome"] == "timeout"


def test_a_deadline_during_a_command_answers_incomplete_with_the_call_cancelled(root):
    model = _Model(_completion(None, ("shell", {"command": "echo partial > partial.txt; sleep 20"}), id="chatcmpl-1"))
    with TestClient(_agent(model, timeout=3).setup_webserver()) as client:
        _seed(client, sandbox_access=_sandbox_access(root))
        response = _respond(client)
        closed = _close(client).json()

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "incomplete"
    assert (root / "partial.txt").read_text() == "partial\n"
    invocation, tool, sandbox = closed["agent_observations"]["records"]
    assert (invocation["status"], invocation["stop_reason"]) == ("incomplete", "timeout")
    assert (tool["tool_call_id"], tool["status"], tool["error_type"]) == (
        "chatcmpl-1-call-0",
        "cancelled",
        "cancelled",
    )
    assert tool["completed_at"] >= tool["started_at"]
    assert sandbox["outcome"] == "timeout"


@pytest.mark.parametrize(
    "seed, body",
    [
        ({"agent_timeout_sec": 2}, {}),
        ({}, {"metadata": {"agent_timeout_sec": "2"}}),
    ],
    ids=["seed", "request_metadata"],
)
def test_a_tasks_agent_timeout_lowers_the_crews_deadline(root, seed, body):
    async def slow() -> _Reply:
        await asyncio.sleep(30)
        raise AssertionError("the task's agent_timeout_sec should have ended the run")

    model = _Model(slow)
    with TestClient(_agent(model, timeout=1800).setup_webserver()) as client:
        _seed(client, sandbox_access=_sandbox_access(root), **seed)
        started = time.monotonic()
        response = _respond(client, {"input": [{"role": "user", "content": _TASK}], **body})

    assert time.monotonic() - started < 15
    assert response.json()["status"] == "incomplete"


def test_a_final_answer_cut_off_at_the_output_cap_is_incomplete(root):
    body = json.loads(_completion("Partial answ", id="chatcmpl-1")._body)
    body["choices"][0]["finish_reason"] = "length"
    with TestClient(_agent(_Model(_Reply(body))).setup_webserver()) as client:
        _seed(client, sandbox_access=_sandbox_access(root))
        response = _respond(client)
        closed = _close(client).json()

    assert response.json()["status"] == "incomplete"
    assert response.json()["incomplete_details"] == {"reason": "max_output_tokens"}
    # The invocation agrees with the response; the crew's use of the sandbox ended normally.
    invocation, sandbox = closed["agent_observations"]["records"]
    assert (invocation["status"], invocation["stop_reason"], sandbox["outcome"]) == ("incomplete", None, "completed")


def test_a_rejected_model_request_fails_the_activation_with_its_status(root):
    model = _Model(_Reply(status=400))
    with TestClient(_agent(model).setup_webserver()) as client:
        _seed(client, sandbox_access=_sandbox_access(root))
        response = _respond(client)
        closed = _close(client).json()

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    # CrewAI does not run the task again after the failed call.
    assert len(model.requests) == 1
    assert json.loads(response.json()["metadata"]["ng_agent_outcome"]) == {
        "failure_kind": "model_request_error",
        "status_code": 400,
    }
    invocation, sandbox = closed["agent_observations"]["records"]
    assert invocation["status"] == "failed"
    assert (sandbox["outcome"], sandbox["error_type"]) == ("failed", invocation["error_type"])


def test_a_rejected_model_request_after_a_command_does_not_run_the_task_again(root):
    model = _Model(
        _completion(None, ("shell", {"command": "echo run >> runs.txt"}), id="chatcmpl-1"),
        _Reply(status=400),
        # What a second attempt at the task would get.
        _completion(None, ("shell", {"command": "echo run >> runs.txt"}), id="chatcmpl-2"),
        _completion("Done.", id="chatcmpl-3"),
    )
    with TestClient(_agent(model).setup_webserver()) as client:
        _seed(client, sandbox_access=_sandbox_access(root))
        response = _respond(client)

    assert response.json()["status"] == "failed"
    assert len(model.requests) == 2
    assert (root / "runs.txt").read_text() == "run\n"


def test_an_empty_answer_is_the_models_invalid_output(root):
    model = _Model(*(_completion("", id=f"chatcmpl-{i}") for i in range(6)))
    with TestClient(_agent(model).setup_webserver()) as client:
        _seed(client, sandbox_access=_sandbox_access(root))
        response = _respond(client)

    assert response.json()["status"] == "failed"
    assert json.loads(response.json()["metadata"]["ng_agent_outcome"]) == {"failure_kind": "agent_invalid_output"}


def test_a_model_request_that_times_out_fails_the_activation_instead_of_ending_it_early(root):
    async def timed_out() -> _Reply:
        raise TimeoutError("model request timed out")

    model = _Model(timed_out)
    with TestClient(_agent(model).setup_webserver()) as client:
        _seed(client, sandbox_access=_sandbox_access(root))
        response = _respond(client)

    # Not the crew's own deadline, which would grade the partial work as incomplete.
    assert response.json()["status"] == "failed"
    assert json.loads(response.json()["metadata"]["ng_agent_outcome"]) == {"failure_kind": "model_transport_error"}


def test_a_transient_model_status_is_retried(root):
    model = _Model(_Reply(status=503), _completion("Done.", id="chatcmpl-1"))
    with TestClient(_agent(model).setup_webserver()) as client:
        _seed(client, sandbox_access=_sandbox_access(root))
        response = _respond(client)
        closed = _close(client).json()

    assert response.json()["status"] == "completed"
    assert len(model.requests) == 2
    # Both attempts carried the session header. Close sends no trajectory of its own, so rollout collection joins the
    # failed attempt, which has no response ID, to the invocation by that header.
    assert all(request["headers"] == {"x-session-id": _EPISODE.capture_key} for request in model.requests)
    assert "trajectory" not in closed


def test_model_server_cookies_are_sent_back_on_later_calls(root):
    first = _completion(None, ("shell", {"command": "true"}), id="chatcmpl-1")
    first.cookies.load({"route": "backend-3"})
    model = _Model(first, _completion("Done.", id="chatcmpl-2"))
    with TestClient(_agent(model).setup_webserver()) as client:
        _seed(client, sandbox_access=_sandbox_access(root))
        assert _respond(client).status_code == 200

    assert [request["cookies"] for request in model.requests] == [{}, {"route": "backend-3"}]


def test_usage_is_unknown_when_a_model_call_reports_none(root):
    model = _Model(
        _completion(None, ("shell", {"command": "true"}), id="chatcmpl-1"),
        _completion("Done.", id="chatcmpl-2", usage=False),
    )
    with TestClient(_agent(model).setup_webserver()) as client:
        _seed(client, sandbox_access=_sandbox_access(root))
        response = _respond(client)
        closed = _close(client).json()

    assert response.json()["usage"] is None
    assert "model_call_usage_unavailable" in {gap["code"] for gap in closed["agent_observations"]["gaps"]}


def test_a_model_reply_without_an_id_is_a_reference_gap(root):
    body = json.loads(_completion("Done.", id="chatcmpl-1")._body)
    del body["id"]
    with TestClient(_agent(_Model(_Reply(body))).setup_webserver()) as client:
        _seed(client, sandbox_access=_sandbox_access(root))
        assert _respond(client).status_code == 200
        closed = _close(client).json()

    assert closed["agent_observations"]["records"][0]["model_calls"] == []
    assert "model_call_reference_unavailable" in {gap["code"] for gap in closed["agent_observations"]["gaps"]}


def test_a_sandbox_that_fails_under_the_crew_fails_the_episode_instead_of_grading_it(root, monkeypatch):
    async def lost(self, command):
        raise ShellRunnerError("sandbox exec ended without a command status")

    monkeypatch.setattr(sandbox_shell.SandboxShell, "run", lost)
    model = _Model(
        _completion(None, ("shell", {"command": "ls"}), id="chatcmpl-1"),
        _completion("Done anyway.", id="chatcmpl-2"),
    )
    with TestClient(_agent(model).setup_webserver()) as client:
        _seed(client, sandbox_access=_sandbox_access(root))
        response = _respond(client)
        closed = _close(client)

    assert response.status_code == 503 and "infrastructure failed" in response.json()["detail"]
    # The crew stopped instead of showing the model the failure.
    assert len(model.requests) == 1
    assert closed.status_code == 200
    invocation, tool, sandbox = closed.json()["agent_observations"]["records"]
    assert (invocation["status"], invocation["error_type"]) == ("failed", "infrastructure_error")
    assert tool["error_type"] == "infrastructure_error"
    assert (sandbox["outcome"], sandbox["error_type"]) == ("failed", "infrastructure_error")


async def test_crews_beyond_the_default_executors_size_all_finish(root, monkeypatch):
    # On a host with one CPU the default executor has 5 threads, fewer than the 8 crews here.
    monkeypatch.setattr(app.os, "process_cpu_count", lambda: 1)
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(5))
    first_calls = 0
    five_in_flight = asyncio.Event()

    class ShellThenDone:
        async def post(self, *, server_name, url_path, json, cookies=None, headers=None) -> _Reply:
            nonlocal first_calls
            if any(message.get("role") == "tool" for message in json["messages"]):
                return _completion("Done.", id="chatcmpl-2")
            # Five crews in a model call at once, as under load: each holds a thread of the executor.
            first_calls += 1
            if first_calls == 5:
                five_in_flight.set()
            await five_in_flight.wait()
            # Then each call needs the executor itself, as aiohttp's default resolver does for the server's address.
            await loop.getaddrinfo("localhost", 80)
            return _completion(None, ("shell", {"command": "echo hi"}), id="chatcmpl-1")

    # A crew that cannot get a thread answers incomplete at this deadline.
    asgi = _agent(ShellThenDone(), concurrency=16, timeout=20).setup_webserver()

    async def episode(i: int) -> list[Any]:
        episode_id = EpisodeId(rollout_id=f"rollout-{i}", attempt=0)
        seed = AgentSeedSessionRequest(
            agent_session_id=f"agent-session-{i}",
            episode_id=episode_id,
            task_id=TaskId(taskset="tb", task_id="t"),
            sandbox_access=_sandbox_access(root),
        )
        close = AgentCloseSessionRequest(agent_session_id=seed.agent_session_id, episode_id=episode_id)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=asgi), base_url="http://agent") as client:
            seeded = await client.post("/v1/agent_sessions", json=seed.model_dump(mode="json"))
            response = await client.post(
                f"/ng-rollout/{episode_id.capture_key}/v1/responses", json={"input": "Say hi."}
            )
            closed = await client.post("/v1/agent_sessions/close", json=close.model_dump(mode="json"))
            return [seeded.status_code, response.json().get("status"), closed.status_code]

    outcomes = await asyncio.wait_for(asyncio.gather(*(episode(i) for i in range(8))), 60)

    assert outcomes == [[200, "completed", 200]] * 8


@pytest.fixture
def owned(tmp_path, monkeypatch) -> dict[str, Any]:
    """A sandbox the agent starts itself on the local provider, in a task directory it does not delete."""
    workdir = tmp_path / "owned"
    workdir.mkdir()
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {"sandbox": {"local": {}}})
    monkeypatch.setattr(sandbox_shell, "SESSION_ROOT", str(tmp_path / "sessions"))
    started: list[SandboxSpec] = []
    stopped: list[AsyncSandbox] = []
    start, stop = AsyncSandbox.start, AsyncSandbox.stop

    async def recording_start(self, spec=None, **options):
        started.append(spec)
        return await start(self, spec, **options)

    async def recording_stop(self):
        stopped.append(self)
        await stop(self)

    monkeypatch.setattr(AsyncSandbox, "start", recording_start)
    monkeypatch.setattr(AsyncSandbox, "stop", recording_stop)
    return {"workdir": workdir, "started": started, "stopped": stopped}


def test_without_a_lent_sandbox_the_agent_runs_its_shell_in_its_own_and_stops_it(owned, tmp_path):
    model = _Model(
        _completion(None, ("shell", {"command": "pwd > where.txt"}), id="chatcmpl-1"),
        _completion("Done.", id="chatcmpl-2"),
    )
    agent = _agent(model, sandbox_provider="sandbox", sandbox_config={"workdir": str(owned["workdir"])}, timeout=300)
    with TestClient(agent.setup_webserver()) as client:
        assert _seed(client).status_code == 200
        response = _respond(client)
        closed = _close(client).json()

    assert response.status_code == 200, response.text
    assert (owned["workdir"] / "where.txt").read_text().strip() == str(owned["workdir"])
    (spec,) = owned["started"]
    assert spec.ttl_s == 900
    assert spec.metadata["nemo_gym_agent"] == "crewai_agent" and spec.metadata["agent_session"] == "agent-session"
    assert len(owned["stopped"]) == 1
    sandbox = closed["agent_observations"]["records"][-1]
    assert (sandbox["role"], sandbox["outcome"], sandbox["teardown_confirmed"]) == ("agent", "completed", True)
    assert list((tmp_path / "sessions").iterdir()) == []


def test_a_sandbox_whose_setup_fails_is_stopped_and_fails_the_seed(owned, monkeypatch):
    async def broken(self):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(sandbox_shell.SandboxShell, "setup", broken)
    agent = _agent(_Model(), sandbox_provider="sandbox", sandbox_config={"workdir": str(owned["workdir"])})
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        seeded = _seed(client)

    assert seeded.status_code == 500
    assert len(owned["started"]) == 1 and owned["stopped"]


def test_without_a_lent_sandbox_or_a_provider_the_crew_has_no_shell():
    model = _Model(_completion("Done.", id="chatcmpl-1"))
    with TestClient(_agent(model).setup_webserver()) as client:
        _seed(client)
        assert _respond(client).status_code == 200

    assert "tools" not in model.requests[0]["json"]
