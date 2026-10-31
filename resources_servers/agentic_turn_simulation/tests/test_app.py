# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The server's endpoints on sandbox rows with scripted model servers: sessions, the /harness endpoint, tool
simulation (in code and by the LLM, with its retries), verification (masks, code gates, the judge and its
retries) and metrics."""

import asyncio
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from fastapi.testclient import TestClient

from nemo_gym import failure_kinds
from nemo_gym.openai_utils import (
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.reward_profile import MASK_SAMPLE_FIELD
from resources_servers.agentic_turn_simulation.app import (
    AgenticTurnSimulationResourcesServer,
    AgenticTurnSimulationResourcesServerConfig,
    AgenticTurnSimulationRunRequest,
    AgenticTurnSimulationVerifyRequest,
    SimulateToolCallRequest,
    _Session,
)
from resources_servers.agentic_turn_simulation.harnesses import HARNESSES
from resources_servers.agentic_turn_simulation.harnesses.sandbox import (
    HARNESS_ERROR,
    IMAGE_READ,
    KNOWN_TOOLS,
    SANDBOX,
    SandboxWorkspace,
    file_hash,
)
from resources_servers.agentic_turn_simulation.items import UNPARSEABLE_ARGS_KEY
from resources_servers.agentic_turn_simulation.judge import evaluation_schema, load_prompt, parse_verdict
from resources_servers.agentic_turn_simulation.simulator import SIMULATOR_FAILED, simulator_failure
from resources_servers.agentic_turn_simulation.tests.support import (
    SERVER,
    VERDICT,
    chat,
    http,
    judge_reply,
    make_server,
    req,
    script,
)


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "agentic_turn_simulation.yaml"
# Rows select the sandbox harness by name.
SANDBOX_METADATA = {"harness": "sandbox"}

GOOD = dict(VERDICT)
BAD = dict(GOOD, correct="no", blocked_by="model")
BASH_OK = '{"duration_ms": 5, "exit_code": 0, "status": "completed", "stderr": "", "stdout": "ok"}'

TOOL_NAMES = ["bash", "read_file", "write_file", "edit_file", "present_file", "generate_image", "ask_user"]
TOOLS = [
    {"type": "function", "name": n, "description": f"the {n} tool", "parameters": {"type": "object"}, "strict": False}
    for n in TOOL_NAMES
]
# A bash tool whose schema requires `command`, types its arguments and forbids any other key.
BASH_SCHEMA = {
    "type": "object",
    "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
    "required": ["command"],
    "additionalProperties": False,
}
STRICT_BASH = {"type": "function", "name": "bash", "description": "run", "parameters": BASH_SCHEMA, "strict": False}
HISTORY = [
    {"type": "message", "role": "system", "content": "You are the sandbox agent."},
    {"type": "message", "role": "user", "content": "Build a page in /home/user/site, use my logo.png"},
    {
        "type": "function_call",
        "call_id": "h1",
        "name": "write_file",
        "arguments": json.dumps({"path": "/home/user/site/index.html", "content": "<h1>Hi</h1>"}),
    },
    {"type": "function_call_output", "call_id": "h1", "output": json.dumps({"hash": "x", "status": "success"})},
    {"type": "message", "role": "assistant", "content": "Done: /home/user/site/index.html"},
    {"type": "message", "role": "user", "content": "Make the heading blue."},
]


def _rcp(input_: Any = None, tools: list[dict[str, Any]] | None = None) -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(
        input=HISTORY if input_ is None else input_, tools=TOOLS if tools is None else tools
    )


async def _seed(server: AgenticTurnSimulationResourcesServer, session_id: str = "s1", **kw: Any) -> _Session:
    await server.seed_session(
        req(session_id),
        AgenticTurnSimulationRunRequest(responses_create_params=_rcp(**kw), verifier_metadata=SANDBOX_METADATA),
    )
    return server._sessions[session_id]


def _sim(name: str, call_id: str = "c1", turn: list[dict[str, Any]] | None = None, **args: Any):
    return SimulateToolCallRequest(call_id=call_id, name=name, arguments=json.dumps(args), turn=turn or [])


def _call(name: str, call_id: str = "c1", **args: Any) -> dict[str, Any]:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(args)}


def _msg(text: str, role: str = "assistant") -> dict[str, Any]:
    return {"type": "message", "role": role, "content": text}


def _out(call_id: str, output: str) -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _message_item(text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(id="m", content=[NeMoGymResponseOutputText(text=text, annotations=[])])


def _call_item(name: str, arguments: str = "{}", call_id: str = "p1") -> NeMoGymResponseFunctionToolCall:
    return NeMoGymResponseFunctionToolCall(call_id=call_id, name=name, arguments=arguments)


def _output_item(call_id: str, output: str) -> NeMoGymFunctionCallOutput:
    return NeMoGymFunctionCallOutput(call_id=call_id, output=output)


def _policy_turn() -> list[Any]:
    return [
        NeMoGymResponseReasoningItem(id="r", summary=[{"type": "summary_text", "text": "SECRET_REASONING"}]),
        _call_item("bash", json.dumps({"command": "sed -n 1p /home/user/site/index.html"}), "p1"),
        _output_item("p1", BASH_OK),
        _call_item("bash", json.dumps({"command": "make"}), "p2"),
        _output_item("p2", json.dumps({"status": "completed", "exit_code": 2, "stdout": ""})),
        _message_item("Made the heading blue in /home/user/site/index.html."),
    ]


def _verify_body(
    items: list[Any] | None = None,
    *,
    stop_reason: str | None = "final_message",
    steps: str | None = "3",
    metadata: dict[str, Any] | None = None,
    response_metadata: dict[str, str] | None = None,
    input_: Any = None,
    tools: list[dict[str, Any]] | None = None,
) -> AgenticTurnSimulationVerifyRequest:
    if response_metadata is None:
        response_metadata = {k: v for k, v in (("stop_reason", stop_reason), ("steps", steps)) if v is not None}
    response = NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="policy",
        object="response",
        output=_policy_turn() if items is None else items,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        metadata=response_metadata or None,
    )
    return AgenticTurnSimulationVerifyRequest(
        responses_create_params=_rcp(input_, tools),
        response=response,
        verifier_metadata={**SANDBOX_METADATA, **({"categories": ["web_dev"]} if metadata is None else metadata)},
    )


# --- server setup and sessions ------------------------------------------------------------------------------


class TestSetup:
    def test_concurrency_limits_and_simulator_prompt(self) -> None:
        server = make_server()
        assert isinstance(server._simulator_semaphore, asyncio.Semaphore)
        assert server._simulator_semaphore._value == 64 and server._judge_semaphore._value == 32
        assert server._simulator_prompts["generic"] == load_prompt("tool-simulator-generic.md")  # the default's
        unlimited = make_server(tool_simulator_max_concurrency=None, judge_max_concurrency=0)
        assert isinstance(unlimited._simulator_semaphore, nullcontext)
        assert isinstance(unlimited._judge_semaphore, nullcontext)

    def test_shipped_config_validates(self) -> None:
        raw = yaml.safe_load(CONFIG_PATH.read_text())
        section = raw[f"{SERVER}_resources_server"]["resources_servers"][SERVER]
        config = AgenticTurnSimulationResourcesServerConfig(host="0.0.0.0", port=0, name=SERVER, **section)
        assert config.score_source == "policy" and len(config.judges) == 1
        assert config.tool_simulator_model_server.name == config.judges[0].model_server.name
        assert config.tool_simulator_chat_completions_create_params.max_tokens == 16384
        assert config.judge_chat_completions_create_params.reasoning_effort == "high"
        assert config.masked_stop_reasons == ["time_cap"]
        agent = raw[f"{SERVER}_agent"]["responses_api_agents"][f"{SERVER}_agent"]
        assert agent["resources_server"]["name"] == f"{SERVER}_resources_server"
        assert agent["stop_tools"] == ["ask_user"]

    def test_http_session_links_seed_and_tool_calls(self) -> None:
        server = make_server()
        client = TestClient(server.setup_webserver())
        seed = AgenticTurnSimulationRunRequest(
            responses_create_params=_rcp(), verifier_metadata=SANDBOX_METADATA
        ).model_dump(mode="json", exclude_none=True)
        assert client.post("/seed_session", json=seed).status_code == 200
        call = {"call_id": "c1", "name": "read_file", "arguments": json.dumps({"path": "site/index.html"})}
        reply = client.post("/simulate_tool_call", json=call)
        assert reply.status_code == 200 and reply.json()["source"] == "exact"
        assert json.loads(reply.json()["output"])["content"] == "<h1>Hi</h1>"
        client.cookies.clear()  # a new rollout without a seeded session
        assert client.post("/simulate_tool_call", json=call).json()["source"] == "failed"


class TestHarnessEndpoint:
    """/harness tells the agent the seeded row's stop tools and tool-call protocol."""

    def test_the_seeded_sessions_harness(self) -> None:
        client = TestClient(make_server().setup_webserver())
        seed = AgenticTurnSimulationRunRequest(
            responses_create_params=_rcp(), verifier_metadata=SANDBOX_METADATA
        ).model_dump(mode="json", exclude_none=True)
        assert client.post("/seed_session", json=seed).json() == {}  # the harness is /harness's answer alone
        reply = client.post("/harness")
        assert reply.status_code == 200 and reply.json() == {"stop_tools": ["ask_user"], "protocol": "native"}

    def test_a_request_without_a_seeded_session_is_not_found(self) -> None:
        reply = TestClient(make_server().setup_webserver()).post("/harness")
        assert reply.status_code == 404 and "/seed_session" in reply.json()["detail"]


class TestSeedSession:
    async def test_stores_history_and_replays_the_workspace(self) -> None:
        server = make_server()
        state = await _seed(server)
        assert [i.get("role") or i["type"] for i in state.prefix] == [
            "system",
            "user",
            "function_call",
            "function_call_output",
            "assistant",
            "user",
        ]
        assert state.tool_names == frozenset(TOOL_NAMES)
        assert state.workspace.files == {"/home/user/site/index.html": "<h1>Hi</h1>"}
        assert "/home/user/site" in json.dumps(state.prefix)
        assert not state.answered_by

    async def test_string_input_and_no_tools(self) -> None:
        server = make_server()
        state = await _seed(server, input_="hello there", tools=[])
        assert state.prefix == [{"type": "message", "role": "user", "content": "hello there"}]
        assert state.tool_names == KNOWN_TOOLS

    async def test_stale_sessions_are_evicted(self) -> None:
        server = make_server(session_ttl_seconds=60)
        server._sessions["old"] = _Session(
            SANDBOX, [], KNOWN_TOOLS, SandboxWorkspace(), created=time.monotonic() - 3600
        )
        server._sessions["recent"] = _Session(SANDBOX, [], KNOWN_TOOLS, SandboxWorkspace())
        await _seed(server, "new")
        assert set(server._sessions) == {"recent", "new"}


# --- /simulate_tool_call ------------------------------------------------------------------------------------


class TestSimulateExact:
    async def test_session_not_found(self) -> None:
        server = make_server()
        for request in (req("missing"), req(None)):
            out = await server.simulate_tool_call(request, _sim("bash", command="ls"))
            assert out.source == "failed"
            assert out.output == simulator_failure("session not found")
            assert SIMULATOR_FAILED in out.output

    async def test_write_then_read_answered_in_code(self) -> None:
        server = make_server()
        state = await _seed(server)
        turn = [_call("write_file", "c1", path="site/app.js", content="let x = 1;")]
        out = await server.simulate_tool_call(
            req(), _sim("write_file", "c1", turn, path="site/app.js", content="let x = 1;")
        )
        assert out.source == "exact"
        assert json.loads(out.output) == {"hash": file_hash("let x = 1;"), "status": "success"}
        out = await server.simulate_tool_call(req(), _sim("read_file", "c2", path="/home/user/site/app.js"))
        assert out.source == "exact" and json.loads(out.output)["content"] == "let x = 1;"
        assert state.answered_by == {"exact": 2}
        server.server_client.post.assert_not_called()

    async def test_a_shell_read_back_of_a_known_file_is_answered_in_code(self) -> None:
        server = make_server()
        state = await _seed(server)
        out = await server.simulate_tool_call(req(), _sim("bash", command="cat site/index.html"))
        assert out.source == "exact" and json.loads(out.output)["stdout"] == "<h1>Hi</h1>"
        sent = script(server, [chat(BASH_OK), chat(BASH_OK)])
        # A formatter may have changed the file: the LLM answers the read-back from then on.
        await server.simulate_tool_call(req(), _sim("bash", "c2", command="npx prettier --write site"))
        again = await server.simulate_tool_call(req(), _sim("bash", "c3", command="cat site/index.html"))
        assert again.source == "llm" and len(sent) == 2
        assert state.answered_by == {"exact": 1, "llm": 2}

    async def test_image_mentioned_in_history_is_read_in_code(self) -> None:
        server = make_server()
        await _seed(server)
        out = await server.simulate_tool_call(req(), _sim("read_file", path="/home/user/uploads/logo.png"))
        assert (out.source, json.loads(out.output)) == ("exact", IMAGE_READ)

    async def test_tool_outside_the_session_tool_list_gets_the_harness_error(self) -> None:
        server = make_server()
        await _seed(server)
        out = await server.simulate_tool_call(req(), _sim("web_search", query="weather"))
        assert (out.source, json.loads(out.output)) == ("exact", HARNESS_ERROR)
        bad = SimulateToolCallRequest(call_id="c", name="bash", arguments="{not json")
        assert json.loads((await server.simulate_tool_call(req(), bad)).output) == HARNESS_ERROR

    async def test_a_crash_while_answering_is_a_failed_simulation(self) -> None:
        # Not a 500: an error reply carries no session cookie, and the policy would see the error page.
        server = make_server()
        state = await _seed(server)
        crash = AsyncMock(side_effect=ValueError("boom"))
        with patch.object(AgenticTurnSimulationResourcesServer, "_answer", crash):
            out = await server.simulate_tool_call(req(), _sim("bash", command="ls"))
        assert (out.source, out.output) == ("failed", simulator_failure("ValueError: boom"))
        assert state.answered_by == {"failed": 1}


class TestSimulateWithLLM:
    async def test_simulated_output_and_prompt(self) -> None:
        server = make_server()
        state = await _seed(server)
        sent = script(server, [chat('Sure. {"status": "completed", "exit_code": 0, "stdout": "/home/user/site"}')])
        turn = [
            _msg("Checking /home/user/site first."),
            _call("bash", "b1", command="grep -c h1 /home/user/site/index.html"),
        ]
        # A known file, but not a read-back the workspace answers.
        out = await server.simulate_tool_call(
            req(), _sim("bash", "b1", turn, command="grep -c h1 /home/user/site/index.html")
        )
        assert out.source == "llm"
        assert json.loads(out.output) == {"status": "completed", "exit_code": 0, "stdout": "/home/user/site"}
        assert state.answered_by == {"llm": 1}

        assert len(sent) == 1 and sent[0]["server_name"] == "simulator"
        assert sent[0]["url_path"] == "/v1/chat/completions"
        params = sent[0]["json"]
        assert (params["max_tokens"], params["temperature"], params["reasoning_effort"]) == (1000, 1.0, "high")
        system, user = params["messages"]
        assert system == {"role": "system", "content": load_prompt("tool-simulator-sandbox.md")}
        user = user["content"]
        assert '<message from="User">\nMake the heading blue.\n</message>' in user
        assert '<message from="Agent">\nChecking /home/user/site first.\n</message>' in user
        assert user.index("Checking") < user.index("<tool_call_to_simulate>")
        assert '<tool_call id="b1" name="bash">' in user.split("<tool_call_to_simulate>")[1]
        assert '<file path="/home/user/site/index.html">\n<h1>Hi</h1>\n</file>' in user

    async def test_simulated_output_updates_the_workspace(self) -> None:
        server = make_server()
        state = await _seed(server)
        read = '{"content": "remember me", "kind": "text", "lines": 1, "size": 11, "status": "success"}'
        script(server, [chat(read)])
        out = await server.simulate_tool_call(req(), _sim("read_file", "r1", path="notes.md"))
        assert out.source == "llm" and state.workspace.files["/home/user/notes.md"] == "remember me"
        again = await server.simulate_tool_call(req(), _sim("read_file", "r2", path="notes.md"))
        assert again.source == "exact" and json.loads(again.output)["content"] == "remember me"
        assert server.server_client.post.await_count == 1

    async def test_simulator_sees_earlier_steps_and_sibling_outputs_not_sibling_calls(self) -> None:
        server = make_server()
        await _seed(server)
        sent = script(server, [chat(BASH_OK)])
        turn = [
            _msg("STEP_ONE_MESSAGE"),
            _call("bash", "a1", command="echo EARLIER_CALL"),
            _out("a1", '{"stdout": "EARLIER_OUTPUT"}'),
            _msg("STEP_TWO_MESSAGE"),
            _call("bash", "b1", command="echo SIBLING_CALL"),
            _call("bash", "b2", command="echo TARGET_CALL"),
            _out("b1", '{"stdout": "SIBLING_OUTPUT"}'),
        ]
        await server.simulate_tool_call(req(), _sim("bash", "b2", turn, command="echo TARGET_CALL"))
        conversation, target = sent[0]["json"]["messages"][1]["content"].split("<tool_call_to_simulate>")
        for seen in ("STEP_ONE_MESSAGE", "EARLIER_CALL", "EARLIER_OUTPUT", "STEP_TWO_MESSAGE", "SIBLING_OUTPUT"):
            assert seen in conversation
        assert "SIBLING_CALL" not in conversation and "TARGET_CALL" not in conversation
        assert "TARGET_CALL" in target and "SIBLING" not in target

    async def test_turn_without_the_call_is_all_history(self) -> None:
        server = make_server()
        await _seed(server)
        sent = script(server, [chat(BASH_OK)])
        turn = [_msg("NOTE_BEFORE"), _call("bash", "other", command="echo OTHER_CALL")]
        await server.simulate_tool_call(req(), _sim("bash", "b9", turn, command="ls"))
        conversation = sent[0]["json"]["messages"][1]["content"].split("<tool_call_to_simulate>")[0]
        assert "NOTE_BEFORE" in conversation and "OTHER_CALL" in conversation

    async def test_retries_with_backoff_then_succeeds(self, sleep: AsyncMock) -> None:
        server = make_server()
        state = await _seed(server)
        sent = script(server, [RuntimeError("connection reset"), http(status=500, body="overloaded"), chat(BASH_OK)])
        out = await server.simulate_tool_call(req(), _sim("bash", command="ls"))
        assert out.source == "llm" and json.loads(out.output)["stdout"] == "ok"
        assert len(sent) == 3 and [c.args[0] for c in sleep.await_args_list] == [1, 2]
        assert state.answered_by == {"llm": 1}

    async def test_all_attempts_fail(self, sleep: AsyncMock) -> None:
        server = make_server(tool_simulator_max_attempts=3, retry_backoff_max_seconds=1.5)
        state = await _seed(server)
        script(server, [http(status=503, body="busy"), RuntimeError("boom"), chat("I would run ls here.")])
        out = await server.simulate_tool_call(req(), _sim("read_file", path="notes.md"))
        assert out.source == "failed" and state.answered_by == {"failed": 1}
        message = json.loads(out.output)["message"]
        assert message.startswith(SIMULATOR_FAILED) and "unparseable (finish=stop): 'I would run ls here.'" in message
        assert [c.args[0] for c in sleep.await_args_list] == [1, 1.5, 1.5]
        assert set(state.workspace.files) == {"/home/user/site/index.html"}

    async def test_http_error_is_the_reported_failure(self) -> None:
        server = make_server(tool_simulator_max_attempts=1)
        await _seed(server)
        script(server, [http(status=502, body="Bad Gateway")])
        out = await server.simulate_tool_call(req(), _sim("bash", command="ls"))
        assert "HTTP 502: Bad Gateway" in json.loads(out.output)["message"]

    async def test_reasoning_rejection_drops_reasoning_effort_without_backoff(self, sleep: AsyncMock) -> None:
        server = make_server()
        await _seed(server)
        sent = script(server, [http(status=400, body="Unsupported parameter: reasoning_effort"), chat(BASH_OK)])
        out = await server.simulate_tool_call(req(), _sim("bash", command="ls"))
        assert out.source == "llm"
        assert sent[0]["json"]["reasoning_effort"] == "high" and "reasoning_effort" not in sent[1]["json"]
        sleep.assert_not_awaited()

    async def test_reasoning_rejection_without_reasoning_effort_is_a_plain_failure(self, sleep: AsyncMock) -> None:
        params = NeMoGymChatCompletionCreateParamsNonStreaming(messages=[], max_tokens=1000)
        server = make_server(tool_simulator_chat_completions_create_params=params)
        await _seed(server)
        script(server, [http(status=400, body="reasoning is not supported"), chat(BASH_OK)])
        assert (await server.simulate_tool_call(req(), _sim("bash", command="ls"))).source == "llm"
        assert sleep.await_count == 1

    @pytest.mark.parametrize(
        "first, max_tokens_cap, expected",
        [
            (chat('{"status": "comp', finish="length"), 49_152, 2000),
            (chat('{"status": "comp', finish="length"), 1500, 1500),
            (chat("", finish="stop"), 49_152, 2000),
            (http({"choices": []}), 49_152, 2000),
            (chat("I refuse to answer", finish="stop"), 49_152, 1000),
        ],
        ids=["length", "length-capped", "empty-content", "no-choices", "unparseable-text-keeps-budget"],
    )
    async def test_output_budget_doubles_on_truncated_or_empty_replies(
        self, first: MagicMock, max_tokens_cap: int, expected: int
    ) -> None:
        server = make_server(tool_simulator_max_tokens_cap=max_tokens_cap)
        await _seed(server)
        sent = script(server, [first, chat(BASH_OK)])
        assert (await server.simulate_tool_call(req(), _sim("bash", command="ls"))).source == "llm"
        assert sent[0]["json"]["max_tokens"] == 1000 and sent[1]["json"]["max_tokens"] == expected

    @pytest.mark.parametrize(
        "params, expected",
        [
            (NeMoGymChatCompletionCreateParamsNonStreaming(messages=[], max_completion_tokens=3000), 6000),
            (NeMoGymChatCompletionCreateParamsNonStreaming(messages=[]), 32_768),
        ],
        ids=["max_completion_tokens", "default"],
    )
    async def test_budget_doubling_starts_from_the_configured_field(self, params, expected: int) -> None:
        server = make_server(tool_simulator_chat_completions_create_params=params)
        await _seed(server)
        sent = script(server, [chat(None, finish="length"), chat(BASH_OK)])
        await server.simulate_tool_call(req(), _sim("bash", command="ls"))
        assert sent[1]["json"]["max_tokens"] == expected and "max_completion_tokens" not in sent[1]["json"]

    async def test_a_cut_off_reply_is_retried_even_when_it_holds_an_object(self) -> None:
        server = make_server()
        await _seed(server)
        cut = '{"status": "success", "results": [{"title": "a", "content": "b"}, {"title": "c", "con'
        sent = script(server, [chat(cut, finish="length"), chat(BASH_OK)])
        out = await server.simulate_tool_call(req(), _sim("bash", command="ls"))
        assert (out.source, json.loads(out.output)) == ("llm", json.loads(BASH_OK))
        assert sent[1]["json"]["max_tokens"] == 2000

    async def test_answer_in_the_reasoning_channel(self) -> None:
        server = make_server()
        await _seed(server)
        script(server, [chat("", reasoning_content='Thinking... {"status": "completed", "exit_code": 0}')])
        out = await server.simulate_tool_call(req(), _sim("bash", command="ls"))
        assert (out.source, json.loads(out.output)) == ("llm", {"status": "completed", "exit_code": 0})

    async def test_each_model_call_holds_its_own_semaphore(self) -> None:
        server = make_server()
        await _seed(server)
        held = []

        async def post(*, server_name: str, url_path: str, json: dict[str, Any], **_: Any) -> MagicMock:
            held.append((server_name, server._simulator_semaphore._value, server._judge_semaphore._value))
            return chat(BASH_OK) if server_name == "simulator" else judge_reply(VERDICT)

        server.server_client.post = AsyncMock(side_effect=post)
        await server.simulate_tool_call(req(), _sim("bash", command="ls"))
        await server.verify(req(), _verify_body())
        assert held == [("simulator", 63, 32), ("judge_0", 64, 31)]


# --- /verify ------------------------------------------------------------------------------------------------


class TestVerifyMasks:
    async def test_the_wall_clock_cap_is_masked(self) -> None:
        server = make_server()
        out = await server.verify(req(), _verify_body(stop_reason="time_cap", steps="12"))
        assert getattr(out, MASK_SAMPLE_FIELD) is True
        assert out.failure_kind == f"{SERVER}:time_cap" and "time_cap" in out.failure_reason
        assert (out.reward, out.stop_reason, out.steps) == (0.0, "time_cap", 12)
        assert out.correct is None and out.judge_verdicts == []
        server.server_client.post.assert_not_called()

    async def test_a_row_too_long_for_the_policy_is_masked(self) -> None:
        # The first model call overflowed: no step was kept.
        server = make_server()
        out = await server.verify(req(), _verify_body([], stop_reason="context_overflow", steps="0"))
        assert getattr(out, MASK_SAMPLE_FIELD) is True and out.failure_kind == f"{SERVER}:context_overflow"
        assert "does not fit the policy's context" in out.failure_reason
        assert (out.reward, out.stop_reason, out.steps) == (0.0, "context_overflow", 0)
        server.server_client.post.assert_not_called()

    @pytest.mark.parametrize(
        "reason, steps",
        [("final_message", "3"), ("ask_user", "2"), ("mistake_limit", "3")],
        ids=["final_message", "stop_tool", "mistake_limit"],
    )
    async def test_a_turn_the_policy_ended_is_judged(self, reason: str, steps: str) -> None:
        server = make_server()
        sent = script(server, [judge_reply(GOOD)])
        out = await server.verify(req(), _verify_body(stop_reason=reason, steps=steps))
        assert not getattr(out, MASK_SAMPLE_FIELD) and getattr(out, "failure_kind", None) is None
        assert (out.reward, out.correct_ladder, out.stop_reason) == (1.0, "yes", reason)
        assert out.steps == int(steps) and len(sent) == 1

    @pytest.mark.parametrize(
        "reason, steps",
        [
            ("max_steps", "50"),
            ("max_output_tokens", "7"),
            ("max_output_tokens", "0"),
            ("context_overflow", "9"),
            ("context_overflow", None),
        ],
        ids=[
            "max_steps",
            "max_output_tokens",
            "max_output_tokens-first-call",
            "context_overflow-later",
            "context_overflow-no-steps",
        ],
    )
    async def test_an_unfinished_turn_scores_zero_without_a_judge(self, reason: str, steps: str | None) -> None:
        server = make_server()
        out = await server.verify(req(), _verify_body(stop_reason=reason, steps=steps))
        assert not getattr(out, MASK_SAMPLE_FIELD) and getattr(out, "failure_kind", None) is None
        assert (out.reward, out.code_gate, out.correct_ladder, out.stop_reason) == (
            0.0,
            "unfinished_turn",
            "no",
            reason,
        )
        assert out.judge_verdicts == []
        server.server_client.post.assert_not_called()

    @pytest.mark.parametrize("status", [None, "completed", "incomplete"])
    async def test_an_empty_final_reply_after_reasoning_scores_zero_without_a_judge(self, status: str | None) -> None:
        # The policy reasoned, then neither said nor called anything.
        server = make_server()
        thought = NeMoGymResponseReasoningItem(id="r2", summary=[{"type": "summary_text", "text": "I will now"}])
        body = _verify_body([*_policy_turn()[:5], thought], stop_reason="empty_output", steps="3")
        body.response = NeMoGymResponse.model_validate({**body.response.model_dump(), "status": status})
        out = await server.verify(req(), body)
        assert not getattr(out, MASK_SAMPLE_FIELD)
        assert (out.reward, out.code_gate, out.correct_ladder, out.stop_reason) == (
            0.0,
            "unfinished_turn",
            "no",
            "empty_output",
        )
        server.server_client.post.assert_not_called()

    @pytest.mark.parametrize("steps", [3, 1], ids=["after-tool-results", "first-step"])
    async def test_an_empty_final_reply_without_any_item_is_judged(self, steps: int) -> None:
        # A reply with no items at all, not even reasoning, is what a model server passes on for a failed request.
        server = make_server()
        sent = script(server, [judge_reply(GOOD)])
        items = _policy_turn()[:5] if steps > 1 else []
        out = await server.verify(req(), _verify_body(items, stop_reason="empty_output", steps=str(steps)))
        assert (out.reward, out.code_gate, out.correct_ladder, len(sent)) == (1.0, None, "yes", 1)

    @pytest.mark.parametrize(
        "failure",
        [{"status": "failed"}, {"status": "cancelled"}, {"error": {"code": "server_error", "message": "engine died"}}],
        ids=["failed", "cancelled", "error"],
    )
    async def test_an_empty_final_reply_the_model_server_failed_is_judged(self, failure: dict[str, Any]) -> None:
        # A Responses-native model server passes on an aborted request's partial output: here reasoning, then nothing.
        server = make_server()
        sent = script(server, [judge_reply(GOOD)])
        thought = NeMoGymResponseReasoningItem(id="r2", summary=[{"type": "summary_text", "text": "I will now"}])
        body = _verify_body([*_policy_turn()[:5], thought], stop_reason="empty_output", steps="3")
        body.response = NeMoGymResponse.model_validate({**body.response.model_dump(), **failure})
        out = await server.verify(req(), body)
        assert (out.reward, out.code_gate, out.correct_ladder, len(sent)) == (1.0, None, "yes", 1)

    async def test_unfinished_stop_reasons_are_configurable(self) -> None:
        server = make_server(unfinished_stop_reasons=[])
        sent = script(server, [judge_reply(GOOD)])
        out = await server.verify(req(), _verify_body(stop_reason="max_steps", steps="50"))
        assert (out.reward, out.code_gate, len(sent)) == (1.0, None, 1)

    async def test_the_cap_masks_before_the_gate_and_a_step_capped_turn_is_gated(self) -> None:
        server = make_server()
        items = [_call_item("browse_web", '{"url": "x"}'), _output_item("p1", json.dumps(HARNESS_ERROR))]
        out = await server.verify(req(), _verify_body(items, stop_reason="time_cap"))
        assert getattr(out, MASK_SAMPLE_FIELD) is True and out.code_gate is None
        out = await server.verify(req(), _verify_body(items, stop_reason="max_steps", steps="50"))
        assert (out.reward, out.code_gate) == (0.0, "nonexistent_tool") and not getattr(out, MASK_SAMPLE_FIELD)
        server.server_client.post.assert_not_called()

    async def test_masked_stop_reasons_are_configurable(self) -> None:
        old_default = ["max_steps", "time_cap", "context_overflow", "max_output_tokens"]
        server = make_server(masked_stop_reasons=old_default)
        for reason in old_default:
            out = await server.verify(req(), _verify_body(stop_reason=reason, steps="9"))
            assert getattr(out, MASK_SAMPLE_FIELD) is True and out.failure_kind == f"{SERVER}:{reason}"
            assert out.failure_reason == f"the turn was cut off ({reason})"
        server.server_client.post.assert_not_called()

        server = make_server(masked_stop_reasons=[])
        script(server, [judge_reply(GOOD)])
        out = await server.verify(req(), _verify_body(stop_reason="time_cap"))
        assert not getattr(out, MASK_SAMPLE_FIELD) and out.reward == 1.0
        # A row too long for the policy is masked whatever the list says.
        out = await server.verify(req(), _verify_body([], stop_reason="context_overflow", steps="0"))
        assert getattr(out, MASK_SAMPLE_FIELD) is True and out.failure_kind == f"{SERVER}:context_overflow"

    async def test_simulator_failure_in_the_turn_is_masked(self) -> None:
        server = make_server()
        failed = simulator_failure("HTTP 503: busy")
        items = [_call_item("bash", '{"command": "ls"}'), _output_item("p1", failed), _message_item("It failed.")]
        out = await server.verify(req(), _verify_body(items))
        assert getattr(out, MASK_SAMPLE_FIELD) is True
        assert out.failure_kind == f"{SERVER}:simulator_failed" and out.failure_reason == failed
        server.server_client.post.assert_not_called()

    async def test_a_tool_result_that_only_mentions_the_failure_phrase_is_judged(self) -> None:
        server = make_server()
        sent = script(server, [judge_reply(GOOD)])
        grep = {"status": "completed", "exit_code": 0, "stdout": f'12: SIMULATOR_FAILED = "{SIMULATOR_FAILED}"\n'}
        items = [
            _call_item("bash", json.dumps({"command": "grep -n SIMULATOR_FAILED simulator.py"})),
            _output_item("p1", json.dumps(grep)),
            _message_item("Line 12 defines it."),
        ]
        out = await server.verify(req(), _verify_body(items))
        assert not getattr(out, MASK_SAMPLE_FIELD) and out.failure_kind is None
        assert out.reward == 1.0 and len(sent) == 1

    async def test_reference_turn_cut_off_after_a_tool_result_is_masked(self) -> None:
        server = make_server(score_source="reference_turn")
        turn = [_call("bash", "r1", command="ls"), _out("r1", BASH_OK)]
        out = await server.verify(req(), _verify_body(metadata={"reference_turn": turn}))
        assert getattr(out, MASK_SAMPLE_FIELD) is True and out.failure_kind == f"{SERVER}:recording_cut_off"
        assert out.score_source == "reference_turn"
        server.server_client.post.assert_not_called()

    async def test_no_usable_verdict_is_masked(self, sleep: AsyncMock) -> None:
        server = make_server(judge_max_attempts=3)
        sent = script(server, [RuntimeError("boom"), http(status=500, body="oops"), judge_reply(text="no idea")])
        out = await server.verify(req(), _verify_body())
        assert getattr(out, MASK_SAMPLE_FIELD) is True and out.failure_kind == failure_kinds.JUDGE_FAILED
        assert out.reward == 0.0 and out.correct_ladder is None and out.judge_verdicts == [None]
        assert len(sent) == 3 and sleep.await_count == 3

    async def test_last_unusable_verdict_is_kept_for_inspection(self) -> None:
        server = make_server(judge_max_attempts=1)
        partial = {"addresses_concern": "yes", "correct": "partly"}
        script(server, [judge_reply(partial)])
        out = await server.verify(req(), _verify_body())
        assert getattr(out, MASK_SAMPLE_FIELD) is True and out.judge_verdicts == [partial]

    async def test_context_length_rejection_gives_up_at_once(self, sleep: AsyncMock) -> None:
        server = make_server()
        body = "This model's maximum context length is 131072 tokens. However, you requested 150000 tokens."
        sent = script(server, [http(status=400, body=body)])
        out = await server.verify(req(), _verify_body())
        assert len(sent) == 1 and getattr(out, MASK_SAMPLE_FIELD) is True
        assert out.failure_kind == failure_kinds.JUDGE_FAILED
        sleep.assert_not_awaited()


class TestVerifyCodeGates:
    @pytest.mark.parametrize(
        "item, gate",
        [
            (_call_item("browse_web", '{"url": "x"}'), "nonexistent_tool"),
            (_call_item("bash", "{not json"), "malformed_arguments"),
            (_call_item("bash", '{"__malformedToolArguments": "ls -"}'), "malformed_arguments"),
            (_call_item("bash", '{"offset": ' + "1" * 5000 + "}"), "malformed_arguments"),
            (_call_item("bash", '{"command": ' + "[" * 100_000), "malformed_arguments"),
        ],
        ids=["nonexistent", "not-json", "marker", "huge-integer", "deep-nesting"],
    )
    async def test_gate_scores_zero_without_a_judge(self, item: Any, gate: str) -> None:
        server = make_server()
        items = [item, _output_item("p1", json.dumps(HARNESS_ERROR)), _message_item("Something went wrong.")]
        out = await server.verify(req(), _verify_body(items))
        assert (out.reward, out.code_gate, out.correct_ladder) == (0.0, gate, "no")
        assert not getattr(out, MASK_SAMPLE_FIELD)
        server.server_client.post.assert_not_called()

    async def test_without_tool_definitions_the_sandbox_tools_are_allowed(self) -> None:
        server = make_server()
        script(server, [judge_reply(GOOD)])
        items = [_call_item("web_search", '{"query": "q"}'), _output_item("p1", '{"results": []}')]
        out = await server.verify(req(), _verify_body(items, tools=[]))
        assert out.code_gate is None and out.reward == 1.0
        assert (
            "(tool definitions omitted)"
            in server.server_client.post.call_args.kwargs["json"]["messages"][0]["content"]
        )

    async def test_the_generic_argument_marker_is_no_gate_for_sandbox_rows(self) -> None:
        # Sandbox rows mark malformed arguments with their own key only (other harnesses gate on both).
        server = make_server()
        sent = script(server, [judge_reply(GOOD)])
        arguments = json.dumps({UNPARSEABLE_ARGS_KEY: "x"})
        items = [_call_item("bash", arguments), _output_item("p1", BASH_OK), _message_item("Done.")]
        out = await server.verify(req(), _verify_body(items))
        assert out.code_gate is None and out.reward == 1.0 and len(sent) == 1

    async def test_argument_gates_score_zero_without_a_judge(self) -> None:
        server = make_server()
        for arguments, gate in [
            ({"command": "ls", "workdir": "/tmp"}, "unknown_argument"),
            ({"command": "ls", "timeout": "30"}, "wrong_argument_type"),
            ({"timeout": 30}, "missing_argument"),
        ]:
            items = [_call_item("bash", json.dumps(arguments)), _output_item("p1", BASH_OK), _message_item("Done.")]
            out = await server.verify(req(), _verify_body(items, tools=[STRICT_BASH]))
            assert (out.reward, out.code_gate, out.correct_ladder) == (0.0, gate, "no")
            assert not getattr(out, MASK_SAMPLE_FIELD) and out.judge_verdicts == []
        server.server_client.post.assert_not_called()

    @pytest.mark.parametrize("harness", sorted(set(HARNESSES) - {"cline_xml"}))
    async def test_missing_arguments_are_gated_only_where_the_harness_rejects_them(self, harness: str) -> None:
        gated = harness in ("sandbox", "claude_code", "openclaw")
        assert HARNESSES[harness].gates_missing_arguments is gated
        server = make_server()
        sent = script(server, [judge_reply(GOOD)])
        metadata = {"harness": harness}

        def turn(arguments: dict[str, Any]) -> list[Any]:
            return [_call_item("bash", json.dumps(arguments)), _output_item("p1", BASH_OK), _message_item("Done.")]

        out = await server.verify(req(), _verify_body(turn({"timeout": 30}), metadata=metadata, tools=[STRICT_BASH]))
        assert out.harness == harness
        assert (out.code_gate, out.reward) == (("missing_argument", 0.0) if gated else (None, 1.0))
        assert len(sent) == (0 if gated else 1)
        # Unknown keys and wrong types are gated for every harness.
        for arguments, gate in [
            ({"command": "ls", "cwd": "/"}, "unknown_argument"),
            ({"command": 7}, "wrong_argument_type"),
        ]:
            out = await server.verify(req(), _verify_body(turn(arguments), metadata=metadata, tools=[STRICT_BASH]))
            assert (out.code_gate, out.reward) == (gate, 0.0)

    async def test_lenient_unusable_or_missing_schemas_are_not_gated(self) -> None:
        server = make_server()
        script(server, [judge_reply(GOOD)])
        lenient = {**STRICT_BASH, "parameters": {k: v for k, v in BASH_SCHEMA.items() if k != "additionalProperties"}}
        unusable = dict(TOOLS[1], parameters={**BASH_SCHEMA, "properties": {"command": {"type": "text"}}})
        no_schema = dict(TOOLS[2], parameters=None)
        items = [
            _call_item("bash", json.dumps({"command": "ls", "workdir": "/tmp"}), "p1"),
            _output_item("p1", BASH_OK),
            _call_item("read_file", json.dumps({"command": 1, "extra": 1}), "p2"),
            _output_item("p2", "{}"),
            _call_item("write_file", json.dumps({"path": 1, "extra": 1}), "p3"),
            _output_item("p3", "{}"),
            _message_item("Done."),
        ]
        out = await server.verify(req(), _verify_body(items, tools=[lenient, unusable, no_schema]))
        assert out.code_gate is None and out.reward == 1.0

    async def test_a_call_written_into_a_message_scores_zero_without_a_judge(self) -> None:
        server = make_server()
        leaked = "Saving it now.\n<function=write_file>\n<parameter=path>\nindex.html\n</parameter>\n</function>"
        items = [*_policy_turn()[:5], _message_item(leaked)]
        out = await server.verify(req(), _verify_body(items))
        assert (out.reward, out.code_gate, out.correct_ladder) == (0.0, "call_in_text", "no")
        assert not getattr(out, MASK_SAMPLE_FIELD) and out.judge_verdicts == []
        server.server_client.post.assert_not_called()

    async def test_markup_in_code_or_in_a_cline_reply_is_judged(self) -> None:
        server = make_server()
        sent = script(server, [judge_reply(GOOD), judge_reply(GOOD)])
        quoted = "The template writes:\n```\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n```"
        out = await server.verify(req(), _verify_body([*_policy_turn()[:5], _message_item(quoted)]))
        assert (out.reward, out.code_gate) == (1.0, None)
        # Cline's calls are text: a reply wrapped in <tool_call> is left to its own parser and the judge.
        wrapped = '<tool_call>\n{"name": "attempt_completion", "arguments": {"result": "Done."}}\n</tool_call>'
        metadata = {"harness": "cline", "protocol": "cline_xml"}
        out = await server.verify(req(), _verify_body([_message_item(wrapped)], metadata=metadata))
        assert (out.harness, out.reward, out.code_gate) == ("cline_xml", 1.0, None)
        assert len(sent) == 2

    async def test_a_reference_turn_is_not_argument_gated(self) -> None:
        server = make_server(score_source="reference_turn")
        script(server, [judge_reply(GOOD)])
        reference = [_call("bash", "r1", workdir="/tmp"), _out("r1", BASH_OK), _msg("Done.")]
        out = await server.verify(req(), _verify_body(metadata={"reference_turn": reference}, tools=[STRICT_BASH]))
        assert out.code_gate is None and out.reward == 1.0


class TestVerifyJudge:
    async def test_correct_turn_scores_one(self) -> None:
        server = make_server()
        state = await _seed(server)
        state.answered_by.update({"exact": 2, "llm": 1})
        sent = script(server, [judge_reply(GOOD)])
        metadata = {"categories": ["web_dev", "styling"], "corrected_turn_item_range": {"steerability": [4, 5]}}
        out = await server.verify(req(), _verify_body(metadata=metadata))

        assert out.reward == 1.0 and not getattr(out, MASK_SAMPLE_FIELD)
        assert (out.addresses_concern, out.correct, out.correct_ladder, out.blocked_by) == (
            "yes",
            "yes",
            "yes",
            "none",
        )
        assert (out.request_questionable, out.discard, out.discard_reason) == ("ok", "no", "none")
        assert out.judge_verdicts == [GOOD] and out.code_gate is None
        assert (out.score_source, out.stop_reason, out.steps) == ("policy", "final_message", 3)
        assert out.shell_failures == 1 and out.categories == ["web_dev", "styling"]
        assert out.simulated_calls == {"exact": 2, "llm": 1} and "s1" not in server._sessions

        (request,) = sent
        assert (request["server_name"], request["url_path"]) == ("judge_0", "/v1/chat/completions")
        params = request["json"]
        assert params["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "Audit", "strict": True, "schema": evaluation_schema()},
        }
        assert (params["max_tokens"], params["temperature"], params["reasoning_effort"]) == (2000, 1.0, "high")
        system, user = (m["content"] for m in params["messages"])
        assert "Check the agent's claims against the evidence." in system
        assert '<tool name="present_file">\nthe present_file tool' in system
        assert "/home/user/site/index.html" in user
        assert "SECRET_REASONING" not in user
        history, turn = user.split("<step_under_evaluation>")
        assert '<rejected_turn>  <!-- the turn the user is correcting -->\n<message from="Agent">\nDone:' in history
        assert "Make the heading blue." in history and "Made the heading blue" in turn
        assert '<tool_result id="p2">' in turn

    async def test_incorrect_turn_scores_zero(self) -> None:
        server = make_server()
        script(server, [judge_reply(BAD)])
        out = await server.verify(req(), _verify_body())
        assert (out.reward, out.correct_ladder, out.blocked_by) == (0.0, "no", "model")
        assert not getattr(out, MASK_SAMPLE_FIELD) and out.simulated_calls == {}

    async def test_judge_recovers_after_garbage(self, sleep: AsyncMock) -> None:
        server = make_server()
        script(
            server,
            [judge_reply(text="I think it is fine."), judge_reply(text="<think>hmm</think>" + json.dumps(GOOD))],
        )
        assert (await server.verify(req(), _verify_body())).reward == 1.0
        assert sleep.await_count == 1

    async def test_json_schema_rejection_falls_back_to_json_object(self, sleep: AsyncMock) -> None:
        server = make_server()
        rejected = "response_format json_schema is not supported by this model"
        sent = script(server, [http(status=400, body=rejected), http(status=400, body=rejected), judge_reply(GOOD)])
        out = await server.verify(req(), _verify_body())
        assert out.reward == 1.0
        assert sent[0]["json"]["response_format"]["type"] == "json_schema"
        assert sent[1]["json"]["response_format"] == {"type": "json_object"}
        # The fallback happens once; a second rejection is retried as is, after a backoff.
        assert sent[2]["json"]["response_format"] == {"type": "json_object"} and sleep.await_count == 1

    async def test_rejected_sampling_parameters_are_dropped(self, sleep: AsyncMock) -> None:
        server = make_server()
        sent = script(
            server,
            [
                http(status=400, body="reasoning_effort is not supported"),
                http(status=400, body="temperature is not supported"),
                http(status=400, body="Bad request"),
                judge_reply(GOOD),
            ],
        )
        assert (await server.verify(req(), _verify_body())).reward == 1.0
        assert "reasoning_effort" in sent[0]["json"] and "reasoning_effort" not in sent[1]["json"]
        assert "temperature" in sent[1]["json"] and "temperature" not in sent[2]["json"]
        assert sent[3]["json"] == sent[2]["json"] and sleep.await_count == 1

    async def test_a_rejection_naming_several_parameters_relaxes_one_at_a_time_in_order(
        self, sleep: AsyncMock
    ) -> None:
        server = make_server()
        rejected = "response_format json_schema, reasoning_effort and temperature are not supported"
        sent = script(server, [http(status=400, body=rejected)] * 3 + [judge_reply(GOOD)])
        assert (await server.verify(req(), _verify_body())).reward == 1.0
        first, schema_relaxed, reasoning_dropped, temperature_dropped = (r["json"] for r in sent)
        assert first["response_format"]["type"] == "json_schema"
        assert schema_relaxed["response_format"] == {"type": "json_object"}
        assert "reasoning_effort" in schema_relaxed and "temperature" in schema_relaxed
        assert "reasoning_effort" not in reasoning_dropped and reasoning_dropped["temperature"] == 1.0
        assert "temperature" not in temperature_dropped
        sleep.assert_not_awaited()

    @pytest.mark.parametrize(
        "params, cap, expected",
        [
            (dict(max_tokens=2000), 65_536, 4000),
            (dict(max_tokens=2000), 3000, 3000),
            (dict(max_completion_tokens=5000), 65_536, 10_000),
            ({}, 65_536, 40_000),
        ],
        ids=["max_tokens", "capped", "max_completion_tokens", "default"],
    )
    async def test_truncated_verdict_doubles_the_budget(self, params: dict[str, Any], cap: int, expected: int) -> None:
        judge_params = NeMoGymChatCompletionCreateParamsNonStreaming(messages=[], **params)
        server = make_server(judge_chat_completions_create_params=judge_params, judge_max_tokens_cap=cap)
        sent = script(server, [judge_reply(text='{"analysis": "The agent first', finish="length"), judge_reply(GOOD)])
        assert (await server.verify(req(), _verify_body())).reward == 1.0
        assert sent[1]["json"]["max_tokens"] == expected and "max_completion_tokens" not in sent[1]["json"]

    async def test_a_truncated_verdict_missing_fields_is_retried(self) -> None:
        server = make_server()
        text = '{"analysis": "ok", "addresses_concern": "yes", "correct": "yes", "blocked_by": "none", "req'
        sent = script(server, [judge_reply(text=text, finish="length"), judge_reply(BAD)])
        out = await server.verify(req(), _verify_body())
        assert len(sent) == 2 and sent[1]["json"]["max_tokens"] == 4000  # retried with double the budget
        assert (out.reward, out.blocked_by) == (0.0, "model") and "truncated" not in out.judge_verdicts[0]

    async def test_a_truncated_verdict_with_every_field_is_used(self) -> None:
        server = make_server()
        sent = script(server, [judge_reply(text=json.dumps(BAD)[:-1], finish="length")])
        out = await server.verify(req(), _verify_body())
        assert len(sent) == 1 and out.reward == 0.0 and out.blocked_by == "model"
        assert out.judge_verdicts[0]["truncated"] is True

    async def test_per_judge_overrides(self) -> None:
        server = make_server()
        server.config.judges[0].overrides = {
            "temperature": None,
            "reasoning_effort": "low",
            "max_tokens": 777,
            "response_format": {"type": "json_object"},
        }
        sent = script(server, [judge_reply(GOOD)])
        await server.verify(req(), _verify_body())
        params = sent[0]["json"]
        assert "temperature" not in params and params["reasoning_effort"] == "low" and params["max_tokens"] == 777
        assert params["response_format"] == {"type": "json_object"}

    async def test_a_null_response_format_override_sends_none(self, sleep: AsyncMock) -> None:
        server = make_server()
        server.config.judges[0].overrides = {"response_format": None}
        sent = script(server, [http(status=400, body="json_schema is not supported"), judge_reply(GOOD)])
        assert (await server.verify(req(), _verify_body())).reward == 1.0
        assert all("response_format" not in s["json"] for s in sent)
        assert sleep.await_count == 1  # a schema rejection has nothing to relax: an ordinary retry

    async def test_verdict_fields_that_are_not_strings_are_dropped(self) -> None:
        # Dropped, they leave the verdict short of the schema: it is retried, then the rollout is masked.
        server = make_server(judge_max_attempts=2)
        verdict = {**GOOD, "discard": False, "request_questionable": ["ok"]}
        sent = script(server, [judge_reply(verdict), judge_reply(verdict)])
        out = await server.verify(req(), _verify_body())
        assert len(sent) == 2 and getattr(out, MASK_SAMPLE_FIELD) is True
        assert out.failure_kind == failure_kinds.JUDGE_FAILED and out.reward == 0.0
        assert "discard" not in out.judge_verdicts[0] and "request_questionable" not in out.judge_verdicts[0]

    @pytest.mark.parametrize(
        "verdict",
        [
            {**GOOD, "addresses_concern": None},
            {k: v for k, v in GOOD.items() if k != "addresses_concern"},
            {**GOOD, "addresses_concern": "maybe"},
            {**GOOD, "discard": True},
            {**GOOD, "blocked_by": "the judge"},
            {k: v for k, v in GOOD.items() if k != "reason"},
        ],
        ids=[
            "steer-null",
            "steer-missing",
            "steer-invalid",
            "discard-boolean",
            "blocked_by-invalid",
            "reason-missing",
        ],
    )
    async def test_a_verdict_short_of_the_schema_is_retried_and_never_pays(
        self, verdict: dict[str, Any], sleep: AsyncMock
    ) -> None:
        server = make_server(judge_max_attempts=3)
        sent = script(server, [judge_reply(verdict)] * 3)
        out = await server.verify(req(), _verify_body())
        assert len(sent) == 3 and sleep.await_count == 3
        assert getattr(out, MASK_SAMPLE_FIELD) is True and out.failure_kind == failure_kinds.JUDGE_FAILED
        assert (out.reward, out.correct_ladder, out.correct) == (0.0, None, None)
        assert out.judge_verdicts == [parse_verdict(json.dumps(verdict))]  # the last one, as parsed, for inspection

    async def test_a_verdict_short_of_the_schema_is_retried_until_a_usable_one(self) -> None:
        server = make_server()
        sent = script(server, [judge_reply({**GOOD, "addresses_concern": None}), judge_reply(BAD)])
        out = await server.verify(req(), _verify_body())
        assert len(sent) == 2 and not getattr(out, MASK_SAMPLE_FIELD)
        assert (out.reward, out.correct_ladder, out.judge_verdicts) == (0.0, "no", [BAD])

    async def test_judges_without_a_usable_verdict_leave_the_vote(self) -> None:
        # Two judges say `correct: yes` with a null steerability: they do not outvote the one valid verdict.
        server = make_server(n_judges=3, judge_max_attempts=1)
        unusable = {**GOOD, "addresses_concern": None}
        replies = {
            "judge_0": [judge_reply(unusable)],
            "judge_1": [judge_reply(unusable)],
            "judge_2": [judge_reply(BAD)],
        }
        script(server, replies)
        out = await server.verify(req(), _verify_body())
        assert (out.reward, out.correct_ladder, out.judge_verdicts) == (0.0, "no", [unusable, unusable, BAD])
        assert not getattr(out, MASK_SAMPLE_FIELD)

    @pytest.mark.parametrize(
        "verdicts, reward, usable",
        [
            ([GOOD, GOOD], 1.0, 2),
            ([GOOD, BAD], 0.0, 2),  # a tie does not give the policy credit
            ([None, GOOD], 1.0, 1),  # one judge without a verdict: the other decides
        ],
        ids=["agree", "tie", "one-failed"],
    )
    async def test_two_judges_vote(self, verdicts: list[dict[str, Any] | None], reward: float, usable: int) -> None:
        server = make_server(n_judges=2, judge_max_attempts=1)
        replies = {
            f"judge_{i}": [
                judge_reply(v) if v is not None else http(status=400, body="maximum context length exceeded")
            ]
            for i, v in enumerate(verdicts)
        }
        sent = script(server, replies)
        out = await server.verify(req(), _verify_body())
        assert out.reward == reward and out.judge_verdicts == verdicts
        assert sum(v is not None for v in out.judge_verdicts) == usable
        assert sorted(s["server_name"] for s in sent) == ["judge_0", "judge_1"]
        assert sent[0]["json"]["messages"] == sent[1]["json"]["messages"]

    async def test_reference_turn_mode_judges_the_reference_turn(self) -> None:
        server = make_server(score_source="reference_turn")
        sent = script(server, [judge_reply(GOOD)])
        reference = [
            _call("bash", "r1", command="ls /home/user"),
            _out("r1", BASH_OK),
            {"type": "function_call", "call_id": "r3", "name": "bash", "arguments": "{truncated"},  # not gated
            _msg("REFERENCE ANSWER"),
        ]
        policy = [_message_item("POLICY ANSWER")]
        body = _verify_body(policy, stop_reason="max_steps", metadata={"reference_turn": reference})
        out = await server.verify(req(), body)
        assert out.reward == 1.0 and out.score_source == "reference_turn" and out.code_gate is None
        assert not getattr(out, MASK_SAMPLE_FIELD)  # the policy's stop reason does not apply
        user = sent[0]["json"]["messages"][1]["content"]
        assert "REFERENCE ANSWER" in user and "POLICY ANSWER" not in user and "ls /home/user" in user

    async def test_reference_turn_mode_with_an_invented_tool_and_empty_turn(self) -> None:
        server = make_server(score_source="reference_turn")
        invented = [_call("browse", "r1", url="x"), _msg("done")]
        out = await server.verify(req(), _verify_body(metadata={"reference_turn": invented}))
        assert out.code_gate == "nonexistent_tool" and out.reward == 0.0
        sent = script(server, [judge_reply(BAD)])
        out = await server.verify(req(), _verify_body(metadata={}))
        assert out.reward == 0.0 and "[the agent produced nothing]" in sent[0]["json"]["messages"][1]["content"]

    async def test_string_prompt_and_missing_metadata(self) -> None:
        server = make_server()
        sent = script(server, [judge_reply(GOOD)])
        body = _verify_body([_message_item("Hello!")], response_metadata={}, input_="just a string prompt")
        out = await server.verify(req(None), body)
        assert out.reward == 1.0 and out.stop_reason is None and out.steps is None
        assert out.categories == ["web_dev"] and out.shell_failures == 0
        assert '<message from="User">\njust a string prompt\n</message>' in sent[0]["json"]["messages"][1]["content"]

    async def test_non_numeric_steps_are_dropped(self) -> None:
        server = make_server()
        script(server, [judge_reply(GOOD)])
        out = await server.verify(req(), _verify_body(steps="many"))
        assert out.steps is None


class TestVerifyTurnEnding:
    async def test_a_final_message_names_no_stop_tool_and_counts_the_calls_that_ran(self) -> None:
        server = make_server()
        script(server, [judge_reply(GOOD)])
        out = await server.verify(req(), _verify_body())
        assert (out.ended_on_stop_tool, out.executed_calls) == (None, 2)

    @pytest.mark.parametrize(
        "items, executed",
        [
            ([_message_item("Which shade of blue?"), _call_item("ask_user", '{"question": "Which?"}', "p1")], 0),
            (
                [
                    _call_item("bash", '{"command": "ls"}', "p1"),
                    _call_item("ask_user", '{"question": "Which?"}', "p2"),
                    _output_item("p1", BASH_OK),
                ],
                1,
            ),
        ],
        ids=["ask-only", "ran-a-sibling-first"],
    )
    async def test_a_stop_tool_ending_is_named_and_its_call_never_counts(
        self, items: list[Any], executed: int
    ) -> None:
        server = make_server()
        script(server, [judge_reply(GOOD)])
        out = await server.verify(req(), _verify_body(items, stop_reason="ask_user", steps="1"))
        assert (out.ended_on_stop_tool, out.executed_calls) == ("ask_user", executed)

    async def test_a_stop_reason_that_is_not_one_of_the_rows_stop_tools_names_none(self) -> None:
        # present_file is not a sandbox stop tool, whatever the agent's config said.
        server = make_server()
        script(server, [judge_reply(GOOD)])
        items = [_call_item("present_file", '{"path": "a.pdf"}', "p1")]
        out = await server.verify(req(), _verify_body(items, stop_reason="present_file", steps="1"))
        assert (out.ended_on_stop_tool, out.executed_calls) == (None, 0)

    async def test_masked_and_gated_turns_report_them_too(self) -> None:
        server = make_server()
        masked = await server.verify(req(), _verify_body(stop_reason="time_cap"))
        gated = await server.verify(req(), _verify_body(stop_reason="max_steps"))
        assert getattr(masked, MASK_SAMPLE_FIELD) and gated.code_gate == "unfinished_turn"
        assert [(o.ended_on_stop_tool, o.executed_calls) for o in (masked, gated)] == [(None, 2), (None, 2)]

    async def test_reference_turn_mode_reports_the_policys_turn(self) -> None:
        server = make_server(score_source="reference_turn")
        script(server, [judge_reply(GOOD)])
        reference = [_call("bash", "r1", command="ls"), _out("r1", BASH_OK), _msg("Listed.")]
        policy = [_call_item("ask_user", '{"question": "Which?"}', "p1")]
        body = _verify_body(policy, stop_reason="ask_user", metadata={"reference_turn": reference})
        out = await server.verify(req(), body)
        assert out.score_source == "reference_turn" and (out.ended_on_stop_tool, out.executed_calls) == ("ask_user", 0)


# Chinese spliced into an English reply to an English history: 5 letters of a script the input never shows.
GARBLED = "Made the heading 标题已更新 blue in /home/user/site/index.html."


def _garbled_turn() -> list[Any]:
    return [*_policy_turn()[:5], _message_item(GARBLED)]


class TestVerifyForeignScript:
    async def test_the_letters_are_reported_and_judged_with_the_gate_off(self) -> None:
        server = make_server()
        sent = script(server, [judge_reply(GOOD), judge_reply(GOOD)])
        clean = await server.verify(req(), _verify_body())
        garbled = await server.verify(req(), _verify_body(_garbled_turn()))
        assert [(o.foreign_script_letters, o.code_gate, o.reward) for o in (clean, garbled)] == [
            (0, None, 1.0),
            (5, None, 1.0),
        ]
        assert len(sent) == 2

    async def test_the_gate_scores_zero_without_a_judge_from_its_threshold_on(self) -> None:
        server = make_server(foreign_script_gate_min_letters=5)
        out = await server.verify(req(), _verify_body(_garbled_turn()))
        assert (out.reward, out.code_gate, out.correct_ladder, out.foreign_script_letters) == (
            0.0,
            "foreign_script",
            "no",
            5,
        )
        assert not getattr(out, MASK_SAMPLE_FIELD) and out.judge_verdicts == []
        server.server_client.post.assert_not_called()
        below = make_server(foreign_script_gate_min_letters=6)
        script(below, [judge_reply(GOOD)])
        out = await below.verify(req(), _verify_body(_garbled_turn()))
        assert (out.reward, out.code_gate, out.foreign_script_letters) == (1.0, None, 5)

    async def test_the_other_gates_come_first(self) -> None:
        server = make_server(foreign_script_gate_min_letters=1)
        items = [_call_item("browse_web", '{"url": "x"}'), _output_item("p1", "{}"), _message_item(GARBLED)]
        out = await server.verify(req(), _verify_body(items))
        assert (out.code_gate, out.foreign_script_letters) == ("nonexistent_tool", 5)
        out = await server.verify(req(), _verify_body(_garbled_turn(), stop_reason="max_steps"))
        assert (out.code_gate, out.foreign_script_letters) == ("unfinished_turn", 5)

    async def test_a_masked_turn_is_still_measured(self) -> None:
        server = make_server(foreign_script_gate_min_letters=1)
        out = await server.verify(req(), _verify_body(_garbled_turn(), stop_reason="time_cap"))
        assert getattr(out, MASK_SAMPLE_FIELD) and (out.code_gate, out.foreign_script_letters) == (None, 5)

    async def test_a_reference_turn_is_neither_measured_nor_gated(self) -> None:
        server = make_server(score_source="reference_turn", foreign_script_gate_min_letters=1)
        script(server, [judge_reply(GOOD)])
        reference = [_call("bash", "r1", command="ls"), _out("r1", BASH_OK), _msg(GARBLED)]
        out = await server.verify(req(), _verify_body(_garbled_turn(), metadata={"reference_turn": reference}))
        assert (out.reward, out.code_gate, out.foreign_script_letters) == (1.0, None, None)

    def test_the_threshold_is_a_positive_number_of_letters(self) -> None:
        with pytest.raises(ValueError, match="foreign_script_gate_min_letters"):
            make_server(foreign_script_gate_min_letters=0)


# --- the sandbox home path ----------------------------------------------------------------------------------


class TestPathsPassThrough:
    """The simulator and the judge see the paths the policy sees, and the policy gets the simulator's paths."""

    async def test_a_simulated_call_reaches_the_simulator_unchanged(self) -> None:
        server = make_server()
        await _seed(server)
        sent = script(server, [chat(BASH_OK)])
        out = await server.simulate_tool_call(req(), _sim("bash", "b1", command="ls -la /home/user/x"))
        assert out.source == "llm"
        system, user = (m["content"] for m in sent[0]["json"]["messages"])
        target = user.split("<tool_call_to_simulate>")[1]
        assert '<tool_call id="b1" name="bash">\n{"command": "ls -la /home/user/x"}\n</tool_call>' in target
        assert '<message from="User">\nBuild a page in /home/user/site, use my logo.png\n</message>' in user

    async def test_the_judge_sees_the_home_path(self) -> None:
        server = make_server()
        await _seed(server)
        sent = script(server, [judge_reply(GOOD)])
        await server.verify(req(), _verify_body())
        system, user = (m["content"] for m in sent[0]["json"]["messages"])
        assert '<message from="User">\nBuild a page in /home/user/site, use my logo.png\n</message>' in user
        assert '{"command": "sed -n 1p /home/user/site/index.html"}' in user
        assert "Made the heading blue in /home/user/site/index.html." in user

    async def test_a_simulator_reply_reaches_the_policy_unchanged(self) -> None:
        server = make_server()
        state = await _seed(server)
        reply = '{"duration_ms": 9, "exit_code": 0, "status": "completed", "stderr": "", "stdout": "/home/user/x\\n"}'
        script(server, [chat(reply)])
        out = await server.simulate_tool_call(req(), _sim("bash", command="pwd"))
        assert (out.source, out.output) == ("llm", reply)
        assert state.answered_by == {"llm": 1}


# --- metrics ------------------------------------------------------------------------------------------------


def _row(reward: float, **kw: Any) -> dict[str, Any]:
    base = {
        "reward": reward,
        "addresses_concern": "yes",
        "code_gate": None,
        "blocked_by": "none",
        "request_questionable": "ok",
        "shell_failures": 0,
        "categories": ["web"],
        "stop_reason": "final_message",
    }
    return {**base, **kw}


class TestMetrics:
    def test_compute_metrics(self) -> None:
        rows = [
            _row(1.0),
            _row(0.0, blocked_by="model", shell_failures=2),
            _row(0.0, code_gate="nonexistent_tool", addresses_concern=None, blocked_by=None, categories=[]),
            _row(
                0.0,
                addresses_concern="no",
                blocked_by=None,
                request_questionable="questionable",
                categories=["code", "web"],
                stop_reason="ask_user",
            ),
        ]
        metrics = make_server().compute_metrics([rows[:2], rows[2:]])
        assert metrics == {
            "stop_reasons": {"final_message": 3, "ask_user": 1},
            "turn_score": 0.25,
            "steerable_rate": 0.5,
            "code_gate_rate": 0.25,
            "questionable_request_rate": 0.25,
            "shell_failures_per_turn": 0.5,
            "failure_attribution": {
                "model": pytest.approx(1 / 3),
                "nonexistent_tool": pytest.approx(1 / 3),
                "steer=no": pytest.approx(1 / 3),
            },
            "turn_score_by_category": {"code": 0.0, "uncategorized": 0.0, "web": pytest.approx(1 / 3)},
        }
        assert list(metrics["turn_score_by_category"]) == ["code", "uncategorized", "web"]

    def test_empty_and_all_correct(self) -> None:
        server = make_server()
        assert server.compute_metrics([]) == {} and server.compute_metrics([[]]) == {}
        metrics = server.compute_metrics([[_row(1.0)], [_row(1.0)]])
        assert metrics["turn_score"] == 1.0 and metrics["failure_attribution"] == {}

    def test_key_metrics(self) -> None:
        server = make_server()
        full = {"turn_score": 0.5, "steerable_rate": 0.9, "code_gate_rate": 0.1, "other": 7}
        assert server.get_key_metrics(full) == {k: full[k] for k in list(full)[:3]}
        assert server.get_key_metrics({"stop_reasons": {}}) == {}

    async def test_ladder_failure_is_attributed_to_steerability(self) -> None:
        server = make_server()
        script(server, [judge_reply(dict(GOOD, addresses_concern="no"))])
        out = await server.verify(req(), _verify_body())
        assert out.reward == 0.0
        metrics = server.compute_metrics([[out.model_dump()]])
        assert metrics["failure_attribution"] == {"steer=no": 1.0}


def _tagged_row(reward: float, metadata: dict[str, Any], shell: int | None = None) -> dict[str, Any]:
    return {"reward": reward, "verifier_metadata": metadata, "shell_failures": shell}


class TestMetricBreakdowns:
    def test_breakdowns_by_the_configured_keys(self) -> None:
        cc = {"harness": "claude_code", "record_kind": "interactive", "split": "s1", "difficulty": {"hard": False}}
        hermes = {"harness": "hermes", "record_kind": "subagent", "split": "s2", "difficulty": {"hard": True}}
        unnamed = {"categories": ["web_dev"]}
        rows = [
            _tagged_row(1.0, cc),
            _tagged_row(0.0, cc),
            _tagged_row(1.0, hermes),
            _tagged_row(1.0, unnamed, shell=2),
        ]
        metrics = make_server().compute_metrics([rows[:2], rows[2:]])
        assert metrics["turn_score_by_harness"] == {"claude_code": 0.5, "hermes": 1.0, "none": 1.0}
        assert metrics["turn_score_by_record_kind"] == {"interactive": 0.5, "none": 1.0, "subagent": 1.0}
        assert "turn_score_by_split" not in metrics  # not a default key, unless listed
        assert metrics["turn_score"] == 0.75
        assert metrics["shell_failures_per_turn"] == 2.0  # only the row with a measured shell count
        listed = make_server(metric_breakdown_keys=["split", "difficulty.hard"]).compute_metrics([rows])
        assert listed["turn_score_by_split"] == {"none": 1.0, "s1": 0.5, "s2": 1.0}
        assert listed["turn_score_by_difficulty_hard"] == {"false": 0.5, "none": 1.0, "true": 1.0}
        assert "turn_score_by_harness" not in listed

    def test_rows_without_the_keys_add_nothing(self) -> None:
        metrics = make_server().compute_metrics([[_tagged_row(1.0, {"categories": ["web"]}, shell=0)]])
        assert not [k for k in metrics if "_by_" in k and k != "turn_score_by_category"]
        assert metrics["shell_failures_per_turn"] == 0.0

    def test_no_shell_measurement_leaves_shell_failures_out(self) -> None:
        metrics = make_server().compute_metrics([[_tagged_row(1.0, {"harness": "hermes"})]])
        assert "shell_failures_per_turn" not in metrics and metrics["turn_score"] == 1.0
