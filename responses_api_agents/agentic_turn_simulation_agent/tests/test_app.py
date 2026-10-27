# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from aiohttp import ClientResponseError, RequestInfo
from fastapi.testclient import TestClient
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.reward_profile import MASK_SAMPLE_FIELD
from nemo_gym.server_utils import ServerClient
from resources_servers.agentic_turn_simulation.app import AgenticTurnSimulationVerifyRequest
from resources_servers.agentic_turn_simulation.cline_xml import NO_TOOLS_USED
from resources_servers.agentic_turn_simulation.simulator import simulator_failure
from resources_servers.agentic_turn_simulation.tests import support as env
from responses_api_agents.agentic_turn_simulation_agent.app import (
    AgenticTurnSimulationAgent,
    AgenticTurnSimulationAgentConfig,
    context_overflow_room,
)
from responses_api_agents.simple_agent.app import ModelServerRef, ResourcesServerRef, SimpleAgentRunRequest


MOD = "responses_api_agents.agentic_turn_simulation_agent.app"
CONFIG_PATH = Path(__file__).parents[1] / "configs" / "agentic_turn_simulation_agent.yaml"
MODEL_PATH = "/v1/responses"
TOOL_PATH = "/simulate_tool_call"
HARNESS_PATH = "/harness"


def _overflow(window: int, prompt: int, completion: int = 8000) -> str:
    return (
        f"This model's maximum context length is {window} tokens. However, you requested "
        f"{prompt + completion} tokens ({prompt} in the messages, {completion} in the completion). "
        "Please reduce the length of the messages or completion."
    )


# --- fakes -------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def sleep():
    """/simulate_tool_call retries back off at once; a test that takes `sleep` can check the delays."""
    with patch(f"{MOD}.sleep", new=AsyncMock()) as mock:
        yield mock


def _http(payload: Any = None, *, status: int = 200, body: str = "", cookies: Optional[dict] = None) -> MagicMock:
    resp = MagicMock(status=status, ok=status < 400, cookies=cookies or {})
    resp.read = AsyncMock(return_value=json.dumps(payload if payload is not None else {}).encode())
    resp.content.read = AsyncMock(return_value=body.encode())
    return resp


def _msg(text: str, item_id: str = "m1") -> Dict[str, Any]:
    return {
        "id": item_id,
        "content": [{"annotations": [], "text": text, "type": "output_text"}],
        "role": "assistant",
        "status": "completed",
        "type": "message",
    }


def _fc(name: str, call_id: str, arguments: str = "{}") -> Dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def _reasoning(text: str = "thinking") -> Dict[str, Any]:
    return {"type": "reasoning", "id": "rs1", "summary": [{"type": "summary_text", "text": text}]}


def _model(
    *items: Dict[str, Any], rid: str = "resp1", incomplete: bool = False, tokens: int = 10, cookies=None
) -> MagicMock:
    payload: Dict[str, Any] = {
        "id": rid,
        "created_at": 0.0,
        "model": "policy",
        "object": "response",
        "output": list(items),
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 100 + tokens,
        },
    }
    if incomplete:
        payload["status"] = "incomplete"
        payload["incomplete_details"] = {"reason": "max_output_tokens"}
    return _http(payload, cookies=cookies)


class Script:
    """Routes the agent's posts: /harness answers with `harness` (a reply, a JSON body, or None for the server's 404
    when it has no session), model calls take the next queued reply, tool calls the next tool reply (default: a
    successful simulated output naming the call)."""

    def __init__(self, models: List[MagicMock], tools: Optional[List[MagicMock]] = None, harness: Any = None) -> None:
        self.models, self.tools, self.harness = list(models), list(tools or []), harness
        self.model_calls: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.harness_calls: List[Dict[str, Any]] = []

    async def post(self, *, server_name: str, url_path: str, json: Any = None, cookies: Any = None) -> MagicMock:
        call = {"server_name": server_name, "url_path": url_path, "json": json, "cookies": cookies}
        if url_path == HARNESS_PATH:
            self.harness_calls.append(call)
            if self.harness is None:
                return _http({"detail": "No seeded session for this request: call /seed_session first."}, status=404)
            return self.harness if isinstance(self.harness, MagicMock) else _http(self.harness)
        if url_path == TOOL_PATH:
            self.tool_calls.append(call)
            return self.tools.pop(0) if self.tools else _http({"output": f"result of {json['call_id']}"})
        self.model_calls.append(call)
        return self.models.pop(0)

    @property
    def budgets(self) -> List[Optional[int]]:
        return [c["json"].max_output_tokens for c in self.model_calls]


def _agent(
    *models: MagicMock,
    tools: Optional[List[MagicMock]] = None,
    harness: Any = None,
    observability: bool = False,
    **config: Any,
):
    cfg = AgenticTurnSimulationAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="turn_sim_agent",
        model_server=ModelServerRef(type="responses_api_models", name="policy"),
        resources_server=ResourcesServerRef(type="resources_servers", name="turn_sim"),
        **config,
    )
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {"observability_enabled": observability}
    script = Script(list(models), tools, harness)
    server_client.post = AsyncMock(side_effect=script.post)
    return AgenticTurnSimulationAgent(config=cfg, server_client=server_client), script


def _body(max_output_tokens: Optional[int] = None, **kw: Any) -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(
        input="make the heading blue", max_output_tokens=max_output_tokens, **kw
    )


async def _episode(agent: AgenticTurnSimulationAgent, body=None, **kw: Any):
    return await agent._create_episode(body or _body(), model_url_path=MODEL_PATH, **kw)


def _types(items: List[Any]) -> List[str]:
    return [i["type"] if isinstance(i, dict) else i.type for i in items]


BASH_TOOL = {
    "type": "function",
    "name": "bash",
    "description": "run",
    "parameters": {"type": "object"},
    "strict": False,
}


def _verify_request(response: Any) -> AgenticTurnSimulationVerifyRequest:
    """The resources server's /verify request for an episode run on `_body(tools=[BASH_TOOL])` (a generic row)."""
    return AgenticTurnSimulationVerifyRequest(responses_create_params=_body(tools=[BASH_TOOL]), response=response)


# --- config and parsing ------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        agent, _ = _agent()
        cfg = agent.config
        assert (cfg.max_steps, cfg.max_time_seconds, cfg.stop_tools) == (50, 7200.0, ["ask_user"])
        assert (cfg.max_output_tokens_cap, cfg.min_output_tokens, cfg.context_margin_tokens) == (65_536, 2_048, 256)
        assert cfg.cline_max_consecutive_mistakes == 3
        assert cfg.simulate_max_attempts == 3

    def test_shipped_config_validates(self) -> None:
        raw = yaml.safe_load(CONFIG_PATH.read_text())
        section = raw["agentic_turn_simulation_agent"]["responses_api_agents"]["agentic_turn_simulation_agent"]
        # A template: the environment config that uses the agent names its resources server.
        assert section["resources_server"]["name"] == "???"
        resources_server = {"type": "resources_servers", "name": "agentic_turn_simulation_resources_server"}
        cfg = AgenticTurnSimulationAgentConfig(
            host="0.0.0.0", port=0, name="x", **{**section, "resources_server": resources_server}
        )
        assert (cfg.max_steps, cfg.max_time_seconds, cfg.stop_tools) == (50, 7200, ["ask_user"])
        assert cfg.resources_server.name == "agentic_turn_simulation_resources_server"


class TestContextOverflowRoom:
    @pytest.mark.parametrize(
        "detail, margin, expected",
        [
            (_overflow(32_768, 30_000), 256, (True, 2_512)),
            (_overflow(32_768, 30_000), 0, (True, 2_768)),
            (
                "Error: This model's MAXIMUM CONTEXT LENGTH is 131072 tokens and your prompt contains 131000 "
                "input tokens",
                100,
                (True, -28),
            ),
            (
                "'max_tokens' or 'max_completion_tokens' is too large: 65536. This model's maximum context length is "
                "131072 tokens and your request has 70000 input tokens (65536 > 131072 - 70000).",
                256,
                (True, 131_072 - 70_000 - 256),
            ),
            ("maximum context length is 32768 tokens; the request is too long", 256, (True, None)),
            ("The input exceeds the context length of the model", 256, (True, None)),
            ("Requested more than the maximum number of tokens", 256, (True, None)),
            ("Internal Server Error", 256, (False, None)),
            ("rate limit exceeded, retry later", 256, (False, None)),
            ("", 256, (False, None)),
        ],
        ids=[
            "vllm-messages",
            "no-margin",
            "prompt-contains",
            "request-has",
            "window-only",
            "context-length-words",
            "maximum-tokens",
            "server-error",
            "rate-limit",
            "empty",
        ],
    )
    def test_parse(self, detail: str, margin: int, expected) -> None:
        assert context_overflow_room(detail, margin=margin) == expected


# --- the turn loop -----------------------------------------------------------------------------


class TestStopReasons:
    async def test_final_message(self) -> None:
        agent, script = _agent(_model(_reasoning(), _msg("All done."), cookies={"m": "1"}))
        response, trajectory, model_cookies, resources_cookies = await _episode(agent)
        assert response.metadata["stop_reason"] == "final_message" and response.metadata["steps"] == "1"
        assert response.metadata["elapsed_seconds"].isdigit()
        assert _types(response.output) == ["reasoning", "message"] and response.id == "resp1"
        assert response.usage.output_tokens == 10 and response.usage.input_tokens == 100
        assert trajectory is None and model_cookies == {"m": "1"} and resources_cookies == {}
        (call,) = script.model_calls
        assert (call["server_name"], call["url_path"], call["cookies"]) == ("policy", MODEL_PATH, None)
        assert call["json"].input == [NeMoGymEasyInputMessage(role="user", content="make the heading blue")]

    async def test_empty_output(self) -> None:
        agent, _ = _agent(_model(_reasoning()))
        response, *_ = await _episode(agent)
        assert response.metadata["stop_reason"] == "empty_output" and response.metadata["steps"] == "1"

    async def test_stop_tool_ends_the_turn_unexecuted(self) -> None:
        question = _fc("ask_user", "c1", '{"question": "Which blue?"}')
        agent, script = _agent(_model(_msg("One question first."), question))
        response, *_ = await _episode(agent)
        assert response.metadata["stop_reason"] == "ask_user" and response.metadata["steps"] == "1"
        assert _types(response.output) == ["message", "function_call"] and response.output[-1].name == "ask_user"
        assert script.tool_calls == [] and len(script.model_calls) == 1

    @pytest.mark.parametrize("question_first", [False, True], ids=["bash-then-ask", "ask-then-bash"])
    async def test_the_calls_beside_a_stop_tool_run_before_the_turn_ends(self, question_first: bool) -> None:
        bash, question = _fc("bash", "c1", '{"command": "ls"}'), _fc("ask_user", "c2", '{"question": "Which blue?"}')
        reply = [question, bash] if question_first else [bash, question]
        agent, script = _agent(_model(*reply), tools=[_http({"output": "index.html"})])
        response, trajectory, *_ = await _episode(agent, collect_trajectory=True)

        assert response.metadata["stop_reason"] == "ask_user" and response.metadata["steps"] == "1"
        assert len(script.model_calls) == 1  # the turn ends: the question goes to the user
        (simulated,) = script.tool_calls
        assert (simulated["json"]["call_id"], simulated["json"]["name"]) == ("c1", "bash")
        assert _types(response.output) == ["function_call", "function_call", "function_call_output"]
        assert (response.output[-1].call_id, response.output[-1].output) == ("c1", "index.html")
        assert [c.tool_call_id for c in trajectory.tool_calls] == ["c1"]
        assert trajectory.invocations[0].status == "completed"

    async def test_custom_stop_tools(self) -> None:
        agent, script = _agent(_model(_fc("present_file", "c1", '{"path": "a.pdf"}')), stop_tools=["present_file"])
        response, *_ = await _episode(agent)
        assert response.metadata["stop_reason"] == "present_file" and script.tool_calls == []

    async def test_max_steps(self) -> None:
        steps = [_model(_fc("bash", f"c{i}", '{"command": "ls"}'), rid=f"r{i}") for i in range(3)]
        agent, script = _agent(*steps, max_steps=2)
        response, *_ = await _episode(agent)
        assert response.metadata["stop_reason"] == "max_steps" and response.metadata["steps"] == "2"
        assert len(script.model_calls) == 2 and len(script.tool_calls) == 2
        assert _types(response.output) == ["function_call", "function_call_output"] * 2
        assert response.id == "r1"

    async def test_time_cap_is_checked_before_each_model_call(self) -> None:
        agent, script = _agent(_model(_fc("bash", "c1")), _model(_msg("never")), max_time_seconds=10)
        with patch(f"{MOD}.monotonic", side_effect=[0.0, 1.0, 11.0, 12.0]):
            response, *_ = await _episode(agent)
        assert response.metadata == {
            "stop_reason": "time_cap",
            "steps": "1",
            "elapsed_seconds": "12",
            "model_calls": json.dumps([{"input_tokens": 100, "output_tokens": 10, "cut_off": False, "kept": True}]),
        }
        assert len(script.model_calls) == 1 and len(script.tool_calls) == 1

    async def test_no_caps_when_disabled(self) -> None:
        steps = [_model(_fc("bash", f"c{i}")) for i in range(3)] + [_model(_msg("done"))]
        agent, _ = _agent(*steps, max_steps=0, max_time_seconds=None)
        response, *_ = await _episode(agent)
        assert response.metadata["stop_reason"] == "final_message" and response.metadata["steps"] == "4"


class TestToolCalls:
    async def test_every_call_goes_to_the_simulator_with_the_turn_so_far(self) -> None:
        step1 = _model(
            _reasoning("private"),
            _msg("Let me check."),
            _fc("bash", "c1", '{"command": "ls"}'),
            _fc("read_file", "c2", "{not json"),
            cookies={"model": "1"},
        )
        tools = [
            _http({"output": "OUT1", "source": "llm"}, cookies={"sess": "b"}),
            _http({"output": "OUT2"}, cookies={"sess": "c"}),
        ]
        harness = {"stop_tools": ["ask_user"], "protocol": "native"}
        agent, script = _agent(step1, _model(_msg("Done.")), tools=tools, harness=harness)
        response, _, model_cookies, resources_cookies = await _episode(agent, resources_server_cookies={"sess": "a"})

        first, second = script.tool_calls
        assert (first["server_name"], first["url_path"]) == ("turn_sim", TOOL_PATH)
        assert first["cookies"] == {"sess": "a"} and second["cookies"] == {"sess": "b"}
        assert {k: first["json"][k] for k in ("call_id", "name", "arguments")} == {
            "call_id": "c1",
            "name": "bash",
            "arguments": '{"command": "ls"}',
        }
        assert second["json"]["arguments"] == "{not json"  # unparseable arguments still go to the simulator
        assert _types(first["json"]["turn"]) == ["message", "function_call", "function_call"]
        assert _types(second["json"]["turn"]) == ["message", "function_call", "function_call", "function_call_output"]
        assert second["json"]["turn"][-1] == {"call_id": "c1", "output": "OUT1", "type": "function_call_output"}
        assert "private" not in json.dumps(second["json"]["turn"])
        assert first["json"]["turn"][0]["content"][0]["text"] == "Let me check."

        next_input = script.model_calls[1]["json"].input
        assert _types(next_input) == [
            "message",
            "reasoning",
            "message",
            "function_call",
            "function_call",
            "function_call_output",
            "function_call_output",
        ]
        assert [o.output for o in next_input[-2:]] == ["OUT1", "OUT2"]
        assert script.model_calls[1]["cookies"] == {"model": "1"}
        assert response.metadata["steps"] == "2" and _types(response.output)[-1] == "message"
        assert resources_cookies == {"sess": "c"} and model_cookies == {}

    async def test_an_error_reply_is_retried_and_the_successful_result_is_used(self, sleep: AsyncMock) -> None:
        tools = [_http(status=500, body="Internal Server Error"), _http({"output": "OUT1"})]
        agent, script = _agent(_model(_fc("bash", "c1", '{"command": "ls"}')), _model(_msg("Done.")), tools=tools)
        response, *_ = await _episode(agent)
        first, second = script.tool_calls
        assert second["json"] == first["json"]  # the same call, resent
        assert response.output[1].output == "OUT1" and script.model_calls[1]["json"].input[-1].output == "OUT1"
        assert [c.args for c in sleep.await_args_list] == [(1,)]

    async def test_a_persistent_error_is_a_simulator_failure_that_verify_masks(self, sleep: AsyncMock) -> None:
        tools = [_http(status=500, body="Internal Server Error") for _ in range(3)]
        agent, script = _agent(_model(_fc("bash", "c1")), _model(_msg("It broke.")), tools=tools)
        response, *_ = await _episode(agent, _body(tools=[BASH_TOOL]))
        failed = simulator_failure("HTTP 500: Internal Server Error")
        assert len(script.tool_calls) == 3 and [c.args for c in sleep.await_args_list] == [(1,), (2,)]
        assert response.output[1].output == failed and script.model_calls[1]["json"].input[-1].output == failed
        assert response.metadata["stop_reason"] == "final_message"
        # The resources server's /verify recognizes its own failure output: masked, and no judge is called.
        server = env.make_server()
        out = await server.verify(env.req(), _verify_request(response))
        assert getattr(out, MASK_SAMPLE_FIELD) is True
        assert out.failure_kind == "agentic_turn_simulation:simulator_failed" and out.reward == 0.0
        server.server_client.post.assert_not_called()

    async def test_a_tool_error_in_a_successful_reply_is_ordinary_output_and_judged(self, sleep: AsyncMock) -> None:
        not_found = json.dumps({"status": "error", "message": "File not found: /home/user/a.txt"})
        agent, script = _agent(
            _model(_fc("bash", "c1")), _model(_msg("No such file.")), tools=[_http({"output": not_found})]
        )
        response, *_ = await _episode(agent, _body(tools=[BASH_TOOL]))
        assert len(script.tool_calls) == 1 and response.output[1].output == not_found
        sleep.assert_not_awaited()
        server = env.make_server()
        sent = env.script(server, [env.judge_reply(env.VERDICT)])
        out = await server.verify(env.req(), _verify_request(response))
        assert not getattr(out, MASK_SAMPLE_FIELD) and (out.reward, len(sent)) == (1.0, 1)

    async def test_the_attempts_are_configurable(self, sleep: AsyncMock) -> None:
        tools = [_http(status=502, body="bad gateway")]
        agent, script = _agent(_model(_fc("bash", "c1")), _model(_msg("ok")), tools=tools, simulate_max_attempts=1)
        response, *_ = await _episode(agent)
        assert len(script.tool_calls) == 1 and response.output[1].output == simulator_failure("HTTP 502: bad gateway")
        sleep.assert_not_awaited()

    async def test_session_cookies_survive_an_error_reply(self) -> None:
        # Gym's error replies set no cookie: the session must still reach the retries, the later calls and /verify.
        errors = [_http(status=500, body="Internal Server Error") for _ in range(4)]
        tools = [*errors, _http({"output": "ok"}, cookies={"session": "b"})]
        step = _model(_fc("bash", "c1"), _fc("bash", "c2"), _fc("bash", "c3"))
        harness = {"stop_tools": ["ask_user"], "protocol": "native"}
        agent, script = _agent(step, _model(_msg("done")), tools=tools, harness=harness)
        *_, resources_cookies = await _episode(agent, resources_server_cookies={"session": "a"})
        # c1 fails its three attempts; c2 fails once, then succeeds and sets a cookie; c3 carries it.
        assert [c["json"]["call_id"] for c in script.tool_calls] == ["c1", "c1", "c1", "c2", "c2", "c3"]
        assert [c["cookies"] for c in script.tool_calls] == [{"session": "a"}] * 5 + [{"session": "b"}]
        assert resources_cookies == {"session": "b"}

    async def test_missing_output_field_is_an_empty_output(self) -> None:
        agent, _ = _agent(_model(_fc("bash", "c1")), _model(_msg("ok")), tools=[_http({"source": "exact"})])
        response, *_ = await _episode(agent)
        assert response.output[1].output == ""


class TestContextOverflow:
    async def test_shrinks_the_output_budget_to_fit_then_succeeds(self) -> None:
        agent, script = _agent(_http(status=400, body=_overflow(10_000, 5_000)), _model(_msg("ok")))
        response, *_ = await _episode(agent, _body(max_output_tokens=8_000))
        assert script.budgets == [8_000, 10_000 - 5_000 - 256]
        assert response.metadata["stop_reason"] == "final_message" and response.metadata["steps"] == "1"

    async def test_shrinks_without_a_requested_budget(self) -> None:
        agent, script = _agent(_http(status=400, body=_overflow(10_000, 5_000)), _model(_msg("ok")))
        await _episode(agent)
        assert script.budgets == [None, 4_744]

    async def test_shrunk_budget_is_never_grown_again(self) -> None:
        agent, script = _agent(_http(status=400, body=_overflow(10_000, 5_000)), _model(_reasoning(), incomplete=True))
        response, *_ = await _episode(agent, _body(max_output_tokens=8_000))
        assert script.budgets == [8_000, 4_744]
        assert response.metadata["stop_reason"] == "max_output_tokens"

    @pytest.mark.parametrize(
        "detail, budget",
        [
            (_overflow(10_000, 9_000), 8_000),  # what fits is below min_output_tokens
            (_overflow(10_000, 5_000), 4_000),  # the budget already fits: shrinking cannot help
            ("prompt is longer than the maximum context length", 8_000),  # no numbers to size it by
        ],
        ids=["room-too-small", "room-not-smaller", "unsized"],
    )
    async def test_gives_up_without_a_model_response(self, detail: str, budget: int) -> None:
        agent, script = _agent(_http(status=400, body=detail))
        response, trajectory, model_cookies, resources_cookies = await _episode(
            agent, _body(max_output_tokens=budget, model="my-policy")
        )
        assert len(script.model_calls) == 1
        assert response.metadata["stop_reason"] == "context_overflow" and response.metadata["steps"] == "0"
        assert (response.id, response.model, response.output) == ("no_model_response", "my-policy", [])
        assert response.usage is None and (model_cookies, resources_cookies) == ({}, {})

    async def test_overflow_mid_turn_keeps_the_partial_turn(self) -> None:
        agent, script = _agent(
            _model(_fc("bash", "c1"), rid="first"), _http(status=400, body="context length exceeded")
        )
        response, *_ = await _episode(agent)
        assert response.metadata["stop_reason"] == "context_overflow" and response.metadata["steps"] == "1"
        assert response.id == "first" and _types(response.output) == ["function_call", "function_call_output"]

    async def test_no_model_name_falls_back_to_unknown(self) -> None:
        agent, _ = _agent(_http(status=413, body="maximum context length exceeded"))
        response, *_ = await _episode(agent)
        assert response.model == "unknown"

    async def test_other_http_errors_raise(self) -> None:
        agent, _ = _agent(_http(status=500, body="CUDA error: device-side assert"))
        with pytest.raises(RuntimeError, match="HTTP 500: CUDA error"):
            await _episode(agent)

    async def test_invalid_model_response_raises(self) -> None:
        agent, _ = _agent(_http({"unexpected": True}))
        with pytest.raises(RuntimeError, match="invalid response from model server"):
            await _episode(agent)


class TestTruncation:
    async def test_truncated_steps_are_retried_with_double_the_budget(self) -> None:
        replies = [_model(_reasoning(), incomplete=True), _model(_msg("cut"), incomplete=True), _model(_msg("ok"))]
        agent, script = _agent(*replies, max_output_tokens_cap=4_000)
        response, *_ = await _episode(agent, _body(max_output_tokens=1_000))
        assert script.budgets == [1_000, 2_000, 4_000]
        assert response.metadata["steps"] == "1" and _types(response.output) == ["message"]
        assert response.output[0].content[0].text == "ok"  # discarded attempts leave nothing behind
        assert response.usage.output_tokens == 10  # their tokens are listed in model_calls, not counted in the usage
        assert [c["kept"] for c in _model_calls(response)] == [False, False, True]

    async def test_doubling_stops_at_the_cap(self) -> None:
        replies = [_model(_reasoning(), incomplete=True) for _ in range(3)]
        agent, script = _agent(*replies, max_output_tokens_cap=3_000)
        response, *_ = await _episode(agent, _body(max_output_tokens=1_000))
        assert script.budgets == [1_000, 2_000, 3_000]
        assert response.metadata["stop_reason"] == "max_output_tokens" and response.metadata["steps"] == "0"
        assert response.output == [] and response.id == "resp1"

    async def test_truncated_at_the_cap_with_visible_output_is_kept(self) -> None:
        agent, script = _agent(_model(_msg("partial answer"), incomplete=True), max_output_tokens_cap=1_000)
        response, *_ = await _episode(agent, _body(max_output_tokens=1_000))
        assert len(script.model_calls) == 1
        assert response.metadata["stop_reason"] == "final_message" and response.output[0].content[0].text == (
            "partial answer"
        )

    @pytest.mark.parametrize("protocol", ["native", "cline_xml"])
    async def test_an_empty_message_at_the_cap_shows_nothing(self, protocol: str) -> None:
        # vllm_model's placeholder for a request it could not serve: cut off, one message with empty text.
        harness = {"stop_tools": ["ask_user"], "protocol": protocol}
        agent, script = _agent(_model(_msg(" "), incomplete=True), harness=harness, max_output_tokens_cap=1_000)
        response, *_ = await _episode(agent, _body(max_output_tokens=1_000))
        assert response.metadata["stop_reason"] == "max_output_tokens" and response.metadata["steps"] == "0"
        assert len(script.model_calls) == 1 and response.output == []

    async def test_a_truncated_tool_call_at_the_cap_ends_the_turn_unrun(self) -> None:
        reply = _model(_msg("Listing."), _fc("bash", "c1", '{"command": "ls'), incomplete=True)
        agent, script = _agent(reply, _model(_msg("ok")), max_output_tokens_cap=0)
        response, *_ = await _episode(agent, _body(max_output_tokens=1_000))
        assert script.budgets == [1_000]  # a cap of 0 disables doubling
        assert script.tool_calls == [] and response.output == []
        assert response.metadata["stop_reason"] == "max_output_tokens" and response.metadata["steps"] == "0"

    async def test_no_doubling_without_a_budget(self) -> None:
        agent, script = _agent(_model(_reasoning(), incomplete=True))
        response, *_ = await _episode(agent)
        assert script.budgets == [None] and response.metadata["stop_reason"] == "max_output_tokens"


def _write_file(call_id: str, arguments: str = '{"path": "app.py"}') -> Dict[str, Any]:
    """vLLM's view of a write_file call the budget cut inside `content`: the parser dropped the unfinished argument."""
    return _fc("write_file", call_id, arguments)


class TestBudgetSpentWithoutIncompleteStatus:
    """vLLM reports a reply cut off inside a tool call as finish_reason "tool_calls", so the response is not marked
    incomplete: a reply whose output tokens reach the requested budget is cut off all the same."""

    @pytest.mark.parametrize("tokens", [1_000, 1_001])
    async def test_it_is_retried_with_double_the_budget(self, tokens: int) -> None:
        replies = [
            _model(_msg("Writing it."), _write_file("c1"), tokens=tokens),
            _model(_write_file("c2", '{"path": "app.py", "content": "x"}'), tokens=1_500),
            _model(_msg("Done.")),
        ]
        agent, script = _agent(*replies, max_output_tokens_cap=4_000)
        response, *_ = await _episode(agent, _body(max_output_tokens=1_000))
        assert script.budgets == [1_000, 2_000, 2_000]
        assert [c["json"]["call_id"] for c in script.tool_calls] == ["c2"]  # the cut call never ran
        assert _types(response.output) == ["function_call", "function_call_output", "message"]
        assert response.metadata["stop_reason"] == "final_message" and response.metadata["steps"] == "2"
        assert _model_calls(response) == [_record(tokens, cut_off=True, kept=False), _record(1_500), _record()]

    async def test_at_the_cap_a_tool_call_ends_the_turn_unrun(self) -> None:
        agent, script = _agent(_model(_write_file("c1"), tokens=1_000), max_output_tokens_cap=1_000)
        response, *_ = await _episode(agent, _body(max_output_tokens=1_000))
        assert len(script.model_calls) == 1 and script.tool_calls == [] and response.output == []
        assert response.metadata["stop_reason"] == "max_output_tokens" and response.metadata["steps"] == "0"
        assert _model_calls(response) == [_record(1_000, cut_off=True, kept=False)]

    async def test_at_the_cap_a_text_only_reply_is_kept(self) -> None:
        # It may have ended on its last allowed token: a final message is judged as written either way.
        agent, _ = _agent(_model(_msg("All done."), tokens=1_000), max_output_tokens_cap=1_000)
        response, *_ = await _episode(agent, _body(max_output_tokens=1_000))
        assert response.metadata["stop_reason"] == "final_message" and _types(response.output) == ["message"]
        assert _model_calls(response) == [_record(1_000, cut_off=True, kept=True)]

    @pytest.mark.parametrize("budget, tokens", [(1_000, 999), (None, 100_000)])
    async def test_a_reply_below_the_budget_or_without_one_is_not_cut_off(self, budget, tokens: int) -> None:
        agent, script = _agent(_model(_write_file("c1"), tokens=tokens), _model(_msg("Done.")))
        response, *_ = await _episode(agent, _body(max_output_tokens=budget))
        assert script.budgets == [budget, budget]
        assert [c["json"]["call_id"] for c in script.tool_calls] == ["c1"]
        assert response.metadata["stop_reason"] == "final_message" and response.metadata["steps"] == "2"
        assert _model_calls(response) == [_record(tokens), _record()]


def _model_calls(response) -> List[Dict[str, Any]]:
    return json.loads(response.metadata["model_calls"])


def _record(output_tokens: int = 10, *, cut_off: bool = False, kept: bool = True) -> Dict[str, Any]:
    return {"input_tokens": 100, "output_tokens": output_tokens, "cut_off": cut_off, "kept": kept}


class TestModelCallUsage:
    async def test_each_reply_is_listed_and_the_turn_usage_is_their_sum(self) -> None:
        replies = [_model(_fc("bash", "c1"), tokens=7), _model(_fc("bash", "c2"), tokens=11), _model(_msg("done"))]
        agent, _ = _agent(*replies)
        response, *_ = await _episode(agent)
        calls = _model_calls(response)
        assert calls == [_record(7), _record(11), _record(10)]
        assert response.usage.output_tokens == sum(c["output_tokens"] for c in calls) == 28
        assert response.usage.input_tokens == sum(c["input_tokens"] for c in calls) == 300

    async def test_a_retried_cut_off_reply_is_listed_once_as_cut_off_and_not_kept(self) -> None:
        replies = [_model(_reasoning(), incomplete=True, tokens=1_000), _model(_msg("ok"), tokens=1_500)]
        agent, script = _agent(*replies, max_output_tokens_cap=4_000)
        response, *_ = await _episode(agent, _body(max_output_tokens=1_000))
        assert script.budgets == [1_000, 2_000]
        assert _model_calls(response) == [_record(1_000, cut_off=True, kept=False), _record(1_500)]
        assert response.usage.output_tokens == 1_500

    async def test_the_usage_counts_only_the_kept_replies(self) -> None:
        # The trajectory references only the kept reply, and Gym's rollout_token_count_mismatch check compares
        # response.usage with the calls it references: the discarded reply's tokens stay in model_calls alone.
        discarded = _model(_msg("Writing it."), _write_file("c1"), tokens=1_000, rid="discarded")
        kept = _model(_msg("Done."), tokens=1_500, rid="kept")
        agent, _ = _agent(discarded, kept, max_output_tokens_cap=4_000)
        response, trajectory, *_ = await _episode(agent, _body(max_output_tokens=1_000), collect_trajectory=True)
        assert _model_calls(response) == [_record(1_000, cut_off=True, kept=False), _record(1_500)]
        assert [ref.response_id for ref in trajectory.invocations[0].model_calls] == ["kept"]
        usage = response.usage
        assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (100, 1_500, 1_600)
        assert (usage.input_tokens_details.cached_tokens, usage.output_tokens_details.reasoning_tokens) == (0, 0)

    async def test_a_cut_off_reply_kept_at_the_cap_is_marked_cut_off(self) -> None:
        agent, _ = _agent(_model(_msg("partial answer"), incomplete=True), max_output_tokens_cap=1_000)
        response, *_ = await _episode(agent, _body(max_output_tokens=1_000))
        assert _model_calls(response) == [_record(cut_off=True, kept=True)]
        assert response.usage.output_tokens == 10  # kept: its tokens count

    async def test_a_cut_off_reply_that_ends_the_turn_is_not_kept(self) -> None:
        agent, _ = _agent(_model(_reasoning(), incomplete=True))
        response, *_ = await _episode(agent)
        assert response.metadata["stop_reason"] == "max_output_tokens"
        assert _model_calls(response) == [_record(cut_off=True, kept=False)]
        assert response.usage is None  # no reply was kept

    async def test_a_model_server_error_is_not_a_reply(self) -> None:
        agent, _ = _agent(_http(status=400, body=_overflow(10_000, 5_000)), _model(_msg("ok")))
        response, *_ = await _episode(agent, _body(max_output_tokens=8_000))
        assert _model_calls(response) == [_record()]
        agent, _ = _agent(_http(status=400, body=_overflow(10_000, 9_000)))
        response, *_ = await _episode(agent, _body(max_output_tokens=8_000))
        assert response.metadata["stop_reason"] == "context_overflow" and _model_calls(response) == []

    async def test_a_reply_without_usage_lists_null_tokens(self) -> None:
        reply = _model(_msg("done"))
        payload = json.loads(reply.read.return_value)
        del payload["usage"]
        agent, _ = _agent(_http(payload))
        response, *_ = await _episode(agent)
        assert response.usage is None
        assert _model_calls(response) == [
            {"input_tokens": None, "output_tokens": None, "cut_off": False, "kept": True}
        ]


class TestTrajectory:
    async def test_turns_tool_calls_and_gaps_are_recorded(self) -> None:
        step1 = _model(_reasoning("plan"), _msg("Checking."), _fc("bash", "c1"), _fc("read_file", "c2"), rid="r1")
        step2 = _model(_msg("Done."), rid="")
        tools = [_http({"output": "OUT1"}), *(_http(status=502, body="bad gateway") for _ in range(3))]
        agent, _ = _agent(step1, step2, tools=tools)
        response, trajectory, *_ = await _episode(agent, task_id="t1", rollout_id="0-1", collect_trajectory=True)

        assert (trajectory.task_id, trajectory.rollout_id) == ("t1", "0-1")
        (invocation,) = trajectory.invocations
        assert invocation.status == "completed" and invocation.invocation_id == "root"
        assert [ref.response_id for ref in invocation.model_calls] == ["r1"]
        assert len(invocation.conversation) == 1 + len(response.output)

        first, second = trajectory.turns
        assert (first.turn_no, second.turn_no) == (1, 2)
        assert _types(first.answer) == ["message", "function_call", "function_call"]
        assert first.reasoning_content[0]["summary"][0]["text"] == "plan" and second.reasoning_content is None
        assert (first.step_count, second.step_count) == (2, 2)
        assert [ref.response_id for ref in first.model_calls] == ["r1"] and second.model_calls == []

        ok, failed = trajectory.tool_calls
        assert (ok.tool_call_id, ok.tool_name, ok.status, ok.error_type, ok.output) == (
            "c1",
            "bash",
            "completed",
            None,
            "OUT1",
        )
        # A call that failed every attempt: the last HTTP status, and the output the model was given.
        assert (failed.status, failed.error_type) == ("failed", "http_502")
        assert failed.output == simulator_failure("HTTP 502: bad gateway")
        assert ok.completed_at >= ok.started_at and ok.duration_ms >= 0
        assert [(g.code, g.detail) for g in trajectory.gaps] == [("model_call_reference_unavailable", "turn:2")]

    @pytest.mark.parametrize(
        "replies, config, status",
        [
            ([_model(_fc("ask_user", "c1"))], {}, "completed"),
            ([_model(_fc("bash", "c1"))], {"max_steps": 1}, "incomplete"),
            ([_model(_reasoning())], {}, "incomplete"),
        ],
        ids=["stop-tool", "max-steps", "empty-output"],
    )
    async def test_invocation_status(self, replies: List[MagicMock], config: Dict[str, Any], status: str) -> None:
        agent, _ = _agent(*replies, **config)
        _, trajectory, *_ = await _episode(agent, collect_trajectory=True)
        assert trajectory.invocations[0].status == status


class TestThroughTheEndpoint:
    def test_context_overflow_before_any_reply_still_answers(self) -> None:
        agent, _ = _agent(_http(status=400, body=_overflow(10_000, 9_900)))
        client = TestClient(agent.setup_webserver())
        reply = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hello"}]})
        assert reply.status_code == 200
        assert reply.json()["metadata"]["stop_reason"] == "context_overflow"
        assert reply.json()["id"] == "no_model_response"

    def test_prefixed_rollout_collects_a_trajectory(self) -> None:
        agent, script = _agent(_model(_msg("hi")), observability=True)
        client = TestClient(agent.setup_webserver())
        reply = client.post("/ng-rollout/0-0/v1/responses", json={"input": [{"role": "user", "content": "hello"}]})
        assert reply.status_code == 200
        assert reply.json()["_ng_trajectory"]["rollout_id"] == "0-0"
        assert script.model_calls[0]["url_path"].endswith(MODEL_PATH)


def _harness_error(status: int) -> MagicMock:
    """A /harness reply other than 404, failing as aiohttp fails it."""
    reply = _http(status=status, body="Internal Server Error")
    url = URL("http://turn_sim/harness")
    info = RequestInfo(url, "POST", CIMultiDictProxy(CIMultiDict()), url)
    reply.raise_for_status = MagicMock(side_effect=ClientResponseError(info, (), status=status))
    return reply


class TestRowHarness:
    """/harness names the row's stop tools and protocol; a bare call (no resources cookies, so no session there) gets
    the config's stop tools and native calls."""

    async def test_the_harness_is_asked_once_with_the_episodes_cookies(self) -> None:
        harness = _http({"stop_tools": ["ask_user"], "protocol": "native"}, cookies={"sess": "other"})
        agent, script = _agent(_model(_fc("bash", "c1")), _model(_msg("done")), harness=harness)
        await _episode(agent, resources_server_cookies={"sess": "a"})
        (lookup,) = script.harness_calls
        assert (lookup["server_name"], lookup["cookies"], lookup["json"]) == ("turn_sim", {"sess": "a"}, None)
        assert script.tool_calls[0]["cookies"] == {"sess": "a"}  # the lookup's reply leaves the episode's jar alone

    async def test_the_rows_stop_tools_replace_the_configs(self) -> None:
        agent, script = _agent(
            _model(_fc("ask_user", "c1")),
            _model(_fc("clarify", "c2", '{"question": "Which?"}')),
            harness={"stop_tools": ["clarify"], "protocol": "native"},
        )
        response, *_ = await _episode(agent)
        assert response.metadata["stop_reason"] == "clarify" and response.metadata["steps"] == "2"
        assert [c["json"]["name"] for c in script.tool_calls] == ["ask_user"]  # not a stop tool for this row

    async def test_an_empty_list_means_no_stop_tools(self) -> None:
        agent, script = _agent(
            _model(_fc("ask_user", "c1")), _model(_msg("done")), harness={"stop_tools": [], "protocol": "native"}
        )
        response, *_ = await _episode(agent)
        assert response.metadata["stop_reason"] == "final_message" and len(script.tool_calls) == 1

    async def test_a_row_stop_tool_completes_the_trajectory(self) -> None:
        agent, _ = _agent(
            _model(_fc("AskUserQuestion", "c1")), harness={"stop_tools": ["AskUserQuestion"], "protocol": "native"}
        )
        _, trajectory, *_ = await _episode(agent, collect_trajectory=True)
        assert trajectory.invocations[0].status == "completed"

    async def test_without_a_session_the_config_stop_tools_and_native_calls_apply(self) -> None:
        # Native: the XML in the text is not a call; the config's stop tool ends the turn.
        reply = _model(_msg("<read_file>\n<path>a.py</path>\n</read_file>"), _fc("clarify", "c1"))
        agent, script = _agent(reply, stop_tools=["clarify"])
        response, *_ = await _episode(agent)
        assert len(script.harness_calls) == 1
        assert response.metadata["stop_reason"] == "clarify" and script.tool_calls == []

    async def test_a_seeded_session_the_server_no_longer_has_is_an_error(self) -> None:
        # /run seeded it (the episode carries its cookies): a Cline row run as native would be scored, not masked.
        agent, script = _agent(_model(_msg("hi")))
        with pytest.raises(RuntimeError, match="/harness has no session for this episode's resources cookies"):
            await _episode(agent, resources_server_cookies={"session": "seeded"})
        assert len(script.harness_calls) == 1 and script.model_calls == [] and script.tool_calls == []

    @pytest.mark.parametrize(
        "harness",
        [{}, {"stop_tools": None, "protocol": None}, {"stop_tools": "clarify", "protocol": "native"}],
        ids=["silent", "nulls", "not-a-list"],
    )
    async def test_stop_tools_it_does_not_name_are_the_configs(self, harness: Dict[str, Any]) -> None:
        replies = [_model(_fc("clarify", "c1")), _model(_fc("ask_user", "c2"))]
        agent, script = _agent(*replies, harness=harness)
        response, *_ = await _episode(agent)
        assert response.metadata["stop_reason"] == "ask_user"
        assert [c["json"]["name"] for c in script.tool_calls] == ["clarify"]

    async def test_an_unknown_protocol_is_an_error(self) -> None:
        agent, script = _agent(_model(_msg("hi")), harness={"stop_tools": [], "protocol": "xml"})
        with pytest.raises(ValueError, match="unknown tool-call protocol: 'xml'"):
            await _episode(agent)
        assert script.model_calls == []

    async def test_other_harness_errors_are_raised(self) -> None:
        agent, script = _agent(_model(_msg("hi")), harness=_harness_error(500))
        with pytest.raises(ClientResponseError):
            await _episode(agent)
        assert script.model_calls == []

    def test_the_harness_reaches_the_episode_through_the_endpoint(self) -> None:
        agent, script = _agent(_model(_fc("clarify", "c1")), harness={"stop_tools": ["clarify"], "protocol": "native"})
        client = TestClient(agent.setup_webserver())
        client.cookies.set("env_session", "s1")
        reply = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hello"}]})
        assert reply.status_code == 200 and reply.json()["metadata"]["stop_reason"] == "clarify"
        assert script.tool_calls == [] and script.harness_calls[0]["cookies"]["env_session"] == "s1"


class TestRun:
    async def test_run_is_simple_agents_seed_turn_and_verify(self) -> None:
        agent, _ = _agent()
        calls: List[Dict[str, Any]] = []
        model_json = json.loads(_model(_msg("done")).read.return_value)
        model_json["metadata"] = {"stop_reason": "final_message", "steps": "1", "elapsed_seconds": "0"}

        async def post(*, server_name: str, url_path: str, json: Any, cookies: Any = None, **kw: Any) -> MagicMock:
            calls.append({"server_name": server_name, "url_path": url_path, "cookies": cookies, **kw})
            if url_path == "/seed_session":
                return _http({}, cookies={"session": "seeded"})
            if url_path == MODEL_PATH:
                return _http(model_json, cookies={"session": "after-turn"})
            return _http({**json, "reward": 1.0})

        agent.server_client.post = AsyncMock(side_effect=post)
        body = SimpleAgentRunRequest(responses_create_params=_body())
        result = await agent.run(SimpleNamespace(cookies={"in": "1"}), body)

        seed_call, turn_call, verify_call = calls
        assert (seed_call["server_name"], seed_call["cookies"]) == ("turn_sim", {"in": "1"})
        # The episode reads the row's harness from /harness with these cookies: nothing else travels with the turn.
        assert (turn_call["server_name"], turn_call["url_path"], turn_call["cookies"]) == (
            "turn_sim_agent",
            MODEL_PATH,
            {"session": "seeded"},
        )
        assert not turn_call.get("headers")
        assert (verify_call["url_path"], verify_call["cookies"]) == ("/verify", {"session": "after-turn"})
        assert result.reward == 1.0 and result.response.metadata["stop_reason"] == "final_message"


class TestUnsupportedEpisodes:
    """Environment Server agent sessions are refused at seed, and /run without executing tools before any call."""

    def test_agent_sessions_are_refused_at_seed(self) -> None:
        agent, _ = _agent()
        client = TestClient(agent.setup_webserver())
        seed = AgentSeedSessionRequest(
            agent_session_id="agent-session",
            episode_id=EpisodeId(rollout_id="rollout", attempt=0),
            task_id=TaskId(taskset="example", task_id="0"),
        )
        with pytest.raises(NotImplementedError, match="supports only the legacy /run route"):
            client.post("/v1/agent_sessions", json=seed.model_dump(mode="json"))
        agent.server_client.post.assert_not_awaited()

    async def test_run_without_executing_tools(self) -> None:
        agent, _ = _agent(execute_tools=False)
        with pytest.raises(ValueError, match="execute_tools=false"):
            await agent.run(SimpleNamespace(cookies={}), SimpleAgentRunRequest(responses_create_params=_body()))
        agent.server_client.post.assert_not_awaited()


CLINE_STOP_TOOLS = ["ask_followup_question", "attempt_completion", "plan_mode_respond"]
CLINE_HARNESS = {"stop_tools": CLINE_STOP_TOOLS, "protocol": "cline_xml"}


def _cline_agent(*models: MagicMock, **kw: Any):
    return _agent(*models, harness=CLINE_HARNESS, **kw)


class TestClineXml:
    """Cline rows: tool calls are XML in the reply text, results come back as user messages."""

    async def test_the_first_xml_call_is_simulated_and_its_result_is_a_user_message(self) -> None:
        read = "Let me look at the config.\n\n<read_file>\n<path>config/app.json</path>\n</read_file>"
        framed = "[read_file for 'config/app.json'] Result:\n{\"debug\": false}"
        agent, script = _cline_agent(
            _model(_reasoning("private"), _msg(read)),
            _model(
                _msg("Debug mode is off.\n\n<attempt_completion>\n<result>It is off.</result>\n</attempt_completion>")
            ),
            tools=[_http({"output": framed, "source": "llm"})],
        )
        response, *_ = await _episode(agent)

        (tool_call,) = script.tool_calls
        assert {k: tool_call["json"][k] for k in ("call_id", "name", "arguments")} == {
            "call_id": "xml_1_1",
            "name": "read_file",
            "arguments": '{"path": "config/app.json"}',
        }
        assert _types(tool_call["json"]["turn"]) == ["message"] and "private" not in json.dumps(tool_call["json"])
        assert response.metadata["stop_reason"] == "attempt_completion" and response.metadata["steps"] == "2"
        assert _types(response.output) == ["reasoning", "message", "message", "message"]
        result = response.output[2]
        assert result.role == "user" and result.content == [{"type": "input_text", "text": framed}]
        assert script.model_calls[1]["json"].input[-1] == result

    async def test_a_reply_without_a_tool_gets_clines_error_and_the_model_retries(self) -> None:
        env = "<environment_details>\n# Current Mode\nACT MODE\n</environment_details>"
        body = _body()
        body.input = [
            NeMoGymEasyInputMessage(
                role="user",
                content=[
                    {"type": "input_text", "text": "<task>\nFix it\n</task>"},
                    {"type": "input_text", "text": env},
                ],
            )
        ]
        agent, script = _cline_agent(
            _model(_msg("First I'll read the file:")),
            _model(_msg("<read_file>\n<path>a.py</path>\n</read_file>")),
            _model(_msg("<attempt_completion>\n<result>Done.</result>\n</attempt_completion>")),
        )
        response, *_ = await _episode(agent, body=body)

        assert response.metadata["stop_reason"] == "attempt_completion" and response.metadata["steps"] == "3"
        nudge = response.output[1]
        assert nudge.role == "user" and nudge.content == [
            {"type": "input_text", "text": NO_TOOLS_USED},
            {"type": "input_text", "text": env},
        ]
        assert NO_TOOLS_USED.startswith("[ERROR] You did not use a tool in your previous response!")
        assert script.model_calls[1]["json"].input[-1] == nudge
        assert [c["json"]["name"] for c in script.tool_calls] == ["read_file"]

    async def test_reasoning_only_and_function_calls_count_as_no_tool(self) -> None:
        agent, script = _cline_agent(
            _model(_reasoning()),
            _model(_fc("bash", "c1", '{"command": "ls"}')),
            _model(_msg("<list_files>\n<path>.</path>\n</list_files>")),
            _model(_msg("<attempt_completion>\n<result>ok</result>\n</attempt_completion>")),
        )
        response, *_ = await _episode(agent)
        nudges = [
            o
            for o in response.output
            if o.type == "message" and o.role == "user" and o.content[0]["text"] == NO_TOOLS_USED
        ]
        assert len(nudges) == 2 and len(nudges[0].content) == 1  # no environment details in this history
        assert response.metadata["stop_reason"] == "attempt_completion" and len(script.tool_calls) == 1

    async def test_the_consecutive_mistake_limit_ends_the_turn(self) -> None:
        replies = [_model(_msg(f"Thinking {i}.")) for i in range(3)]
        agent, script = _cline_agent(*replies)
        response, *_ = await _episode(agent)
        assert response.metadata["stop_reason"] == "mistake_limit" and response.metadata["steps"] == "3"
        assert len(script.model_calls) == 3 and script.tool_calls == []
        # Two errors reached the model; the third reply ends the turn (Cline stops to ask the user).
        assert [(o.type, getattr(o, "role", None)) for o in response.output] == [
            ("message", "assistant"),
            ("message", "user"),
        ] * 2 + [("message", "assistant")]

    async def test_a_tool_call_resets_the_count_and_the_limit_is_configurable(self) -> None:
        replies = [
            _model(_msg("hm")),
            _model(_msg("<list_files>\n<path>.</path>\n</list_files>")),
            _model(_msg("hm")),
            _model(_msg("hm")),
        ]
        agent, _ = _cline_agent(*replies, cline_max_consecutive_mistakes=2)
        response, *_ = await _episode(agent)
        assert response.metadata["stop_reason"] == "mistake_limit" and response.metadata["steps"] == "4"
        assert AgenticTurnSimulationAgentConfig.model_fields["cline_max_consecutive_mistakes"].default == 3

    async def test_asking_the_user_ends_the_turn_unexecuted(self) -> None:
        question = "<ask_followup_question>\n<question>Which port?</question>\n</ask_followup_question>"
        agent, script = _cline_agent(_model(_msg(question)))
        response, *_ = await _episode(agent)
        assert response.metadata["stop_reason"] == "ask_followup_question" and script.tool_calls == []

    async def test_every_call_of_a_reply_runs_and_the_results_share_one_message(self) -> None:
        text = "<list_files>\n<path>.</path>\n</list_files>\n<read_file>\n<path>a.py</path>\n</read_file>"
        tools = [
            _http({"output": "[list_files for '.'] Result:\na.py"}),
            _http({"output": "[read_file for 'a.py'] Result:\n1 | x"}),
        ]
        agent, script = _cline_agent(
            _model(_msg(text)),
            _model(_msg("<attempt_completion>\n<result>Done.</result>\n</attempt_completion>")),
            tools=tools,
        )
        response, *_ = await _episode(agent)

        first, second = script.tool_calls
        assert [(c["json"]["call_id"], c["json"]["name"]) for c in (first, second)] == [
            ("xml_1_1", "list_files"),
            ("xml_1_2", "read_file"),
        ]
        # The second call is simulated with the first one's result, as the start of the reply's result message.
        assert _types(first["json"]["turn"]) == ["message"]
        assert second["json"]["turn"][-1] == {
            "content": [{"text": "[list_files for '.'] Result:\na.py", "type": "input_text"}],
            "role": "user",
            "type": "message",
        }
        (result,) = [o for o in response.output if o.type == "message" and o.role == "user"]
        assert result.content[0]["text"] == "[list_files for '.'] Result:\na.py\n[read_file for 'a.py'] Result:\n1 | x"

    async def test_the_calls_before_a_stop_tool_run_and_the_turn_ends_on_it(self) -> None:
        text = "<read_file>\n<path>a.py</path>\n</read_file>\n<attempt_completion>\n<result>ok</result>\n</attempt_completion>"
        tools = [_http({"output": "[read_file for 'a.py'] Result:\n1 | x"})]
        agent, script = _cline_agent(_model(_msg(text)), tools=tools)
        response, *_ = await _episode(agent)
        assert response.metadata["stop_reason"] == "attempt_completion" and len(script.model_calls) == 1
        assert [(c["json"]["call_id"], c["json"]["name"]) for c in script.tool_calls] == [("xml_1_1", "read_file")]
        assert _types(response.output) == ["message", "message"]
        assert response.output[-1].role == "user"
        assert response.output[-1].content[0]["text"] == "[read_file for 'a.py'] Result:\n1 | x"

    async def test_the_calls_after_a_stop_tool_never_run(self) -> None:
        question = "<ask_followup_question>\n<question>Which file?</question>\n</ask_followup_question>"
        agent, script = _cline_agent(_model(_msg(f"{question}\n<read_file>\n<path>a.py</path>\n</read_file>")))
        response, *_ = await _episode(agent)
        assert response.metadata["stop_reason"] == "ask_followup_question" and script.tool_calls == []
        assert _types(response.output) == ["message"]

    async def test_max_steps(self) -> None:
        steps = [_model(_msg(f"<execute_command>\n<command>ls {i}</command>\n</execute_command>")) for i in range(3)]
        agent, script = _cline_agent(*steps, max_steps=2)
        response, *_ = await _episode(agent)
        assert response.metadata["stop_reason"] == "max_steps" and len(script.tool_calls) == 2
        assert [c["json"]["call_id"] for c in script.tool_calls] == ["xml_1_1", "xml_2_1"]
        assert _types(response.output) == ["message", "message"] * 2

    async def test_tool_records_in_the_trajectory(self) -> None:
        agent, _ = _cline_agent(
            _model(_msg("<list_files>\n<path>src</path>\n</list_files>")),
            _model(_msg("<attempt_completion>\n<result>Done.</result>\n</attempt_completion>")),
        )
        _, trajectory, *_ = await _episode(agent, collect_trajectory=True)
        (record,) = trajectory.tool_calls
        assert (record.tool_call_id, record.tool_name, record.status) == ("xml_1_1", "list_files", "completed")
        assert trajectory.invocations[0].status == "completed"

    def test_the_protocol_reaches_the_episode_through_the_endpoint(self) -> None:
        agent, script = _cline_agent(
            _model(_msg("<read_file>\n<path>a</path>\n</read_file>")),
            _model(_msg("<attempt_completion>\n<result>Done.</result>\n</attempt_completion>")),
        )
        client = TestClient(agent.setup_webserver())
        reply = client.post("/v1/responses", json={"input": "hello"})
        assert reply.status_code == 200 and reply.json()["output"][1]["role"] == "user"
        assert reply.json()["metadata"]["stop_reason"] == "attempt_completion"
        assert [c["json"]["name"] for c in script.tool_calls] == ["read_file"]
