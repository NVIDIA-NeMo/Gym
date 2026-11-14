# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests with a scripted model server (no GPU, no network). The sandbox and the verifier are the real ones."""

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.responses_converter import VLLMConverter
from nemo_gym.server_utils import ServerClient
from resources_servers.arc_code.app import ArcCodeResourcesServer, ArcCodeResourcesServerConfig, ArcCodeVerifyRequest
from responses_api_agents.arc_code_agent import arc_prompt
from responses_api_agents.arc_code_agent.app import (
    ArcCodeAgent,
    ArcCodeAgentConfig,
    ArcCodeAgentRunRequest,
)


EXAMPLES = [
    {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
    {"input": [[5, 0], [0, 6]], "output": [[6, 0], [0, 5]]},
]
TEST_INPUT = [[1, 0], [2, 3]]
EXPECTED = [[3, 2], [0, 1]]  # the examples rotate by 180 degrees

CHECKED_CODE = "def transform(g):\n    return g[::-1, ::-1]\ncheck(transform)"
GOOD_ANSWER = "<answer>\n3 2\n0 1\n</answer>"


class _FakeHTTPResponse:
    def __init__(self, payload: Dict[str, Any]):
        self.payload, self.cookies, self.ok = payload, {}, True

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def model_reply(
    text: str, reasoning: Optional[str] = "thinking", out_tokens: int = 100, in_tokens: int = 3000
) -> dict:
    output: List[dict] = []
    if reasoning is not None:
        output.append({"id": "rs", "type": "reasoning", "summary": [{"type": "summary_text", "text": reasoning}]})
    output.append(
        {
            "id": "msg",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
    )
    return {
        "id": "r",
        "created_at": 1,
        "model": "scripted",
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": in_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": out_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": in_tokens + out_tokens,
        },
    }


def tool_reply(
    calls: List[tuple],
    text: str = "",
    reasoning: Optional[str] = "thinking",
    out_tokens: int = 100,
    in_tokens: int = 3000,
) -> dict:
    """A reply with native function calls: ``calls`` is a list of (call_id, name, arguments JSON string)."""
    reply = model_reply(text, reasoning, out_tokens, in_tokens)
    if not text:
        reply["output"] = [i for i in reply["output"] if i["type"] != "message"]
    for call_id, name, arguments in calls:
        reply["output"].append(
            {
                "id": f"fc-{call_id}",
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
                "status": "completed",
            }
        )
    return reply


def py(code: str) -> str:
    return json.dumps({"code": code})


class Harness:
    """Scripted model server plus the real resources server behind a fake ServerClient."""

    def __init__(self, replies: List[dict], **config: Any) -> None:
        self.replies = list(replies)
        self.model_calls: List[Any] = []
        cfg = ArcCodeAgentConfig(
            host="localhost",
            port=10001,
            entrypoint="app.py",
            name="arc_code_agent",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            resources_server=ResourcesServerRef(type="resources_servers", name="arc_code"),
            num_sandbox_workers=1,
            **config,
        )
        self.verifier = ArcCodeResourcesServer(
            config=ArcCodeResourcesServerConfig(host="localhost", port=10002, entrypoint="app.py", name="arc_code"),
            server_client=MagicMock(spec=ServerClient),
        )
        client = MagicMock(spec=ServerClient)
        client.post = self.post
        self.agent = ArcCodeAgent(config=cfg, server_client=client)

    async def post(self, *, server_name: str, url_path: str, json: Any, cookies: Any = None, **_: Any):
        if url_path == "/verify":
            result = await self.verifier.verify(ArcCodeVerifyRequest.model_validate(json))
            return _FakeHTTPResponse(result.model_dump())
        assert server_name == "policy_model" and url_path == "/v1/responses"
        self.model_calls.append(json)
        # The model server converts the request exactly as the wire carries it (exclude_unset): it must be convertible.
        VLLMConverter(return_token_id_information=False).responses_to_chat_completion_create_params(
            NeMoGymResponseCreateParamsNonStreaming.model_validate(json.model_dump(exclude_unset=True))
        )
        return _FakeHTTPResponse(self.replies.pop(0))

    async def run(self, expected: Optional[list] = None, rollout_index: int = 0):
        body = ArcCodeAgentRunRequest.model_validate(
            {
                "task_id": "puzzle-a",
                "examples": EXAMPLES,
                "test_input": TEST_INPUT,
                "expected_output": expected or EXPECTED,
                "responses_create_params": {"input": [{"role": "user", "content": "arc"}]},
                "_ng_rollout_index": rollout_index,
            }
        )
        return await self.agent.run(SimpleNamespace(cookies={}), body)


@pytest.fixture
async def make():
    made: List[Harness] = []

    def factory(replies: List[dict], **config: Any) -> Harness:
        made.append(Harness(replies, **config))
        return made[-1]

    yield factory
    for h in made:
        await h.agent._pool.close()


def roles(result) -> List[str]:
    return [getattr(i, "role", None) or i.type for i in result.response.output]


@pytest.mark.asyncio
async def test_happy_path_code_run_then_answer(make) -> None:
    h = make(
        [
            model_reply("Rotate by 180.\n<code>\nprint(test_inputs[0].shape)\n</code>"),
            model_reply(GOOD_ANSWER, out_tokens=50, in_tokens=3300),
        ]
    )
    result = await h.run()
    assert result.reward == 1.0
    assert result.turns == 2 and result.code_runs == 1 and result.cuts == 0
    assert result.finished_with_answer and result.answer_source == "typed"
    assert result.generated_tokens == 150
    assert result.predicted_output == EXPECTED

    # Second request: prompt, reasoning + assistant turn, then the tool turn with the sandbox output.
    second = h.model_calls[1]
    last = second.input[-1]
    assert last.role == "user" and last.content == "<tool_response>\n(2, 2)\n\n</tool_response>"
    assert any(getattr(i, "type", None) == "reasoning" for i in second.input)
    # Sampling and template kwargs travel as metadata the model server understands.
    first = h.model_calls[0]
    assert json.loads(first.metadata["chat_template_kwargs"]) == {"enable_thinking": True, "reasoning_effort": "xhigh"}
    extra = json.loads(first.metadata["extra_body"])
    assert extra["stop"] == ["</code>"] and extra["include_stop_str_in_output"] is True and extra["top_k"] == 20
    assert first.temperature == 0.7 and first.max_output_tokens == 73728 - 4096 - 64 - 0
    # Prompt: system, then one user message with a digit text and an image per grid (2 pairs + test = 5 grids).
    assert first.input[0].role == "system"
    parts = first.input[1].content
    assert sum(p["type"] == "input_image" for p in parts) == 5
    assert parts[1]["image_url"].startswith("data:image/png;base64,")
    # The full history is returned: prompt (with images) in responses_create_params, turns in response.output.
    assert result.responses_create_params.input[1].content[1]["type"] == "input_image"
    assert roles(result) == ["reasoning", "assistant", "user", "reasoning", "assistant"]


@pytest.mark.asyncio
async def test_auto_answer_from_passing_check(make) -> None:
    h = make([model_reply(f"<code>\n{CHECKED_CODE}\n</code>")])
    result = await h.run()
    assert result.reward == 1.0 and result.answer_source == "auto" and result.finished_with_answer
    assert len(h.model_calls) == 1
    tool_turn = result.response.output[2]
    assert "check: passed 2/2" in tool_turn.content
    assert result.response.output[-1].content[0].text == GOOD_ANSWER


@pytest.mark.asyncio
async def test_nudge_when_turn_has_no_code(make) -> None:
    h = make(
        [model_reply("I think the rule is a rotation."), model_reply(GOOD_ANSWER)],
        nocode_nudge=1,
    )
    result = await h.run()
    assert result.reward == 1.0 and result.turns == 2 and result.code_runs == 0
    nudge = h.model_calls[1].input[-1]
    assert nudge.role == "user" and nudge.content == arc_prompt.NOCODE_NUDGE


@pytest.mark.asyncio
async def test_no_code_turn_ends_attempt_by_default(make) -> None:
    h = make([model_reply("I think the rule is a rotation.")])
    result = await h.run()
    assert result.reward == 0.0 and not result.finished_with_answer and result.turns == 1


@pytest.mark.asyncio
async def test_cut_turn_gets_nudge(make) -> None:
    h = make([model_reply("", reasoning="long thinking", out_tokens=1000), model_reply(GOOD_ANSWER)], turn_cap=1000)
    result = await h.run()
    assert result.cuts == 1 and result.reward == 1.0
    nudge = h.model_calls[1].input[-1]
    assert nudge.content == arc_prompt.CUT_NUDGE
    # The cut turn (reasoning only) keeps an assistant turn boundary before the nudge.
    assert roles(result) == ["reasoning", "assistant", "user", "reasoning", "assistant"]


@pytest.mark.asyncio
async def test_budget_forces_final_answer_with_thinking_off(make) -> None:
    h = make(
        [
            model_reply("<code>\nprint(1)\n</code>", out_tokens=3900, in_tokens=1500),
            model_reply(GOOD_ANSWER, reasoning=None, out_tokens=30, in_tokens=5500),
        ],
        cap=5000,
        answer_tokens=600,
    )
    result = await h.run()
    assert result.forced_final and result.answer_source == "forced" and result.reward == 1.0
    assert result.turns == 1
    final_request = h.model_calls[1]
    assert json.loads(final_request.metadata["chat_template_kwargs"]) == {"enable_thinking": False}
    assert "stop" not in json.loads(final_request.metadata["extra_body"])
    assert final_request.max_output_tokens == 600
    assert final_request.input[-1].content == arc_prompt.FINAL_NUDGE


@pytest.mark.asyncio
async def test_sandbox_timeout_is_reported_and_cell_dropped(make) -> None:
    h = make(
        [
            model_reply("<code>\nx = 5\nwhile True:\n    pass\n</code>"),
            model_reply("<code>\nprint(x)\n</code>"),
            model_reply(GOOD_ANSWER),
        ],
        timeout_s=2,
    )
    result = await h.run()
    first_tool = h.model_calls[1].input[-1].content
    assert "timed out after 2s" in first_tool and "the cell was discarded" in first_tool
    # The dropped cell is not replayed, so `x` from it is undefined in the next run.
    second_tool = h.model_calls[2].input[-1].content
    assert "NameError" in second_tool
    assert result.code_runs == 2 and result.reward == 1.0


@pytest.mark.asyncio
async def test_sandbox_blocks_open_and_truncates_output(make) -> None:
    h = make(
        [
            model_reply("<code>\nprint('a' * 6000)\n</code>"),
            model_reply("<code>\nopen('/etc/passwd')\n</code>"),
            model_reply("<code>\nimport socket\n</code>"),
            model_reply(GOOD_ANSWER),
        ]
    )
    await h.run()
    assert "[truncated 2001 chars]" in h.model_calls[1].input[-1].content
    assert "NameError" in h.model_calls[2].input[-1].content or "not defined" in h.model_calls[2].input[-1].content
    assert "import of 'socket' is not allowed" in h.model_calls[3].input[-1].content


@pytest.mark.asyncio
async def test_malformed_answer_gets_reward_zero(make) -> None:
    h = make([model_reply("<answer>\n3 2\n0\n</answer>")], answer_resend=0)
    result = await h.run()
    assert result.reward == 0.0
    assert result.predicted_output is None and not result.extraction_successful
    assert not result.finished_with_answer and result.answer_source == "none"


@pytest.mark.asyncio
async def test_malformed_answer_is_resent_once(make) -> None:
    h = make([model_reply("<answer>\n3 2\n0\n</answer>"), model_reply(GOOD_ANSWER)], answer_resend=1)
    result = await h.run()
    assert result.reward == 1.0 and result.resends == 1
    assert "different lengths" in h.model_calls[1].input[-1].content


@pytest.mark.asyncio
async def test_wrong_grid_gets_reward_zero_but_counts_as_answered(make) -> None:
    h = make([model_reply("<answer>\n0 0\n0 0\n</answer>")])
    result = await h.run()
    assert result.reward == 0.0 and result.finished_with_answer and result.predicted_output == [[0, 0], [0, 0]]


@pytest.mark.asyncio
async def test_run_cap_goes_to_forced_final(make) -> None:
    h = make(
        [model_reply("<code>\nprint(1)\n</code>"), model_reply("<code>\nprint(2)\n</code>"), model_reply(GOOD_ANSWER)],
        max_runs=1,
    )
    result = await h.run()
    assert result.code_runs == 1 and result.forced_final and result.reward == 1.0


def test_system_prompt_matches_settings() -> None:
    prompt = arc_prompt.build_system_prompt(73728, 24, 4000)
    assert "about 73728 tokens" in prompt and "up to 24 runs" in prompt
    assert "Printed output beyond 4000 characters" in prompt and "`panels(g, sep=None)`" in prompt
    assert "`examples`" in prompt and "also preloaded: `ndimage`" in prompt


def test_image_rendering_matches_pad32_style() -> None:
    img = arc_prompt.render_grid([[1, 2, 3], [0, 0, 9]], 336, 12, "pad32")
    assert img.size == (64, 32)  # 3x2 cells of 12 px, padded to multiples of 32
    assert img.getpixel((0, 0)) == (0, 116, 217) and img.getpixel((12, 0)) == (255, 65, 54)
    assert img.getpixel((35, 23)) == (135, 12, 37) and img.getpixel((40, 0)) == arc_prompt.CANVAS


# -- native function calling ----------------------------------------------------------------------------------------


def kinds(result) -> List[str]:
    return [i.type if getattr(i, "type", None) != "message" else f"message:{i.role}" for i in result.response.output]


def outputs(request) -> List[Any]:
    """The function_call_output items of a model request."""
    return [i for i in request.input if getattr(i, "type", None) == "function_call_output"]


@pytest.mark.asyncio
async def test_native_tool_call_then_answer(make) -> None:
    h = make(
        [
            tool_reply([("call_1", "python", py("print(test_inputs[0].shape)"))], reasoning="plan"),
            model_reply(GOOD_ANSWER, out_tokens=50, in_tokens=3300),
        ],
        native_tools=True,
    )
    result = await h.run()
    assert result.reward == 1.0 and result.turns == 2 and result.code_runs == 1
    assert result.answer_source == "typed" and result.finished_with_answer and result.generated_tokens == 150

    # The request offers the python tool, has no stop string, and the system prompt names the tool.
    first = h.model_calls[0]
    assert [t["name"] for t in first.tools] == ["python"] and first.tool_choice == "auto"
    assert first.tools[0]["parameters"]["required"] == ["code"]
    assert "stop" not in json.loads(first.metadata["extra_body"])
    system = first.input[0].content
    assert "calling the `python` tool" in system and "<code>" not in system and "</code>" not in system
    # The second request replays reasoning and the call, then the function result with the same call id.
    second = h.model_calls[1]
    assert [getattr(i, "type", None) for i in second.input[2:]] == [
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    out = second.input[-1]
    assert out.call_id == "call_1" and out.output == "(2, 2)\n" and "tool_response" not in out.output
    # The returned history holds everything for export.
    assert kinds(result) == ["reasoning", "function_call", "function_call_output", "reasoning", "message:assistant"]


@pytest.mark.asyncio
async def test_native_parallel_tool_calls_share_sandbox_state(make) -> None:
    h = make(
        [
            tool_reply([("a", "python", py("y = 7")), ("b", "python", py("print(y + 1)"))]),
            model_reply(GOOD_ANSWER),
        ],
        native_tools=True,
    )
    result = await h.run()
    assert result.code_runs == 2 and result.turns == 2 and result.reward == 1.0
    got = {o.call_id: o.output for o in outputs(h.model_calls[1])}
    assert got == {"a": "", "b": "8\n"}
    assert kinds(result)[:4] == ["reasoning", "function_call", "function_call", "function_call_output"]


@pytest.mark.asyncio
async def test_native_bad_arguments_are_returned_as_errors_and_count_as_runs(make) -> None:
    h = make(
        [
            tool_reply([("a", "python", "{not json"), ("b", "python", json.dumps({"src": "1"})), ("c", "nope", "{}")]),
            model_reply(GOOD_ANSWER),
        ],
        native_tools=True,
    )
    result = await h.run()
    got = {o.call_id: o.output for o in outputs(h.model_calls[1])}
    assert "not valid JSON" in got["a"] and "non-empty string field `code`" in got["b"] and "unknown tool" in got["c"]
    assert all(v.startswith("[error]") for v in got.values())
    assert result.code_runs == 3 and result.reward == 1.0


@pytest.mark.asyncio
async def test_native_turn_without_call_or_answer_is_nudged(make) -> None:
    h = make(
        [model_reply("I think the rule is a rotation."), model_reply(GOOD_ANSWER)], native_tools=True, nocode_nudge=1
    )
    result = await h.run()
    assert result.reward == 1.0 and result.turns == 2 and result.code_runs == 0
    nudge = h.model_calls[1].input[-1]
    assert nudge.role == "user" and nudge.content == arc_prompt.NOCODE_NUDGE


@pytest.mark.asyncio
async def test_native_turn_without_call_or_answer_ends_attempt_by_default(make) -> None:
    h = make([model_reply("I think the rule is a rotation.")], native_tools=True)
    result = await h.run()
    assert result.reward == 0.0 and not result.finished_with_answer and result.answer_source == "none"


@pytest.mark.asyncio
async def test_native_auto_answer_from_passing_check(make) -> None:
    h = make([tool_reply([("a", "python", py(CHECKED_CODE))])], native_tools=True)
    result = await h.run()
    assert result.reward == 1.0 and result.answer_source == "auto" and len(h.model_calls) == 1
    assert kinds(result) == ["reasoning", "function_call", "function_call_output", "message:assistant"]
    assert "check: passed 2/2" in result.response.output[2].output
    assert result.response.output[-1].content[0].text == GOOD_ANSWER


@pytest.mark.asyncio
async def test_native_budget_forces_final_answer_with_thinking_off(make) -> None:
    h = make(
        [
            tool_reply([("a", "python", py("print(1)"))], out_tokens=3900, in_tokens=1500),
            model_reply(GOOD_ANSWER, reasoning=None, out_tokens=30, in_tokens=5500),
        ],
        native_tools=True,
        cap=5000,
        answer_tokens=600,
    )
    result = await h.run()
    assert result.forced_final and result.answer_source == "forced" and result.reward == 1.0 and result.turns == 1
    final = h.model_calls[1]
    assert json.loads(final.metadata["chat_template_kwargs"]) == {"enable_thinking": False}
    assert final.tool_choice == "none" and final.max_output_tokens == 600
    assert final.input[-1].content == arc_prompt.FINAL_NUDGE
    assert getattr(final.input[-2], "type", None) == "function_call_output"  # the call was answered before


@pytest.mark.asyncio
async def test_native_run_cap_answers_the_call_and_goes_to_forced_final(make) -> None:
    h = make(
        [
            tool_reply([("a", "python", py("print(1)"))]),
            tool_reply([("b", "python", py("print(2)")), ("c", "python", py("print(3)"))]),
            model_reply(GOOD_ANSWER),
        ],
        native_tools=True,
        max_runs=1,
    )
    result = await h.run()
    assert result.code_runs == 1 and result.forced_final and result.reward == 1.0
    got = {o.call_id: o.output for o in outputs(h.model_calls[2])}
    assert got["a"] == "1\n" and "limit of code runs" in got["b"] and "limit of code runs" in got["c"]


@pytest.mark.asyncio
async def test_native_sandbox_timeout_is_reported_and_cell_dropped(make) -> None:
    h = make(
        [
            tool_reply([("a", "python", py("x = 5\nwhile True:\n    pass"))]),
            tool_reply([("b", "python", py("print(x)"))]),
            model_reply(GOOD_ANSWER),
        ],
        native_tools=True,
        timeout_s=2,
    )
    result = await h.run()
    first = outputs(h.model_calls[1])[0].output
    assert "timed out after 2s" in first and "the cell was discarded" in first
    assert "NameError" in outputs(h.model_calls[2])[-1].output
    assert result.code_runs == 2 and result.reward == 1.0


@pytest.mark.asyncio
async def test_native_cut_turn_with_a_call_answers_it(make) -> None:
    h = make(
        [tool_reply([("a", "python", '{"code": "pri')], out_tokens=1000), model_reply(GOOD_ANSWER)],
        native_tools=True,
        turn_cap=1000,
    )
    result = await h.run()
    assert result.cuts == 1 and result.code_runs == 0 and result.reward == 1.0
    assert (
        "not run" in outputs(h.model_calls[1])[0].output and h.model_calls[1].input[-1].content == arc_prompt.CUT_NUDGE
    )


def test_native_system_prompt_differs_only_in_the_run_sentence() -> None:
    text = arc_prompt.build_system_prompt(73728, 24, 4000)
    native = arc_prompt.build_system_prompt(73728, 24, 4000, native_tools=True)
    assert text.replace(arc_prompt.TEXT_RUN_SENTENCE, arc_prompt.NATIVE_RUN_SENTENCE) == native


def test_harness_assistant_message_keeps_type_and_role_on_the_wire() -> None:
    # The request is serialized with exclude_unset; an item without `type` and `role` made the model server fail with
    # KeyError('type') after a reasoning-only cut turn. The wire form must carry both.
    from responses_api_agents.arc_code_agent.app import _assistant_message

    wire = _assistant_message("", "empty").model_dump(exclude_unset=True)
    assert wire["type"] == "message" and wire["role"] == "assistant"
    assert wire["content"][0]["type"] == "output_text"
    # The whole request, as the model server receives it, must validate and convert.
    request = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {"input": [{"role": "user", "content": "puzzle"}, wire, {"role": "user", "content": "continue"}]}
    )
    VLLMConverter(return_token_id_information=False).responses_to_chat_completion_create_params(request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "printed, expected",
    [
        ("5", None),  # a scalar made predict_answer raise TypeError and lose the rollout
        ("[1, 2]", None),
        ("[[1, 2], [3]]", None),
        ("[[1, 12]]", None),
        ('"text"', None),
        ("[[1, 2], [3, 4]]", [[1, 2], [3, 4]]),
    ],
)
async def test_predict_answer_returns_only_valid_grids(printed, expected) -> None:
    from responses_api_agents.arc_code_agent.arc_sandbox import SandboxPool

    pool = SandboxPool.__new__(SandboxPool)

    async def fake_run(header, codes, code, out_chars=None):
        return {"output": f"log line\n{printed}\n"}

    pool.run = fake_run
    assert await pool.predict_answer("", [], "transform") == expected


@pytest.mark.asyncio
async def test_sandbox_call_reports_a_failed_restart_instead_of_raising() -> None:
    from responses_api_agents.arc_code_agent import arc_sandbox

    server = arc_sandbox._Server("python", [])
    starts = []

    async def failing_start():
        starts.append(1)
        raise TimeoutError()

    server.start = failing_start
    # No process (an earlier restart failed): the call tries to start one and reports the failure.
    reply = await server.call({"id": 0}, wait_s=1)
    assert reply["output"] == "" and "sandbox unavailable" in reply["error"] and server.proc is None
    assert len(starts) == 1


@pytest.mark.asyncio
async def test_predict_answer_without_output_is_no_answer() -> None:
    from responses_api_agents.arc_code_agent.arc_sandbox import SandboxPool

    pool = SandboxPool.__new__(SandboxPool)

    async def fake_run(header, codes, code, out_chars=None):
        return {"timed_out": False}  # a reply that lacks "output"

    pool.run = fake_run
    assert await pool.predict_answer("", [], "transform") is None
