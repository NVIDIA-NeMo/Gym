# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for the progress-board feature (config.progress / update_progress).

Ported from bc_frankie_bash_tool_w_progress_tracking tests/test_progress_tracking.py
(commits 43e59bc, 5099589, 8cdd075). Key behaviours:
  * the update_progress tool is offered only when progress=True;
  * each call OVERWRITES the whole board (not appended);
  * the board is rendered into the system prompt and rebuilt on every context
    reset (never mid-segment), so it SURVIVES "discard all" resets;
  * crossing the reset threshold grants ONE warned board-save turn (a
    [SYSTEM NOTE] appended to the latest tool result) before the reset fires,
    on both the post-call and the tokenize pre-call reset paths;
  * an answer committed only to the board is recovered into the trajectory at
    return time so the judge can extract it.
"""

import json as jsonlib
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.browsecomp_agent.app import (
    PROGRESS_TOOL,
    BrowsecompAgent,
    BrowsecompAgentConfig,
)


SYSTEM = {"role": "system", "content": "BASE SYSTEM PROMPT."}
USER = {"role": "user", "content": "Q?"}


def _make_config(**kwargs) -> BrowsecompAgentConfig:
    defaults = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="test_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="test_resources"),
        model_server=ModelServerRef(type="responses_api_models", name="test_model"),
        nudge_steps=False,
    )
    return BrowsecompAgentConfig(**(defaults | kwargs))


def _make_msg(text: str, msg_id: str = "msg_001") -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id=msg_id,
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _make_fn_call(name: str, call_id: str = "call_001", args: dict | None = None) -> NeMoGymResponseFunctionToolCall:
    return NeMoGymResponseFunctionToolCall(
        id="fc_001",
        call_id=call_id,
        name=name,
        arguments=jsonlib.dumps(args or {}),
        type="function_call",
    )


def _progress_call(board: str, call_id: str = "c1") -> NeMoGymResponseFunctionToolCall:
    return _make_fn_call("update_progress", call_id=call_id, args={"board": board})


def _model_response(outputs: list, input_tokens: int = 0, response_id: str = "resp_001") -> dict:
    return NeMoGymResponse(
        id=response_id,
        created_at=0.0,
        model="test_model",
        object="response",
        output=outputs,
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        usage=NeMoGymResponseUsage(
            input_tokens=input_tokens,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
            output_tokens=0,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            total_tokens=input_tokens,
        ),
    ).model_dump()


class _FakeServerClient:
    """Serves queued model responses on /v1/responses and a canned envelope for
    every tool endpoint; records each model-call body (the exact
    NeMoGymResponseCreateParamsNonStreaming the agent sends) so tests can
    inspect the system prompt and tools the model would have seen."""

    def __init__(self, model_responses: list):
        self._model_responses = list(model_responses)
        self.model_bodies = []
        self.tool_paths = []

    async def post(self, server_name=None, url_path=None, json=None, cookies=None):
        http = MagicMock()
        http.ok = True
        http.status = 200
        http.cookies = {}
        if url_path == "/v1/responses":
            self.model_bodies.append(json)
            payload = self._model_responses.pop(0)
            http.read = AsyncMock(return_value=jsonlib.dumps(payload).encode())
        else:
            self.tool_paths.append(url_path)
            http.content.read = AsyncMock(return_value=b'{"results_string": "tool result"}')
        return http


def _make_agent(model_responses: list, **config_kwargs) -> tuple[BrowsecompAgent, _FakeServerClient]:
    fake = _FakeServerClient(model_responses)
    server_client = MagicMock(spec=ServerClient)
    server_client.post = fake.post
    return BrowsecompAgent(config=_make_config(**config_kwargs), server_client=server_client), fake


async def _run(agent: BrowsecompAgent, input_messages: list) -> NeMoGymResponse:
    request_mock = MagicMock()
    request_mock.cookies = {}
    response_mock = MagicMock()
    response_mock.set_cookie = MagicMock()
    body = NeMoGymResponseCreateParamsNonStreaming(input=input_messages)
    return await agent.responses(request_mock, response_mock, body)


def _tool_names(body: NeMoGymResponseCreateParamsNonStreaming) -> list:
    return [t["name"] if isinstance(t, dict) else t.name for t in body.tools]


def _fn_call_outputs(result: NeMoGymResponse) -> list:
    return [o.output for o in result.output if getattr(o, "type", None) == "function_call_output"]


def _last_message_text(result: NeMoGymResponse) -> str:
    for o in reversed(result.output):
        if getattr(o, "type", None) == "message":
            text = "".join(c.text for c in o.content if getattr(c, "type", None) == "output_text")
            if text:
                return text
    return ""


class TestGating:
    """The update_progress tool is offered only when progress=True."""

    async def test_progress_tool_present_when_enabled(self) -> None:
        agent, fake = _make_agent([_model_response([_make_msg("Exact Answer: X")])], progress=True)
        await _run(agent, [SYSTEM, USER])
        assert "update_progress" in _tool_names(fake.model_bodies[0])

    async def test_progress_tool_absent_when_disabled(self) -> None:
        agent, fake = _make_agent([_model_response([_make_msg("Exact Answer: X")])])
        await _run(agent, [SYSTEM, USER])
        assert "update_progress" not in _tool_names(fake.model_bodies[0])

    async def test_progress_tool_inserted_before_bash_command(self) -> None:
        """bc_frankie parity: active_tools = TOOLS + [PROGRESS_TOOL] + [BASH_TOOL],
        so update_progress must render before bash_command in the prompt."""
        agent, fake = _make_agent([_model_response([_make_msg("Exact Answer: X")])], progress=True)
        request_mock = MagicMock()
        request_mock.cookies = {}
        response_mock = MagicMock()
        response_mock.set_cookie = MagicMock()
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[SYSTEM, USER],
            tools=[
                {
                    "type": "function",
                    "name": name,
                    "description": "d",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": False,
                }
                for name in ("search", "browse", "bash_command")
            ],
        )
        await agent.responses(request_mock, response_mock, body)
        assert _tool_names(fake.model_bodies[0]) == ["search", "browse", "update_progress", "bash_command"]

    def test_progress_config_default_off(self) -> None:
        assert _make_config().progress is False


class TestSystemPromptAddendum:
    async def test_empty_board_renders_hint(self) -> None:
        agent, fake = _make_agent([_model_response([_make_msg("done")])], progress=True)
        await _run(agent, [SYSTEM, USER])
        sys_content = fake.model_bodies[0].input[0].content
        assert sys_content.startswith("BASE SYSTEM PROMPT.")
        assert "## Progress Board" in sys_content
        assert "start by listing the question's constraints" in sys_content

    async def test_no_addendum_when_disabled(self) -> None:
        agent, fake = _make_agent([_model_response([_make_msg("done")])])
        await _run(agent, [SYSTEM, USER])
        assert "## Progress Board" not in fake.model_bodies[0].input[0].content

    async def test_addendum_has_ledger_slots_and_no_leading_cue(self) -> None:
        """De-anchored board (bc_frankie Fix A): settled-state ledger slots, no
        example block (the '<- leading' cue was the anchor), and a post-reset
        re-derivation rule."""
        agent, fake = _make_agent([_model_response([_make_msg("done")])], progress=True)
        await _run(agent, [SYSTEM, USER])
        sys_content = fake.model_bodies[0].input[0].content
        assert "Confirmed facts" in sys_content
        assert "Ruled-out candidates" in sys_content
        assert "Search angles tried" in sys_content
        assert "UNVERIFIED" in sys_content
        assert "After a context reset" in sys_content
        assert "<- leading" not in sys_content
        assert "Example:" not in sys_content

    async def test_system_message_synthesized_when_absent(self) -> None:
        """A dataset row without a system message still gets the board pinned."""
        agent, fake = _make_agent([_model_response([_make_msg("done")])], progress=True)
        await _run(agent, [USER])
        first = fake.model_bodies[0].input[0]
        assert first.role == "system"
        assert "## Progress Board" in first.content

    def test_tool_description_frames_ledger_not_final_answer(self) -> None:
        desc = PROGRESS_TOOL["description"]
        assert "final answer" in desc.lower()
        assert "ruled-out" in desc.lower()


class TestBoardLifecycle:
    """Overwrite semantics + reset survival, driven through the post-call reset
    path. With progress on, crossing the threshold first grants a warned
    board-save turn; the reset fires after that turn. So the sequences use two
    over-threshold tool-call turns: one to arm, one to absorb the reset."""

    async def test_board_renders_and_survives_reset(self) -> None:
        board = "constraints: C1 | candidate: Jane Doe"
        steps = [
            _model_response([_progress_call(board, "a")], input_tokens=100),
            _model_response([_progress_call(board, "b")], input_tokens=999999),  # arm
            _model_response([_progress_call(board, "c")], input_tokens=999999),  # warned turn -> reset
            _model_response([_make_msg("Exact Answer: X")], input_tokens=100),
        ]
        agent, fake = _make_agent(steps, progress=True, context_reset_tokens=1000, max_reset_count=1)
        result = await _run(agent, [SYSTEM, USER])
        assert result.reset_count == 1
        assert board in fake.model_bodies[-1].input[0].content

    async def test_update_overwrites_previous_board(self) -> None:
        steps = [
            _model_response([_progress_call("FIRST board text", "a")], input_tokens=100),
            _model_response([_progress_call("SECOND board text", "b")], input_tokens=999999),  # arm
            _model_response([_progress_call("SECOND board text", "c")], input_tokens=999999),  # warned turn -> reset
            _model_response([_make_msg("Exact Answer: X")], input_tokens=100),
        ]
        agent, fake = _make_agent(steps, progress=True, context_reset_tokens=1000, max_reset_count=1)
        await _run(agent, [SYSTEM, USER])
        last_sys = fake.model_bodies[-1].input[0].content
        assert "SECOND board text" in last_sys
        assert "FIRST board text" not in last_sys

    async def test_board_not_rendered_mid_segment(self) -> None:
        """The board is re-rendered only at resets (bc_frankie parity): a write on
        turn 1 is not visible in the system prompt of turn 2."""
        steps = [
            _model_response([_progress_call("MID SEGMENT WRITE", "a")], input_tokens=100),
            _model_response([_make_msg("Exact Answer: X")], input_tokens=100),
        ]
        agent, fake = _make_agent(steps, progress=True, context_reset_tokens=1000)
        await _run(agent, [SYSTEM, USER])
        assert "MID SEGMENT WRITE" not in fake.model_bodies[1].input[0].content


class TestCommitGuard:
    """bc_frankie Fix B.1: the handler warns when the board contains an
    answer-commit marker instead of the model committing it as a message."""

    async def test_handler_warns_when_board_contains_answer_commit(self) -> None:
        steps = [
            _model_response([_progress_call("Constraints: C1\nExact Answer: Foo", "a")], input_tokens=100),
            _model_response([_make_msg("Exact Answer: Foo")], input_tokens=100),
        ]
        agent, _ = _make_agent(steps, progress=True)
        result = await _run(agent, [SYSTEM, USER])
        assert any("NOT your final answer" in o for o in _fn_call_outputs(result))

    async def test_handler_silent_on_normal_board(self) -> None:
        steps = [
            _model_response([_progress_call("Constraints: C1 | Ruled-out: John Roe", "a")], input_tokens=100),
            _model_response([_make_msg("Exact Answer: X")], input_tokens=100),
        ]
        agent, fake = _make_agent(steps, progress=True)
        result = await _run(agent, [SYSTEM, USER])
        outputs = _fn_call_outputs(result)
        assert any("Progress board updated" in o for o in outputs)
        assert not any("NOT your final answer" in o for o in outputs)
        # update_progress is handled by the agent, never posted to the resources server
        assert fake.tool_paths == []


class TestPreResetWarning:
    """The model cannot see a discard-all coming. With progress on, crossing the
    reset threshold injects a save-the-board warning and delays the reset by
    exactly one turn; the post-warning board write must survive the reset."""

    async def test_pre_reset_warning_gives_board_save_turn(self) -> None:
        steps = [
            _model_response([_progress_call("EARLY BOARD", "a")], input_tokens=5000),  # over -> arm + warn
            _model_response([_progress_call("SAVED AT WARNING", "b")], input_tokens=5000),  # warned turn -> reset
            _model_response([_make_msg("Exact Answer: X")], input_tokens=10),
        ]
        agent, fake = _make_agent(steps, progress=True, context_reset_tokens=1000, context_reset_keep_rounds=3)
        result = await _run(agent, [SYSTEM, USER])
        assert result.reset_count == 1
        assert result.pre_reset_warning_steps == [1]
        outputs = _fn_call_outputs(result)
        # The warning was injected into a tool result and names exactly what survives.
        assert any("about to be RESET" in o for o in outputs)
        assert any("the last 3 rounds and your progress board" in o for o in outputs)
        # The write made in the warned turn survives into the rebuilt system prompt.
        assert "SAVED AT WARNING" in fake.model_bodies[-1].input[0].content

    async def test_no_pre_reset_warning_without_progress(self) -> None:
        steps = [
            _model_response([_make_fn_call("search", call_id="a")], input_tokens=5000),  # over -> immediate reset
            _model_response([_make_msg("done")], input_tokens=10),
        ]
        agent, _ = _make_agent(steps, context_reset_tokens=1000)
        result = await _run(agent, [SYSTEM, USER])
        assert result.reset_count == 1
        assert result.pre_reset_warning_steps == []
        assert not any("about to be RESET" in o for o in _fn_call_outputs(result))

    async def test_precall_tokenize_path_warned_turn_then_reset(self, monkeypatch) -> None:
        """Same warned-turn semantics on the /tokenize pre-call reset path: the
        over-threshold estimate injects the warning (instead of resetting), the
        model gets one turn, and the next pre-call check fires the reset."""
        steps = [
            _model_response([_progress_call("EARLY", "a")]),
            _model_response([_progress_call("SAVED AT WARNING", "b")]),
            _model_response([_make_msg("Exact Answer: X")]),
        ]
        agent, fake = _make_agent(
            steps,
            progress=True,
            context_reset_tokens=1000,
            context_reset_keep_rounds=3,
            save_model_call_using_vllm_tokenize_endpoint=True,
        )
        # it1: 10 (under) -> turn 1; it2: 5000 (over -> arm + warn) -> turn 2 (warned);
        # it3: 5000 (over + armed -> reset), shrink probe n=3: 800 fits; it4: 500 -> final turn.
        monkeypatch.setattr(BrowsecompAgent, "_count_prompt_tokens", AsyncMock(side_effect=[10, 5000, 5000, 800, 500]))
        result = await _run(agent, [SYSTEM, USER])
        assert result.reset_count == 1
        assert result.pre_reset_warning_steps == [2]
        assert any("about to be RESET" in o for o in _fn_call_outputs(result))
        assert "SAVED AT WARNING" in fake.model_bodies[-1].input[0].content


class TestBoardAnswerFallback:
    """bc_frankie Fix B.2: recover an answer committed only to the board so the
    judge can extract it. Appends only — a trajectory that already commits an
    answer is returned unchanged."""

    async def test_fallback_recovers_answer_from_board(self) -> None:
        steps = [
            _model_response([_progress_call("Constraints: C1\nANSWER: The Sensation of Sight", "a")]),
            _model_response([_make_msg("I have recorded the answer on the board.")]),
        ]
        agent, _ = _make_agent(steps, progress=True)
        result = await _run(agent, [SYSTEM, USER])
        last = _last_message_text(result)
        assert "[Recovered from progress board]" in last
        assert "The Sensation of Sight" in last

    async def test_fallback_recovers_on_max_steps_fallthrough(self) -> None:
        """No assistant message at all when max_steps runs out -> a synthetic
        recovery message is appended so grading still sees the board's answer."""
        steps = [
            _model_response([_progress_call("ANSWER: Foo", "a")]),
            _model_response([_make_fn_call("search", call_id="b")]),
        ]
        agent, _ = _make_agent(steps, progress=True, max_steps=2)
        result = await _run(agent, [SYSTEM, USER])
        assert result.output[-1].type == "message"
        assert "Foo" in _last_message_text(result)

    async def test_fallback_inert_when_response_commits(self) -> None:
        steps = [
            _model_response([_progress_call("ANSWER: Foo", "a")]),
            _model_response([_make_msg("Exact Answer: Bar")]),
        ]
        agent, _ = _make_agent(steps, progress=True)
        result = await _run(agent, [SYSTEM, USER])
        last = _last_message_text(result)
        assert "Recovered from progress board" not in last
        assert "Foo" not in last

    async def test_fallback_inert_without_board_answer(self) -> None:
        steps = [
            _model_response([_progress_call("Constraints: C1", "a")]),
            _model_response([_make_msg("Some text without a commit marker.")]),
        ]
        agent, _ = _make_agent(steps, progress=True)
        result = await _run(agent, [SYSTEM, USER])
        assert _last_message_text(result) == "Some text without a commit marker."

    async def test_fallback_matches_prose_safe_markers_only(self) -> None:
        """The commit marker is line-anchored: 'candidate answer: ...' prose in
        the board must NOT trigger a recovery."""
        steps = [
            _model_response([_progress_call("notes: the candidate answer: maybe X, unverified", "a")]),
            _model_response([_make_msg("Still researching.")]),
        ]
        agent, _ = _make_agent(steps, progress=True)
        result = await _run(agent, [SYSTEM, USER])
        assert "Recovered from progress board" not in _last_message_text(result)
