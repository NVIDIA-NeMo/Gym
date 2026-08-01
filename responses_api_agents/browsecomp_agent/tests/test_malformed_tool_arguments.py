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
"""A model tool call whose `arguments` is not valid JSON must not kill the rollout.

Regression test for the Inkling-Small eval (job 5748664, 2026-07-31), which died at
34/400 samples with:

    File ".../responses_api_agents/browsecomp_agent/app.py", line 492, in responses
        json=json.loads(output_function_call.arguments),
    json.decoder.JSONDecodeError: Expecting ',' delimiter: line 1 column 199 (char 198)

The parse was unguarded, so ONE malformed tool call raised out of the /responses
handler -> agent server 500 -> resources `/run` 500 -> `raise_for_status` ->
the whole `ng_e2e_collect_rollouts` process exited, taking all 400 samples with it.

The surrounding code already treats a bad model tool call as a recoverable, model-visible
error (see the comment at the `server_client.post` call: "it's a valid return for the API
to error e.g. if the model outputs an invalid call or something", and the 422s that get fed
back as tool output). Only the JSON parse of `arguments` was missing that treatment — on
BOTH branches: the resources-server tool path and the `update_progress` board path.
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
from responses_api_agents.browsecomp_agent.app import BrowsecompAgent, BrowsecompAgentConfig


SYSTEM = {"role": "system", "content": "BASE SYSTEM PROMPT."}
USER = {"role": "user", "content": "Q?"}

# Shaped like the payload that killed 5748664: a truncated JSON object (the model ran out of
# budget mid-arguments), which raises "Expecting ',' delimiter" rather than a clean EOF error.
MALFORMED_ARGS = '{"queries":["who won the 1998 prize","runner up 1998 prize""goal":"identify winner"}'


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


def _raw_fn_call(name: str, arguments: str, call_id: str = "call_001") -> NeMoGymResponseFunctionToolCall:
    """A function call carrying an ARBITRARY arguments string (valid JSON or not)."""
    return NeMoGymResponseFunctionToolCall(
        id="fc_001",
        call_id=call_id,
        name=name,
        arguments=arguments,
        type="function_call",
    )


def _model_response(outputs: list, response_id: str = "resp_001") -> dict:
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
            input_tokens=0,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
            output_tokens=0,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            total_tokens=0,
        ),
    ).model_dump()


class _FakeServerClient:
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


def _fn_call_outputs(result: NeMoGymResponse) -> list:
    return [o.output for o in result.output if getattr(o, "type", None) == "function_call_output"]


class TestMalformedArgumentsAreRecoverable:
    """The rollout survives and the model is told what went wrong."""

    async def test_resources_tool_with_malformed_args_does_not_raise(self) -> None:
        """The exact 5748664 crash: a resources-server tool call with unparseable arguments."""
        agent, _ = _make_agent(
            [
                _model_response([_raw_fn_call("search", MALFORMED_ARGS)]),
                _model_response([_make_msg("Exact Answer: X")], response_id="resp_002"),
            ]
        )
        result = await _run(agent, [SYSTEM, USER])
        assert result is not None

    async def test_malformed_args_reported_back_to_the_model(self) -> None:
        agent, _ = _make_agent(
            [
                _model_response([_raw_fn_call("search", MALFORMED_ARGS)]),
                _model_response([_make_msg("Exact Answer: X")], response_id="resp_002"),
            ]
        )
        result = await _run(agent, [SYSTEM, USER])
        outputs = _fn_call_outputs(result)
        assert len(outputs) == 1
        # Names the failure and the offending tool so the model can retry deliberately.
        assert "JSON" in outputs[0]
        assert "search" in outputs[0]

    async def test_malformed_args_do_not_reach_the_resources_server(self) -> None:
        """Nothing is posted for an uninterpretable call — there are no arguments to send."""
        agent, fake = _make_agent(
            [
                _model_response([_raw_fn_call("search", MALFORMED_ARGS)]),
                _model_response([_make_msg("Exact Answer: X")], response_id="resp_002"),
            ]
        )
        await _run(agent, [SYSTEM, USER])
        assert fake.tool_paths == []

    async def test_rollout_continues_after_a_malformed_call(self) -> None:
        """The model recovers on the next step and its answer is still returned."""
        agent, fake = _make_agent(
            [
                _model_response([_raw_fn_call("search", MALFORMED_ARGS)]),
                _model_response(
                    [_raw_fn_call("search", jsonlib.dumps({"queries": ["retry"]}), call_id="call_002")],
                    response_id="resp_002",
                ),
                _model_response([_make_msg("Exact Answer: X")], response_id="resp_003"),
            ]
        )
        result = await _run(agent, [SYSTEM, USER])
        assert fake.tool_paths == ["/search"]
        assert len(_fn_call_outputs(result)) == 2
        texts = [
            "".join(c.text for c in o.content if getattr(c, "type", None) == "output_text")
            for o in result.output
            if getattr(o, "type", None) == "message"
        ]
        assert any("Exact Answer: X" in t for t in texts)

    async def test_malformed_update_progress_args_do_not_raise(self) -> None:
        """Same defect on the board branch (app.py:473), which parses arguments separately."""
        agent, _ = _make_agent(
            [
                _model_response([_raw_fn_call("update_progress", MALFORMED_ARGS)]),
                _model_response([_make_msg("Exact Answer: X")], response_id="resp_002"),
            ],
            progress=True,
        )
        result = await _run(agent, [SYSTEM, USER])
        outputs = _fn_call_outputs(result)
        assert len(outputs) == 1
        assert "JSON" in outputs[0]


class TestWellFormedArgumentsUnaffected:
    """Regression guard: the happy paths must behave exactly as before."""

    async def test_valid_resources_tool_call_still_posts(self) -> None:
        agent, fake = _make_agent(
            [
                _model_response([_raw_fn_call("search", jsonlib.dumps({"queries": ["q"]}))]),
                _model_response([_make_msg("Exact Answer: X")], response_id="resp_002"),
            ]
        )
        result = await _run(agent, [SYSTEM, USER])
        assert fake.tool_paths == ["/search"]
        # The results_string envelope is still unwrapped for the model.
        assert _fn_call_outputs(result) == ["tool result"]

    async def test_valid_update_progress_still_updates_the_board(self) -> None:
        agent, fake = _make_agent(
            [
                _model_response([_raw_fn_call("update_progress", jsonlib.dumps({"board": "findings so far"}))]),
                _model_response([_make_msg("Exact Answer: X")], response_id="resp_002"),
            ],
            progress=True,
        )
        result = await _run(agent, [SYSTEM, USER])
        assert fake.tool_paths == []  # board writes stay inside the agent
        assert "Progress board updated" in _fn_call_outputs(result)[0]
