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
"""tool_images: images a tool returns reach the model in a user message after the round's tool outputs."""

import json as jsonlib
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
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
    TOOL_IMAGE_DROPPED,
    TOOL_IMAGES_HEADER,
    BrowsecompAgent,
    BrowsecompAgentConfig,
    _tool_image_message,
    _unwrap_tool_output,
)


SYSTEM = {"role": "system", "content": "BASE SYSTEM PROMPT."}
USER = {"role": "user", "content": "Q?"}
IMAGE_REPLY = {"results_string": "[Image]: https://example.com/a.png", "images": ["data:image/jpeg;base64,AAAA"]}


def _config(**kwargs) -> BrowsecompAgentConfig:
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


def _call(name: str, args: dict, call_id: str) -> NeMoGymResponseFunctionToolCall:
    return NeMoGymResponseFunctionToolCall(
        id=f"fc_{call_id}", call_id=call_id, name=name, arguments=jsonlib.dumps(args), type="function_call"
    )


def _answer(text: str = "Exact Answer: X") -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id="msg",
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _model_response(outputs: list) -> dict:
    return NeMoGymResponse(
        id="resp",
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
    """Answers view_image with IMAGE_REPLY and every other tool with plain text."""

    def __init__(self, model_responses: list):
        self._model_responses = list(model_responses)
        self.model_bodies = []

    async def post(self, server_name=None, url_path=None, json=None, cookies=None):
        http = MagicMock(ok=True, status=200, cookies={})
        if url_path == "/v1/responses":
            self.model_bodies.append(json)
            http.read = AsyncMock(return_value=jsonlib.dumps(self._model_responses.pop(0)).encode())
        else:
            reply = IMAGE_REPLY if url_path == "/view_image" else {"results_string": "tool result"}
            http.content.read = AsyncMock(return_value=jsonlib.dumps(reply).encode())
        return http


async def _run(model_responses: list, **config_kwargs) -> tuple[NeMoGymResponse, _FakeServerClient]:
    fake = _FakeServerClient(model_responses)
    server_client = MagicMock(spec=ServerClient)
    server_client.post = fake.post
    agent = BrowsecompAgent(config=_config(**config_kwargs), server_client=server_client)
    request, response = MagicMock(cookies={}), MagicMock()
    body = NeMoGymResponseCreateParamsNonStreaming(input=[SYSTEM, USER])
    return await agent.responses(request, response, body), fake


def _items(body) -> list:
    return [i if isinstance(i, dict) else i.model_dump() for i in body.input]


def _image_parts(items: list) -> list:
    return [
        part
        for item in items
        if item.get("role") == "user" and isinstance(item.get("content"), list)
        for part in item["content"]
        if part.get("type") == "input_image"
    ]


class TestUnwrap:
    def test_text_envelope_is_unwrapped(self) -> None:
        assert _unwrap_tool_output('{"results_string": "a\\nb"}', accept_images=True) == ("a\nb", [])

    def test_image_envelope_needs_the_flag(self) -> None:
        raw = jsonlib.dumps(IMAGE_REPLY)
        assert _unwrap_tool_output(raw, accept_images=False) == (raw, [])
        assert _unwrap_tool_output(raw, accept_images=True) == (IMAGE_REPLY["results_string"], IMAGE_REPLY["images"])

    def test_error_bodies_are_left_alone(self) -> None:
        raw = '{"detail": "Not Found"}'
        assert _unwrap_tool_output(raw, accept_images=True) == (raw, [])
        assert _unwrap_tool_output("not json", accept_images=True) == ("not json", [])


class TestToolImagesInContext:
    async def test_image_follows_the_tool_outputs_of_its_round(self) -> None:
        _, fake = await _run(
            [
                _model_response(
                    [
                        _call("view_image", {"url": "https://example.com/a.png"}, "c1"),
                        _call("search", {"queries": ["q"]}, "c2"),
                    ]
                ),
                _model_response([_answer()]),
            ],
            tool_images=True,
        )
        items = _items(fake.model_bodies[1])
        assert [i.get("type") for i in items[-3:]] == ["function_call_output", "function_call_output", "message"]
        assert items[-1]["content"][0]["text"] == TOOL_IMAGES_HEADER
        assert _image_parts(items) == [
            {"type": "input_image", "image_url": "data:image/jpeg;base64,AAAA", "detail": "auto"}
        ]

    async def test_images_stay_out_of_the_returned_trajectory(self) -> None:
        result, _ = await _run(
            [
                _model_response([_call("view_image", {"url": "https://example.com/a.png"}, "c1")]),
                _model_response([_answer()]),
            ],
            tool_images=True,
        )
        assert [o.type for o in result.output] == ["function_call", "function_call_output", "message"]
        assert result.output[1].output == IMAGE_REPLY["results_string"]

    async def test_without_the_flag_the_model_sees_json_and_no_image(self) -> None:
        _, fake = await _run(
            [
                _model_response([_call("view_image", {"url": "https://example.com/a.png"}, "c1")]),
                _model_response([_answer()]),
            ]
        )
        items = _items(fake.model_bodies[1])
        assert _image_parts(items) == []
        assert jsonlib.loads(items[-1]["output"]) == IMAGE_REPLY

    async def test_old_images_are_replaced_by_a_marker(self) -> None:
        rounds = [
            _model_response([_call("view_image", {"url": f"https://example.com/{n}.png"}, f"c{n}")]) for n in range(3)
        ]
        _, fake = await _run(rounds + [_model_response([_answer()])], tool_images=True, max_tool_images_in_context=2)
        items = _items(fake.model_bodies[-1])
        assert len(_image_parts(items)) == 2
        dropped = [
            p for i in items if i.get("role") == "user" and isinstance(i.get("content"), list) for p in i["content"]
        ]
        assert {"type": "input_text", "text": TOOL_IMAGE_DROPPED} in dropped

    async def test_milestone_nudge_lands_on_the_tool_output_not_the_image(self) -> None:
        # max_steps=4: the quarter mark is step 1, the first round.
        _, fake = await _run(
            [
                _model_response([_call("view_image", {"url": "https://example.com/a.png"}, "c1")]),
                _model_response([_answer()]),
            ],
            tool_images=True,
            nudge_steps=True,
            max_steps=4,
        )
        items = _items(fake.model_bodies[1])
        assert "[SYSTEM NOTE: You have used 1 out of 4 turns." in items[-2]["output"]
        assert items[-1]["role"] == "user"


class TestToolImagesSurviveContextReset:
    def test_kept_rounds_keep_their_images(self) -> None:
        agent = BrowsecompAgent(config=_config(tool_images=True), server_client=MagicMock(spec=ServerClient))
        image = _tool_image_message(["data:image/jpeg;base64,AAAA"])
        outputs = []
        for n in range(3):
            outputs += [
                _call("view_image", {"url": f"https://example.com/{n}.png"}, f"c{n}"),
                NeMoGymFunctionCallOutput(call_id=f"c{n}", output=f"[Image]: {n}"),
            ]
            if n != 1:
                outputs.append(image)
        kept = agent._extract_last_rounds(outputs, n=2)
        assert [getattr(i, "type", None) for i in kept] == ["function_call", "function_call_output"] + [
            "function_call",
            "function_call_output",
            "message",
        ]
        assert kept[-1] is image

    def test_text_rounds_are_unchanged(self) -> None:
        agent = BrowsecompAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        outputs = [
            _call("search", {"queries": ["q"]}, "c1"),
            NeMoGymFunctionCallOutput(call_id="c1", output="r1"),
            _answer("thinking"),
            _call("search", {"queries": ["q2"]}, "c2"),
            NeMoGymFunctionCallOutput(call_id="c2", output="r2"),
        ]
        assert agent._extract_last_rounds(outputs, n=5) == [outputs[0], outputs[1], outputs[3], outputs[4]]
