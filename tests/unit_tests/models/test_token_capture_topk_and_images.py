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
"""Sampler top-k and tool-call images through local token capture and the projection.

Each test serves model calls through the capture middleware, reads the recorded entries back
from the store and rebuilds the rollout the way delivery does.
"""

import asyncio
from time import time
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import Body, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.client.server_client import ServerClient
from nemo_gym.core.responses_api_types import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.models.base_responses_api_model import BaseResponsesAPIModelConfig, SimpleResponsesAPIModel
from nemo_gym.models.chat_completions_converter import ChatCompletionsConverter
from nemo_gym.models.token_capture.builder import (
    assert_prefix_contiguity,
    prefix_merging,
    project_main_chain_response,
)
from nemo_gym.models.token_capture.consumer import trajectories_from_source
from nemo_gym.models.token_capture.records import (
    TOKEN_FIELDS,
    ParentResolutionStatus,
    TokenEntry,
    extract_token_fields,
    interstitial_image_sources,
)
from nemo_gym.models.token_capture.store import TokenCaptureStore


PROMPT = [1, 2, 3]
GENERATED = [4, 5]
LOG_PROBS = [-0.1, -0.2]
TOPK_IDS = [[4, 9], [5, 8]]
TOPK_LOG_PROBS = [[-0.1, -2.4], [-0.2, -1.7]]

INPUT_IMAGE = "data:image/png;base64,aW5wdXQ="
CROP = "data:image/png;base64,Y3JvcA=="
PLOT = "https://example.test/plot.png"


def _chat_completion(message: dict) -> NeMoGymChatCompletion:
    return NeMoGymChatCompletion.model_validate(
        {
            "id": f"chatcmpl_{uuid4().hex}",
            "created": int(time()),
            "model": "policy",
            "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        }
    )


class _TopkModel(SimpleResponsesAPIModel):
    """Serve one call whose generated tokens carry the sampler's top-k, in both dialects."""

    config: BaseResponsesAPIModelConfig
    model_config = {"arbitrary_types_allowed": True}

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model="policy",
            object="response",
            output=[
                {
                    "type": "message",
                    "id": f"msg_{uuid4().hex}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "hi", "annotations": []}],
                    "prompt_token_ids": PROMPT,
                    "generation_token_ids": GENERATED,
                    "generation_log_probs": LOG_PROBS,
                    "sampler_topk_token_ids": TOPK_IDS,
                    "sampler_topk_logprobs": TOPK_LOG_PROBS,
                }
            ],
            tool_choice="auto",
            parallel_tool_calls=True,
            tools=[],
        )

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        return _chat_completion(
            {
                "role": "assistant",
                "content": "hi",
                "prompt_token_ids": PROMPT,
                "generation_token_ids": GENERATED,
                "generation_log_probs": LOG_PROBS,
                "sampler_topk_token_ids": TOPK_IDS,
                "sampler_topk_logprobs": TOPK_LOG_PROBS,
            }
        )


# A vision agent's rollout: the first call sees the task image and asks a tool for a crop; the
# second call sees the crop the tool returned and answers. Each prompt extends the previous one.
_CROP_CALL = {"id": "call_crop", "type": "function", "function": {"name": "crop", "arguments": '{"box":[0,0,8,8]}'}}
_TASK_TURN = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Where is the cat?"},
        {"type": "image_url", "image_url": {"url": INPUT_IMAGE}},
    ],
}
_SECOND_REQUEST = [
    _TASK_TURN,
    {"role": "assistant", "content": None, "tool_calls": [_CROP_CALL]},
    {"role": "tool", "tool_call_id": "call_crop", "content": "Cropped the region."},
    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": CROP}}]},
]


class _ToolImageModel(SimpleResponsesAPIModel):
    """Answer the first request with a crop tool call and the second with the answer."""

    config: BaseResponsesAPIModelConfig
    model_config = {"arbitrary_types_allowed": True}

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        raise NotImplementedError("this rollout speaks Chat Completions")

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        if not any(message.get("role") == "assistant" for message in body.messages):
            return _chat_completion(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_CROP_CALL],
                    "prompt_token_ids": [1, 2, 3],
                    "generation_token_ids": [4, 5],
                    "generation_log_probs": [-0.1, -0.2],
                }
            )
        return _chat_completion(
            {
                "role": "assistant",
                "content": "On the sofa.",
                # The tool result's text and the crop's placeholder tokens sit between the calls.
                "prompt_token_ids": [1, 2, 3, 4, 5, 6, 7],
                "generation_token_ids": [8],
                "generation_log_probs": [-0.3],
            }
        )


def _client(tmp_path, model_cls: type) -> TestClient:
    server = model_cls(
        config=BaseResponsesAPIModelConfig(host="0.0.0.0", port=8099, entrypoint="", name="policy"),
        server_client=MagicMock(
            spec=ServerClient,
            global_config_dict={"token_id_capture": {"enabled": True, "dir": str(tmp_path)}},
        ),
    )
    return TestClient(server.setup_webserver())


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/responses", {"input": "hi"}),
    ],
)
def test_sampler_topk_round_trips_through_capture_and_projection(tmp_path, path, body):
    client = _client(tmp_path, _TopkModel)

    served = client.post(f"/ng-rollout/task0-topk/training-token-capture{path}", json=body)

    assert served.status_code == 200
    [entry] = TokenCaptureStore(tmp_path).read_entries("task0-topk")
    assert entry.sampler_topk_token_ids == TOPK_IDS
    assert entry.sampler_topk_logprobs == TOPK_LOG_PROBS
    # Stored once on the entry, like the token arrays, not again on the content items.
    for item in entry.output_items:
        assert not any(field in item for field in TOKEN_FIELDS)

    response = project_main_chain_response("task0-topk", prefix_merging([entry]), model="policy")

    [generated] = [item for item in response["output"] if item.get("generation_token_ids")]
    assert generated["generation_token_ids"] == GENERATED
    assert generated["sampler_topk_token_ids"] == TOPK_IDS
    assert generated["sampler_topk_logprobs"] == TOPK_LOG_PROBS


def test_a_later_calls_tool_image_reaches_the_rebuilt_rollout(tmp_path):
    """A multi-call chain in which a tool adds an image between the calls.

    The second call records the crop it saw after the first call's turn, and the rebuilt rollout
    places it right before that call's generated item. The task image is the rollout input and
    is not repeated.
    """
    client = _client(tmp_path, _ToolImageModel)
    first = client.post(
        "/ng-rollout/task0-crop/training-token-capture/v1/chat/completions", json={"messages": [_TASK_TURN]}
    )
    assert first.status_code == 200
    second = client.post(
        "/ng-rollout/task0-crop/training-token-capture/v1/chat/completions", json={"messages": _SECOND_REQUEST}
    )
    assert second.status_code == 200

    source = TokenCaptureStore(tmp_path)
    entries = {entry.generation_token_ids[0]: entry for entry in source.read_entries("task0-crop")}
    assert entries[4].interstitial_images is None
    assert entries[8].interstitial_images == [CROP]
    assert entries[8].parent_resolution == ParentResolutionStatus.RESOLVED
    assert entries[8].parent_call_id == entries[4].model_call_id

    built = asyncio.run(trajectories_from_source("task0-crop", source, model="policy"))

    assert not built["mask_sample"]
    output = built["rebuilt_response"]["output"]
    assert [item.get("role") for item in output] == ["assistant", "user", "assistant"]
    assert output[0]["generation_token_ids"] == [4, 5]
    assert output[1]["content"] == [{"type": "input_image", "image_url": CROP, "detail": "auto"}]
    assert "prompt_token_ids" not in output[1]
    assert output[2]["prompt_token_ids"] == [1, 2, 3, 4, 5, 6, 7]
    assert INPUT_IMAGE not in str(output)
    assert_prefix_contiguity(built["rebuilt_response"])


@pytest.mark.parametrize(
    ("request_items", "expected"),
    [
        # A root request: its images are the rollout input.
        ([_TASK_TURN], None),
        # Chat: the crop arrives in a user turn after the tool result.
        (_SECOND_REQUEST, [CROP]),
        # Chat: the image is a part of the tool message itself.
        (
            [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": None, "tool_calls": [_CROP_CALL]},
                {
                    "role": "tool",
                    "tool_call_id": "call_crop",
                    "content": [{"type": "text", "text": "done"}, {"type": "image_url", "image_url": {"url": CROP}}],
                },
            ],
            [CROP],
        ),
        # Responses: a user turn's image counts. An image part of a tool output does not: the
        # model server sends it as a text placeholder unless it inlines tool-output images.
        (
            [
                {"role": "user", "content": [{"type": "input_image", "image_url": INPUT_IMAGE}]},
                {"type": "reasoning", "summary": []},
                {"type": "function_call", "call_id": "c1", "name": "plot", "arguments": "{}"},
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": [{"type": "input_text", "text": "plotted"}, {"type": "input_image", "image_url": CROP}],
                },
                {"role": "user", "content": [{"type": "input_image", "image_url": PLOT}]},
            ],
            [PLOT],
        ),
        # Images before the last model turn were in the parent's prompt already; a tool output
        # that is a plain string is text to the model even when it spells a data URL.
        (
            [
                {"role": "user", "content": [{"type": "input_image", "image_url": INPUT_IMAGE}]},
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": CROP},
            ],
            [],
        ),
    ],
    ids=["root", "chat-user-turn", "chat-tool-part", "responses", "text-only-tail"],
)
def test_interstitial_image_sources_reads_every_dialect(request_items, expected):
    assert interstitial_image_sources(request_items) == expected


def test_extract_token_fields_carries_the_sampler_topk():
    fields = extract_token_fields(
        {
            "choices": [
                {
                    "message": {
                        "prompt_token_ids": PROMPT,
                        "generation_token_ids": GENERATED,
                        "generation_log_probs": LOG_PROBS,
                        "sampler_topk_token_ids": TOPK_IDS,
                        "sampler_topk_logprobs": TOPK_LOG_PROBS,
                    }
                }
            ]
        }
    )
    assert fields["sampler_topk_token_ids"] == TOPK_IDS
    assert fields["sampler_topk_logprobs"] == TOPK_LOG_PROBS


@pytest.mark.parametrize(
    ("topk_ids", "topk_log_probs", "error"),
    [
        (TOPK_IDS, None, "needs both"),
        (TOPK_IDS[:1], TOPK_LOG_PROBS[:1], "one row per generated token"),
        ([[4, 9], [5]], [[-0.1, -2.4], [-0.2]], "every row must have 2"),
        ([[4, 9], [5, 8]], [[-0.1, -2.4], [-0.2]], "every row must have 2"),
        ([[], []], [[], []], "must not be empty"),
    ],
)
def test_token_entry_rejects_a_malformed_sampler_topk(topk_ids, topk_log_probs, error):
    with pytest.raises(ValidationError, match=error):
        TokenEntry(
            capture_key="task0-topk",
            model_call_id="c1",
            prompt_token_ids=PROMPT,
            generation_token_ids=GENERATED,
            generation_log_probs=LOG_PROBS,
            sampler_topk_token_ids=topk_ids,
            sampler_topk_logprobs=topk_log_probs,
        )


def test_a_record_without_the_new_fields_still_reads():
    """Records written before these fields read with no top-k and no interstitial images."""
    entry = TokenEntry.model_validate(
        {
            "capture_key": "task0-old",
            "model_call_id": "c1",
            "prompt_token_ids": PROMPT,
            "generation_token_ids": GENERATED,
            "generation_log_probs": LOG_PROBS,
        }
    )
    assert entry.sampler_topk_token_ids is None
    assert entry.sampler_topk_logprobs is None
    assert entry.interstitial_images is None


def test_an_echoed_turn_sends_its_token_ids_but_not_its_sampler_topk_to_the_engine():
    converter = ChatCompletionsConverter(return_token_id_information=True)
    chat = converter.responses_to_chat_completion_create_params(
        NeMoGymResponseCreateParamsNonStreaming.model_validate(
            {
                "input": [
                    {"role": "user", "content": "hi"},
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "hi", "annotations": []}],
                        "prompt_token_ids": PROMPT,
                        "generation_token_ids": GENERATED,
                        "generation_log_probs": LOG_PROBS,
                        "sampler_topk_token_ids": TOPK_IDS,
                        "sampler_topk_logprobs": TOPK_LOG_PROBS,
                    },
                ]
            }
        )
    )
    [assistant] = [message for message in chat.messages if message["role"] == "assistant"]
    assert assistant["generation_token_ids"] == GENERATED
    assert "sampler_topk_token_ids" not in assistant
    assert "sampler_topk_logprobs" not in assistant
