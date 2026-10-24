# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invented media: no decoder, backend, credentials or benchmark data."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.openai_utils import NeMoGymFunctionCallOutput, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.responses_converter import ResponsesConverter
from responses_api_models.vllm_video_model.app import ToolMediaConverter, VideoVLLMModel


VIDEO = "data:video/mp4;base64,ZmFrZQ=="
IMAGE = "data:image/png;base64,ZmFrZQ=="


def request(parts):
    return NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {"role": "user", "content": "Watch the video."},
            {"type": "function_call", "call_id": "original-call", "name": "watch_video", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "original-call", "output": parts},
        ]
    )


@pytest.mark.parametrize(
    "parts",
    [
        [{"type": "input_video", "video_url": VIDEO}],
        [{"type": "input_image", "image_url": IMAGE, "detail": "auto"}],
        [
            {"type": "input_text", "text": "window"},
            {"type": "input_video", "video_url": VIDEO},
            {"type": "input_image", "image_url": IMAGE, "detail": "auto"},
        ],
    ],
)
def test_media_stays_in_original_tool_result(parts):
    converter = ToolMediaConverter(return_token_id_information=False)
    chat = converter.responses_to_chat_completion_create_params(request(parts)).model_dump(exclude_none=True)
    messages = chat["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant", "tool"]
    assert messages[1]["tool_calls"][0]["id"] == "original-call"
    assert messages[2]["tool_call_id"] == "original-call"
    restored = converter.chat_completions_messages_to_responses_items(messages)
    returned = [item for item in restored if item.type == "function_call_output"]
    assert len(returned) == 1 and returned[0].call_id == "original-call"
    assert returned[0].output == parts


@pytest.mark.parametrize(
    ("source", "url"),
    [
        ("/videos/a b.mp4", "file:///videos/a%20b.mp4"),
        ("file:///videos/a.mp4", "file:///videos/a.mp4"),
        (VIDEO, VIDEO),
    ],
)
def test_local_video_paths_are_sent_as_file_urls(source, url):
    params = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {
                "role": "user",
                "content": [{"type": "input_video", "video_url": source}, {"type": "input_text", "text": "Q"}],
            }
        ]
    )
    converter = ToolMediaConverter(return_token_id_information=False)
    chat = converter.responses_to_chat_completion_create_params(params).model_dump(exclude_none=True)
    assert chat["messages"][0]["content"][0] == {"type": "video_url", "video_url": {"url": url}}


def test_generic_converter_still_refuses_tool_video():
    converter = ResponsesConverter(return_token_id_information=False)
    with pytest.raises(NotImplementedError, match="input_video"):
        converter.responses_to_chat_completion_create_params(request([{"type": "input_video", "video_url": VIDEO}]))


def test_empty_video_source_is_rejected_by_schema():
    with pytest.raises(ValidationError):
        NeMoGymFunctionCallOutput(call_id="call", output=[{"type": "input_video", "video_url": ""}])


def test_text_only_conversion_is_unchanged():
    body = request("ordinary result")
    plain = ResponsesConverter(return_token_id_information=False)
    media = ToolMediaConverter(return_token_id_information=False)
    assert plain.responses_to_chat_completion_create_params(body) == media.responses_to_chat_completion_create_params(
        body
    )


def test_parent_model_uses_opt_in_converter_without_replacing_its_loop():
    parent = MagicMock()
    parent.config.return_token_id_information = False
    parent.config.uses_reasoning_parser = True
    converter = VideoVLLMModel.get_converter(parent)
    assert isinstance(converter, ToolMediaConverter)
