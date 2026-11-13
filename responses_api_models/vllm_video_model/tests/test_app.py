# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invented media: no decoder, backend, credentials or benchmark data."""

import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.core import NeMoGymChatCompletion
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


def model_converter(strategy="passthrough"):
    parent = MagicMock()
    parent.config.return_token_id_information = False
    parent.config.uses_reasoning_parser = True
    parent.config.custom_tool_strategy = strategy
    return VideoVLLMModel.get_converter(parent)


def test_parent_model_uses_opt_in_converter_without_replacing_its_loop():
    converter = model_converter()
    assert isinstance(converter, ToolMediaConverter)
    assert converter.custom_tool_strategy == "passthrough"


EXEC_TOOL = {
    "type": "custom",
    "name": "exec",
    "description": "Run invented code.",
    "format": {"type": "grammar", "syntax": "lark", "definition": "start: SOURCE\nSOURCE: /[\\s\\S]+/\n"},
}
EXEC_CODE = 'print("invented")'


def custom_tool_request(*history):
    return NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {"input": [{"role": "user", "content": "Run it."}, *history], "tools": [EXEC_TOOL], "model": "policy"}
    )


def test_configured_function_bridge_lowers_replays_and_lifts_custom_tools():
    converter = model_converter("function_bridge")
    body = custom_tool_request(
        {"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "exec", "input": EXEC_CODE},
        {"type": "custom_tool_call_output", "id": "ctco_1", "call_id": "call_1", "output": "done"},
    )
    chat = converter.responses_to_chat_completion_create_params(body)
    (tool,) = chat.tools
    assert tool["type"] == "function" and tool["function"]["name"] == "exec"
    assert tool["function"]["parameters"]["required"] == ["input"]
    assert tool["function"]["parameters"]["properties"]["input"]["type"] == "string"
    (call,) = next(message for message in chat.messages if message["role"] == "assistant")["tool_calls"]
    assert call["id"] == "call_1" and json.loads(call["function"]["arguments"]) == {"input": EXEC_CODE}
    assert next(message for message in chat.messages if message["role"] == "tool")["tool_call_id"] == "call_1"
    backend_call = {
        "id": "call_2",
        "type": "function",
        "function": {"name": "exec", "arguments": json.dumps({"input": EXEC_CODE})},
    }
    completion = NeMoGymChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "policy",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {"role": "assistant", "content": None, "tool_calls": [backend_call]},
                }
            ],
        }
    )
    (item,) = converter.chat_completion_to_response(body, completion).output
    assert item.type == "custom_tool_call" and item.call_id == "call_2" and item.input == EXEC_CODE


def test_passthrough_keeps_custom_tools_and_refuses_their_replay():
    converter = model_converter("passthrough")
    (tool,) = converter.responses_to_chat_completion_create_params(custom_tool_request()).tools
    assert tool["type"] == "custom" and tool["custom"]["name"] == "exec"
    replay = custom_tool_request(
        {"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "exec", "input": EXEC_CODE}
    )
    with pytest.raises(NotImplementedError, match="custom_tool_call"):
        converter.responses_to_chat_completion_create_params(replay)


@pytest.mark.parametrize(
    "parts",
    [
        [{"type": "input_image", "image_url": IMAGE, "detail": "original"}],
        [
            {"type": "input_text", "text": "window"},
            {"type": "input_video", "video_url": VIDEO},
            {"type": "input_image", "image_url": IMAGE, "detail": "original"},
        ],
    ],
)
def test_tool_image_original_detail_is_sent_as_high(parts):
    chat = ToolMediaConverter(return_token_id_information=False).responses_to_chat_completion_create_params(
        request(parts)
    )
    tool = chat.messages[-1]
    assert tool["role"] == "tool" and tool["tool_call_id"] == "original-call"
    assert [part["type"] for part in tool["content"]] == [
        {"input_text": "text", "input_video": "video_url", "input_image": "image_url"}[part["type"]] for part in parts
    ]
    assert tool["content"][-1]["image_url"] == {"url": IMAGE, "detail": "high"}


@pytest.mark.parametrize("detail", ["auto", "low", "high"])
def test_other_tool_image_details_are_unchanged(detail):
    chat = ToolMediaConverter(return_token_id_information=False).responses_to_chat_completion_create_params(
        request([{"type": "input_image", "image_url": IMAGE, "detail": detail}])
    )
    assert chat.messages[-1]["content"][0]["image_url"] == {"url": IMAGE, "detail": detail}


@pytest.mark.parametrize(
    "body",
    [
        # A user image keeps the shared converter's refusal of "original".
        NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": IMAGE, "detail": "original"}],
                }
            ]
        ),
        # A tool image without a URL has nothing to send.
        request([{"type": "input_image", "file_id": "file-invented", "detail": "original"}]),
    ],
)
def test_unrepresentable_images_still_fail(body):
    with pytest.raises(NotImplementedError):
        ToolMediaConverter(return_token_id_information=False).responses_to_chat_completion_create_params(body)
