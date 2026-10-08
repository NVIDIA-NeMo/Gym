# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from resources_servers.image_tools import (
    has_malformed_image_tool_markup,
)
from responses_api_agents.image_tools_agent.app import (
    ImageToolsAgent,
    ImageToolsAgentConfig,
    ImageToolsAgentRunRequest,
)


class _FakeClientResponse:
    def __init__(self, payload: dict[str, Any], cookies: dict[str, str] | None = None):
        self.payload = payload
        self.cookies = cookies or {}
        self.ok = True

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _assistant_response(
    *,
    response_id: str,
    text: str,
    prompt_token_ids: list[int],
    generation_token_ids: list[int],
) -> dict[str, Any]:
    return {
        "id": response_id,
        "created_at": 1,
        "model": "unit-model",
        "object": "response",
        "output": [
            {
                "id": f"{response_id}-message",
                "content": [
                    {
                        "annotations": [],
                        "text": text,
                        "type": "output_text",
                    }
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
                "prompt_token_ids": prompt_token_ids,
                "generation_token_ids": generation_token_ids,
                "generation_log_probs": [-0.1] * len(generation_token_ids),
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _base_verify_response(request_body: dict[str, Any], reward: float) -> dict[str, Any]:
    return {
        "responses_create_params": request_body["responses_create_params"],
        "response": request_body["response"],
        "reward": reward,
    }


def test_image_tool_markup_validation_allows_clean_xml() -> None:
    text = (
        "<tool_call><function=image_zoom_in_tool>"
        "<parameter=bbox_2d>[0, 0, 500, 500]</parameter>"
        "<parameter=label>upper left</parameter>"
        "<parameter=img_idx>0</parameter>"
        "</function></tool_call>"
    )

    assert not has_malformed_image_tool_markup(text)


def test_image_tool_markup_validation_rejects_nested_or_extra_tags() -> None:
    text = (
        "<tool_call><function=image_zoom_in_tool>"
        "<parameter=bbox_2d>[0, 0, 500, 500]</parameter>"
        "<parameter=label>bad\n</think>\n<tool_call>"
        "<function=image_zoom_in_tool>"
        "<parameter=bbox_2d>[10, 10, 100, 100]</parameter>"
        "<parameter=label>nested</parameter>"
        "<parameter=img_idx>0</parameter>"
        "</function></tool_call>"
    )

    assert has_malformed_image_tool_markup(text)


@pytest.mark.asyncio
async def test_image_tools_agent_runs_tool_loop_and_delegates_reward(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (256, 192), color=(120, 80, 40)).save(image_path)

    config = ImageToolsAgentConfig(
        host="localhost",
        port=10001,
        entrypoint="app.py",
        name="image_tools_simple_agent",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        resource_servers_by_agent={
            "string_match_simple_agent": ResourcesServerRef(
                type="resources_servers",
                name="string_match",
            )
        },
        crop_dir=str(tmp_path / "crops"),
        crop_format="jpeg",
        crop_min_pixels=262144,
        crop_max_pixels=1048576,
        tool_success_reward=0.02,
        tool_success_reward_cap=0.05,
    )

    tool_text = (
        "<tool_call><function=image_zoom_in_tool>"
        "<parameter=bbox_2d>[0, 0, 500, 500]</parameter>"
        "<parameter=label>upper left</parameter>"
        "<parameter=img_idx>0</parameter>"
        "</function></tool_call>"
    )
    final_text = "The answer is car."

    server_client_post = AsyncMock()

    async def _post_side_effect(*, server_name: str, url_path: str, **kwargs: Any):
        if url_path == "/seed_session":
            assert server_name == "string_match"
            return _FakeClientResponse({})
        if server_name == "policy_model" and url_path == "/v1/responses":
            call_index = server_client_post.await_count
            if call_index == 2:
                request = kwargs["json"]
                extra_body = json.loads(request.metadata["extra_body"])
                assert extra_body["stop"] == ["</tool_call>"]
                assert extra_body["include_stop_str_in_output"] is True
                return _FakeClientResponse(
                    _assistant_response(
                        response_id="tool",
                        text=tool_text,
                        prompt_token_ids=[1, 2, 3],
                        generation_token_ids=[10, 11, 12],
                    )
                )
            request = kwargs["json"]
            assert len(request.input) == 4
            assert request.input[-2].role == "assistant"
            assert request.input[-1].role == "user"
            return _FakeClientResponse(
                _assistant_response(
                    response_id="final",
                    text=final_text,
                    prompt_token_ids=[1, 2, 3, 10, 11, 12, 20],
                    generation_token_ids=[30, 31],
                )
            )
        if server_name == "string_match" and url_path == "/verify":
            verify_payload = kwargs["json"]
            assert len(verify_payload["response"]["output"]) == 1
            assert verify_payload["response"]["output"][0]["content"][0]["text"] == final_text
            return _FakeClientResponse(_base_verify_response(kwargs["json"], reward=1.0))
        raise AssertionError(f"Unexpected call: {server_name} {url_path}")

    server_client_post.side_effect = _post_side_effect
    server_client = MagicMock(spec=ServerClient)
    server_client.post = server_client_post

    agent = ImageToolsAgent(config=config, server_client=server_client)
    body = ImageToolsAgentRunRequest.model_validate(
        {
            "image_tools_base_agent_ref": {
                "type": "responses_api_agents",
                "name": "string_match_simple_agent",
            },
            "responses_create_params": {
                "input": [
                    {
                        "role": "system",
                        "content": "Use the image zoom tool if needed.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "What object is shown?"},
                            {
                                "type": "input_image",
                                "image_url": str(image_path),
                                "detail": "auto",
                            },
                        ],
                    },
                ],
            },
        }
    )
    result = await agent.run(SimpleNamespace(cookies={}), body)
    payload = result.model_dump(mode="json")
    assert payload["base_reward"] == 1.0
    assert payload["image_tools_aux_reward"] == 0.02
    assert payload["reward"] == 1.02
    assert payload["image_tools_call_count"] == 1
    assert payload["image_tools_error_count"] == 0
    assert len(payload["image_tools_output_paths"]) == 1
    assert Path(payload["image_tools_output_paths"][0]).exists()
    # Generation 0 produced no images; generation 1 carries the crop the tool wrote.
    assert payload["image_tools_generation_image_paths"][0] == []
    assert payload["image_tools_generation_image_paths"][1] == payload["image_tools_output_paths"]
    assert payload["image_tools_base_agent_ref"]["name"] == "string_match_simple_agent"

    output = payload["response"]["output"]
    assert len(output) == 3
    assert output[0]["role"] == "assistant"
    assert output[1]["role"] == "user"
    assert output[2]["role"] == "assistant"
    assert server_client_post.await_count == 4
