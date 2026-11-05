# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from PIL import Image

from nemo_gym.client import ServerClient
from resources_servers.image_tools.base import IMAGE_TOOLS_ROLLOUT_KEY
from resources_servers.image_tools.tool_reward_app import (
    ImageToolsRewardConfig,
    ImageToolsRewardServer,
    ImageToolsRewardVerifyRequest,
)


ZOOM = (
    "<tool_call><function=image_zoom_in_tool>"
    "<parameter=bbox_2d>[0, 0, 500, 500]</parameter>"
    "<parameter=label>upper left</parameter>"
    "<parameter=img_idx>0</parameter>"
    "</function></tool_call>"
)
UNKNOWN_TOOL = "<tool_call><function=image_teleport_tool><parameter=img_idx>0</parameter></function></tool_call>"


class _Reply:
    ok = True
    status = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _message(text: str, role: str = "assistant") -> dict[str, Any]:
    if role == "user":
        return {"type": "message", "role": "user", "content": text}
    return {
        "type": "message",
        "id": f"msg-{len(text)}",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _server(agent_settings: dict[str, Any], base_reward: float = 0.5) -> tuple[ImageToolsRewardServer, AsyncMock]:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {
        "image_tools_simple_agent": {
            "responses_api_agents": {
                "image_tools_agent": {
                    "resource_servers_by_agent": {
                        "string_match_simple_agent": {"type": "resources_servers", "name": "string_match"}
                    },
                    **agent_settings,
                }
            }
        }
    }

    async def verify(*, server_name: str, url_path: str, json: dict[str, Any]) -> _Reply:
        assert (server_name, url_path) == ("string_match", "/verify")
        return _Reply(json | {"reward": base_reward})

    client.post = AsyncMock(side_effect=verify)
    server = ImageToolsRewardServer(
        config=ImageToolsRewardConfig(
            host="localhost",
            port=9000,
            entrypoint="tool_reward_app.py",
            name="image_tools_reward",
            agent_server={"type": "responses_api_agents", "name": "image_tools_simple_agent"},
        ),
        server_client=client,
    )
    return server, client.post


def _request(image: Path, output: list[dict[str, Any]], reported_paths: list[str] | None = None) -> Any:
    response = {
        "id": "rollout",
        "created_at": 0,
        "model": "policy",
        "object": "response",
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }
    if reported_paths is not None:
        response[IMAGE_TOOLS_ROLLOUT_KEY] = {
            "image_tools_output_paths": reported_paths,
            "image_tools_generation_image_paths": [[], reported_paths],
        }
    return ImageToolsRewardVerifyRequest.model_validate(
        {
            "responses_create_params": {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "What object is shown?"},
                            {"type": "input_image", "image_url": str(image), "detail": "auto"},
                        ],
                    }
                ]
            },
            "response": response,
            "image_tools_base_agent_ref": {"type": "responses_api_agents", "name": "string_match_simple_agent"},
            "expected_answer": "car",
        }
    )


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "source.png"
    Image.new("RGB", (256, 192), color=(120, 80, 40)).save(path)
    return path


@pytest.mark.parametrize("base_reward,expected_reward", [(0.0, 0.02), (0.5, 0.52), (0.99, 1.0), (1.0, 1.0)])
async def test_the_reward_adds_the_replayed_tool_reward_capped_at_one(
    image: Path, base_reward: float, expected_reward: float
) -> None:
    server, post = _server({"tool_success_reward": 0.02, "tool_success_reward_cap": 0.05}, base_reward)
    output = [_message(ZOOM), _message("<tool_response>...</tool_response>", role="user"), _message("A car.")]

    result = (await server.verify(_request(image, output, reported_paths=["/agent/crop.jpeg"]))).model_dump()

    assert result["base_reward"] == base_reward
    assert result["image_tools_aux_reward"] == pytest.approx(0.02)
    assert result["reward"] == pytest.approx(expected_reward)
    assert result["image_tools_call_count"] == 1
    assert result["image_tools_error_count"] == 0
    assert result["image_tools_output_paths"] == ["/agent/crop.jpeg"]
    assert result["image_tools_generation_image_paths"] == [[], ["/agent/crop.jpeg"]]
    assert result["expected_answer"] == "car"
    # The verifier scored the final answer only; the result keeps the whole trajectory.
    sent = post.await_args.kwargs["json"]["response"]["output"]
    assert [item["content"][0]["text"] for item in sent] == ["A car."]
    assert len(result["response"]["output"]) == 3


async def test_a_terminating_invalid_call_costs_the_penalty_even_at_full_marks(image: Path) -> None:
    server, _ = _server({"invalid_tool_call_penalty": -0.05}, base_reward=1.0)

    result = (await server.verify(_request(image, [_message(UNKNOWN_TOOL)], reported_paths=[]))).model_dump()

    assert result["image_tools_aux_reward"] == pytest.approx(-0.05)
    assert result["reward"] == pytest.approx(0.95)
    assert result["image_tools_error_count"] == 1


async def test_settings_the_agent_leaves_unset_take_the_agents_defaults(image: Path) -> None:
    server, _ = _server({}, base_reward=0.0)
    output = [_message(ZOOM), _message("observation", role="user"), _message("A car.")]

    result = (await server.verify(_request(image, output))).model_dump()

    # ImageToolsSettings' default tool_success_reward, as the agent would have used.
    assert result["image_tools_aux_reward"] == pytest.approx(0.02)


async def test_a_replay_that_disagrees_with_the_agent_fails_verification(image: Path) -> None:
    server, _ = _server({}, base_reward=1.0)
    output = [_message(ZOOM), _message("observation", role="user"), _message("A car.")]

    with pytest.raises(RuntimeError, match="ran 1 image tools, but the agent reports 2"):
        await server.verify(_request(image, output, reported_paths=["/a.jpeg", "/b.jpeg"]))


async def test_a_row_without_a_known_base_agent_is_refused(image: Path) -> None:
    server, post = _server({})
    body = _request(image, [_message("A car.")]).model_copy(update={"image_tools_base_agent_ref": {"name": "other"}})

    with pytest.raises(HTTPException) as error:
        await server.verify(body)

    assert error.value.status_code == 422
    post.assert_not_awaited()
