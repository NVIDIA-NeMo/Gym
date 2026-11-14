# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invented fixtures only; no real benchmark annotations or answers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from pydantic import BaseModel, ConfigDict

from nemo_gym.agents import verify_agent_response
from nemo_gym.config import ResourcesServerRef
from nemo_gym.core import NeMoGymResponse
from nemo_gym.eval.rollout_reverification import InputRolloutPair, _build_verify_payload
from nemo_gym.server_runtime import BaseRunRequest
from resources_servers.video_mcqa.dataset import VideoSource


REVISION = "a" * 40


def media(**updates: object) -> dict[str, VideoSource]:
    return {
        "shapes": VideoSource.model_validate(
            {
                "source_url": "https://example.invalid/shapes.mp4",
                "sha256": "b" * 64,
                "video_url": "https://example.invalid/prepared-video",
                **updates,
            }
        )
    }


def v1_record(identity: str = "one", **updates: object) -> dict[str, object]:
    return {
        "video_id": "001",
        "videoID": "shapes",
        "question_id": identity,
        "duration": "short",
        "domain": "Invented shapes",
        "sub_category": "Motion",
        "task_type": "Counting",
        "question": "Which invented shape moves?",
        "options": ["A. Circle", "B. Square", "C. Triangle", "D. Star"],
        "answer": "B",
        **updates,
    }


def lv_record(identity: str = "one", **updates: object) -> dict[str, object]:
    return {
        "key": "shapes",
        "type": "Invented shapes",
        "uid": identity,
        "question": "Which invented shape moves?\n(A) Circle\n(B) Square\n(C) Triangle\n(D) Star",
        "answer": "B",
        "question_type": ["entity recognition", "reasoning"],
        "time_reference": "SECRET_EVIDENCE_DO_NOT_FORWARD",
        **updates,
    }


def response(text: str | None = r"\boxed{B}", **updates: object) -> dict[str, object]:
    return {
        "id": "response",
        "created_at": 0,
        "object": "response",
        "model": "fixture",
        "output": []
        if text is None
        else [
            {
                "id": "message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "status": "completed",
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
        **updates,
    }


class RunRequest(BaseRunRequest):
    """The agent's run request, which carries the row's task fields to the verifier."""

    model_config = ConfigDict(extra="allow")


async def grade_failed_run_and_replay(server, request_model: type[BaseModel], row: dict, generated: dict):
    """Grade a failed agent run as Gym's agent helper does, then replay the row it saves."""

    async def post(*, server_name: str, url_path: str, json: dict, cookies: dict) -> SimpleNamespace:
        result = await server.verify(request_model.model_validate(json))
        return SimpleNamespace(ok=True, read=AsyncMock(return_value=result.model_dump_json().encode()))

    saved = await verify_agent_response(
        SimpleNamespace(post=post),
        ResourcesServerRef(type="resources_servers", name="verifier"),
        RunRequest.model_validate(row),
        NeMoGymResponse.model_validate(generated),
        {},
        force_zero_reward=True,
    )
    payload = _build_verify_payload(InputRolloutPair(input=row, rollout=saved))
    return saved, await server.verify(request_model.model_validate(payload))
