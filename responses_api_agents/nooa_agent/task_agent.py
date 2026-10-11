# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal generic NOOA agent and its Gym Responses invocation adapter."""

from __future__ import annotations

from typing import Any

from nooa import Agent, CodeActStrategy, strategy
from pydantic import BaseModel

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


class TaskAgent(Agent):
    """Solve a task using resource methods supplied by Gym at runtime."""

    @strategy(CodeActStrategy())
    async def solve_task(self, task: str) -> str:
        """Complete ``task`` by calling the available resource methods.

        Perform the requested actions against the real environment instead of
        simulating their results, then return a concise summary.
        """

        ...


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, dict):
        return value
    return {}


def _latest_user_text(request: NeMoGymResponseCreateParamsNonStreaming) -> str:
    if isinstance(request.input, str):
        if request.input.strip():
            return request.input
        raise ValueError("NOOA task input must not be empty")

    for raw_item in reversed(request.input):
        item = _as_dict(raw_item)
        if item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = []
            for raw_part in content:
                part = _as_dict(raw_part)
                if part.get("type") in {"input_text", "text"} and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            text = "\n".join(parts).strip()
            if text:
                return text
        raise ValueError("latest NOOA user message must contain text")
    raise ValueError("NOOA task input must contain a user message")


async def invoke_solve_task(agent: TaskAgent, request: NeMoGymResponseCreateParamsNonStreaming) -> object:
    """Translate standard Responses input into ``TaskAgent.solve_task``."""

    return await agent.solve_task(_latest_user_text(request))
