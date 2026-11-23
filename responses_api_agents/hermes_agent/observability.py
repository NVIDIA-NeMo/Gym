# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert Hermes conversation messages into Gym Responses input items."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from nemo_gym.core import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)


_SOURCE = "hermes"


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(getattr(part, "text", "")) for part in content
        )
    return "" if content is None else str(content)


def _input_image(part: Any) -> dict[str, str] | None:
    """Return a Hermes image part as a Responses ``input_image`` part, or None for any other part.

    Hermes reads both Chat Completions ``image_url`` parts, whose ``image_url`` is an object or a bare URL, and
    Responses ``input_image`` parts. An exported conversation must not fail on either form after a completed run.
    """
    if not isinstance(part, dict) or part.get("type") not in ("image_url", "input_image"):
        return None
    image = part.get("image_url")
    url, detail = (image.get("url"), image.get("detail")) if isinstance(image, dict) else (image, part.get("detail"))
    if not isinstance(url, str) or not url:
        return None
    return {"type": "input_image", "image_url": url, "detail": detail or "auto"}


def normalize_hermes_messages(messages: Iterable[Any], *, id_prefix: str = "hermes") -> list[NeMoGymResponseInputItem]:
    """Convert a Hermes conversation to ordered Gym Responses items."""
    output: list[NeMoGymResponseInputItem] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role, content = message.get("role"), _text(message.get("content"))
        if role in {"system", "user", "developer"}:
            raw_content = message.get("content")
            if role == "user" and isinstance(raw_content, list):
                images = [_input_image(part) for part in raw_content]
                if any(images):
                    content = [
                        image or {"type": "input_text", "text": _text([part])}
                        for part, image in zip(raw_content, images)
                    ]
            output.append(NeMoGymEasyInputMessage(role=role, content=content))
        elif role == "assistant":
            reasoning = message.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                output.append(
                    NeMoGymResponseReasoningItem(
                        id=f"{id_prefix}-reasoning-{index}",
                        summary=[NeMoGymSummary(text=reasoning, type="summary_text")],
                    )
                )
            output.append(
                NeMoGymResponseOutputMessage(
                    id=str(message.get("id") or f"{id_prefix}-message-{index}"),
                    content=[NeMoGymResponseOutputText(text=content, annotations=[])],
                )
            )
            for call in message.get("tool_calls") or []:
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict):
                    continue
                arguments = function.get("arguments", "")
                if not isinstance(arguments, str):
                    try:
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    except (TypeError, ValueError):
                        arguments = "{}"
                call_id = str(call.get("id") or "")
                output.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=arguments,
                        call_id=call_id,
                        id=call_id or None,
                        name=str(function.get("name") or ""),
                        status="completed",
                    )
                )
        elif role == "tool":
            output.append(
                NeMoGymFunctionCallOutput(
                    call_id=str(message.get("tool_call_id") or ""),
                    output=content,
                    status="completed",
                )
            )
    return output
