# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import io
from copy import deepcopy
from typing import Any

from PIL import Image

from nemo_gym.openai_utils import NeMoGymFunctionCallOutput
from resources_servers.webarena.schemas import (
    WebArenaEnvStateEasyInputMessage,
    WebArenaTaskRow,
)


TASK_INPUT_IMAGE_REDACTION_NOTICE = (
    "Task images have been redacted from this turn and are available in the first user turn."
)


def format_tab_context(context: dict[str, Any] | None) -> str:
    if context is None:
        return "Tab Context:\n- (unavailable)"
    lines = [
        "Tab Context:",
        f"- current_tab_id: {context.get('current_tab_id', -1)}",
        f"- tab_count: {context.get('tab_count', 0)}",
        "- available_tabs:",
    ]
    tabs = context.get("tabs") or []
    for tab in tabs:
        lines.append(
            "  - tab_id: {tab_id}, title: {title}, url: {url}".format(
                tab_id=tab.get("tab_id", -1),
                title=tab.get("title", "unknown"),
                url=tab.get("url", "unknown"),
            )
        )
    if not tabs:
        lines.append("  - (none)")
    return "\n".join(lines)


def screenshot_to_data_url(
    screenshot: bytes,
    *,
    image_format: str,
    jpeg_quality: int,
) -> str:
    normalized = image_format.upper()
    if normalized == "PNG":
        payload = screenshot
        mime = "png"
    elif normalized == "JPEG":
        image = Image.open(io.BytesIO(screenshot)).convert("RGB")
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
        payload = output.getvalue()
        mime = "jpeg"
    else:
        raise ValueError(f"Unsupported screenshot format: {image_format}")
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def build_observation_message(
    *,
    screenshot: bytes,
    task: WebArenaTaskRow,
    step: int,
    tab_context: dict[str, Any] | None,
    image_format: str,
    jpeg_quality: int,
    feedback: str | None = None,
) -> WebArenaEnvStateEasyInputMessage:
    text_parts: list[str] = []
    if step == 1 or task.task_input_images:
        text_parts.append(f"# Task Instruction:\n\n{task.intent}")
    if step > 1 and task.task_input_images:
        text_parts.append(TASK_INPUT_IMAGE_REDACTION_NOTICE)
    if feedback:
        text_parts.append(feedback)
    text_parts.append(f"You are currently on Step {step}.")
    text_parts.append(format_tab_context(tab_context))

    screenshot_part = {
        "type": "input_image",
        "image_url": screenshot_to_data_url(
            screenshot,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
        ),
        "detail": "auto",
    }
    if task.task_input_images and step == 1:
        content = [
            screenshot_part,
            {"type": "input_text", "text": f"# Task Instruction:\n\n{task.intent}"},
            *deepcopy(task.task_input_images),
            {"type": "input_text", "text": "\n\n".join(text_parts[1:])},
        ]
    else:
        content = [
            screenshot_part,
            {"type": "input_text", "text": "\n\n".join(text_parts)},
        ]
    return WebArenaEnvStateEasyInputMessage(
        role="user",
        content=content,
        step=step,
        tab_context=tab_context,
    )


def build_function_output(call_id: str, output: str) -> NeMoGymFunctionCallOutput:
    return NeMoGymFunctionCallOutput(
        type="function_call_output",
        call_id=call_id,
        output=output,
    )
