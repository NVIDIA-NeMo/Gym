# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render visual-browser observations as Responses API multimodal messages."""

from __future__ import annotations

import re
from typing import Any

from nemo_gym.openai_utils import NeMoGymEasyInputMessage
from nemo_gym.web.models import WebObservation, WebObservationProfile, WebTask
from nemo_gym.web.task_images import resolve_task_image_url


VISUAL_OBSERVATION_TEXT_MODES = frozenset({"full_axtree", "som_only", "none"})
TASK_INPUT_IMAGE_REDACTION_NOTICE = (
    "Task images have been redacted from this turn and are available in the first user turn."
)
_BID_LINE = re.compile(r"^\s*\[[^\]]+\]\s+")
_SOM_MARKER = re.compile(r",\s*som(?=,|$)")


def _goal_text(goal: list[dict[str, Any]], fallback: str) -> str:
    texts = [str(item.get("text", "")) for item in goal if item.get("type") == "text"]
    return "\n".join(text for text in texts if text).strip() or fallback


def _goal_images(goal: list[dict[str, Any]]) -> list[str]:
    images: list[str] = []
    for item in goal:
        if item.get("type") != "image_url":
            continue
        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        if isinstance(image_url, str) and image_url:
            images.append(image_url)
    return images


def compact_som_text(axtree_text: str, *, max_chars: int = 12_000) -> str:
    """Keep only labelled interactive nodes when a backend supplies an AXTree."""

    retained: list[str] = []
    retained_chars = 0
    for raw_line in axtree_text.splitlines():
        line = raw_line.strip()
        if not _BID_LINE.match(line) or _SOM_MARKER.search(line) is None:
            continue
        line = _SOM_MARKER.sub("", line)
        if len(line) > 220:
            line = f"{line[:217]}..."
        additional = len(line) + (1 if retained else 0)
        if retained_chars + additional > max_chars:
            retained.append("[Additional labelled elements omitted.]")
            break
        retained.append(line)
        retained_chars += additional
    return "\n".join(retained)


def render_observation(
    observation: WebObservation,
    task: WebTask,
    *,
    step_index: int,
    visual_observation_text: str = "none",
    task_image_root: str | None = None,
    max_task_image_bytes: int = 25 * 1024 * 1024,
) -> NeMoGymEasyInputMessage:
    """Build one Nano-style visual computer-use turn.

    Qwen constructs its model-specific folded history in
    ``qwen_computer_use.QwenPolicyState``. Both adapters consume the same
    ``WebObservation`` and produce normalized environment actions.
    """

    if visual_observation_text not in VISUAL_OBSERVATION_TEXT_MODES:
        raise ValueError(f"unsupported visual observation text mode: {visual_observation_text}")

    text_parts: list[str] = []
    if step_index == 0 or task.input_images:
        text_parts.append(f"# Task Instruction:\n\n{_goal_text(observation.goal, task.intent)}")
    if step_index > 0 and task.input_images:
        text_parts.append(TASK_INPUT_IMAGE_REDACTION_NOTICE)
    text_parts.append(f"You are currently on Step {step_index + 1}.")
    tab_lines = [
        "Tab Context:",
        f"- current_tab_id: {observation.active_tab_index}",
        f"- tab_count: {len(observation.tabs)}",
        "- available_tabs:",
    ]
    tab_lines.extend(f"  - tab_id: {tab.index}, title: {tab.title}, url: {tab.url}" for tab in observation.tabs)
    if not observation.tabs:
        tab_lines.append("  - (none)")
    text_parts.append("\n".join(tab_lines))
    if observation.last_action:
        text_parts.append(f"Previous action: {observation.last_action}")
    if observation.last_action_error:
        text_parts.append(f"Previous action failed: {observation.last_action_error}")

    profile = task.observation_profile or WebObservationProfile.SCREENSHOT
    if observation.axtree_text and profile == WebObservationProfile.A11Y:
        text_parts.append(f"Accessibility tree (element ids are in brackets):\n{observation.axtree_text}")
    elif observation.axtree_text and visual_observation_text == "full_axtree":
        text_parts.append(f"Accessibility tree (element ids are in brackets):\n{observation.axtree_text}")
    elif observation.axtree_text and visual_observation_text == "som_only":
        compact_text = compact_som_text(observation.axtree_text)
        if compact_text:
            text_parts.append(f"Labelled interactive elements (ids match the screenshot):\n{compact_text}")

    content: list[dict[str, Any]] = []
    if observation.screenshot is not None and observation.screenshot.data_url:
        content.append({"type": "input_image", "image_url": observation.screenshot.data_url, "detail": "high"})

    image_references = [*_goal_images(observation.goal), *task.input_images] if step_index == 0 else []
    if image_references:
        content.append({"type": "input_text", "text": text_parts[0]})
        for index, image_reference in enumerate(image_references, start=1):
            image_url = resolve_task_image_url(
                image_reference,
                image_root=task_image_root,
                max_bytes=max_task_image_bytes,
            )
            content.extend(
                [
                    {
                        "type": "input_text",
                        "text": f"Task image {index} of {len(image_references)}:",
                    },
                    {"type": "input_image", "image_url": image_url, "detail": "high"},
                ]
            )
        content.append({"type": "input_text", "text": "\n\n".join(text_parts[1:])})
    else:
        content.append({"type": "input_text", "text": "\n\n".join(text_parts)})
    return NeMoGymEasyInputMessage(role="user", content=content)


__all__ = ["compact_som_text", "render_observation"]
