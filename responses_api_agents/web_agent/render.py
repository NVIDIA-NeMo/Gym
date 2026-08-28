# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render common web observations as Responses API multimodal messages."""

from __future__ import annotations

import re
from typing import Any

from nemo_gym.openai_utils import NeMoGymEasyInputMessage
from nemo_gym.web.models import WebActionProfile, WebObservation, WebObservationProfile, WebTask
from nemo_gym.web.task_images import resolve_task_image_url


BROWSERGYM_CODE_BLOCK_FORMAT = """## Action:
concise description of the next action
## Code:
```python
click('bid')
```"""


BROWSERGYM_STANDARD_ACTION_GUIDANCE = """Return one Action using BrowserGym high-level calls only.
Common calls:
- click('bid'), fill('bid', 'text'), select_option('bid', 'value'), hover('bid')
- scroll(0, 500), keyboard_press('Enter'), go_back(), go_forward(), goto('https://...')
- new_tab(), tab_focus(0), tab_close()
- send_msg_to_user('final answer') when the task is complete
- report_infeasible('reason') only when the task cannot be completed
You may return at most two calls on separate lines. Arguments must be literals; do not emit arbitrary Python.
Use this exact shape:
Thought: concise reasoning
Action: click('bid')"""


BROWSERGYM_CODE_BLOCK_ACTION_GUIDANCE = f"""Return one action using BrowserGym high-level calls only.
Common calls:
- click('bid'), fill('bid', 'text'), select_option('bid', 'value'), hover('bid')
- scroll(0, 500), keyboard_press('Enter'), go_back(), go_forward(), goto('https://...')
- new_tab(), tab_focus(0), tab_close()
- send_msg_to_user('final answer') when the task is complete
- report_infeasible('reason') only when the task cannot be completed
You may return at most two calls on separate lines. Arguments must be literals; do not emit arbitrary Python.
For element actions, use the exact id shown in square brackets in the accessibility tree or screenshot.
For example, `[297] link 'pics'` must be called as `click('297')`; never pass visible text such as `click('pics')`.
The Code block must contain the executable call, not a natural-language instruction.
Use exactly this response shape:
{BROWSERGYM_CODE_BLOCK_FORMAT}"""


WEBVOYAGER_ACTION_GUIDANCE = """Return exactly one WebVoyager-style Action:
- Click [bid]
- Type [bid]; [text] (typing also submits with Enter)
- Scroll [WINDOW]; up or Scroll [WINDOW]; down
- Wait, GoBack, or Google
- ANSWER; [final answer] when complete
For Click and Type, replace `bid` with the exact numeric id shown in the
labelled elements. For example, `[38] textbox 'Search'` must be
`Type [38]; [SimCSE]`; never omit the id or describe the action in prose.
Stop immediately after the Action line. Do not predict its result or write
future Thought/Action turns in the same response.
Use this exact shape:
Thought: concise reasoning
Action: Click [bid]"""


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


def action_guidance(task: WebTask, action_prompt_profile: str = "standard") -> str:
    if task.action_profile == WebActionProfile.NATIVE_TOOLCALL:
        return ""
    if task.action_profile == WebActionProfile.WEBVOYAGER_LEGACY:
        return WEBVOYAGER_ACTION_GUIDANCE
    if action_prompt_profile == "standard":
        return BROWSERGYM_STANDARD_ACTION_GUIDANCE
    if action_prompt_profile == "code_block":
        return BROWSERGYM_CODE_BLOCK_ACTION_GUIDANCE
    raise ValueError(f"unsupported action prompt profile: {action_prompt_profile}")


def compact_som_text(axtree_text: str, *, max_chars: int = 12_000) -> str:
    """Keep the BrowserGym-labelled interactive elements from a visual AXTree.

    BrowserGym marks the nodes actually represented in its SoM overlay with a
    ``som`` property.  WebVoyager's upstream prompt contains a similarly short
    list of labelled interactive elements, rather than the complete AXTree.
    """

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
    visual_observation_text: str = "full_axtree",
    action_prompt_profile: str = "standard",
    task_image_root: str | None = None,
    max_task_image_bytes: int = 25 * 1024 * 1024,
) -> NeMoGymEasyInputMessage:
    """Build one model turn without leaking raw BrowserGym Python objects."""

    if visual_observation_text not in VISUAL_OBSERVATION_TEXT_MODES:
        raise ValueError(f"unsupported visual observation text mode: {visual_observation_text}")

    profile = task.observation_profile
    if profile is None:
        profile = WebObservationProfile.SOM
    if task.action_profile == WebActionProfile.NATIVE_TOOLCALL:
        text_parts = []
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
    else:
        text_parts = [
            f"Task: {_goal_text(observation.goal, task.intent)}",
            f"Step: {step_index}",
            f"Current URL: {observation.url}",
        ]
    if observation.tabs and task.action_profile != WebActionProfile.NATIVE_TOOLCALL:
        text_parts.append(
            "Tabs:\n"
            + "\n".join(
                f"- [{tab.index}] {'ACTIVE ' if tab.active else ''}{tab.title} — {tab.url}" for tab in observation.tabs
            )
        )
    if observation.last_action:
        text_parts.append(f"Previous action: {observation.last_action}")
    if observation.last_action_error:
        text_parts.append(f"Previous action failed: {observation.last_action_error}")
    if observation.axtree_text and profile == WebObservationProfile.A11Y:
        text_parts.append(f"Accessibility tree (element ids are in brackets):\n{observation.axtree_text}")
    elif observation.axtree_text and visual_observation_text == "full_axtree":
        text_parts.append(f"Accessibility tree (element ids are in brackets):\n{observation.axtree_text}")
    elif observation.axtree_text and visual_observation_text == "som_only":
        compact_text = compact_som_text(observation.axtree_text)
        if compact_text:
            text_parts.append(f"Labelled interactive elements (ids match the screenshot):\n{compact_text}")
    guidance = action_guidance(task, action_prompt_profile)
    if guidance:
        text_parts.append(guidance)

    content: list[dict[str, Any]] = []
    # The native recipe places the current browser screenshot before text and
    # task-reference images. BrowserGym profiles retain their historical order.
    image_references = [*_goal_images(observation.goal), *task.input_images] if step_index == 0 else []
    if task.action_profile == WebActionProfile.NATIVE_TOOLCALL:
        screenshot = observation.screenshot
        if screenshot is not None and screenshot.data_url:
            content.append({"type": "input_image", "image_url": screenshot.data_url, "detail": "high"})
        if image_references:
            # The pinned native agent places reference images between the task
            # instruction and step/tab context, with a deterministic label for
            # each image. Screenshot-history compaction recognizes the
            # adjacent deterministic label and preserves the task image.
            content.append({"type": "input_text", "text": text_parts[0]})
            for index, image_reference in enumerate(image_references, start=1):
                image_url = resolve_task_image_url(
                    image_reference,
                    image_root=task_image_root,
                    max_bytes=max_task_image_bytes,
                )
                content.append(
                    {
                        "type": "input_text",
                        "text": f"Task image {index} of {len(image_references)}:",
                    }
                )
                content.append(
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": "high",
                    }
                )
            content.append({"type": "input_text", "text": "\n\n".join(text_parts[1:])})
        else:
            content.append({"type": "input_text", "text": "\n\n".join(text_parts)})
    else:
        content.append({"type": "input_text", "text": "\n\n".join(text_parts)})
    if step_index == 0 and task.action_profile != WebActionProfile.NATIVE_TOOLCALL:
        for image_reference in image_references:
            image_url = resolve_task_image_url(
                image_reference,
                image_root=task_image_root,
                max_bytes=max_task_image_bytes,
            )
            content.append({"type": "input_image", "image_url": image_url, "detail": "high"})
    if task.action_profile != WebActionProfile.NATIVE_TOOLCALL and profile in {
        WebObservationProfile.SCREENSHOT,
        WebObservationProfile.SOM,
    }:
        screenshot = observation.screenshot
        if screenshot is not None and screenshot.data_url:
            content.append(
                {
                    "type": "input_image",
                    "image_url": screenshot.data_url,
                    "detail": "high",
                }
            )
    return NeMoGymEasyInputMessage(role="user", content=content)


def parse_error_message(
    error: Exception,
    action_prompt_profile: str = "standard",
    action_profile: WebActionProfile | str | None = None,
) -> NeMoGymEasyInputMessage:
    if action_profile is not None and WebActionProfile(action_profile) == WebActionProfile.NATIVE_TOOLCALL:
        content = f"Tool-call parse error: {error}. Return corrected structured browser tool calls only."
    elif action_profile is not None and WebActionProfile(action_profile) == WebActionProfile.WEBVOYAGER_LEGACY:
        content = (
            f"Action parse error: {error}. Return one corrected response and stop after its Action line.\n"
            f"{WEBVOYAGER_ACTION_GUIDANCE}"
        )
    elif action_prompt_profile == "standard":
        content = (
            f"Action parse error: {error}. Return a corrected Thought and Action only, "
            "using the required action grammar."
        )
    elif action_prompt_profile == "code_block":
        content = (
            f"Action parse error: {error}. Return a corrected response using exactly this format; "
            "put an executable BrowserGym call in the Python Code block, not a description:\n"
            f"{BROWSERGYM_CODE_BLOCK_FORMAT}"
        )
    else:
        raise ValueError(f"unsupported action prompt profile: {action_prompt_profile}")
    return NeMoGymEasyInputMessage(
        role="user",
        content=content,
    )
