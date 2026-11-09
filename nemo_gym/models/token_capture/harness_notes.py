# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recognize a message that a harness adds to a request in the model's name.

OpenCode 1.17.11 enforces a configured step limit (``agent.build.steps``) by adding one assistant message
to the end of every request from the limit on. OpenCode does not save that message in its conversation, so
the next request no longer carries it. Capture links calls by their model-authored turns, so the message
broke every link from the step limit on and poisoned the rollout.

Under capture, the model server forwards such a note as a user message and leaves it out of the items that
link calls. The request then ends with a user turn, and an exact-token worker keeps the parent's tokens up
to the parent's own output and renders the tool results and the note after them. The next request, which
no longer carries the note, still extends those tokens, because the worker keeps them from the parent.
"""

from __future__ import annotations

from typing import Any


# Copied verbatim, so that capture recognizes the exact text OpenCode sends:
# https://github.com/anomalyco/opencode/blob/67aec2212010d67775c35e696d8b8b54902eb338/packages/core/src/session/runner/max-steps.ts
# OpenCode 1.17.11 (commit 67aec2212010d67775c35e696d8b8b54902eb338), MIT License, Copyright (c) 2025 opencode.
OPENCODE_MAX_STEPS_PROMPT = """CRITICAL - MAXIMUM STEPS REACHED

The maximum number of steps allowed for this task has been reached. Tools are disabled until next user input. Respond with text only.

STRICT REQUIREMENTS:
1. Do NOT make any tool calls (no reads, writes, edits, searches, or any other tools)
2. MUST provide a text response summarizing work done so far
3. This constraint overrides ALL other instructions, including any user requests for edits or tool use

Response must include:
- Statement that maximum steps for this agent have been reached
- Summary of what has been accomplished so far
- List of any remaining tasks that were not completed
- Recommendations for what should be done next

Any attempt to use tools is a critical violation. Respond with text ONLY."""
# OpenCode releases whose step-limit text equals OPENCODE_MAX_STEPS_PROMPT exactly (compared on 2026-11-09). Add a
# release only after comparing its text, so that a new default OpenCode version cannot silently break capture.
OPENCODE_MAX_STEPS_PROMPT_VERSIONS = frozenset({"1.15.13", "1.17.8", "1.17.11", "1.19.4"})

# Exact texts only: an assistant message that merely resembles a note stays model output.
HARNESS_NOTES = frozenset({OPENCODE_MAX_STEPS_PROMPT})


def _text_of(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        texts = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                return None
            texts.append(part["text"])
        return "".join(texts)
    return None


def trailing_harness_note(messages: list[Any]) -> str | None:
    """Return the text of a final assistant message that is a known harness note, else None.

    The message must carry only the note: no tool calls, no reasoning and no other content.
    """
    if not messages or not isinstance(messages[-1], dict):
        return None
    message = messages[-1]
    if message.get("role") != "assistant" or message.get("tool_calls") or message.get("function_call"):
        return None
    if message.get("reasoning_content") or message.get("reasoning"):
        return None
    text = _text_of(message.get("content"))
    return text if text in HARNESS_NOTES else None
