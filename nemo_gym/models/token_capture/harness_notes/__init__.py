# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recognize a message that a harness adds to a request in the model's name.

Some harnesses add an assistant message to a request that the model never wrote, and drop it from the next request,
such as OpenCode's note at its step limit. Capture links calls by their model-authored turns, so such a message would
break every link after it and poison the rollout.

Under capture, the model server forwards such a note as a user message and leaves it out of the items that link calls.
The request then ends with a user turn, and an exact-token worker keeps the parent's tokens up to the parent's own
output and renders the tool results and the note after them. The next request, which no longer carries the note,
still extends those tokens, because the worker keeps them from the parent.

This keeps calls chained where the parent's exact tokens carry over: external staging, or prefix supply on the model
server. With ``parent_resolution: source_history``, the replayed history leaves the note out while the forwarded
messages end with it, so each call past the step limit starts a new chain, as it did before. In-process capture
without prefix supply still fails closed: the trajectory build finds ``parent_digest_mismatch`` and does not link
the call to its parent.

Each harness keeps its notes in a module of its own (``opencode.py``), as :class:`HarnessNote` records with the
releases checked to send that exact text. To support another harness, add its module and its ``NOTES`` to
``HARNESS_NOTES``; the shared tests then cover its notes, and the harness's own tests check its default release.
"""

from __future__ import annotations

from typing import Any

from nemo_gym.models.token_capture.harness_notes import opencode
from nemo_gym.models.token_capture.harness_notes.base import HarnessNote


HARNESS_NOTES: tuple[HarnessNote, ...] = (*opencode.NOTES,)
# Exact texts only: an assistant message that merely resembles a note stays model output.
_NOTE_TEXTS = frozenset(note.text for note in HARNESS_NOTES)


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
    return text if text in _NOTE_TEXTS else None
