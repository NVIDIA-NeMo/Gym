# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The text the judge and the tool simulator are shown: a conversation with the items under evaluation, the
agent's own system prompt, and the tool definitions. Each reader has its render settings (`JUDGE_RENDER`,
`SIMULATOR_RENDER`)."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel

from resources_servers.agentic_turn_simulation.items import Items, content_text


ELISION = "\n[... {n:,} characters elided ...]\n"


class RenderConfig(BaseModel):
    """How much of a conversation a reader is shown, and how."""

    tool_output_max_chars: int = 2_000
    tool_args_max_chars: int = 2_000
    # "head_tail" keeps the opening (the task) and the most recent context, in whole blocks.
    prefix_mode: Literal["full", "head_tail"] = "full"
    head_chars: int = 20_000
    tail_chars: int = 60_000
    # Wrap the span of the history the user went on to correct (the row's corrected_turn_item_range) in
    # <rejected_turn>.
    mark_corrected_turn: bool = True
    # Hard cap on the rendered history, applied last. 0 = uncapped.
    max_conversation_chars: int = 0
    # Who the user-role messages are from (e.g. "Parent agent" when another agent delegated the task).
    user_label: str = "User"


# The judge sees the whole history (capped), with the corrected turn marked.
JUDGE_RENDER = RenderConfig(max_conversation_chars=200_000)
# The simulator runs on every tool call: it sees the opening and the recent end of a long history.
SIMULATOR_RENDER = RenderConfig(
    tool_output_max_chars=4_000,
    tool_args_max_chars=6_000,
    prefix_mode="head_tail",
    head_chars=20_000,
    tail_chars=100_000,
    mark_corrected_turn=False,
)
# Likewise the head and tail of a long system prompt, where harnesses put the environment (cwd, OS, date),
# memory and project instructions. The judge sees it whole.
SIMULATOR_SYSTEM_PROMPT_CHARS = 60_000


def truncate(text: str, limit: int) -> str:
    """Keep the head and tail around an explicit elision marker."""
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head] + ELISION.format(n=len(text) - limit) + text[-tail:]


def render_item(item: dict[str, Any], cfg: RenderConfig) -> str | None:
    """One Responses item to a text block, or None to skip it (reasoning, empty messages)."""
    kind = item.get("type")
    if kind == "message" or (kind is None and "role" in item):
        text = content_text(item.get("content"))
        if not text.strip():
            return None
        role = item.get("role")
        if role == "user":
            return f'<message from="{cfg.user_label}">\n{text}\n</message>'
        if role == "assistant":
            return f'<message from="Agent">\n{text}\n</message>'
        return None
    if kind == "function_call":
        args = truncate(item.get("arguments") or "", cfg.tool_args_max_chars)
        return f'<tool_call id="{item.get("call_id")}" name="{item.get("name")}">\n{args}\n</tool_call>'
    if kind == "function_call_output":
        output = item.get("output") or ""
        if not isinstance(output, str):
            output = json.dumps(output)
        return (
            f'<tool_result id="{item.get("call_id")}">\n{truncate(output, cfg.tool_output_max_chars)}\n</tool_result>'
        )
    return None


def _fitting(blocks: list[str], budget: int) -> int:
    """How many of the leading blocks fit in `budget` characters."""
    used = 0
    for n, block in enumerate(blocks):
        used += len(block)
        if used > budget:
            return n
    return len(blocks)


def _head_tail(blocks: list[str], cfg: RenderConfig) -> list[str]:
    """The opening and closing blocks that fit cfg.head_chars and cfg.tail_chars, with a marker between."""
    if len("\n\n".join(blocks)) <= cfg.head_chars + cfg.tail_chars:
        return blocks
    n_head = _fitting(blocks, cfg.head_chars)
    n_tail = _fitting(blocks[n_head:][::-1], cfg.tail_chars)
    omitted = len(blocks) - n_head - n_tail
    middle = [f"[... {omitted} blocks omitted from the middle of the conversation ...]"] if omitted > 0 else []
    return blocks[:n_head] + middle + blocks[len(blocks) - n_tail :]  # not [-n_tail:]: n_tail may be 0


def render_conversation(
    input_items: Items,
    step_items: Items,
    cfg: RenderConfig,
    corrected_turn_range: list[int] | None = None,
) -> str:
    """The history, then the items under evaluation (the turn, or the tool call to simulate)."""
    blocks: list[str] = []
    mark = bool(cfg.mark_corrected_turn and corrected_turn_range)
    lo, hi = corrected_turn_range or (-1, -1)
    open_span = False

    for idx, item in enumerate(input_items):
        block = render_item(item, cfg)
        if block is not None:
            if mark and not open_span and lo <= idx < hi:
                block = "<rejected_turn>  <!-- the turn the user is correcting -->\n" + block
                open_span = True
            blocks.append(block)
        # Close on the span's last item, even when that item rendered to nothing.
        if mark and open_span and idx == hi - 1:
            blocks[-1] = blocks[-1] + "\n</rejected_turn>"
            open_span = False
    if open_span and blocks:  # the span runs past the end of the history
        blocks[-1] = blocks[-1] + "\n</rejected_turn>"

    if cfg.prefix_mode == "head_tail" and blocks:
        blocks = _head_tail(blocks, cfg)
    prefix = "\n\n".join(blocks)
    if cfg.max_conversation_chars:
        prefix = truncate(prefix, cfg.max_conversation_chars)

    step = "\n\n".join(b for b in (render_item(i, cfg) for i in step_items) if b)
    step = step or "[the agent produced nothing]"
    return f"<conversation>\n{prefix}\n</conversation>\n\n<step_under_evaluation>\n{step}\n</step_under_evaluation>"


def agent_system_prompt(items: Items) -> str:
    """The agent's own system prompt: the text of the history's system and developer messages, in order."""
    texts = (
        content_text(x.get("content"))
        for x in items
        if x.get("type", "message") == "message" and x.get("role") in ("system", "developer")
    )
    return "\n\n".join(t for t in texts if t.strip())


def render_system_prompt(text: str, max_chars: int = 0) -> str:
    """The agent's system prompt as a block to put before the conversation ("" when there is none);
    `max_chars` > 0 keeps its head and tail around an elision marker."""
    if not text.strip():
        return ""
    return f"<agent_system_prompt>\n{truncate(text, max_chars)}\n</agent_system_prompt>\n\n"


def format_tools(tools: Items | None) -> str:
    """The tool definitions as a <tools> block."""
    if not tools:
        return "<tools>\n(tool definitions omitted)\n</tools>"
    body = "\n".join(
        f'<tool name="{t.get("name")}">\n{t.get("description", "")}\n'
        f"parameters: {json.dumps(t.get('parameters', {}))}\n</tool>"
        for t in tools
    )
    return f"<tools>\n{body}\n</tools>"
