# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NeMoAgent — Stirrup ``Agent`` subclass with Gym-specific behaviour.

1. **tool_response_as_user** — Return a ``UserMessage`` (not ``ToolMessage``)
   from ``run_tool()`` so the conversation history presents tool results
   with ``role=user``.  Reasoning-trained models tend to keep expanding the
   work and emit auxiliary artifacts (charts, methodology notes) when they
   see tool output as a user turn rather than terminating early.

2. **skip_input_file_listing** — Suppresses the file-path listing Stirrup
   injects into the system prompt; useful when a task prompt already lists
   its own reference files (e.g. GDPVal).

3. **safe context compaction** — Removes inline reasoning from generated
   summaries and retries empty or degenerate summaries before replacing the
   live history.

To preserve tool-call metadata that ``Agent.step()`` reads immediately after
``run_tool()`` returns (``.name``, ``.success``, ``.tool_call_id``), the
conversion uses :class:`NeMoUserMessage` — a ``UserMessage`` subclass with
those fields.  Serialisation still renders ``role=user`` so the LLM sees a
user turn.
"""

from __future__ import annotations

import logging
import re
from itertools import takewhile
from typing import Any, Optional

from pydantic import ConfigDict
from stirrup import Agent
from stirrup.core.agent import SessionAgent
from stirrup.core.exceptions import ContextOverflowError
from stirrup.core.models import AssistantMessage, ChatMessage, SummaryMessage, ToolCall, ToolMessage, UserMessage
from stirrup.prompts import MESSAGE_SUMMARIZER, MESSAGE_SUMMARIZER_BRIDGE_TEMPLATE


LOGGER = logging.getLogger(__name__)

_MAX_COMPACTION_SUMMARY_ATTEMPTS = 3
_LEADING_THINK_OPEN_RE = re.compile(r"\A\s*<think\s*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)


def _sanitize_compaction_summary(content: Any) -> str:
    """Remove leading inline-thinking blocks from a generated summary.

    Only a leading block is special: a later literal ``<think>`` may be part
    of task content and must survive. An unclosed leading block has no safe
    answer portion, so it is treated as empty and retried.
    """
    if not isinstance(content, str):
        return ""

    sanitized = content.strip()
    while match := _LEADING_THINK_OPEN_RE.match(sanitized):
        closing = _THINK_CLOSE_RE.search(sanitized, match.end())
        if closing is None:
            return ""
        sanitized = sanitized[closing.end() :].lstrip()
    return sanitized.strip()


def _is_meaningful_compaction_summary(summary: str, *, min_words: int = 1) -> bool:
    """Return whether a summary is large enough to replace a full context."""
    return len(summary.split()) >= min_words


def _summary_text_only_instruction(min_words: int) -> str:
    minimum = (
        f"The summary must contain at least {min_words} words."
        if min_words > 1
        else "The summary must contain useful, non-empty text."
    )
    return (
        "Return a concise but comprehensive summary as text only. "
        "Do not call tools. Do not include analysis, reasoning, or <think> tags. "
        f"{minimum}"
    )


def _summary_rejection_reason(content: Any, sanitized: str, min_words: int) -> str:
    if not isinstance(content, str):
        return f"non-text content ({type(content).__name__})"
    if not sanitized:
        return "empty after removing leading reasoning"
    return f"{len(sanitized.split())} words, below configured minimum {min_words}"


class NeMoUserMessage(UserMessage):
    """``UserMessage`` that also carries tool-call metadata.

    When ``tool_response_as_user`` is enabled, ``run_tool()`` returns one of
    these instead of a ``ToolMessage``.  The extra fields mirror what
    ``Agent.step()`` reads on the returned object (``.success``, ``.name``)
    so the agent loop keeps working after the conversion.  Serialisation
    still yields ``role=user`` so the LLM sees a user turn.
    """

    # Allow ``model_dump()`` to include extra fields without the Pydantic
    # V2 warning the base model would otherwise emit.
    model_config = ConfigDict(extra="allow")

    name: Optional[str] = None
    success: bool = False
    args_was_valid: bool = True
    tool_call_id: Optional[str] = None
    tool_start_time: Optional[float] = None
    tool_end_time: Optional[float] = None


# With `from __future__ import annotations`, Pydantic stores field types as
# strings and resolves them lazily.  Force resolution now so construction
# inside async code paths doesn't hit `PydanticUserError: not fully defined`.
NeMoUserMessage.model_rebuild()


class NeMoAgent(Agent):
    """``Agent`` with tool-response-as-user conversion and system-prompt control."""

    def __init__(
        self,
        *,
        tool_response_as_user: bool = False,
        skip_input_file_listing: bool = False,
        min_compaction_summary_words: int = 1,
        **kwargs: Any,
    ) -> None:
        if min_compaction_summary_words < 1:
            raise ValueError("min_compaction_summary_words must be at least 1")
        super().__init__(**kwargs)
        self._tool_response_as_user = tool_response_as_user
        self._skip_input_file_listing = skip_input_file_listing
        self._min_compaction_summary_words = min_compaction_summary_words

    def _build_system_prompt(self) -> str:
        """Override to optionally skip the input file listing."""
        if not self._skip_input_file_listing:
            return super()._build_system_prompt()

        # Temporarily clear uploaded_file_paths so the parent doesn't list them
        from stirrup.core.agent import _SESSION_STATE

        state = _SESSION_STATE.get(None)
        saved_paths = None
        if state and state.uploaded_file_paths:
            saved_paths = state.uploaded_file_paths
            state.uploaded_file_paths = []

        result = super()._build_system_prompt()

        if saved_paths is not None and state is not None:
            state.uploaded_file_paths = saved_paths

        return result

    async def summarize_messages(
        self,
        messages: list[ChatMessage],
        run_metadata_by_turn: dict[str, dict[str, list[Any]]],
    ) -> tuple[list[ChatMessage], list[ChatMessage]]:
        """Compact history without replaying reasoning or erasing progress.

        Stirrup 0.1 inserts ``summary.content`` into the next context verbatim.
        Some reasoning-model endpoints return a leading ``<think>`` block in
        that field, including think-only responses. Sanitize at the adapter
        boundary and retry bad summaries before replacing any history.
        """
        current_messages = messages
        while True:
            try:
                task_context: list[ChatMessage] = list(
                    takewhile(
                        lambda message: not isinstance(message, (AssistantMessage, SummaryMessage)),
                        current_messages,
                    )
                )

                tool_docs = "\n".join(f"- {tool.name}: {tool.description}" for tool in self._active_tools.values())
                strict_prompt = (
                    f"{MESSAGE_SUMMARIZER}\n\n{_summary_text_only_instruction(self._min_compaction_summary_words)}"
                )
                no_tools_prompt = f"{strict_prompt}\n\nTools are disabled for this response." + (
                    f" Tools used earlier were:\n{tool_docs}" if tool_docs else ""
                )
                attempts = (
                    (MESSAGE_SUMMARIZER, self._active_tools),
                    (strict_prompt, self._active_tools),
                    (no_tools_prompt, {}),
                )

                summary_content = ""
                for attempt, (prompt, tools) in enumerate(attempts, start=1):
                    # Compaction uses the policy client internally, but its
                    # one-shot truncation recovery belongs to policy turns.
                    # Do not let a summary consume an armed recovery or arm a
                    # recovery that leaks into the next policy turn.
                    sentinel = object()
                    saved_recovery = getattr(self._client, "_recover_from_truncation", sentinel)
                    saved_overruns = getattr(self._client, "_truncation_overruns", sentinel)
                    if saved_recovery is not sentinel:
                        self._client._recover_from_truncation = False
                    try:
                        summary = await self._client.generate(
                            [*current_messages, UserMessage(content=prompt)],
                            tools,
                        )
                    finally:
                        if saved_recovery is not sentinel:
                            self._client._recover_from_truncation = saved_recovery
                        if saved_overruns is not sentinel:
                            self._client._truncation_overruns = saved_overruns
                    summary_content = _sanitize_compaction_summary(summary.content)
                    if _is_meaningful_compaction_summary(
                        summary_content,
                        min_words=self._min_compaction_summary_words,
                    ):
                        break
                    usage = getattr(summary, "token_usage", None)
                    LOGGER.warning(
                        "Compaction summary attempt %d/%d rejected: %s (input=%s answer=%s reasoning=%s); %s",
                        attempt,
                        _MAX_COMPACTION_SUMMARY_ATTEMPTS,
                        _summary_rejection_reason(
                            summary.content,
                            summary_content,
                            self._min_compaction_summary_words,
                        ),
                        getattr(usage, "input", None),
                        getattr(usage, "answer", None),
                        getattr(usage, "reasoning", None),
                        "retrying" if attempt < _MAX_COMPACTION_SUMMARY_ATTEMPTS else "no attempts remain",
                    )
                else:
                    raise RuntimeError(
                        "Compaction summarizer produced no usable text after "
                        f"{_MAX_COMPACTION_SUMMARY_ATTEMPTS} attempts; refusing to erase context"
                    )

                summary_bridge_prompt = MESSAGE_SUMMARIZER_BRIDGE_TEMPLATE.format(summary=summary_content)
                summary_bridge = SummaryMessage(content=summary_bridge_prompt)
                acknowledgement_msg = UserMessage(content="Got it, thanks!")

                self._logger.context_summarization_complete(summary_content, summary_bridge_prompt)
                return current_messages, [*task_context, summary_bridge, acknowledgement_msg]
            except ContextOverflowError:
                # Preserve Stirrup's recovery contract: discard the latest
                # completed turn, including its per-turn metadata, and retry.
                current_messages, dropped_turn_id = self._unwind_context_overflow(current_messages)
                run_metadata_by_turn.pop(dropped_turn_id, None)

    async def run_tool(self, tool_call: ToolCall, run_metadata: dict[str, list[Any]]) -> ToolMessage:
        """Run a tool and optionally return a ``NeMoUserMessage`` instead of ``ToolMessage``.

        Preserves all tool metadata on the returned message but flips its
        serialised role from ``tool`` to ``user``.  ``Agent.step()`` inspects
        ``.success`` and ``.name`` on the returned object immediately, so
        ``NeMoUserMessage`` carries those fields.
        """
        tool_message: ToolMessage = await super().run_tool(tool_call, run_metadata)

        if not self._tool_response_as_user:
            return tool_message

        return NeMoUserMessage(  # type: ignore[return-value]
            content=tool_message.content,
            name=tool_message.name,
            success=tool_message.success,
            args_was_valid=getattr(tool_message, "args_was_valid", True),
            tool_call_id=tool_message.tool_call_id,
            tool_start_time=getattr(tool_message, "tool_start_time", None),
            tool_end_time=getattr(tool_message, "tool_end_time", None),
        )

    async def __aenter__(self):  # type: ignore[override]
        """Upgrade the SessionAgent returned by Stirrup to a NeMoSessionAgent.

        Stirrup's ``Agent.__aenter__`` returns ``SessionAgent.from_agent(self)``,
        a plain ``SessionAgent`` that inherits from ``Agent`` directly and
        therefore bypasses any methods we override on ``NeMoAgent`` (MRO stops
        at ``Agent``).  We cannot cleanly re-implement ``__aenter__`` (it runs
        ~100 lines of tool/state setup), so we let Stirrup do its work, then
        reassign the returned instance's ``__class__`` to ``NeMoSessionAgent``
        — a layout-compatible subclass that inherits from both
        ``SessionAgent`` (for tool/session state) and ``NeMoAgent`` (for our
        overrides).  After the reassignment, ``self.run_tool`` and any
        other NeMoAgent method dispatch through our overrides.
        """
        sa = await super().__aenter__()
        sa.__class__ = NeMoSessionAgent
        return sa


class NeMoSessionAgent(SessionAgent, NeMoAgent):
    """``SessionAgent`` variant whose MRO also includes ``NeMoAgent``.

    Python's C3 linearisation gives us

        NeMoSessionAgent -> SessionAgent -> NeMoAgent -> Agent -> ...

    so method lookups for ``run_tool`` / ``_build_system_prompt`` (which
    SessionAgent inherits from Agent without overriding) resolve to our
    NeMoAgent overrides.  Used via ``agent.__aenter__`` → reassign
    ``__class__``; no ``__init__`` is called (the instance's ``__dict__``
    was already populated by ``Agent.__init__`` on the parent NeMoAgent).
    """

    pass
