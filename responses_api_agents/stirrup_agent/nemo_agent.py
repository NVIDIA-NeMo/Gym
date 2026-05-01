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
"""NeMoAgent — Stirrup ``Agent`` subclass with two behavioural overrides.

1. **tool_response_as_user** — Return a ``UserMessage`` (not ``ToolMessage``)
   from ``run_tool()`` so the conversation history presents tool results
   with ``role=user``.  Reasoning-trained models tend to keep expanding the
   work and emit auxiliary artifacts (charts, methodology notes) when they
   see tool output as a user turn rather than terminating early.

2. **skip_input_file_listing** — Suppresses the file-path listing Stirrup
   injects into the system prompt; useful when a task prompt already lists
   its own reference files (e.g. GDPVal).

To preserve tool-call metadata that ``Agent.step()`` reads immediately after
``run_tool()`` returns (``.name``, ``.success``, ``.tool_call_id``), the
conversion uses :class:`NeMoUserMessage` — a ``UserMessage`` subclass with
those fields.  Serialisation still renders ``role=user`` so the LLM sees a
user turn.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import ConfigDict
from stirrup import Agent
from stirrup.core.agent import SessionAgent
from stirrup.core.models import AssistantMessage, ChatMessage, ToolCall, ToolMessage, UserMessage


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


def _restore_tool_messages_for_model(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Convert NeMoUserMessage tool results back to ToolMessage for model calls.

    NeMoUserMessage intentionally presents tool results to the agent as user
    turns during normal execution. OpenAI-compatible chat completions APIs,
    however, require assistant messages with tool_calls to be followed by
    matching tool-role messages. Summarization sends prior history back to the
    model, so it needs the provider-valid representation.
    """
    pending_tool_call_ids: set[str] = set()
    restored: list[ChatMessage] = []

    for message in messages:
        if isinstance(message, AssistantMessage):
            pending_tool_call_ids = {tc.tool_call_id for tc in message.tool_calls if tc.tool_call_id}
            restored.append(message)
            continue

        if isinstance(message, NeMoUserMessage) and message.tool_call_id in pending_tool_call_ids:
            restored.append(
                ToolMessage(
                    content=message.content,
                    name=message.name,
                    success=message.success,
                    args_was_valid=message.args_was_valid,
                    tool_call_id=message.tool_call_id,
                    tool_start_time=message.tool_start_time,
                    tool_end_time=message.tool_end_time,
                )
            )
            pending_tool_call_ids.discard(message.tool_call_id)
            continue

        restored.append(message)

    return restored


class NeMoAgent(Agent):
    """``Agent`` with tool-response-as-user conversion and system-prompt control."""

    def __init__(
        self,
        *,
        tool_response_as_user: bool = False,
        skip_input_file_listing: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._tool_response_as_user = tool_response_as_user
        self._skip_input_file_listing = skip_input_file_listing

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

    async def summarize_messages(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Summarize with provider-valid tool-call history."""
        return await super().summarize_messages(_restore_tool_messages_for_model(messages))

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
