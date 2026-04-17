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
"""NeMoAgent — Stirrup Agent subclass with two overrides for GDPVal.

1. **tool_response_as_user** — Converts ``ToolMessage`` objects to
   ``UserMessage`` in the ``step()`` return so the LLM sees tool results
   as user messages (some models perform better this way).

2. **skip_input_file_listing** — Suppresses the file-path listing that
   Stirrup injects into the system prompt via ``_build_system_prompt()``.
   When we build the GDPVal user prompt externally (with its own
   ``<reference_files>`` section), the duplicate listing wastes tokens
   and can confuse the model.

These two overrides allow using Stirrup as-is, without a fork.
"""

from __future__ import annotations

from typing import Any

from stirrup import Agent


class NeMoAgent(Agent):
    """Agent with optional tool-response-as-user conversion and system prompt control."""

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

    async def step(self, messages: list, run_metadata: Any, turn: int = 0, max_turns: int = 0) -> tuple:
        assistant_msg, tool_msgs, finish_params = await super().step(
            messages, run_metadata, turn=turn, max_turns=max_turns
        )

        if self._tool_response_as_user and tool_msgs:
            from stirrup.core.models import UserMessage

            converted = []
            for tm in tool_msgs:
                content = tm.content if isinstance(tm.content, str) else str(tm.content)
                converted.append(UserMessage(content=content))
            tool_msgs = converted  # type: ignore[assignment]

        return assistant_msg, tool_msgs, finish_params
