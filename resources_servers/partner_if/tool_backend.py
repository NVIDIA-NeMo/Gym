# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a mock-tool backend implements and raises.

Gym runs mock_tools_app.py as `__main__`, so a backend imports these from here: importing them from
mock_tools_app would load a second copy of that module, whose ToolCallError the server never catches.
"""

from typing import Any, Protocol

from pydantic import JsonValue


class ToolCallError(Exception):
    """A mistake the model made in a tool call, such as an unknown key; the model sees it as the output."""


class ToolBackend(Protocol):
    """A task's simulated tools. The server holds one state per episode and calls these methods synchronously
    on its event loop, so each must return quickly and must not block on I/O."""

    # Whether an episode must end with a text answer; one that does not scores 0 without the judge.
    requires_final_answer: bool

    def tools(self, row: dict[str, JsonValue]) -> dict[str, dict[str, JsonValue]]:
        """The tools this row's task may call, by name."""

    def new_state(self, row: dict[str, JsonValue]) -> Any:
        """A fresh state for one episode of this row."""

    def call(self, state: Any, name: str, arguments: dict[str, JsonValue]) -> JsonValue:
        """Run one tool call against the episode's state and return its output. Raise ToolCallError for the
        model's mistakes: the server records the call with `{"error": ...}` as its output, which the model sees.
        Any other exception is an infrastructure error and leaves no record, so the call has no receipt and the
        rollout is masked."""

    def artifacts(self, state: Any) -> dict[str, str]:
        """The files the episode produced, by name; the judge sees them as artifact evidence."""

    def receipts(self, state: Any) -> dict[str, JsonValue]:
        """The backend's own records of the episode, returned with the verify response."""
