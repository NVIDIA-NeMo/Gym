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
    requires_final_answer: bool

    def tools(self, row: dict[str, JsonValue]) -> dict[str, dict[str, JsonValue]]: ...
    def new_state(self, row: dict[str, JsonValue]) -> Any: ...
    # Raises ToolCallError for the model's mistakes; any other exception is an infrastructure error.
    def call(self, state: Any, name: str, arguments: dict[str, JsonValue]) -> JsonValue: ...
    def artifacts(self, state: Any) -> dict[str, str]: ...
    def receipts(self, state: Any) -> dict[str, JsonValue]: ...
