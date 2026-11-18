# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An example `ToolBackend` for mock_tools_app.py: packaged notes to look up and a log to append to."""

import json
from pathlib import Path

from pydantic import JsonValue

from resources_servers.partner_if.tool_backend import ToolCallError


TOOLS = {
    "lookup": {
        "type": "function",
        "name": "lookup",
        "description": "Look up a note by its key.",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
        "strict": False,
    },
    "append": {
        "type": "function",
        "name": "append",
        "description": "Append a line to the log.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        "strict": False,
    },
}


class NotesBackend:
    """Reads packaged notes and appends to a per-session log, which is the episode's artifact."""

    requires_final_answer = True

    def __init__(self, fixture_path: Path):
        self.notes = json.loads(fixture_path.read_text())

    def tools(self, row: dict[str, JsonValue]) -> dict[str, dict[str, JsonValue]]:
        return TOOLS

    def new_state(self, row: dict[str, JsonValue]) -> list[str]:
        return []

    def call(self, state: list[str], name: str, arguments: dict[str, JsonValue]) -> JsonValue:
        if name == "lookup":
            key = arguments.get("key")
            if not isinstance(key, str) or key not in self.notes:
                raise ToolCallError(f"No note has the key {key!r}")
            return {"note": self.notes[key]}
        state.append(str(arguments.get("text", "")))
        return "Appended"

    def artifacts(self, state: list[str]) -> dict[str, str]:
        return {"log": "\n".join(state)}

    def receipts(self, state: list[str]) -> dict[str, JsonValue]:
        return {"appended": list(state)}
