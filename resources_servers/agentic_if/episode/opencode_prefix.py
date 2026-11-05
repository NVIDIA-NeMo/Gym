# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read a continuation's recorded OpenCode prefix.

A continuation row carries the OpenCode export of the session so far and the workspace diff at its cut.
``opencode_replay`` re-runs the prefix's shell commands and restores the files at seed; the harness imports the export
(``session_import_path``). These helpers are pure; ``opencode_replay`` does the sandbox work.
"""

import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

from resources_servers.agentic_if.episode.opencode_profile import PROFILE_METADATA_KEY


class ReplayError(ValueError):
    """A continuation prefix that cannot be replayed as recorded."""


@dataclass(frozen=True)
class RecordedCommand:
    """One completed shell call of the prefix, with the directory and deadline it ran under."""

    command: str
    workdir: str
    output: str
    timeout_s: float


def tool_parts(export: Mapping[str, Any]) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """(tool id, state) of every tool part of the export, in order."""
    for message in export.get("messages") or []:
        for part in message.get("parts") or []:
            if part.get("type") == "tool" and isinstance(part.get("state"), Mapping):
                yield part.get("tool", ""), part["state"]


def recorded_commands(export: Mapping[str, Any], *, default_workdir: str) -> list[RecordedCommand]:
    """The prefix's completed shell commands, in order, with the directory they ran in and their recorded output."""
    out = []
    for tool, state in tool_parts(export):
        if tool != "bash" or state.get("status") != "completed":
            continue
        args = state.get("input") or {}
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            continue
        # OpenCode 1.17.11's bash tool uses milliseconds and a 120s default.
        # Dropping this bound lets replay inherit the multi-hour episode
        # limit, even for a prefix command that originally timed out.
        timeout_ms = args.get("timeout", 120_000)
        if type(timeout_ms) not in (int, float):
            raise ReplayError("the replayed bash command has an invalid timeout")
        try:
            timeout_s = timeout_ms / 1000
        except OverflowError as error:
            raise ReplayError("the replayed bash command has an invalid timeout") from error
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ReplayError("the replayed bash command has an invalid timeout")
        out.append(
            RecordedCommand(
                command=command,
                workdir=args.get("workdir") or default_workdir,
                output=str(state.get("output") or ""),
                timeout_s=timeout_s,
            )
        )
    return out


def prefix_message_ids(export: Mapping[str, Any]) -> set[str]:
    """The message IDs of an export."""
    return {m["info"]["id"] for m in export.get("messages") or [] if m.get("info", {}).get("id")}


FILE_TOOLS = {"edit", "write", "bash"}


def recorded_workspace_diff(export: Mapping[str, Any]) -> str | None:
    """The workspace diff at the end of the export, from the plugin's per-call records; None if no call recorded one.

    Refuses an export where a file-changing call after the last record has no record of its own (a failed call: the
    plugin records only after a tool succeeds), because the files could then differ from the recorded diff."""
    last: str | None = None
    unrecorded_after_last = False
    parts = [part for message in export.get("messages") or [] for part in message.get("parts") or []]
    for part in parts:
        if part.get("type") != "tool":
            continue
        tool, state = part.get("tool"), part.get("state") or {}
        if tool not in FILE_TOOLS:
            continue
        stamp = (state.get("metadata") or {}).get(PROFILE_METADATA_KEY)
        if isinstance(stamp, Mapping) and isinstance(stamp.get("workspace_diff"), str):
            last = stamp["workspace_diff"]
            unrecorded_after_last = False
        else:
            unrecorded_after_last = True
    if last is not None and unrecorded_after_last:
        raise ReplayError("a file-changing call after the last recorded workspace diff has no record of its own")
    return last
