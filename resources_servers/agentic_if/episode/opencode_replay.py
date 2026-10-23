# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restore a continuation's recorded prefix; no separate rollout or grading lifecycle.

The workspace half runs at seed: replay the prefix's shell commands, reset the repository and apply the recorded diff.
The session half is OpenCode's: the harness imports the staged prefix export and continues it. Verification binds the
continued root session back to the prefix it restored (``continued_session``).
"""

import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from shlex import quote
from uuid import uuid4

from pydantic import JsonValue

from resources_servers.agentic_if.episode import opencode_prefix as replay
from resources_servers.agentic_if.episode.runtime import BoundSandbox
from resources_servers.agentic_if.episode.workspace import WORKSPACE_DIFF
from resources_servers.agentic_if.if_grading.contracts import SessionEvidence
from resources_servers.agentic_if.task_variants.runtime import ContinuationSpec, RuntimeSpec


def workspace_patches_equal(left: str, right: str) -> bool:
    """Compare recorded Git patches, allowing only compatible object-ID abbreviations.

    Git's default abbreviation length depends on repository configuration. It is
    not a file-content difference. Keep every other line (including modes, binary
    bodies, and no-newline markers) exact; even index IDs must agree by prefix.
    """
    first, second = left.splitlines(), right.splitlines()
    if len(first) != len(second):
        return False
    index = re.compile(r"index ([0-9a-f]{4,64})\.\.([0-9a-f]{4,64})( [0-7]{6})?")
    for a, b in zip(first, second, strict=True):
        if a == b:
            continue
        x, y = index.fullmatch(a), index.fullmatch(b)
        if x is None or y is None or x[3] != y[3]:
            return False
        if any(not (x[k].startswith(y[k]) or y[k].startswith(x[k])) for k in (1, 2)):
            return False
    return True


async def _upload(sandbox: BoundSandbox, directory: str, *, filename: str, text: str) -> str:
    remote = directory + "/" + filename
    with tempfile.TemporaryDirectory(prefix="if-replay-") as temporary:
        path = Path(temporary) / filename
        path.write_text(text)
        await sandbox.upload(path, remote)
    mode = await sandbox.transport.exec(command=f"chmod 0644 {quote(remote)}", user="root", timeout_s=30)
    if mode.return_code != 0 or mode.error_type:
        raise RuntimeError("cannot stage the prefix patch")
    return remote


async def _workspace_diff(sandbox: BoundSandbox, *, exclude: tuple[str, ...] = ()) -> str:
    command = WORKSPACE_DIFF
    if exclude:
        paths = " ".join(quote(":(top,literal,exclude)" + path) for path in exclude)
        command = command.replace(
            "git diff --binary --full-index",
            f"git diff --binary --full-index -- . {paths}",
        )
    current = await sandbox.exec(command=command, timeout_s=600)
    if current.return_code != 0 or current.error_type:
        raise RuntimeError("cannot record replay workspace diff")
    return current.stdout or ""


async def restore_workspace(
    sandbox: BoundSandbox, spec: RuntimeSpec, continuation: ContinuationSpec, *, staging: str
) -> dict[str, JsonValue]:
    """Restore recorded repository state after replaying shell environment effects.

    ``staging`` is a directory the task identity can read, outside the repository, for the recorded patch.
    """
    source = continuation.prefix_export
    workdir = sandbox.workdir or str(source["info"].get("directory") or ".")
    mismatches = []
    commands = replay.recorded_commands(source, default_workdir=workdir)
    remote = None
    touched = set()
    if continuation.workspace_diff.strip():
        remote = await _upload(sandbox, staging, filename="prefix.patch", text=continuation.workspace_diff)
        stats = await sandbox.exec(command=f"git apply --numstat -z {quote(remote)}", timeout_s=30)
        if stats.return_code or stats.error_type:
            raise RuntimeError("cannot inspect recorded prefix workspace diff")
        touched = {entry.split("\t", 2)[2] for entry in (stats.stdout or "").split("\0") if entry}
    initial_paths = []
    for path in spec.workspace_files:
        if path.startswith("/"):
            prefix = workdir.rstrip("/") + "/"
            if not path.startswith(prefix):
                # Repository cleanup cannot touch a declared file outside it.
                continue
            path = path[len(prefix) :]
        initial_paths.append(path)
    untouched = tuple(path for path in initial_paths if path not in touched)
    for index, recorded in enumerate(commands):
        result = await sandbox.exec(
            command=f'"${{SHELL:-/bin/sh}}" -c {quote(recorded.command)}',
            cwd=recorded.workdir,
            timeout_s=recorded.timeout_s,
        )
        if result.error_type:
            raise RuntimeError("prefix shell replay failed at the sandbox transport")
        observed = ((result.stdout or "") + (result.stderr or "")).strip()
        if result.return_code != 0 or observed != recorded.output.strip():
            mismatches.append(index)
    # This is a fresh, resource-owned task sandbox, never the host checkout.
    checkout = " ".join(quote(":(top,literal,exclude)" + path) for path in untouched)
    clean = " ".join("-e " + quote("/" + path) for path in untouched)
    reset = await sandbox.exec(command=f"git checkout -- . {checkout} && git clean -fdq {clean}", timeout_s=600)
    if reset.return_code != 0 or reset.error_type:
        raise RuntimeError("cannot reset the sandbox repository before restoring the prefix")
    if continuation.workspace_diff.strip():
        applied = await sandbox.exec(command=f"git apply --binary --whitespace=nowarn {quote(remote)}", timeout_s=600)
        if applied.return_code != 0 or applied.error_type:
            raise RuntimeError("recorded prefix workspace diff does not apply")
    current = await _workspace_diff(sandbox, exclude=untouched)
    return {
        "mode": "restore",
        "replayed_commands": len(commands),
        "command_output_mismatch_indices": mismatches,
        "workspace_matches_diff": workspace_patches_equal(current, continuation.workspace_diff),
    }


async def restore_prefix_workspace(sandbox: BoundSandbox, spec: RuntimeSpec) -> dict[str, JsonValue]:
    """The seed half of a continuation: check the recorded diff, then restore the workspace as the task identity.

    The patch is staged in a private directory of the task identity, removed once the restore is done.
    """
    continuation = spec.continuation
    if continuation is None:
        raise ValueError("a continuation prefix is required")
    recorded = replay.recorded_workspace_diff(continuation.prefix_export)
    if recorded is not None and not workspace_patches_equal(recorded, continuation.workspace_diff):
        raise ValueError("continuation workspace diff disagrees with the recorded prefix")
    staging = f"/tmp/nemo-gym-agentic-if-replay-{uuid4().hex}"
    created = await sandbox.exec(command=f"umask 077; mkdir -p {quote(staging)}", timeout_s=30)
    if created.return_code or created.error_type:
        raise RuntimeError("could not prepare a private prefix staging directory")
    try:
        return await restore_workspace(sandbox, spec, continuation, staging=staging)
    finally:
        await sandbox.transport.exec(command=f"rm -rf {quote(staging)}", user="root", timeout_s=30)


@dataclass(frozen=True)
class ContinuedSession:
    """Actual prefix session and message IDs, before the fresh user continuation."""

    session_id: str
    prefix_ids: frozenset[str]
    receipt: dict[str, JsonValue]
    artifacts: dict[str, str]

    def attach(self, evidence: SessionEvidence) -> SessionEvidence:
        """Return a copy of ``evidence`` with this replay's receipt, artifacts and continuation start."""
        result = evidence.model_copy(deep=True)
        starts = [
            index
            for index, step in enumerate(result.steps)
            if step.session_id == self.session_id and step.role == "user" and step.id not in self.prefix_ids
        ]
        result.facts["replay"] = self.receipt
        result.artifacts.update(self.artifacts)
        if starts:
            result.continuation_start = starts[0]
        else:
            result.status = "incomplete"
            result.issues.append("missing_continuation_user_message")
        if not self.receipt["workspace_matches_diff"]:
            result.status = "incomplete"
            result.issues.append("replay_workspace_diverged")
        if not self.receipt["tool_outputs_match"]:
            result.status = "incomplete"
            result.issues.append("restored_prefix_history_diverged")
        return result


def continued_session(
    continuation: ContinuationSpec,
    *,
    session_id: str,
    export: Mapping[str, JsonValue] | None,
    receipt: Mapping[str, JsonValue],
    artifacts: Mapping[str, str],
) -> ContinuedSession:
    """Bind the continued root session to the prefix restored at seed, retaining real replay evidence.

    OpenCode's import keeps the prefix's message IDs, so the prefix is the root's messages with those IDs. Workspace
    divergence and real tool-result divergence are distinct facts.
    """
    prefix_ids = frozenset(replay.prefix_message_ids(continuation.prefix_export))
    messages = [message for message in (export or {}).get("messages") or [] if isinstance(message, dict)]
    restored = {"messages": [m for m in messages if (m.get("info") or {}).get("id") in prefix_ids]}
    result = dict(receipt)
    result["prefix_messages"] = len(restored["messages"])
    expected = [state.get("output", state.get("error")) for _, state in replay.tool_parts(continuation.prefix_export)]
    observed = [state.get("output", state.get("error")) for _, state in replay.tool_parts(restored)]
    result["tool_outputs_match"] = expected == observed
    return ContinuedSession(session_id, prefix_ids, result, dict(artifacts))
