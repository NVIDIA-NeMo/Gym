# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restore a continuation's recorded prefix.

The workspace half runs at seed: replay the prefix's shell commands, reset the files to the episode's base tree and
apply the recorded diff, which is taken against that tree (or reset them to the Git index the replayed commands leave,
for a prefix whose diffs were taken against the index).
The session half is OpenCode's: the harness imports the staged prefix export and continues it. Verification binds the
continued root session back to the prefix it restored (``continued_session``).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from shlex import quote

from pydantic import JsonValue

from resources_servers.agentic_if.episode import opencode_prefix
from resources_servers.agentic_if.episode.runtime import BoundSandbox, private_staging, upload_text
from resources_servers.agentic_if.episode.workspace import workspace_diff_command
from resources_servers.agentic_if.if_grading.contracts import SessionEvidence
from resources_servers.agentic_if.if_grading.git_diff import workspace_patches_equal
from resources_servers.agentic_if.task_variants.runtime import ContinuationSpec, RuntimeSpec


async def _workspace_diff(sandbox: BoundSandbox, *, base: str | None, exclude: tuple[str, ...] = ()) -> str:
    current = await sandbox.exec(command=workspace_diff_command(base, exclude=exclude), timeout_s=600)
    if current.return_code != 0 or current.error_type:
        raise RuntimeError("cannot record replay workspace diff")
    return current.stdout or ""


async def restore_workspace(
    sandbox: BoundSandbox, spec: RuntimeSpec, continuation: ContinuationSpec, *, base: str, staging: str
) -> dict[str, JsonValue]:
    """Restore recorded repository state after replaying shell environment effects.

    ``base`` is the episode's base tree, which the recorded diff is taken against, unless the continuation's diffs were
    taken against the Git index (``workspace_diff_base``): the files are then reset to, and compared against, the index
    the replayed commands leave. ``staging`` is a directory of the task identity, outside the repository, for the
    patches. A replayed command that times out fails the restore, unless the continuation counts it as a mismatch
    (``replay_timeouts``).
    """
    source = continuation.prefix_export
    workdir = sandbox.workdir or str(source["info"].get("directory") or ".")
    against = None if continuation.workspace_diff_base == "index" else base
    mismatches, timeouts = [], []
    commands = opencode_prefix.recorded_commands(source, default_workdir=workdir)
    remote = None
    touched = set()
    if continuation.workspace_diff.strip():
        remote = f"{staging}/prefix.patch"
        await upload_text(sandbox, remote, continuation.workspace_diff)
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
        if result.error_type == "timeout" and continuation.replay_timeouts == "mismatch":
            # Typically a command that also timed out when it was recorded: what it did before the deadline stays.
            timeouts.append(index)
            mismatches.append(index)
            continue
        if result.error_type:
            raise RuntimeError("prefix shell replay failed at the sandbox transport")
        observed = ((result.stdout or "") + (result.stderr or "")).strip()
        if result.return_code != 0 or observed != recorded.output.strip():
            mismatches.append(index)
    # Back to the recorded diff's base: undo, as a patch, what the replayed commands left in the files. The index they
    # may have staged stays as it is. Against the base tree, the recorded diff describes the files whatever was staged;
    # against the index, it describes them relative to what the replayed commands staged again. This is a fresh,
    # resource-owned task sandbox, never the host checkout.
    replayed = quote(f"{staging}/replayed.patch")
    reset = await sandbox.exec(
        command=f"( {workspace_diff_command(against, exclude=untouched)} ) > {replayed} && "
        f"{{ [ ! -s {replayed} ] || git apply -R --binary --whitespace=nowarn {replayed}; }}",
        timeout_s=600,
    )
    if reset.return_code != 0 or reset.error_type:
        raise RuntimeError("cannot reset the sandbox repository before restoring the prefix")
    if continuation.workspace_diff.strip():
        applied = await sandbox.exec(command=f"git apply --binary --whitespace=nowarn {quote(remote)}", timeout_s=600)
        if applied.return_code != 0 or applied.error_type:
            raise RuntimeError("recorded prefix workspace diff does not apply")
    current = await _workspace_diff(sandbox, base=against, exclude=untouched)
    receipt: dict[str, JsonValue] = {
        "mode": "restore",
        "replayed_commands": len(commands),
        "command_output_mismatch_indices": mismatches,
        "workspace_matches_diff": workspace_patches_equal(current, continuation.workspace_diff),
    }
    if continuation.replay_timeouts is not None:
        receipt["command_timeout_indices"] = timeouts
    return receipt


async def restore_prefix_workspace(
    sandbox: BoundSandbox, spec: RuntimeSpec, *, base: str | None
) -> dict[str, JsonValue]:
    """The seed half of a continuation: check the recorded diff, then restore the workspace as the task identity.

    ``base`` is the episode's base tree (``workspace_base``). The patches are staged in a private directory of the task
    identity, removed once the restore is done.
    """
    continuation = spec.continuation
    if continuation is None:
        raise ValueError("a continuation prefix is required")
    if base is None:
        raise RuntimeError("a continuation needs a Git repository in the task's working directory")
    recorded = opencode_prefix.recorded_workspace_diff(continuation.prefix_export)
    if recorded is not None and not workspace_patches_equal(recorded, continuation.workspace_diff):
        raise ValueError("continuation workspace diff disagrees with the recorded prefix")
    async with private_staging(sandbox, "replay") as staging:
        return await restore_workspace(sandbox, spec, continuation, base=base, staging=staging)


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
    prefix_ids = frozenset(opencode_prefix.prefix_message_ids(continuation.prefix_export))
    messages = [message for message in (export or {}).get("messages") or [] if isinstance(message, dict)]
    restored = {"messages": [m for m in messages if (m.get("info") or {}).get("id") in prefix_ids]}
    result = dict(receipt)
    result["prefix_messages"] = len(restored["messages"])
    expected = [
        state.get("output", state.get("error")) for _, state in opencode_prefix.tool_parts(continuation.prefix_export)
    ]
    observed = [state.get("output", state.get("error")) for _, state in opencode_prefix.tool_parts(restored)]
    result["tool_outputs_match"] = expected == observed
    return ContinuedSession(session_id, prefix_ids, result, dict(artifacts))
