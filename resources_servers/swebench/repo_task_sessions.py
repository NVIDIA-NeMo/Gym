# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoints for the servers whose task is a repository and whose verdict grades the agent's patch.

swe_rebench, swemer_v1, swemer_v2 and swe_next share one session shape: the row names an image, the seed starts it
and prepares the repository, the agent edits the checkout, and ``/verify`` captures the agent's patch once and grades
it in a fresh sandbox. Each server keeps its own seed and verify; ``RepoTaskSessionResourcesServer`` holds what they
would otherwise repeat to take part in partial-rollout checkpoints, on ``CheckpointedSandboxSessionResourcesServer``:

- the files the checkout held untracked before the agent started, and the commit the patch is taken against when the
  seed records one, are exported with the session, so a restore that forks the agent's sandbox captures the same patch;
- the captured patch is saved in a form a checkpoint can store (``saved_capture`` and ``loaded_capture``), so a
  ``/verify`` that runs again after a restore grades the patch the checkpoint saved (``checkpoint_verify`` ``replay``);
- golden-patch grading seeds no episode sessions and runs ``restart_only``.

The seed's reads of a new sandbox are retried: ``_pristine_untracked_files`` here, and each server's own read of HEAD.

A server starts its agent's sandbox with ``new_task_sandbox`` and sets ``task_model``; its config adds
``RepoTaskSessionFields``. Grading sandboxes are not session state: each grading run starts its own.

The agent sandbox's egress block is ``AgentNetworkConfig`` (``agent_network.py``). Its rule is recorded with the
sandbox, so the fork a restore makes keeps the rule of the job that seeded it, which allows that job's model server.
"""

import dataclasses
import shlex
from typing import Any, ClassVar

from pydantic import JsonValue

from nemo_gym.agents import exec_idempotent
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.server_runtime import CheckpointMode, StepType
from resources_servers.swebench.patch_capture import PatchCapture
from resources_servers.swebench.sandbox_session_checkpoints import (
    CheckpointedSandboxSessionResourcesServer,
    SandboxCheckpointFields,
)


class RepoTaskSessionFields(SandboxCheckpointFields):
    """The config fields of a repository-task server, besides its own."""

    # Fail a seed whose anti-cheat scrub did not complete instead of handing the agent a history that may hold the
    # task's solution; the rollout is then retried. Off keeps the scrub best effort.
    anti_cheat_required: bool = False


def saved_capture(capture: PatchCapture, error: str | None, incomplete: bool) -> dict[str, JsonValue]:
    """What ``/verify`` passes to ``capture_task_sandbox_patch``: the capture as a JSON value a checkpoint can store.

    ``error`` is the extraction error, if any; ``incomplete`` says the sandbox could not run the capture, which is
    not evidence of an empty patch.
    """
    return {"capture": dataclasses.asdict(capture), "error": error, "incomplete": incomplete}


def loaded_capture(saved: dict[str, Any]) -> tuple[PatchCapture, str | None, bool]:
    """The capture, extraction error and incomplete flag that ``saved_capture`` stored."""
    return PatchCapture(**saved["capture"]), saved["error"], saved["incomplete"]


class RepoTaskSessionResourcesServer(CheckpointedSandboxSessionResourcesServer):
    """A repository-task server whose episode sessions take part in checkpoints; see the module docstring."""

    # The patch is captured once per session, so a verify repeated after a restore grades the same patch.
    checkpoint_verify: ClassVar[StepType] = "replay"

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_id_to_pristine_untracked: dict[str, frozenset[str]] = {}
        # Only for a server whose seed records the commit; another takes it from the row.
        self._session_id_to_base_commit: dict[str, str] = {}

    def checkpoint_mode_in_effect(self) -> CheckpointMode:
        # Golden-patch grading seeds no episode sessions, so it has nothing to checkpoint.
        if self.config.is_verifying_golden_patch:
            return "restart_only"
        return super().checkpoint_mode_in_effect()

    async def _pristine_untracked_files(self, sandbox: AsyncSandbox, workdir: str) -> frozenset[str]:
        """The files ``workdir`` holds untracked before the agent touches it, which its patch leaves out.

        A new sandbox can refuse its first commands for a few seconds, so a listing the provider raised on is
        retried. A listing that still cannot run, that the provider reports as not run, or that git refuses fails the
        seed: an empty set would make the patch capture count the image's own untracked files as the agent's.
        """
        result = await exec_idempotent(
            sandbox,
            f"git -C {shlex.quote(workdir)} ls-files --others --exclude-standard",
            what=f"{self.config.name}: listing pristine untracked files",
        )
        if result.error_type is not None or result.return_code != 0:
            reason = result.error_type or f"exit {result.return_code}"
            raise RuntimeError(
                f"Task sandbox could not list its pristine untracked files: {reason}: "
                f"{(result.stderr or '').strip()[-500:]}"
            )
        return frozenset(line.strip() for line in (result.stdout or "").splitlines() if line.strip())

    def _export_task_sandbox_state(self, session_id: str) -> dict[str, JsonValue]:
        # What a restored session's patch capture needs; nothing once verify captured the patch and stopped the sandbox.
        state: dict[str, JsonValue] = {}
        if session_id in self._session_id_to_pristine_untracked:
            state["pristine_untracked"] = sorted(self._session_id_to_pristine_untracked[session_id])
        if session_id in self._session_id_to_base_commit:
            state["base_commit"] = self._session_id_to_base_commit[session_id]
        return state

    def _restore_task_sandbox_state(self, session_id: str, state: dict[str, JsonValue]) -> None:
        if "pristine_untracked" in state:
            self._session_id_to_pristine_untracked[session_id] = frozenset(
                str(path) for path in state["pristine_untracked"] or ()
            )
        if "base_commit" in state:
            self._session_id_to_base_commit[session_id] = str(state["base_commit"])
