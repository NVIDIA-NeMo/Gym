# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The sandbox owner base for repository tasks graded by the agent's patch.

swe_rebench, scale_swe, swemer_v1, swemer_v2 and swe_next share one session shape: the row names an image, the
seed starts it and prepares the repository (git history, anti-cheat scrub, committer identity), the agent edits the
checkout, and ``/verify`` captures the agent's patch with :func:`capture_model_patch` and grades it in a fresh
sandbox. :class:`RepoTaskOwnerResourcesServer` runs that shape on :class:`SandboxOwnerResourcesServer`, so these
servers take part in partial-rollout checkpoints:

- the session's base commit and the files the checkout held untracked before the agent started are kept in the
  session's data, so a restore that forks the agent's sandbox grades against the same base;
- the patch is captured once per session (``capture_once``) and the agent's sandbox stopped after it, so a ``/verify``
  that a checkpoint replays, or that runs again after a restore, grades the same patch (``checkpoint_verify``
  ``replay``);
- grading sandboxes run on the provider the sessions share.

A subclass describes its task: the instance model a typed seed's ``task_data`` must validate as, the sandbox spec,
the agent's working directory, and the repository setup that runs before the agent gets the sandbox.
"""

import logging
import shlex
from collections.abc import Iterable
from dataclasses import asdict
from typing import Any, ClassVar

from fastapi import Request
from pydantic import BaseModel, ValidationError

from nemo_gym._checkpoint.control import CheckpointMode
from nemo_gym._checkpoint.steps import StepType
from nemo_gym.base_resources_server import (
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.rollout_correlation import current_episode_id
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.sandbox_owner import SandboxOwnerConfig, SandboxOwnerResourcesServer, SessionSetup
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.swebench.patch_capture import PatchCapture, PatchCaptureMode, capture_model_patch


LOGGER = logging.getLogger(__name__)

# Session data keys.
BASE_COMMIT = "base_commit"
PRISTINE_UNTRACKED = "pristine_untracked"
PATCH = "patch"


class RepoTaskOwnerConfig(SandboxOwnerConfig):
    is_verifying_golden_patch: bool = False
    # "worktree" (diff of the working tree) or "committed" (committed work only); see swebench/patch_capture.py.
    patch_capture_mode: PatchCaptureMode = "worktree"
    sandbox_config: dict[str, Any]


class RepoTaskOwnerResourcesServer(SandboxOwnerResourcesServer):
    """A resources server whose sessions own one repository task sandbox each; see the module docstring."""

    config: RepoTaskOwnerConfig
    checkpoint_verify: ClassVar[StepType] = "replay"
    # The flat instance a row carries; a typed seed's task_data must validate as it.
    instance_model: ClassVar[type[BaseModel]]

    def checkpoint_mode_in_effect(self) -> CheckpointMode:
        # A golden-patch server grades the dataset's own patch and never seeds: it holds no session state.
        if self.config.is_verifying_golden_patch:
            return "restart_only"
        return super().checkpoint_mode_in_effect()

    # -- what a subclass describes -----------------------------------------------------------------------------

    def _sandbox_spec(self, instance: Any, files: dict[str, str] | None = None) -> SandboxSpec:
        """The sandbox the task runs in, for the agent's session and for each grading run."""
        raise NotImplementedError

    def _task_workdir(self, instance: Any) -> str:
        """The repository checkout the agent works in and the patch is captured from."""
        raise NotImplementedError

    async def _prepare_repository(self, sandbox: AsyncSandbox, instance: Any) -> None:
        """Prepare the checkout before the agent gets the sandbox: git history, anti-cheat scrub, committer identity."""
        raise NotImplementedError

    def _drop_patch_sections(self, patch: str, paths: Iterable[str]) -> str:
        """Drop the sections of ``patch`` for ``paths``, files the checkout held untracked before the agent started."""
        raise NotImplementedError

    def _session_files(self, instance: Any) -> dict[str, str]:
        """Files delivered to the agent's sandbox at start, in addition to the spec's."""
        return {}

    async def _after_start(self, sandbox: AsyncSandbox, instance: Any) -> None:
        """Runs once any sandbox of the task started, before the repository is prepared or a grading runs."""

    def _base_commit(self, instance: Any, seed_head: str) -> str:
        """The commit the agent's patch is taken against; by default HEAD once the repository is prepared."""
        return seed_head

    # -- seeds -------------------------------------------------------------------------------------------------

    async def _set_up_session(self, session_id: str, instance: Any) -> None:
        """Start the session's sandbox, prepare the repository, and keep what grading needs in the session data."""
        workdir = self._task_workdir(instance)

        async def setup(sandbox: AsyncSandbox) -> SessionSetup:
            await self._after_start(sandbox, instance)
            await self._prepare_repository(sandbox, instance)
            head = await sandbox.exec(f"git -C {shlex.quote(workdir)} rev-parse HEAD")
            seed_head = (head.stdout or "").strip() if head.return_code == 0 else ""
            return SessionSetup(
                workdir=workdir,
                data={
                    BASE_COMMIT: self._base_commit(instance, seed_head),
                    PRISTINE_UNTRACKED: sorted(await self._pristine_untracked_files(sandbox, workdir)),
                },
            )

        spec = self._sandbox_spec(instance, files=self._session_files(instance) or None)
        await self.start_session_sandbox(session_id, spec, setup)

    async def _set_up_typed_session(self, session_id: str, body: ResourcesSeedSessionRequest) -> None:
        try:
            instance = self.instance_model.model_validate(body.task_data)
        except ValidationError as error:
            raise ResourcesSeedRejectedError(f"task_data is not a {self.instance_model.__name__}: {error}") from error
        if body.task_id.task_id != instance.instance_id:
            raise ResourcesSeedRejectedError("TaskId does not match the instance_id")
        await self._set_up_session(session_id, instance)

    async def seed_typed_session(
        self, request: Request, body: ResourcesSeedSessionRequest
    ) -> ResourcesSeedSessionResponse:
        """An Environment Server's seed: lends the session's sandbox; a restored session's re-seed lends the fork."""
        return await self.seed_task_sandbox_session(request, body, self._set_up_typed_session)

    async def seed_flat_session(self, request: Request, instance: Any) -> str:
        """An agent's own ``/run`` seeds with the flat instance under its session cookie; returns the sandbox id."""
        session_id = request.session[SESSION_ID_KEY]
        await self.seed_sandbox_session(
            session_id,
            episode_id=current_episode_id() or EpisodeId(rollout_id=session_id),
            task_id=TaskId(taskset=self.config.name or type(self).__name__, task_id=instance.instance_id),
            setup=lambda: self._set_up_session(session_id, instance),
        )
        return str(self.require_session_sandbox(session_id).sandbox_id)

    # -- verify ------------------------------------------------------------------------------------------------

    async def captured_patch(self, session_id: str, instance: Any) -> tuple[PatchCapture, str | None]:
        """The patch to grade and an extraction error, if any.

        The agent's patch is captured once per session and its sandbox stopped after; later calls, a replayed
        verify included, return the stored capture. A capture that fails grades an empty patch and leaves the
        agent's sandbox for a retry (close, retire or shutdown stop it).
        """
        mode = self.config.patch_capture_mode
        if self.config.is_verifying_golden_patch:
            return PatchCapture.static(instance.patch, mode, "golden"), None
        try:
            stored = await self.capture_once(
                session_id, PATCH, lambda: self._capture_patch(session_id), release_sandbox_after=True
            )
        except Exception as exc:
            LOGGER.warning(
                "%s: failed to extract the model patch of session %s: %s", self.config.name, session_id, exc
            )
            return PatchCapture.static("", mode, "none"), f"Failed to extract model patch: {exc}"
        return PatchCapture(**stored), None

    async def _capture_patch(self, session_id: str) -> dict[str, Any]:
        sandbox = self.require_session_sandbox(session_id)
        data = self.session_data(session_id)
        base_commit = str(data.get(BASE_COMMIT) or "")
        if not base_commit:
            raise RuntimeError("no base commit recorded (seed_session did not run for this session)")
        capture = await capture_model_patch(
            sandbox,
            self.session_workdir(session_id) or "",
            base_commit,
            mode=self.config.patch_capture_mode,
            pristine_untracked=frozenset(data.get(PRISTINE_UNTRACKED) or ()),
            drop_sections=self._drop_patch_sections,
        )
        # Raised rather than stored: capture_once keeps nothing from a failed capture, so a retry captures again.
        return asdict(capture)

    async def start_grading_sandbox(self, instance: Any, files: dict[str, str]) -> AsyncSandbox:
        """A fresh sandbox for one grading run, on the provider the sessions share; the caller stops it."""
        spec = self._sandbox_spec(instance, files=files)
        sandbox = AsyncSandbox(self.shared_provider(), spec, owns_provider=False)
        await sandbox.start_with_setup(spec, lambda started: self._after_start(started, instance))
        return sandbox

    # -- internals ---------------------------------------------------------------------------------------------

    async def _pristine_untracked_files(self, sandbox: AsyncSandbox, workdir: str) -> frozenset[str]:
        """The files ``workdir`` holds untracked before the agent touches it."""
        try:
            result = await sandbox.exec(f"git -C {shlex.quote(workdir)} ls-files --others --exclude-standard")
        except Exception:
            LOGGER.exception("%s: failed to list pristine untracked files", self.config.name)
            return frozenset()
        if result.return_code != 0:
            LOGGER.warning("%s: failed to list pristine untracked files: %s", self.config.name, result.stderr)
            return frozenset()
        return frozenset(line.strip() for line in (result.stdout or "").splitlines() if line.strip())
