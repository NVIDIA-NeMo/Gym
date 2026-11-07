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
"""Episode sessions, with checkpoints, for repository tasks graded by the agent's patch.

swe_rebench, swemer_v1, swemer_v2 and swe_next share one session shape: the row names an image, the seed starts it
and prepares the repository (git history, anti-cheat scrub, committer identity), the agent edits the checkout, and
``/verify`` captures the agent's patch with :func:`capture_model_patch` and grades it in a fresh sandbox.
:class:`RepoTaskSessionResourcesServer` runs that shape on :class:`CheckpointedSandboxSessionResourcesServer`, so
these servers serve episode sessions and take part in partial-rollout checkpoints:

- the commit the agent's patch is taken against and the files the checkout held untracked before the agent started
  are exported with the session, so a restore that forks the agent's sandbox captures against the same base;
- the patch is captured once per session and the agent's sandbox stopped after it, so a ``/verify`` that a checkpoint
  replays, or that runs again after a restore, grades the same patch (``checkpoint_verify`` ``replay``);
- grading sandboxes are not session state: each grading run starts its own.

An agent's own ``/run`` still seeds with the flat row under its session cookie.

A subclass describes its task: ``task_model`` (the row, which a seed's ``task_data`` must validate as), the sandbox
spec, the agent's working directory, and the repository setup that runs before the agent gets the sandbox.
"""

import dataclasses
import logging
import shlex
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, ClassVar

from fastapi import Request
from pydantic import JsonValue, ValidationError

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, resolve_provider_config
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    CheckpointMode,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    StepType,
    lost_session_verification,
)
from resources_servers.swebench.patch_capture import PatchCapture, PatchCaptureMode, capture_model_patch
from resources_servers.swebench.sandbox_session_checkpoints import (
    CheckpointedSandboxSessionResourcesServer,
    SandboxCheckpointFields,
)


LOG = logging.getLogger(__name__)


class RepoTaskSessionConfig(SandboxCheckpointFields, BaseResourcesServerConfig):
    is_verifying_golden_patch: bool = False
    # "worktree" (diff of the working tree) or "committed" (committed work only); see swebench/patch_capture.py.
    patch_capture_mode: PatchCaptureMode = "worktree"
    sandbox_provider: str
    sandbox_config: dict[str, Any]


class RepoTaskSessionResourcesServer(CheckpointedSandboxSessionResourcesServer):
    """A resources server whose sessions each hold one repository task sandbox; see the module docstring."""

    config: RepoTaskSessionConfig
    # The patch is captured once per session, so a verify repeated after a restore grades the same patch.
    checkpoint_verify: ClassVar[StepType] = "replay"

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_id_to_base_commit: dict[str, str] = {}
        self._session_id_to_pristine_untracked: dict[str, frozenset[str]] = {}

    def checkpoint_mode_in_effect(self) -> CheckpointMode:
        # Golden-patch grading takes the row's own patch and seeds no episode sessions: it has nothing to checkpoint.
        if self.config.is_verifying_golden_patch:
            return "restart_only"
        return super().checkpoint_mode_in_effect()

    # -- what a subclass describes -----------------------------------------------------------------------------

    def _sandbox_spec(self, task: Any, files: dict[str, str] | None = None) -> SandboxSpec:
        """The sandbox the task runs in, for the agent's session and for each grading run."""
        raise NotImplementedError

    def _task_workdir(self, task: Any) -> str:
        """The repository checkout the agent works in and the patch is captured from."""
        raise NotImplementedError

    async def _prepare_repository(self, sandbox: AsyncSandbox, task: Any) -> None:
        """Prepare the checkout before the agent gets the sandbox: git history, anti-cheat scrub, committer identity."""
        raise NotImplementedError

    def _drop_patch_sections(self, patch: str, paths: Iterable[str]) -> str:
        """Drop the sections of ``patch`` for ``paths``, files the checkout held untracked before the agent started."""
        raise NotImplementedError

    def _session_files(self, task: Any) -> dict[str, str]:
        """Files delivered to the agent's sandbox at start, in addition to the spec's."""
        return {}

    async def _after_start(self, sandbox: AsyncSandbox, task: Any) -> None:
        """Runs once any sandbox of the task started, before the repository is prepared or a grading runs."""

    def _base_commit(self, task: Any, seed_head: str) -> str:
        """The commit the agent's patch is taken against; by default HEAD once the repository is prepared."""
        return seed_head

    # -- seeds -------------------------------------------------------------------------------------------------

    async def seed_episode_session(
        self, request: Request, body: ResourcesSeedSessionRequest
    ) -> ResourcesSeedSessionResponse:
        """An Environment Server's seed: lends the session's sandbox; a restored session's seed lends the fork."""
        try:
            task = self.task_model.model_validate(body.task_data)
        except ValidationError as error:
            raise ResourcesSeedRejectedError(f"task_data is not a {self.task_model.__name__}: {error}") from error
        if body.task_id.task_id != task.instance_id:
            raise ResourcesSeedRejectedError("TaskId does not match the instance_id")
        return await self.seed_task_sandbox_session(request, body, self.task_model)

    async def seed_agent_run_session(self, request: Request, task: Any) -> str:
        """An agent's own ``/run`` seeds with the flat row under its session cookie; returns the sandbox id."""
        session_id = request.session[SESSION_ID_KEY]
        async with self._locked_task_session(session_id) as record:
            if record.seed is not None or self._task_session_fences.get(session_id) is not None:
                raise ResourcesSessionConflictError("An agent-run seed cannot replace an episode session")
            await self._stop_task_sandbox(session_id)
            try:
                await self._start_task_sandbox(session_id, task)
            except BaseException:
                await self._stop_failed_seed_sandbox(session_id)
                raise
            return str(self._session_id_to_sandbox[session_id].sandbox_id)

    async def _start_task_sandbox(self, session_id: str, task: Any) -> str:
        """Start the session's sandbox, prepare the repository, and keep what the patch capture needs."""
        workdir = self._task_workdir(task)

        async def setup(sandbox: AsyncSandbox) -> None:
            await self._after_start(sandbox, task)
            await self._prepare_repository(sandbox, task)
            head = await sandbox.exec(f"git -C {shlex.quote(workdir)} rev-parse HEAD")
            seed_head = (head.stdout or "").strip() if head.return_code == 0 else ""
            self._session_id_to_base_commit[session_id] = self._base_commit(task, seed_head)
            self._session_id_to_pristine_untracked[session_id] = await self._pristine_untracked_files(sandbox, workdir)

        spec = self._sandbox_spec(task, files=self._session_files(task) or None)
        await self.new_task_sandbox(session_id, spec).start_with_setup(spec, setup)
        return workdir

    def _forget_task_sandbox_state(self, session_id: str) -> None:
        self._session_id_to_base_commit.pop(session_id, None)
        self._session_id_to_pristine_untracked.pop(session_id, None)

    def _export_task_sandbox_state(self, session_id: str) -> dict[str, JsonValue]:
        # What a restored session's patch capture needs; nothing once verify captured the patch and stopped the sandbox.
        if session_id not in self._session_id_to_base_commit:
            return {}
        return {
            "base_commit": self._session_id_to_base_commit[session_id],
            "pristine_untracked": sorted(self._session_id_to_pristine_untracked.get(session_id, frozenset())),
        }

    def _restore_task_sandbox_state(self, session_id: str, state: dict[str, JsonValue]) -> None:
        if "base_commit" in state:
            self._session_id_to_base_commit[session_id] = str(state["base_commit"])
            self._session_id_to_pristine_untracked[session_id] = frozenset(
                str(path) for path in state.get("pristine_untracked") or ()
            )

    # -- verify ------------------------------------------------------------------------------------------------

    async def verify_repo_task[ResponseT: BaseVerifyResponse](
        self,
        request: Request,
        body: BaseVerifyRequest,
        response_model: type[ResponseT],
        grade: Callable[[PatchCapture, str | None], Awaitable[ResponseT]],
    ) -> ResponseT:
        """Grade the patch ``captured_patch`` returns with ``grade(capture, extraction_error)``.

        An episode session's first verdict is kept and replayed to an identical retry; a session this server no longer
        holds gets a masked ``response_model``.
        """
        session_id = request.session[SESSION_ID_KEY]

        async def verify() -> ResponseT:
            capture, extraction_error = await self.captured_patch(session_id, body)
            return await grade(capture, extraction_error)

        def lost_response(reason: str, failure_kind: str) -> ResponseT:
            return response_model.model_validate(
                body.model_dump()
                | {
                    "evaluation_completed": False,
                    "resolved": False,
                    "patch_applied": False,
                    "test_results": None,
                    "test_output": "",
                    "error": reason,
                    "eval_sandbox_start_time_taken": 0.0,
                    "patch_verification_time_taken": 0.0,
                }
                | lost_session_verification(reason, failure_kind=failure_kind)
            )

        return await self.verify_task_sandbox_session(
            request,
            body,
            verify=verify,
            lost_response=lost_response,
            allow_seedless=self.config.is_verifying_golden_patch,
        )

    async def captured_patch(self, session_id: str, task: Any) -> tuple[PatchCapture, str | None]:
        """The patch to grade and an extraction error, if any.

        The agent's patch is captured once per session and its sandbox stopped after; a later verify, a replayed or
        restored one included, gets the saved capture. A capture that fails grades an empty patch, which is saved too.
        """
        mode = self.config.patch_capture_mode
        if self.config.is_verifying_golden_patch:
            return PatchCapture.static(task.patch, mode, "golden"), None

        async def capture_or_fallback() -> dict[str, JsonValue]:
            # Saved as JSON, so a checkpoint can export it.
            try:
                capture = await self._capture_patch(session_id, self._task_workdir(task))
                return {"capture": dataclasses.asdict(capture), "error": None}
            except Exception as error:
                LOG.warning("%s: failed to extract the patch of session %s: %s", self.config.name, session_id, error)
                fallback = PatchCapture.static("", mode, "none")
                return {"capture": dataclasses.asdict(fallback), "error": f"Failed to extract model patch: {error}"}

        saved = await self.capture_task_sandbox_patch(session_id, capture_or_fallback)
        await self.stop_task_sandbox_after_capture(session_id)
        return PatchCapture(**saved["capture"]), saved["error"]

    async def _capture_patch(self, session_id: str, workdir: str) -> PatchCapture:
        base_commit = self._session_id_to_base_commit.get(session_id, "")
        if not base_commit:
            raise RuntimeError("no base commit recorded (seed_session did not run for this session)")
        return await capture_model_patch(
            self._session_id_to_sandbox[session_id],
            workdir,
            base_commit,
            mode=self.config.patch_capture_mode,
            pristine_untracked=self._session_id_to_pristine_untracked.get(session_id, frozenset()),
            drop_sections=self._drop_patch_sections,
        )

    async def start_grading_sandbox(self, task: Any, files: dict[str, str]) -> AsyncSandbox:
        """A fresh sandbox for one grading run, on a provider of its own; the caller stops it."""
        spec = self._sandbox_spec(task, files=files)
        sandbox = AsyncSandbox(resolve_provider_config(self.config.sandbox_provider, get_global_config_dict()))
        await sandbox.start_with_setup(spec, lambda started: self._after_start(started, task))
        return sandbox

    # -- internals ---------------------------------------------------------------------------------------------

    async def _pristine_untracked_files(self, sandbox: AsyncSandbox, workdir: str) -> frozenset[str]:
        """The files ``workdir`` holds untracked before the agent touches it (excluded from its patch)."""
        try:
            result = await sandbox.exec(f"git -C {shlex.quote(workdir)} ls-files --others --exclude-standard")
        except Exception:
            LOG.exception("%s: failed to list pristine untracked files", self.config.name)
            return frozenset()
        if result.return_code != 0:
            LOG.warning("%s: failed to list pristine untracked files: %s", self.config.name, result.stderr)
            return frozenset()
        return frozenset(line.strip() for line in (result.stdout or "").splitlines() if line.strip())
