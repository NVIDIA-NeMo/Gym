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
"""Resources server for AweAI-Team/Scale-SWE.

Scale-SWE is 20,181 rows, entirely Python. Each row ships its own prebuilt image
(``image_url``), an explicit ``workdir``, and a ``pre_commands`` string that checks out
``parent_commit`` and scrubs git history so the fix commit is unreachable. Unlike SWE-rebench,
there is a single test runner (pytest) for the whole set, so no parser dispatch is needed.

The failing tests arrive as a literal test file (``f2p_script``, 92% of rows), a patch that adds
them (``f2p_patch``, 69%), or both. Verification: run ``pre_commands``, apply the candidate
patch and whichever of f2p_patch/f2p_script the row carries, run pytest against the named test
files, grade FAIL_TO_PASS/PASS_TO_PASS from the output.

Set ``is_verifying_golden_patch: true`` to grade the dataset's own patch instead of an agent's,
the dataset-health check: a row whose golden patch does not resolve is a broken row.

An episode session keeps its first verification and replays it to an identical retry. A verification that
produced no verdict is masked with a ``failure_kind``.

Episode sessions run on the shared session base (``resources_servers/swebench/sandbox_sessions.py``) and take part in
partial-rollout checkpoints like the other repository-task servers
(``resources_servers/swebench/repo_task_sessions.py``): a commit snapshots the agent's sandbox, or after verify
captured the patch, saves the capture alone. Seed, verify and close follow the base; ``ScaleSWEResourcesServer``
lists what this server adds.
"""

import asyncio
import logging
import re
import shlex
import sys
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path
from time import time
from traceback import format_exc
from typing import Any, ClassVar

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import get_global_config_dict
from nemo_gym.core import PROVIDER_UNAVAILABLE, SESSION_LOST, VERIFIER_ERROR
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxResources,
    SandboxSpec,
    cpu_cap_env,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    lost_session_verification,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.scale_swe.verification import (
    VerificationInputs,
    VerificationResult,
    as_id_list,
    clean_commands,
    drop_patch_sections,
    drop_test_patch_files,
    infrastructure_failure,
    run_verification,
    verification_files,
)
from resources_servers.swebench.agent_network import AgentNetworkConfig
from resources_servers.swebench.anti_cheat import apply_anti_cheat_setup
from resources_servers.swebench.patch_capture import (
    CaptureCommandError,
    PatchCapture,
    PatchCaptureMode,
    capture_model_patch,
    prepare_git_for_commits,
)
from resources_servers.swebench.repo_task_sessions import (
    RepoTaskSessionFields,
    RepoTaskSessionResourcesServer,
    loaded_capture,
    saved_capture,
)
from resources_servers.swebench.sandbox_sessions import _TaskSession


LOG = logging.getLogger(__name__)

# What `git rev-parse HEAD` prints for a commit: a SHA-1 or SHA-256 object name.
_COMMIT_ID = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")


class ScaleSWEResourcesServerConfig(RepoTaskSessionFields, BaseResourcesServerConfig, AgentNetworkConfig):
    is_verifying_golden_patch: bool = False
    # "worktree" (diff of the working tree) or "committed" (committed work only); see swebench/patch_capture.py.
    patch_capture_mode: PatchCaptureMode = "worktree"
    include_model_patch_in_response: bool = True
    evaluation_timeout: int | None = 1200
    # A verdict-less run is retried on a fresh sandbox: an image pull or a flaky provider start
    # is not evidence about the patch.
    inconclusive_verification_retries: int = 1
    apply_anti_cheating: bool = True
    sandbox_stop_timeout_seconds: float = Field(
        default=120,
        gt=0,
        allow_inf_nan=False,
        description=(
            "How long an episode session waits for each sandbox stop: its task sandbox (at close, after a "
            "failed seed, after patch capture, at expiry) and each verifier sandbox. The stop itself is never cut "
            "short. One that is still running, or that failed, stays with the session, and close or shutdown waits "
            "for it again or retries it. Shutdown also uses it for every sandbox an episode session still holds, and "
            "as the longest wait for a session's request in progress. The cookie path's stops inside requests are "
            "not bounded this way."
        ),
    )
    sandbox_provider: str
    sandbox_config: dict[str, Any]


class ScaleSWEInstanceRequest(BaseModel):
    """One row of AweAI-Team/Scale-SWE."""

    model_config = ConfigDict(extra="allow")

    instance_id: str
    repo: str = ""
    language: str = "python"
    workdir: str
    image_url: str
    patch: str = ""
    pre_commands: str = ""
    problem_statement: str = ""
    f2p_patch: str = ""
    f2p_script: str = ""
    # Upstream spells these in caps and JSON-encodes the list; keep both as-received so a row
    # round-trips unchanged.
    FAIL_TO_PASS: str | list[str] = []
    PASS_TO_PASS: str | list[str] = []


class ScaleSWESeedSessionRequest(ScaleSWEInstanceRequest, BaseSeedSessionRequest):
    sandbox_spec: dict[str, Any] | None = None


class ScaleSWESeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str


class ScaleSWEVerifyRequest(ScaleSWEInstanceRequest, BaseVerifyRequest):
    pass


class ScaleSWEVerifyResponse(BaseVerifyResponse):
    evaluation_completed: bool
    resolved: bool
    patch_applied: bool
    instance_id: str
    language: str
    test_results: dict[str, Any] | None
    test_output: str
    error: str | None
    eval_sandbox_start_time_taken: float
    patch_verification_time_taken: float
    test_patch_failed: bool = False
    # Patch-capture provenance; see resources_servers/swebench/patch_capture.py.
    patch_source: str = "none"
    patch_branch: str | None = None
    patch_commits: int = 0
    worktree_dirty: bool = False
    model_patch_bytes: int = 0
    model_patch: str | None = None
    model_patch_sha256: str = ""
    model_patch: str | None = None


class _PatchCaptureError(Exception):
    """The agent's patch could not be read for a reason outside the agent's control, so there is no verdict."""

    def __init__(self, failure_kind: str, message: str) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind


class _UnreadablePatchError(Exception):
    """Git failed on the repository the agent left, so its patch cannot be read: a measured 0, not a lost verdict."""


class ScaleSWEResourcesServer(RepoTaskSessionResourcesServer):
    """Scale-SWE episode sessions on the shared session base, with checkpoints; see the module docstring.

    The base supplies the session record, its lock, seed, verify and close, the close fences, the expiry of a session
    that is never closed, and the checkpoint export and restore. This server adds:

    - an episode seed is refused in golden-patch mode (422), for task data that is not a row (422, without the
      values), during shutdown (503), and when the prepared checkout has no base commit (503);
    - each stop gets a bounded wait (``sandbox_stop_timeout_seconds``) but is never cut short, and each grading
      attempt's verifier sandbox stays with the session until its stop succeeds;
    - shutdown waits a bounded time for a session's request in progress, never raises, and leaves what an agent's
      ``/run`` still holds to the provider's TTL.

    The cookie path of an agent's ``/run`` keeps its seed; its task sandbox is created on the shared provider
    (``new_task_sandbox``), so a checkpointing server says its episode restarts.
    """

    config: ScaleSWEResourcesServerConfig
    task_model: ClassVar[type[BaseModel]] = ScaleSWEInstanceRequest

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self.config.check_agent_network_provider(get_global_config_dict())
        # Verifier sandboxes of episode sessions whose stop has not succeeded yet.
        self._session_id_to_verifier_sandboxes: dict[str, list[AsyncSandbox]] = {}
        # Stops that a bounded wait gave up on, still running, so a later wait joins them instead of starting another.
        self._stop_tasks: dict[AsyncSandbox, asyncio.Task[None]] = {}
        self._shutting_down = False

    async def shutdown(self) -> None:
        """Refuse new episode seeds, then stop what sessions hold as the base does (``_stop_abandoned_session``)."""
        self._shutting_down = True
        await super().shutdown()

    async def _stop_abandoned_session(self, session_id: str, stopping: asyncio.Semaphore) -> None:
        """Stop every sandbox an episode session still holds at shutdown.

        A request still in progress on the session gets ``sandbox_stop_timeout_seconds`` to finish; after that the
        sandboxes are stopped without the session's lock. Never raises: a stop that fails or runs long is logged, and
        the provider's TTL reclaims the sandbox. What an agent's ``/run`` still holds is left to the provider's TTL.
        """
        record = self._task_sessions.get(session_id)
        if record is None or record.seed is None:
            return
        async with stopping, AsyncExitStack() as held:
            try:
                async with asyncio.timeout(self.config.sandbox_stop_timeout_seconds):
                    await held.enter_async_context(self._task_session_locks.locked(session_id))
            except TimeoutError:
                LOG.warning("Scale-SWE session %s is still busy at shutdown; stopping its sandboxes", session_id)
            try:
                await self._stop_session_sandboxes(session_id)
            except Exception:
                LOG.exception(
                    "Failed to stop Scale-SWE session %s on shutdown; provider TTL is the backstop", session_id
                )

    def _inputs(self, body: ScaleSWEInstanceRequest, patch: str) -> VerificationInputs:
        return VerificationInputs(
            instance_id=body.instance_id,
            workdir=body.workdir,
            patch=drop_test_patch_files(patch, body.f2p_patch),
            pre_commands=body.pre_commands,
            f2p_patch=body.f2p_patch,
            f2p_script=body.f2p_script,
            fail_to_pass=as_id_list(body.FAIL_TO_PASS),
            pass_to_pass=as_id_list(body.PASS_TO_PASS),
        )

    async def _create_sandbox(
        self,
        body: ScaleSWEInstanceRequest,
        files: dict[str, str] | None = None,
        *,
        for_agent: bool = False,
        session_id: str | None = None,
    ) -> AsyncSandbox:
        """Start a sandbox from the row's image; with ``session_id``, the session's task sandbox.

        A task sandbox is created with ``new_task_sandbox``, before it starts, so a failed setup still stops it and,
        for an episode session, a checkpoint can snapshot it. A grading sandbox is not session state.
        """
        global_config_dict = get_global_config_dict()
        provider_config = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        provider_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)

        # Cap pytest's own parallelism (xdist, if a row's suite uses it) to the CPU limit. A
        # container sees the HOST core count, not the cgroup quota, so an unconstrained worker
        # count fans out against a small quota and CFS-throttles. Lower stakes here than the
        # compiled-language sets, since plain pytest is single-process by default, but cheap
        # insurance for the rows that do use xdist or multiprocessing test runners.
        sandbox_resources = SandboxResources.from_mapping(self.config.sandbox_config.get("resources", {}))
        env = dict(self.config.sandbox_config.get("env", {}))
        if self.config.sandbox_config.get("derive_cpu_env", True):
            env = cpu_cap_env(sandbox_resources.cpu) | env

        spec = SandboxSpec(
            # The row names its own image; there is no repository template to apply.
            image=body.image_url,
            ttl_s=self.config.sandbox_config.get("ttl_s"),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s"),
            workdir=body.workdir,
            env=env,
            files=files or {},
            metadata=provider_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {
                "nemo_gym_agent": self.config.name,
                "instance_id": body.instance_id[:63],
            },
            resources=sandbox_resources,
            provider_options=self.config.sandbox_provider_options(provider_config, for_agent=for_agent),
        )
        if session_id is not None:
            # On the provider every task sandbox shares, and registered for the session and its checkpoints.
            sandbox = self.new_task_sandbox(session_id, spec)
        else:
            sandbox = AsyncSandbox(provider_config)
        await sandbox.start(spec)
        return sandbox

    async def _stop_sandbox(self, sandbox: AsyncSandbox | None) -> None:
        if sandbox is None:
            return
        try:
            await sandbox.stop()
        except Exception:
            print("Failed to stop Scale-SWE sandbox", format_exc(), file=sys.stderr)

    async def _wait_for_stop(self, sandbox: AsyncSandbox) -> None:
        """Wait up to ``sandbox_stop_timeout_seconds`` for ``sandbox`` to stop, without ever cutting the stop short.

        A stop still running after the wait goes on, and the next wait for the same sandbox joins it instead of
        starting another; a stop that failed is started again. Raises the stop's error, or ``TimeoutError``.
        """
        task = self._stop_tasks.get(sandbox)
        if task is None or (task.done() and (task.cancelled() or task.exception() is not None)):
            # No limit of the sandbox's own: this wait bounds the caller, and the stop goes on after it.
            task = asyncio.create_task(sandbox.stop(timeout_s=None))
            # Retrieve the error of a stop nobody waits for anymore, so it is not reported as never retrieved.
            task.add_done_callback(lambda done: done.cancelled() or done.exception())
            self._stop_tasks[sandbox] = task
        try:
            await asyncio.wait_for(asyncio.shield(task), self.config.sandbox_stop_timeout_seconds)
        finally:
            if task.done() and self._stop_tasks.get(sandbox) is task:
                del self._stop_tasks[sandbox]

    async def _stop_session_sandboxes(self, session_id: str) -> None:
        """Stop every sandbox the session still holds, at once: its task sandbox and any verifier sandbox left over.

        Each stop gets the bounded wait of ``_wait_for_stop``, and a sandbox is forgotten only once its stop succeeds.
        If any stop failed or is still running, this raises and the session keeps that sandbox, so close, expiry or
        shutdown can try again.
        """
        outcomes = await asyncio.gather(
            self._stop_task_sandbox(session_id), self._stop_verifier_sandbox(session_id), return_exceptions=True
        )
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome

    async def _stop_unclaimed_task_sandbox(self, session_id: str) -> None:
        """Stop the task sandbox with the bounded wait of ``_wait_for_stop``, not with a stop cut short.

        Under the session's export lock, as the checkpointed base stops it, so a commit never snapshots the sandbox
        while this waits for its stop. The session's task state is forgotten once the sandbox stopped.
        """
        async with self._export_locks.locked(session_id):
            sandbox = self._session_id_to_sandbox.get(session_id)
            if sandbox is not None:
                await self._wait_for_stop(sandbox)
                if self._session_id_to_sandbox.get(session_id) is sandbox:
                    del self._session_id_to_sandbox[session_id]
            if session_id not in self._session_id_to_sandbox:
                self._forget_task_sandbox_state(session_id)
                await self._untrack(session_id)

    async def _stop_verifier_sandbox(self, session_id: str) -> None:
        """Stop the verifier sandboxes of the session whose stop failed or ran long during its verification."""
        held = list(self._session_id_to_verifier_sandboxes.get(session_id, []))
        outcomes = await asyncio.gather(*(self._wait_for_stop(sandbox) for sandbox in held), return_exceptions=True)
        for sandbox, outcome in zip(held, outcomes):
            if outcome is None:
                self._forget_verifier_sandbox(session_id, sandbox)
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome

    def _forget_verifier_sandbox(self, session_id: str, sandbox: AsyncSandbox) -> None:
        """Drop a verifier sandbox whose stop succeeded from what the session holds."""
        verifiers = self._session_id_to_verifier_sandboxes.get(session_id, [])
        if any(held is sandbox for held in verifiers):
            verifiers[:] = [held for held in verifiers if held is not sandbox]
            if not verifiers:
                del self._session_id_to_verifier_sandboxes[session_id]

    async def _stop_attempt_verifier_sandbox(self, session_id: str, sandbox: AsyncSandbox | None) -> None:
        """Stop the verifier sandbox of one grading attempt; one whose stop fails or runs long stays with the session.

        The sandbox is recorded before the wait, so even a cancelled verification leaves it for close and shutdown.
        """
        if sandbox is None:
            return
        self._session_id_to_verifier_sandboxes.setdefault(session_id, []).append(sandbox)
        try:
            await self._wait_for_stop(sandbox)
        except Exception:
            LOG.warning("Verifier sandbox of Scale-SWE session %s is not stopped yet; close retries", session_id)
        else:
            self._forget_verifier_sandbox(session_id, sandbox)

    def _forget_task_sandbox_state(self, session_id: str) -> None:
        self._session_id_to_pristine_untracked.pop(session_id, None)
        self._session_id_to_base_commit.pop(session_id, None)

    async def _ensure_git_repo(self, sandbox: AsyncSandbox, workdir: str) -> None:
        """Some images ship with no git history at all. Init one fresh, but only when missing, so
        ``seed_session`` always has a real base commit to diff the agent's changes against without
        disturbing the git history everywhere else.
        """
        precheck = await sandbox.exec(f"git -C {shlex.quote(workdir)} rev-parse --git-dir")
        if precheck.return_code == 0:
            return
        result = await sandbox.exec(
            f"cd {shlex.quote(workdir)} && git init -q "
            f"&& git config user.email nemo-gym@nvidia.com && git config user.name nemo-gym "
            f"&& git add -A && git commit -q -m 'nemo_gym: initial snapshot' --allow-empty"
        )
        if result.return_code != 0:
            print(f"Failed to init git repo at {workdir}: {result.stdout}\n{result.stderr}", file=sys.stderr)

    async def _extract_model_patch(self, session_id: str, workdir: str, base_commit: str) -> PatchCapture:
        """Capture the agent's patch per ``config.patch_capture_mode``, then stop its sandbox."""
        original_sandbox = self._session_id_to_sandbox.pop(session_id)
        pristine_untracked = self._session_id_to_pristine_untracked.pop(session_id, frozenset())
        try:
            return await capture_model_patch(
                original_sandbox,
                workdir,
                base_commit,
                mode=self.config.patch_capture_mode,
                pristine_untracked=pristine_untracked,
                drop_sections=drop_patch_sections,
            )
        finally:
            await self._stop_sandbox(original_sandbox)

    async def _capture_model_patch(self, session_id: str, workdir: str) -> PatchCapture:
        """Capture an episode session's patch per ``config.patch_capture_mode``, then stop its sandbox.

        Raises ``_UnreadablePatchError`` when git fails on the repository the agent left, and ``_PatchCaptureError``
        when the capture failed for any other reason. The patch is graded in a fresh sandbox, so the agent's sandbox
        is stopped either way (``stop_task_sandbox_after_capture``); a stop that fails or runs long fails the verify and
        leaves the sandbox with the session, and a retry grades the saved capture.

        The outcome, failure included, is saved with the session as a JSON value (``capture_task_sandbox_patch``), so
        a checkpoint can store it and a verify after a restore grades the same patch without capturing it again.
        """
        mode = self.config.patch_capture_mode

        async def capture() -> dict[str, JsonValue]:
            failure_kind = None
            try:
                patch, error, incomplete = await self._read_model_patch(session_id, workdir), None, False
            except _PatchCaptureError as failure:
                patch, error, incomplete = PatchCapture.static("", mode, "none"), str(failure), True
                failure_kind = failure.failure_kind
            except _UnreadablePatchError as failure:
                patch, error, incomplete = PatchCapture.static("", mode, "none"), str(failure), False
            # With the failure kind, so a verify after a restore masks a lost capture as the first verify did.
            return saved_capture(patch, error, incomplete) | {"failure_kind": failure_kind}

        saved = await self.capture_task_sandbox_patch(session_id, capture)
        await self.stop_task_sandbox_after_capture(session_id)
        patch, error, incomplete = loaded_capture(saved)
        if error is None:
            return patch
        if incomplete:
            raise _PatchCaptureError(saved.get("failure_kind") or VERIFIER_ERROR, error)
        raise _UnreadablePatchError(error)

    async def _read_model_patch(self, session_id: str, workdir: str) -> PatchCapture:
        """Read the agent's patch from the session's task sandbox; raises as ``_capture_model_patch`` says."""
        sandbox = self._session_id_to_sandbox.get(session_id)
        base_commit = self._session_id_to_base_commit.get(session_id, "")
        if sandbox is None:
            raise _PatchCaptureError(
                SESSION_LOST,
                "Failed to extract model patch: no task sandbox for this session "
                "(seed_session did not run, or an earlier verification already used it)",
            )
        if not base_commit:
            raise _PatchCaptureError(
                VERIFIER_ERROR, "Failed to extract model patch: seed_session recorded no base commit"
            )
        try:
            return await capture_model_patch(
                sandbox,
                workdir,
                base_commit,
                mode=self.config.patch_capture_mode,
                pristine_untracked=self._session_id_to_pristine_untracked.get(session_id, frozenset()),
                drop_sections=drop_patch_sections,
            )
        except CaptureCommandError as error:
            if error.error_type is None:
                # Git itself exited non-zero on the repository the agent left, such as a stale .git/index.lock
                # from an interrupted git command, a deleted .git, or a pruned base commit.
                raise _UnreadablePatchError(
                    f"Failed to extract model patch: git exited {error.return_code}: {error}"
                ) from error
            # The provider ended the command without an exit status. A timeout counts as infrastructure too:
            # it can come from an overloaded sandbox as well as from the repository, and only a git exit is
            # evidence about the repository.
            raise _PatchCaptureError(
                VERIFIER_ERROR, f"Failed to extract model patch ({error.error_type}): {error}"
            ) from error
        except Exception as error:
            # The command never ran or never answered: the agent's sandbox could not be reached.
            raise _PatchCaptureError(VERIFIER_ERROR, f"Failed to extract model patch: {error}") from error

    async def seed_session(
        self, request: Request, body: ScaleSWESeedSessionRequest | ResourcesSeedSessionRequest
    ) -> ScaleSWESeedSessionResponse | ResourcesSeedSessionResponse:
        """Start the instance's image, then run ``pre_commands`` so the agent sees the prepared
        checkout (parent_commit, fix-commit history scrubbed) rather than the image's raw,
        unprepared state -- ``pre_commands`` used to only run inside the eval script at /verify
        time, which left an agent's own working session never actually checked out to the right
        commit at all.

        An Environment Server seeds with a ``ResourcesSeedSessionRequest`` whose ``task_data`` is the row, gets the
        sandbox as ``sandbox_access``, verifies through the session cookie and closes with ``/close_session``.
        """
        if isinstance(body, ResourcesSeedSessionRequest):
            return await self._seed_resources_session(request, body)
        session_id = request.session[SESSION_ID_KEY]
        record = self._task_sessions.get(session_id)
        if (record is not None and record.seed is not None) or session_id in self._task_session_fences:
            raise ResourcesSessionConflictError("An agent-run seed cannot replace an episode session")
        return await self._seed_session(session_id, body)

    async def _seed_resources_session(
        self, request: Request, body: ResourcesSeedSessionRequest
    ) -> ResourcesSeedSessionResponse:
        if self.config.is_verifying_golden_patch:
            # Golden-patch mode grades the row's own patch, so an agent's sandbox would go unused.
            raise HTTPException(422, "Golden-patch mode cannot be used with episode sessions")
        try:
            ScaleSWESeedSessionRequest.model_validate(body.task_data)
        except ValidationError as error:
            raise HTTPException(422, detail=error.errors(include_input=False)) from error
        if self._shutting_down:
            raise HTTPException(503, "Scale-SWE server is shutting down")
        return await self.seed_task_sandbox_session(request, body, ScaleSWEInstanceRequest)

    async def _start_task_sandbox(self, session_id: str, task: ScaleSWEInstanceRequest) -> str:
        """Start and prepare an episode session's task sandbox, under the session's lock; see ``_seed_session``.

        A shutdown that began meanwhile fails the seed, which then stops the sandbox.
        """
        self._raise_if_shutting_down()
        await self._seed_session(session_id, task)
        if not _COMMIT_ID.fullmatch(self._session_id_to_base_commit.get(session_id, "")):
            # Without a base commit the agent's patch could never be captured: fail the seed instead, so the episode
            # is retried.
            raise HTTPException(503, "git rev-parse HEAD did not name a commit in the task sandbox; retry the episode")
        self._raise_if_shutting_down()
        return task.workdir

    async def _task_sandbox_response(self, session_id: str, record: _TaskSession) -> ResourcesSeedSessionResponse:
        response = await super()._task_sandbox_response(session_id, record)
        # Serializing the sandbox awaits the provider, and the server may have begun shutting down meanwhile.
        self._raise_if_shutting_down()
        return response

    def _raise_if_shutting_down(self) -> None:
        """Refuse to lend a sandbox once the server's shutdown began during setup."""
        if self._shutting_down:
            raise HTTPException(503, "Scale-SWE server is shutting down")

    async def _seed_session(self, session_id: str, body: ScaleSWEInstanceRequest) -> ScaleSWESeedSessionResponse:
        """Start the session's task sandbox and prepare its checkout, for an episode seed and an agent's ``/run``."""
        await self._stop_sandbox(self._session_id_to_sandbox.pop(session_id, None))
        self._forget_task_sandbox_state(session_id)

        # Registered for the session before its setup runs, so a failed setup can still stop it.
        sandbox = await self._create_sandbox(body, for_agent=True, session_id=session_id)

        pre = clean_commands(body.pre_commands)
        if pre:
            result = await sandbox.exec(f"cd {shlex.quote(body.workdir)} && {pre}")
            if result.return_code != 0:
                print(
                    f"[scale_swe] {body.instance_id}: pre_commands failed during seed_session "
                    f"(non-fatal, matching build_eval_script's own tolerance): {result.stderr}",
                    file=sys.stderr,
                )

        await self._ensure_git_repo(sandbox, body.workdir)
        if self.config.apply_anti_cheating:
            required = self.config.anti_cheat_required
            await apply_anti_cheat_setup(sandbox, body.workdir, body.instance_id, "scale_swe", required=required)
        # The anti-cheat scrub leaves no committer identity, so the agent's `git commit` would fail.
        await prepare_git_for_commits(sandbox, body.workdir, "scale_swe")

        head_result = await sandbox.exec(f"git -C {shlex.quote(body.workdir)} rev-parse HEAD")
        self._session_id_to_base_commit[session_id] = (head_result.stdout or "").strip()
        self._session_id_to_pristine_untracked[session_id] = await self._pristine_untracked_files(
            sandbox, body.workdir
        )
        return ScaleSWESeedSessionResponse(sandbox_handle=str(sandbox._handle.sandbox_id))

    async def verify(self, request: Request, body: ScaleSWEVerifyRequest) -> ScaleSWEVerifyResponse:
        """Grade the agent's patch in a fresh sandbox; an episode session's retry gets its saved verdict back."""
        session_id = request.session[SESSION_ID_KEY]

        async def grade() -> ScaleSWEVerifyResponse:
            record = self._task_sessions.get(session_id)
            episode = record is not None and record.seed is not None
            return await self._verify(session_id, body, resources_session=episode)

        return await self.verify_task_sandbox_session(
            request,
            body,
            verify=grade,
            lost_response=lambda reason, kind: self._lost_response(body, reason, kind),
            allow_seedless=self.config.is_verifying_golden_patch,
        )

    def _lost_response(self, body: ScaleSWEVerifyRequest, reason: str, failure_kind: str) -> ScaleSWEVerifyResponse:
        """The masked reply for a session this server no longer holds, such as one it closed when it expired."""
        return ScaleSWEVerifyResponse.model_validate(
            body.model_dump()
            | {
                "evaluation_completed": False,
                "resolved": False,
                "patch_applied": False,
                "instance_id": body.instance_id,
                "language": body.language,
                "test_results": None,
                "test_output": "",
                "error": reason,
                "eval_sandbox_start_time_taken": 0.0,
                "patch_verification_time_taken": 0.0,
            }
            | lost_session_verification(reason, failure_kind=failure_kind)
        )

    async def _verify(
        self, session_id: str, body: ScaleSWEVerifyRequest, *, resources_session: bool
    ) -> ScaleSWEVerifyResponse:
        """Capture the agent's patch (or take the golden one) and grade it.

        For an episode session (``resources_session``):

        - A patch lost to infrastructure is not graded, since an empty patch would read as a completed 0.
        - A repository git cannot read is the agent's doing and scores a completed 0.
        - A verification without a verdict, including a verifier run that never tested the patch, is masked with a
          registered failure_kind.
        - Verifier sandbox stops are waited for within ``sandbox_stop_timeout_seconds``.

        The cookie path (an Agent's ``/run``) masks nothing and replays nothing.
        """
        extraction_error = None
        unreadable_patch = None
        failure_kind = VERIFIER_ERROR
        mode = self.config.patch_capture_mode
        if self.config.is_verifying_golden_patch:
            capture = PatchCapture.static(body.patch, mode, "golden")
        elif resources_session:
            try:
                capture = await self._capture_model_patch(session_id, body.workdir)
            except _PatchCaptureError as error:
                capture = PatchCapture.static("", mode, "none")
                extraction_error, failure_kind = str(error), error.failure_kind
            except _UnreadablePatchError as error:
                capture = PatchCapture.static("", mode, "none")
                unreadable_patch = str(error)
        else:
            base_commit = self._session_id_to_base_commit.pop(session_id, "")
            if not base_commit:
                capture = PatchCapture.static("", mode, "none")
                extraction_error = "Failed to extract model patch: no base commit recorded (seed_session did not run for this session)"
            else:
                try:
                    capture = await self._extract_model_patch(session_id, body.workdir, base_commit)
                except Exception as exc:
                    capture = PatchCapture.static("", mode, "none")
                    extraction_error = f"Failed to extract model patch: {exc}"
        patch = capture.patch

        inputs = self._inputs(body, patch)
        log_dir = Path(__file__).parent / "logs" / body.instance_id

        files = verification_files(inputs)
        attempts = 1 + max(self.config.inconclusive_verification_retries, 0)
        if resources_session and (extraction_error or unreadable_patch):
            # Nothing was captured, so there is nothing to grade.
            attempts = 0
        start_time_taken = 0.0
        verification_time_taken = 0.0
        result = VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=False,
            test_results=None,
            test_output="",
            error="unattempted",
        )
        if unreadable_patch:
            result = VerificationResult(
                completed=True,
                resolved=False,
                patch_applied=False,
                test_results=None,
                test_output="",
                error=unreadable_patch,
            )
        for attempt in range(1, attempts + 1):
            sandbox: AsyncSandbox | None = None
            started = time()
            failure_kind = VERIFIER_ERROR
            try:
                sandbox = await self._create_sandbox(body, files=files)
                start_time_taken = time() - started
                verification_started = time()
                result = await run_verification(
                    sandbox=sandbox,
                    inputs=inputs,
                    timeout_s=self.config.evaluation_timeout,
                    log_dir=log_dir,
                )
                verification_time_taken = time() - verification_started
                if resources_session and (reason := infrastructure_failure(result)) is not None:
                    # The run never tested the patch, so it is no verdict and is retried on a fresh sandbox.
                    result = replace(result, completed=False, resolved=False, error=reason)
            except Exception as exc:
                # A verifier sandbox that never started is the provider's failure.
                failure_kind = PROVIDER_UNAVAILABLE if sandbox is None else VERIFIER_ERROR
                start_time_taken = time() - started
                verification_time_taken = 0.0
                result = VerificationResult(
                    completed=False,
                    resolved=False,
                    patch_applied=False,
                    test_results=None,
                    test_output="",
                    error=f"Verification failed: {exc}",
                )
            finally:
                if resources_session:
                    await self._stop_attempt_verifier_sandbox(session_id, sandbox)
                else:
                    await self._stop_sandbox(sandbox)
            if result.completed:
                break
            if attempt < attempts:
                print(
                    f"[scale_swe] {body.instance_id}: inconclusive ({result.error}); "
                    f"retrying on a fresh sandbox ({attempt}/{attempts - 1})",
                    flush=True,
                )

        error = extraction_error or result.error
        diagnosis = {}
        if resources_session and not result.completed:
            diagnosis = {"mask_sample": True, "failure_kind": failure_kind, "failure_reason": error or "No verdict"}
        elif unreadable_patch:
            # A measured 0, so it stays unmasked; the reason says why the patch could not be read.
            diagnosis = {"failure_reason": unreadable_patch}
        return ScaleSWEVerifyResponse.model_validate(
            body.model_dump()
            | {
                # An unresolved-but-completed run is a real 0. An incomplete run is also 0, but
                # evaluation_completed distinguishes them so a broken row is not read as a hard task.
                "reward": 1.0 if result.resolved else 0.0,
                "evaluation_completed": result.completed,
                "resolved": result.resolved,
                "patch_applied": result.patch_applied,
                "instance_id": body.instance_id,
                "language": body.language,
                "test_results": result.test_results,
                "test_output": result.test_output[-100_000:],
                "error": error,
                "eval_sandbox_start_time_taken": start_time_taken,
                "patch_verification_time_taken": verification_time_taken,
                "test_patch_failed": result.test_patch_failed,
                **capture.response_fields(self.config.include_model_patch_in_response),
            }
            | diagnosis
        )


if __name__ == "__main__":
    ScaleSWEResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # Required whenever num_workers > 1. Multi-worker uvicorn re-imports this entrypoint BY PATH
    # in each forked child and expects a module-level `app`; without it every child exits, and
    # uvicorn responds by stopping the parent. The server then never binds, so the symptom is not
    # an import error but a flood of connection failures from clients talking to a dead port.
    app = ScaleSWEResourcesServer.run_webserver()  # noqa: F401
