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

An Environment Server session (a typed seed) keeps its first completed verification and replays it to an identical
retry. A verification that produced no verdict is masked with a ``failure_kind`` and not kept: an identical retry
gets a 503 that asks the caller to retry the episode.
"""

import asyncio
import logging
import re
import shlex
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import time
from traceback import format_exc
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ClosedResourcesSessionRecords,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    SimpleResourcesServer,
)
from nemo_gym.config import get_global_config_dict
from nemo_gym.episode_types import EpisodeId
from nemo_gym.failure_kinds import PROVIDER_UNAVAILABLE, SESSION_LOST, VERIFIER_ERROR
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxResources,
    SandboxSpec,
    cpu_cap_env,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_utils import SESSION_ID_KEY
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
from resources_servers.swebench.anti_cheat import apply_anti_cheat_setup
from resources_servers.swebench.patch_capture import (
    CaptureCommandError,
    PatchCapture,
    PatchCaptureMode,
    capture_model_patch,
    prepare_git_for_commits,
)


LOG = logging.getLogger(__name__)

# What `git rev-parse HEAD` prints for a commit: a SHA-1 or SHA-256 object name.
_COMMIT_ID = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")


class ScaleSWEResourcesServerConfig(BaseResourcesServerConfig):
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
            "How long an Environment Server session waits for each sandbox stop: its task sandbox (at close, after a "
            "failed seed, after patch capture) and each verifier sandbox. The stop itself is never cut short. One "
            "that is still running, or that failed, stays with the session, and close or shutdown waits for it again "
            "or retries it. Shutdown also uses it for every sandbox still held, cookie sessions included, and as "
            "the longest wait for a session's request in progress. The cookie path's stops inside requests are not "
            "bounded."
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


@dataclass
class _ResourcesSession:
    """One Environment Server session: its lock, identity, seed reply and first completed verification.

    The sandbox stays in ``_session_id_to_sandbox``, which keeps it until a stop succeeds.
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    episode_id: EpisodeId | None = None
    request: ResourcesSeedSessionRequest | None = None
    response: ResourcesSeedSessionResponse | None = None
    # Set by a close before it waits for the lock, so a seed still running stops its own sandbox.
    closing: bool = False
    verification_request: ScaleSWEVerifyRequest | None = None
    verification_response: ScaleSWEVerifyResponse | None = None


class _PatchCaptureError(Exception):
    """The agent's patch could not be read for a reason outside the agent's control, so there is no verdict."""

    def __init__(self, failure_kind: str, message: str) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind


class _UnreadablePatchError(Exception):
    """Git failed on the repository the agent left, so its patch cannot be read: a measured 0, not a lost verdict."""


class ScaleSWEResourcesServer(SimpleResourcesServer):
    config: ScaleSWEResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_id_to_sandbox: dict[str, AsyncSandbox] = {}
        self._session_id_to_pristine_untracked: dict[str, frozenset[str]] = {}
        self._session_id_to_base_commit: dict[str, str] = {}
        # Environment Server sessions by resources_session_id. A close drops the session and is remembered for
        # session_close_retry_window_seconds, so a late seed cannot recreate it.
        self._resources_sessions: dict[str, _ResourcesSession] = {}
        self._closed_sessions: ClosedResourcesSessionRecords[EpisodeId] = ClosedResourcesSessionRecords(
            self.config.session_close_retry_window_seconds
        )
        # Verifier sandboxes of Environment Server sessions whose stop has not succeeded yet.
        self._session_id_to_verifier_sandboxes: dict[str, list[AsyncSandbox]] = {}
        # Stops that a bounded wait gave up on, still running, so a later wait joins them instead of starting another.
        self._stop_tasks: dict[AsyncSandbox, asyncio.Task[None]] = {}
        self._shutting_down = False

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[Any]:
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                self._shutting_down = True
                # The sandboxes Environment Server sessions still hold. The cookie path keeps its behavior: what an
                # Agent's /run still holds is left to the provider's TTL.
                session_ids = set(self._resources_sessions) | set(self._session_id_to_verifier_sandboxes)
                await asyncio.gather(*(self._shutdown_session(session_id) for session_id in session_ids))

        app.router.lifespan_context = lifespan
        return app

    async def _shutdown_session(self, session_id: str) -> None:
        """Stop every sandbox a session still holds at shutdown.

        A request still in progress on an Environment Server session gets ``sandbox_stop_timeout_seconds`` to
        finish; after that the sandboxes are stopped without the session's lock. Never raises: a stop that fails
        or runs long is logged, and the provider's TTL reclaims the sandbox.
        """
        session = self._resources_sessions.get(session_id)
        locked = False
        try:
            if session is not None:
                try:
                    await asyncio.wait_for(session.lock.acquire(), self.config.sandbox_stop_timeout_seconds)
                    locked = True
                except TimeoutError:
                    LOG.warning("Scale-SWE session %s is still busy at shutdown; stopping its sandboxes", session_id)
            await self._stop_session_sandbox(session_id)
        except Exception:
            LOG.exception("Failed to stop Scale-SWE session %s on shutdown; provider TTL is the backstop", session_id)
        finally:
            if locked:
                session.lock.release()

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
        self, body: ScaleSWEInstanceRequest, files: dict[str, str] | None = None
    ) -> AsyncSandbox:
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
            provider_options=self.config.sandbox_config.get("provider_options", {}),
        )
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
            task = asyncio.create_task(sandbox.stop())
            # Retrieve the error of a stop nobody waits for anymore, so it is not reported as never retrieved.
            task.add_done_callback(lambda done: done.cancelled() or done.exception())
            self._stop_tasks[sandbox] = task
        try:
            await asyncio.wait_for(asyncio.shield(task), self.config.sandbox_stop_timeout_seconds)
        finally:
            if task.done() and self._stop_tasks.get(sandbox) is task:
                del self._stop_tasks[sandbox]

    def _forget_sandbox(self, session_id: str, sandbox: AsyncSandbox) -> None:
        """Drop a sandbox whose stop succeeded from what the session holds."""
        if self._session_id_to_sandbox.get(session_id) is sandbox:
            del self._session_id_to_sandbox[session_id]
        verifiers = self._session_id_to_verifier_sandboxes.get(session_id, [])
        if any(held is sandbox for held in verifiers):
            verifiers[:] = [held for held in verifiers if held is not sandbox]
            if not verifiers:
                del self._session_id_to_verifier_sandboxes[session_id]

    async def _stop_session_sandbox(self, session_id: str) -> None:
        """Stop every sandbox the session still holds, then forget the session's sandbox state.

        That is its task sandbox and any verifier sandbox whose stop failed or ran long. Each stop gets the bounded
        wait of ``_wait_for_stop``, and a sandbox is forgotten only once its stop succeeds. If any stop failed or
        is still running, this raises and the session keeps that sandbox, so close or shutdown can try again.
        """
        task_sandbox = self._session_id_to_sandbox.get(session_id)
        held = ([task_sandbox] if task_sandbox is not None else []) + self._session_id_to_verifier_sandboxes.get(
            session_id, []
        )
        outcomes = await asyncio.gather(*(self._wait_for_stop(sandbox) for sandbox in held), return_exceptions=True)
        for sandbox, outcome in zip(held, outcomes):
            if outcome is None:
                self._forget_sandbox(session_id, sandbox)
        if session_id not in self._session_id_to_sandbox:
            self._session_id_to_pristine_untracked.pop(session_id, None)
            self._session_id_to_base_commit.pop(session_id, None)
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome

    async def _stop_verifier_sandbox(self, session_id: str, sandbox: AsyncSandbox | None) -> None:
        """Stop an Environment Server session's verifier sandbox; one whose stop fails or runs long stays with it.

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
            self._forget_sandbox(session_id, sandbox)

    async def _stop_failed_seed(self, session_id: str, session: _ResourcesSession) -> None:
        """Stop the sandbox of a typed seed that failed; its caller never got it, so nobody else can stop it.

        A failed stop is logged, so the seed's own error propagates, and the sandbox stays with the session for close.
        Once the sandbox is stopped, a close that fenced the seed is complete.
        """
        try:
            await self._stop_session_sandbox(session_id)
        except Exception:
            LOG.exception("Failed to stop the sandbox of failed Scale-SWE seed %s; close can retry", session_id)
            return
        if session.closing:
            self._finish_close(session_id, session)

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

    async def _pristine_untracked_files(self, sandbox: AsyncSandbox, workdir: str) -> frozenset[str]:
        """List of files ``workdir`` holds untracked before the agent touches it."""
        try:
            result = await sandbox.exec(f"git -C {shlex.quote(workdir)} ls-files --others --exclude-standard")
            if result.return_code != 0:
                print(f"Failed to list pristine untracked files: {result.stderr}", file=sys.stderr)
                return frozenset()
            return frozenset(line.strip() for line in (result.stdout or "").splitlines() if line.strip())
        except Exception:
            print("Failed to list pristine untracked files", format_exc(), file=sys.stderr)
            return frozenset()

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
        """Capture an Environment Server session's patch per ``config.patch_capture_mode``, then stop its sandbox.

        Raises ``_UnreadablePatchError`` when git fails on the repository the agent left, and ``_PatchCaptureError``
        when the capture failed for any other reason. The patch is graded in a fresh sandbox, so the agent's sandbox
        is stopped either way; a stop that fails or runs long leaves it with the session for close.
        """
        sandbox = self._session_id_to_sandbox.get(session_id)
        base_commit = self._session_id_to_base_commit.get(session_id, "")
        try:
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
        finally:
            try:
                await self._stop_session_sandbox(session_id)
            except Exception:
                LOG.exception(
                    "Failed to stop Scale-SWE task sandbox %s after patch capture; close can retry", session_id
                )

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
        return await self._seed_session(request.session[SESSION_ID_KEY], body)

    @asynccontextmanager
    async def _locked_session(
        self, session_id: str, *, close_episode_id: EpisodeId | None = None
    ) -> AsyncIterator[_ResourcesSession]:
        """Hold the Environment Server session's lock.

        A close fences the session before it waits, so a seed still running stops its own sandbox and a close that
        times out still takes effect. A record that holds nothing is dropped on the way out.
        """
        while True:
            session = self._resources_sessions.setdefault(session_id, _ResourcesSession())
            if close_episode_id is not None:
                if session.episode_id not in (None, close_episode_id):
                    raise HTTPException(409, "episode_id does not match the resources session")
                session.episode_id = close_episode_id
                session.closing = True
            async with session.lock:
                # A close can finish and drop the record while this caller waits; start over from the current one.
                if self._resources_sessions.get(session_id) is not session:
                    continue
                try:
                    yield session
                finally:
                    if (
                        session.request is None
                        and not session.closing
                        and session_id not in self._session_id_to_sandbox
                        and self._resources_sessions.get(session_id) is session
                    ):
                        del self._resources_sessions[session_id]
                return

    def _finish_close(self, session_id: str, session: _ResourcesSession) -> None:
        """Drop a session whose sandbox is stopped, and remember its close for the retry window."""
        # Only a close finishes a session, and it set episode_id when it fenced the session.
        assert session.episode_id is not None
        if self._resources_sessions.get(session_id) is session:
            del self._resources_sessions[session_id]
        self._closed_sessions.remember(session_id, session.episode_id)

    async def _release_session(self, session_id: str, session: _ResourcesSession) -> None:
        """Stop the session's sandbox and complete its close; a failed stop raises and keeps the sandbox and fence."""
        await self._stop_session_sandbox(session_id)
        self._finish_close(session_id, session)

    async def _seed_resources_session(
        self, request: Request, body: ResourcesSeedSessionRequest
    ) -> ResourcesSeedSessionResponse:
        if self.config.is_verifying_golden_patch:
            # Golden-patch mode grades the row's own patch, so an agent's sandbox would go unused.
            raise HTTPException(422, "Golden-patch mode cannot be used with Environment Server sessions")
        try:
            task = ScaleSWESeedSessionRequest.model_validate(body.task_data)
        except ValidationError as error:
            raise HTTPException(422, detail=error.errors(include_input=False)) from error
        session_id = body.resources_session_id
        async with self._locked_session(session_id) as session:
            if session.closing or session_id in self._closed_sessions:
                raise HTTPException(409, "Resources session is already closed")
            if self._shutting_down:
                raise HTTPException(503, "Scale-SWE server is shutting down")
            if session.request is None and session.episode_id in (None, body.episode_id):
                session.episode_id = body.episode_id
                # An earlier failed seed may still hold a sandbox it could not stop; never start a second one.
                await self._stop_session_sandbox(session_id)
                try:
                    await self._seed_session(session_id, task, own_before_setup=True)
                    if not _COMMIT_ID.fullmatch(self._session_id_to_base_commit.get(session_id, "")):
                        # Without a base commit the agent's patch could never be captured: fail the seed instead, so
                        # the episode is retried.
                        raise HTTPException(
                            503, "git rev-parse HEAD did not name a commit in the task sandbox; retry the episode"
                        )
                    self._raise_if_closing(session)
                    descriptor = await self._session_id_to_sandbox[session_id].serialize()
                    self._raise_if_closing(session)
                except BaseException:
                    await self._stop_failed_seed(session_id, session)
                    raise
                session.request = body.model_copy(deep=True)
                session.response = ResourcesSeedSessionResponse(
                    resources_session_id=session_id,
                    sandbox_access=SandboxAccess(
                        connection=DirectSandboxConnection(
                            provider_config_ref=self.config.sandbox_provider, descriptor=descriptor
                        ),
                        workdir=task.workdir,
                    ),
                )
            elif session.request != body:
                raise HTTPException(409, "resources_session_id is already bound to a different request")
            elif session.verification_request is not None:
                raise HTTPException(409, "Resources session was already verified")
            request.session[SESSION_ID_KEY] = session_id
            return session.response

    def _raise_if_closing(self, session: _ResourcesSession) -> None:
        """Refuse to lend a sandbox that a close, or the server's shutdown, has claimed during setup."""
        if session.closing:
            raise HTTPException(409, "Resources session closed during setup")
        if self._shutting_down:
            raise HTTPException(503, "Scale-SWE server is shutting down")

    async def _seed_session(
        self, session_id: str, body: ScaleSWESeedSessionRequest, *, own_before_setup: bool = False
    ) -> ScaleSWESeedSessionResponse:
        await self._stop_sandbox(self._session_id_to_sandbox.pop(session_id, None))
        self._session_id_to_pristine_untracked.pop(session_id, None)
        self._session_id_to_base_commit.pop(session_id, None)

        sandbox = await self._create_sandbox(body)
        if own_before_setup:
            # An Environment Server session owns its sandbox from the start, so the failed seed's cleanup or a close
            # can stop it when the setup below fails.
            self._session_id_to_sandbox[session_id] = sandbox

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
            await apply_anti_cheat_setup(sandbox, body.workdir, body.instance_id, "scale_swe")
        # The anti-cheat scrub leaves no committer identity, so the agent's `git commit` would fail.
        await prepare_git_for_commits(sandbox, body.workdir, "scale_swe")

        head_result = await sandbox.exec(f"git -C {shlex.quote(body.workdir)} rev-parse HEAD")
        self._session_id_to_base_commit[session_id] = (head_result.stdout or "").strip()
        self._session_id_to_pristine_untracked[session_id] = await self._pristine_untracked_files(
            sandbox, body.workdir
        )
        self._session_id_to_sandbox[session_id] = sandbox
        return ScaleSWESeedSessionResponse(sandbox_handle=str(sandbox._handle.sandbox_id))

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        """Stop an Environment Server session's sandbox and remember the close for the retry window.

        A repeated close, or one for a session that was never seeded, succeeds; a later seed is refused. A stop that
        fails or times out raises and keeps the sandbox and the fence, so the caller can retry the close.
        """
        session_id = body.resources_session_id
        if session_id in self._closed_sessions:
            if self._closed_sessions.get(session_id) != body.episode_id:
                raise HTTPException(409, "episode_id does not match the closed resources session")
            return ResourcesCloseSessionResponse(resources_session_id=session_id)
        async with self._locked_session(session_id, close_episode_id=body.episode_id) as session:
            await self._release_session(session_id, session)
        return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def verify(self, request: Request, body: ScaleSWEVerifyRequest) -> ScaleSWEVerifyResponse:
        """Grade the agent's patch in a fresh sandbox.

        An Environment Server session keeps its first completed verification and replays it to an identical retry:
        the first verification stopped the agent's sandbox, so grading again would score an empty patch.
        """
        session_id = request.session[SESSION_ID_KEY]
        if session_id in self._resources_sessions:
            return await self._verify_resources_session(session_id, body)
        if session_id in self._closed_sessions:
            raise HTTPException(409, "Resources session is closed")
        return await self._verify(session_id, body, resources_session=False)

    async def _verify_resources_session(self, session_id: str, body: ScaleSWEVerifyRequest) -> ScaleSWEVerifyResponse:
        async with self._locked_session(session_id) as session:
            try:
                if session.closing or session.request is None:
                    raise HTTPException(409, "Resources session is not available for verification")
                instance_fields = set(ScaleSWEInstanceRequest.model_fields)
                seeded = ScaleSWEInstanceRequest.model_validate(session.request.task_data)
                if body.model_dump(include=instance_fields) != seeded.model_dump(include=instance_fields):
                    raise HTTPException(409, "Verification task does not match the seeded task")
                if session.verification_request is None:
                    # Recorded first: an attempt that ends without a verdict has already used the agent's sandbox.
                    session.verification_request = body.model_copy(deep=True)
                    response = await self._verify(session_id, body, resources_session=True)
                    if not response.evaluation_completed:
                        # A missing measurement is not the session's result; a retry gets 503 below.
                        return response
                    session.verification_response = response
                elif session.verification_request != body:
                    raise HTTPException(409, "Verification request does not match the first request")
                elif session.verification_response is None:
                    raise HTTPException(503, "Previous verification did not complete; retry the episode")
                return session.verification_response.model_copy(deep=True)
            finally:
                if session.closing:
                    # A close arrived while this verification ran, and may have given up waiting; complete it.
                    try:
                        await self._release_session(session_id, session)
                    except Exception:
                        LOG.exception("Failed to stop closing Scale-SWE session %s; close can retry", session_id)

    async def _verify(
        self, session_id: str, body: ScaleSWEVerifyRequest, *, resources_session: bool
    ) -> ScaleSWEVerifyResponse:
        """Capture the agent's patch (or take the golden one) and grade it.

        For an Environment Server session (``resources_session``):

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
                    await self._stop_verifier_sandbox(session_id, sandbox)
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
