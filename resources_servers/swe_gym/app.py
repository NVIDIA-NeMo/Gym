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
"""Resources server for SWE-Gym/SWE-Gym (2,438 SWE-bench-format Python tasks across 11 repos).

Every row names its own prebuilt image (``docker.io/xingyaoww/sweb.eval.x86_64.<owner>_s_<repo>-<pr>``)
with the repository at ``/testbed`` on ``base_commit`` and a conda env ``testbed``. Verification is
the SWE-bench recipe run in a fresh sandbox: apply the patch, re-run the repo's install step, apply the
held-out test patch, run the repo's pytest command, grade FAIL_TO_PASS / PASS_TO_PASS.

Two sandboxes per task, on purpose. The agent works in one created by ``seed_session`` that holds only
the image plus a git-history scrub; the golden patch, the test patch and the test lists stay in the
server process. Grading happens in a second sandbox created from the same image, seeded with those
files. Nothing an agent can read or run in its sandbox reveals the fix or the hidden tests.

Set ``is_verifying_golden_patch: true`` to grade the dataset's own patch instead of an agent's. That is
the dataset-health check: a row whose golden patch does not resolve is a broken row, and scoring an
agent against it measures noise. ``apply_golden_patch.py`` runs it over the training jsonl.
"""

import asyncio
import sys
from pathlib import Path
from shlex import quote
from time import time
from traceback import format_exc
from typing import Any

from fastapi import Request
from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
)
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from resources_servers.swe_gym.swebench_specs import spec_for
from resources_servers.swe_gym.verification import (
    CONDA_ENV,
    REPO_DIRECTORY,
    VerificationInputs,
    VerificationResult,
    as_list,
    drop_patch_sections,
    drop_test_patch_files,
    run_verification,
    verification_files,
)
from resources_servers.swebench.anti_cheat import apply_anti_cheat_setup
from resources_servers.swebench.patch_capture import (
    PatchCapture,
    PatchCaptureMode,
    capture_model_patch,
    prepare_git_for_commits,
)
from resources_servers.swebench.sandbox_sessions import SandboxSessionResourcesServer


# The image's default PATH points at the conda *base* interpreter (Python 3.11 without the repo's
# deps); the per-instance environment is the ``testbed`` env. Putting it first on PATH is the same
# effect as ``conda activate testbed`` for every shell the agent opens, without relying on the
# agent's tool to source conda. Explicit ``sandbox_config.env`` keys still win.
TESTBED_ENV = {
    "PATH": f"/opt/miniconda3/envs/{CONDA_ENV}/bin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "CONDA_DEFAULT_ENV": CONDA_ENV,
    "CONDA_PREFIX": f"/opt/miniconda3/envs/{CONDA_ENV}",
}


class SWEGymResourcesServerConfig(BaseResourcesServerConfig):
    is_verifying_golden_patch: bool = False
    # "worktree" (diff of the working tree) or "committed" (committed work only); see swebench/patch_capture.py.
    patch_capture_mode: PatchCaptureMode = "worktree"
    include_model_patch_in_response: bool = True
    evaluation_timeout: int | None = 3600
    # A verdict-less run is retried on a fresh sandbox: an image pull or a flaky provider start
    # is not evidence about the patch.
    inconclusive_verification_retries: int = 1
    apply_anti_cheating: bool = True
    sandbox_provider: str
    sandbox_config: dict[str, Any]


class SWEGymInstanceRequest(BaseModel):
    """One row of SWE-Gym/SWE-Gym, as written by prepare_swe_gym.py."""

    model_config = ConfigDict(extra="allow")

    instance_id: str
    repo: str
    version: str
    base_commit: str
    patch: str = ""
    test_patch: str = ""
    problem_statement: str = ""
    language: str = "python"
    image_name: str
    # Upstream spells these in caps; keep the dataset's own names so a row round-trips unchanged.
    FAIL_TO_PASS: list[str] | str = []
    PASS_TO_PASS: list[str] | str = []


class SWEGymSeedSessionRequest(SWEGymInstanceRequest, BaseSeedSessionRequest):
    sandbox_spec: dict[str, Any] | None = None


class SWEGymSeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str
    workdir: str


class SWEGymVerifyRequest(SWEGymInstanceRequest, BaseVerifyRequest):
    pass


class SWEGymVerifyResponse(BaseVerifyResponse):
    evaluation_completed: bool
    resolved: bool
    patch_applied: bool
    instance_id: str
    # Echoed so per-row sweep outputs can be grouped without re-reading the training jsonl.
    repo: str = ""
    version: str = ""
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


class SWEGymResourcesServer(SandboxSessionResourcesServer):
    ray_enabled = False
    config: SWEGymResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_id_to_sandbox: dict[str, AsyncSandbox] = {}
        self._session_id_to_pristine_untracked: dict[str, frozenset[str]] = {}

    def _inputs(self, body: SWEGymInstanceRequest, patch: str) -> VerificationInputs:
        return VerificationInputs(
            instance_id=body.instance_id,
            repo=body.repo,
            version=str(body.version),
            base_commit=body.base_commit,
            patch=drop_test_patch_files(patch, body.test_patch),
            test_patch=body.test_patch,
            fail_to_pass=as_list(body.FAIL_TO_PASS),
            pass_to_pass=as_list(body.PASS_TO_PASS),
        )

    async def _create_sandbox(self, body: SWEGymInstanceRequest, files: dict[str, str] | None = None) -> AsyncSandbox:
        global_config_dict = get_global_config_dict()
        provider_config = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        provider_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)

        # Cap test parallelism to the CPU limit: a container sees the HOST core count, so pytest-xdist
        # and BLAS fan out ~96 workers against a small quota and spend their time CFS-throttled.
        sandbox_resources = SandboxResources.from_mapping(self.config.sandbox_config.get("resources", {}))
        env = dict(self.config.sandbox_config.get("env", {}))
        if self.config.sandbox_config.get("derive_cpu_env", True):
            env = cpu_cap_env(sandbox_resources.cpu) | env
        env = TESTBED_ENV | env

        spec = SandboxSpec(
            # The row names its own image; there is no repository template to apply.
            image=body.image_name,
            ttl_s=self.config.sandbox_config.get("ttl_s"),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s"),
            workdir=REPO_DIRECTORY,
            env=env,
            files=files or {},
            metadata=provider_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {
                "nemo_gym_agent": self.config.name,
                "instance_id": body.instance_id[:63],
                "language": body.language[:63],
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
            print("Failed to stop SWE-Gym sandbox", format_exc(), file=sys.stderr)

    async def _pristine_untracked_files(self, sandbox: AsyncSandbox, workdir: str) -> frozenset[str]:
        """Files ``workdir`` holds untracked before the agent touches it (excluded from its patch)."""
        try:
            result = await sandbox.exec(f"git -C {quote(workdir)} ls-files --others --exclude-standard")
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
            await self._release_task_sandbox(session_id, original_sandbox)

    async def seed_session(
        self, request: Request, body: SWEGymSeedSessionRequest | ResourcesSeedSessionRequest
    ) -> SWEGymSeedSessionResponse | ResourcesSeedSessionResponse:
        """Start the instance's image so an agent can work in it.

        An Environment Server seeds a typed session and gets the sandbox back as ``sandbox_access``; an
        agent's ``/run`` seeds with the row and gets the sandbox handle.
        """
        if isinstance(body, ResourcesSeedSessionRequest):
            return await self.seed_task_sandbox_session(request, body, SWEGymInstanceRequest)
        session_id = request.session[SESSION_ID_KEY]
        await self._stop_sandbox(self._session_id_to_sandbox.pop(session_id, None))
        await self._start_task_sandbox(session_id, body)
        return SWEGymSeedSessionResponse(
            sandbox_handle=str(self._session_id_to_sandbox[session_id]._handle.sandbox_id), workdir=REPO_DIRECTORY
        )

    async def _start_task_sandbox(self, session_id: str, body: SWEGymInstanceRequest) -> str:
        """Start the task sandbox for ``session_id`` and return the directory the agent works in.

        The sandbox gets no files from the row. The git scrub drops every ref but HEAD and prunes the
        objects behind them: SWE-bench images clone the whole upstream history, so without it the fix
        commit and every later release tag sit in ``.git`` within a ``git log --all`` of the agent.
        """
        self._forget_task_sandbox_state(session_id)
        sandbox = await self._create_sandbox(body)
        # Own the sandbox before preparing it, so a failed seed can still stop it.
        self._session_id_to_sandbox[session_id] = sandbox
        if self.config.apply_anti_cheating:
            await apply_anti_cheat_setup(sandbox, REPO_DIRECTORY, body.instance_id, "swe_gym")
        # The anti-cheat scrub leaves no committer identity, so the agent's `git commit` would fail.
        await prepare_git_for_commits(sandbox, REPO_DIRECTORY, "swe_gym")
        self._session_id_to_pristine_untracked[session_id] = await self._pristine_untracked_files(
            sandbox, REPO_DIRECTORY
        )
        return REPO_DIRECTORY

    def _forget_task_sandbox_state(self, session_id: str) -> None:
        self._session_id_to_pristine_untracked.pop(session_id, None)

    def _response(self, body: SWEGymVerifyRequest, **fields: Any) -> SWEGymVerifyResponse:
        # Spread the request: BaseVerifyResponse extends BaseVerifyRequest, so responses_create_params
        # and response are required and must be echoed back.
        return SWEGymVerifyResponse.model_validate(
            body.model_dump()
            | {
                "instance_id": body.instance_id,
                "repo": body.repo,
                "version": str(body.version),
                "language": body.language,
            }
            | fields
        )

    async def verify(self, request: Request, body: SWEGymVerifyRequest) -> SWEGymVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        extraction_error = None
        mode = self.config.patch_capture_mode
        if self.config.is_verifying_golden_patch:
            capture = PatchCapture.static(body.patch, mode, "golden")
        else:
            self._claim_task_sandbox(session_id)
            try:
                capture = await self._extract_model_patch(session_id, REPO_DIRECTORY, body.base_commit)
            except Exception as exc:
                capture = PatchCapture.static("", mode, "none")
                extraction_error = f"Failed to extract model patch: {exc}"
        patch = capture.patch

        # Resolve the spec before spending a sandbox: a (repo, version) without one can never be graded.
        try:
            spec_for(body.repo, str(body.version))
        except KeyError as exc:
            return self._response(
                body,
                reward=0.0,
                evaluation_completed=False,
                resolved=False,
                patch_applied=False,
                test_results=None,
                test_output="",
                error=str(exc),
                eval_sandbox_start_time_taken=0.0,
                patch_verification_time_taken=0.0,
                **capture.response_fields(self.config.include_model_patch_in_response),
            )

        inputs = self._inputs(body, patch)
        log_dir = Path(__file__).parent / "logs" / body.instance_id
        files = verification_files(inputs)
        attempts = 1 + max(self.config.inconclusive_verification_retries, 0)
        start_time_taken = 0.0
        verification_time_taken = 0.0
        result = VerificationResult(False, False, False, None, "", "unattempted")
        for attempt in range(1, attempts + 1):
            sandbox: AsyncSandbox | None = None
            started = time()
            try:
                sandbox = await self._create_sandbox(body, files=files)
                start_time_taken = time() - started
                verification_started = time()
                result = await run_verification(
                    sandbox=sandbox, inputs=inputs, timeout_s=self.config.evaluation_timeout, log_dir=log_dir
                )
                verification_time_taken = time() - verification_started
            except asyncio.CancelledError:
                # A cancellation that is OURS (the client went away, or shutdown) must propagate. One that
                # surfaced from inside the sandbox client -- a long test run tripping an inner timeout -- is
                # an infrastructure verdict about this attempt, not a reason to drop the whole request.
                if asyncio.current_task().cancelling():
                    await self._stop_sandbox(sandbox)
                    raise
                start_time_taken = time() - started
                verification_time_taken = 0.0
                result = VerificationResult(
                    False, False, False, None, "", "Verification cancelled inside the sandbox client"
                )
            except Exception as exc:
                start_time_taken = time() - started
                verification_time_taken = 0.0
                result = VerificationResult(False, False, False, None, "", f"Verification failed: {exc}")
            finally:
                await self._stop_sandbox(sandbox)
            if result.completed:
                break
            if attempt < attempts:
                print(
                    f"[swe_gym] {body.instance_id}: inconclusive ({result.error}); "
                    f"retrying on a fresh sandbox ({attempt}/{attempts - 1})",
                    flush=True,
                )

        return self._response(
            body,
            # An unresolved-but-completed run is a real 0. An incomplete run is also 0, but
            # evaluation_completed distinguishes them so a broken row is not read as a hard task.
            reward=1.0 if result.resolved else 0.0,
            evaluation_completed=result.completed,
            resolved=result.resolved,
            patch_applied=result.patch_applied,
            test_results=result.test_results,
            test_output=result.test_output[-100_000:],
            error=extraction_error or result.error,
            eval_sandbox_start_time_taken=start_time_taken,
            patch_verification_time_taken=verification_time_taken,
            test_patch_failed=result.test_patch_failed,
            **capture.response_fields(self.config.include_model_patch_in_response),
        )


if __name__ == "__main__":
    SWEGymResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # Required whenever num_workers > 1: multi-worker uvicorn re-imports this entrypoint BY PATH in
    # each forked child and expects a module-level `app`.
    app = SWEGymResourcesServer.run_webserver()  # noqa: F401
