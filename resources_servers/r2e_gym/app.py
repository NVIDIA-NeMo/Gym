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
"""Resources server for R2E-Gym/R2E-Gym-Subset (4,578 synthetic-issue Python tasks across 10 repos).

Every row names its own prebuilt image (``docker.io/namanjain12/<repo>_final:<commit>``) holding the
repository at ``/testbed`` in its pre-fix state, with the project venv already on PATH, the held-out
tests under ``/r2e_tests`` and their runner at ``/testbed/run_tests.sh``. Verification is R2E-Gym's
own recipe run in a fresh sandbox: apply the patch, stage the runner and the tests, run them, and
require the per-test statuses to equal the row's ``expected_output_json`` exactly.

Two sandboxes per task, on purpose. The agent works in one created by ``seed_session`` that is scrubbed
first: ``/r2e_tests``, the runner and any R2E sidecar JSON are deleted (they describe the hidden tests
and the fix), and the git history is cut down to HEAD (the image's ``origin/master`` reaches the fix
commit). Grading happens in a second sandbox created from the same image, where those files are intact,
seeded with the candidate patch. Nothing an agent can read or run in its sandbox reveals the fix or the
hidden tests.

Set ``is_verifying_golden_patch: true`` to grade the dataset's own patch instead of an agent's. That is
the dataset-health check; ``apply_golden_patch.py`` runs it over the training jsonl.
"""

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
from resources_servers.r2e_gym.verification import (
    REPO_DIRECTORY,
    VerificationInputs,
    VerificationResult,
    drop_hidden_test_sections,
    drop_patch_sections,
    hide_from_agent_command,
    parse_expected,
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


class R2EGymResourcesServerConfig(BaseResourcesServerConfig):
    is_verifying_golden_patch: bool = False
    # "worktree" (diff of the working tree) or "committed" (committed work only); see swebench/patch_capture.py.
    patch_capture_mode: PatchCaptureMode = "worktree"
    include_model_patch_in_response: bool = True
    evaluation_timeout: int | None = 1800
    # A verdict-less run is retried on a fresh sandbox: an image pull or a flaky provider start
    # is not evidence about the patch.
    inconclusive_verification_retries: int = 1
    apply_anti_cheating: bool = True
    sandbox_provider: str
    sandbox_config: dict[str, Any]


class R2EGymInstanceRequest(BaseModel):
    """One row of R2E-Gym-Subset, as written by prepare_r2e_gym.py."""

    model_config = ConfigDict(extra="allow")

    instance_id: str
    repo_name: str
    commit_hash: str = ""
    patch: str = ""
    problem_statement: str = ""
    language: str = "python"
    image_name: str
    # The Hub ships this as a JSON string; the prepare script keeps it that way so a row round-trips.
    expected_output_json: str | dict[str, Any] = ""


class R2EGymSeedSessionRequest(R2EGymInstanceRequest, BaseSeedSessionRequest):
    sandbox_spec: dict[str, Any] | None = None


class R2EGymSeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str
    workdir: str


class R2EGymVerifyRequest(R2EGymInstanceRequest, BaseVerifyRequest):
    pass


class R2EGymVerifyResponse(BaseVerifyResponse):
    evaluation_completed: bool
    resolved: bool
    patch_applied: bool
    instance_id: str
    # Echoed so per-row sweep outputs can be grouped without re-reading the training jsonl.
    repo_name: str = ""
    language: str
    test_results: dict[str, Any] | None
    test_output: str
    error: str | None
    eval_sandbox_start_time_taken: float
    patch_verification_time_taken: float
    # Patch-capture provenance; see resources_servers/swebench/patch_capture.py.
    patch_source: str = "none"
    patch_branch: str | None = None
    patch_commits: int = 0
    worktree_dirty: bool = False
    model_patch_bytes: int = 0
    model_patch: str | None = None
    model_patch_sha256: str = ""


class R2EGymResourcesServer(SandboxSessionResourcesServer):
    ray_enabled = False
    config: R2EGymResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_id_to_sandbox: dict[str, AsyncSandbox] = {}
        self._session_id_to_pristine_untracked: dict[str, frozenset[str]] = {}

    def _inputs(self, body: R2EGymInstanceRequest, patch: str) -> VerificationInputs:
        return VerificationInputs(
            instance_id=body.instance_id,
            repo_name=body.repo_name,
            patch=drop_hidden_test_sections(patch),
            expected=parse_expected(body.expected_output_json),
        )

    async def _create_sandbox(self, body: R2EGymInstanceRequest, files: dict[str, str] | None = None) -> AsyncSandbox:
        global_config_dict = get_global_config_dict()
        provider_config = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        provider_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)

        # Cap test parallelism to the CPU limit: a container sees the HOST core count, so numpy/pandas
        # BLAS pools and xdist fan out ~96 workers against a small quota and CFS-throttle.
        sandbox_resources = SandboxResources.from_mapping(self.config.sandbox_config.get("resources", {}))
        env = dict(self.config.sandbox_config.get("env", {}))
        if self.config.sandbox_config.get("derive_cpu_env", True):
            env = cpu_cap_env(sandbox_resources.cpu) | env

        spec = SandboxSpec(
            # The row names its own image; the project venv is already first on its PATH.
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
            print("Failed to stop R2E-Gym sandbox", format_exc(), file=sys.stderr)

    async def _hide_tests_from_agent(self, sandbox: AsyncSandbox, instance_id: str) -> None:
        """Delete the hidden tests, their runner and any R2E sidecar JSON from the agent's sandbox.

        Not best-effort: if this fails the agent could read the held-out tests, so the rollout is
        refused rather than silently graded on a leaked task.
        """
        result = await sandbox.exec(hide_from_agent_command(), timeout_s=120)
        if result.return_code != 0:
            raise RuntimeError(
                f"[r2e_gym] {instance_id}: could not remove hidden-test artifacts (rc={result.return_code}): "
                f"{result.stderr}"
            )

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

    async def _extract_model_patch(self, session_id: str, workdir: str) -> PatchCapture:
        """Capture the agent's patch per ``config.patch_capture_mode``, then stop its sandbox."""
        original_sandbox = self._session_id_to_sandbox.pop(session_id)
        pristine_untracked = self._session_id_to_pristine_untracked.pop(session_id, frozenset())
        try:
            return await capture_model_patch(
                original_sandbox,
                workdir,
                "HEAD",
                mode=self.config.patch_capture_mode,
                pristine_untracked=pristine_untracked,
                drop_sections=drop_patch_sections,
            )
        finally:
            await self._release_task_sandbox(session_id, original_sandbox)

    async def seed_session(
        self, request: Request, body: R2EGymSeedSessionRequest | ResourcesSeedSessionRequest
    ) -> R2EGymSeedSessionResponse | ResourcesSeedSessionResponse:
        """Start the instance's image so an agent can work in it, minus everything that gives the task away.

        An Environment Server seeds a typed session and gets the sandbox back as ``sandbox_access``; an
        agent's ``/run`` seeds with the row and gets the sandbox handle.
        """
        if isinstance(body, ResourcesSeedSessionRequest):
            return await self.seed_task_sandbox_session(request, body, R2EGymInstanceRequest)
        session_id = request.session[SESSION_ID_KEY]
        await self._stop_sandbox(self._session_id_to_sandbox.pop(session_id, None))
        try:
            await self._start_task_sandbox(session_id, body)
        except BaseException:
            # A sandbox that leaked hidden tests must not be handed out or left running.
            await self._stop_sandbox(self._session_id_to_sandbox.pop(session_id, None))
            raise
        return R2EGymSeedSessionResponse(
            sandbox_handle=str(self._session_id_to_sandbox[session_id]._handle.sandbox_id), workdir=REPO_DIRECTORY
        )

    async def _start_task_sandbox(self, session_id: str, body: R2EGymInstanceRequest) -> str:
        """Start the task sandbox for ``session_id`` and return the directory the agent works in."""
        self._forget_task_sandbox_state(session_id)
        sandbox = await self._create_sandbox(body)
        # Own the sandbox before preparing it, so a failed seed can still stop it.
        self._session_id_to_sandbox[session_id] = sandbox
        await self._hide_tests_from_agent(sandbox, body.instance_id)
        if self.config.apply_anti_cheating:
            await apply_anti_cheat_setup(sandbox, REPO_DIRECTORY, body.instance_id, "r2e_gym")
        # The anti-cheat scrub leaves no committer identity, so the agent's `git commit` would fail.
        await prepare_git_for_commits(sandbox, REPO_DIRECTORY, "r2e_gym")
        self._session_id_to_pristine_untracked[session_id] = await self._pristine_untracked_files(
            sandbox, REPO_DIRECTORY
        )
        return REPO_DIRECTORY

    def _forget_task_sandbox_state(self, session_id: str) -> None:
        self._session_id_to_pristine_untracked.pop(session_id, None)

    def _response(self, body: R2EGymVerifyRequest, **fields: Any) -> R2EGymVerifyResponse:
        # Spread the request: BaseVerifyResponse extends BaseVerifyRequest, so responses_create_params
        # and response are required and must be echoed back.
        return R2EGymVerifyResponse.model_validate(
            body.model_dump()
            | {"instance_id": body.instance_id, "repo_name": body.repo_name, "language": body.language}
            | fields
        )

    async def verify(self, request: Request, body: R2EGymVerifyRequest) -> R2EGymVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        extraction_error = None
        mode = self.config.patch_capture_mode
        if self.config.is_verifying_golden_patch:
            capture = PatchCapture.static(body.patch, mode, "golden")
        else:
            self._claim_task_sandbox(session_id)
            try:
                capture = await self._extract_model_patch(session_id, REPO_DIRECTORY)
            except Exception as exc:
                capture = PatchCapture.static("", mode, "none")
                extraction_error = f"Failed to extract model patch: {exc}"

        inputs = self._inputs(body, capture.patch)
        if not inputs.expected:
            # Without the expected statuses there is nothing to compare against; the row cannot be graded.
            return self._response(
                body,
                reward=0.0,
                evaluation_completed=False,
                resolved=False,
                patch_applied=False,
                test_results=None,
                test_output="",
                error="row has no expected_output_json",
                eval_sandbox_start_time_taken=0.0,
                patch_verification_time_taken=0.0,
                **capture.response_fields(self.config.include_model_patch_in_response),
            )

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
                    f"[r2e_gym] {body.instance_id}: inconclusive ({result.error}); "
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
            **capture.response_fields(self.config.include_model_patch_in_response),
        )


if __name__ == "__main__":
    R2EGymResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # Required whenever num_workers > 1: multi-worker uvicorn re-imports this entrypoint BY PATH in
    # each forked child and expects a module-level `app`.
    app = R2EGymResourcesServer.run_webserver()  # noqa: F401
