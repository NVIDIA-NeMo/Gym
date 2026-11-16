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
"""Resources server for Swemer Ultra SWE task packages (the swe_bench_ext OTS delivery).

The dataset is not on the Hub: it is internal, sourced from a local delivery folder
(``ext-nvidia-ots-delivery/<delivery>/swe_bench_ext/tasks/<task>/``) plus the image manifest
``swebext_final_manifest.tsv`` (``status == "ready"`` rows only); ``data/swemer_ultra_training.jsonl`` is built offline (see the README).

Every task carries its own prebuilt image with ``/workspace/repo`` checked out at the task's
base commit. Verification: apply the candidate patch and the dataset's held-out ``test_patch``,
run the row's own ``test_command`` (with swe-bench-ext's own output-flag injection, see
``verification.build_eval_script``), and grade FAIL_TO_PASS/PASS_TO_PASS via
``responses_api_agents.swe_agents.swe_bench_ext.parsing`` rather than a hand-rolled parser --
this delivery spans 13 test frameworks, not the 5 swemer_v2 supports.

Set ``is_verifying_golden_patch: true`` to grade the dataset's own patch instead of an agent's,
the dataset-health check: a row whose golden patch does not resolve is a broken row.
"""

import shlex
import sys
from pathlib import Path
from time import time
from traceback import format_exc
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from fastapi import Request
from pydantic import BaseModel, ConfigDict

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import get_global_config_dict
from nemo_gym.core import VERIFIER_ERROR
from nemo_gym.sandbox import (
    AsyncSandbox,
    CpuCappedSandboxConfig,
    resolve_provider_config,
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
from resources_servers.swebench.anti_cheat import apply_anti_cheat_setup
from resources_servers.swebench.patch_capture import (
    CaptureCommandError,
    PatchCapture,
    PatchCaptureMode,
    capture_model_patch,
    prepare_git_for_commits,
)
from resources_servers.swebench.sandbox_sessions import SandboxSessionResourcesServer
from resources_servers.swemer_ultra.verification import (
    VerificationInputs,
    VerificationResult,
    drop_test_patch_files,
    mirror_files,
    run_verification,
    verification_files,
)


JVM_MIRROR_ENV = {
    "GRADLE_USER_HOME": "/root/.gradle",
}


class SwemerUltraResourcesServerConfig(BaseResourcesServerConfig):
    is_verifying_golden_patch: bool = False
    # "worktree" (diff of the working tree) or "committed" (committed work only); see swebench/patch_capture.py.
    patch_capture_mode: PatchCaptureMode = "worktree"
    include_model_patch_in_response: bool = True
    evaluation_timeout: int | None = 1200
    # None disables disk logs. Set an output directory to retain per-task, per-attempt logs.
    verification_log_dir: Path | None = None
    # A verdict-less run is retried on a fresh sandbox: an image pull or a flaky provider start
    # is not evidence about the patch.
    inconclusive_verification_retries: int = 1
    apply_anti_cheating: bool = True
    # Write the Maven Central mirror to ~/.m2/settings.xml in the agent and eval sandboxes. Turn off for providers
    # that mount their own read-only settings.xml, e.g. one that already mirrors Maven Central.
    overwrite_maven_settings: bool = True
    sandbox_provider: str
    sandbox_config: CpuCappedSandboxConfig


class SwemerUltraInstanceRequest(BaseModel):
    """One row of ``data/swemer_ultra_training.jsonl``, a Swemer Ultra task package."""

    model_config = ConfigDict(extra="allow")

    instance_id: str
    delivery: str = ""
    language: str = ""
    workdir: str = "/workspace/repo"
    image_ref: str
    patch: str = ""
    test_patch: str = ""
    problem_statement: str = ""
    test_framework: str
    test_command: str
    FAIL_TO_PASS: list[str] = []
    PASS_TO_PASS: list[str] = []


class SwemerUltraSeedSessionRequest(SwemerUltraInstanceRequest, BaseSeedSessionRequest):
    sandbox_spec: dict[str, Any] | None = None


class SwemerUltraSeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str
    workdir: str


class SwemerUltraVerifyRequest(SwemerUltraInstanceRequest, BaseVerifyRequest):
    pass


class SwemerUltraVerifyResponse(BaseVerifyResponse):
    verification_log_dir: str | None = None
    evaluation_completed: bool
    resolved: bool
    patch_applied: bool
    instance_id: str
    language: str
    test_framework: str
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


class SwemerUltraResourcesServer(SandboxSessionResourcesServer):
    config: SwemerUltraResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_id_to_sandbox: dict[str, AsyncSandbox] = {}
        self._session_id_to_pristine_untracked: dict[str, frozenset[str]] = {}
        self._session_id_to_base_commit: dict[str, str] = {}

    def _inputs(self, body: SwemerUltraInstanceRequest, patch: str) -> VerificationInputs:
        return VerificationInputs(
            instance_id=body.instance_id,
            workdir=body.workdir,
            patch=drop_test_patch_files(patch, body.test_patch),
            test_patch=body.test_patch,
            test_framework=body.test_framework,
            test_command=body.test_command,
            fail_to_pass=list(body.FAIL_TO_PASS),
            pass_to_pass=list(body.PASS_TO_PASS),
        )

    async def _create_sandbox(
        self,
        body: SwemerUltraInstanceRequest,
        files: dict[str, str] | None = None,
        *,
        session_id: str | None = None,
        verifier_session_id: str | None = None,
    ) -> AsyncSandbox:
        global_config_dict = get_global_config_dict()
        provider_config = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        spec = self.config.sandbox_config.spec(
            self.config.sandbox_provider,
            # The row names its own image; there is no repository template to apply.
            image=body.image_ref,
            workdir=body.workdir,
            # Gradle only auto-loads init scripts from $GRADLE_USER_HOME/init.d, so the mirror script
            # in verification._mirror_files() is invisible unless this points at the same home. Maven
            # needs no equivalent: it reads ~/.m2/settings.xml by default.
            env=JVM_MIRROR_ENV,
            files=files,
            metadata={
                "nemo_gym_agent": self.config.name,
                "instance_id": body.instance_id[:63],
            },
            named_configs=global_config_dict,
        )
        sandbox = AsyncSandbox(provider_config)
        if session_id is not None:
            self._session_id_to_sandbox[session_id] = sandbox
        if verifier_session_id is not None:
            self._verifier_sandboxes[verifier_session_id] = sandbox
        await sandbox.start(spec)
        return sandbox

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
        """Capture the agent's patch per ``config.patch_capture_mode``, leaving cleanup to the session lifecycle."""
        from resources_servers.swemer_ultra.verification import drop_patch_sections

        original_sandbox = self._session_id_to_sandbox[session_id]
        pristine_untracked = self._session_id_to_pristine_untracked.get(session_id, frozenset())
        return await capture_model_patch(
            original_sandbox,
            workdir,
            base_commit,
            mode=self.config.patch_capture_mode,
            pristine_untracked=pristine_untracked,
            drop_sections=drop_patch_sections,
        )

    async def seed_session(
        self, request: Request, body: SwemerUltraSeedSessionRequest | ResourcesSeedSessionRequest
    ) -> SwemerUltraSeedSessionResponse | ResourcesSeedSessionResponse:
        """Seed an episode session or the existing agent-run sandbox-handle response."""
        if isinstance(body, ResourcesSeedSessionRequest):
            return await self.seed_task_sandbox_session(request, body, SwemerUltraInstanceRequest)
        session_id = request.session[SESSION_ID_KEY]
        async with self._locked_task_session(session_id) as record:
            if record.seed is not None or self._task_session_fences.get(session_id) is not None:
                raise ResourcesSessionConflictError("An agent-run seed cannot replace an episode session")
            await self._stop_task_sandbox(session_id)
            try:
                await self._start_task_sandbox(session_id, body)
                sandbox = self._session_id_to_sandbox[session_id]
                response = SwemerUltraSeedSessionResponse(sandbox_handle=str(sandbox.sandbox_id), workdir=body.workdir)
            except BaseException:
                await self._stop_failed_seed_sandbox(session_id)
                raise
            return response

    async def _start_task_sandbox(self, session_id: str, body: SwemerUltraInstanceRequest) -> str:
        self._forget_task_sandbox_state(session_id)
        sandbox = await self._create_sandbox(
            body, files=mirror_files(maven_settings=self.config.overwrite_maven_settings), session_id=session_id
        )
        await self._ensure_git_repo(sandbox, body.workdir)
        if self.config.apply_anti_cheating:
            await apply_anti_cheat_setup(sandbox, body.workdir, body.instance_id, "swemer_ultra")
        # The anti-cheat scrub leaves no committer identity, so the agent's `git commit` would fail.
        await prepare_git_for_commits(sandbox, body.workdir, "swemer_ultra")

        head_result = await sandbox.exec(f"git -C {shlex.quote(body.workdir)} rev-parse HEAD")
        if (
            head_result.error_type is not None
            or head_result.return_code != 0
            or not (head_result.stdout or "").strip()
        ):
            raise RuntimeError("Task sandbox did not produce a base commit")
        self._session_id_to_base_commit[session_id] = head_result.stdout.strip()
        self._session_id_to_pristine_untracked[session_id] = await self._pristine_untracked_files(
            sandbox, body.workdir
        )
        return body.workdir

    def _forget_task_sandbox_state(self, session_id: str) -> None:
        self._session_id_to_pristine_untracked.pop(session_id, None)
        self._session_id_to_base_commit.pop(session_id, None)

    def _incomplete_response(
        self, body: SwemerUltraVerifyRequest, reason: str, **fields: Any
    ) -> SwemerUltraVerifyResponse:
        return SwemerUltraVerifyResponse.model_validate(
            body.model_dump()
            | {
                "reward": 0.0,
                "evaluation_completed": False,
                "mask_sample": True,
                "failure_kind": VERIFIER_ERROR,
                "failure_reason": reason,
                "resolved": False,
                "patch_applied": False,
                "test_results": None,
                "test_output": "",
                "error": reason,
                "eval_sandbox_start_time_taken": 0.0,
                "patch_verification_time_taken": 0.0,
            }
            | fields
        )

    async def verify(self, request: Request, body: SwemerUltraVerifyRequest) -> SwemerUltraVerifyResponse:
        """Serialize verification with close and reuse the saved episode verdict on retries."""
        return await self.verify_task_sandbox_session(
            request,
            body,
            verify=lambda: self._verify(request, body),
            lost_response=lambda reason, kind: self._incomplete_response(
                body, reason, **lost_session_verification(reason, failure_kind=kind)
            ),
            allow_seedless=self.config.is_verifying_golden_patch,
        )

    async def _verify(self, request: Request, body: SwemerUltraVerifyRequest) -> SwemerUltraVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        extraction_error = None
        mode = self.config.patch_capture_mode
        if self.config.is_verifying_golden_patch:
            capture = PatchCapture.static(body.patch, mode, "golden")
        else:

            async def capture_or_fallback() -> tuple[PatchCapture, str | None, bool]:
                try:
                    capture = await self._extract_model_patch(
                        session_id, body.workdir, self._session_id_to_base_commit[session_id]
                    )
                    return capture, None, False
                except CaptureCommandError as exc:
                    # A real git failure can reflect a repository damaged by the agent. Keep its empty-patch
                    # grading behavior; unavailable execution belongs to the infrastructure denominator.
                    incomplete = exc.error_type is not None or not isinstance(exc.return_code, int)
                    return PatchCapture.static("", mode, "none"), f"Failed to extract model patch: {exc}", incomplete
                except Exception as exc:
                    return PatchCapture.static("", mode, "none"), f"Failed to extract model patch: {exc}", True

            capture, extraction_error, capture_incomplete = await self.capture_task_sandbox_patch(
                session_id, capture_or_fallback
            )
            await self.stop_task_sandbox_after_capture(session_id)
            if capture_incomplete:
                # An unavailable patch is not evidence that the agent submitted an empty patch.
                return self._incomplete_response(
                    body, extraction_error, **capture.response_fields(self.config.include_model_patch_in_response)
                )
        patch = capture.patch

        inputs = self._inputs(body, patch)
        log_dir = (
            self.config.verification_log_dir / f"instance-{quote(body.instance_id, safe='')}" / uuid4().hex
            if self.config.verification_log_dir is not None
            else None
        )

        files = verification_files(inputs, overwrite_maven_settings=self.config.overwrite_maven_settings)
        attempts = 1 + max(self.config.inconclusive_verification_retries, 0)
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
        for attempt in range(1, attempts + 1):
            sandbox: AsyncSandbox | None = None
            started = time()
            try:
                sandbox = await self._create_sandbox(body, files=files, verifier_session_id=session_id)
                start_time_taken = time() - started
                verification_started = time()
                result = await run_verification(
                    sandbox=sandbox,
                    inputs=inputs,
                    timeout_s=self.config.evaluation_timeout,
                    log_dir=log_dir / f"attempt-{attempt}" if log_dir is not None else None,
                )
                verification_time_taken = time() - verification_started
            except Exception as exc:
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
                if not result.completed and attempt < attempts:
                    await self._stop_verifier_sandbox(session_id)
            if result.completed:
                break
            if attempt < attempts:
                print(
                    f"[swemer_ultra] {body.instance_id}: inconclusive ({result.error}); "
                    f"retrying on a fresh sandbox ({attempt}/{attempts - 1})",
                    flush=True,
                )

        return SwemerUltraVerifyResponse.model_validate(
            body.model_dump()
            | {
                # An unresolved-but-completed run is a real 0. An incomplete run is also 0, but
                # evaluation_completed distinguishes them so a broken row is not read as a hard task.
                "reward": 1.0 if result.resolved else 0.0,
                "evaluation_completed": result.completed,
                "verification_log_dir": str(log_dir) if log_dir is not None else None,
                **(
                    {"mask_sample": True, "failure_kind": VERIFIER_ERROR, "failure_reason": result.error}
                    if not result.completed
                    else {}
                ),
                "resolved": result.resolved,
                "patch_applied": result.patch_applied,
                "instance_id": body.instance_id,
                "language": body.language,
                "test_framework": body.test_framework,
                "test_results": result.test_results,
                "test_output": result.test_output[-100_000:],
                "error": extraction_error or result.error,
                "eval_sandbox_start_time_taken": start_time_taken,
                "patch_verification_time_taken": verification_time_taken,
                "test_patch_failed": result.test_patch_failed,
                **capture.response_fields(self.config.include_model_patch_in_response),
            }
        )


if __name__ == "__main__":
    SwemerUltraResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # Required whenever num_workers > 1. Multi-worker uvicorn re-imports this entrypoint BY PATH
    # in each forked child and expects a module-level `app`; without it every child exits, and
    # uvicorn responds by stopping the parent. The server then never binds, so the symptom is not
    # an import error but a flood of connection failures from clients talking to a dead port.
    app = SwemerUltraResourcesServer.run_webserver()  # noqa: F401
