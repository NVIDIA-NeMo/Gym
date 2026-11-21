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
"""Resources server for SWE tasks graded by their own test-runner report.

Each task row names a prebuilt image that holds the repository at its base commit with dependencies installed, plus
what the task was validated with: the exact test argv, the runner's JSON report format, setup commands, the editable
source paths, and FAIL_TO_PASS / PASS_TO_PASS ids in that runner's own ``<spec>::<full test name>`` form. The episode
lifecycle (agent sandbox, anti-cheat scrub, patch capture, a fresh verifier sandbox, bounded retries of inconclusive
runs) is ``swemer_v2``'s; this server changes what happens around it:

* the task's setup commands run before the baseline commit, so their outputs are part of the agent's starting tree,
  and again in the verifier after the patches;
* only edits to editable source files are replayed into the verifier;
* the hidden test patch must apply, and the runner's report is graded against the recorded ids (see verification.py).

Set ``is_verifying_golden_patch: true`` to grade each row's own ``patch`` instead of an agent's.
"""

import shlex
from contextvars import ContextVar
from time import time
from typing import Any, Sequence
from urllib.parse import quote
from uuid import uuid4

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import VERIFIER_ERROR
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.server_runtime import (
    BaseSeedSessionRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.swe_runner_report.verification import (
    ResultsFormat,
    VerificationInputs,
    VerificationResult,
    run_verification,
    verification_files,
)
from resources_servers.swebench.anti_cheat import apply_anti_cheat_setup
from resources_servers.swebench.patch_capture import (
    CaptureCommandError,
    PatchCapture,
    capture_model_patch,
    prepare_git_for_commits,
)
from resources_servers.swemer_v2.app import (
    SwemerV2ResourcesServer,
    SwemerV2ResourcesServerConfig,
    SwemerV2SeedSessionResponse,
)
from resources_servers.swemer_v2.verification import drop_patch_sections


class SweRunnerReportResourcesServerConfig(SwemerV2ResourcesServerConfig):
    # Bounds the whole verifier script; the test command itself is bounded by the row's test_timeout_s.
    evaluation_timeout: int | None = 2400


class SweRunnerReportInstanceRequest(BaseModel):
    """One task row (see task_data.py for the field contract)."""

    model_config = ConfigDict(extra="allow")

    instance_id: str
    language: str = ""
    workdir: str = "/app/repo"
    image_ref: str
    patch: str = ""
    test_patch: str = ""
    problem_statement: str = ""
    test_argv: list[str] = Field(min_length=1)
    results_format: ResultsFormat
    spec_files: list[str] = []
    source_prefixes: list[str] = Field(min_length=1)
    source_suffixes: list[str] = Field(min_length=1)
    setup_commands: list[list[str]] = []
    setup_timeout_s: int = Field(default=900, gt=0)
    test_timeout_s: int = Field(default=1200, gt=0)
    FAIL_TO_PASS: list[str] = []
    PASS_TO_PASS: list[str] = []


class SweRunnerReportSeedSessionRequest(SweRunnerReportInstanceRequest, BaseSeedSessionRequest):
    sandbox_spec: dict[str, Any] | None = None


class SweRunnerReportVerifyRequest(SweRunnerReportInstanceRequest, BaseVerifyRequest):
    pass


class SweRunnerReportVerifyResponse(BaseVerifyResponse):
    verification_log_dir: str | None = None
    evaluation_completed: bool
    resolved: bool
    patch_applied: bool
    instance_id: str
    language: str
    results_format: ResultsFormat
    test_results: dict[str, Any] | None
    test_output: str
    error: str | None
    eval_sandbox_start_time_taken: float
    patch_verification_time_taken: float
    test_patch_failed: bool = False
    # Paths of the captured patch that were not replayed (outside the editable source files).
    unreplayed_patch_paths: list[str] = []
    # Patch-capture provenance; see resources_servers/swebench/patch_capture.py.
    patch_source: str = "none"
    patch_branch: str | None = None
    patch_commits: int = 0
    worktree_dirty: bool = False
    model_patch_bytes: int = 0
    model_patch: str | None = None
    model_patch_sha256: str = ""


# The row being seeded, for the _ensure_git_repo hook (a ContextVar is per asyncio task, so per request).
_SEEDING: ContextVar[SweRunnerReportInstanceRequest | None] = ContextVar("swe_runner_report_seeding", default=None)


class SweRunnerReportResourcesServer(SwemerV2ResourcesServer):
    config: SweRunnerReportResourcesServerConfig

    def _inputs(self, body: SweRunnerReportInstanceRequest, patch: str) -> VerificationInputs:
        return VerificationInputs(
            instance_id=body.instance_id,
            workdir=body.workdir,
            patch=patch,
            test_patch=body.test_patch,
            test_argv=body.test_argv,
            results_format=body.results_format,
            spec_files=body.spec_files,
            setup_commands=body.setup_commands,
            setup_timeout_s=body.setup_timeout_s,
            test_timeout_s=body.test_timeout_s,
            fail_to_pass=body.FAIL_TO_PASS,
            pass_to_pass=body.PASS_TO_PASS,
            source_prefixes=body.source_prefixes,
            source_suffixes=body.source_suffixes,
        )

    async def _run_setup(self, sandbox: AsyncSandbox, body: SweRunnerReportInstanceRequest) -> None:
        for index, command in enumerate(body.setup_commands):
            result = await sandbox.exec(
                f"cd {shlex.quote(body.workdir)} && timeout -k 10 {body.setup_timeout_s} {shlex.join(command)}",
                timeout_s=body.setup_timeout_s + 60,
            )
            if result.error_type is not None or result.return_code != 0:
                raise RuntimeError(
                    f"{body.instance_id}: setup command {index} failed "
                    f"(return_code={result.return_code}, error_type={result.error_type})"
                )

    async def _ensure_git_repo(self, sandbox: AsyncSandbox, workdir: str) -> None:
        body = _SEEDING.get()
        if body is not None and body.setup_commands:
            await self._run_setup(sandbox, body)
        await super()._ensure_git_repo(sandbox, workdir)

    async def _extract_model_patch(self, session_id: str, workdir: str, base_commit: str) -> PatchCapture:
        return await capture_model_patch(
            self._session_id_to_sandbox[session_id],
            workdir,
            base_commit,
            mode=self.config.patch_capture_mode,
            pristine_untracked=self._session_id_to_pristine_untracked.get(session_id, frozenset()),
            drop_sections=drop_patch_sections,
        )

    async def seed_session(
        self, request: Request, body: SweRunnerReportSeedSessionRequest | ResourcesSeedSessionRequest
    ) -> SwemerV2SeedSessionResponse | ResourcesSeedSessionResponse:
        """Seed an episode session or the existing agent-run sandbox-handle response."""
        if isinstance(body, ResourcesSeedSessionRequest):
            return await self.seed_task_sandbox_session(request, body, SweRunnerReportInstanceRequest)
        session_id = request.session[SESSION_ID_KEY]
        async with self._locked_task_session(session_id) as record:
            if record.seed is not None or self._task_session_fences.get(session_id) is not None:
                raise ResourcesSessionConflictError("An agent-run seed cannot replace an episode session")
            await self._stop_task_sandbox(session_id)
            try:
                await self._start_task_sandbox(session_id, body)
                sandbox = self._session_id_to_sandbox[session_id]
                return SwemerV2SeedSessionResponse(sandbox_handle=str(sandbox.sandbox_id), workdir=body.workdir)
            except BaseException:
                await self._stop_failed_seed_sandbox(session_id)
                raise

    async def _start_task_sandbox(self, session_id: str, body: SweRunnerReportInstanceRequest) -> str:
        token = _SEEDING.set(body)
        try:
            self._forget_task_sandbox_state(session_id)
            sandbox = await self._create_sandbox(body, session_id=session_id)
            await self._ensure_git_repo(sandbox, body.workdir)
        finally:
            _SEEDING.reset(token)
        if self.config.apply_anti_cheating:
            await apply_anti_cheat_setup(sandbox, body.workdir, body.instance_id, "swe_runner_report")
        # The anti-cheat scrub leaves no committer identity, so the agent's `git commit` would fail.
        await prepare_git_for_commits(sandbox, body.workdir, "swe_runner_report")
        head = await sandbox.exec(f"git -C {shlex.quote(body.workdir)} rev-parse HEAD")
        if head.error_type is not None or head.return_code != 0 or not (head.stdout or "").strip():
            raise RuntimeError("Task sandbox did not produce a base commit")
        self._session_id_to_base_commit[session_id] = head.stdout.strip()
        self._session_id_to_pristine_untracked[session_id] = await self._pristine_untracked_files(
            sandbox, body.workdir
        )
        return body.workdir

    def _incomplete_response(
        self, body: SweRunnerReportVerifyRequest, reason: str, **fields: Any
    ) -> SweRunnerReportVerifyResponse:
        return SweRunnerReportVerifyResponse.model_validate(
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

    async def verify(self, request: Request, body: SweRunnerReportVerifyRequest) -> SweRunnerReportVerifyResponse:
        return await super().verify(request, body)

    async def _verify(self, request: Request, body: SweRunnerReportVerifyRequest) -> SweRunnerReportVerifyResponse:
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
                    # A git failure can reflect a repository the agent damaged: grade it as an empty patch.
                    # Unavailable execution belongs to the infrastructure denominator.
                    incomplete = exc.error_type is not None or not isinstance(exc.return_code, int)
                    return PatchCapture.static("", mode, "none"), f"Failed to extract model patch: {exc}", incomplete
                except Exception as exc:
                    return PatchCapture.static("", mode, "none"), f"Failed to extract model patch: {exc}", True

            capture, extraction_error, capture_incomplete = await self.capture_task_sandbox_patch(
                session_id, capture_or_fallback
            )
            await self.stop_task_sandbox_after_capture(session_id)
            if capture_incomplete:
                return self._incomplete_response(
                    body, extraction_error, **capture.response_fields(self.config.include_model_patch_in_response)
                )

        inputs = self._inputs(body, capture.patch)
        unreplayed: Sequence[str] = inputs.replayed_patch()[1]
        log_dir = (
            self.config.verification_log_dir / f"instance-{quote(body.instance_id, safe='')}" / uuid4().hex
            if self.config.verification_log_dir is not None
            else None
        )
        files = verification_files(inputs)
        attempts = 1 + max(self.config.inconclusive_verification_retries, 0)
        start_time_taken = verification_time_taken = 0.0
        result = VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=False,
            test_results=None,
            test_output="",
            error="unattempted",
        )
        for attempt in range(1, attempts + 1):
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
                start_time_taken, verification_time_taken = time() - started, 0.0
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
                    f"[swe_runner_report] {body.instance_id}: inconclusive ({result.error}); "
                    f"retrying on a fresh sandbox ({attempt}/{attempts - 1})",
                    flush=True,
                )

        return SweRunnerReportVerifyResponse.model_validate(
            body.model_dump()
            | {
                # An unresolved-but-completed run is a real 0; an incomplete run is masked, not read as a hard task.
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
                "results_format": body.results_format,
                "test_results": result.test_results,
                "test_output": result.test_output[-100_000:],
                "error": extraction_error or result.error,
                "eval_sandbox_start_time_taken": start_time_taken,
                "patch_verification_time_taken": verification_time_taken,
                "test_patch_failed": result.test_patch_failed,
                "unreplayed_patch_paths": list(unreplayed)[:50],
                **capture.response_fields(self.config.include_model_patch_in_response),
            }
        )


if __name__ == "__main__":
    SweRunnerReportResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # Required whenever num_workers > 1: forked uvicorn workers re-import this entrypoint by path and need `app`.
    app = SweRunnerReportResourcesServer.run_webserver()  # noqa: F401
