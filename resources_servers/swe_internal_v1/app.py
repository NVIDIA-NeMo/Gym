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
"""Resources server for the internal-v1 SWE tasks (swe_internal_v1).

Same shape as swemer_v1/swemer_oml: the dataset is internal, every row names its own prebuilt
image with the repository checked out at ``/app`` on the task's base commit, and
``data/swe_internal_v1_training.jsonl`` is distributed offline (there is no prepare script here).
Grading follows the vendor's own harness contract: the row's ``run_script.sh`` runs the selected
test files, its ``parsing_script.py`` turns the output into per-test statuses, and the task is
resolved when every FAIL_TO_PASS and PASS_TO_PASS id is observed passing (see ``verification.py``).

Set ``is_verifying_golden_patch: true`` to grade the dataset's own patch instead of an agent's,
the dataset-health check: a row whose golden patch does not resolve is a broken row.
"""

import shlex
import sys
from pathlib import Path
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
    SimpleResourcesServer,
)
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from resources_servers.swe_internal_v1.verification import (
    VerificationInputs,
    VerificationResult,
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


class SweInternalV1ResourcesServerConfig(BaseResourcesServerConfig):
    is_verifying_golden_patch: bool = False
    # "worktree" (diff of the working tree) or "committed" (committed work only); see swebench/patch_capture.py.
    patch_capture_mode: PatchCaptureMode = "worktree"
    include_model_patch_in_response: bool = True
    # The vendor's recorded evaluation_time is < 1 min for 95% of rows and < 10 min for all.
    evaluation_timeout: int | None = 1800
    # A verdict-less run is retried on a fresh sandbox: an image pull or a flaky provider start
    # is not evidence about the patch.
    inconclusive_verification_retries: int = 1
    apply_anti_cheating: bool = True
    sandbox_provider: str
    sandbox_config: dict[str, Any]


class SweInternalV1InstanceRequest(BaseModel):
    """One row of ``data/swe_internal_v1_training.jsonl``, an internal-v1 SWE task."""

    model_config = ConfigDict(extra="allow")

    instance_id: str
    delivery: str = ""
    language: str = ""
    repo: str = ""
    workdir: str = "/app"
    image_ref: str
    base_commit: str = ""
    solution_commit: str = ""
    patch: str = ""
    test_patch: str = ""
    problem_statement: str = ""
    run_script: str
    parsing_script: str
    test_files: list[str] = []
    test_patch_checkout_cmd: str = ""
    env_exports: list[str] = []
    test_framework: str = ""
    FAIL_TO_PASS: list[str] = []
    PASS_TO_PASS: list[str] = []


class SweInternalV1SeedSessionRequest(SweInternalV1InstanceRequest, BaseSeedSessionRequest):
    sandbox_spec: dict[str, Any] | None = None


class SweInternalV1SeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str
    workdir: str


class SweInternalV1VerifyRequest(SweInternalV1InstanceRequest, BaseVerifyRequest):
    pass


class SweInternalV1VerifyResponse(BaseVerifyResponse):
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


class SweInternalV1ResourcesServer(SimpleResourcesServer):
    config: SweInternalV1ResourcesServerConfig
    ray_enabled = False

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_id_to_sandbox: dict[str, AsyncSandbox] = {}
        self._session_id_to_pristine_untracked: dict[str, frozenset[str]] = {}
        self._session_id_to_base_commit: dict[str, str] = {}

    def _inputs(self, body: SweInternalV1InstanceRequest, patch: str) -> VerificationInputs:
        return VerificationInputs(
            instance_id=body.instance_id,
            workdir=body.workdir,
            base_commit=body.base_commit,
            patch=drop_test_patch_files(patch, body.test_patch),
            run_script=body.run_script,
            parsing_script=body.parsing_script,
            test_files=list(body.test_files),
            test_patch=body.test_patch,
            test_patch_checkout_cmd=body.test_patch_checkout_cmd,
            env_exports=list(body.env_exports),
            fail_to_pass=list(body.FAIL_TO_PASS),
            pass_to_pass=list(body.PASS_TO_PASS),
        )

    async def _create_sandbox(
        self, body: SweInternalV1InstanceRequest, files: dict[str, str] | None = None
    ) -> AsyncSandbox:
        global_config_dict = get_global_config_dict()
        provider_config = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        provider_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)

        # jest/vitest/pytest-xdist/cmake spawn worker pools sized off the visible core count, which
        # is the HOST count, not the cgroup quota.
        sandbox_resources = SandboxResources.from_mapping(self.config.sandbox_config.get("resources", {}))
        env = dict(self.config.sandbox_config.get("env", {}))
        if self.config.sandbox_config.get("derive_cpu_env", True):
            env = cpu_cap_env(sandbox_resources.cpu) | env

        spec = SandboxSpec(
            image=body.image_ref,
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
            print("Failed to stop swe_internal_v1 sandbox", format_exc(), file=sys.stderr)

    async def _ensure_git_repo(self, sandbox: AsyncSandbox, workdir: str) -> None:
        """Init a git repo only when the image has none, so ``seed_session`` always has a base
        commit to diff the agent's changes against."""
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

    async def _restore_missing_blobs(self, sandbox: AsyncSandbox, workdir: str) -> None:
        """Same guard as swemer_oml: an image whose ``.git`` had its blobs stripped loses every
        tracked file the anti-cheat ``git reset --hard`` cannot re-read. Rewrite blobs from the
        working tree first; a no-op (one probe) when HEAD's blobs are readable."""
        wd = shlex.quote(workdir)
        probe = await sandbox.exec(
            f'cd {wd} && for f in $(git ls-files | head -3); do git cat-file -e "HEAD:$f" || exit 3; done',
            timeout_s=120,
        )
        if probe.return_code == 0:
            return
        print(
            f"[swe_internal_v1] {workdir}: HEAD blobs missing (blob-stripped image); restoring from the working tree",
            flush=True,
        )
        result = await sandbox.exec(
            f"cd {wd} && git ls-files -z --stage | while IFS= read -r -d '' e; do "
            "m=${e%% *}; p=${e#*$'\\t'}; case $m in 100644|100755) printf '%s\\0' \"$p\";; esac; done "
            "| xargs -0 -r -n 500 git hash-object -w -- || true; git add -u && "
            "(git -c user.email=nemo-gym@nvidia.com -c user.name=nemo-gym commit -q -m "
            "'nemo_gym: restore blobs stripped from the image' || true)",
            timeout_s=900,
        )
        if result.return_code != 0:
            print(
                f"[swe_internal_v1] blob restore failed ({result.return_code}): "
                f"{(result.stderr or '')[-400:]} {(result.stdout or '')[-400:]}",
                file=sys.stderr,
            )

    async def _pristine_untracked_files(self, sandbox: AsyncSandbox, workdir: str) -> frozenset[str]:
        """Files ``workdir`` holds untracked before the agent touches it."""
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

    async def seed_session(
        self, request: Request, body: SweInternalV1SeedSessionRequest
    ) -> SweInternalV1SeedSessionResponse:
        """Start the instance's image so an agent can work in it."""
        session_id = request.session[SESSION_ID_KEY]
        await self._stop_sandbox(self._session_id_to_sandbox.pop(session_id, None))
        self._session_id_to_pristine_untracked.pop(session_id, None)
        self._session_id_to_base_commit.pop(session_id, None)

        sandbox = await self._create_sandbox(body)
        await self._ensure_git_repo(sandbox, body.workdir)
        await self._restore_missing_blobs(sandbox, body.workdir)
        if self.config.apply_anti_cheating:
            await apply_anti_cheat_setup(sandbox, body.workdir, body.instance_id, "swe_internal_v1")
        # The anti-cheat scrub leaves no committer identity, so the agent's `git commit` would fail.
        await prepare_git_for_commits(sandbox, body.workdir, "swe_internal_v1")

        head_result = await sandbox.exec(f"git -C {shlex.quote(body.workdir)} rev-parse HEAD")
        self._session_id_to_base_commit[session_id] = (head_result.stdout or "").strip()
        self._session_id_to_pristine_untracked[session_id] = await self._pristine_untracked_files(
            sandbox, body.workdir
        )
        self._session_id_to_sandbox[session_id] = sandbox
        return SweInternalV1SeedSessionResponse(sandbox_handle=str(sandbox._handle.sandbox_id), workdir=body.workdir)

    async def verify(self, request: Request, body: SweInternalV1VerifyRequest) -> SweInternalV1VerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        extraction_error = None
        mode = self.config.patch_capture_mode
        if self.config.is_verifying_golden_patch:
            capture = PatchCapture.static(body.patch, mode, "golden")
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
                await self._stop_sandbox(sandbox)
            if result.completed:
                break
            if attempt < attempts:
                print(
                    f"[swe_internal_v1] {body.instance_id}: inconclusive ({result.error}); "
                    f"retrying on a fresh sandbox ({attempt}/{attempts - 1})",
                    flush=True,
                )

        return SweInternalV1VerifyResponse.model_validate(
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
    SweInternalV1ResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # Required whenever num_workers > 1: multi-worker uvicorn re-imports this entrypoint by path
    # in each forked child and expects a module-level `app`.
    app = SweInternalV1ResourcesServer.run_webserver()  # noqa: F401
