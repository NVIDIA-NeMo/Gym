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
"""Resources server for AweAI-Team/DeNovoSWE.

A "document-to-repository" benchmark: each image ships the original package source at
``base_commit`` plus a ``document`` (a written spec/README). The agent must regenerate the
package from scratch given only the document -- so unlike every other SWE resources server here,
``seed_session`` wipes the pre-existing source (``_denovoswe_clean.sh``) and re-injects the
document as ``README.md`` BEFORE the agent starts, and there is no golden ``patch`` to apply for
golden-patch validation: the image's pre-existing source IS the golden answer.

Verification: wipe again (fresh sandbox, non-golden only) + re-inject the document to match the
baseline the agent's patch was diffed against, apply the candidate patch, lay the canonical test
suite from ``test_patch``, reinstall the package, and grade via
``responses_api_agents/swe_agents/_denovoswe_eval.py`` (a verbatim local copy of the
already-validated per-file pytest evaluator -- see ``verification.py``).

Set ``is_verifying_golden_patch: true`` to grade the image's own pre-existing source instead of an
agent's, the dataset-health check: a row whose golden source does not resolve is a broken row.
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
from resources_servers.denovo_swe.verification import (
    DOCUMENT_PATH,
    VerificationInputs,
    VerificationResult,
    drop_patch_sections,
    run_verification,
    seed_prep_script,
    verification_files,
)


class DeNovoSWEResourcesServerConfig(BaseResourcesServerConfig):
    is_verifying_golden_patch: bool = False
    evaluation_timeout: int | None = 1800
    # A verdict-less run is retried on a fresh sandbox: an image pull or a flaky provider start
    # is not evidence about the patch.
    inconclusive_verification_retries: int = 1
    sandbox_provider: str
    sandbox_config: dict[str, Any]


class DeNovoSWEInstanceRequest(BaseModel):
    """One row of AweAI-Team/DeNovoSWE."""

    model_config = ConfigDict(extra="allow")

    instance_id: str
    repo: str = ""
    github_url: str = ""
    language: str = "python"
    workdir: str
    image_ref: str
    base_commit: str
    patch: str = ""
    test_patch: str = ""
    document: str = ""
    problem_statement: str = ""
    pypi_name: str = ""
    import_names: list[str] = []
    passed_ptp: list[str] = []
    failed_ptp: list[str] = []
    test_binary_archive_b64: str = ""
    expected_coverage_percent: float = 0.0


class DeNovoSWESeedSessionRequest(DeNovoSWEInstanceRequest, BaseSeedSessionRequest):
    sandbox_spec: dict[str, Any] | None = None


class DeNovoSWESeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str


class DeNovoSWEVerifyRequest(DeNovoSWEInstanceRequest, BaseVerifyRequest):
    pass


class DeNovoSWEVerifyResponse(BaseVerifyResponse):
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


class DeNovoSWEResourcesServer(SimpleResourcesServer):
    config: DeNovoSWEResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_id_to_sandbox: dict[str, AsyncSandbox] = {}
        self._session_id_to_pristine_untracked: dict[str, frozenset[str]] = {}
        self._session_id_to_base_commit: dict[str, str] = {}

    def _inputs(self, body: DeNovoSWEInstanceRequest, patch: str) -> VerificationInputs:
        return VerificationInputs(
            instance_id=body.instance_id,
            workdir=body.workdir,
            base_commit=body.base_commit,
            patch=patch,
            test_patch=body.test_patch,
            document=body.document or body.problem_statement,
            pypi_name=body.pypi_name,
            passed_ptp=list(body.passed_ptp),
            failed_ptp=list(body.failed_ptp),
            test_binary_archive_b64=body.test_binary_archive_b64,
            expected_coverage_percent=body.expected_coverage_percent,
        )

    async def _create_sandbox(
        self, body: DeNovoSWEInstanceRequest, files: dict[str, str] | None = None
    ) -> AsyncSandbox:
        global_config_dict = get_global_config_dict()
        provider_config = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        provider_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)

        sandbox_resources = SandboxResources.from_mapping(self.config.sandbox_config.get("resources", {}))
        env = dict(self.config.sandbox_config.get("env", {}))
        if self.config.sandbox_config.get("derive_cpu_env", True):
            env = cpu_cap_env(sandbox_resources.cpu) | env

        spec = SandboxSpec(
            # The row names its own image; there is no repository template to apply.
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
            print("Failed to stop DeNovoSWE sandbox", format_exc(), file=sys.stderr)

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

    async def _extract_model_patch(self, session_id: str, workdir: str, base_commit: str) -> str:
        """Diff the agent's own sandbox against the post-wipe/README-amend commit, then stop it.

        ``git add -N`` (intent-to-add) is what makes brand-new files show up in ``git diff`` too,
        not just edits to already-tracked files.
        """
        original_sandbox = self._session_id_to_sandbox.pop(session_id)
        pristine_untracked = self._session_id_to_pristine_untracked.pop(session_id, frozenset())
        try:
            result = await original_sandbox.exec(
                f"git -C {shlex.quote(workdir)} add -N . "
                f"&& git -C {shlex.quote(workdir)} --no-pager diff {shlex.quote(base_commit)}"
            )
            if result.return_code != 0:
                raise RuntimeError(result.stderr or "git diff failed")
            return drop_patch_sections(result.stdout or "", pristine_untracked)
        finally:
            await self._stop_sandbox(original_sandbox)

    async def seed_session(self, request: Request, body: DeNovoSWESeedSessionRequest) -> DeNovoSWESeedSessionResponse:
        """Start the instance's image, then wipe the pre-existing source and re-inject the spec
        as README.md -- without this the agent could just read the implementation it is supposed
        to regenerate (see module docstring).
        """
        session_id = request.session[SESSION_ID_KEY]
        await self._stop_sandbox(self._session_id_to_sandbox.pop(session_id, None))
        self._session_id_to_pristine_untracked.pop(session_id, None)
        self._session_id_to_base_commit.pop(session_id, None)

        document = body.document or body.problem_statement
        sandbox = await self._create_sandbox(
            body, files={"/tmp/nemo_gym_seed_prep.sh": seed_prep_script(body.workdir), DOCUMENT_PATH: document}
        )
        prep_result = await sandbox.exec("bash /tmp/nemo_gym_seed_prep.sh")
        if prep_result.return_code != 0:
            print(
                f"[denovo_swe] {body.instance_id}: seed prep (wipe+README) failed during seed_session "
                f"(non-fatal, matching build_eval_script's own tolerance): {prep_result.stderr}",
                file=sys.stderr,
            )

        head_result = await sandbox.exec(f"git -C {shlex.quote(body.workdir)} rev-parse HEAD")
        self._session_id_to_base_commit[session_id] = (head_result.stdout or "").strip()
        self._session_id_to_pristine_untracked[session_id] = await self._pristine_untracked_files(
            sandbox, body.workdir
        )
        self._session_id_to_sandbox[session_id] = sandbox
        return DeNovoSWESeedSessionResponse(sandbox_handle=str(sandbox._handle.sandbox_id))

    async def verify(self, request: Request, body: DeNovoSWEVerifyRequest) -> DeNovoSWEVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        is_golden = self.config.is_verifying_golden_patch
        extraction_error = None
        if is_golden:
            patch = body.patch
        else:
            base_commit = self._session_id_to_base_commit.pop(session_id, "")
            if not base_commit:
                patch = ""
                extraction_error = "Failed to extract model patch: no base commit recorded (seed_session did not run for this session)"
            else:
                try:
                    patch = await self._extract_model_patch(session_id, body.workdir, base_commit)
                except Exception as exc:
                    patch = ""
                    extraction_error = f"Failed to extract model patch: {exc}"

        inputs = self._inputs(body, patch)
        log_dir = Path(__file__).parent / "logs" / body.instance_id

        files = verification_files(inputs, is_golden)
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
                    is_golden=is_golden,
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
                    f"[denovo_swe] {body.instance_id}: inconclusive ({result.error}); "
                    f"retrying on a fresh sandbox ({attempt}/{attempts - 1})",
                    flush=True,
                )

        return DeNovoSWEVerifyResponse.model_validate(
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
                "error": extraction_error or result.error,
                "eval_sandbox_start_time_taken": start_time_taken,
                "patch_verification_time_taken": verification_time_taken,
            }
        )


if __name__ == "__main__":
    DeNovoSWEResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # Required whenever num_workers > 1. Multi-worker uvicorn re-imports this entrypoint BY PATH
    # in each forked child and expects a module-level `app`; without it every child exits, and
    # uvicorn responds by stopping the parent. The server then never binds, so the symptom is not
    # an import error but a flood of connection failures from clients talking to a dead port.
    app = DeNovoSWEResourcesServer.run_webserver()  # noqa: F401
