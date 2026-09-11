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
"""

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
from resources_servers.scale_swe.verification import (
    VerificationInputs,
    VerificationResult,
    as_id_list,
    run_verification,
    verification_files,
)


class ScaleSWEResourcesServerConfig(BaseResourcesServerConfig):
    is_verifying_golden_patch: bool = False
    evaluation_timeout: int | None = 1200
    # A verdict-less run is retried on a fresh sandbox: an image pull or a flaky provider start
    # is not evidence about the patch.
    inconclusive_verification_retries: int = 1
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


class ScaleSWEResourcesServer(SimpleResourcesServer):
    config: ScaleSWEResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_id_to_sandbox: dict[str, AsyncSandbox] = {}

    def _inputs(self, body: ScaleSWEInstanceRequest, patch: str) -> VerificationInputs:
        return VerificationInputs(
            instance_id=body.instance_id,
            workdir=body.workdir,
            patch=patch,
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

    async def seed_session(self, request: Request, body: ScaleSWESeedSessionRequest) -> ScaleSWESeedSessionResponse:
        """Start the instance's image so an agent can work in it."""
        session_id = request.session[SESSION_ID_KEY]
        await self._stop_sandbox(self._session_id_to_sandbox.pop(session_id, None))
        sandbox = await self._create_sandbox(body)
        self._session_id_to_sandbox[session_id] = sandbox
        return ScaleSWESeedSessionResponse(sandbox_handle=str(sandbox.sandbox_id))

    async def verify(self, request: Request, body: ScaleSWEVerifyRequest) -> ScaleSWEVerifyResponse:
        if self.config.is_verifying_golden_patch:
            patch = body.patch
        else:
            patch = str(getattr(body, "model_patch", "") or "")

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
                    f"[scale_swe] {body.instance_id}: inconclusive ({result.error}); "
                    f"retrying on a fresh sandbox ({attempt}/{attempts - 1})",
                    flush=True,
                )

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
                "error": result.error,
                "eval_sandbox_start_time_taken": start_time_taken,
                "patch_verification_time_taken": verification_time_taken,
            }
        )


if __name__ == "__main__":
    ScaleSWEResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # Required whenever num_workers > 1. Multi-worker uvicorn re-imports this entrypoint BY PATH
    # in each forked child and expects a module-level `app`; without it every child exits, and
    # uvicorn responds by stopping the parent. The server then never binds, so the symptom is not
    # an import error but a flood of connection failures from clients talking to a dead port.
    app = ScaleSWEResourcesServer.run_webserver()  # noqa: F401
