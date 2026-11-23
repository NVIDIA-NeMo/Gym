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

"""SWE-bench Pro v2 resources server: v1's server with the v2 image and verifier."""

from dataclasses import asdict
from ipaddress import ip_address
from pathlib import Path
from typing import ClassVar

from fastapi import Request
from pydantic import Field

from nemo_gym.config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.server_runtime import ResourcesSeedSessionRequest, ResourcesSeedSessionResponse
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.swebench_pro.app import (
    PatchCaptureLostError,
    SWEBenchProInstanceRequest,
    SWEBenchProResourcesServer,
    SWEBenchProSeedSessionRequest,
    SWEBenchProSeedSessionResponse,
    SWEBenchProVerifyRequest,
    SWEBenchProVerifyResponse,
)
from resources_servers.swebench_pro.verification import VerificationInputs, VerificationResult
from resources_servers.swebench_pro_v2.verification import (
    VerificationInputsV2,
    assemble_workspace_files,
    run_verification,
)


class SWEBenchProV2InstanceRequest(SWEBenchProInstanceRequest):
    docker_image: str
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    test_sh: str
    test_config: str


class SWEBenchProV2SeedSessionRequest(SWEBenchProV2InstanceRequest, SWEBenchProSeedSessionRequest):
    pass


class SWEBenchProV2VerifyRequest(SWEBenchProV2InstanceRequest, SWEBenchProVerifyRequest):
    pass


class SWEBenchProV2ResourcesServer(SWEBenchProResourcesServer):
    instance_request_cls: ClassVar[type[SWEBenchProInstanceRequest]] = SWEBenchProV2InstanceRequest
    workspace_assembler = staticmethod(assemble_workspace_files)

    async def seed_session(
        self,
        request: Request,
        body: SWEBenchProV2SeedSessionRequest | ResourcesSeedSessionRequest,
    ) -> SWEBenchProSeedSessionResponse | ResourcesSeedSessionResponse:
        return await super().seed_session(request, body)

    async def verify(self, request: Request, body: SWEBenchProV2VerifyRequest) -> SWEBenchProVerifyResponse:
        return await super().verify(request, body)

    def _sandbox_workdir(self) -> str:
        return "/"

    async def _repository_dir(self, sandbox: AsyncSandbox) -> str:
        try:
            result = await sandbox.exec(
                "git -C /app rev-parse --show-toplevel 2>/dev/null || "
                "git -C /testbed rev-parse --show-toplevel 2>/dev/null"
            )
        except Exception as exc:
            raise PatchCaptureLostError(f"Repository lookup failed: {exc}") from exc
        if result.error_type is not None:
            raise PatchCaptureLostError(f"Repository lookup failed: {result.error_type}")
        directory = (result.stdout or "").strip()
        if result.return_code != 0 or directory not in {"/app", "/testbed"}:
            raise RuntimeError("No task git repository at /app or /testbed")
        return directory

    async def _create_sandbox(
        self,
        body: SWEBenchProInstanceRequest,
        files: dict[str, str] | None = None,
        *,
        task: bool = False,
    ) -> AsyncSandbox:
        if task and self.config.task_network_policy is not None:
            host = (get_global_config_dict().get("head_server") or {}).get("host", "127.0.0.1")
            try:
                loopback = ip_address(host).is_loopback
            except ValueError:
                loopback = host.lower().rstrip(".") == "localhost"
            if not loopback:
                raise ValueError(
                    "SWE-bench Pro V2's network lock requires the head server on loopback; "
                    "set absolute_ip_server_types: [responses_api_models]"
                )
        return await super()._create_sandbox(body, files=files, task=task)

    def _image(self, body: SWEBenchProInstanceRequest) -> str:
        return f"{body.docker_image.rsplit(':', 1)[0]}@{body.image_digest}"

    def _verification_inputs(self, body: SWEBenchProInstanceRequest, patch: str) -> VerificationInputsV2:
        return VerificationInputsV2(
            **asdict(super()._verification_inputs(body, patch)),
            test_sh=body.test_sh,
            test_config=body.test_config,
            test_patch=body.test_patch,
            evaluation_timeout=self.config.evaluation_timeout,
        )

    async def _run_verification(
        self,
        sandbox: AsyncSandbox,
        inputs: VerificationInputs,
        log_dir: Path,
        timeout_s: int | None,
    ) -> VerificationResult:
        if not isinstance(inputs, VerificationInputsV2):
            raise TypeError("SWE-bench Pro V2 requires its pinned per-task verifier files")
        return await run_verification(sandbox=sandbox, inputs=inputs, log_dir=log_dir, timeout_s=timeout_s)

    def _inconclusive_reason(self, result: VerificationResult) -> str | None:
        return None if result.completed else result.error or "V2 verification produced no verdict"


if __name__ == "__main__":
    SWEBenchProV2ResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SWEBenchProV2ResourcesServer.run_webserver()
