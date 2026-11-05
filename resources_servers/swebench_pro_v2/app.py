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

from fastapi import Request

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.server_runtime import ResourcesSeedSessionRequest, ResourcesSeedSessionResponse
from resources_servers.swebench_pro.app import (
    SWEBenchProInstanceRequest,
    SWEBenchProResourcesServer,
    SWEBenchProSeedSessionRequest,
    SWEBenchProSeedSessionResponse,
    SWEBenchProVerifyRequest,
    SWEBenchProVerifyResponse,
)
from resources_servers.swebench_pro_v2.verification import VerificationInputsV2, assemble_workspace_files


class SWEBenchProV2InstanceRequest(SWEBenchProInstanceRequest):
    docker_image: str
    test_sh: str
    test_config: str


class SWEBenchProV2SeedSessionRequest(SWEBenchProV2InstanceRequest, SWEBenchProSeedSessionRequest):
    pass


class SWEBenchProV2VerifyRequest(SWEBenchProV2InstanceRequest, SWEBenchProVerifyRequest):
    pass


class SWEBenchProV2ResourcesServer(SWEBenchProResourcesServer):
    instance_request_cls = SWEBenchProV2InstanceRequest
    workspace_assembler = staticmethod(assemble_workspace_files)

    async def seed_session(
        self,
        request: Request,
        body: SWEBenchProV2SeedSessionRequest | ResourcesSeedSessionRequest,
    ) -> SWEBenchProSeedSessionResponse | ResourcesSeedSessionResponse:
        return await super().seed_session(request, body)

    async def verify(self, request: Request, body: SWEBenchProV2VerifyRequest) -> SWEBenchProVerifyResponse:
        return await super().verify(request, body)

    async def _repository_dir(self, sandbox: AsyncSandbox) -> str:
        # Some v2 images keep the repository in /testbed; /app is only used when it is the repository.
        result = await sandbox.exec("[ ! -e /app/.git ] && [ -e /testbed/.git ] && echo /testbed || echo /app")
        return (result.stdout or "").strip() or "/app"

    def _image(self, body: SWEBenchProInstanceRequest) -> str:
        # v2 images live on GHCR and are named by the dataset's ``docker_image`` column.
        return body.docker_image

    def _verification_inputs(self, body: SWEBenchProInstanceRequest, patch: str) -> VerificationInputsV2:
        return VerificationInputsV2(
            **asdict(super()._verification_inputs(body, patch)),
            test_sh=body.test_sh,
            test_config=body.test_config,
            test_patch=body.test_patch,
        )


if __name__ == "__main__":
    SWEBenchProV2ResourcesServer.run_webserver()
