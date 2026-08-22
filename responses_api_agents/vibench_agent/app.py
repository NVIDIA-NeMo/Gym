# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""ViBench agent: owns the build sandbox and copies the finished app out.

ViBench is "case 2" in NVIDIA-NeMo/Gym#2082 -- the rollout produces an artifact and the
verifier grades it in a fresh box -- so the sandbox never has to be shared. That matters
practically: only OpenSandbox implements ``serialize()``/``connect()``, so a design where
the resources server creates the box and the agent attaches to it cannot run on Docker,
Apptainer or enroot at all.

Flow, mirroring ``responses_api_agents/cvdp_agent``:

    POST /seed_session          -> PRD text + asset dirs (no sandbox handle)
    create sandbox              -> ViBench's app-bench-base image, WORKDIR /app
    stage PRD + assets          -> via SandboxSpec.files, before the harness starts
    run the OpenCode harness    -> inherited wholesale from OpenCodeSandboxedAgent
    harvest /app                -> tarball written into the shared artifact_dir
    POST /verify                -> resources server unpacks and grades it

Only the sandbox acquisition and the harvest differ from ``opencode_sandboxed_agent``;
everything about installing and driving OpenCode is inherited.
"""

import sys
from pathlib import Path
from shlex import quote
from traceback import format_exc
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import Request

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec, create_provider
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status
from responses_api_agents.opencode_sandboxed_agent.app import (
    OpenCodeSandboxedAgent,
    OpenCodeSandboxedAgentConfig,
    OpenCodeSandboxedAgentVerifyRequest,
    OpenCodeSandboxedAgentVerifyResponse,
)


# ViBench's coding agent reads its brief from this path; the seeding and evaluation agents
# are handed the same PRD text separately at grade time.
PRD_FILENAME = "prd.txt"


class VibenchAgentConfig(OpenCodeSandboxedAgentConfig):
    # ViBench's base image. Its WORKDIR is /app, which is where the harness lands.
    build_image: str
    app_workdir: str = "/app"

    # Shared with the resources server; built-app tarballs are written here.
    artifact_dir: str

    # Ceiling on taring and downloading the finished app.
    harvest_timeout_s: int = 900


class VibenchAgent(OpenCodeSandboxedAgent):
    config: VibenchAgentConfig

    async def _create_build_sandbox(self, prd_text: str, asset_paths: list[str]) -> AsyncSandbox:
        """Start a fresh build box with the PRD already staged.

        ``SandboxSpec.files`` writes the PRD before anything runs, so the harness sees it on
        its first `ls` and there is no upload race.
        """
        global_config_dict = get_global_config_dict()
        provider = create_provider(resolve_provider_config(self.config.sandbox_provider, global_config_dict))
        provider_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)

        spec = SandboxSpec(
            image=self.config.build_image,
            ttl_s=self.config.sandbox_config.get("ttl_s", None),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s", None),
            workdir=self.config.app_workdir,
            env=self.config.sandbox_config.get("env", {}),
            files={f"{self.config.app_workdir}/{PRD_FILENAME}": prd_text},
            metadata=provider_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {"nemo_gym_agent": self.config.name},
            resources=SandboxResources.from_mapping(dict(self.config.sandbox_config.get("resources", {}))),
            entrypoint=None,
            provider_options=self.config.sandbox_config.get("provider_options", {}),
        )
        sandbox = AsyncSandbox(provider, spec)
        await sandbox.start()

        for asset_dir in asset_paths:
            src = Path(asset_dir)
            if not src.is_dir():
                continue
            for item in sorted(src.rglob("*")):
                if item.is_file():
                    remote = f"{self.config.app_workdir}/assets/{item.relative_to(src).as_posix()}"
                    await sandbox.upload(item, remote)

        return sandbox

    async def _harvest_app(self, sandbox: AsyncSandbox, session_id: str) -> Optional[str]:
        """Tar the built app out of the sandbox into ``artifact_dir``.

        node_modules is excluded because it is huge and platform-specific -- the grading
        stack reinstalls dependencies anyway -- and prd.txt because the grader supplies its
        own copy. Returns None when nothing could be harvested, which the resources server
        scores as a build failure.
        """
        artifact_root = Path(self.config.artifact_dir).expanduser()
        artifact_root.mkdir(parents=True, exist_ok=True)
        local = artifact_root / f"vibench-app-{session_id}-{uuid4().hex[:8]}.tar"
        remote = "/tmp/vibench-app.tar"

        try:
            result = await sandbox.exec(
                f"cd {quote(self.config.app_workdir)} && tar "
                f"--exclude=./node_modules --exclude=./.git --exclude=./{PRD_FILENAME} "
                f"-cf {quote(remote)} .",
                timeout_s=self.config.harvest_timeout_s,
            )
            if result.return_code != 0:
                print(f"Failed to tar app dir: {result.stderr or result.stdout}", file=sys.stderr)
                return None
            await sandbox.download(remote, local)
        except Exception:
            print("Failed to harvest app from sandbox", format_exc(), file=sys.stderr)
            return None

        return str(local)

    async def run(self, request: Request, body: BaseRunRequest) -> OpenCodeSandboxedAgentVerifyResponse:
        cookies = request.cookies

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = cookies | seed_session_response.cookies
        seed_session_result = await seed_session_response.json()

        session_id = request.session[SESSION_ID_KEY]
        sandbox = await self._create_build_sandbox(
            prd_text=seed_session_result["prd_text"],
            asset_paths=seed_session_result.get("asset_paths", []),
        )
        self._sandbox_id_to_sandbox[session_id] = sandbox
        cookies["sandbox_id"] = session_id
        request._cookies = cookies

        try:
            response = await self.responses(request, body.responses_create_params)
            artifact_path = await self._harvest_app(sandbox, session_id)
        finally:
            # Harvest first, then release the box: the tarball is the only thing that
            # survives, and grading happens in a fresh stack.
            try:
                await sandbox.stop()
            except Exception:
                print("Failed to stop build sandbox", format_exc(), file=sys.stderr)
            self._sandbox_id_to_sandbox.pop(session_id, None)

        # OpenCodeSandboxedAgentVerifyRequest is extra="allow", so artifact_path rides along
        # to the resources server without a ViBench-specific request type.
        verify_request = OpenCodeSandboxedAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": response, "artifact_path": artifact_path}
        )
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)

        response_dict: Dict[str, Any] = await get_response_json(verify_response)
        response_dict |= self._sandbox_id_to_run_result.pop(session_id, {})
        return OpenCodeSandboxedAgentVerifyResponse.model_validate(response_dict)


if __name__ == "__main__":
    VibenchAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VibenchAgent.run_webserver()  # noqa: F401
