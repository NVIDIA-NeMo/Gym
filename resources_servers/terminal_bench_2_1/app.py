# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, ClassVar, Dict, Optional

from fastapi import Request

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY


class TerminalBench21ResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    is_verifying_golden_patch: bool = False
    evaluation_timeout: Optional[int] = None

    # Sandbox config
    sandbox_provider: str
    sandbox_config: Dict[str, Any]

    clear_terminal_bench_debug_logs: bool = True

    def model_post_init(self, context: Any, /) -> None:
        if self.is_verifying_golden_patch and self.clear_terminal_bench_debug_logs:
            print("Turning off logs clear since `is_verifying_golden_patch=true`")
            self.clear_terminal_bench_debug_logs = False


class TerminalBench21VerifyRequest(BaseVerifyRequest):
    task_name: str
    docker_image: str
    task_folder: str


class TerminalBench21SeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str  # @bxyu-nvidia: Just a plain string URI for now for OpenSandbox backend.


class TerminalBench21ResourcesServer(SimpleResourcesServer):
    config: TerminalBench21ResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)

        self._session_id_to_sandbox: Dict[str, AsyncSandbox] = dict()

    async def _create_sandbox(self, verify_request: TerminalBench21VerifyRequest) -> AsyncSandbox:
        # TODO @bxyu-nvidia: Refactor this after Hemil's swap from Python dataclass to Pydantic BaseModel
        global_config_dict = get_global_config_dict()
        resolved_sandbox_provider = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        provider_default_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)
        resources = dict(self.config.sandbox_config.get("resources", {}))

        eval_sandbox_spec = SandboxSpec(
            image=verify_request.docker_image,
            ttl_s=self.config.sandbox_config.get("ttl_s", None),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s", None),
            workdir=None,  # Default to container's WORKDIR
            env=self.config.sandbox_config.get("env", {}),
            files=dict(),
            metadata=provider_default_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {
                "nemo_gym_agent": self.config.name,
                "instance_id": verify_request.task_name,
            },
            resources=SandboxResources.from_mapping(resources),
            entrypoint=None,
            provider_options=self.config.sandbox_config.get("provider_options", {}),
        )
        eval_sandbox = AsyncSandbox(resolved_sandbox_provider)
        await eval_sandbox.start(eval_sandbox_spec)

        return eval_sandbox

    async def seed_session(self, request: Request, body: TerminalBench21VerifyRequest) -> BaseSeedSessionResponse:
        eval_sandbox = await self._create_sandbox(body)
        self._session_id_to_sandbox[request.session[SESSION_ID_KEY]] = eval_sandbox

        return TerminalBench21SeedSessionResponse(sandbox_handle=eval_sandbox._handle.sandbox_id)

    async def verify(self, body: TerminalBench21VerifyRequest) -> BaseVerifyResponse:
        reward = float(body.response.output_text.strip() == body.expected_answer.strip())
        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    TerminalBench21ResourcesServer.run_webserver()
