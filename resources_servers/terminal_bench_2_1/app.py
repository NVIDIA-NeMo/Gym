# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from glob import glob
from pathlib import Path
from sys import stderr
from tempfile import NamedTemporaryFile
from time import time
from traceback import format_exc
from typing import Any, ClassVar, Dict, Optional, Tuple

from fastapi import Request

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxPtySession, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY


class TerminalBench21ResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    is_verifying_golden_patch: bool = False
    evaluation_timeout: Optional[int] = None

    # Sandbox config
    sandbox_provider: str
    sandbox_config: Dict[str, Any]

    debug: bool = False


class TerminalBench21SeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str  # @bxyu-nvidia: Just a plain string URI for now for OpenSandbox backend.


class TerminalBench21VerifyRequest(BaseVerifyRequest):
    task_name: str
    docker_image: str
    task_folder: str


class TerminalBench21VerifyResponse(BaseVerifyResponse):
    evaluation_completed: bool

    # Misc metrics
    verification_time_taken: float

    task_name: str
    test_output: str
    golden_patch_output: Optional[str]


class TerminalBench21ResourcesServer(SimpleResourcesServer):
    config: TerminalBench21ResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)

        self._session_id_to_sandbox: Dict[str, Tuple[AsyncSandbox, SandboxPtySession]] = dict()

    async def _create_sandbox(
        self, verify_request: TerminalBench21VerifyRequest
    ) -> Tuple[AsyncSandbox, SandboxPtySession]:
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

        pty_session = await eval_sandbox.pty.create()

        return eval_sandbox, pty_session

    async def seed_session(self, request: Request, body: TerminalBench21VerifyRequest) -> BaseSeedSessionResponse:
        eval_sandbox, pty_session = await self._create_sandbox(body)
        self._session_id_to_sandbox[request.session[SESSION_ID_KEY]] = eval_sandbox, pty_session

        return TerminalBench21SeedSessionResponse(sandbox_handle=eval_sandbox._handle.sandbox_id)

    async def _upload_folder(self, sandbox: AsyncSandbox, local_dirpath: Path, target_dirpath: str) -> None:
        if not local_dirpath.is_absolute():
            local_dirpath = PARENT_DIR / local_dirpath

        for file in glob("**", root_dir=str(local_dirpath), recursive=True):
            local_fpath = local_dirpath / file
            if not local_fpath.is_file():
                continue

            target_fpath = f"{target_dirpath}/{file}"
            mkdir_result = await sandbox.exec(f"mkdir -p {Path(target_fpath).parent}")
            assert mkdir_result.return_code == 0, mkdir_result
            await sandbox.upload(local_path=local_fpath, remote_path=target_fpath)

    async def verify(self, request: Request, body: TerminalBench21VerifyRequest) -> TerminalBench21VerifyResponse:
        task_folder = Path(body.task_folder)

        if self.config.is_verifying_golden_patch:
            if self.config.debug:
                print(f"Creating eval sandbox for {body.task_name}", file=stderr)
            eval_sandbox, pty_session = await self._create_sandbox(body)
            cwd = (await eval_sandbox.exec("pwd")).stdout.strip()
            await self._upload_folder(eval_sandbox, task_folder / "solution", cwd)

            if self.config.debug:
                print(f"Running golden patch for {body.task_name}", file=stderr)
            golden_patch_result = await eval_sandbox.pty.exec(
                f"bash {cwd}/solve.sh", session=pty_session, timeout_s=self.config.evaluation_timeout
            )
            # assert golden_patch_result.return_code == 0, (
            #     f"Failed to apply golden patch for {body.task_name}: {golden_patch_result}"
            # )
            golden_patch_output = (golden_patch_result.stderr or "") + (golden_patch_result.stdout or "")
            if self.config.debug:
                print(f"Golden patch output for {body.task_name}: {golden_patch_output}", file=stderr)
        else:
            # Re-use the original sandbox
            eval_sandbox, pty_session = self._session_id_to_sandbox.pop(request.session[SESSION_ID_KEY])
            golden_patch_output = None
            raise NotImplementedError

        if self.config.debug:
            print(f"Running tests for {body.task_name}", file=stderr)
        start_time = time()
        await self._upload_folder(eval_sandbox, task_folder / "tests", "/tests")
        eval_result = await eval_sandbox.pty.exec(
            "bash /tests/test.sh", session=pty_session, timeout_s=self.config.evaluation_timeout, detach=True
        )
        verification_time_taken = time() - start_time
        test_output = (eval_result.stderr or "") + (eval_result.stdout or "")

        if self.config.debug:
            print(f"Test output for {body.task_name}: {test_output}", file=stderr)

        try:
            with NamedTemporaryFile(mode="w+", suffix=".txt") as temp_file:
                await eval_sandbox.download("/logs/verifier/reward.txt", temp_file.name)
                temp_file.seek(0)
                reward = float(temp_file.read())

            evaluation_completed = True
        except:
            if self.config.debug:
                print(f"Hit an exception downloading and converting reward: {format_exc()}", file=stderr)
            evaluation_completed = False
            reward = 0.0

        return TerminalBench21VerifyResponse(
            **body.model_dump(),
            evaluation_completed=evaluation_completed,
            reward=reward,
            verification_time_taken=verification_time_taken,
            test_output=test_output,
            golden_patch_output=golden_patch_output,
        )


if __name__ == "__main__":
    TerminalBench21ResourcesServer.run_webserver()
