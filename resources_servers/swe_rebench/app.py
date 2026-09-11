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
"""Resources server for nebius/SWE-rebench-V2.

Each dataset row ships its own prebuilt image plus the commands to install and test it, so this
server does not build environments — it starts the row's image in a sandbox, applies patches,
runs the row's test command, and grades the output with the row's named upstream parser.

Set ``is_verifying_golden_patch: true`` to grade the dataset's own patch instead of an agent's.
That is the dataset-health check: a row whose golden patch does not resolve is a broken row, and
scoring an agent against it is measuring noise.
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
from resources_servers.swe_rebench.log_parsers import default_cache_dir, resolve_parser
from resources_servers.swe_rebench.verification import (
    VerificationInputs,
    VerificationResult,
    as_command_list,
    repo_directory,
    run_verification,
    verification_files,
)


# Gradle only auto-loads init scripts from $GRADLE_USER_HOME/init.d, so the mirror script
# shipped by verification._mirror_files() is invisible unless this points at the same home.
# Maven needs no equivalent: it reads ~/.m2/settings.xml by default.
#
# An earlier attempt set -Djava.net.preferIPv6Addresses=false here on the theory that the JVM
# was misrouting over IPv6. It changed nothing -- the failures were identical -- so the cause is
# Maven Central being unreachable from this network rather than an IP-version problem. Removed
# rather than left in place, since an ineffective workaround invites the same dead end later.
JVM_MIRROR_ENV = {
    "GRADLE_USER_HOME": "/root/.gradle",
}


class SWERebenchResourcesServerConfig(BaseResourcesServerConfig):
    is_verifying_golden_patch: bool = False
    evaluation_timeout: int | None = 3600
    # A verdict-less run is retried on a fresh sandbox: an image pull or a flaky provider start
    # is not evidence about the patch.
    inconclusive_verification_retries: int = 1
    # Where the upstream log-parser repo is cloned. Empty uses the shared nemo_gym cache.
    log_parser_cache_dir: str = ""
    sandbox_provider: str
    sandbox_config: dict[str, Any]


class SWERebenchInstanceRequest(BaseModel):
    """One row of nebius/SWE-rebench-V2."""

    model_config = ConfigDict(extra="allow")

    instance_id: str
    repo: str
    base_commit: str
    patch: str = ""
    test_patch: str = ""
    problem_statement: str = ""
    language: str = ""
    image_name: str
    install_config: dict[str, Any] = {}
    # Upstream spells these in caps; keep the dataset's own names so a row round-trips unchanged.
    FAIL_TO_PASS: list[str] = []
    PASS_TO_PASS: list[str] = []


class SWERebenchSeedSessionRequest(SWERebenchInstanceRequest, BaseSeedSessionRequest):
    sandbox_spec: dict[str, Any] | None = None


class SWERebenchSeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str


class SWERebenchVerifyRequest(SWERebenchInstanceRequest, BaseVerifyRequest):
    pass


class SWERebenchVerifyResponse(BaseVerifyResponse):
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


class SWERebenchResourcesServer(SimpleResourcesServer):
    config: SWERebenchResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_id_to_sandbox: dict[str, AsyncSandbox] = {}

    @property
    def _parser_cache_dir(self) -> Path:
        return Path(self.config.log_parser_cache_dir) if self.config.log_parser_cache_dir else default_cache_dir()

    def _inputs(self, body: SWERebenchInstanceRequest, patch: str) -> VerificationInputs:
        install_config = body.install_config or {}
        return VerificationInputs(
            instance_id=body.instance_id,
            repo=body.repo,
            base_commit=body.base_commit,
            patch=patch,
            test_patch=body.test_patch,
            install=as_command_list(install_config.get("install")),
            test_cmd=as_command_list(install_config.get("test_cmd")),
            log_parser=str(install_config.get("log_parser") or ""),
            fail_to_pass=list(body.FAIL_TO_PASS or []),
            pass_to_pass=list(body.PASS_TO_PASS or []),
        )

    async def _create_sandbox(
        self, body: SWERebenchInstanceRequest, files: dict[str, str] | None = None
    ) -> AsyncSandbox:
        global_config_dict = get_global_config_dict()
        provider_config = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        provider_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)

        # Cap build/test parallelism to the CPU limit. A container sees the HOST core count (96
        # here), so `make -j$(nproc)`, cargo, gradle and `go test` fan out ~96 workers against a
        # 4-CPU quota and spend most of their time CFS-throttled. This matters more for this set
        # than any other: it is the compile-heaviest, with 6144 Go, 3123 Rust, 1716 Java, 411
        # Scala and 182 C++ rows. Explicit sandbox_config.env keys still win.
        sandbox_resources = SandboxResources.from_mapping(self.config.sandbox_config.get("resources", {}))
        env = dict(self.config.sandbox_config.get("env", {}))
        if self.config.sandbox_config.get("derive_cpu_env", True):
            env = cpu_cap_env(sandbox_resources.cpu) | env
        if self.config.sandbox_config.get("use_maven_mirror", True):
            # See verification._mirror_files(): Maven Central is unreachable from this network
            # while other registries are, so both build tools are pointed at the Google mirror.
            env = JVM_MIRROR_ENV | env

        spec = SandboxSpec(
            # The row names its own image; there is no repository template to apply.
            image=body.image_name,
            ttl_s=self.config.sandbox_config.get("ttl_s"),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s"),
            workdir=repo_directory(body.repo),
            env=env,
            files=files or {},
            metadata=provider_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {
                "nemo_gym_agent": self.config.name,
                "instance_id": body.instance_id[:63],
                "language": body.language[:63],
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
            print("Failed to stop SWE-rebench sandbox", format_exc(), file=sys.stderr)

    async def seed_session(
        self, request: Request, body: SWERebenchSeedSessionRequest
    ) -> SWERebenchSeedSessionResponse:
        """Start the instance's image so an agent can work in it."""
        session_id = request.session[SESSION_ID_KEY]
        await self._stop_sandbox(self._session_id_to_sandbox.pop(session_id, None))
        sandbox = await self._create_sandbox(body)
        self._session_id_to_sandbox[session_id] = sandbox
        return SWERebenchSeedSessionResponse(sandbox_handle=str(sandbox.sandbox_id))

    async def verify(self, request: Request, body: SWERebenchVerifyRequest) -> SWERebenchVerifyResponse:
        if self.config.is_verifying_golden_patch:
            patch = body.patch
        else:
            patch = str(getattr(body, "model_patch", "") or "")

        inputs = self._inputs(body, patch)
        log_dir = Path(__file__).parent / "logs" / body.instance_id

        # Resolve the parser before spending a sandbox on the run: an unknown parser name means
        # this row can never be graded, and finding that out after the tests have run wastes the
        # most expensive part.
        try:
            parser = resolve_parser(inputs.log_parser, self._parser_cache_dir)
        except Exception as exc:
            # Spread the request: BaseVerifyResponse extends BaseVerifyRequest, so
            # responses_create_params and response are required and must be echoed back.
            return SWERebenchVerifyResponse.model_validate(
                body.model_dump()
                | {
                    "reward": 0.0,
                    "evaluation_completed": False,
                    "resolved": False,
                    "patch_applied": False,
                    "instance_id": body.instance_id,
                    "language": body.language,
                    "test_results": None,
                    "test_output": "",
                    "error": f"{exc}",
                    "eval_sandbox_start_time_taken": 0.0,
                    "patch_verification_time_taken": 0.0,
                }
            )

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
                    parser=parser,
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
                    f"[swe_rebench] {body.instance_id}: inconclusive ({result.error}); "
                    f"retrying on a fresh sandbox ({attempt}/{attempts - 1})",
                    flush=True,
                )

        return SWERebenchVerifyResponse.model_validate(
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
    SWERebenchResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # Required whenever num_workers > 1. Multi-worker uvicorn re-imports this entrypoint BY PATH
    # in each forked child and expects a module-level `app`; without it every child exits, and
    # uvicorn responds by stopping the parent. The server then never binds, so the symptom is not
    # an import error but a flood of connection failures from clients talking to a dead port.
    app = SWERebenchResourcesServer.run_webserver()  # noqa: F401
