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

"""SWE-bench Pro resources server."""

import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from shlex import quote
from time import time
from traceback import format_exc
from typing import Any

from fastapi import FastAPI, Request
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
from nemo_gym.sandbox.providers.base import SandboxPtySession
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.swebench_pro.verification import (
    DEFAULT_ENVIRONMENT_REPAIRS,
    VerificationInputs,
    VerificationResult,
    assemble_workspace_files,
    build_seed_normalization,
    drop_patch_sections,
    inconclusive_reason,
    run_verification,
)


# K8s maps localhost to ::1 and Node 17+ honours that, but servers under test bind IPv4.
SANDBOX_ENV_OVERRIDES = {"NODE_OPTIONS": "--dns-result-order=ipv4first"}


# Blanked on the spec so the agent sees a clean env; the entryscript unsets them for real.
HARNESS_ENV_TO_SCRUB = (
    "OTEL_SERVICE_NAME",
    "OTEL_SERVICE_VERSION",
    "OTEL_RESOURCE_ATTRIBUTES",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
    "OTEL_METRIC_EXPORT_INTERVAL",
    "OTEL_TRACES_EXPORTER",
    "OTEL_METRICS_EXPORTER",
    "OTEL_LOGS_EXPORTER",
)


class SWEBenchProResourcesServerConfig(BaseResourcesServerConfig):
    is_verifying_golden_patch: bool = False
    apply_anti_cheating: bool = True
    prefetch_go_modules: bool = False
    evaluation_timeout: int | None = None
    # A verdict-less run is retried on a new sandbox; see `inconclusive_reason`.
    inconclusive_verification_retries: int = 2
    # Which container repairs to apply; see `ENVIRONMENT_REPAIRS`.
    environment_repairs: tuple[str, ...] = DEFAULT_ENVIRONMENT_REPAIRS
    image_repository: str = "docker.io/jefzda/sweap-images"
    sandbox_provider: str
    sandbox_config: dict[str, Any]


class SWEBenchProInstanceRequest(BaseModel):
    """One row from ScaleAI/SWE-bench_Pro plus pinned evaluator assets."""

    model_config = ConfigDict(extra="allow")

    repo: str
    instance_id: str
    base_commit: str
    patch: str
    test_patch: str = ""
    problem_statement: str
    requirements: str = ""
    interface: str = ""
    repo_language: str = ""
    fail_to_pass: str | list[str]
    pass_to_pass: str | list[str]
    issue_specificity: str = ""
    issue_categories: str = ""
    before_repo_set_cmd: str = ""
    selected_test_files_to_run: str | list[str]
    dockerhub_tag: str
    image_digest: str = ""
    run_script: str
    parser_script: str
    base_dockerfile: str = ""
    instance_dockerfile: str = ""
    subset: str = "pro"
    split: str = "test"


class SWEBenchProSeedSessionRequest(SWEBenchProInstanceRequest, BaseSeedSessionRequest):
    sandbox_spec: dict[str, Any] | None = None


class SWEBenchProSeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str
    # The agent attaches to this session; without it, it builds its own sandbox instead.
    pty_session_id: str


class SWEBenchProVerifyRequest(SWEBenchProInstanceRequest, BaseVerifyRequest):
    pass


class SWEBenchProVerifyResponse(BaseVerifyResponse):
    evaluation_completed: bool
    resolved: bool
    patch_applied: bool
    eval_sandbox_start_time_taken: float
    patch_verification_time_taken: float
    instance_id: str
    model_patch: str | None
    test_results: dict[str, Any] | None
    test_output: str
    error: str | None
    log_dir: str


class SWEBenchProResourcesServer(SimpleResourcesServer):
    config: SWEBenchProResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_id_to_sandbox: dict[str, AsyncSandbox] = {}
        # The agent's terminal for the session. Leading underscore: pydantic needs it.
        self._session_id_to_pty: dict[str, SandboxPtySession] = {}
        # Untracked files the image ships, per session. Leading underscore: pydantic needs it.
        self._session_id_to_pristine_untracked: dict[str, frozenset[str]] = {}

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as maybe_state:
                    yield maybe_state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        return app

    async def close_pty_session(self, session: SandboxPtySession | None) -> None:
        """Close the agent's terminal; a session outliving its sandbox leaks its connection."""
        if session is None:
            return
        try:
            await session.close()
        except Exception:
            print("Failed to close SWE-bench Pro PTY session", format_exc(), file=sys.stderr)

    async def shutdown(self) -> None:
        sandboxes = list(self._session_id_to_sandbox.values())
        sessions = list(self._session_id_to_pty.values())
        self._session_id_to_sandbox.clear()
        self._session_id_to_pty.clear()
        self._session_id_to_pristine_untracked.clear()
        for session in sessions:
            await self.close_pty_session(session)
        for sandbox in sandboxes:
            try:
                await sandbox.stop()
            except Exception:
                print("Failed to stop abandoned SWE-bench Pro sandbox", format_exc(), file=sys.stderr)

    def _image(self, body: SWEBenchProInstanceRequest) -> str:
        if body.image_digest:
            return f"{self.config.image_repository}@{body.image_digest}"
        return f"{self.config.image_repository}:{body.dockerhub_tag}"

    async def _create_sandbox(
        self,
        body: SWEBenchProInstanceRequest,
        files: dict[str, str] | None = None,
    ) -> AsyncSandbox:
        global_config_dict = get_global_config_dict()
        provider_config = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        provider_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)
        spec = SandboxSpec(
            image=self._image(body),
            ttl_s=self.config.sandbox_config.get("ttl_s"),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s"),
            workdir="/app",
            env=dict.fromkeys(HARNESS_ENV_TO_SCRUB, "") | SANDBOX_ENV_OVERRIDES,
            files=files or {},
            metadata=provider_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {
                "nemo_gym_agent": self.config.name,
                "instance_id": body.instance_id[:63],
            },
            resources=SandboxResources.from_mapping(self.config.sandbox_config.get("resources", {})),
            entrypoint=None,
            provider_options=self.config.sandbox_config.get("provider_options", {}),
        )
        sandbox = AsyncSandbox(provider_config)
        await sandbox.start(spec)
        return sandbox

    def _verification_inputs(self, body: SWEBenchProInstanceRequest, patch: str) -> VerificationInputs:
        return VerificationInputs(
            instance_id=body.instance_id,
            base_commit=body.base_commit,
            patch=patch,
            run_script=body.run_script,
            parser_script=body.parser_script,
            selected_test_files_to_run=body.selected_test_files_to_run,
            fail_to_pass=body.fail_to_pass,
            pass_to_pass=body.pass_to_pass,
            before_repo_set_cmd=body.before_repo_set_cmd,
            base_dockerfile=body.base_dockerfile,
            instance_dockerfile=body.instance_dockerfile,
            repo_language=body.repo_language,
            prefetch_go_modules=self.config.prefetch_go_modules,
            environment_repairs=tuple(self.config.environment_repairs),
        )

    async def seed_session(
        self, request: Request, body: SWEBenchProSeedSessionRequest
    ) -> SWEBenchProSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        self._session_id_to_pristine_untracked.pop(session_id, None)
        await self.close_pty_session(self._session_id_to_pty.pop(session_id, None))
        previous = self._session_id_to_sandbox.pop(session_id, None)
        if previous is not None:
            try:
                await previous.stop()
            except Exception:
                print("Failed to stop previous SWE-bench Pro sandbox", format_exc(), file=sys.stderr)

        sandbox = await self._create_sandbox(body)
        pty_session = await sandbox.pty.create()
        if self.config.apply_anti_cheating:
            anti_cheat_setup_fpath = Path(__file__).parent.parent / "swebench" / "anti_cheat_setup.sh"
            await sandbox.upload(anti_cheat_setup_fpath, "/app/anti_cheat_setup.sh")
            result = await sandbox.exec(
                "git reset --hard && WORKING_DIRECTORY=/app bash anti_cheat_setup.sh && rm anti_cheat_setup.sh",
                timeout_s=600,
            )
            if result.return_code != 0:
                print(
                    f"Failed to setup anti-cheating for {body.instance_id}. Return code: {result.return_code}\n"
                    f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
                )
        await self.normalize_sandbox_environment(sandbox, body.instance_id)
        self._session_id_to_pristine_untracked[session_id] = await self.pristine_untracked_files(sandbox)
        self._session_id_to_sandbox[session_id] = sandbox
        self._session_id_to_pty[session_id] = pty_session
        return SWEBenchProSeedSessionResponse(
            sandbox_handle=sandbox._handle.sandbox_id, pty_session_id=pty_session.session_id
        )

    async def normalize_sandbox_environment(self, sandbox: AsyncSandbox, instance_id: str) -> None:
        """Give the agent container the same repairs the verifier gets; best effort."""
        try:
            script = build_seed_normalization(self.config.environment_repairs)
            result = await sandbox.exec(f"bash -c {quote(script)}", timeout_s=300)
            if result.return_code != 0:
                print(
                    f"Failed to normalize sandbox environment for {instance_id}. "
                    f"Return code: {result.return_code}\nStderr:\n{result.stderr}",
                    file=sys.stderr,
                )
        except Exception:
            print(f"Failed to normalize sandbox environment for {instance_id}", format_exc(), file=sys.stderr)

    async def pristine_untracked_files(self, sandbox: AsyncSandbox) -> frozenset[str]:
        """List the untracked files ``/app`` holds before the agent touches it."""
        try:
            result = await sandbox.exec("git -C /app ls-files --others --exclude-standard")
            if result.return_code != 0:
                print(f"Failed to list pristine untracked files: {result.stderr}", file=sys.stderr)
                return frozenset()
            return frozenset(line.strip() for line in (result.stdout or "").splitlines() if line.strip())
        except Exception:
            print("Failed to list pristine untracked files", format_exc(), file=sys.stderr)
            return frozenset()

    async def _extract_model_patch(self, session_id: str, base_commit: str) -> str:
        original_sandbox = self._session_id_to_sandbox.pop(session_id)
        original_pty_session = self._session_id_to_pty.pop(session_id, None)
        pristine_untracked = self._session_id_to_pristine_untracked.pop(session_id, frozenset())
        try:
            result = await original_sandbox.exec(
                f"git -C /app add -N . && git -C /app --no-pager diff {quote(base_commit)}"
            )
            if result.return_code != 0:
                raise RuntimeError(result.stderr or "git diff failed")
            return drop_patch_sections(result.stdout or "", pristine_untracked)
        finally:
            await self.close_pty_session(original_pty_session)
            try:
                await original_sandbox.stop()
            except Exception:
                print("Failed to stop agent sandbox", format_exc(), file=sys.stderr)

    async def verify(self, request: Request, body: SWEBenchProVerifyRequest) -> SWEBenchProVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        extraction_error = None
        if self.config.is_verifying_golden_patch:
            model_patch = body.patch
        else:
            try:
                model_patch = await self._extract_model_patch(session_id, body.base_commit)
            except Exception as exc:
                model_patch = ""
                extraction_error = f"Failed to extract model patch: {exc}"

        inputs = self._verification_inputs(body, model_patch)
        workspace_files, _ = assemble_workspace_files(body.instance_id, None, model_patch, asdict(inputs))
        sandbox_files = {
            f"/workspace/{relative_path}": contents for relative_path, contents in workspace_files.items()
        }

        run_log_dir = Path(__file__).parent / "logs" / "run_evaluation" / session_id / body.instance_id
        eval_sandbox_start_time_taken = 0.0
        patch_verification_time_taken = 0.0
        attempts = 1 + max(self.config.inconclusive_verification_retries, 0)
        for attempt in range(1, attempts + 1):
            eval_sandbox: AsyncSandbox | None = None
            start_time = time()
            try:
                eval_sandbox = await self._create_sandbox(body, files=sandbox_files)
                eval_sandbox_start_time_taken = time() - start_time
                verification_start = time()
                result = await run_verification(
                    sandbox=eval_sandbox,
                    inputs=inputs,
                    log_dir=run_log_dir,
                    timeout_s=self.config.evaluation_timeout,
                )
                patch_verification_time_taken = time() - verification_start
            except Exception as exc:
                eval_sandbox_start_time_taken = time() - start_time
                patch_verification_time_taken = 0.0
                result = VerificationResult(
                    completed=False,
                    resolved=False,
                    patch_applied=False,
                    test_results=None,
                    error=f"Verification failed: {exc}",
                )
            finally:
                if eval_sandbox is not None:
                    try:
                        await eval_sandbox.stop()
                    except Exception:
                        print("Failed to stop verification sandbox", format_exc(), file=sys.stderr)

            reason = inconclusive_reason(result, asdict(inputs))
            if reason is None or attempt == attempts:
                if reason is not None:
                    print(
                        f"Verification for {body.instance_id} still inconclusive after {attempt} attempt(s): {reason}",
                        file=sys.stderr,
                    )
                break
            print(
                f"Retrying verification for {body.instance_id} on a new sandbox "
                f"(attempt {attempt}/{attempts} was inconclusive: {reason})",
                file=sys.stderr,
            )

        response_data = body.model_dump() | {
            "reward": float(result.resolved),
            "evaluation_completed": result.completed,
            "resolved": result.resolved,
            "patch_applied": result.patch_applied,
            "eval_sandbox_start_time_taken": eval_sandbox_start_time_taken,
            "patch_verification_time_taken": patch_verification_time_taken,
            "instance_id": body.instance_id,
            "model_patch": model_patch or None,
            "test_results": result.test_results,
            "test_output": result.test_output,
            "error": extraction_error or result.error,
            "log_dir": str(run_log_dir),
        }
        return SWEBenchProVerifyResponse.model_validate(response_data)


if __name__ == "__main__":
    SWEBenchProResourcesServer.run_webserver()
