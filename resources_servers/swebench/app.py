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

import asyncio
import sys
from functools import partial
from glob import glob
from pathlib import Path
from shlex import quote
from shutil import rmtree
from time import monotonic, time
from traceback import format_exc
from typing import Any, ClassVar, Dict, Optional, Tuple
from uuid import uuid4

from anyio import CancelScope
from docker.models.containers import ExecResult
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, PrivateAttr
from swebench.harness.run_evaluation import make_test_spec
from swebench.harness.test_spec.test_spec import LATEST, TestSpec

from nemo_gym.base_resources_server import (
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.judge import judge_failsafe
from nemo_gym.rollout_observability import SandboxObservation
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_metadata
from nemo_gym.sandbox.owner import SandboxOwnerConfig, SandboxOwnerResourcesServer
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import SESSION_ID_KEY
from nemo_gym.telemetry.endpoints import traced_verify_endpoint
from resources_servers.swebench.swebench_patches import (
    patch_swebench_multilingual_golden_patch_pass,
    patch_swebench_multilingual_log_parsing,
    patch_swebench_multilingual_sandbox,
    patch_swebench_verified_env_vars,
    run_instance,
)


class SwebenchResourcesServerConfig(SandboxOwnerConfig):
    is_verifying_golden_patch: bool = False
    apply_anti_cheating: bool = True
    # Write the Maven Central mirror to ~/.m2/settings.xml in the eval sandbox. Turn off for providers
    # that mount their own read-only settings.xml, e.g. one that already mirrors Maven Central.
    overwrite_maven_settings: bool = True

    evaluation_timeout: Optional[int] = None

    # Sandbox config; sandbox_provider and sandbox_checkpoint come from SandboxOwnerConfig.
    sandbox_config: Dict[str, Any]

    clear_swebench_debug_logs: bool = True

    def model_post_init(self, context: Any, /) -> None:
        if self.is_verifying_golden_patch and self.clear_swebench_debug_logs:
            print("Turning off logs clear since `is_verifying_golden_patch=true`")
            self.clear_swebench_debug_logs = False


class SWEBenchInstanceRequest(BaseModel):
    # See https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified
    # See swebench.harness.run_evaluation.TestSpec https://github.com/SWE-bench/SWE-bench/blob/f7bbbb2ccdf479001d6467c9e34af59e44a840f9/swebench/harness/test_spec/test_spec.py#L28
    repo: str
    instance_id: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str
    created_at: str
    version: str
    # These are JSON strings.
    FAIL_TO_PASS: str
    PASS_TO_PASS: str
    environment_setup_commit: str
    difficulty: str
    subset: str
    split: str


class SWEBenchVerifyRequest(SWEBenchInstanceRequest, BaseVerifyRequest):
    pass


class SWEBenchPreparedVerifyRequest(SWEBenchVerifyRequest):
    # Required even for an empty diff: this route never reads an agent sandbox.
    model_patch: str


class SWEBenchPreparedPatch(BaseModel):
    model_patch: str


class SWEBenchVerifyResponse(BaseVerifyResponse):
    evaluation_completed: bool
    resolved: bool

    # Misc metrics
    eval_sandbox_start_time_taken: float
    patch_verification_time_taken: float

    instance_id: str
    test_output: str
    model_patch: Optional[str]

    log_dir: str

    verifier_sandbox_observation: Optional[SandboxObservation] = Field(
        default=None, exclude_if=lambda value: value is None
    )


# @bxyu-nvidia: This is a wrapper that can be passed directly to a very lightly modified version of `run_instance`
# The method is almost identical to the original, just with async awaits rather than sync.
# See resources_servers/swebench/swebench_patches.py
class DockerContainer(BaseModel):
    id: str
    instance_id: str

    _inner_container: AsyncSandbox
    _eval_return_code: Optional[int] = None
    _sandbox_error_type: Optional[str] = None

    async def exec_run(
        self,
        command: str,
        workdir: Optional[str] = None,
        user: Optional[str] = None,
    ) -> ExecResult:
        try:
            res = await self._inner_container.exec(
                command=command,
                cwd=workdir,
                user=user,
            )
        except Exception as exc:
            self._sandbox_error_type = self._sandbox_error_type or type(exc).__name__
            print(f"Failed to exec in SWE Bench for {self.instance_id}", format_exc(), file=sys.stderr)
            raise
        if res.error_type is not None:
            self._sandbox_error_type = self._sandbox_error_type or res.error_type

        return ExecResult(
            exit_code=res.return_code,
            # @bxyu-nvidia: This is not entirely 1:1, but it works for the purposes of this patch.
            # The sandbox API returns None for an empty stream (docker-py returned bytes).
            output=((res.stdout or "") + (res.stderr or "")).encode(),
        )

    async def exec_run_with_timeout(self, command: str, timeout: int) -> Tuple[str, bool, float]:
        # Returns: test_output: str, timed_out: bool, total_runtime: float
        start_time = time()
        try:
            res = await self._inner_container.exec(
                command=command,
                # AsyncSandbox.exec takes timeout_s, not docker-py's timeout.
                timeout_s=timeout,
            )
            self._eval_return_code = res.return_code if res.error_type is None else None
            if res.error_type is not None:
                self._sandbox_error_type = res.error_type
            timed_out = False

            stdout = res.stdout or ""
            stderr = res.stderr or ""

            maybe_test_output = patch_swebench_multilingual_log_parsing(stdout, stderr, self.instance_id)
            test_output = maybe_test_output or (stdout + stderr)
        except TimeoutError:
            # Gym Sandbox API will throw a timeout error on actual timeout.
            timed_out = True
            self._sandbox_error_type = "TimeoutError"
            test_output = ""
        except Exception as exc:
            self._sandbox_error_type = type(exc).__name__
            print(
                f"Failed to exec_run_with_timeout in SWE Bench for {self.instance_id}", format_exc(), file=sys.stderr
            )
            raise

        return (test_output, timed_out, time() - start_time)

    async def copy(self, src: Path, dest: Path) -> None:
        if "eval.sh" in str(src):
            data = src.read_text()
            src.write_text(patch_swebench_multilingual_golden_patch_pass(data, self.instance_id))

        try:
            await self._inner_container.upload(local_path=src, remote_path=str(dest))
        except Exception as exc:
            self._sandbox_error_type = self._sandbox_error_type or type(exc).__name__
            print(
                f"Failed to upload to verification sandbox in SWE Bench for {self.instance_id}",
                format_exc(),
                file=sys.stderr,
            )
            raise

    async def cleanup(self) -> None:
        try:
            # A cancelled verification still stops its sandbox.
            with CancelScope(shield=True):
                await self._inner_container.stop()
        except Exception as exc:
            self._sandbox_error_type = self._sandbox_error_type or type(exc).__name__
            print("Failed to stop verification sandbox", format_exc(), file=sys.stderr)

    def observation(self, *, wall_time_s: float, evaluation_completed: bool) -> SandboxObservation:
        handle = self._inner_container._handle
        normalized_error = self._sandbox_error_type.lower() if isinstance(self._sandbox_error_type, str) else ""
        if "timeout" in normalized_error:
            outcome = "timeout"
        elif self._sandbox_error_type is not None:
            outcome = "sandbox_error"
        elif evaluation_completed:
            outcome = "completed"
        else:
            outcome = "failed"

        return SandboxObservation(
            role="verifier",
            provider=handle.provider_name if handle is not None else None,
            sandbox_id=handle.sandbox_id if handle is not None else None,
            outcome=outcome,
            exit_code=self._eval_return_code,
            wall_time_s=wall_time_s,
            error_type=self._sandbox_error_type,
        )


# TODO @bxyu-nvidia: Eventually once the sandbox server infra is ready, these seed_session types need to upgrade to pass a sandbox spec.
# They can possibly even omitted once this graduates to core infra.
class SWEBenchSeedSessionRequest(SWEBenchInstanceRequest, BaseSeedSessionRequest):
    sandbox_spec: Optional[Dict[str, Any]] = None


class SWEBenchSeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str  # @bxyu-nvidia: Just a plain string URI for now for OpenSandbox backend.


class SwebenchResourcesServer(SandboxOwnerResourcesServer):
    """Each session owns the sandbox the agent works in, from /seed_session until /verify extracts the patch."""

    config: SwebenchResourcesServerConfig
    # Prepared verification: /prepare_verify captures the patch and ends the session (a short wait step for the
    # caller, which records the patch in its bookmark); /verify_prepared grades that patch in a fresh sandbox and
    # touches no session state, so a checkpoint neither waits for the grading nor keeps the agent's sandbox for it.
    checkpoint_replayable_paths: ClassVar[tuple[str, ...]] = ("/verify_prepared",)
    # Only for a /prepare_verify reply lost after the sandbox was stopped: the retry gets the same patch.
    _prepared_patches: Dict[str, str] = PrivateAttr(default_factory=dict)
    _prepare_locks: Dict[str, asyncio.Lock] = PrivateAttr(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/prepare_verify")(self.prepare_verify)
        app.post("/verify_prepared")(
            traced_verify_endpoint(
                judge_failsafe(self.verify_prepared),
                static_attributes={"nemo.gym.server.name": self.config.name},
            )
        )
        return app

    async def prepare_verify(self, request: Request, body: SWEBenchVerifyRequest) -> SWEBenchPreparedPatch:
        """Capture the agent's patch and end the session, stopping its sandbox; a retry gets the same patch."""
        session_id = request.session[SESSION_ID_KEY]
        async with self._prepare_locks.setdefault(session_id, asyncio.Lock()):
            result = await self._prepare_verification_patch(session_id, body)
        self.checkpoint_session_ended(request)
        return result

    async def _prepare_verification_patch(self, session_id: str, body: SWEBenchVerifyRequest) -> SWEBenchPreparedPatch:
        """Capture the patch once per session, then end the session; a failed capture leaves the sandbox running."""
        if session_id not in self._prepared_patches:
            if self.config.is_verifying_golden_patch:
                patch = body.patch
            else:
                # 409 for a session without a sandbox: an empty patch would grade as reward 0.
                sandbox = self.require_session_sandbox(session_id)
                workdir = self._workdirs.get(session_id) or (await sandbox.exec("pwd")).stdout.strip()
                captured = await sandbox.exec(f"cd {quote(workdir)} && git --no-pager diff")
                if captured.return_code != 0 or captured.error_type:
                    raise RuntimeError(
                        f"Failed to capture the generated patch of session {session_id}: "
                        f"return_code={captured.return_code}, error_type={captured.error_type}, "
                        f"stderr={captured.stderr!r}"
                    )
                patch = captured.stdout or ""
            self._prepared_patches[session_id] = patch
        # The continuation now lives in the caller's bookmark, which holds the patch.
        await self.end_session_sandbox(session_id)
        return SWEBenchPreparedPatch(model_patch=self._prepared_patches[session_id])

    async def verify_prepared(self, request: Request, body: SWEBenchPreparedVerifyRequest) -> SWEBenchVerifyResponse:
        """Grade a patch /prepare_verify captured, in a fresh sandbox, without reading or changing an agent sandbox."""
        session_id = request.session[SESSION_ID_KEY]
        self._prepared_patches.pop(session_id, None)
        self._prepare_locks.pop(session_id, None)
        verify_body = SWEBenchVerifyRequest.model_validate(body.model_dump(exclude={"model_patch"}))
        return await self._evaluate_patch(request, verify_body, body.model_patch)

    def lends_sandboxes(self) -> bool:
        # A golden-patch instance grades the dataset's own patch and never seeds: it lends no sandbox.
        return not self.config.is_verifying_golden_patch

    async def _create_sandbox(self, test_spec: TestSpec) -> AsyncSandbox:
        return await self._start_sandbox(self._sandbox_spec(test_spec), test_spec)

    def _sandbox_spec(self, test_spec: TestSpec) -> SandboxSpec:
        # TODO @bxyu-nvidia: Refactor this after Hemil's swap from Python dataclass to Pydantic BaseModel
        global_config_dict = get_global_config_dict()
        provider_default_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)
        resources = dict(self.config.sandbox_config.get("resources", {}))

        # Derive from the final resources map (after the multilingual bump);
        # explicit sandbox_config.env keys win over the derived caps.
        sandbox_resources = SandboxResources.from_mapping(resources)
        env = dict(self.config.sandbox_config.get("env", {}))
        if self.config.sandbox_config.get("derive_cpu_env", True):
            env = cpu_cap_env(sandbox_resources.cpu) | env

        patch_swebench_verified_env_vars(test_spec.repo, env)

        eval_sandbox_spec = SandboxSpec(
            image=test_spec.instance_image_key,
            ttl_s=self.config.sandbox_config.get("ttl_s", None),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s", None),
            workdir=None,  # Default to container's WORKDIR
            env=env,
            files=dict(),
            metadata=provider_default_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {
                "nemo_gym_agent": self.config.name,
                "instance_id": test_spec.instance_id[:63],
            },
            resources=sandbox_resources,
            entrypoint=None,
            provider_options=self.config.sandbox_config.get("provider_options", {}),
        )
        return eval_sandbox_spec

    async def _start_sandbox(self, spec: SandboxSpec, test_spec: TestSpec) -> AsyncSandbox:
        eval_sandbox = AsyncSandbox(self.shared_provider(), spec, owns_provider=False)

        await eval_sandbox.start_with_setup(spec, partial(self._patch_multilingual, test_spec))

        return eval_sandbox

    async def _patch_multilingual(self, test_spec: TestSpec, sandbox: AsyncSandbox) -> None:
        await patch_swebench_multilingual_sandbox(
            test_spec.repo,
            test_spec.instance_id,
            sandbox,
            overwrite_maven_settings=self.config.overwrite_maven_settings,
        )

    def _make_test_spec(self, body: SWEBenchVerifyRequest) -> TestSpec:
        return make_test_spec(
            # This accepts a SWEbenchInstance which is identically our body.
            body.model_dump(),
            namespace="swebench",  # Dockerhub namespace
            instance_image_tag=LATEST,
            env_image_tag=LATEST,
        )

    async def seed_session(self, request: Request, body: SWEBenchSeedSessionRequest) -> SWEBenchSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        sandbox = self.session_sandbox(session_id)
        if sandbox is None:
            test_spec = self._make_test_spec(body)
            sandbox = await self.start_session_sandbox(
                session_id, self._sandbox_spec(test_spec), setup=partial(self._setup_agent_sandbox, test_spec)
            )
        # Otherwise a replacement attempt re-seeds the session a checkpoint restored: it continues in the restored
        # sandbox, the fork of the checkpoint's snapshot, whose setup already ran.
        return SWEBenchSeedSessionResponse(sandbox_handle=sandbox.handle.sandbox_id)

    async def _setup_agent_sandbox(self, test_spec: TestSpec, sandbox: AsyncSandbox) -> None:
        """The agent's sandbox before the agent starts: the multilingual patch, then the anti-cheat scrub.

        Verify diffs the working tree in the sandbox, so the session needs no data beyond the sandbox itself.
        """
        await self._patch_multilingual(test_spec, sandbox)
        if not self.config.apply_anti_cheating:
            return
        wd = (await sandbox.exec("pwd")).stdout.strip()
        # Remove the current Git repo's future history beyond the current commit to prevent the model from cheating.
        anti_cheat_setup_fpath = Path(__file__).parent / "anti_cheat_setup.sh"
        await sandbox.upload(anti_cheat_setup_fpath, f"{wd}/anti_cheat_setup.sh")
        result = await sandbox.exec(
            f"""git reset --hard && WORKING_DIRECTORY={wd} bash anti_cheat_setup.sh && rm anti_cheat_setup.sh"""
        )
        if result.return_code != 0:
            print(f"""Failed to setup anti-cheating for {test_spec.instance_id}. Return code: {result.return_code}
Stdout:
{result.stdout}
Stderr:
{result.stderr}""")

    async def verify(self, request: Request, body: SWEBenchVerifyRequest) -> SWEBenchVerifyResponse:
        """
        Key requirements:
        1. Extract the model_patch from the input container
            Proposal
                1. Spinup a fresh container (need this anyways for running eval)
                2. pwd in the fresh container (defaults to WORKDIR)
                3. cd into WORKDIR in the input container
                4. extract the patch via git
            Notes
                1. DeepSWE expects the model to commit. That will go in the DeepSWE resources server and not this one.
        2. Docker Client - Make a mock client class here that wraps our sandbox client.
        3. Harnesses like OpenCode open a new terminal rather than reusing the existing one. Grab the environment variables and workdir from the outer terminal first, and then export/cd as appropriate in the new terminal
            Notes
                1. This is a harness-specific thing that the harness will handle across benchmarks.
        4. Interleaved thinking - Verify how is the harness behaving i.e. it has interleaved thinking or not and to force interleaved thinking unconditionally.
            Proposal
                1. For seeing what the harness is doing, use model call capture
                2. For forcing, we can add it in the Responses API model proxy i.e. save all the past requests/responses and populate as necessary.
        5. Restrict number of turns - same as interleaved thinking, we could add in Responses API model proxy
        """

        test_spec = self._make_test_spec(body)

        session_id = request.session[SESSION_ID_KEY]
        original_sandbox = None
        if not self.config.is_verifying_golden_patch:
            # 409 for a session without a sandbox: an empty patch would grade as reward 0.
            original_sandbox = self.require_session_sandbox(session_id)

        verifier_sandbox_lifecycle_started_at = monotonic()
        eval_sandbox = await self._create_sandbox(test_spec)
        eval_sandbox_start_time_taken = monotonic() - verifier_sandbox_lifecycle_started_at

        model_patch = ""
        if self.config.is_verifying_golden_patch:
            model_patch = body.patch
        else:
            try:
                original_workdir = (await eval_sandbox.exec("pwd")).stdout.strip()
                model_patch_result = await original_sandbox.exec(f"cd {original_workdir} && git --no-pager diff")
                model_patch = model_patch_result.stdout
            except Exception:
                print("Failed to extract patch from container", format_exc(), file=sys.stderr)
            # The episode is over; its snapshots stay until a release drops the checkpoints that name them.
            await self.end_session_sandbox(session_id)

        return await self._evaluate_patch(
            request,
            body,
            model_patch,
            eval_sandbox=eval_sandbox,
            started_at=verifier_sandbox_lifecycle_started_at,
            start_time_taken=eval_sandbox_start_time_taken,
        )

    async def _evaluate_patch(
        self,
        request: Request,
        body: SWEBenchVerifyRequest,
        model_patch: str,
        *,
        eval_sandbox: Optional[AsyncSandbox] = None,
        started_at: Optional[float] = None,
        start_time_taken: Optional[float] = None,
    ) -> SWEBenchVerifyResponse:
        """Grade ``model_patch`` in ``eval_sandbox``, or in a fresh sandbox for the instance."""
        test_spec = self._make_test_spec(body)
        verifier_sandbox_lifecycle_started_at = monotonic() if started_at is None else started_at
        if eval_sandbox is None:
            eval_sandbox = await self._create_sandbox(test_spec)
        eval_sandbox_start_time_taken = (
            monotonic() - verifier_sandbox_lifecycle_started_at if start_time_taken is None else start_time_taken
        )

        # A replayed /verify_prepared may grade while the original still runs: they must not share log files.
        run_id = f"{request.session[SESSION_ID_KEY]}-{uuid4().hex[:8]}"
        mock_container = DockerContainer(id=run_id, instance_id=test_spec.instance_id)
        mock_container._inner_container = eval_sandbox

        # Res has 2 keys: completed (whether evaluation completed or not), resolved (whether the issue is resolved)
        start_time = time()
        try:
            res = await run_instance(
                test_spec=test_spec,
                pred={
                    "instance_id": test_spec.instance_id,
                    "model_patch": model_patch,
                },
                rm_image=False,
                force_rebuild=False,
                client=mock_container,
                run_id=run_id,
                timeout=self.config.evaluation_timeout,
                rewrite_reports=False,
            )
        finally:
            # run_instance normally stops the sandbox; a failure or cancellation before its cleanup starts must too.
            if eval_sandbox.handle is not None:
                await mock_container.cleanup()
        patch_verification_time_taken = time() - start_time
        verifier_sandbox_wall_time_s = monotonic() - verifier_sandbox_lifecycle_started_at

        try:
            verifier_sandbox_observation = mock_container.observation(
                wall_time_s=verifier_sandbox_wall_time_s,
                evaluation_completed=res["completed"],
            )
        except Exception:
            verifier_sandbox_observation = None
            print("Failed to build verification sandbox observation", format_exc(), file=sys.stderr)

        log_dir = Path(__file__).parent / "logs/run_evaluation" / run_id

        test_output_fpaths = glob(str(log_dir / "**" / "test_output.txt"), recursive=True)
        test_output = ""
        if test_output_fpaths:
            test_output_fpath = Path(test_output_fpaths[0])
            test_output = test_output_fpath.read_text()

        if self.config.clear_swebench_debug_logs:
            rmtree(str(log_dir), ignore_errors=True)
            log_dir = ""

        return SWEBenchVerifyResponse(
            **body.model_dump(),
            # run_instance returns "completed"; the response field is "evaluation_completed".
            evaluation_completed=res["completed"] and mock_container._sandbox_error_type is None,
            resolved=res["resolved"],
            reward=int(res["resolved"]),
            eval_sandbox_start_time_taken=eval_sandbox_start_time_taken,
            patch_verification_time_taken=patch_verification_time_taken,
            model_patch=model_patch or None,
            test_output=test_output,
            log_dir=str(log_dir),
            verifier_sandbox_observation=verifier_sandbox_observation,
        )


if __name__ == "__main__":
    SwebenchResourcesServer.run_webserver()
