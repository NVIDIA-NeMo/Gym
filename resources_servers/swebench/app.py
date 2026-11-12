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

import hashlib
import re
import sys
from glob import glob
from pathlib import Path
from shlex import quote
from shutil import rmtree
from tempfile import TemporaryDirectory
from time import monotonic, time
from traceback import format_exc
from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple
from uuid import uuid4

from fastapi import Request
from pydantic import BaseModel, Field, JsonValue, field_validator
from swebench.harness.run_evaluation import make_test_spec
from swebench.harness.test_spec.test_spec import LATEST, TestSpec

from docker.models.containers import ExecResult
from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import get_global_config_dict
from nemo_gym.core import (
    SESSION_LOST,
    VERIFIER_ERROR,
    FailureBoundary,
    NgFailureEvent,
    current_verifier_invocation_id,
    observe_failure,
)
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxResources,
    SandboxSpec,
    cpu_cap_env,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    CheckpointMode,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    StepType,
    lost_session_verification,
)
from nemo_gym.trajectory import SandboxObservation
from nemo_gym.utils import bounded_text, is_nemo_gym_fastapi_entrypoint
from resources_servers.swebench.sandbox_session_checkpoints import (
    CheckpointedSandboxSessionResourcesServer,
    SandboxCheckpointFields,
)
from resources_servers.swebench.swebench_patches import (
    patch_swebench_multilingual_golden_patch_pass,
    patch_swebench_multilingual_log_parsing,
    patch_swebench_multilingual_sandbox,
    patch_swebench_verified_env_vars,
    run_instance,
)
from resources_servers.swebench.task_data import SWEBenchImageAssets, SWEBenchTestAsset, image_assets_from_json


# The longest failure_reason a failure event accepts.
_MAX_FAILURE_REASON_CHARS = 2000


def staged_test_asset_path(index: int) -> str:
    """Where verification uploads the test patch's binary file ``index`` before the evaluation script runs."""
    return f"/tmp/nemo-gym-swebench-test-asset-{index}"


def restore_test_assets_from_staging(test_spec: TestSpec, assets: Sequence[SWEBenchTestAsset]) -> None:
    """Make the evaluation script copy each of the test patch's binary files from its staged upload.

    SWE-bench's JavaScript evaluation script runs ``curl -o <path> <url>`` for each file right after resetting the
    test files. The verifier sandbox may have no internet access, and curl without ``--fail`` would also write an
    error page where the image belongs, so each download is replaced by a copy of the uploaded bytes.
    """
    for index, asset in enumerate(assets):
        download = f"curl -o {asset.path} {asset.url}"
        if download not in test_spec.eval_script_list:
            raise ValueError(f"The evaluation script of {test_spec.instance_id} does not download {asset.path}")
        position = test_spec.eval_script_list.index(download)
        test_spec.eval_script_list[position] = f"cp {quote(staged_test_asset_path(index))} {quote(asset.path)}"


class SwebenchResourcesServerConfig(SandboxCheckpointFields, BaseResourcesServerConfig):
    is_verifying_golden_patch: bool = False
    apply_anti_cheating: bool = True
    # Write the Maven Central mirror to ~/.m2/settings.xml in the eval sandbox. Turn off for providers
    # that mount their own read-only settings.xml, e.g. one that already mirrors Maven Central.
    overwrite_maven_settings: bool = True

    evaluation_timeout: Optional[int] = None

    # Sandbox config
    sandbox_provider: str
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
    # SWE-bench Multimodal only. Its evaluation script restores the test patch's binary files (reference images)
    # from here, so the field must reach make_test_spec.
    image_assets: Optional[SWEBenchImageAssets] = None

    @field_validator("image_assets", mode="before")
    @classmethod
    def _parse_image_assets(cls, value: object) -> object:
        return image_assets_from_json(value)

    def test_assets(self) -> list[SWEBenchTestAsset]:
        return self.image_assets.test_patch if self.image_assets else []


class SWEBenchVerifyRequest(SWEBenchInstanceRequest, BaseVerifyRequest):
    pass


class SWEBenchVerifyResponse(BaseVerifyResponse):
    evaluation_completed: bool
    resolved: bool

    # Misc metrics
    eval_sandbox_start_time_taken: float
    patch_verification_time_taken: float

    instance_id: str
    test_output: str
    model_patch: Optional[str]
    # sha256 of the patch that was verified (UTF-8), so a later re-verification can prove it grades the same patch;
    # None when the patch could not be extracted from the agent's sandbox.
    model_patch_sha256: Optional[str] = None

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
            stage="verification",
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


# The agent's working tree is graded by the diff it leaves behind. A plain
# `git diff` shows only unstaged changes to tracked files, so a fix the agent
# staged (`git add`) or committed scored 0 with an empty patch: 14% of Claude
# Code rollouts on SWE-bench Verified did one of the two (2026-10-19). Diff the
# index against the commit the session was seeded at instead, after staging
# the tracked files' changes, so staged and committed fixes count. Untracked
# files are left out on purpose: `git add -A` swept a generated Sphinx `_build`
# tree (1.1 MB of jquery) into one patch; a new file the agent committed is
# still included. Binary sections are dropped, since `git apply` cannot take
# them without binary data.
def patch_extraction_command(workdir: str, seed_head: str) -> str:
    return (
        f"cd {quote(workdir)} && git add -u >/dev/null 2>&1; "
        f"cd {quote(workdir)} && git --no-pager diff --cached {quote(seed_head)}"
    )


def strip_binary_diffs(patch: str) -> str:
    """Remove `diff --git` sections for binary files (git prints "Binary files ... differ" without the data)."""
    if not patch:
        return patch
    sections = re.split(r"(?m)^(?=diff --git )", patch)
    return "".join(sec for sec in sections if not re.search(r"(?m)^Binary files .* differ$", sec))


class SwebenchResourcesServer(CheckpointedSandboxSessionResourcesServer):
    config: SwebenchResourcesServerConfig
    task_model: ClassVar[type[BaseModel]] = SWEBenchInstanceRequest
    # The patch is captured once per session, so a verify repeated after a restore grades the same patch.
    checkpoint_verify: ClassVar[StepType] = "replay"

    def checkpoint_mode_in_effect(self) -> CheckpointMode:
        # Golden-patch grading seeds no episode sessions, so it has nothing to checkpoint.
        if self.config.is_verifying_golden_patch:
            return "restart_only"
        return super().checkpoint_mode_in_effect()

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)

        self._session_id_to_seed_head: Dict[str, str] = dict()

    async def _create_sandbox(
        self,
        test_spec: TestSpec,
        *,
        test_assets: Sequence[SWEBenchTestAsset] = (),
        task_session_id: str | None = None,
        verifier_session_id: str | None = None,
    ) -> AsyncSandbox:
        """Start a sandbox from the instance image. ``test_assets`` are uploaded for the evaluation script only: the
        agent's sandbox must not hold the hidden tests' reference files."""
        # TODO @bxyu-nvidia: Refactor this after Hemil's swap from Python dataclass to Pydantic BaseModel
        global_config_dict = get_global_config_dict()
        resolved_sandbox_provider = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
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
        if task_session_id is not None:
            # On the provider every task sandbox shares, and registered for the session and its checkpoints.
            eval_sandbox = self.new_task_sandbox(task_session_id, eval_sandbox_spec)
        else:
            eval_sandbox = AsyncSandbox(resolved_sandbox_provider)
        if verifier_session_id is not None:
            self._verifier_sandboxes[verifier_session_id] = eval_sandbox

        async def _run_setup(sandbox: AsyncSandbox) -> None:
            await patch_swebench_multilingual_sandbox(
                test_spec.repo,
                test_spec.instance_id,
                sandbox,
                overwrite_maven_settings=self.config.overwrite_maven_settings,
            )
            if test_assets:
                with TemporaryDirectory(prefix="nemo-gym-swebench-test-assets-") as staging_dir:
                    for index, asset in enumerate(test_assets):
                        local_path = Path(staging_dir) / str(index)
                        local_path.write_bytes(asset.content)
                        await sandbox.upload(local_path, staged_test_asset_path(index))

        await eval_sandbox.start_with_setup(eval_sandbox_spec, _run_setup)

        return eval_sandbox

    def _make_test_spec(self, body: SWEBenchInstanceRequest) -> TestSpec:
        test_spec = make_test_spec(
            # This accepts a SWEbenchInstance: our body's instance fields, without the agent's request and response.
            # SWE-bench reads some optional keys whenever they are present, such as image_assets for JavaScript,
            # so an unset optional field is left out rather than sent as null.
            body.model_dump(include=set(SWEBenchInstanceRequest.model_fields), exclude_none=True),
            namespace="swebench",  # Dockerhub namespace
            instance_image_tag=LATEST,
            env_image_tag=LATEST,
        )
        restore_test_assets_from_staging(test_spec, body.test_assets())
        return test_spec

    async def seed_session(
        self, request: Request, body: SWEBenchSeedSessionRequest | ResourcesSeedSessionRequest
    ) -> SWEBenchSeedSessionResponse | ResourcesSeedSessionResponse:
        """Create the task sandbox for an agent-run seed's instance body or an Environment Server's episode seed.

        An episode seed returns ``sandbox_access`` with the serialized task sandbox and the image's working
        directory; an agent-run seed returns the sandbox handle.
        """
        if isinstance(body, ResourcesSeedSessionRequest):
            return await self.seed_task_sandbox_session(request, body, SWEBenchInstanceRequest)
        session_id = request.session[SESSION_ID_KEY]
        async with self._locked_task_session(session_id) as record:
            if record.seed is not None or session_id in self._task_session_fences:
                raise ResourcesSessionConflictError("An agent-run seed cannot replace an episode session")
            await self._stop_task_sandbox(session_id)
            try:
                record.workdir = await self._start_task_sandbox(session_id, body)
            except BaseException:
                await self._stop_failed_seed_sandbox(session_id)
                raise
            return SWEBenchSeedSessionResponse(
                sandbox_handle=self._session_id_to_sandbox[session_id]._handle.sandbox_id
            )

    async def _start_task_sandbox(self, session_id: str, body: SWEBenchInstanceRequest) -> str:
        """Create the task sandbox for ``session_id``, apply the anti-cheating setup, and return its WORKDIR.

        The agent works where verify later extracts the patch: the image's WORKDIR.
        """
        test_spec = self._make_test_spec(body)
        eval_sandbox = await self._create_sandbox(test_spec, task_session_id=session_id)
        self._session_id_to_sandbox[session_id] = eval_sandbox

        pwd = await eval_sandbox.exec("pwd")
        wd = (pwd.stdout or "").strip()
        if pwd.error_type is not None or pwd.return_code != 0 or not wd.startswith("/"):
            raise RuntimeError(f"Could not resolve the task sandbox workdir: {pwd.stderr}")

        if self.config.apply_anti_cheating:
            # Remove the current Git repo's future history beyond the current commit to prevent the model from cheating.
            anti_cheat_setup_fpath = Path(__file__).parent / "anti_cheat_setup.sh"
            await eval_sandbox.upload(anti_cheat_setup_fpath, f"{wd}/anti_cheat_setup.sh")
            result = await eval_sandbox.exec(
                f"""git reset --hard && WORKING_DIRECTORY={wd} bash anti_cheat_setup.sh && rm anti_cheat_setup.sh"""
            )
            if result.return_code != 0:
                print(f"""Failed to setup anti-cheating for {test_spec.instance_id}. Return code: {result.return_code}
Stdout:
{result.stdout}
Stderr:
{result.stderr}""")

        # The commit to diff against at verify time: HEAD once the sandbox is
        # ready (anti-cheating rewrites refs but not the tree), falling back to
        # the task's base commit if git is unavailable.
        head_result = await eval_sandbox.exec("git rev-parse HEAD")
        seed_head = (head_result.stdout or "").strip() if head_result.return_code == 0 else ""
        self._session_id_to_seed_head[session_id] = seed_head or test_spec.base_commit

        return wd

    def _forget_task_sandbox_state(self, session_id: str) -> None:
        self._session_id_to_seed_head.pop(session_id, None)

    def _export_task_sandbox_state(self, session_id: str) -> dict[str, JsonValue]:
        # The commit a restored session's patch is captured against.
        seed_head = self._session_id_to_seed_head.get(session_id)
        return {"seed_head": seed_head} if seed_head else {}

    def _restore_task_sandbox_state(self, session_id: str, state: dict[str, JsonValue]) -> None:
        if state.get("seed_head"):
            self._session_id_to_seed_head[session_id] = str(state["seed_head"])

    async def verify(self, request: Request, body: SWEBenchVerifyRequest) -> SWEBenchVerifyResponse:
        """Grade the resources-owned sandbox, reusing the patch and verdict on an episode session's retries."""
        return await self.verify_task_sandbox_session(
            request,
            body,
            verify=lambda: self._verify(request, body),
            lost_response=lambda reason, failure_kind: _lost_session_response(body, reason, failure_kind=failure_kind),
            allow_seedless=self.config.is_verifying_golden_patch,
        )

    async def _verify(self, request: Request, body: SWEBenchVerifyRequest) -> SWEBenchVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        test_spec = self._make_test_spec(body)
        patch_extraction_failed = False
        if self.config.is_verifying_golden_patch:
            model_patch = body.patch
        else:

            async def capture() -> str:
                original_sandbox = self._session_id_to_sandbox[session_id]
                record = self._task_sessions.get(session_id)
                if record is not None and record.workdir:
                    workdir = record.workdir
                else:
                    pwd = await original_sandbox.exec("pwd")
                    workdir = (pwd.stdout or "").strip()
                    if pwd.error_type is not None or pwd.return_code != 0 or not workdir.startswith("/"):
                        raise RuntimeError("Could not resolve the task sandbox workdir for patch capture")
                seed_head = self._session_id_to_seed_head.get(session_id) or test_spec.base_commit
                result = await original_sandbox.exec(patch_extraction_command(workdir, seed_head))
                if result.error_type is not None:
                    raise RuntimeError("Failed to extract the model patch from the task sandbox")
                if result.return_code != 0:
                    # The agent can break the repository this runs in (removing .git, pruning the seed commit), so
                    # a git failure is its graded result: the empty patch fails to apply and scores 0, unmasked.
                    print(f"Patch extraction exited {result.return_code}: {result.stderr}", file=sys.stderr)
                return strip_binary_diffs(result.stdout or "")

            try:
                model_patch = await self.capture_task_sandbox_patch(session_id, capture)
            except Exception:
                patch_extraction_failed = True
                model_patch = ""
                print("Failed to extract patch from container", format_exc(), file=sys.stderr)
            # A failed stop of an episode session retains the handle and captured patch for a retry.
            await self.stop_task_sandbox_after_capture(session_id)

        verifier_sandbox_lifecycle_started_at = monotonic()
        eval_sandbox = await self._create_sandbox(
            test_spec, test_assets=body.test_assets(), verifier_session_id=session_id
        )
        self._verifier_sandboxes[session_id] = eval_sandbox
        eval_sandbox_start_time_taken = monotonic() - verifier_sandbox_lifecycle_started_at

        # A verify replayed after a restore may grade while the first one still runs: they must not share log files.
        run_id = f"{session_id}-{uuid4().hex[:8]}"
        mock_container = DockerContainer(id=run_id, instance_id=test_spec.instance_id)
        mock_container._inner_container = eval_sandbox

        # Res has 2 keys: completed (whether evaluation completed or not), resolved (whether the issue is resolved)
        start_time = time()
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

        evaluation_completed = res["completed"] and mock_container._sandbox_error_type is None
        # A patch the harness could not apply is the model's own graded result when the patch was collected intact
        # and no sandbox operation failed. Any other evaluation that did not complete has no verdict: its zero is not
        # a grade, so the sample is masked and the failure is reported as a verifier error.
        model_patch_unapplied = (
            bool(res.get("patch_apply_failed"))
            and mock_container._sandbox_error_type is None
            and not patch_extraction_failed
        )
        failure_reason = None
        if not evaluation_completed and not model_patch_unapplied:
            if patch_extraction_failed:
                failure_reason = "Failed to extract the model patch from the agent sandbox"
            elif mock_container._sandbox_error_type is not None:
                failure_reason = f"Verification sandbox error: {mock_container._sandbox_error_type}"
            else:
                failure_reason = res.get("error") or "SWE-bench evaluation did not complete"
            failure_reason = bounded_text(failure_reason, _MAX_FAILURE_REASON_CHARS)
            observe_failure(
                NgFailureEvent(
                    boundary=FailureBoundary.RESOURCES_SERVER,
                    failure_kind=VERIFIER_ERROR,
                    failure_reason=failure_reason,
                    operation="verify",
                    operation_id=current_verifier_invocation_id() or run_id,
                    stage="verification",
                )
            )

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
            evaluation_completed=evaluation_completed,
            resolved=res["resolved"],
            reward=int(res["resolved"]),
            eval_sandbox_start_time_taken=eval_sandbox_start_time_taken,
            patch_verification_time_taken=patch_verification_time_taken,
            model_patch=model_patch or None,
            model_patch_sha256=None
            if patch_extraction_failed
            else hashlib.sha256(model_patch.encode("utf-8")).hexdigest(),
            mask_sample=failure_reason is not None,
            failure_kind=VERIFIER_ERROR if failure_reason is not None else None,
            failure_reason=failure_reason,
            test_output=test_output,
            log_dir=str(log_dir),
            verifier_sandbox_observation=verifier_sandbox_observation,
        )


def _lost_session_response(
    body: SWEBenchVerifyRequest, reason: str, *, failure_kind: str = SESSION_LOST
) -> SWEBenchVerifyResponse:
    """Return an unscored, masked result when the task state is unavailable."""
    return SWEBenchVerifyResponse.model_validate(
        body.model_dump()
        | lost_session_verification(reason, failure_kind=failure_kind)
        | {
            "evaluation_completed": False,
            "resolved": False,
            "eval_sandbox_start_time_taken": 0.0,
            "patch_verification_time_taken": 0.0,
            "instance_id": body.instance_id,
            "model_patch": None,
            "model_patch_sha256": None,
            "test_output": "",
            "log_dir": "",
        }
    )


if __name__ == "__main__":
    SwebenchResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = SwebenchResourcesServer.run_webserver()  # noqa: F401
