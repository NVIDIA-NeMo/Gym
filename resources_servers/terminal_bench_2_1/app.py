# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tarfile
import tomllib
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from shlex import join
from shlex import quote as shlex_quote
from sys import stderr
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import time
from traceback import format_exc
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

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
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import SESSION_ID_KEY


# Bullseye security packages were removed from the live mirror after LTS ended.
# Keep the signed final-LTS repository and package versions available to task setup.
_BULLSEYE_SECURITY_SNAPSHOT_SETUP = r"""set -eu
. "$1"
if [ "${ID:-}:${VERSION_CODENAME:-}" = "debian:bullseye" ]; then
    snapshot=https://snapshot.debian.org/archive/debian-security/20260831T235959Z/
    old='deb http://deb[.]debian[.]org/debian-security bullseye-security main'
    new="deb [check-valid-until=no] $snapshot bullseye-security main"
    sed -i -E "s|^$old$|$new|" "$2"
fi
"""


class TerminalBench21ResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    is_verifying_golden_patch: bool = False
    evaluation_timeout: Optional[int] = None
    # Wall budget for `bash /tests/test.sh`: max(this floor, the task's own limit), where the task's limit is the
    # row's `verifier_timeout_sec` or else `[verifier] timeout_sec` in its task.toml.
    verifier_timeout_floor_sec: int = 1800
    # Run `bash /tests/test.sh` as the row's non-root `agent_user` (the account the vendor images declare as their
    # USER) instead of the image default (root on the `-userroot` derivatives).
    verifier_runs_as_agent_user: bool = True

    # Sandbox config
    sandbox_provider: str
    sandbox_config: Dict[str, Any]

    # Run the Debian/apt source preparation after the sandbox starts. Images whose tests are self-contained
    # (or sandboxes with a deny-all network policy) skip it: with no egress `apt-get update` only burns time.
    prepare_apt_sources: bool = True
    # One small JSON per session (`<session_id>.json`: phase open/closed, request, termination, verdict) so a
    # campaign driver can tell which rollouts are still in flight server-side after it restarts. None disables.
    session_records_dir: Optional[Path] = None

    debug: bool = False


def task_verifier_timeout_sec(task_folder: Path) -> Optional[float]:
    """The task's own ``[verifier] timeout_sec`` from ``task.toml``, or None when missing, unreadable or invalid."""
    try:
        with open(task_folder / "task.toml", "rb") as stream:
            value = tomllib.load(stream).get("verifier", {}).get("timeout_sec")
    except (OSError, tomllib.TOMLDecodeError, AttributeError):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)


def is_root_identity(user: str | int | None) -> bool:
    """``None``, ``"root"`` and ``0`` all mean the image default (root on the supported images)."""
    return user is None or user == "root" or user == 0 or user == "0"


class TerminalBench21SeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str  # @bxyu-nvidia: Just a plain string URI for now for OpenSandbox backend.
    session_id: str
    sandbox_descriptor: Dict[str, Any]
    sandbox_provider: Dict[str, Any]
    instruction: str
    task_id: str
    agent_timeout_sec: float
    # Identity the agent harness runs as (row `agent_user`); None = the image default, as for OpenCode/Terminus.
    user: str | int | None = None


class TerminalBench21SeedSessionRequest(BaseModel):
    task_name: str
    docker_image: str
    task_folder: str


class TerminalBench21RunRequest(TerminalBench21SeedSessionRequest):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming | None = None
    agent_timeout_sec: float = Field(default=28800, gt=0)
    # Optional per-row identity for harnesses that honour the seed's `user` (the in-sandbox mini-SWE agent).
    agent_user: str | int | None = None
    # Wall budget for `bash /tests/test.sh` (defaults to the server's evaluation_timeout) and extra environment
    # for it (e.g. a task's own VERIFIER_WALL_SEC), both taken from the task's metadata by the row builder.
    verifier_timeout_sec: Optional[float] = Field(default=None, gt=0)
    verifier_env: Dict[str, str] = Field(default_factory=dict)
    # Campaign bookkeeping only (recorded in the session record, never interpreted here).
    rollout_id: Optional[str] = None


class TerminalBench21VerifyRequest(TerminalBench21SeedSessionRequest, BaseVerifyRequest):
    pass


class TerminalBench21SessionVerifyRequest(BaseVerifyRequest):
    session_id: str
    termination: Dict[str, Any]
    agent_started: bool = False
    agent_timings: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    harness_metadata: Dict[str, Any] = Field(default_factory=dict)


class TerminalBench21VerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    evaluation_completed: bool
    # Set (to the agent's termination detail) when the tests were NOT run because the agent harness itself
    # failed; mirrors the TB4 resources server so campaign drivers can treat both the same way.
    infrastructure_error: Optional[str] = None

    # Misc metrics
    verification_time_taken: float

    task_name: str
    test_output: str
    golden_patch_output: Optional[str]


GOLDEN_PATCH_SOLVE_SH_PATCHES = {
    "terminal-bench/build-cython-ext": [
        (
            "pip install setuptools==80.9.0 cython==3.1.3",
            "pip install setuptools==80.9.0 cython==3.1.3 planarity==0.6",
        ),
    ],
    "terminal-bench/build-pov-ray": [
        ("wget=1.21.4-1ubuntu4.1", "wget"),
        ("ncompress=5.0-1", "ncompress"),
        (
            "wget https://www.povray.org/ftp/pub/povray/Old-Versions/Official-2.2/POVDOC.TAR.Z",
            "wget --tries=5 --timeout=60 --output-document=POVDOC.TAR.Z "
            "http://grumbeer.dyndns.org/ftp/cdroms/freebsd/freebsd-2.1.7-2/ports/distfiles/povdoc.tar.Z",
        ),
        (
            "wget https://www.povray.org/ftp/pub/povray/Old-Versions/Official-2.2/POVSCN.TAR.Z",
            "wget --tries=5 --timeout=60 --output-document=POVSCN.TAR.Z "
            "http://grumbeer.dyndns.org/ftp/cdroms/freebsd/freebsd-2.1.7-2/ports/distfiles/povscn.tar.Z",
        ),
        (
            "wget https://www.povray.org/ftp/pub/povray/Old-Versions/Official-2.2/POVSRC.TAR.Z",
            """wget --tries=5 --timeout=60 --output-document=POVSRC.TAR.Z \\
  http://grumbeer.dyndns.org/ftp/cdroms/freebsd/freebsd-2.1.7-2/ports/distfiles/povsrc.tar.Z
cat <<'EOF' | sha256sum --check -
e70e44d1fe8835c4dff7c7a55bd6629b15e6a15b2ab7f2f49ee9e2dc016cc470  POVDOC.TAR.Z
4272e2d4724d8dfd916d68827194577221d17b733d99e84e7040f3a9f7eb92a7  POVSCN.TAR.Z
4d8a7073fadaca82827f1354428393cd13e4d3f71a5a3149fd7d6fffd77293d4  POVSRC.TAR.Z
EOF""",
        ),
    ],
}

TEST_SH_PATCHES = {
    "terminal-bench/mcmc-sampling-stan": [
        ("sudo apt-get install -y \\\n    gfortran", "sudo apt-get install -y \\\n    cmake \\\n    gfortran"),
    ],
    "terminal-bench/pytorch-model-recovery": [
        ("-w torch==2.7.1", "-w torch==2.7.1 --index https://download.pytorch.org/whl/cpu"),
    ],
    "terminal-bench/torch-tensor-parallelism": [
        ("-w torch==2.7.0", "-w torch==2.7.0 --index https://download.pytorch.org/whl/cpu"),
    ],
    "terminal-bench/torch-pipeline-parallelism": [
        ("-w torch==2.7.0", "-w torch==2.7.0 --index https://download.pytorch.org/whl/cpu"),
    ],
    "terminal-bench/mteb-retrieve": [
        ("-w mteb==1.36.8", "-w mteb==1.36.8 --index https://download.pytorch.org/whl/cpu"),
    ],
}


class TerminalBench21ResourcesServer(SimpleResourcesServer):
    config: TerminalBench21ResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)

        self._session_id_to_sandbox: Dict[str, AsyncSandbox] = dict()
        self._session_id_to_task: Dict[str, TerminalBench21RunRequest] = dict()
        if self.config.session_records_dir is not None:
            self.config.session_records_dir.mkdir(parents=True, exist_ok=True)

    def _record_session(self, session_id: str, **fields: Any) -> None:
        """Merge `fields` into the session's JSON record (atomic replace; best effort, never fails a request)."""
        directory = self.config.session_records_dir
        if directory is None:
            return
        path = directory / f"{session_id}.json"
        try:
            record = json.loads(path.read_text()) if path.exists() else {}
        except (OSError, ValueError):
            record = {}
        record.update(fields, session_id=session_id, updated_at=datetime.now(timezone.utc).isoformat())
        try:
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, default=str))
            os.replace(temporary, path)
        except OSError:
            print(f"Could not write the session record {path}: {format_exc()}", file=stderr)

    def _patch_sandbox_provider_options_for_instances(
        self, task_name: str, resources: SandboxResources, provider_options: Dict[str, Any]
    ) -> None:
        # TODO @bxyu-nvidia: These patches may not be necessary eventually, but for now we need them in order for the below instance golden patches to pass.
        tasks_to_increase_initial_resources_for = {
            "terminal-bench/torch-pipeline-parallelism",
            "terminal-bench/torch-tensor-parallelism",
            "terminal-bench/pytorch-model-recovery",
            "terminal-bench/mteb-retrieve",
            "terminal-bench/caffe-cifar-10",
        }
        if task_name in tasks_to_increase_initial_resources_for:
            provider_options["resource_requests"] = {
                "cpu": resources.cpu,
                "memory_mib": resources.memory_mib,
                "disk_gib": resources.disk_gib,
            }

    async def _create_sandbox(self, verify_request: TerminalBench21SeedSessionRequest) -> AsyncSandbox:
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

        provider_options = deepcopy(self.config.sandbox_config.get("provider_options") or {})
        self._patch_sandbox_provider_options_for_instances(
            verify_request.task_name, sandbox_resources, provider_options
        )

        eval_sandbox_spec = SandboxSpec(
            image=verify_request.docker_image,
            ttl_s=self.config.sandbox_config.get("ttl_s", None),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s", None),
            workdir=None,  # Default to container's WORKDIR
            env=env,
            files=dict(),
            metadata=provider_default_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {
                "nemo_gym_agent": self.config.name,
                "instance_id": verify_request.task_name,
            },
            resources=SandboxResources.from_mapping(resources),
            entrypoint=None,
            provider_options=provider_options,
        )
        eval_sandbox = AsyncSandbox(resolved_sandbox_provider)

        async def _run_setup(sandbox: AsyncSandbox) -> None:
            result = await sandbox.exec(
                join(
                    ["bash", "-c", _BULLSEYE_SECURITY_SNAPSHOT_SETUP, "--", "/etc/os-release", "/etc/apt/sources.list"]
                ),
                timeout_s=self.config.evaluation_timeout,
            )
            if result.return_code != 0:
                raise RuntimeError(f"Failed to prepare TerminalBench package sources: {result}")

            result = await sandbox.exec("apt-get update", timeout_s=self.config.evaluation_timeout)
            if result.return_code != 0:
                print(f"Failed to apt-get update: {result}")

        # start_with_setup stops the container if _run_setup raises, instead of
        # leaving it running until its TTL.
        if self.config.prepare_apt_sources:
            await eval_sandbox.start_with_setup(eval_sandbox_spec, _run_setup)
        else:
            await eval_sandbox.start(eval_sandbox_spec)

        return eval_sandbox

    async def seed_session(
        self, request: Request, body: TerminalBench21RunRequest
    ) -> TerminalBench21SeedSessionResponse:
        instruction = []
        for item in body.responses_create_params.input if body.responses_create_params else []:
            if getattr(item, "role", None) == "user":
                content = item.model_dump()["content"]
                instruction.append(
                    content if isinstance(content, str) else "\n".join(part["text"] for part in content)
                )
        provider = resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
        eval_sandbox = await self._create_sandbox(body)
        session_id = request.session[SESSION_ID_KEY]
        self._session_id_to_sandbox[session_id] = eval_sandbox
        self._session_id_to_task[session_id] = body
        self._record_session(
            session_id,
            phase="open",
            request=body.model_dump(exclude={"responses_create_params"}),
            sandbox_id=eval_sandbox._handle.sandbox_id,
            seeded_at=datetime.now(timezone.utc).isoformat(),
        )

        # Keep the handle for OpenCode/Terminus; mini-SWE consumes the structured fields.
        return TerminalBench21SeedSessionResponse(
            sandbox_handle=eval_sandbox._handle.sandbox_id,
            session_id=session_id,
            sandbox_descriptor=await eval_sandbox.serialize(),
            sandbox_provider=provider,
            instruction="\n".join(instruction),
            task_id=body.task_name,
            agent_timeout_sec=body.agent_timeout_sec,
            user=body.agent_user,
        )

    @contextmanager
    def _patch_golden_patch_solve_sh(
        self, task_name: str, local_fpath: Path, patches: Dict[str, List[Tuple[str, str]]]
    ):
        if task_name not in patches or local_fpath.suffix != ".sh":
            yield local_fpath
            return

        content = local_fpath.read_text()
        for old, new in patches[task_name]:
            content = content.replace(old, new)

        with NamedTemporaryFile(mode="w+", suffix=".sh", delete_on_close=False) as temp_file:
            temp_file.write(content)
            temp_file.flush()

            yield temp_file.name

    async def _upload_folder(
        self,
        sandbox: AsyncSandbox,
        local_dirpath: Path,
        target_dirpath: str,
        patches: Dict[str, List[Tuple[str, str]]],
        task_name: Optional[str] = None,
    ) -> None:
        if not local_dirpath.is_absolute():
            local_dirpath = PARENT_DIR / local_dirpath

        # One archive, one upload, one exec: a tests/ folder of hundreds of fixture files used to cost two
        # control-plane calls per file, which is what bounds a resources server at high concurrency.
        with TemporaryDirectory(prefix="tb21-upload-") as scratch:
            archive = Path(scratch) / "folder.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                for local_fpath in sorted(local_dirpath.rglob("*")):
                    if not local_fpath.is_file():
                        continue
                    relative = local_fpath.relative_to(local_dirpath).as_posix()
                    with self._patch_golden_patch_solve_sh(task_name, local_fpath, patches) as new_local_fpath:
                        info = tar.gettarinfo(str(new_local_fpath), arcname=relative)
                        info.mode = local_fpath.stat().st_mode & 0o7777
                        with open(new_local_fpath, "rb") as stream:
                            tar.addfile(info, stream)
            remote_archive = f"{target_dirpath.rstrip('/')}.upload.tar.gz"
            mkdir_result = await sandbox.exec(f"mkdir -p {json.dumps(target_dirpath)}")
            assert mkdir_result.return_code == 0, mkdir_result
            await sandbox.upload(local_path=archive, remote_path=remote_archive)
            unpack = await sandbox.exec(
                f"tar -xzf {json.dumps(remote_archive)} -C {json.dumps(target_dirpath)} && rm -f {json.dumps(remote_archive)}",
                timeout_s=600,
            )
            assert unpack.return_code == 0, unpack

    async def verify(
        self, request: Request, body: TerminalBench21VerifyRequest | TerminalBench21SessionVerifyRequest
    ) -> TerminalBench21VerifyResponse:
        metadata: Dict[str, Any] = {}
        session_id = request.session[SESSION_ID_KEY]
        run_request: Optional[TerminalBench21RunRequest] = None
        agent_failure: Optional[str] = None
        termination_reason: Optional[str] = None
        if isinstance(body, TerminalBench21SessionVerifyRequest):
            if body.session_id != session_id:
                raise HTTPException(409, "Verification session does not match the seeded session cookie")
            run_request = self._session_id_to_task.get(body.session_id)
            if run_request is None:
                raise HTTPException(404, "No seeded TB2 task for this session")
            metadata = body.harness_metadata | {
                "termination": body.termination,
                "agent_started": body.agent_started,
                "agent_timings": body.agent_timings,
            }
            # The agent harness failing (or never starting) is not the model's doing: release the sandbox
            # without grading, as the TB4 resources server does, and mark the sample unusable.
            termination_reason = body.termination.get("reason") if isinstance(body.termination, dict) else None
            if termination_reason in ("infrastructure_error", "cancelled") or not body.agent_started:
                agent_failure = str(
                    (body.termination.get("detail") if isinstance(body.termination, dict) else None)
                    or termination_reason
                    or "Agent did not start"
                )
            body = TerminalBench21VerifyRequest.model_validate(run_request.model_dump() | body.model_dump())
        task_folder = Path(body.task_folder)
        local_task_folder = task_folder if task_folder.is_absolute() else PARENT_DIR / task_folder
        task_limit = run_request.verifier_timeout_sec if run_request is not None else None
        verifier_timeout = float(
            max(
                self.config.verifier_timeout_floor_sec, task_limit or task_verifier_timeout_sec(local_task_folder) or 0
            )
        )
        verifier_env: Dict[str, str] = {}
        agent_user = getattr(body, "agent_user", None)
        if run_request is not None:
            verifier_env = dict(run_request.verifier_env)
            agent_user = run_request.agent_user
        verifier_user = (
            agent_user if self.config.verifier_runs_as_agent_user and not is_root_identity(agent_user) else None
        )

        if self.config.is_verifying_golden_patch:
            if self.config.debug:
                print(f"Creating eval sandbox for {body.task_name}", file=stderr)
            eval_sandbox = await self._create_sandbox(body)
            cwd = (await eval_sandbox.exec("pwd")).stdout.strip()
            await self._upload_folder(
                eval_sandbox, task_folder / "solution", cwd, GOLDEN_PATCH_SOLVE_SH_PATCHES, task_name=body.task_name
            )

            if self.config.debug:
                print(f"Running golden patch for {body.task_name}", file=stderr)
            golden_patch_result = await eval_sandbox.exec(
                f"bash {cwd}/solve.sh",
                timeout_s=self.config.evaluation_timeout,
            )
            golden_patch_output = (golden_patch_result.stderr or "") + (golden_patch_result.stdout or "")
            if self.config.debug:
                print(f"Golden patch output for {body.task_name}: {golden_patch_output}", file=stderr)
        else:
            # Re-use the original sandbox
            eval_sandbox = self._session_id_to_sandbox.pop(session_id)
            self._session_id_to_task.pop(session_id, None)
            golden_patch_output = None

        start_time = time()
        eval_result = None
        test_output = ""
        if agent_failure is not None:
            print(f"Skipping tests for {body.task_name}: agent harness failure: {agent_failure[:300]}", file=stderr)
        else:
            if self.config.debug:
                print(f"Running tests for {body.task_name}", file=stderr)
            try:
                # /tests is uploaded as the image default (root), so the verifier account can read but not change
                # it. `bash /tests/test.sh` then runs as `verifier_user`: the agent's account (as the vendor images
                # declare it), or the image default when the agent ran as root or the switch is off.
                await self._upload_folder(
                    eval_sandbox, task_folder / "tests", "/tests", TEST_SH_PATCHES, body.task_name
                )
                if verifier_user is not None:
                    prepare = await eval_sandbox.exec(
                        f"mkdir -p /logs/verifier && chown -R {shlex_quote(str(verifier_user))} /logs/verifier"
                    )
                    if prepare.return_code != 0:
                        print(f"Failed to hand /logs/verifier to {verifier_user!r}: {prepare}", file=stderr)
                eval_result = await eval_sandbox.exec(
                    "bash /tests/test.sh",
                    timeout_s=verifier_timeout,
                    env=verifier_env or None,
                    user=verifier_user,
                )
                test_output = (eval_result.stderr or "") + (eval_result.stdout or "")
            except:
                print(f"Hit exception running TerminalBench 2.1 tests: {format_exc()}", file=stderr)
                eval_result = None
                test_output = ""
        verification_time_taken = time() - start_time

        if self.config.debug:
            print(f"Test output for {body.task_name}: {test_output}", file=stderr)

        evaluation_completed = False
        reward = 0.0
        if eval_result is not None:
            try:
                with NamedTemporaryFile(mode="w+", suffix=".txt") as temp_file:
                    await eval_sandbox.download("/logs/verifier/reward.txt", temp_file.name)
                    temp_file.seek(0)
                    reward = float(temp_file.read())

                evaluation_completed = True
            except:
                if self.config.debug:
                    print(f"Hit an exception downloading and converting reward: {format_exc()}", file=stderr)

        try:
            await eval_sandbox.stop()
        except:
            print(f"Hit an exception stopping sandbox: {format_exc()}", file=stderr)

        # A missing reward is the verifier's doing, never a model failure: keep the reward at 0 for
        # compatibility but flag the sample so downstream scores exclude it.
        mask_sample = False
        failure_kind = None
        failure_reason = None
        if agent_failure is not None:
            mask_sample = True
            failure_kind = "cancelled" if termination_reason == "cancelled" else "agent_run_error"
            failure_reason = agent_failure
        elif not evaluation_completed:
            mask_sample = True
            failure_kind = "verifier_error"
            failure_reason = (
                "MissingOfficialReward: the verifier left no /logs/verifier/reward.txt"
                if eval_result is not None
                else "VerifierExecutionError: the tests could not be uploaded or run"
            )

        result = TerminalBench21VerifyResponse(
            **body.model_dump(),
            evaluation_completed=evaluation_completed,
            reward=reward,
            mask_sample=mask_sample,
            failure_kind=failure_kind,
            failure_reason=failure_reason,
            infrastructure_error=agent_failure,
            verification_time_taken=verification_time_taken,
            test_output=test_output,
            golden_patch_output=golden_patch_output,
            verifier_user=verifier_user,
            verifier_timeout_sec=verifier_timeout,
        )
        self._record_session(
            session_id,
            phase="closed",
            termination=metadata.get("termination"),
            agent_started=metadata.get("agent_started"),
            verified_response={
                "reward": reward,
                "evaluation_completed": evaluation_completed,
                "mask_sample": mask_sample,
                "failure_kind": failure_kind,
                "failure_reason": failure_reason,
                "infrastructure_error": agent_failure,
                "verification_time_taken": verification_time_taken,
                "test_output_tail": test_output[-4000:],
            },
            verified_at=datetime.now(timezone.utc).isoformat(),
        )
        return TerminalBench21VerifyResponse.model_validate(metadata | result.model_dump()) if metadata else result


if __name__ == "__main__":
    TerminalBench21ResourcesServer.run_webserver()
