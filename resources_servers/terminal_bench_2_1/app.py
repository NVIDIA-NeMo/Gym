# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from copy import deepcopy
from glob import glob
from pathlib import Path, PurePosixPath
from shlex import quote as shlex_quote
from sys import stderr
from tempfile import NamedTemporaryFile
from time import time
from traceback import format_exc
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from fastapi import Request
from pydantic import BaseModel, field_validator

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
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.agent_user import AgentUser, is_root_agent_user, normalize_agent_user
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import SESSION_ID_KEY


class TerminalBench21ResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    is_verifying_golden_patch: bool = False
    evaluation_timeout: Optional[int] = None

    # Sandbox config
    sandbox_provider: str
    sandbox_config: Dict[str, Any]

    debug: bool = False


def uploaded_subdirectories(target_dirpath: str, remote_paths: List[str]) -> List[str]:
    """Directories strictly below ``target_dirpath`` that hold ``remote_paths`` (deduplicated, parents first).

    ``remote_paths`` must come from ``_upload_folder(..., target_dirpath)``, i.e. each is ``f"{target_dirpath}/<rel>"``;
    the directories are the ones its ``mkdir -p`` created or used for nested files, never the upload root itself.
    """
    prefix = f"{target_dirpath}/"
    root = target_dirpath.rstrip("/")
    directories: Dict[str, None] = {}
    for remote_path in remote_paths:
        assert remote_path.startswith(prefix), (remote_path, target_dirpath)
        relative_dir = PurePosixPath(remote_path[len(prefix) :]).parent
        for depth in range(1, len(relative_dir.parts) + 1):
            directories.setdefault(f"{root}/{'/'.join(relative_dir.parts[:depth])}", None)
    return list(directories)


class TerminalBench21SeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str  # @bxyu-nvidia: Just a plain string URI for now for OpenSandbox backend.
    # Echo of the row's agent_user: the resources server is the authority the agent harness reads it from.
    agent_user: AgentUser = None


class TerminalBench21SeedSessionRequest(BaseModel):
    task_name: str
    docker_image: str
    task_folder: str
    # Identity the agent harness runs as inside the task sandbox (mirrors Harbor `task.toml` `[agent] user`).
    # `None` keeps the image default. The verifier always runs as the image default (root on the supported
    # images), so a non-root `agent_user` requires a root-default image on which that account exists.
    # A row value of "root" (or 0) is the explicit per-row image default and beats any lane-level setting.
    agent_user: AgentUser = None

    @field_validator("agent_user", mode="before")
    @classmethod
    def _normalize_agent_user(cls, value: Any) -> Any:
        return normalize_agent_user(value)


class TerminalBench21VerifyRequest(TerminalBench21SeedSessionRequest, BaseVerifyRequest):
    pass


class TerminalBench21VerifyResponse(BaseVerifyResponse):
    evaluation_completed: bool

    # Misc metrics
    verification_time_taken: float

    task_name: str
    test_output: str
    golden_patch_output: Optional[str]
    # Identity the agent ran as (the base response chain drops undeclared request fields, so declare it here
    # to record it in rollouts).
    agent_user: AgentUser = None


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
        await eval_sandbox.start(eval_sandbox_spec)

        result = await eval_sandbox.exec("apt-get update", timeout_s=self.config.evaluation_timeout)
        if result.return_code != 0:
            print(f"Failed to apt-get update: {result}")

        return eval_sandbox

    async def seed_session(
        self, request: Request, body: TerminalBench21SeedSessionRequest
    ) -> TerminalBench21SeedSessionResponse:
        eval_sandbox = await self._create_sandbox(body)
        self._session_id_to_sandbox[request.session[SESSION_ID_KEY]] = eval_sandbox

        return TerminalBench21SeedSessionResponse(
            sandbox_handle=eval_sandbox._handle.sandbox_id, agent_user=body.agent_user
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
    ) -> List[str]:
        """Upload every file under ``local_dirpath`` to ``target_dirpath``; returns the remote paths written."""
        if not local_dirpath.is_absolute():
            local_dirpath = PARENT_DIR / local_dirpath

        remote_paths: List[str] = []
        for file in glob("**", root_dir=str(local_dirpath), recursive=True):
            local_fpath = local_dirpath / file
            if not local_fpath.is_file():
                continue

            target_fpath = f"{target_dirpath}/{file}"
            mkdir_result = await sandbox.exec(f"mkdir -p {Path(target_fpath).parent}")
            assert mkdir_result.return_code == 0, mkdir_result

            with self._patch_golden_patch_solve_sh(task_name, local_fpath, patches) as new_local_fpath:
                await sandbox.upload(local_path=new_local_fpath, remote_path=target_fpath)
            remote_paths.append(target_fpath)

        return remote_paths

    async def verify(self, request: Request, body: TerminalBench21VerifyRequest) -> TerminalBench21VerifyResponse:
        task_folder = Path(body.task_folder)

        if self.config.is_verifying_golden_patch:
            if self.config.debug:
                print(f"Creating eval sandbox for {body.task_name}", file=stderr)
            eval_sandbox = await self._create_sandbox(body)
            cwd = (await eval_sandbox.exec("pwd")).stdout.strip()
            solution_paths = await self._upload_folder(
                eval_sandbox, task_folder / "solution", cwd, GOLDEN_PATCH_SOLVE_SH_PATCHES, task_name=body.task_name
            )
            # The reference solution stands in for the agent, so it runs under the agent identity. Uploads (and the
            # directories `_upload_folder` created for nested files) land owned by the image default; hand them to a
            # non-root agent_user first (mirrors Harbor's agents/installed/base.py chown after upload) so solve.sh
            # can read and modify its own files. One non-recursive chown, parents first, never the upload root.
            if not is_root_agent_user(body.agent_user) and solution_paths:
                chown_targets = uploaded_subdirectories(cwd, solution_paths) + solution_paths
                quoted_paths = " ".join(shlex_quote(path) for path in chown_targets)
                chown_result = await eval_sandbox.exec(f"chown {shlex_quote(str(body.agent_user))} {quoted_paths}")
                if chown_result.return_code != 0:
                    print(f"Failed to chown solution files to {body.agent_user!r}: {chown_result}", file=stderr)

            if self.config.debug:
                print(f"Running golden patch for {body.task_name}", file=stderr)
            golden_patch_result = await eval_sandbox.exec(
                f"bash {cwd}/solve.sh",
                timeout_s=self.config.evaluation_timeout,
                user=body.agent_user,
            )
            golden_patch_output = (golden_patch_result.stderr or "") + (golden_patch_result.stdout or "")
            if self.config.debug:
                print(f"Golden patch output for {body.task_name}: {golden_patch_output}", file=stderr)
        else:
            # Re-use the original sandbox
            eval_sandbox = self._session_id_to_sandbox.pop(request.session[SESSION_ID_KEY])
            golden_patch_output = None

        if self.config.debug:
            print(f"Running tests for {body.task_name}", file=stderr)
        start_time = time()
        try:
            # The verifier execs below (the /tests mkdir + upload and `bash /tests/test.sh`) intentionally carry
            # no user override: they run as the image default (root on the supported images) regardless of
            # `agent_user`, which is why a non-root agent_user requires a root-default image.
            await self._upload_folder(eval_sandbox, task_folder / "tests", "/tests", TEST_SH_PATCHES, body.task_name)
            eval_result = await eval_sandbox.exec(
                "bash /tests/test.sh",
                timeout_s=self.config.evaluation_timeout,
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
