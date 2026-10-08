# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from contextlib import asynccontextmanager, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from shlex import join
from sys import stderr
from tempfile import NamedTemporaryFile
from time import time
from traceback import format_exc
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.episode_types import EpisodeId
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import SESSION_ID_KEY


LOG = logging.getLogger(__name__)


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
    session_close_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)

    # Sandbox config
    sandbox_provider: str
    sandbox_config: Dict[str, Any]

    debug: bool = False


class TerminalBench21SeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str  # @bxyu-nvidia: Just a plain string URI for now for OpenSandbox backend.


class TerminalBench21SeedSessionRequest(BaseModel):
    task_name: str
    docker_image: str
    task_folder: str


class TerminalBench21VerifyRequest(TerminalBench21SeedSessionRequest, BaseVerifyRequest):
    pass


class TerminalBench21VerifyResponse(BaseVerifyResponse):
    evaluation_completed: bool

    # Misc metrics
    verification_time_taken: float

    task_name: str
    test_output: str
    golden_patch_output: Optional[str]


@dataclass
class _ResourcesSessionState:
    """Resources-owned request identity and replay results for one episode.

    EnvironmentServer supplies the session ID and closes it after verification.
    Sandbox ownership is retained separately in ``_session_id_to_sandbox``.
    """

    request: ResourcesSeedSessionRequest
    response: ResourcesSeedSessionResponse | None = None
    verification_request: TerminalBench21VerifyRequest | None = None
    verification_response: TerminalBench21VerifyResponse | None = None


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
    ray_enabled = False
    config: TerminalBench21ResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("Terminal-Bench process-local sessions require num_workers=1")

        self._session_id_to_sandbox: Dict[str, AsyncSandbox] = dict()
        self._session_id_to_state: dict[str, _ResourcesSessionState] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._closed_sessions: dict[str, EpisodeId] = {}

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                for session_id in list(self._session_id_to_sandbox):
                    try:
                        await self._stop_session_sandbox(session_id)
                    except Exception:
                        LOG.exception("Failed to stop Terminal-Bench session %s on shutdown", session_id)

        app.router.lifespan_context = lifespan
        return app

    async def _stop_session_sandbox(self, session_id: str) -> None:
        sandbox = self._session_id_to_sandbox.get(session_id)
        if sandbox is not None:
            async with asyncio.timeout(self.config.session_close_timeout_seconds):
                await sandbox.stop()
            # Keep ownership when stop fails so close can retry.
            self._session_id_to_sandbox.pop(session_id, None)

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        """Stop an owner-managed sandbox, including after failed or lost seed calls."""
        session_id = body.resources_session_id
        async with self._session_locks.setdefault(session_id, asyncio.Lock()):
            closed_episode = self._closed_sessions.get(session_id)
            session = self._session_id_to_state.get(session_id)
            expected = closed_episode or (session.request.episode_id if session is not None else None)
            if expected is not None and expected != body.episode_id:
                raise HTTPException(409, "episode_id does not match the resources session")
            await self._stop_session_sandbox(session_id)
            self._session_id_to_state.pop(session_id, None)
            # A close that arrives before seed also fences a delayed seed request.
            self._closed_sessions[session_id] = body.episode_id
            return ResourcesCloseSessionResponse(resources_session_id=session_id)

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

    async def _create_sandbox(
        self, verify_request: TerminalBench21SeedSessionRequest, *, session_id: str | None = None
    ) -> AsyncSandbox:
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

        if session_id is None:
            # Direct-agent callers have no Resources session to retain failed cleanup.
            await eval_sandbox.start_with_setup(eval_sandbox_spec, _run_setup)
        else:
            # Resources session setup owns cleanup, preserving the error and a handle
            # for close/shutdown retries even if the first stop fails.
            self._session_id_to_sandbox[session_id] = eval_sandbox
            await eval_sandbox.start(eval_sandbox_spec)
            await _run_setup(eval_sandbox)

        return eval_sandbox

    async def seed_session(
        self, request: Request, body: ResourcesSeedSessionRequest | TerminalBench21SeedSessionRequest
    ) -> ResourcesSeedSessionResponse | TerminalBench21SeedSessionResponse:
        if isinstance(body, ResourcesSeedSessionRequest):
            return await self._seed_resources_session(request, body)
        eval_sandbox = await self._create_sandbox(body)
        self._session_id_to_sandbox[request.session[SESSION_ID_KEY]] = eval_sandbox

        return TerminalBench21SeedSessionResponse(sandbox_handle=eval_sandbox._handle.sandbox_id)

    async def _seed_resources_session(
        self, request: Request, body: ResourcesSeedSessionRequest
    ) -> ResourcesSeedSessionResponse:
        """Bind an EnvironmentServer episode to its benchmark-owned sandbox."""
        if self.config.is_verifying_golden_patch:
            raise HTTPException(422, "Golden-patch mode cannot be used with agent sandbox sessions")
        task = TerminalBench21SeedSessionRequest.model_validate(body.task_data)
        task_folder = Path(task.task_folder)
        if not task_folder.is_absolute():
            task_folder = PARENT_DIR / task_folder
        if not (task_folder / "tests/test.sh").is_file():
            raise HTTPException(422, f"Missing local task verifier: {task_folder / 'tests/test.sh'}")
        session_id = body.resources_session_id
        async with self._session_locks.setdefault(session_id, asyncio.Lock()):
            if session_id in self._closed_sessions:
                raise HTTPException(409, "Resources session is already closed")
            session = self._session_id_to_state.get(session_id)
            if session is not None:
                if session.request != body:
                    raise HTTPException(409, "resources_session_id is already bound to a different request")
                if session.response is None or session.verification_request is not None:
                    raise HTTPException(409, "Resources session is no longer available for seeding")
                request.session[SESSION_ID_KEY] = session_id
                return session.response
            session = _ResourcesSessionState(request=body.model_copy(deep=True))
            self._session_id_to_state[session_id] = session
            try:
                sandbox = await self._create_sandbox(task, session_id=session_id)
                working_directory = await sandbox.exec("pwd", timeout_s=30)
                workdir = (working_directory.stdout or "").strip()
                if working_directory.return_code != 0 or not Path(workdir).is_absolute():
                    raise RuntimeError("Could not determine the task sandbox's absolute working directory")
                session.response = ResourcesSeedSessionResponse(
                    resources_session_id=session_id,
                    sandbox_access=SandboxAccess(
                        connection=DirectSandboxConnection(
                            provider_config_ref=self.config.sandbox_provider,
                            descriptor=await sandbox.serialize(),
                        ),
                        workdir=workdir,
                    ),
                )
            except BaseException:
                try:
                    await self._stop_session_sandbox(session_id)
                except Exception:
                    LOG.exception("Failed to stop partially seeded Terminal-Bench session %s", session_id)
                raise
            request.session[SESSION_ID_KEY] = session_id
            return session.response

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

        for file in glob("**", root_dir=str(local_dirpath), recursive=True):
            local_fpath = local_dirpath / file
            if not local_fpath.is_file():
                continue

            target_fpath = f"{target_dirpath}/{file}"
            mkdir_result = await sandbox.exec(f"mkdir -p {Path(target_fpath).parent}")
            assert mkdir_result.return_code == 0, mkdir_result

            with self._patch_golden_patch_solve_sh(task_name, local_fpath, patches) as new_local_fpath:
                await sandbox.upload(local_path=new_local_fpath, remote_path=target_fpath)

    async def verify(self, request: Request, body: TerminalBench21VerifyRequest) -> TerminalBench21VerifyResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        if session_id in self._closed_sessions:
            raise HTTPException(409, "Resources session is already closed")
        if session_id in self._session_id_to_state:
            async with self._session_locks[session_id]:
                session = self._session_id_to_state.get(session_id)
                if session is None or session.response is None:
                    raise HTTPException(409, "Resources session is not available for verification")
                task = TerminalBench21SeedSessionRequest.model_validate(session.request.task_data)
                if task != TerminalBench21SeedSessionRequest.model_validate(body.model_dump()):
                    raise HTTPException(409, "Verification task does not match the seeded task")
                if session.verification_request is not None:
                    if session.verification_request != body:
                        raise HTTPException(409, "Verification request does not match the first request")
                    if session.verification_response is None:
                        # The verifier may have mutated the task before failing.
                        raise HTTPException(503, "Previous verification did not complete; retry the episode")
                else:
                    session.verification_request = body.model_copy(deep=True)
                    session.verification_response = await self._verify(request, body, keep_sandbox_for_close=True)
                # Replay a lost response without grading the mutated task twice.
                return session.verification_response.model_copy(deep=True)
        return await self._verify(request, body, keep_sandbox_for_close=False)

    async def _verify(
        self, request: Request, body: TerminalBench21VerifyRequest, *, keep_sandbox_for_close: bool
    ) -> TerminalBench21VerifyResponse:
        task_folder = Path(body.task_folder)

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
                preserve_background_services=True,
            )
            golden_patch_output = (golden_patch_result.stderr or "") + (golden_patch_result.stdout or "")
            if self.config.debug:
                print(f"Golden patch output for {body.task_name}: {golden_patch_output}", file=stderr)
        else:
            # Re-use the original sandbox
            eval_sandbox = self._session_id_to_sandbox[request.session[SESSION_ID_KEY]]
            golden_patch_output = None

        if self.config.debug:
            print(f"Running tests for {body.task_name}", file=stderr)
        start_time = time()
        try:
            await self._upload_folder(eval_sandbox, task_folder / "tests", "/tests", TEST_SH_PATCHES, body.task_name)
            # Test scripts write rewards here even if their test runner fails to start.
            setup_result = await eval_sandbox.exec("mkdir -p /logs/verifier", timeout_s=self.config.evaluation_timeout)
            if setup_result.return_code != 0:
                raise RuntimeError(f"Failed to prepare TerminalBench verifier output directory: {setup_result}")
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
                    # Providers such as Docker can replace the destination file during download.
                    reward = float(Path(temp_file.name).read_text())

                evaluation_completed = True
            except:
                if self.config.debug:
                    print(f"Hit an exception downloading and converting reward: {format_exc()}", file=stderr)

        if not keep_sandbox_for_close:
            try:
                if self.config.is_verifying_golden_patch:
                    await eval_sandbox.stop()
                else:
                    await self._stop_session_sandbox(request.session[SESSION_ID_KEY])
            except Exception:
                LOG.exception("Failed to stop Terminal-Bench sandbox after legacy verification")

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
