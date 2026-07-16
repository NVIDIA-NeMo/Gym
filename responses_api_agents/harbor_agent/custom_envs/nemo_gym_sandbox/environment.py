# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Harbor environment backed by the NeMo Gym sandbox API.

This adapter lets any Harbor agent (e.g. Terminus-2) execute a task inside a
sandbox managed by ``nemo_gym.sandbox`` — so every provider Gym supports
(``opensandbox``, ``docker``, ``daytona``, ``apptainer``, ``ecs_fargate``, or a
custom registration) becomes a Harbor execution backend without touching the
harbor_agent server code. Select it purely via config:

    harbor_environment_import_path: "responses_api_agents.harbor_agent.\
custom_envs.nemo_gym_sandbox.environment:NemoGymSandboxEnvironment"
    harbor_environment_kwargs:
      sandbox_provider:
        opensandbox:
          connection: {domain: ..., api_key: ..., use_server_proxy: true}

The task's ``docker_image`` must be a pullable image ref (Harbor-side image
builds are not supported here). Logging directories are not mounted
(``is_mounted == False``); Harbor downloads ``/logs`` from the sandbox at the
end of the trial through :meth:`download_dir`.
"""

import shlex
import tarfile
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.paths import EnvironmentPaths

from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxSpec,
    resolve_provider_config,
    resolve_provider_metadata,
    rewrite_image,
)


# The sandbox-side scratch directory used for tar-based directory transfer.
_TRANSFER_DIR = "/tmp"


class NemoGymSandboxEnvironment(BaseEnvironment):
    """Harbor ``BaseEnvironment`` that runs the task in a NeMo Gym sandbox.

    Optional kwargs (via ``harbor_environment_kwargs``):
        sandbox_provider: Required. Single-key mapping ``{provider_name: kwargs}``
            (optionally wrapped with a reserved ``default_metadata`` key, same
            shape as the shipped ``sandbox:`` config blocks).
        sandbox_metadata: Extra ``SandboxSpec.metadata`` entries.
        sandbox_env: Extra environment variables set in the sandbox.
        sandbox_ttl_s: Sandbox server-side TTL safety net (default 21600).
        sandbox_ready_timeout_s: Create/readiness timeout incl. image pull
            (default 900).
        default_exec_timeout_s: Timeout applied when Harbor calls ``exec``
            without ``timeout_sec`` (default 300). This is a *per-command*
            bound, not a trial bound: it caps how long a single sandbox exec
            (submit + background poll) may run before the provider gives up and
            raises ``TimeoutError`` (which Terminus-2 handles as a command
            timeout). Terminus-2 only omits ``timeout_sec`` for its own short
            tmux plumbing (``send-keys``/``capture-pane`` are sub-second; blocked
            waits self-bound at ``timeout <=180s tmux wait done``), so 300s is
            ample headroom while still bounding a wedged exec on a hostile
            cross-cluster path. Agent commands pass their own ``timeout_sec``
            (``command.duration_sec``) and are unaffected. Do NOT set this to a
            trial-length value (e.g. hours): a value that large lets a single
            stuck exec hang the whole task until the trial deadline.
        exec_shell: Shell prefix wrapped around every Harbor-issued command
            (default ``"bash -ic"``, matching Harbor's docker/daytona
            backends). Set to null to run commands verbatim.
        image_rewrites: Ordered ``[{from: ..., to: ...}]`` prefix rewrites
            applied to the task's ``docker_image`` (see
            ``nemo_gym.sandbox.rewrite_image``).
        workdir: Container working directory override (defaults to the image's
            own WORKDIR).
        allow_unenforced_internet_isolation: Accept tasks that request
            ``allow_internet = false`` even though this environment cannot
            enforce network isolation (default False). Each affected trial
            logs a prominent warning.
    """

    def __init__(
        self,
        *args,
        sandbox_provider: Optional[Mapping[str, Any]] = None,
        sandbox_metadata: Optional[Mapping[str, Any]] = None,
        sandbox_env: Optional[Mapping[str, str]] = None,
        sandbox_ttl_s: Optional[float] = 21600,
        sandbox_ready_timeout_s: Optional[float] = 900,
        default_exec_timeout_s: Optional[float] = 300,
        exec_shell: Optional[str] = "bash -ic",
        image_rewrites: Optional[list[Mapping[str, str]]] = None,
        workdir: Optional[str] = None,
        allow_unenforced_internet_isolation: bool = False,
        **kwargs,
    ):
        # Set before super().__init__: the base constructor runs the
        # _validate_* hooks, which read these.
        self._sandbox_provider = sandbox_provider
        self._sandbox_metadata = dict(sandbox_metadata or {})
        self._sandbox_env = {str(k): str(v) for k, v in dict(sandbox_env or {}).items()}
        self._sandbox_ttl_s = sandbox_ttl_s
        self._sandbox_ready_timeout_s = sandbox_ready_timeout_s
        self._default_exec_timeout_s = default_exec_timeout_s
        self._exec_shell = exec_shell
        self._image_rewrites = [dict(rewrite) for rewrite in (image_rewrites or [])]
        self._workdir = workdir
        self._allow_unenforced_internet_isolation = allow_unenforced_internet_isolation
        self._sandbox: Optional[AsyncSandbox] = None

        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> EnvironmentType:
        # Harbor only calls this from validation error messages. The pinned
        # harbor release has no member for external environments, so fall back
        # to DOCKER purely for display.
        return getattr(EnvironmentType, "NEMO_GYM_SANDBOX", EnvironmentType.DOCKER)

    @property
    def is_mounted(self) -> bool:
        return False

    @property
    def supports_gpus(self) -> bool:
        return False

    @property
    def can_disable_internet(self) -> bool:
        return self._allow_unenforced_internet_isolation

    def _validate_definition(self):
        if not self._sandbox_provider:
            raise ValueError(
                "NemoGymSandboxEnvironment requires harbor_environment_kwargs.sandbox_provider "
                "({provider_name: kwargs})."
            )
        # Fails fast on malformed provider blocks (e.g. multiple provider keys).
        resolve_provider_config(self._sandbox_provider)
        if not self.task_env_config.docker_image:
            raise ValueError(
                f"Task {self.environment_name!r} does not define environment.docker_image; "
                "NemoGymSandboxEnvironment cannot build images from a Dockerfile."
            )

    def _validate_internet_config(self):
        if not self.task_env_config.allow_internet and not self.can_disable_internet:
            raise ValueError(
                f"Task {self.environment_name!r} requires allow_internet=false, which "
                "NemoGymSandboxEnvironment cannot enforce. Set "
                "harbor_environment_kwargs.allow_unenforced_internet_isolation=true to run "
                "the task anyway (without isolation)."
            )

    @property
    def _resolved_image(self) -> str:
        return rewrite_image(self.task_env_config.docker_image, self._image_rewrites)

    def _build_spec(self) -> SandboxSpec:
        config = self.task_env_config
        resources: dict[str, Any] = {}
        if config.cpus:
            resources["cpu"] = float(config.cpus)
        if config.memory_mb:
            resources["memory_mib"] = int(config.memory_mb)
        if config.storage_mb:
            resources["disk_gib"] = max(1, round(config.storage_mb / 1024))
        if config.gpus:
            resources["gpu"] = int(config.gpus)

        metadata = {
            "harbor-session": self.session_id,
            "harbor-task": self.environment_name,
            **resolve_provider_metadata(self._sandbox_provider),
            **self._sandbox_metadata,
        }

        return SandboxSpec(
            image=self._resolved_image,
            ttl_s=self._sandbox_ttl_s,
            ready_timeout_s=self._sandbox_ready_timeout_s,
            workdir=self._workdir,
            env=self._sandbox_env,
            metadata=metadata,
            resources=resources,
        )

    async def start(self, force_build: bool) -> None:
        if force_build:
            self.logger.warning(
                "force_build is not supported by NemoGymSandboxEnvironment; using the task's prebuilt image %r.",
                self._resolved_image,
            )
        if not self.task_env_config.allow_internet:
            self.logger.warning(
                "Task %r requests allow_internet=false but NemoGymSandboxEnvironment does "
                "not enforce network isolation; the sandbox keeps cluster-default egress.",
                self.environment_name,
            )

        sandbox = AsyncSandbox(
            resolve_provider_config(self._sandbox_provider),
            self._build_spec(),
        )
        await sandbox.start()
        self._sandbox = sandbox

        # /logs/{agent,verifier} are bind mounts in Harbor's Docker backend; here
        # they must exist inside the sandbox so the agent and verifier can write
        # to them (e.g. the verifier tees test output under /logs/verifier).
        log_dirs = f"{EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir}"
        result = await self._sandbox.exec(f"mkdir -p {log_dirs}", timeout_s=60)
        if result.return_code != 0:
            raise RuntimeError(
                f"Failed to create log directories in sandbox: {result.stderr or result.stdout or '<no output>'}"
            )

    def _require_sandbox(self) -> AsyncSandbox:
        if self._sandbox is None:
            raise RuntimeError("Sandbox is not running; call start() first.")
        return self._sandbox

    async def stop(self, delete: bool):
        if self._sandbox is None:
            return
        if not delete:
            # Remote sandboxes are single-use; keeping them alive after the trial
            # would leak cluster resources, so `delete=False` is intentionally not
            # honored here.
            self.logger.debug(
                "delete=False is ignored by NemoGymSandboxEnvironment; the sandbox is always terminated on stop()."
            )
        sandbox, self._sandbox = self._sandbox, None
        await sandbox.stop()

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        timeout_s = timeout_sec if timeout_sec is not None else self._default_exec_timeout_s
        # Match Harbor's docker/daytona backends, which run every agent/verifier
        # command through an interactive bash so .bashrc-based task setups
        # (conda, pyenv, custom PATH) behave identically.
        if self._exec_shell:
            command = f"{self._exec_shell} {shlex.quote(command)}"
        result = await self._require_sandbox().exec(
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
        )
        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.return_code,
        )

    async def upload_file(self, source_path: Path | str, target_path: str):
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        sandbox = self._require_sandbox()
        parent = str(PurePosixPath(target_path).parent)
        if parent and parent != ".":
            await sandbox.exec(f"mkdir -p {shlex.quote(parent)}", timeout_s=60)
        await sandbox.upload(source, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        source = Path(source_dir)
        if not source.exists():
            raise FileNotFoundError(f"Source directory not found: {source}")
        sandbox = self._require_sandbox()

        remote_tar = f"{_TRANSFER_DIR}/.nemo-gym-upload-{uuid.uuid4().hex}.tar.gz"
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_tar = Path(tmp_dir) / "upload.tar.gz"
            with tarfile.open(local_tar, "w:gz") as tar:
                # Archive the *contents* of source_dir so they land directly in
                # target_dir (Harbor's upload_dir contract).
                tar.add(source, arcname=".")
            await sandbox.upload(local_tar, remote_tar)

        quoted_target = shlex.quote(target_dir)
        quoted_tar = shlex.quote(remote_tar)
        result = await sandbox.exec(
            f"mkdir -p {quoted_target} && tar -xzf {quoted_tar} -C {quoted_target}; "
            f"status=$?; rm -f {quoted_tar}; exit $status",
            timeout_s=600,
        )
        if result.return_code != 0:
            self.logger.warning(
                "tar-based upload_dir failed (rc=%s, stderr=%r); falling back to per-file upload.",
                result.return_code,
                (result.stderr or "")[:500],
            )
            await self._upload_dir_file_by_file(source, target_dir)

    async def _upload_dir_file_by_file(self, source: Path, target_dir: str):
        sandbox = self._require_sandbox()
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            remote_path = str(PurePosixPath(target_dir) / PurePosixPath(*relative.parts))
            if path.is_dir():
                await sandbox.exec(f"mkdir -p {shlex.quote(remote_path)}", timeout_s=60)
            elif path.is_file():
                await self.upload_file(path, remote_path)

    async def download_file(self, source_path: str, target_path: Path | str):
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        await self._require_sandbox().download(source_path, target)

    async def download_dir(self, source_dir: str, target_dir: Path | str):
        sandbox = self._require_sandbox()
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        remote_tar = f"{_TRANSFER_DIR}/.nemo-gym-download-{uuid.uuid4().hex}.tar.gz"
        quoted_source = shlex.quote(source_dir)
        quoted_tar = shlex.quote(remote_tar)
        result = await sandbox.exec(
            f"tar -czf {quoted_tar} -C {quoted_source} .",
            timeout_s=600,
        )
        if result.return_code != 0:
            await sandbox.exec(f"rm -f {quoted_tar}", timeout_s=60)
            self.logger.warning(
                "tar-based download_dir failed (rc=%s, stderr=%r); falling back to per-file download.",
                result.return_code,
                (result.stderr or "")[:500],
            )
            await self._download_dir_file_by_file(source_dir, target)
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            local_tar = Path(tmp_dir) / "download.tar.gz"
            try:
                await sandbox.download(remote_tar, local_tar)
            finally:
                await sandbox.exec(f"rm -f {quoted_tar}", timeout_s=60)
            with tarfile.open(local_tar, "r:gz") as tar:
                tar.extractall(target, filter="data")

    async def _download_dir_file_by_file(self, source_dir: str, target: Path):
        sandbox = self._require_sandbox()
        listing = await sandbox.exec(
            f"find {shlex.quote(source_dir)} -type f",
            timeout_s=120,
        )
        if listing.return_code != 0:
            raise RuntimeError(
                f"Failed to list sandbox directory {source_dir!r}: {listing.stderr or listing.stdout or '<no output>'}"
            )
        source_root = PurePosixPath(source_dir)
        for line in (listing.stdout or "").splitlines():
            remote_path = line.strip()
            if not remote_path:
                continue
            relative = PurePosixPath(remote_path).relative_to(source_root)
            await self.download_file(remote_path, target / Path(*relative.parts))
