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

Makes every provider ``nemo_gym.sandbox`` supports a Harbor execution backend,
selected purely via config:

    harbor_environment:
      import_path: "responses_api_agents.harbor_agent_general.\
custom_envs.nemo_gym_sandbox.environment:NemoGymSandboxEnvironment"
      kwargs:
        sandbox_provider:
          opensandbox:
            connection: {domain: ..., api_key: ..., use_server_proxy: true}

The task's ``docker_image`` must be a pullable image ref; Harbor-side image
builds are not supported. Logging directories are not mounted, so Harbor
downloads ``/logs`` at the end of the trial via :meth:`download_dir`.
"""

import json
import re
import shlex
import tarfile
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fnmatch import fnmatch
from pathlib import Path, PurePath, PurePosixPath
from typing import Any, Mapping, Optional

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.definition import should_upload_environment_dir
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import NetworkMode
from harbor.models.trial.paths import EnvironmentPaths
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxSpec,
    resolve_provider_config,
    resolve_provider_metadata,
    rewrite_image,
)


# The sandbox-side scratch directory used for ordinary tar-based directory transfer.
_TRANSFER_DIR = "/tmp"
_SHARED_WORKSPACE_MARKER = ".nemo-gym-shared-workspace.json"
_SHARED_WORKSPACE_SCHEMA_VERSION = 1
_SHARED_WORKSPACE_CLEANUP_MOUNT = "/nemo-gym-workspace-cleanup"
_SANDBOX_TEMPLATE_PATTERN = re.compile(r"\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}")
_SUPPORTED_SANDBOX_TEMPLATE_NAMES = frozenset({"context_id", "environment_name", "session_id", "task_id", "task_name"})


def validate_sandbox_template_placeholders(
    value: Any,
    *,
    context: str = "sandbox configuration",
) -> None:
    """Reject template names the sandbox environment cannot render."""

    unsupported: set[str] = set()

    def collect(item: Any) -> None:
        if isinstance(item, str):
            unsupported.update(
                match.group("name")
                for match in _SANDBOX_TEMPLATE_PATTERN.finditer(item)
                if match.group("name") not in _SUPPORTED_SANDBOX_TEMPLATE_NAMES
            )
            return
        if isinstance(item, Mapping):
            for key, nested in item.items():
                collect(key)
                collect(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                collect(nested)

    collect(value)
    if unsupported:
        rendered_unsupported = ", ".join(f"{{{name}}}" for name in sorted(unsupported))
        rendered_supported = ", ".join(f"{{{name}}}" for name in sorted(_SUPPORTED_SANDBOX_TEMPLATE_NAMES))
        raise ValueError(
            f"Unsupported template placeholder(s) in {context}: {rendered_unsupported}. "
            f"Supported placeholders: {rendered_supported}."
        )


class SandboxPathCopy(BaseModel):
    """Copy a sandbox-visible directory into the task workspace before setup."""

    model_config = ConfigDict(extra="forbid")

    source: str
    destination: str

    @model_validator(mode="after")
    def validate_paths(self) -> "SandboxPathCopy":
        for field_name, value in (("source", self.source), ("destination", self.destination)):
            path = PurePosixPath(value)
            if (
                not path.is_absolute()
                or path == PurePosixPath("/")
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise ValueError(
                    f"sandbox_path_copies.{field_name} must be an absolute non-root path without "
                    f"'.' or '..' components (got {value!r})."
                )
        if PurePosixPath(self.source) == PurePosixPath(self.destination):
            raise ValueError("sandbox_path_copies source and destination must differ.")
        return self


class SandboxPathSymlink(BaseModel):
    """Expose one sandbox-visible path through a stable compatibility path."""

    model_config = ConfigDict(extra="forbid")

    source: str
    destination: str

    @model_validator(mode="after")
    def validate_paths(self) -> "SandboxPathSymlink":
        for field_name, value in (("source", self.source), ("destination", self.destination)):
            path = PurePosixPath(value)
            if (
                not path.is_absolute()
                or path == PurePosixPath("/")
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise ValueError(
                    "sandbox_path_symlinks."
                    f"{field_name} must be an absolute non-root path without '.' or '..' components "
                    f"(got {value!r})."
                )
        if PurePosixPath(self.source) == PurePosixPath(self.destination):
            raise ValueError("sandbox_path_symlinks source and destination must differ.")
        return self


class SharedWorkspaceHostConfig(BaseModel, extra="forbid"):
    """Physical host path for one direct shared workspace."""

    path: str

    @model_validator(mode="after")
    def validate_path(self) -> "SharedWorkspaceHostConfig":
        path = PurePosixPath(self.path)
        if not path.is_absolute() or path == PurePosixPath("/") or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError(
                "shared_workspace.volume.host.path must be an absolute non-root path without "
                f"'.' or '..' components (got {self.path!r})"
            )
        if path.name != "{context_id}":
            raise ValueError("shared_workspace.volume.host.path must end with the {context_id} placeholder")
        return self


class SharedWorkspaceVolumeConfig(BaseModel):
    """Provider volume mounted read-write for policy and read-only for verifier."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    host: SharedWorkspaceHostConfig
    mount_path: str = Field(alias="mountPath")

    @model_validator(mode="after")
    def validate_volume(self) -> "SharedWorkspaceVolumeConfig":
        if not self.name:
            raise ValueError("shared_workspace.volume.name must not be empty")
        mount_path = PurePosixPath(self.mount_path)
        if (
            not mount_path.is_absolute()
            or mount_path == PurePosixPath("/")
            or any(part in {"", ".", ".."} for part in mount_path.parts)
        ):
            raise ValueError(
                "shared_workspace.volume.mountPath must be an absolute non-root path without "
                f"'.' or '..' components (got {self.mount_path!r})"
            )
        self.mount_path = str(mount_path)
        return self


class SharedWorkspaceConfig(BaseModel, extra="forbid"):
    """Direct EFS-backed policy workspace shared with a separate verifier."""

    volume: SharedWorkspaceVolumeConfig
    handoff_timeout_s: float = 600
    cleanup_timeout_s: float = 600
    cleanup_ttl_s: float = 900
    cleanup_entrypoint: list[str] = Field(default_factory=lambda: ["tail", "-f", "/dev/null"])

    @model_validator(mode="after")
    def validate_timeouts(self) -> "SharedWorkspaceConfig":
        for field_name in ("handoff_timeout_s", "cleanup_timeout_s", "cleanup_ttl_s"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"shared_workspace.{field_name} must be positive")
        if not self.cleanup_entrypoint:
            raise ValueError("shared_workspace.cleanup_entrypoint must not be empty")
        return self


@dataclass(frozen=True)
class SharedWorkspaceMarker:
    """Host-side capability authorizing one immutable shared workspace handoff."""

    context_id: str
    source: str
    host_path: str
    exclude: tuple[str, ...]
    sha256: str
    source_environment_stopped: bool = False
    consumed: bool = False

    def to_json(self) -> str:
        return (
            json.dumps(
                {
                    "schema_version": _SHARED_WORKSPACE_SCHEMA_VERSION,
                    "context_id": self.context_id,
                    "source": self.source,
                    "host_path": self.host_path,
                    "exclude": list(self.exclude),
                    "sha256": self.sha256,
                    "source_environment_stopped": self.source_environment_stopped,
                    "consumed": self.consumed,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    @classmethod
    def from_path(cls, path: Path) -> "SharedWorkspaceMarker":
        raw = json.loads(path.read_text())
        expected = {
            "schema_version",
            "context_id",
            "source",
            "host_path",
            "exclude",
            "sha256",
            "source_environment_stopped",
            "consumed",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError(f"Invalid shared workspace marker fields in {path}")
        if raw["schema_version"] != _SHARED_WORKSPACE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported shared workspace marker schema in {path}")
        for field_name in ("context_id", "source", "host_path", "sha256"):
            if not isinstance(raw[field_name], str) or not raw[field_name]:
                raise ValueError(f"Invalid {field_name} in shared workspace marker {path}")
        if not isinstance(raw["exclude"], list) or any(not isinstance(value, str) for value in raw["exclude"]):
            raise ValueError(f"Invalid exclude in shared workspace marker {path}")
        if len(raw["sha256"]) != 64 or any(character not in "0123456789abcdef" for character in raw["sha256"]):
            raise ValueError(f"Invalid sha256 in shared workspace marker {path}")
        for field_name in ("source_environment_stopped", "consumed"):
            if not isinstance(raw[field_name], bool):
                raise ValueError(f"Invalid {field_name} in shared workspace marker {path}")
        return cls(
            context_id=raw["context_id"],
            source=raw["source"],
            host_path=raw["host_path"],
            exclude=tuple(raw["exclude"]),
            sha256=raw["sha256"],
            source_environment_stopped=raw["source_environment_stopped"],
            consumed=raw["consumed"],
        )


def _cpu_pin_prefix(width: int) -> str:
    """Shell prefix pinning the following command to a random contiguous
    ``width``-core block.

    Containers see the host's full core count (cgroup cpu limits don't shrink
    ``nproc``), so build tools fan out and get CFS throttled. A random block
    also spreads co-resident sandboxes instead of stacking them on cores 0..n.

    Must be POSIX sh, and fails open: no ``taskset``, ``nproc`` <= width, or
    unreadable urandom leaves the command unpinned.
    """
    return (
        f"__osb_w={width}; __osb_n=$(nproc 2>/dev/null || echo 0); "
        f'if [ "$__osb_n" -gt "$__osb_w" ] && command -v taskset >/dev/null 2>&1; then '
        f'__osb_s=$(( $(od -An -N2 -tu2 /dev/urandom | tr -d " ") % (__osb_n - __osb_w) )); '
        f'__osb_pin="taskset -c $__osb_s-$((__osb_s + __osb_w - 1))"; '
        f'else __osb_pin=""; fi; $__osb_pin'
    )


class NemoGymSandboxEnvironment(BaseEnvironment):
    """Harbor ``BaseEnvironment`` that runs the task in a NeMo Gym sandbox.

    Optional kwargs (via ``harbor_environment_kwargs``):
        sandbox_provider: Required. Single-key mapping ``{provider_name: kwargs}``
            (optionally wrapped with a reserved ``default_metadata`` key, same
            shape as the shipped ``sandbox:`` config blocks).
        sandbox_metadata: Extra ``SandboxSpec.metadata`` entries.
        sandbox_provider_options: ``SandboxSpec.provider_options`` passed through
            to the provider, e.g. ``resource_requests`` to schedule sandboxes
            below their resource limits. String values may contain
            ``{context_id}``, ``{environment_name}``, ``{task_name}``,
            ``{task_id}``, or ``{session_id}``; these are expanded per trial.
            ``{task_name}`` is an alias for ``{environment_name}``.
        environment_upload_excludes: Relative paths omitted when Harbor uploads
            the task's ``environment/`` directory. Use this only when the image
            or a sandbox volume supplies those paths, e.g. ``["data"]`` with a
            task-specific volume mounted at ``/data`` and exposed through an
            ``/app/data`` compatibility link.
        sandbox_path_copies: Ordered ``[{source: ..., destination: ...}]``
            directory copies performed inside the sandbox before Harbor uploads
            the task environment. Strings support the same per-trial placeholders
            as ``sandbox_provider_options``. This lets a read-only dataset mount
            seed a writable workspace without transferring the data through the
            OpenSandbox API.
        sandbox_path_copy_timeout_s: Timeout for each sandbox-local directory
            copy (default 1200).
        sandbox_path_symlinks: Ordered ``[{source: ..., destination: ...}]``
            compatibility links prepared after provider volumes are mounted and
            before Harbor uploads the task environment. A separate verifier
            validates links inside its read-only shared workspace instead of
            modifying them. Strings support the same per-trial placeholders as
            ``sandbox_provider_options``.
        shared_workspace: Optional direct shared-workspace volume. Its physical
            host path must end in ``{context_id}``. The policy receives the
            volume read-write at ``volume.mountPath`` and the separate verifier
            receives the same volume read-only. Harbor passes only a local
            integrity marker between roles; policy artifacts are never copied.
        sandbox_env: Extra environment variables set in the sandbox.
        sandbox_ttl_s: Sandbox server-side TTL safety net (default 21600).
        sandbox_ready_timeout_s: Create/readiness timeout incl. image pull
            (default 900).
        default_exec_timeout_s: Per-command bound applied when Harbor calls
            ``exec`` without ``timeout_sec`` (default 1800), not a trial bound.

            Do not set this small. Harbor's verifier never passes
            ``timeout_sec``, so this is what bounds verification for every
            benchmark, and benchmarks routinely declare multi-minute budgets.
            A default below the task's own budget silently truncates
            verification and scores the task 0 rather than failing loudly.
        exec_shell: Shell prefix wrapped around every Harbor-issued command
            (default ``"bash -ic"``, matching Harbor's docker/daytona
            backends). Set to null to run commands verbatim.
        cpu_pin_enabled: Pin every Harbor-issued command to a random
            contiguous core block sized by the task's cpu count (default
            False). Terminus-2's tmux server is itself launched via ``exec``,
            so the whole agent session inherits the affinity. See
            ``_cpu_pin_prefix``.
        image_rewrites: Ordered ``[{from: ..., to: ...}]`` prefix rewrites
            applied to the task's ``docker_image`` (see
            ``nemo_gym.sandbox.rewrite_image``).
        image_override: Optional image replacing the task's ``docker_image``.
            Prefer a digest-pinned reference for reproducible runs.
        entrypoint: Optional command replacing the sandbox image's startup
            command. This is useful when Harbor starts task services later.
        workdir: Container working directory override (defaults to the image's
            own WORKDIR).
        allow_unenforced_internet_isolation: Accept tasks that request
            ``network_mode = "no-network"`` even though this environment cannot
            enforce network isolation (default False). Each affected trial
            logs a prominent warning.
    """

    def __init__(
        self,
        *args,
        sandbox_provider: Optional[Mapping[str, Any]] = None,
        sandbox_metadata: Optional[Mapping[str, Any]] = None,
        sandbox_provider_options: Optional[Mapping[str, Any]] = None,
        environment_upload_excludes: Optional[Sequence[str]] = None,
        sandbox_path_copies: Optional[Sequence[Mapping[str, str]]] = None,
        sandbox_path_copy_timeout_s: Optional[float] = 1200,
        sandbox_path_symlinks: Optional[Sequence[Mapping[str, str]]] = None,
        shared_workspace: Optional[Mapping[str, Any] | SharedWorkspaceConfig] = None,
        sandbox_env: Optional[Mapping[str, str]] = None,
        sandbox_ttl_s: Optional[float] = 21600,
        sandbox_ready_timeout_s: Optional[float] = 900,
        default_exec_timeout_s: Optional[float] = 1800,
        exec_shell: Optional[str] = "bash -ic",
        cpu_pin_enabled: bool = False,
        image_rewrites: Optional[list[Mapping[str, str]]] = None,
        image_override: Optional[str] = None,
        entrypoint: Optional[list[str]] = None,
        workdir: Optional[str] = None,
        allow_unenforced_internet_isolation: bool = False,
        **kwargs,
    ):
        # Set before super().__init__: the base constructor runs the
        # _validate_* hooks, which read these.
        self._sandbox_provider = sandbox_provider
        self._sandbox_metadata = dict(sandbox_metadata or {})
        self._sandbox_provider_options = dict(sandbox_provider_options or {})
        self._environment_upload_excludes = self._validate_upload_excludes(environment_upload_excludes or ())
        self._sandbox_path_copies = tuple(SandboxPathCopy.model_validate(item) for item in (sandbox_path_copies or ()))
        self._sandbox_path_copy_timeout_s = sandbox_path_copy_timeout_s
        self._sandbox_path_symlinks = tuple(
            SandboxPathSymlink.model_validate(item) for item in (sandbox_path_symlinks or ())
        )
        validate_sandbox_template_placeholders(shared_workspace, context="shared_workspace")
        self._shared_workspace = (
            shared_workspace
            if isinstance(shared_workspace, SharedWorkspaceConfig)
            else SharedWorkspaceConfig.model_validate(shared_workspace)
            if shared_workspace is not None
            else None
        )
        self._shared_workspace_markers: set[Path] = set()
        self._shared_workspace_cleanup_required = False
        self._sandbox_env = {str(k): str(v) for k, v in dict(sandbox_env or {}).items()}
        self._sandbox_ttl_s = sandbox_ttl_s
        self._sandbox_ready_timeout_s = sandbox_ready_timeout_s
        self._default_exec_timeout_s = default_exec_timeout_s
        self._exec_shell = exec_shell
        self._cpu_pin_enabled = cpu_pin_enabled
        self._image_rewrites = [dict(rewrite) for rewrite in (image_rewrites or [])]
        self._image_override = image_override
        self._entrypoint = list(entrypoint) if entrypoint is not None else None
        self._workdir = workdir
        self._allow_unenforced_internet_isolation = allow_unenforced_internet_isolation
        self._sandbox: Optional[AsyncSandbox] = None

        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> EnvironmentType:
        # Only used in validation error messages, and the pinned harbor release
        # has no member for external environments, so DOCKER is display-only.
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

    @property
    def _resolved_image(self) -> str:
        if self._image_override is not None:
            return self._image_override
        return rewrite_image(self.task_env_config.docker_image, self._image_rewrites)

    @staticmethod
    def _validate_upload_excludes(excludes: Sequence[str]) -> tuple[PurePosixPath, ...]:
        normalized: list[PurePosixPath] = []
        for value in excludes:
            path = PurePosixPath(str(value))
            if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
                raise ValueError(
                    "environment_upload_excludes entries must be non-empty relative paths "
                    f"without '.' or '..' components (got {value!r})."
                )
            normalized.append(path)
        return tuple(normalized)

    def _render_provider_option_templates(self, value: Any) -> Any:
        validate_sandbox_template_placeholders(value, context="sandbox provider options")
        task_id = self.environment_name.rsplit("__", 1)[-1]
        replacements = {
            "context_id": str(self.context_id),
            "environment_name": self.environment_name,
            "task_name": self.environment_name,
            "task_id": task_id,
            "session_id": self.session_id,
        }

        def render(item: Any) -> Any:
            if isinstance(item, str):
                return _SANDBOX_TEMPLATE_PATTERN.sub(
                    lambda match: replacements[match.group("name")],
                    item,
                )
            if isinstance(item, Mapping):
                return {render(key): render(nested) for key, nested in item.items()}
            if isinstance(item, list):
                return [render(nested) for nested in item]
            if isinstance(item, tuple):
                return tuple(render(nested) for nested in item)
            return item

        return render(value)

    def _is_environment_upload(self, source: Path) -> bool:
        return source.resolve() == Path(self.environment_dir).resolve()

    @property
    def _is_separate_verifier(self) -> bool:
        # Harbor assigns separate verifier environments a stable
        # ``<trial>__verifier__<step>`` session id.
        return "__verifier__" in self.session_id

    def _is_upload_excluded(self, relative: PurePosixPath) -> bool:
        return any(
            relative == excluded or excluded in relative.parents for excluded in self._environment_upload_excludes
        )

    def _shared_workspace_volume(self, *, read_only: bool) -> dict[str, Any]:
        config = self._shared_workspace
        if config is None:
            raise RuntimeError("Shared workspace is not configured")
        volume = self._render_provider_option_templates(config.volume.model_dump(by_alias=True))
        volume["readOnly"] = read_only
        return volume

    def _provider_options_for_role(self) -> dict[str, Any]:
        rendered = self._render_provider_option_templates(self._sandbox_provider_options)
        if not isinstance(rendered, Mapping):
            raise ValueError("sandbox_provider_options must be a mapping")
        options = dict(rendered)
        if self._shared_workspace is None:
            return options

        configured_volumes = options.get("volumes", [])
        if not isinstance(configured_volumes, list):
            raise ValueError("sandbox_provider_options.volumes must be a list")
        shared_volume = self._shared_workspace_volume(read_only=self._is_separate_verifier)
        shared_name = shared_volume["name"]
        shared_mount = shared_volume["mountPath"]
        for volume in configured_volumes:
            if not isinstance(volume, Mapping):
                raise ValueError("sandbox_provider_options.volumes entries must be mappings")
            mount_path = volume.get("mountPath", volume.get("mount_path"))
            if volume.get("name") == shared_name:
                raise ValueError(f"sandbox_provider_options.volumes duplicates shared workspace name {shared_name!r}")
            if mount_path == shared_mount:
                raise ValueError(
                    f"sandbox_provider_options.volumes duplicates shared workspace mount {shared_mount!r}"
                )
        options["volumes"] = [shared_volume, *configured_volumes]
        return options

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
            "harbor-role": "verifier" if self._is_separate_verifier else "agent",
            **resolve_provider_metadata(self._sandbox_provider),
            **self._sandbox_metadata,
        }

        return SandboxSpec(
            image=self._resolved_image,
            ttl_s=self._sandbox_ttl_s,
            ready_timeout_s=self._sandbox_ready_timeout_s,
            workdir=self._workdir if self._workdir is not None else self.task_env_config.workdir,
            env={**self._startup_env(), **self._sandbox_env},
            metadata=metadata,
            resources=resources,
            entrypoint=self._entrypoint,
            provider_options=self._provider_options_for_role(),
        )

    async def _upload_environment_dir_after_start(self) -> None:
        if not self._is_separate_verifier:
            await super()._upload_environment_dir_after_start()
            return

        # In Harbor separate-verifier mode ``environment_dir`` is the hidden
        # tests directory, not the task's policy environment directory. Local
        # Harbor backends build this context as the verifier image; prebuilt
        # OpenSandbox images need the equivalent files uploaded to /tests.
        if not should_upload_environment_dir(
            self.environment_dir,
            docker_image=self.task_env_config.docker_image,
        ):
            return
        self.logger.debug("Uploading separate verifier context to /tests")
        await self.upload_dir(self.environment_dir, str(EnvironmentPaths.tests_dir))

    async def _copy_sandbox_paths(self) -> None:
        sandbox = self._require_sandbox()
        for configured_copy in self._sandbox_path_copies:
            rendered = SandboxPathCopy.model_validate(
                self._render_provider_option_templates(configured_copy.model_dump())
            )
            source = shlex.quote(rendered.source)
            destination = shlex.quote(rendered.destination)
            result = await sandbox.exec(
                f"test -d {source} && mkdir -p {destination} && cp -a -- {source}/. {destination}/",
                cwd="/",
                timeout_s=self._sandbox_path_copy_timeout_s,
            )
            if result.return_code != 0:
                output = result.stderr or result.stdout or "<no output>"
                raise RuntimeError(
                    f"Failed to copy sandbox directory {rendered.source!r} to {rendered.destination!r}: {output}"
                )

    def _rendered_sandbox_path_symlinks(self) -> tuple[SandboxPathSymlink, ...]:
        return tuple(
            SandboxPathSymlink.model_validate(self._render_provider_option_templates(configured_symlink.model_dump()))
            for configured_symlink in self._sandbox_path_symlinks
        )

    def _verifier_must_validate_symlink(self, destination: PurePosixPath) -> bool:
        if not self._is_separate_verifier or self._shared_workspace is None:
            return False
        shared_mount = PurePosixPath(self._shared_workspace.volume.mount_path)
        return destination == shared_mount or shared_mount in destination.parents

    @staticmethod
    def _sandbox_path_symlink_validation_command(config: SandboxPathSymlink) -> str:
        source = shlex.quote(config.source)
        destination = shlex.quote(config.destination)
        return (
            f"test -e {source} && test -L {destination} && "
            f'[ "$(readlink -f -- {destination})" = "$(readlink -f -- {source})" ]'
        )

    async def _prepare_sandbox_path_symlinks(self) -> None:
        sandbox = self._require_sandbox()
        for rendered in self._rendered_sandbox_path_symlinks():
            destination_path = PurePosixPath(rendered.destination)
            validation = self._sandbox_path_symlink_validation_command(rendered)
            if self._verifier_must_validate_symlink(destination_path):
                command = validation
                action = "validate"
            else:
                parent = shlex.quote(str(destination_path.parent))
                destination = shlex.quote(rendered.destination)
                source = shlex.quote(rendered.source)
                command = (
                    f"test -e {source} && mkdir -p {parent} && "
                    f"if [ -L {destination} ]; then {validation}; "
                    f"elif [ -e {destination} ]; then exit 74; "
                    f"else ln -s -- {source} {destination}; fi"
                )
                action = "prepare"
            result = await sandbox.exec(command, cwd="/", timeout_s=60, user="root")
            if result.return_code != 0:
                output = result.stderr or result.stdout or "<no output>"
                self.logger.error(
                    "Failed to %s sandbox symlink %r -> %r: %s",
                    action,
                    rendered.destination,
                    rendered.source,
                    output,
                )
                raise RuntimeError(
                    f"Failed to {action} sandbox symlink {rendered.destination!r} -> {rendered.source!r}: {output}"
                )

    def _uses_shared_workspace(self, source_dir: str) -> bool:
        config = self._shared_workspace
        return config is not None and str(PurePosixPath(source_dir)) == config.volume.mount_path

    @staticmethod
    def _write_shared_workspace_marker(path: Path, marker: SharedWorkspaceMarker) -> None:
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_text(marker.to_json())
        temporary.replace(path)

    def _shared_workspace_host_path(self) -> PurePosixPath:
        volume = self._shared_workspace_volume(read_only=self._is_separate_verifier)
        host = volume.get("host")
        if not isinstance(host, Mapping) or not isinstance(host.get("path"), str):
            raise ValueError("shared_workspace.volume.host.path must be a string")
        path = PurePosixPath(host["path"])
        if path.name != str(self.context_id):
            raise ValueError("Rendered shared workspace host path must end with the Harbor context ID")
        return path

    @staticmethod
    def _validate_shared_workspace_excludes(exclude: Sequence[str]) -> tuple[str, ...]:
        normalized: list[str] = []
        for value in exclude:
            path = PurePosixPath(value)
            if (
                path.is_absolute()
                or not path.parts
                or any(part in {"", ".", ".."} for part in path.parts)
                or any(character in value for character in "*?[")
            ):
                raise ValueError(
                    "Direct shared workspace exclusions must be literal relative paths without "
                    f"'.', '..', or glob components (got {value!r})"
                )
            normalized.append(str(path))
        return tuple(normalized)

    def _shared_workspace_digest_command(
        self,
        *,
        source_dir: str,
        exclude: Sequence[str],
        prune_excluded: bool,
    ) -> str:
        source = PurePosixPath(source_dir)
        mounts = set(self._sandbox_volume_mount_paths())
        commands = ["set -euo pipefail"]
        if prune_excluded:
            symlink_sources = {
                PurePosixPath(config.destination): config for config in self._rendered_sandbox_path_symlinks()
            }
            for relative in exclude:
                excluded = source / relative
                if excluded in mounts:
                    continue
                configured_symlink = symlink_sources.get(excluded)
                if configured_symlink is not None:
                    commands.append(self._sandbox_path_symlink_validation_command(configured_symlink))
                    continue
                if any(excluded in mount.parents or mount in excluded.parents for mount in mounts if mount != source):
                    raise ValueError(f"Cannot safely exclude {relative!r} because it overlaps a nested volume")
                commands.append(f"rm -rf -- {shlex.quote(str(excluded))}")
        quoted_source = shlex.quote(str(source))
        commands.extend(
            [
                f"bad=$(find {quoted_source} -xdev -mindepth 1 ! -type d ! -type f ! -type l -print -quit)",
                'if [ -n "$bad" ]; then printf \'unsupported workspace entry: %s\\n\' "$bad"; exit 73; fi',
            ]
        )
        exclude_flags = " ".join(
            f"--exclude={shlex.quote(relative)} --exclude={shlex.quote(f'./{relative}')}" for relative in exclude
        )
        commands.append(
            "tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner --format=gnu "
            f"{exclude_flags} -C {quoted_source} -cf - . | sha256sum | awk '{{print $1}}'"
        )
        return f"bash -o pipefail -c {shlex.quote('; '.join(commands))}"

    @staticmethod
    def _shared_workspace_error_output(result: Any) -> str:
        return (
            "\n".join(
                output
                for output in (
                    (result.stdout or "").strip(),
                    (result.stderr or "").strip(),
                )
                if output
            )
            or "<no output>"
        )

    async def _shared_workspace_digest(
        self,
        *,
        source_dir: str,
        exclude: Sequence[str],
        prune_excluded: bool,
    ) -> str:
        config = self._shared_workspace
        if config is None:
            raise RuntimeError("Shared workspace is not configured")
        result = await self._require_sandbox().exec(
            self._shared_workspace_digest_command(
                source_dir=source_dir,
                exclude=exclude,
                prune_excluded=prune_excluded,
            ),
            cwd="/",
            timeout_s=config.handoff_timeout_s,
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"Failed to validate shared workspace {source_dir!r}: {self._shared_workspace_error_output(result)}"
            )
        digest = (result.stdout or "").strip().splitlines()[-1:]
        if not digest or len(digest[0]) != 64 or any(character not in "0123456789abcdef" for character in digest[0]):
            raise RuntimeError("Shared workspace validation did not return a valid SHA-256 digest")
        return digest[0]

    async def _mark_shared_workspace(
        self,
        *,
        source_dir: str,
        target_dir: Path,
        exclude: Sequence[str],
    ) -> None:
        if self._is_separate_verifier:
            raise RuntimeError("Only a policy environment may produce a shared workspace marker")
        target_dir.mkdir(parents=True, exist_ok=True)
        if any(target_dir.iterdir()):
            raise RuntimeError(f"Shared workspace marker directory is not empty: {target_dir}")
        normalized_exclude = self._validate_shared_workspace_excludes(exclude)
        digest = await self._shared_workspace_digest(
            source_dir=source_dir,
            exclude=normalized_exclude,
            prune_excluded=True,
        )
        marker_path = target_dir / _SHARED_WORKSPACE_MARKER
        marker = SharedWorkspaceMarker(
            context_id=str(self.context_id),
            source=str(PurePosixPath(source_dir)),
            host_path=str(self._shared_workspace_host_path()),
            exclude=normalized_exclude,
            sha256=digest,
        )
        self._write_shared_workspace_marker(marker_path, marker)
        self._shared_workspace_markers.add(marker_path)

    def _validate_shared_workspace_marker(self, marker: SharedWorkspaceMarker, target_dir: str) -> None:
        config = self._shared_workspace
        if config is None:
            raise RuntimeError("Shared workspace marker found but shared_workspace is disabled")
        if marker.context_id != str(self.context_id):
            raise RuntimeError("Shared workspace marker belongs to a different Harbor trial")
        if marker.source != config.volume.mount_path or str(PurePosixPath(target_dir)) != marker.source:
            raise RuntimeError("Shared workspace handoff target does not match its original source")
        if marker.host_path != str(self._shared_workspace_host_path()):
            raise RuntimeError("Shared workspace marker points to a different physical source path")
        if not marker.source_environment_stopped:
            raise RuntimeError("Refusing shared workspace handoff before source sandbox teardown is confirmed")
        if not self._is_separate_verifier:
            raise RuntimeError("Shared workspace handoff may only be consumed by a separate verifier environment")
        if marker.consumed:
            raise RuntimeError("Shared workspace handoff has already been consumed")

    async def _accept_shared_workspace(self, source: Path, target_dir: str) -> bool:
        marker_path = source / _SHARED_WORKSPACE_MARKER
        if not marker_path.is_file():
            return False
        if {path.name for path in source.iterdir()} != {_SHARED_WORKSPACE_MARKER}:
            raise RuntimeError(f"Shared workspace marker directory contains unexpected entries: {source}")
        marker = SharedWorkspaceMarker.from_path(marker_path)
        self._validate_shared_workspace_marker(marker, target_dir)
        self._shared_workspace_cleanup_required = True
        actual_digest = await self._shared_workspace_digest(
            source_dir=target_dir,
            exclude=marker.exclude,
            prune_excluded=False,
        )
        if actual_digest != marker.sha256:
            self.logger.error(
                "Shared workspace digest mismatch for %r: expected %s, got %s",
                target_dir,
                marker.sha256,
                actual_digest,
            )
            raise RuntimeError(
                "Shared workspace changed between policy teardown and verifier mount: "
                f"expected {marker.sha256}, got {actual_digest}"
            )
        self._write_shared_workspace_marker(marker_path, replace(marker, consumed=True))
        return True

    async def _cleanup_shared_workspace(self) -> None:
        config = self._shared_workspace
        if config is None:
            return
        host_path = self._shared_workspace_host_path()
        context_id = str(self.context_id)
        if host_path.name != context_id:
            raise RuntimeError("Refusing cleanup outside the context-scoped shared workspace")

        rendered_options = self._render_provider_option_templates(self._sandbox_provider_options)
        if not isinstance(rendered_options, Mapping):
            raise ValueError("sandbox_provider_options must be a mapping")
        cleanup_options = {key: value for key, value in rendered_options.items() if key != "volumes"}
        cleanup_options["volumes"] = [
            {
                "name": f"{config.volume.name}-cleanup",
                "host": {"path": str(host_path.parent)},
                "mountPath": _SHARED_WORKSPACE_CLEANUP_MOUNT,
                "readOnly": False,
            }
        ]
        cleanup_spec = SandboxSpec(
            image=self._resolved_image,
            ttl_s=config.cleanup_ttl_s,
            ready_timeout_s=min(float(self._sandbox_ready_timeout_s or config.cleanup_ttl_s), config.cleanup_ttl_s),
            workdir="/",
            env={},
            metadata={
                "harbor-session": self.session_id,
                "harbor-task": self.environment_name,
                "harbor-role": "workspace-cleanup",
                **resolve_provider_metadata(self._sandbox_provider),
                **self._sandbox_metadata,
            },
            resources={},
            entrypoint=config.cleanup_entrypoint,
            provider_options=cleanup_options,
        )
        cleanup_sandbox = AsyncSandbox(resolve_provider_config(self._sandbox_provider), cleanup_spec)
        cleanup_started = False
        operation_error: Exception | None = None
        try:
            await cleanup_sandbox.start()
            cleanup_started = True
            target = PurePosixPath(_SHARED_WORKSPACE_CLEANUP_MOUNT) / context_id
            result = await cleanup_sandbox.exec(
                f"rm -rf -- {shlex.quote(str(target))}; test ! -e {shlex.quote(str(target))}",
                cwd="/",
                timeout_s=config.cleanup_timeout_s,
                user="root",
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"Failed to clean direct shared workspace: {self._shared_workspace_error_output(result)}"
                )
        except Exception as error:  # noqa: BLE001 - cleanup sandbox must still terminate
            operation_error = error

        stop_error: Exception | None = None
        if cleanup_started:
            try:
                await cleanup_sandbox.stop()
            except Exception as error:  # noqa: BLE001 - report both cleanup operation and termination failures
                stop_error = error
        if operation_error is not None:
            if stop_error is not None:
                raise RuntimeError(
                    "Shared workspace cleanup and cleanup-sandbox termination both failed: "
                    f"cleanup_error={operation_error!r}, stop_error={stop_error!r}"
                ) from operation_error
            raise operation_error
        if stop_error is not None:
            raise stop_error
        self._shared_workspace_cleanup_required = False

    def _confirm_shared_workspace_source_stopped(self) -> None:
        for marker_path in self._shared_workspace_markers:
            marker = SharedWorkspaceMarker.from_path(marker_path)
            self._write_shared_workspace_marker(
                marker_path,
                replace(marker, source_environment_stopped=True),
            )

    def _sandbox_volume_mount_paths(self) -> tuple[PurePosixPath, ...]:
        rendered = self._provider_options_for_role()
        volumes = rendered.get("volumes", []) if isinstance(rendered, Mapping) else []
        mount_paths: list[PurePosixPath] = []
        for volume in volumes:
            if not isinstance(volume, Mapping):
                continue
            value = volume.get("mountPath", volume.get("mount_path"))
            if isinstance(value, str) and PurePosixPath(value).is_absolute():
                mount_paths.append(PurePosixPath(value))
        return tuple(mount_paths)

    def _empty_dirs_preserving_mounts_command(
        self,
        dirs: Sequence[str | PurePath],
        *,
        chmod: bool,
    ) -> str:
        mounts = self._sandbox_volume_mount_paths()
        commands: list[str] = []
        for value in dirs:
            path = PurePosixPath(str(value))
            quoted = shlex.quote(str(path))
            if path in mounts:
                commands.append("true")
                continue
            preserved_children = {path / mount.relative_to(path).parts[0] for mount in mounts if path in mount.parents}
            commands.extend(
                [
                    f"if [ -L {quoted} ] || {{ [ -e {quoted} ] && [ ! -d {quoted} ]; }}; then rm -rf {quoted}; fi",
                    f"mkdir -p {quoted}",
                ]
            )
            exclusions = " ".join(f"! -path {shlex.quote(str(child))}" for child in sorted(preserved_children))
            commands.append(f"find {quoted} -mindepth 1 -maxdepth 1 {exclusions} -exec rm -rf -- {{}} +")
            if chmod:
                commands.append(f"chmod 777 {quoted}")
        return " && ".join(commands)

    async def start(self, force_build: bool) -> None:
        if force_build:
            self.logger.warning(
                "force_build is not supported by NemoGymSandboxEnvironment; using the task's prebuilt image %r.",
                self._resolved_image,
            )
        if self._network_policy.network_mode == NetworkMode.NO_NETWORK:
            self.logger.warning(
                "Task %r requests network_mode='no-network' but NemoGymSandboxEnvironment does "
                "not enforce network isolation; the sandbox keeps cluster-default egress.",
                self.environment_name,
            )

        sandbox = AsyncSandbox(
            resolve_provider_config(self._sandbox_provider),
            self._build_spec(),
        )
        await sandbox.start()
        self._sandbox = sandbox

        # Harbor's local backends bind mount these convention directories. The
        # remote provider must create all of them before agents start writing.
        log_dirs = f"{EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir} {EnvironmentPaths.artifacts_dir}"
        result = await self._sandbox.exec(f"mkdir -p {log_dirs}", cwd="/", timeout_s=60)
        if result.return_code != 0:
            raise RuntimeError(
                f"Failed to create log directories in sandbox: {result.stderr or result.stdout or '<no output>'}"
            )
        await self._prepare_sandbox_path_symlinks()
        await self._copy_sandbox_paths()
        await self._upload_environment_dir_after_start()

    def _emptying_removes_sandbox_copy(self, dirs: Sequence[str | PurePath]) -> bool:
        emptied = [PurePosixPath(str(path)) for path in dirs]
        for configured_copy in self._sandbox_path_copies:
            rendered = SandboxPathCopy.model_validate(
                self._render_provider_option_templates(configured_copy.model_dump())
            )
            destination = PurePosixPath(rendered.destination)
            if any(path == destination or path in destination.parents for path in emptied):
                return True
        return False

    def _emptying_removes_sandbox_symlink(self, dirs: Sequence[str | PurePath]) -> bool:
        emptied = [PurePosixPath(str(path)) for path in dirs]
        return any(
            any(path == destination or path in destination.parents for path in emptied)
            for destination in (PurePosixPath(config.destination) for config in self._rendered_sandbox_path_symlinks())
        )

    async def ensure_dirs(
        self,
        dirs: Sequence[str | PurePath],
        *,
        chmod: bool = True,
    ) -> ExecResult | None:
        """Create infrastructure directories without relying on the task workdir."""
        if not dirs:
            return None
        return await self.exec(
            self._ensure_dirs_command(dirs, chmod=chmod),
            cwd="/",
            user=self._reset_dirs_user() if chmod else None,
        )

    async def empty_dirs(
        self,
        dirs: Sequence[str | PurePath],
        *,
        chmod: bool = True,
    ) -> ExecResult | None:
        """Empty directories, then restore verifier-only sandbox-local baselines."""
        if not dirs:
            return None
        result = await self.exec(
            self._empty_dirs_preserving_mounts_command(dirs, chmod=chmod),
            cwd="/",
            user=self._reset_dirs_user(),
        )
        if result.return_code == 0:
            if self._emptying_removes_sandbox_symlink(dirs):
                await self._prepare_sandbox_path_symlinks()
            if self._is_separate_verifier and self._emptying_removes_sandbox_copy(dirs):
                await self._copy_sandbox_paths()
        return result

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        result = await self.exec(
            self._path_kind_check_command(path, require_dir=True),
            cwd="/",
            timeout_sec=10,
            user=user,
        )
        return result.return_code == 0

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        result = await self.exec(
            self._path_kind_check_command(path, require_dir=False),
            cwd="/",
            timeout_sec=10,
            user=user,
        )
        return result.return_code == 0

    def _require_sandbox(self) -> AsyncSandbox:
        if self._sandbox is None:
            raise RuntimeError("Sandbox is not running; call start() first.")
        return self._sandbox

    async def stop(self, delete: bool):
        if self._sandbox is None:
            return
        if not delete:
            # Remote sandboxes are single-use, so honoring delete=False would
            # leak cluster resources.
            self.logger.debug(
                "delete=False is ignored by NemoGymSandboxEnvironment; the sandbox is always terminated on stop()."
            )
        sandbox = self._sandbox
        stop_error: Exception | None = None
        try:
            await sandbox.stop()
        except Exception as error:  # noqa: BLE001 - preserve cleanup and stop failures
            stop_error = error
        finally:
            self._sandbox = None
        if stop_error is None:
            self._confirm_shared_workspace_source_stopped()
        if stop_error is not None:
            raise stop_error
        should_cleanup = self._shared_workspace_cleanup_required or (
            self._shared_workspace is not None
            and not self._is_separate_verifier
            and not self._shared_workspace_markers
        )
        if should_cleanup:
            await self._cleanup_shared_workspace()

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        timeout_s = timeout_sec if timeout_sec is not None else self._default_exec_timeout_s
        # Harbor's docker/daytona backends use an interactive bash, so
        # .bashrc-based task setups (conda, pyenv, PATH) must behave the same.
        if self._exec_shell:
            command = f"{self._exec_shell} {shlex.quote(command)}"
        if self._cpu_pin_enabled:
            width = int(self.task_env_config.cpus or 0)
            if width > 0:
                command = f"{_cpu_pin_prefix(width)} {command}"
        result = await self._require_sandbox().exec(
            command,
            cwd=cwd,
            env=self._merge_env(env),
            timeout_s=timeout_s,
            user=self._resolve_user(user),
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
            await sandbox.exec(f"mkdir -p {shlex.quote(parent)}", cwd="/", timeout_s=60)
        await sandbox.upload(source, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        source = Path(source_dir)
        if not source.exists():
            raise FileNotFoundError(f"Source directory not found: {source}")
        if await self._accept_shared_workspace(source, target_dir):
            return
        sandbox = self._require_sandbox()
        excludes = self._environment_upload_excludes if self._is_environment_upload(source) else ()

        remote_tar = f"{_TRANSFER_DIR}/.nemo-gym-upload-{uuid.uuid4().hex}.tar.gz"
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_tar = Path(tmp_dir) / "upload.tar.gz"
            with tarfile.open(local_tar, "w:gz") as tar:
                # Archive the *contents* of source_dir so they land directly in
                # target_dir (Harbor's upload_dir contract).
                def _filter(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
                    relative = PurePosixPath(member.name).relative_to(".")
                    if excludes and self._is_upload_excluded(relative):
                        return None
                    return member

                tar.add(source, arcname=".", filter=_filter)
            await sandbox.upload(local_tar, remote_tar)

        quoted_target = shlex.quote(target_dir)
        quoted_tar = shlex.quote(remote_tar)
        result = await sandbox.exec(
            f"mkdir -p {quoted_target} && tar -xzf {quoted_tar} -C {quoted_target}; "
            f"status=$?; rm -f {quoted_tar}; exit $status",
            cwd="/",
            timeout_s=600,
        )
        if result.return_code != 0:
            self.logger.warning(
                "tar-based upload_dir failed (rc=%s, stderr=%r); falling back to per-file upload.",
                result.return_code,
                (result.stderr or "")[:500],
            )
            await self._upload_dir_file_by_file(source, target_dir, excludes=excludes)

    async def _upload_dir_file_by_file(
        self,
        source: Path,
        target_dir: str,
        *,
        excludes: tuple[PurePosixPath, ...] = (),
    ):
        sandbox = self._require_sandbox()
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            relative_posix = PurePosixPath(*relative.parts)
            if excludes and self._is_upload_excluded(relative_posix):
                continue
            remote_path = str(PurePosixPath(target_dir) / relative_posix)
            if path.is_dir():
                await sandbox.exec(f"mkdir -p {shlex.quote(remote_path)}", cwd="/", timeout_s=60)
            elif path.is_file():
                await self.upload_file(path, remote_path)

    async def download_file(self, source_path: str, target_path: Path | str):
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        await self._require_sandbox().download(source_path, target)

    async def download_dir(self, source_dir: str, target_dir: Path | str):
        if self._uses_shared_workspace(source_dir):
            await self._mark_shared_workspace(
                source_dir=source_dir,
                target_dir=Path(target_dir),
                exclude=(),
            )
            return
        await self._download_dir_archive(
            source_dir=source_dir,
            target_dir=target_dir,
            exclude=(),
            snapshot=True,
        )

    async def download_dir_with_exclusions(
        self,
        *,
        source_dir: str,
        target_dir: Path | str,
        exclude: list[str],
    ) -> None:
        if self._uses_shared_workspace(source_dir):
            await self._mark_shared_workspace(
                source_dir=source_dir,
                target_dir=Path(target_dir),
                exclude=exclude,
            )
            return
        await self._download_dir_archive(
            source_dir=source_dir,
            target_dir=target_dir,
            exclude=exclude,
            snapshot=False,
        )

    async def _download_dir_archive(
        self,
        *,
        source_dir: str,
        target_dir: Path | str,
        exclude: Sequence[str],
        snapshot: bool,
    ) -> None:
        sandbox = self._require_sandbox()
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        transfer_id = uuid.uuid4().hex
        remote_tar = f"{_TRANSFER_DIR}/.nemo-gym-download-{transfer_id}.tar.gz"
        remote_snapshot = f"{_TRANSFER_DIR}/.nemo-gym-download-{transfer_id}"
        quoted_source = shlex.quote(source_dir)
        quoted_tar = shlex.quote(remote_tar)
        quoted_snapshot = shlex.quote(remote_snapshot)
        exclude_flags = " ".join(f"--exclude={shlex.quote(pattern)}" for pattern in exclude)
        if snapshot:
            archive_command = (
                f"rm -rf {quoted_snapshot}; mkdir -p {quoted_snapshot} "
                f"&& cp -a -- {quoted_source}/. {quoted_snapshot}/ "
                f"&& tar -czf {quoted_tar} -C {quoted_snapshot} .; "
                f"status=$?; rm -rf {quoted_snapshot}; exit $status"
            )
        else:
            archive_command = f"tar -czf {quoted_tar} {exclude_flags} -C {quoted_source} ."
        result = await sandbox.exec(
            archive_command,
            cwd="/",
            timeout_s=600,
        )
        if result.return_code != 0:
            await sandbox.exec(f"rm -f {quoted_tar}", cwd="/", timeout_s=60)
            self.logger.warning(
                "tar-based download_dir failed (rc=%s, stderr=%r); falling back to per-file download.",
                result.return_code,
                (result.stderr or "")[:500],
            )
            await self._download_dir_file_by_file(source_dir, target, exclude=exclude)
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            local_tar = Path(tmp_dir) / "download.tar.gz"
            try:
                await sandbox.download(remote_tar, local_tar)
            finally:
                await sandbox.exec(f"rm -f {quoted_tar}", cwd="/", timeout_s=60)
            with tarfile.open(local_tar, "r:gz") as tar:
                tar.extractall(target, filter="data")

    @staticmethod
    def _is_download_excluded(relative: PurePosixPath, exclude: Sequence[str]) -> bool:
        candidates = [relative.as_posix(), *(parent.as_posix() for parent in relative.parents if parent.parts)]
        return any(fnmatch(candidate, pattern) for candidate in candidates for pattern in exclude)

    async def _download_dir_file_by_file(
        self,
        source_dir: str,
        target: Path,
        *,
        exclude: Sequence[str] = (),
    ) -> None:
        sandbox = self._require_sandbox()
        listing = await sandbox.exec(
            f"find {shlex.quote(source_dir)} -type f",
            cwd="/",
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
            if self._is_download_excluded(relative, exclude):
                continue
            await self.download_file(remote_path, target / Path(*relative.parts))
