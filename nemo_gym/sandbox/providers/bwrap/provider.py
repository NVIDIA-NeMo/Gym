# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Bubblewrap (``bwrap``) sandbox provider.

A sandbox is a directory, ``<base_dir>/<sandbox_id>/{workspace,tmp,home}``, on
storage local to the Gym process. There is no long-lived container process:
every ``exec`` spawns ``bwrap <flags> -- <shell> -c <command>`` in fresh user,
pid, ipc and uts namespaces (and a fresh network namespace unless
``network: host``). The host's system directories are bound read-only; only
the sandbox's own directories are writable. File transfer is plain host-side
file I/O with containment checks. See README.md for the design and its limits.
"""

import asyncio
import contextlib
import errno
import glob
import json
import logging
import os
import posixpath
import re
import shutil
import signal
import socket
import stat
import tempfile
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxCreateVerificationError,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)


LOGGER = logging.getLogger(__name__)

PROVIDER_NAME = "bwrap"
# Sandbox-runtime failure (bwrap could not run the command), not the command's exit code.
SANDBOX_RUNTIME_RETURN_CODE = 125
SANDBOX_ID_PREFIX = "bwrap-"
SANDBOX_ID_PATTERN = re.compile(r"bwrap-[0-9a-f]{32}")
ENV_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
STATE_FILE = "sandbox.json"
CLOSING_MARKER = "closing"
# Host-only subdirectories of a sandbox root. Only these three are bound into the sandbox;
# ``run/`` (process registry) and the state file stay invisible to sandboxed code.
WORKSPACE_DIR = "workspace"
TMP_DIR = "tmp"
HOME_DIR = "home"
RUN_DIR = "run"
# ``run/<bwrap pid>.init``: "<host pid of the exec's namespace init> <its start time>".
INIT_RECORD_SUFFIX = ".init"
# run/ entry of an exec that is queued or about to spawn: "<owner pid> <owner start time>".
PENDING_RECORD_PREFIX = "pending-"
# Whether /proc reports process start times (Linux); without it records carry no start time.
_HAVE_PROC = os.path.exists("/proc/self/stat")
DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# Paths a workdir must not shadow: the root, or the kernel/device mounts bwrap creates.
FORBIDDEN_WORKDIRS = ("/", "/proc", "/dev")

# Measured on a Slurm + pyxis GB300 cluster (a CUDA task image, bwrap 0.11.2). /usr, /bin and /lib are required;
# the rest are existence-checked. /venv is that image's Python environment.
DEFAULT_RO_BINDS: tuple[Any, ...] = (
    "/usr",
    "/bin",
    "/lib",
    {"path": "/sbin", "optional": True},
    {"path": "/lib64", "optional": True},
    {"path": "/opt", "optional": True},
    {"path": "/venv", "optional": True},
    {"path": "/etc/ld.so.cache", "optional": True},
    {"path": "/etc/ld.so.conf", "optional": True},
    {"path": "/etc/ld.so.conf.d", "optional": True},
    {"path": "/etc/alternatives", "optional": True},
)
# Extra read-only binds for ``network: host`` so name resolution, TLS and user lookup work.
# A smaller set (without nsswitch.conf and ca-certificates) broke HTTPS on that cluster (curl rc 77).
DEFAULT_NETWORK_HOST_RO_BINDS: tuple[Any, ...] = (
    "/etc/resolv.conf",
    "/etc/hosts",
    "/etc/nsswitch.conf",
    "/etc/ssl",
    "/etc/ca-certificates",
    "/etc/passwd",
    "/etc/group",
    {"path": "/etc/pki", "optional": True},
)
DEFAULT_ENV_PASSTHROUGH: tuple[str, ...] = ("PATH", "LANG", "LD_LIBRARY_PATH", "CUDA_HOME")


class BwrapCreateError(SandboxCreateError):
    """Raised when the bwrap provider cannot create a sandbox."""


class BwrapCreateVerificationError(SandboxCreateVerificationError):
    """Raised when a new bwrap sandbox fails its readiness probe."""


class BwrapPathError(PermissionError):
    """Raised when a transfer path escapes the sandbox's mounts or crosses a symlink."""


# ----------------------------------------------------------------------------- config


@dataclass(frozen=True)
class BwrapBind:
    """A read-only system bind: ``path`` may be a glob; ``optional`` entries may be absent."""

    path: str
    optional: bool = False

    @classmethod
    def coerce(cls, value: Any) -> "BwrapBind":
        if isinstance(value, BwrapBind):
            return value
        if isinstance(value, str):
            return cls(path=value)
        if isinstance(value, Mapping):
            unknown = set(value) - {"path", "optional"}
            if unknown:
                raise ValueError(f"Unknown bwrap bind keys: {sorted(unknown)}")
            return cls(path=str(value["path"]), optional=bool(value.get("optional", False)))
        raise TypeError(f"A bwrap bind must be a path string or {{path, optional}}, got {value!r}")

    def __post_init__(self) -> None:
        if not self.path.startswith("/"):
            raise ValueError(f"bwrap bind paths must be absolute: {self.path!r}")


@dataclass(frozen=True)
class BwrapMount:
    """An extra bind mount: host ``src`` at sandbox ``dst``; read-only unless ``ro`` is false."""

    src: str
    dst: str
    ro: bool = True

    @classmethod
    def coerce(cls, value: Any) -> "BwrapMount":
        """Accept ``{src, dst, ro}`` or the enroot/apptainer string form ``src:dst[:ro|rw]``."""
        if isinstance(value, BwrapMount):
            return value
        if isinstance(value, str):
            parts = value.split(":")
            if len(parts) == 2:
                return cls(src=parts[0], dst=parts[1])
            if len(parts) == 3 and parts[2] in ("ro", "rw"):
                return cls(src=parts[0], dst=parts[1], ro=parts[2] == "ro")
            raise ValueError(f"bwrap mount strings must be 'src:dst[:ro|rw]', got {value!r}")
        if isinstance(value, Mapping):
            unknown = set(value) - {"src", "dst", "ro"}
            if unknown:
                raise ValueError(f"Unknown bwrap mount keys: {sorted(unknown)}")
            return cls(src=str(value["src"]), dst=str(value["dst"]), ro=bool(value.get("ro", True)))
        raise TypeError(f"A bwrap mount must be a mapping or 'src:dst[:ro|rw]' string, got {value!r}")

    def __post_init__(self) -> None:
        for label, path in (("src", self.src), ("dst", self.dst)):
            if not path.startswith("/") or ".." in PurePosixPath(path).parts:
                raise ValueError(f"bwrap mount {label} must be an absolute path without '..': {path!r}")
        if posixpath.normpath(self.dst) in FORBIDDEN_WORKDIRS:
            raise ValueError(f"bwrap mount dst cannot be {self.dst!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"src": self.src, "dst": self.dst, "ro": self.ro}


def _validate_env(env: Mapping[str, Any], *, source: str) -> dict[str, str]:
    validated: dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(key, str) or ENV_NAME_PATTERN.fullmatch(key) is None:
            raise ValueError(f"{source}: environment variable names must be POSIX identifiers, got {key!r}")
        if not isinstance(value, str):
            raise TypeError(f"{source}: environment variable {key} must be a string")
        if "\x00" in value:
            raise ValueError(f"{source}: environment variable {key} contains NUL")
        validated[key] = value
    return validated


def _validate_sandbox_dir(path: str, *, label: str) -> str:
    if not isinstance(path, str) or not path.startswith("/") or ".." in PurePosixPath(path).parts:
        raise ValueError(f"bwrap {label} must be an absolute path without '..', got {path!r}")
    normalized = posixpath.normpath(path)
    if normalized in FORBIDDEN_WORKDIRS or normalized.startswith(("/proc/", "/dev/")):
        raise ValueError(f"bwrap {label} cannot be {path!r}")
    return normalized


@dataclass(frozen=True)
class BwrapConfig:
    """Provider config: the ``bwrap:`` block of a sandbox provider config."""

    # Absolute path to a pinned bwrap binary. Required: no PATH lookup.
    bwrap_path: str
    # Parent of every sandbox directory; node-local storage. Default: <tmp>/nemo-gym-bwrap-<uid>.
    base_dir: str | None = None
    # Workdir used when SandboxSpec.workdir is unset.
    workdir: str = "/workspace"
    # Where the per-sandbox home directory appears inside; HOME points here.
    home_path: str = "/home/sandbox"
    # "none": fresh network namespace (loopback only). "host": share the Gym process's network.
    network: Literal["none", "host"] = "none"
    shell: str = "/bin/bash"
    ro_binds: list[Any] = field(default_factory=lambda: list(DEFAULT_RO_BINDS))
    network_host_ro_binds: list[Any] = field(default_factory=lambda: list(DEFAULT_NETWORK_HOST_RO_BINDS))
    # Extra binds for every sandbox; SandboxSpec.provider_options["mounts"] adds per-sandbox ones.
    mounts: list[Any] = field(default_factory=list)
    # Fixed environment values, and names copied from the Gym process's environment when set.
    env: dict[str, str] = field(default_factory=dict)
    env_passthrough: list[str] = field(default_factory=lambda: list(DEFAULT_ENV_PASSTHROUGH))
    # Force CUDA_VISIBLE_DEVICES="" and NVIDIA_VISIBLE_DEVICES=void in every exec.
    hide_gpus: bool = True
    # Remount bwrap's tmpfs root read-only after all binds, so writes outside the writable
    # mounts (e.g. to / or /etc) fail instead of landing silently in a throwaway tmpfs.
    remount_root_ro: bool = True
    # Text files (sandbox path -> content) written into every new sandbox.
    default_files: dict[str, str] = field(default_factory=dict)
    # Extra bwrap arguments inserted before the command.
    extra_args: list[str] = field(default_factory=list)
    # Let sandboxed processes create a nested user namespace, e.g. bwrap for the Codex agent's
    # tool_pid_namespace. A sandboxed root can map uid 0 into one only with CAP_SETFCAP (Linux
    # >= 5.12), so when the Gym process is root the sandbox keeps that one capability; file
    # capabilities set inside stay bounded by the sandbox's bounding set. Others need nothing.
    allow_nested_userns: bool = False
    # Concurrent bwrap processes per provider instance; null = unbounded.
    exec_concurrency: int | None = 256
    default_timeout_s: float | None = 180
    # Bound on reaping a killed bwrap process tree.
    kill_timeout_s: float = 10
    # Run `true` in each new sandbox before returning it.
    probe_on_create: bool = True
    probe_timeout_s: float = 30

    def __post_init__(self) -> None:
        if not isinstance(self.bwrap_path, str) or not os.path.isabs(self.bwrap_path):
            raise ValueError(f"bwrap.bwrap_path must be an absolute path, got {self.bwrap_path!r}")
        if self.network not in ("none", "host"):
            raise ValueError(f"bwrap.network must be 'none' or 'host', got {self.network!r}")
        _validate_sandbox_dir(self.workdir, label="workdir")
        home = _validate_sandbox_dir(self.home_path, label="home_path")
        if home == "/tmp":
            raise ValueError("bwrap.home_path cannot be /tmp")
        if not self.shell.startswith("/"):
            raise ValueError("bwrap.shell must be an absolute path")
        if self.exec_concurrency is not None and self.exec_concurrency < 1:
            raise ValueError("bwrap.exec_concurrency must be >= 1 or null")
        if self.default_timeout_s is not None and self.default_timeout_s <= 0:
            raise ValueError("bwrap.default_timeout_s must be > 0 or null")
        if self.kill_timeout_s <= 0 or self.probe_timeout_s <= 0:
            raise ValueError("bwrap.kill_timeout_s and probe_timeout_s must be > 0")
        object.__setattr__(self, "ro_binds", [BwrapBind.coerce(b) for b in self.ro_binds])
        object.__setattr__(self, "network_host_ro_binds", [BwrapBind.coerce(b) for b in self.network_host_ro_binds])
        object.__setattr__(self, "mounts", [BwrapMount.coerce(m) for m in self.mounts])
        object.__setattr__(self, "env", _validate_env(self.env, source="bwrap.env"))
        for name in self.env_passthrough:
            if ENV_NAME_PATTERN.fullmatch(name) is None:
                raise ValueError(f"bwrap.env_passthrough entries must be POSIX identifiers, got {name!r}")
        for path in self.default_files:
            if not isinstance(path, str) or not path.startswith("/"):
                raise ValueError(f"bwrap.default_files keys must be absolute sandbox paths, got {path!r}")


def resolve_binds(binds: list[BwrapBind]) -> list[str]:
    """Expand globs and drop absent optional entries; a missing required entry raises."""
    resolved: list[str] = []
    for bind in binds:
        if glob.has_magic(bind.path):
            matches = sorted(glob.glob(bind.path))
        else:
            matches = [bind.path] if os.path.lexists(bind.path) else []
        if not matches and not bind.optional:
            raise FileNotFoundError(f"Required bwrap read-only bind {bind.path!r} does not exist")
        resolved.extend(match for match in matches if match not in resolved)
    return resolved


# ----------------------------------------------------------------------------- state


@dataclass
class _BwrapSandbox:
    """Provider-private state on ``SandboxHandle.raw``."""

    sandbox_id: str
    root: Path
    workdir: str
    network: str
    env: dict[str, str] = field(default_factory=dict)
    mounts: list[BwrapMount] = field(default_factory=list)

    @property
    def workspace(self) -> Path:
        return self.root / WORKSPACE_DIR

    @property
    def tmp(self) -> Path:
        return self.root / TMP_DIR

    @property
    def home(self) -> Path:
        return self.root / HOME_DIR

    @property
    def run_dir(self) -> Path:
        return self.root / RUN_DIR

    def state(self) -> dict[str, Any]:
        return {
            "sandbox_id": self.sandbox_id,
            "workdir": self.workdir,
            "network": self.network,
            "env": dict(self.env),
            "mounts": [mount.to_dict() for mount in self.mounts],
        }


def _process_start_time(pid: int) -> str | None:
    """The kernel start time of ``pid`` (Linux), guarding kills against PID reuse."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # The command name (field 2) is parenthesized and may contain spaces.
    fields = raw.rpartition(")")[2].split()
    # Field 3 (state) is fields[0]; field 22 (starttime) is fields[19].
    if len(fields) < 20 or fields[0] in ("Z", "X"):
        return None
    return fields[19]


def _pid_alive(pid: int, start_time: str) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False
    if not start_time:
        return True
    return _process_start_time(pid) == start_time


def _pid_may_run(pid: int, start_time: str) -> bool:
    """Conservative liveness for a cleanup proof: only a missing, zombie or reused pid is gone."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if not start_time:
        return True
    return _process_start_time(pid) == start_time


def _read_init_record(path: Path) -> tuple[int, str] | None:
    """``(pid, start time)`` of a namespace-init record, or None when it is unreadable."""
    try:
        pid, _, start_time = path.read_text().partition(" ")
        return int(pid), start_time.strip()
    except (OSError, ValueError):
        return None


def _rmtree(path: Path) -> None:
    """Remove a sandbox tree, restoring owner permissions that sandboxed code may have dropped."""

    def onexc(function: Any, failed: str, error: BaseException) -> None:
        if isinstance(error, FileNotFoundError):
            return
        if not isinstance(error, PermissionError):
            raise error
        parent = os.path.dirname(failed)
        for target in (parent, failed):
            with contextlib.suppress(OSError):
                if not os.path.islink(target):
                    os.chmod(target, stat.S_IRWXU)
        if function is os.rmdir or os.path.isdir(failed) and not os.path.islink(failed):
            shutil.rmtree(failed, onexc=onexc)
        else:
            os.unlink(failed)

    shutil.rmtree(path, onexc=onexc)


def _regular_file_or_close(fd: int, path: str) -> None:
    """Refuse (closing ``fd``) anything but a regular file, then make ``fd`` blocking again."""
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise IsADirectoryError(f"sandbox path {path!r} is not a regular file")
        os.set_blocking(fd, True)
    except BaseException:
        os.close(fd)
        raise


# ----------------------------------------------------------------------------- provider


class BwrapProvider:
    """Sandbox provider backed by a pinned bubblewrap binary on the Gym process's host."""

    name = PROVIDER_NAME

    def __init__(self, **config: Any) -> None:
        self._config = BwrapConfig(**config)
        binary = self._config.bwrap_path
        if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
            raise RuntimeError(f"bwrap.bwrap_path {binary!r} is not an executable file")
        base = self._config.base_dir or os.path.join(tempfile.gettempdir(), f"nemo-gym-bwrap-{os.getuid()}")
        base = os.path.expanduser(base)
        if not os.path.isabs(base):
            raise ValueError(f"bwrap.base_dir must be absolute, got {base!r}")
        self._base_dir = Path(os.path.realpath(base))
        # Resolved once: the Gym process's filesystem is the sandbox's system image.
        self._system_binds = resolve_binds(self._config.ro_binds)
        self._network_host_binds = (
            resolve_binds(self._config.network_host_ro_binds) if self._config.network == "host" else []
        )
        self._check_target(self._config.workdir, label="workdir")
        self._check_target(self._config.home_path, label="home_path")
        self._check_own_dirs(self._config.workdir)
        for mount in self._config.mounts:
            if not os.path.exists(mount.src):
                raise FileNotFoundError(f"bwrap mount source {mount.src!r} does not exist")
            self._check_target(mount.dst, label="mount dst")
            self._check_mount_dst(mount.dst, workdir=self._config.workdir)
        self._semaphore = (
            asyncio.Semaphore(self._config.exec_concurrency) if self._config.exec_concurrency is not None else None
        )
        # sandbox_id -> {pid: process} for bwrap processes this instance started.
        self._processes: dict[str, dict[int, asyncio.subprocess.Process]] = {}
        # Sandboxes with an exec whose namespace init could not be recorded (see _record_init).
        self._untracked_inits: set[str] = set()
        self._image_warned = False

    def _check_target(self, target: str, *, label: str) -> None:
        """Refuse a writable/extra mount target inside a read-only system bind.

        bwrap cannot create a mount point inside a read-only bind (measured on that cluster:
        a target under /opt failed), so use a top-level target such as /codex-runtime.
        """
        normalized = posixpath.normpath(target)
        for bound in [*self._system_binds, *self._network_host_binds]:
            if normalized == bound or normalized.startswith(bound.rstrip("/") + "/"):
                raise ValueError(
                    f"bwrap {label} {target!r} is inside the read-only bind {bound!r}; "
                    f"use a top-level path such as /codex-runtime"
                )

    def _sandbox_writable_dirs(self, workdir: str) -> tuple[str, ...]:
        return ("/tmp", posixpath.normpath(self._config.home_path), posixpath.normpath(workdir))

    def _check_own_dirs(self, workdir: str) -> None:
        """Refuse a workdir that equals, contains or lies inside /tmp or home_path (or vice versa).

        Each of the three is a separate bind; nested, a later bind hides an earlier one from commands while
        transfers still resolve to it, and a mount point inside /tmp or home is created in sandbox-writable space.
        """
        own = {"/tmp": "/tmp", "home_path": posixpath.normpath(self._config.home_path)}
        own["workdir"] = posixpath.normpath(workdir)
        names = list(own)
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                a, b = own[first], own[second]
                if a == b or a.startswith(b.rstrip("/") + "/") or b.startswith(a.rstrip("/") + "/"):
                    raise ValueError(f"bwrap {first} {a!r} and {second} {b!r} overlap; they must be disjoint")

    def _check_mount_dst(self, target: str, *, workdir: str) -> None:
        """Refuse an extra mount at, above or inside the sandbox's own /tmp, home or workdir.

        Inside one: bwrap creates the mount point in a directory that sandboxed code can write, and follows
        symlinks while doing so, so an earlier exec could make it create entries on the host. At or above one:
        the mount shadows that directory, and transfers to it would land where the sandbox cannot see them.
        """
        normalized = posixpath.normpath(target)
        for own in self._sandbox_writable_dirs(workdir):
            if (
                normalized == own
                or normalized.startswith(own.rstrip("/") + "/")
                or own.startswith(normalized.rstrip("/") + "/")
            ):
                raise ValueError(
                    f"bwrap mount dst {target!r} overlaps the sandbox's own {own!r}; "
                    "use a top-level path such as /codex-runtime"
                )

    def _binds_workspace_host_path(self, inst: _BwrapSandbox) -> bool:
        """Whether the workspace is also bound at its host path. Not when that path lies inside the sandbox's own
        /tmp, home or workdir: bwrap would create its mount point in sandbox-writable space (see _check_mount_dst)."""
        host = str(inst.workspace)
        if host == inst.workdir:
            return False
        return not any(
            host == own or host.startswith(own.rstrip("/") + "/") for own in self._sandbox_writable_dirs(inst.workdir)
        )

    @property
    def config(self) -> BwrapConfig:
        return self._config

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    # ------------------------------------------------------------------ argv

    def _environment(self, inst: _BwrapSandbox, env: Mapping[str, str] | None) -> dict[str, str]:
        """The complete environment of a sandboxed command (bwrap passes its own env through)."""
        config = self._config
        result = {name: os.environ[name] for name in config.env_passthrough if name in os.environ}
        result.update(config.env)
        result.setdefault("PATH", DEFAULT_PATH)
        result.setdefault("LANG", "C.UTF-8")
        result["HOME"] = config.home_path
        result.update(inst.env)
        if env:
            result.update(_validate_env(env, source="exec env"))
        if config.hide_gpus:
            result["CUDA_VISIBLE_DEVICES"] = ""
            result["NVIDIA_VISIBLE_DEVICES"] = "void"
        return result

    def bwrap_argv(self, inst: _BwrapSandbox, command: str, *, cwd: str | None = None) -> list[str]:
        """The full ``bwrap ... -- <shell> -c <command>`` argv for one exec."""
        config = self._config
        argv = [
            config.bwrap_path,
            "--die-with-parent",
            "--new-session",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
        ]
        if inst.network != "host":
            argv.append("--unshare-net")
        argv += ["--cap-drop", "ALL"]
        if config.allow_nested_userns and os.geteuid() == 0:
            # Only root needs it; a non-root sandbox's nested bwrap refuses to start with capabilities.
            argv += ["--cap-add", "CAP_SETFCAP"]
        network_binds = self._network_host_binds
        if inst.network == "host" and config.network != "host":
            # A sandbox created as host-network by a differently configured provider.
            network_binds = resolve_binds(config.network_host_ro_binds)
        for path in [*self._system_binds, *(network_binds if inst.network == "host" else [])]:
            argv += ["--ro-bind", path, path]
        argv += ["--proc", "/proc", "--dev", "/dev"]
        argv += ["--bind", str(inst.tmp), "/tmp"]
        argv += ["--bind", str(inst.home), config.home_path]
        argv += ["--bind", str(inst.workspace), inst.workdir]
        if self._binds_workspace_host_path(inst):
            argv += ["--bind", str(inst.workspace), str(inst.workspace)]
        for mount in [*config.mounts, *inst.mounts]:
            argv += ["--ro-bind" if mount.ro else "--bind", mount.src, mount.dst]
        if config.remount_root_ro:
            # Last: bwrap must still create the mount points above in its root tmpfs.
            argv += ["--remount-ro", "/"]
        argv += ["--chdir", cwd or inst.workdir]
        argv += list(config.extra_args)
        argv += ["--", config.shell, "-c", command]
        return argv

    # ------------------------------------------------------------------ process registry

    def _register(self, inst: _BwrapSandbox, process: asyncio.subprocess.Process) -> None:
        self._processes.setdefault(inst.sandbox_id, {})[process.pid] = process
        # A host-only record lets any provider instance (e.g. the creator after a borrower ran
        # this exec) kill the process at close; the start time guards against PID reuse.
        with contextlib.suppress(OSError):
            (inst.run_dir / str(process.pid)).write_text(_process_start_time(process.pid) or "")

    async def _watch_info(self, inst: _BwrapSandbox, outer_pid: int, info_fd: int) -> bool:
        """Read bwrap's --info-fd pipe without a thread, then record the namespace init.

        True when bwrap reported a namespace init (``child-pid``), i.e. its own setup got as far as starting the
        sandboxed command. The command cannot write to this pipe, so a False is bwrap's own failure.
        """
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        stream = os.fdopen(info_fd, "rb", buffering=0)
        try:
            transport, _ = await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), stream)
        except BaseException:
            stream.close()
            raise
        try:
            return await self._record_init(inst, outer_pid, reader)
        finally:
            transport.close()

    async def _record_init(self, inst: _BwrapSandbox, outer_pid: int, info: asyncio.StreamReader) -> bool:
        """Record the host pid of the exec's namespace init from bwrap's --info-fd JSON.

        bwrap returns as soon as the command (pid 2) exits; its init (pid 1) is killed by
        --die-with-parent right after, and the kernel kills the rest of the namespace
        only when that init exits. The record lets ``exec_processes_terminated`` wait out
        that instant instead of trusting bwrap's own exit. No JSON means bwrap created no
        namespace (it failed first), so there is nothing to record.
        """
        data = b""
        while b"}" not in data and len(data) < 65536:
            chunk = await info.read(4096)
            if not chunk:
                break
            data += chunk
        try:
            payload = json.loads(data.decode(errors="replace")) if data.strip() else None
        except ValueError:
            payload = None
        child = payload.get("child-pid") if isinstance(payload, dict) else None
        if data.strip() and not (isinstance(child, int) and not isinstance(child, bool) and child > 0):
            LOGGER.warning("bwrap --info-fd output has no child-pid: %r", data[:200])
            return False
        if child is None:
            return False
        start_time = _process_start_time(child)
        if start_time is None and _HAVE_PROC:
            return True  # already gone (or a zombie): its namespace is empty
        try:
            (inst.run_dir / f"{outer_pid}{INIT_RECORD_SUFFIX}").write_text(f"{child} {start_time or ''}")
        except OSError:
            # Without the record this instance can no longer prove the namespace gone.
            LOGGER.warning("cannot record bwrap namespace init %s of sandbox %s", child, inst.sandbox_id)
            self._untracked_inits.add(inst.sandbox_id)
        return True

    def _unregister(self, inst: _BwrapSandbox, process: asyncio.subprocess.Process) -> None:
        processes = self._processes.get(inst.sandbox_id)
        if processes is not None:
            processes.pop(process.pid, None)
            if not processes:
                self._processes.pop(inst.sandbox_id, None)
        with contextlib.suppress(OSError):
            (inst.run_dir / str(process.pid)).unlink()
        # The namespace init is usually still dying here; a later check or close() drops its record.
        init = inst.run_dir / f"{process.pid}{INIT_RECORD_SUFFIX}"
        record = _read_init_record(init) if init.exists() else None
        if record is not None and not _pid_may_run(*record):
            with contextlib.suppress(OSError):
                init.unlink()

    async def _kill_process(self, process: asyncio.subprocess.Process) -> None:
        """SIGKILL bwrap's group: --die-with-parent then tears down the sandbox's pid namespace."""
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(process.wait(), timeout=self._config.kill_timeout_s)

    async def _kill_recorded(self, inst: _BwrapSandbox) -> None:
        """Kill every bwrap process recorded for this sandbox, by any provider instance."""
        local = dict(self._processes.get(inst.sandbox_id, {}))
        foreign: dict[int, str] = {}
        try:
            entries = list(inst.run_dir.iterdir())
        except OSError:
            entries = []
        for entry in entries:
            if entry.name.endswith(INIT_RECORD_SUFFIX):
                # A namespace init that outlived its bwrap (normally only for an instant): killing
                # it tears down its pid namespace.
                record = _read_init_record(entry)
                if record is not None and _pid_alive(*record):
                    foreign[record[0]] = record[1]
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        os.kill(record[0], signal.SIGKILL)
                continue
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid in local:
                continue
            try:
                start_time = entry.read_text().strip()
            except OSError:
                continue
            if _pid_alive(pid, start_time):
                foreign[pid] = start_time
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(pid, signal.SIGKILL)
        await asyncio.gather(*(self._kill_process(process) for process in local.values()))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._config.kill_timeout_s
        while foreign and loop.time() < deadline:
            foreign = {pid: start for pid, start in foreign.items() if _pid_alive(pid, start)}
            if foreign:
                await asyncio.sleep(0.05)
        if foreign:
            raise RuntimeError(f"bwrap sandbox {inst.sandbox_id}: processes {sorted(foreign)} survived SIGKILL")

    # ------------------------------------------------------------------ lifecycle

    def _warn_ignored(self, spec: SandboxSpec) -> None:
        ignored = [
            name
            for name, value in (
                ("ttl_s", spec.ttl_s),
                ("resources.cpu", spec.resources.cpu),
                ("resources.memory_mib", spec.resources.memory_mib),
                ("resources.disk_gib", spec.resources.disk_gib),
                ("ports", tuple(spec.ports)),
            )
            if value
        ]
        if ignored:
            LOGGER.warning("bwrap sandbox provider ignores %s", ", ".join(ignored))
        if spec.image and not self._image_warned:
            self._image_warned = True
            LOGGER.warning(
                "bwrap sandboxes use the Gym process's own filesystem as their image; spec.image=%r is ignored",
                spec.image,
            )

    def _prepare_base_dir(self) -> None:
        self._base_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        owner = self._base_dir.stat().st_uid
        if owner != os.getuid():
            raise BwrapCreateError(f"bwrap base_dir {self._base_dir} is owned by uid {owner}, not {os.getuid()}")

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        if spec.entrypoint:
            raise BwrapCreateError("bwrap sandboxes have no long-lived process; spec.entrypoint is unsupported")
        if spec.resources.gpu:
            raise BwrapCreateError("bwrap sandboxes are CPU-only; GPUs are hidden from sandboxed commands")
        self._warn_ignored(spec)
        try:
            workdir = _validate_sandbox_dir(spec.workdir or self._config.workdir, label="workdir")
            env = _validate_env(spec.env, source="spec.env")
            raw_mounts = spec.provider_options.get("mounts") or []
            if isinstance(raw_mounts, (str, Mapping)):
                raw_mounts = [raw_mounts]
            mounts = [BwrapMount.coerce(mount) for mount in raw_mounts]
            self._check_target(workdir, label="workdir")
            for mount in mounts:
                self._check_target(mount.dst, label="mount dst")
            for mount in [*self._config.mounts, *mounts]:
                self._check_mount_dst(mount.dst, workdir=workdir)
        except (TypeError, ValueError) as error:
            raise BwrapCreateError(f"invalid bwrap sandbox spec: {error}") from error
        unknown_options = set(spec.provider_options) - {"mounts"}
        if unknown_options:
            raise BwrapCreateError(f"unsupported bwrap provider_options: {sorted(unknown_options)}")
        for mount in mounts:
            if not os.path.exists(mount.src):
                raise BwrapCreateError(f"bwrap mount source {mount.src!r} does not exist")
        try:
            self._check_own_dirs(workdir)
        except ValueError as error:
            raise BwrapCreateError(str(error)) from error

        sandbox_id = SANDBOX_ID_PREFIX + uuid.uuid4().hex
        try:
            self._prepare_base_dir()
            root = self._base_dir / sandbox_id
            root.mkdir(mode=0o700)
        except OSError as error:
            raise BwrapCreateError(f"cannot create bwrap sandbox directory under {self._base_dir}: {error}") from error
        inst = _BwrapSandbox(
            sandbox_id=sandbox_id, root=root, workdir=workdir, network=self._config.network, env=env, mounts=mounts
        )
        handle = SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=inst)
        try:
            for directory in (inst.workspace, inst.tmp, inst.home, inst.run_dir):
                directory.mkdir(mode=0o700)
            state_tmp = root / f".{STATE_FILE}.tmp"
            state_tmp.write_text(json.dumps(inst.state()))
            state_tmp.replace(root / STATE_FILE)
            for path, content in self._config.default_files.items():
                await asyncio.to_thread(self._write_bytes, inst, path, content.encode("utf-8"), 0o644)
            if self._config.probe_on_create:
                result = await self.exec(handle, "true", timeout_s=self._config.probe_timeout_s)
                if result.return_code != 0 or result.error_type:
                    raise BwrapCreateVerificationError(
                        f"bwrap sandbox {sandbox_id} failed its readiness probe: "
                        f"return_code={result.return_code}, error_type={result.error_type}, "
                        f"stderr={(result.stderr or '').strip()!r}"
                    )
        except BaseException as error:
            with contextlib.suppress(Exception):
                await self.close(handle)
            if isinstance(error, (SandboxCreateError, asyncio.CancelledError)) or not isinstance(error, Exception):
                raise
            raise BwrapCreateError(f"cannot initialize bwrap sandbox {sandbox_id}: {error}") from error
        return handle

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        """Run ``command`` in a fresh bwrap namespace set over this sandbox's directories.

        Processes do not outlive their exec: when the command exits, its pid
        namespace (and every background process in it) is torn down. On timeout or
        cancellation the bwrap process group is SIGKILLed, which also kills the namespace.
        """
        if user is not None:
            raise ValueError("The bwrap sandbox provider runs commands as the Gym process's user only")
        inst: _BwrapSandbox = handle.raw
        if cwd is not None:
            cwd = _validate_sandbox_dir(cwd, label="cwd") if cwd != "/" else "/"
        if self._stopped(inst):
            return self._stopped_result(inst)
        argv = self.bwrap_argv(inst, command, cwd=cwd)
        process_env = self._environment(inst, env)
        timeout = timeout_s if timeout_s is not None else self._config.default_timeout_s
        # From here until its bwrap process is registered (or it returns without one), the exec is pending: queued
        # on exec_concurrency or about to spawn. The record in run/ makes that visible to every provider instance:
        # exec_processes_terminated() is False and close() waits while any live owner has one.
        pending = self._add_pending(inst)
        if pending is None:
            return self._stopped_result(inst)
        try:
            async with self._semaphore if self._semaphore is not None else contextlib.nullcontext():
                # A stop may have run while this exec waited for its slot: never start after it.
                if self._stopped(inst):
                    return self._stopped_result(inst)
                return await self._run(inst, argv, process_env, timeout, pending=pending)
        finally:
            self._remove_pending(pending)

    @staticmethod
    def _stopped(inst: _BwrapSandbox) -> bool:
        return not inst.root.is_dir() or (inst.root / CLOSING_MARKER).exists()

    @staticmethod
    def _stopped_result(inst: _BwrapSandbox) -> SandboxExecResult:
        return SandboxExecResult(
            stdout=None,
            stderr=f"bwrap sandbox {inst.sandbox_id} has been stopped",
            return_code=SANDBOX_RUNTIME_RETURN_CODE,
            error_type="sandbox",
        )

    @staticmethod
    def _add_pending(inst: _BwrapSandbox) -> Path | None:
        """Record a pending exec in run/ (owner pid and its start time); None when the sandbox is gone."""
        path = inst.run_dir / f"{PENDING_RECORD_PREFIX}{uuid.uuid4().hex}"
        try:
            path.write_text(f"{os.getpid()} {_process_start_time(os.getpid()) or ''}")
        except OSError:
            return None
        return path

    @staticmethod
    def _remove_pending(path: Path | None) -> None:
        if path is not None:
            with contextlib.suppress(OSError):
                path.unlink()

    @staticmethod
    def _live_pending(inst: _BwrapSandbox) -> list[Path] | None:
        """Pending-exec records whose owner may still run (records of dead owners are removed); None if unreadable."""
        try:
            entries = [entry for entry in inst.run_dir.iterdir() if entry.name.startswith(PENDING_RECORD_PREFIX)]
        except FileNotFoundError:
            return []
        except OSError:
            return None
        live = []
        for entry in entries:
            record = _read_init_record(entry)
            if record is None or _pid_may_run(*record):
                live.append(entry)  # unreadable counts as live: it may be a record still being written
            else:
                with contextlib.suppress(OSError):
                    entry.unlink()
        return live

    @staticmethod
    def _open_bind_source(path: str | Path) -> int:
        """Open a bind source without following a symlink in any path component."""
        parts = Path(path).parts
        if not parts or parts[0] != "/" or ".." in parts:
            raise BwrapPathError(f"invalid bwrap mount source {str(path)!r}")
        fd = os.open("/", os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            for index, part in enumerate(parts[1:]):
                flags = os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC
                if index < len(parts) - 2:
                    flags |= os.O_DIRECTORY
                next_fd = os.open(part, flags, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            mode = os.fstat(fd).st_mode
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise BwrapPathError(f"bwrap mount source {str(path)!r} is a symlink or special file")
            return fd
        except BaseException as error:
            os.close(fd)
            if isinstance(error, OSError) and error.errno in (errno.ELOOP, errno.ENOTDIR):
                raise BwrapPathError(f"bwrap mount source {str(path)!r} crosses a symlink or non-directory") from error
            raise

    @classmethod
    @contextlib.contextmanager
    def _bind_sources(cls, argv: list[str]) -> Iterator[tuple[list[str], tuple[int, ...]]]:
        """Pin sources until bwrap inherits them; no pathname is reopened during mount setup."""
        with contextlib.ExitStack() as stack:
            pinned, descriptors = [], []
            index = 0
            while index < len(argv):
                option = argv[index]
                if option == "--":
                    pinned.extend(argv[index:])
                    break
                if option in ("--bind", "--ro-bind"):
                    fd = cls._open_bind_source(argv[index + 1])
                    stack.callback(os.close, fd)
                    descriptors.append(fd)
                    pinned.extend((option + "-fd", str(fd), argv[index + 2]))
                    index += 3
                else:
                    pinned.append(option)
                    index += 1
            yield pinned, tuple(descriptors)

    async def _run(
        self,
        inst: _BwrapSandbox,
        argv: list[str],
        env: dict[str, str],
        timeout: float | None,
        *,
        pending: Path | None = None,
    ) -> SandboxExecResult:
        # bwrap writes {"child-pid": <host pid of the namespace init>, ...} to --info-fd.
        info_read, info_write = os.pipe()
        try:
            with self._bind_sources(argv) as (pinned_argv, source_fds):
                process = await asyncio.create_subprocess_exec(
                    pinned_argv[0],
                    "--info-fd",
                    str(info_write),
                    *pinned_argv[1:],
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                    env=env,
                    pass_fds=(info_write, *source_fds),
                )
        except BaseException as error:
            os.close(info_read)
            if not isinstance(error, OSError):
                raise
            return SandboxExecResult(
                stdout=None,
                stderr=f"cannot start bwrap: {error}",
                return_code=SANDBOX_RUNTIME_RETURN_CODE,
                error_type="sandbox",
            )
        finally:
            os.close(info_write)
        self._register(inst, process)
        if pending is not None:
            # Registered before the record goes: a concurrent close() or proof never sees neither.
            self._remove_pending(pending)
        info_task = asyncio.create_task(self._watch_info(inst, process.pid, info_read))
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except (TimeoutError, asyncio.TimeoutError):
            await self._kill_process(process)
            return SandboxExecResult(
                stdout=None,
                stderr=f"bwrap command timed out after {timeout:g}s",
                return_code=SANDBOX_RUNTIME_RETURN_CODE,
                error_type="timeout",
            )
        except BaseException:
            await self._kill_process(process)
            raise
        finally:
            # The init record must exist (or be known absent) before the exec counts as returned.
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(info_task), timeout=self._config.kill_timeout_s)
            if not info_task.done():
                info_task.cancel()
                self._untracked_inits.add(inst.sandbox_id)
            if process.returncode is not None:
                self._unregister(inst, process)
        stderr_text = stderr.decode(errors="replace")
        return_code = process.returncode if process.returncode is not None else SANDBOX_RUNTIME_RETURN_CODE
        command_started = (
            info_task.done() and not info_task.cancelled() and info_task.exception() is None and info_task.result()
        )
        if return_code != 0 and not command_started:
            # bwrap's own setup failure (bad bind, no user namespaces, ...), not the command's: bwrap never reported
            # a namespace init on --info-fd, a channel the command cannot write. Text the command prints, such as
            # a "bwrap: ..." line on stderr, never decides this.
            return SandboxExecResult(
                stdout=stdout.decode(errors="replace"),
                stderr=stderr_text,
                return_code=SANDBOX_RUNTIME_RETURN_CODE,
                error_type="sandbox",
            )
        return SandboxExecResult(
            stdout=stdout.decode(errors="replace"), stderr=stderr_text, return_code=return_code, error_type=None
        )

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        inst: _BwrapSandbox = handle.raw
        if not inst.root.is_dir() or (inst.root / CLOSING_MARKER).exists():
            return SandboxStatus.STOPPED
        if not (inst.root / STATE_FILE).is_file():
            return SandboxStatus.ERROR
        return SandboxStatus.RUNNING

    async def close(self, handle: SandboxHandle) -> None:
        """Kill every bwrap process of the sandbox, then delete its directory. Idempotent."""
        inst: _BwrapSandbox = handle.raw
        if inst.root.is_dir():
            with contextlib.suppress(OSError):
                (inst.root / CLOSING_MARKER).touch()
            await self._kill_recorded(inst)
            # Pending execs of any provider instance: queued ones re-check the marker after their slot, and the
            # others register right after spawning, so wait them out (bounded), then kill.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._config.kill_timeout_s
            while loop.time() < deadline:
                live = await asyncio.to_thread(self._live_pending, inst)
                if live == []:
                    break
                await asyncio.sleep(0.02)
            # A second pass catches an exec that started while the marker was being written.
            await self._kill_recorded(inst)
            try:
                await asyncio.to_thread(_rmtree, inst.root)
            except OSError as error:
                raise RuntimeError(f"cannot remove bwrap sandbox directory {inst.root}: {error}") from error
        self._processes.pop(inst.sandbox_id, None)

    async def aclose(self) -> None:
        """Kill bwrap processes this instance still runs; sandbox directories are left to close()."""
        processes = [process for group in self._processes.values() for process in group.values()]
        await asyncio.gather(*(self._kill_process(process) for process in processes), return_exceptions=True)
        self._processes.clear()

    async def exec_processes_terminated(self, handle: SandboxHandle) -> bool:
        """Whether no process of any exec of this sandbox can still run (SupportsSandboxExecScopedProcesses).

        Every sandboxed process lives in the pid namespace of the exec that started it,
        and the kernel kills a namespace when its init exits. So the proof is: no bwrap
        process recorded by any provider instance is alive, and no recorded namespace
        init is alive (it is killed by --die-with-parent right after its bwrap exits;
        a zombie init has finished killing its namespace). Anything unprovable, such as
        an unreadable record or an init this instance failed to record, is False.
        """
        inst: _BwrapSandbox = handle.raw
        if inst.sandbox_id in self._untracked_inits:
            return False
        if any(process.returncode is None for process in self._processes.get(inst.sandbox_id, {}).values()):
            return False
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._config.kill_timeout_s
        while True:
            live = await asyncio.to_thread(self._recorded_processes_may_run, inst)
            if live is not None and not live:
                return True
            if live is None or loop.time() >= deadline:
                return False
            # An init outlives its bwrap only for an instant: wait it out, bounded.
            await asyncio.sleep(0.05)

    def _recorded_processes_may_run(self, inst: _BwrapSandbox) -> list[str] | None:
        """Records whose process may still run (dead init records are removed); None if unprovable."""
        try:
            entries = list(inst.run_dir.iterdir())
        except OSError:
            return None
        live: list[str] = []
        for entry in entries:
            if entry.name.startswith(PENDING_RECORD_PREFIX):
                # An exec of some provider instance is queued or about to spawn: a process may yet run.
                record = _read_init_record(entry)
                if record is None or _pid_may_run(*record):
                    live.append(entry.name)  # unreadable counts as live: it may be a record still being written
                else:
                    with contextlib.suppress(OSError):
                        entry.unlink()
                continue
            if entry.name.endswith(INIT_RECORD_SUFFIX):
                record = _read_init_record(entry)
                if record is None:
                    return None
                if _pid_may_run(*record):
                    live.append(entry.name)
                else:
                    with contextlib.suppress(OSError):
                        entry.unlink()
                continue
            if not entry.name.isdigit():
                continue
            try:
                start_time = entry.read_text().strip()
            except OSError:
                return None
            if _pid_may_run(int(entry.name), start_time):
                live.append(entry.name)
        return live

    # ------------------------------------------------------------------ connect

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        """A same-host, same-uid descriptor; ``connect`` reads the rest from the sandbox's state file.

        Like apptainer's, this is a trusted control-plane descriptor, not a lease:
        either side can stop the sandbox, so callers coordinate ownership.
        """
        inst: _BwrapSandbox = handle.raw
        return {
            "provider": self.name,
            "host": socket.gethostname(),
            "uid": os.getuid(),
            "base_dir": str(self._base_dir),
            "sandbox_id": inst.sandbox_id,
            "workdir": inst.workdir,
            "network": inst.network,
        }

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        if descriptor.get("provider") != self.name:
            raise ValueError("bwrap connect requires a descriptor produced by the bwrap provider")
        if descriptor.get("host") != socket.gethostname() or descriptor.get("uid") != os.getuid():
            raise ValueError("bwrap connect requires the same host and uid as the sandbox's creator")
        base_dir = descriptor.get("base_dir")
        if not isinstance(base_dir, str) or Path(os.path.realpath(base_dir)) != self._base_dir:
            raise ValueError(f"bwrap descriptor base_dir {base_dir!r} differs from this provider's {self._base_dir}")
        sandbox_id = descriptor.get("sandbox_id")
        if not isinstance(sandbox_id, str) or SANDBOX_ID_PATTERN.fullmatch(sandbox_id) is None:
            raise ValueError(f"invalid bwrap sandbox id {sandbox_id!r}")
        root = self._base_dir / sandbox_id
        if not root.is_dir() or (root / CLOSING_MARKER).exists():
            raise ValueError(f"bwrap sandbox {sandbox_id} does not exist or has been stopped")
        try:
            state = json.loads((root / STATE_FILE).read_text())
        except (OSError, ValueError) as error:
            raise ValueError(f"bwrap sandbox {sandbox_id} has no readable state file") from error
        if state.get("sandbox_id") != sandbox_id:
            raise ValueError(f"bwrap sandbox {sandbox_id} state file does not match its directory")
        # The descriptor's workdir is the facade's current default cwd (set_default_workdir may move it); the
        # workspace bind destination comes from the host-only state file below.
        for key in ("network",):
            if key in descriptor and descriptor[key] is not None and descriptor[key] != state.get(key):
                raise ValueError(f"bwrap descriptor {key} {descriptor[key]!r} does not match the sandbox's")
        inst = _BwrapSandbox(
            sandbox_id=sandbox_id,
            root=root,
            workdir=_validate_sandbox_dir(state["workdir"], label="workdir"),
            network=state["network"] if state.get("network") in ("none", "host") else "none",
            env=_validate_env(state.get("env") or {}, source="sandbox state env"),
            mounts=[BwrapMount.coerce(mount) for mount in state.get("mounts") or []],
        )
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=inst)

    # ------------------------------------------------------------------ files

    def _mount_table(self, inst: _BwrapSandbox) -> list[tuple[str, Path, bool]]:
        """(sandbox prefix, host path, writable) of the binds a command sees, most specific prefix first.

        Built in bind order (as ``bwrap_argv`` applies them): a later bind at or above an earlier one's prefix
        hides it, so its entry is dropped and transfers resolve to the visible bind. A regular-file mount's host
        path is the resolved file.
        """
        table = [
            ("/tmp", inst.tmp, True),
            (posixpath.normpath(self._config.home_path), inst.home, True),
            (inst.workdir, inst.workspace, True),
        ]
        if self._binds_workspace_host_path(inst):
            table.append((str(inst.workspace), inst.workspace, True))
        for mount in [*self._config.mounts, *inst.mounts]:
            src = Path(mount.src)
            table.append((posixpath.normpath(mount.dst), src, not mount.ro))
        effective: list[tuple[str, Path, bool]] = []
        for entry in table:
            prefix = entry[0]
            effective = [
                previous
                for previous in effective
                if not (previous[0] == prefix or previous[0].startswith(prefix.rstrip("/") + "/"))
            ]
            effective.append(entry)
        return sorted(effective, key=lambda entry: len(entry[0]), reverse=True)

    def host_path(self, inst: _BwrapSandbox, path: str, *, write: bool) -> tuple[Path, tuple[str, ...]]:
        """Map a sandbox path to (host root, relative parts) under one of its writable mounts.

        Relative paths are relative to the workdir. ``..`` components are refused, as
        are paths outside the sandbox's own directories and extra mounts.
        """
        if not isinstance(path, str) or not path or "\x00" in path:
            raise BwrapPathError(f"invalid sandbox path {path!r}")
        absolute = path if path.startswith("/") else posixpath.join(inst.workdir, path)
        if ".." in PurePosixPath(absolute).parts:
            raise BwrapPathError(f"sandbox path {path!r} contains '..'")
        normalized = posixpath.normpath(absolute)
        if normalized.startswith("//"):
            normalized = "/" + normalized.lstrip("/")
        for prefix, root, writable in self._mount_table(inst):
            if normalized == prefix or normalized.startswith(prefix.rstrip("/") + "/"):
                if write and not writable:
                    raise BwrapPathError(f"sandbox path {path!r} is on a read-only mount")
                if root.is_file():
                    # A regular-file mount: only its exact destination addresses it.
                    if normalized != prefix:
                        raise BwrapPathError(f"sandbox path {path!r} is below a file mount")
                    return root.parent, (root.name,)
                relative = normalized[len(prefix) :].strip("/")
                return root, tuple(part for part in relative.split("/") if part)
        raise BwrapPathError(f"sandbox path {path!r} is outside the sandbox's workspace, /tmp, home and mounts")

    @staticmethod
    def _open_parent(root: Path, parts: tuple[str, ...], *, create: bool) -> int:
        """Open the directory holding ``parts[-1]`` without following any symlink below ``root``."""
        fd = BwrapProvider._open_bind_source(root)
        try:
            for part in parts[:-1]:
                try:
                    next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                except FileNotFoundError:
                    if not create:
                        raise
                    with contextlib.suppress(FileExistsError):
                        os.mkdir(part, 0o755, dir_fd=fd)
                    next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = next_fd
        except OSError as error:
            os.close(fd)
            if error.errno in (errno.ELOOP, errno.ENOTDIR):
                raise BwrapPathError(f"refusing to traverse a symlink or non-directory under {root}") from error
            raise
        return fd

    def _write_bytes(self, inst: _BwrapSandbox, path: str, data: bytes, mode: int) -> None:
        root, parts = self.host_path(inst, path, write=True)
        if not parts:
            raise IsADirectoryError(f"sandbox path {path!r} is a mount root")
        parent = self._open_parent(root, parts, create=True)
        try:
            try:
                # O_NONBLOCK and no O_TRUNC: a FIFO the sandbox planted here must neither block this worker thread
                # nor be written to, and nothing is truncated before the file is known to be a regular file.
                fd = os.open(
                    parts[-1], os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, mode & 0o777, dir_fd=parent
                )
            except OSError as error:
                if error.errno == errno.ELOOP:
                    raise BwrapPathError(f"refusing to write through symlink {path!r}") from error
                if error.errno == errno.ENXIO:  # a FIFO with no reader
                    raise BwrapPathError(f"sandbox path {path!r} is not a regular file") from error
                raise
        finally:
            os.close(parent)
        _regular_file_or_close(fd, path)
        os.ftruncate(fd, 0)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)

    def _upload(self, inst: _BwrapSandbox, source_path: Path, target_path: str) -> None:
        mode = source_path.stat().st_mode
        self._write_bytes(inst, target_path, source_path.read_bytes(), mode)

    def _download(self, inst: _BwrapSandbox, source_path: str, target_path: Path) -> None:
        root, parts = self.host_path(inst, source_path, write=False)
        if not parts:
            raise IsADirectoryError(f"sandbox path {source_path!r} is a mount root")
        parent = self._open_parent(root, parts, create=False)
        try:
            try:
                # O_NONBLOCK: a blocking open of a FIFO the sandbox planted here would hold this worker thread
                # forever (cancelling the coroutine does not free it).
                fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            except OSError as error:
                if error.errno == errno.ELOOP:
                    raise BwrapPathError(f"refusing to read through symlink {source_path!r}") from error
                raise
        finally:
            os.close(parent)
        _regular_file_or_close(fd, source_path)
        with os.fdopen(fd, "rb") as stream:
            data = stream.read()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(data)

    def _exists(self, inst: _BwrapSandbox, path: str) -> bool:
        root, parts = self.host_path(inst, path, write=False)
        if not parts:
            return root.is_dir()
        try:
            parent = self._open_parent(root, parts, create=False)
        except (FileNotFoundError, BwrapPathError):
            return False
        try:
            os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return False
        finally:
            os.close(parent)
        return True

    def uploads_independent_of_execs(self) -> bool:
        # Uploads are host-side copies in a worker thread; only commands take the exec semaphore.
        return True

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        await asyncio.to_thread(self._upload, handle.raw, Path(source_path), target_path)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        await asyncio.to_thread(self._download, handle.raw, source_path, Path(target_path))

    async def exists(self, handle: SandboxHandle, path: str) -> bool:
        """Whether ``path`` exists in the sandbox (bwrap-specific; not part of the provider protocol)."""
        return await asyncio.to_thread(self._exists, handle.raw, path)
