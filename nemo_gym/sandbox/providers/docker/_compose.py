# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker implementations of the optional Compose provider interfaces."""

import ipaddress
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from nemo_gym.sandbox.providers.base import SandboxSpec


_RUNTIME_METADATA = "nemo-gym.docker.runtime"


@dataclass(frozen=True)
class DockerNetworkingConfig:
    loopback_forwarding: bool = False
    python_executable: str = "python3"
    setup_command: str | None = None


@dataclass(frozen=True)
class DockerSharedStorageConfig:
    host_path: str | None = None


class DockerComposeSupport:
    async def _inspect_container(self, name):
        code, out, err = await self._run(
            [self._binary, "inspect", "--format", "{{json .}}", name],
            timeout_s=self._exec_config.default_timeout_s,
        )
        if code:
            raise RuntimeError(f"Cannot inspect Docker sandbox {name!r}: {err.strip()}")
        return json.loads(out)

    def validate_networking(self):
        network = self._create_config.network or "bridge"
        if network in {"none", "host"} or network.startswith("container:"):
            raise NotImplementedError("Docker Compose requires a network with peer-reachable container addresses")

    async def network_address(self, handle):
        self.validate_networking()
        info = await self._inspect_container(handle.raw.name)
        networks = (info.get("NetworkSettings") or {}).get("Networks") or {}
        selected = networks.get(self._create_config.network or "bridge")
        # Docker accepts a network ID at create time, but inspect keys by name.
        if selected is None and len(networks) == 1:
            selected = next(iter(networks.values()))
        address = ipaddress.ip_address(
            (selected or {}).get("IPAddress") or (selected or {}).get("GlobalIPv6Address") or ""
        )
        if address.is_loopback or address.is_unspecified or address.is_multicast:
            raise ValueError("Docker Compose requires a peer-reachable container address")
        return str(address)

    async def set_hosts(self, handle, hosts):
        self.validate_networking()
        entries = []
        for name, address in hosts.items():
            if len(name) > 253 or not all(
                re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9_-]{0,61}[A-Za-z0-9])?", label) for label in name.split(".")
            ):
                raise ValueError(f"Invalid sandbox hostname: {name!r}")
            entries.append(shlex.quote(f"{ipaddress.ip_address(address)} {name}\n"))
        if entries:
            result = await self.exec(handle, "printf '%s' " + " ".join(entries) + " >> /etc/hosts", user="root")
            if result.return_code:
                raise RuntimeError(f"Cannot configure Docker service aliases: {result.stderr}")

    def validate_runtime_requirements(self, *, cap_add, shm_size):
        if shm_size is not None and (isinstance(shm_size, bool) or not isinstance(shm_size, int) or shm_size <= 0):
            raise ValueError("shm_size must be a positive number of bytes")
        if any(not isinstance(cap, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", cap) for cap in cap_add):
            raise ValueError("cap_add must contain Docker capability names")
        # Provider-owned metadata carries native create flags through the shared
        # validation hook, without teaching the adapter about Docker arguments.
        return {_RUNTIME_METADATA: json.dumps({"cap_add": list(cap_add), "shm_size": shm_size})}

    def _runtime_flags(self, spec):
        if _RUNTIME_METADATA not in spec.metadata:
            return []
        requirements = json.loads(spec.metadata[_RUNTIME_METADATA])
        self.validate_runtime_requirements(**requirements)
        flags = [arg for cap in requirements["cap_add"] for arg in ("--cap-add", cap)]
        if requirements["shm_size"] is not None:
            flags += ["--shm-size", str(requirements["shm_size"])]
        return flags

    async def configure_runtime(self, handle, *, cap_add, shm_size):
        self.validate_runtime_requirements(cap_add=cap_add, shm_size=shm_size)
        config = (await self._inspect_container(handle.raw.name))["HostConfig"]
        if shm_size is not None and config.get("ShmSize") != shm_size:
            raise RuntimeError("Docker did not apply the requested shm_size")
        actual = {cap.removeprefix("CAP_") for cap in config.get("CapAdd") or []}
        if "ALL" not in actual and not {cap.removeprefix("CAP_") for cap in cap_add} <= actual:
            raise RuntimeError("Docker did not apply the requested cap_add")

    def shared_volume_metadata(self):
        if not self._shared_storage.host_path:
            raise NotImplementedError("Docker Compose volumes require shared_storage.host_path on the daemon host")
        return {}

    def shared_volume_options(self, source, target, *, read_only=False):
        self.shared_volume_metadata()
        root = self._shared_storage.host_path
        for path in (root, target):
            if not PurePosixPath(path).is_absolute() or ".." in path.split("/") or any(c in path for c in ":\x00\\"):
                raise ValueError(
                    "Docker shared paths must be absolute POSIX paths without traversal or volume separators"
                )
        if source is not None and (
            not source
            or not PurePosixPath(source).parts
            or PurePosixPath(source).is_absolute()
            or ".." in source.split("/")
            or any(c in source for c in ":\x00\\")
        ):
            raise ValueError("Docker shared source must be a safe relative path")
        host = str(PurePosixPath(root) / source) if source is not None else root
        return {"volumes": [f"{host}:{target}" + (":ro" if read_only else "")]}

    def validate_port_forwarding(self):
        self.validate_networking()
        if not self._networking.loopback_forwarding:
            raise NotImplementedError("Docker loopback forwarding requires networking.loopback_forwarding=true")

    async def forward_ports(self, handle, target_address, ports, *, ready_file):
        from nemo_gym.sandbox.providers import _port_forward

        self.validate_port_forwarding()
        address = str(ipaddress.ip_address(target_address))
        ports = SandboxSpec(ports=ports).ports
        if self._networking.setup_command:
            result = await self.exec(handle, self._networking.setup_command, user="root", timeout_s=180)
            if result.return_code:
                raise RuntimeError(f"Docker forwarding setup failed: {result.stderr}")
        # A relay lasts for the collection's lifetime and must not occupy an exec
        # slot needed by health checks or cleanup (including concurrency=1).
        code, out, err = await self._run(
            [
                self._binary,
                "exec",
                "--user",
                "0",
                handle.raw.name,
                self._networking.python_executable,
                "-u",
                "-c",
                Path(_port_forward.__file__).read_text(),
                address,
                ready_file,
                *(str(port) for port in ports),
            ],
            timeout_s=None,
            bounded=False,
        )
        raise RuntimeError(f"Docker forwarding exited ({code}): {out}; {err}")
