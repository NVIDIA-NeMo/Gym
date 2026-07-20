# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OSWorld provider compatibility layer backed by :mod:`nemo_gym.sandbox`.

OSWorld keeps ownership of ``DesktopEnv``, controllers, task setup, actions,
and evaluation. This adapter moves only the VM container lifecycle, dynamic
port publication, and cleanup into Gym's provider-neutral Sandbox API.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import os
import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import requests

from nemo_gym.sandbox import Sandbox, SandboxEndpoint, SandboxSpec, SandboxStatus


LOG = logging.getLogger("nemo_gym.osworld_agent.sandbox_provider")

OSWORLD_SERVICE_PORTS = (5000, 9222, 8006, 8080)
OSWORLD_IMAGE_ENTRYPOINT = ("/usr/bin/tini", "-s", "/run/entry.sh")
OSWORLD_QCOW2_MOUNT = "/System.qcow2"
OSWORLD_WORKLOAD_LABEL = "nemo-gym.workload=osworld"
OSWORLD_RUN_ID_LABEL = "nemo-gym.run-id"


def _string_list(value: Any, *, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return list(value)
    raise TypeError(f"{field} must be a string or list of strings")


def _has_option(args: list[str], flag: str, value: str) -> bool:
    return any(
        (item == flag and index + 1 < len(args) and args[index + 1] == value)
        or item == f"{flag}={value}"
        for index, item in enumerate(args)
    )


def _parse_plain_http_endpoint(resolved: SandboxEndpoint, port: int) -> tuple[str, int]:
    """Return the host/port shape accepted by unmodified OSWorld controllers."""

    if resolved.headers:
        raise ValueError(
            f"Sandbox endpoint for OSWorld port {port} requires headers; "
            "the current OSWorld controllers support direct endpoints only"
        )
    parsed = urlsplit(resolved.endpoint)
    if parsed.scheme != "http":
        raise ValueError(
            f"Sandbox endpoint for OSWorld port {port} must use direct HTTP, "
            f"got scheme {parsed.scheme!r}"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError(
            f"Sandbox endpoint for OSWorld port {port} must be a plain origin without credentials, path, or query"
        )
    if parsed.hostname is None or parsed.port is None:
        raise ValueError(f"Sandbox endpoint for OSWorld port {port} has no host/port")
    return parsed.hostname, parsed.port


def _http_origin(host: str, port: int) -> str:
    formatted_host = f"[{host}]" if ":" in host else host
    return f"http://{formatted_host}:{port}"


class GymSandboxDesktopProvider:
    """Implement OSWorld's provider contract with one Gym Docker Sandbox per VM."""

    def __init__(
        self,
        sandbox_provider: Mapping[str, Any],
        sandbox_spec: Mapping[str, Any],
        *,
        require_kvm: bool = True,
        ready_timeout_s: float = 600.0,
        ready_poll_s: float = 2.0,
    ) -> None:
        if not isinstance(sandbox_provider, Mapping) or len(sandbox_provider) != 1:
            raise ValueError("sandbox_provider must be a single-key Gym Sandbox provider config")
        if not isinstance(sandbox_spec, Mapping):
            raise TypeError("sandbox_spec must be a mapping")
        if ready_timeout_s <= 0:
            raise ValueError("ready_timeout_s must be > 0")
        if ready_poll_s <= 0:
            raise ValueError("ready_poll_s must be > 0")

        self._sandbox_provider = copy.deepcopy(dict(sandbox_provider))
        self._sandbox_provider_name = str(next(iter(self._sandbox_provider))).lower().strip()
        if self._sandbox_provider_name != "docker":
            raise ValueError(
                "The OSWorld Gym Sandbox deployment requires Gym's Docker provider, "
                f"got {self._sandbox_provider_name!r}"
            )
        self._sandbox_spec = copy.deepcopy(dict(sandbox_spec))
        self._require_kvm = bool(require_kvm)
        self._ready_timeout_s = float(ready_timeout_s)
        self._ready_poll_s = float(ready_poll_s)
        self._sandbox: Sandbox | None = None
        self._host: str | None = None
        self.server_port: int | None = None
        self.chromium_port: int | None = None
        self.vnc_port: int | None = None
        self.vlc_port: int | None = None

    def _build_spec(self, path_to_vm: str, *, headless: bool, os_type: str) -> SandboxSpec:
        if os_type.lower() not in {"ubuntu", "linux"}:
            raise ValueError(f"Gym Sandbox OSWorld adapter currently supports Ubuntu only, got {os_type!r}")
        vm_path = os.path.realpath(os.path.abspath(os.path.expanduser(path_to_vm)))
        if not os.path.isfile(vm_path) or not os.access(vm_path, os.R_OK):
            raise FileNotFoundError(f"OSWorld base qcow2 is not readable: {vm_path}")
        if (
            self._require_kvm
            and (not os.path.exists("/dev/kvm") or not os.access("/dev/kvm", os.R_OK | os.W_OK))
        ):
            raise RuntimeError("OSWorld Gym Sandbox requires readable/writable /dev/kvm")

        values = copy.deepcopy(self._sandbox_spec)
        if not values.get("image"):
            raise ValueError("sandbox_spec.image is required for OSWorld")

        environment = dict(values.get("env") or {})
        environment.setdefault("DISK_SIZE", "32G")
        environment.setdefault("RAM_SIZE", "4G")
        environment.setdefault("CPU_CORES", "4")
        environment.setdefault("HEADLESS", "Y" if headless else "N")
        environment["KVM"] = "Y" if self._require_kvm else "N"
        values["env"] = environment
        values.setdefault("entrypoint", list(OSWORLD_IMAGE_ENTRYPOINT))
        values["ports"] = list(dict.fromkeys([*(values.get("ports") or ()), *OSWORLD_SERVICE_PORTS]))

        metadata = dict(values.get("metadata") or {})
        metadata.setdefault("workload", "osworld")
        metadata.setdefault("osworld-provider", "gym-docker-sandbox")
        values["metadata"] = metadata

        provider_options = dict(values.get("provider_options") or {})
        volumes = _string_list(provider_options.get("volumes"), field="volumes")
        if any(f":{OSWORLD_QCOW2_MOUNT}" in volume for volume in volumes):
            raise ValueError(f"sandbox_spec already mounts {OSWORLD_QCOW2_MOUNT}; the adapter owns this mount")
        volumes.append(f"{vm_path}:{OSWORLD_QCOW2_MOUNT}:ro")
        provider_options["volumes"] = volumes

        run_args = _string_list(provider_options.get("run_args"), field="run_args")
        if not _has_option(run_args, "--label", OSWORLD_WORKLOAD_LABEL):
            run_args.extend(["--label", OSWORLD_WORKLOAD_LABEL])
        run_id = os.environ.get("OSWORLD_RUN_ID", "").strip()
        run_id_label = f"{OSWORLD_RUN_ID_LABEL}={run_id}"
        if run_id and not _has_option(run_args, "--label", run_id_label):
            run_args.extend(["--label", run_id_label])
        if not _has_option(run_args, "--cap-add", "NET_ADMIN"):
            run_args.extend(["--cap-add", "NET_ADMIN"])
        if self._require_kvm and not _has_option(run_args, "--device", "/dev/kvm"):
            run_args.extend(["--device", "/dev/kvm"])
        provider_options["run_args"] = run_args

        values["provider_options"] = provider_options
        return SandboxSpec(**values)

    def _resolve_service_endpoints(self, sandbox: Sandbox) -> tuple[str, dict[int, int]]:
        host: str | None = None
        resolved_ports: dict[int, int] = {}
        for container_port in OSWORLD_SERVICE_PORTS:
            endpoint_host, endpoint_port = _parse_plain_http_endpoint(
                sandbox.endpoint(container_port),
                container_port,
            )
            if host is None:
                host = endpoint_host
            elif endpoint_host != host:
                raise ValueError(
                    "OSWorld requires all Sandbox service endpoints to share one host; "
                    f"got {host!r} and {endpoint_host!r}"
                )
            resolved_ports[container_port] = endpoint_port
        if host is None:
            raise RuntimeError("Gym Sandbox returned no OSWorld service endpoints")
        return host, resolved_ports

    def _wait_for_vm_ready(self, sandbox: Sandbox, host: str, server_port: int) -> None:
        deadline = time.monotonic() + self._ready_timeout_s
        last_error = "guest readiness was not attempted"
        with requests.Session() as session:
            while time.monotonic() < deadline:
                try:
                    response = session.get(
                        f"{_http_origin(host, server_port)}/screenshot",
                        timeout=(5.0, 10.0),
                    )
                    if response.status_code == 200 and response.content:
                        return
                    last_error = f"HTTP {response.status_code}, bytes={len(response.content)}"
                except requests.RequestException as exc:
                    last_error = f"{type(exc).__name__}: {exc}"

                status = sandbox.status()
                if status in {SandboxStatus.ERROR, SandboxStatus.STOPPED}:
                    raise RuntimeError(
                        f"Gym Sandbox stopped before the OSWorld guest became ready: status={status.value}"
                    )
                time.sleep(min(self._ready_poll_s, max(deadline - time.monotonic(), 0.0)))
        raise TimeoutError(
            f"OSWorld guest did not become ready within {self._ready_timeout_s:g}s: {last_error}"
        )

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str) -> None:
        if self._sandbox is not None:
            raise RuntimeError("OSWorld Gym Sandbox emulator is already running")

        sandbox = Sandbox(self._sandbox_provider)
        try:
            sandbox.start(self._build_spec(path_to_vm, headless=headless, os_type=os_type))
            host, ports = self._resolve_service_endpoints(sandbox)
            self._wait_for_vm_ready(sandbox, host, ports[5000])
        except BaseException:
            # Preserve the startup failure if best-effort cleanup also fails.
            with contextlib.suppress(Exception):
                sandbox.stop()
            raise

        self._sandbox = sandbox
        self._host = host
        self.server_port = ports[5000]
        self.chromium_port = ports[9222]
        self.vnc_port = ports[8006]
        self.vlc_port = ports[8080]
        LOG.info("OSWorld guest is ready in Gym Sandbox provider=%s", self._sandbox_provider_name)

    def get_ip_address(self, path_to_vm: str) -> str:
        del path_to_vm
        if self._host is None or None in {
            self.server_port,
            self.chromium_port,
            self.vnc_port,
            self.vlc_port,
        }:
            raise RuntimeError("OSWorld Gym Sandbox emulator has not started")
        host = f"[{self._host}]" if ":" in self._host else self._host
        return f"{host}:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}"

    def save_state(self, path_to_vm: str, snapshot_name: str) -> None:
        del path_to_vm, snapshot_name
        raise NotImplementedError(
            "Live VM snapshots are not available for Gym Sandbox OSWorld; "
            "the read-only qcow2 base is restored by recreating the sandbox"
        )

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str) -> None:
        del snapshot_name
        self.stop_emulator(path_to_vm)

    def stop_emulator(self, path_to_vm: str, region: str | None = None, *args: Any, **kwargs: Any) -> None:
        del path_to_vm, region, args, kwargs
        sandbox = self._sandbox
        self._sandbox = None
        self._host = None
        self.server_port = None
        self.chromium_port = None
        self.vnc_port = None
        self.vlc_port = None
        if sandbox is not None:
            sandbox.stop()
