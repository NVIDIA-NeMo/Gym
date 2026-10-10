# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional service networking through the deployment's WebSocket ingress."""

import asyncio
import ipaddress
import json
import math
import re
import secrets
import shlex
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from nemo_gym.sandbox.providers.base import SandboxEndpoint


@dataclass
class E2BEndpointConfig:
    url_template: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class E2BNetworkingConfig:
    enabled: bool = False
    address_cidr: str = "198.18.0.0/15"
    python_executable: str = "python3"
    setup_command: str | None = None
    tunnel_port: int = 49152
    startup_timeout_s: float = 180

    def __post_init__(self):
        network = ipaddress.IPv4Network(self.address_cidr)
        if network.is_loopback or network.is_multicast or network.prefixlen > 30:
            raise ValueError("networking.address_cidr must provide non-loopback unicast addresses")
        if (
            isinstance(self.tunnel_port, bool)
            or not isinstance(self.tunnel_port, int)
            or not 1 <= self.tunnel_port <= 65535
        ):
            raise ValueError("networking.tunnel_port must be between 1 and 65535")
        if not math.isfinite(self.startup_timeout_s) or self.startup_timeout_s <= 0:
            raise ValueError("networking.startup_timeout_s must be positive")


@dataclass
class E2BRuntimeRequirementsConfig:
    capability_probes: dict[str, str] = field(default_factory=dict)
    resize_shared_memory: bool = False


class E2BComposeSupport:
    async def endpoint(self, handle, port):
        values = {"sandbox_id": handle.sandbox_id, "port": port}
        url = self._endpoints.url_template
        return SandboxEndpoint(
            endpoint=url.format(**values) if url else "https://" + self._sandbox(handle).get_host(port),
            headers={key: value.format(**values) for key, value in self._endpoints.headers.items()},
        )

    def validate_networking(self):
        if not self._networking.enabled:
            raise NotImplementedError("E2B Compose requires networking.enabled=true and WebSocket ingress")

    def validate_port_forwarding(self):
        self.validate_networking()

    def _register_network(self, handle, spec):
        used_ports = {port for member in self._network_members.values() for port in member["ports"]}
        if used_ports.intersection(spec.ports):
            raise NotImplementedError("E2B relays require distinct declared TCP ports across services")
        network = ipaddress.IPv4Network(self._networking.address_cidr)
        if self._network_index >= network.num_addresses - 2:
            raise RuntimeError("E2B networking address pool exhausted; create a new provider")
        self._network_members[handle.sandbox_id] = {
            "handle": handle,
            "address": str(network.network_address + self._network_index + 1),
            "ports": list(spec.ports),
            "token": secrets.token_urlsafe(32),
        }
        self._network_index += 1

    async def _peer(self, member, addresses=None):
        endpoint = await self.endpoint(member["handle"], self._networking.tunnel_port)
        return {
            "addresses": addresses or [member["address"]],
            "ports": member["ports"],
            "url": endpoint.endpoint.replace("https://", "wss://", 1).replace("http://", "ws://", 1),
            "headers": {**endpoint.headers, "X-Gym-Tunnel-Token": member["token"]},
        }

    async def _run_tunnel(self, handle, config):
        from nemo_gym.sandbox.providers.e2b import _tunnel

        root = "/tmp/gym-tunnel-" + uuid.uuid4().hex
        result = await self.exec(handle, f"mkdir -m 700 {root}", user="root", timeout_s=30)
        if result.return_code:
            raise RuntimeError("Cannot create E2B tunnel directory")
        config.setdefault("ready_file", root + "/ready")
        await self.write_file(handle, root + "/config.json", json.dumps(config))
        await self.write_file(handle, root + "/tunnel.py", Path(_tunnel.__file__).read_text())
        command = shlex.join([self._networking.python_executable, root + "/tunnel.py", root + "/config.json"])
        task = asyncio.create_task(self.exec(handle, command, user="root", timeout_s=0))
        self._network_tasks.setdefault(handle.sandbox_id, []).append(task)
        async with asyncio.timeout(self._networking.startup_timeout_s):
            while (
                await self.exec(handle, "test -f " + shlex.quote(config["ready_file"]), user="root", timeout_s=30)
            ).return_code:
                if task.done():
                    result = await task
                    raise RuntimeError(f"E2B tunnel exited during startup: {result.stderr}")
                await asyncio.sleep(0.1)
        return task

    async def network_address(self, handle):
        self.validate_networking()
        member = self._network_members[handle.sandbox_id]
        if handle.sandbox_id not in self._network_tasks:
            if self._networking.setup_command:
                result = await self.exec(
                    handle, self._networking.setup_command, user="root", timeout_s=self._networking.startup_timeout_s
                )
                if result.return_code:
                    raise RuntimeError(f"E2B networking setup failed: {result.stderr}")
            peers = [await self._peer(peer) for peer in self._network_members.values() if peer is not member]
            await self._run_tunnel(
                handle,
                {
                    "token": member["token"],
                    "ports": member["ports"],
                    "tunnel_port": self._networking.tunnel_port,
                    "peers": peers,
                    "local_addresses": [peer["address"] for peer in self._network_members.values()],
                },
            )
        return member["address"]

    async def set_hosts(self, handle, hosts):
        self.validate_networking()
        entries = []
        for name, address in hosts.items():
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,252}", name):
                raise ValueError(f"Invalid sandbox hostname: {name!r}")
            entries.append(shlex.quote(f"{ipaddress.ip_address(address)} {name}\n"))
        if entries:
            result = await self.exec(handle, "printf '%s' " + " ".join(entries) + " >> /etc/hosts", user="root")
            if result.return_code:
                raise RuntimeError(f"Could not configure E2B hosts: {result.stderr}")

    async def forward_ports(self, handle, target_address, ports, *, ready_file):
        self.validate_port_forwarding()
        member = next(peer for peer in self._network_members.values() if peer["address"] == target_address)
        peer = await self._peer(member, ["127.0.0.1", "::1"])
        peer["ports"] = list(ports)
        task = await self._run_tunnel(handle, {"peers": [peer], "ready_file": ready_file})
        result = await task
        raise RuntimeError(f"E2B port forwarding exited: {result.stderr}")

    def validate_runtime_requirements(self, *, cap_add, shm_size):
        for capability in cap_add:
            if not self._runtime_requirements.capability_probes.get(capability, "").strip():
                raise NotImplementedError(f"E2B requires a capability probe for {capability!r}")
        if shm_size is not None:
            if isinstance(shm_size, bool) or not isinstance(shm_size, int) or shm_size <= 0:
                raise ValueError("shm_size must be a positive number of bytes")
            if not self._runtime_requirements.resize_shared_memory:
                raise NotImplementedError("E2B shm_size requires runtime_requirements.resize_shared_memory=true")

    async def configure_runtime(self, handle, *, cap_add, shm_size):
        self.validate_runtime_requirements(cap_add=cap_add, shm_size=shm_size)
        commands = [(capability, self._runtime_requirements.capability_probes[capability]) for capability in cap_add]
        if shm_size is not None:
            check = (
                "set -- $(stat -fc '%S %b' /dev/shm); "
                f"expected=$((({shm_size} + $1 - 1) / $1 * $1)); "
                '[ "$(($1 * $2))" -eq "$expected" ]'
            )
            commands.append(("shm_size", f"mount -o remount,size={shm_size} /dev/shm && {check}"))
        for requirement, command in commands:
            result = await self.exec(handle, command, user="root", timeout_s=60)
            if result.return_code:
                raise RuntimeError(f"E2B cannot satisfy {requirement}: {result.stderr}")

    async def _close_network(self, sandbox_id):
        tasks = self._network_tasks.pop(sandbox_id, [])
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._network_members.pop(sandbox_id, None)
