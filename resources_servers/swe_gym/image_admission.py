# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted-host admission for SWE-Gym's optional offline agent images."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from resources_servers.swe_gym.clean_image import SCANNER_VERSION


@dataclass(frozen=True)
class AdmittedImage:
    image_id: str
    agent_base_commit: str
    rootfs_sha256: str
    scanner_version: int = SCANNER_VERSION


def admit_agent_image(
    *, manifest_path: str, task_id: str, base_commit: str, provider_config: dict, sandbox_config: dict
) -> AdmittedImage:
    """Reject unsupported controls before allocation. Manifests are trusted host configuration.

    Docker's isolated network namespace blocks direct IP and upstream DNS. No arbitrary run/exec flags,
    mounts or network-enabled providers are accepted. The agent harness must run outside the sandbox.

    Manifests are per Docker daemon: importing the same rootfs elsewhere can yield a
    different image ID because Docker records import time. Import and bind a manifest
    on each deployment daemon. rootfs_sha256 identifies the portable artifact.
    """
    if set(provider_config) != {"docker"}:
        raise ValueError("Audited agent images currently require the offline Docker provider")
    config = provider_config["docker"] or {}
    create = config.get("create", {})
    if (
        create.get("network") != "none"
        or create.get("cap_drop") != ["ALL"]
        or create.get("security_opt") != ["no-new-privileges"]
        or not 1 <= create.get("pids_limit", 0) <= 512
        or not create.get("apply_resource_limits", True)
        or create.get("extra_run_args")
        or config.get("exec", {}).get("extra_exec_args")
    ):
        raise ValueError(
            "Offline Docker requires network=none, ALL capabilities dropped, no-new-privileges, "
            "pids_limit<=512, resource limits and no raw flags"
        )
    if sandbox_config.get("provider_options"):
        raise ValueError("Audited agent images do not permit provider options or host mounts")
    if not sandbox_config.get("ttl_s") or sandbox_config["ttl_s"] <= 0:
        raise ValueError("Audited agent images require a positive ttl_s")
    resources = sandbox_config.get("resources", {})
    if not resources.get("cpu") or not resources.get("memory_mib") or resources.get("gpu"):
        raise ValueError("Audited agent images require CPU/memory limits and no GPU")
    manifest = json.loads(Path(manifest_path).read_text())
    if (
        manifest.get("scanner_version") != SCANNER_VERSION
        or manifest.get("task_id") != task_id
        or manifest.get("base_commit") != base_commit
    ):
        raise ValueError("Image manifest does not match scanner, task or base commit")
    for key, pattern in (
        ("image_id", r"sha256:[0-9a-f]{64}"),
        ("rootfs_sha256", r"[0-9a-f]{64}"),
        ("agent_base_commit", r"[0-9a-f]{40}|[0-9a-f]{64}"),
    ):
        if not re.fullmatch(pattern, str(manifest.get(key, ""))):
            raise ValueError(f"Image manifest has invalid {key}")
    required = {"base-export", "fresh-git", "artifact-paths", "reference-fragments", "mtime-zero", "flat-rootfs"}
    if not required <= set(manifest.get("checks", [])):
        raise ValueError("Image manifest lacks required admission checks")
    return AdmittedImage(manifest["image_id"], manifest["agent_base_commit"], manifest["rootfs_sha256"])


# Executed before handing the sandbox to the agent. No DNS resolution or HTTP proxy environment is used
# for direct-IP tests. A DNS error response still proves packets escaped and must reject admission.
OFFLINE_NETWORK_PROBE = r"""
import ipaddress, os, socket, struct
if {name for _, name in socket.if_nameindex()} != {"lo"}:
    raise RuntimeError("offline sandbox has a non-loopback network interface")
resolvers = {"1.1.1.1", "8.8.8.8"}
with open("/etc/resolv.conf") as stream:
    for line in stream:
        words = line.split()
        if words and words[0] == "nameserver":
            resolvers.add(words[1])
query = struct.pack("!HHHHHH", 0xA123, 0x0100, 1, 0, 0, 0) + b"\x0eadmission-" + os.urandom(2).hex().encode() + b"\x07invalid\0\0\x01\0\x01"
for address in ["1.1.1.1", "2606:4700:4700::1111"]:
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        if sock.connect_ex((address, 443)) == 0:
            raise RuntimeError("direct-IP egress succeeded")
for address in sorted(resolvers):
    family = socket.AF_INET6 if ipaddress.ip_address(address).version == 6 else socket.AF_INET
    for kind in [socket.SOCK_DGRAM, socket.SOCK_STREAM]:
        with socket.socket(family, kind) as sock:
            sock.settimeout(0.5)
            try:
                if kind == socket.SOCK_STREAM:
                    sock.connect((address, 53))
                    raise RuntimeError("TCP DNS egress succeeded")
                sock.sendto(query, (address, 53))
                sock.recvfrom(4096)
                raise RuntimeError("UDP DNS egress succeeded")
            except OSError:
                pass
print("offline-network-admitted")
"""
