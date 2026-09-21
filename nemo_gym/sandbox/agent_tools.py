# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Endpoint and authenticated MCP helpers for dedicated sandboxed agents."""

import ipaddress
import socket
from collections.abc import Mapping
from urllib.parse import urlsplit, urlunsplit

from nemo_gym.base_resources_server import NEMO_GYM_MCP_METADATA_KEY, BaseRunRequest
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.server_utils import ServerClient, get_response_json, get_server_url, raise_for_status


def sandbox_server_url(name: str) -> str:
    """Advertise the Gym host instead of the sandbox's loopback interface."""
    base = get_server_url(name).rstrip("/")
    parsed = urlsplit(base)
    if parsed.hostname in {"localhost", "127.0.0.1", "0.0.0.0", "::1", "::"}:
        host = socket.gethostbyname(socket.gethostname())
        host = f"[{host}]" if ":" in host else host
        netloc = f"{host}:{parsed.port}" if parsed.port else host
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
    return base


def restricted_network_policy(provider_name: str, urls: list[str]) -> dict[str, object]:
    """Fail closed when a provider or endpoint cannot support network isolation."""
    if provider_name != "opensandbox":
        raise ValueError("Restricted network access requires the OpenSandbox network-policy provider")
    targets = set()
    for url in urls:
        target = urlsplit(url).hostname
        if not target or target.lower() == "localhost":
            raise ValueError("Model/tool endpoint must advertise a sandbox-reachable host")
        try:
            address = ipaddress.ip_address(target)
        except ValueError:
            address = None
        if address and (address.is_loopback or address.is_unspecified):
            raise ValueError("Model/tool endpoint must advertise a sandbox-reachable host")
        targets.add(target)
    return {"defaultAction": "deny", "egress": [{"action": "allow", "target": t} for t in sorted(targets)]}


async def seed_mcp_servers(
    client: ServerClient,
    servers: list[ResourcesServerRef],
    body: BaseRunRequest,
    cookies: Mapping[str, str],
    *,
    timeout_s: float,
) -> dict[str, dict[str, object]]:
    """Seed host-side tools; pass only per-session MCP authentication into the sandbox."""
    entries = {}
    for server in servers:
        seeded = await client.post(
            server_name=server.name, url_path="/seed_session", json=body.model_dump(), cookies=cookies
        )
        await raise_for_status(seeded)
        metadata = (await get_response_json(seeded)).get(NEMO_GYM_MCP_METADATA_KEY)
        if not isinstance(metadata, dict) or not metadata.get("headers"):
            raise ValueError(f"Tool server {server.name} must expose authenticated MCP tools")
        entries[server.name] = {
            "url": sandbox_server_url(server.name) + "/" + metadata.get("url_path", "/mcp").lstrip("/"),
            "headers": metadata["headers"],
            "enabled": True,
            "timeout": int(timeout_s * 1000),
        }
    return entries
