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

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from nemo_gym.client.http_client import get_response_json, raise_for_status
from nemo_gym.client.server_client import ServerClient, get_server_url
from nemo_gym.config.config_types import ResourcesServerRef
from nemo_gym.core.failure_kinds import AGENT_RUN_ERROR
from nemo_gym.core.responses_api_types import NeMoGymResponse
from nemo_gym.server_runtime.base_resources_server import NEMO_GYM_MCP_METADATA_KEY, BaseRunRequest

# Defined beside the other import-free helpers, so a resources server's sandbox lending can use them too.
from nemo_gym.utils.network_policy import reachable_host as _reachable_host
from nemo_gym.utils.network_policy import restricted_network_policy as restricted_network_policy


def sandbox_server_url(name: str, *, require_reachable: bool = False) -> str:
    """Preserve inherited networking; reject loopback binds for remote sandboxes."""
    base = get_server_url(name).rstrip("/")
    parsed = urlsplit(base)
    if require_reachable:
        _reachable_host(base)
    elif parsed.hostname in {"0.0.0.0", "::"}:
        # Wildcards are bind addresses, not destinations. Host-network sandboxes
        # can use loopback; remote deployments must advertise a concrete host.
        host = "[::1]" if parsed.hostname == "::" else "127.0.0.1"
        return urlunsplit((parsed.scheme, f"{host}:{parsed.port}" if parsed.port else host, parsed.path, "", ""))
    return base


async def seed_mcp_servers(
    client: ServerClient,
    servers: list[ResourcesServerRef],
    body: BaseRunRequest,
    cookies: Mapping[str, str],
    *,
    timeout_s: float,
    require_reachable: bool = False,
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
            "url": sandbox_server_url(server.name, require_reachable=require_reachable)
            + "/"
            + metadata.get("url_path", "/mcp").lstrip("/"),
            "headers": metadata["headers"],
            "enabled": True,
            "timeout": int(timeout_s * 1000),
        }
    return entries


def verification_failure_fields(response: NeMoGymResponse | None = None) -> dict[str, Any]:
    """Forward bounded diagnostic events, including a restored response's report, without selecting a failure."""
    from nemo_gym.core.failure_transport import RequestFailureLog

    log = RequestFailureLog.current() or RequestFailureLog()
    saved = getattr(response, "failure_report", None) if response is not None else None
    if saved is not None:
        log.add_saved(saved)
    report = log.report
    # Explicit None clears any unrelated report copied from the run-request body.
    return {"failure_report": report.model_dump(mode="json") if report.events or report.omitted else None}


async def verify_agent_response(
    client: ServerClient,
    server: ResourcesServerRef,
    body: BaseRunRequest,
    response: NeMoGymResponse,
    cookies: Mapping[str, str],
    *,
    force_zero_reward: bool,
) -> dict[str, Any]:
    """Let the verifier own score fields, retaining failed generations for inspection."""
    grading_response = response.model_copy(update={"output": []}) if force_zero_reward else response
    verified = await client.post(
        server_name=server.name,
        url_path="/verify",
        cookies=cookies,
        json=body.model_dump(mode="json")
        | {"response": grading_response.model_dump(mode="json")}
        | verification_failure_fields(grading_response),
    )
    await raise_for_status(verified)
    result = await get_response_json(verified)
    if force_zero_reward:
        if result.get("reward") != 0 or result.get("mask_sample", False):
            raise ValueError(
                "execution_failure_reward_zero requires a verifier that scores an empty response as unmasked zero"
            )
        result.update(
            response=response.model_dump(mode="json"),
            failure_kind=AGENT_RUN_ERROR,
            failure_reason="Agent execution failed; execution_failure_reward_zero applied via empty-response verification.",
        )
    return result
