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

"""Multi-resources-server MCP seeding (design R3 / 3.3): one verifier seeds the
task and owns verify; tool providers are seeded tool_only and their tools are
merged, namespaced by server name so they can't clash."""

from __future__ import annotations

import pytest

from responses_api_agents.external_harness.app import (
    ExternalHarnessConfig,
    HarnessSettings,
    _seed_rollout,
    resolve_rollout_id,
)


def _cfg(**kw) -> ExternalHarnessConfig:
    return ExternalHarnessConfig(
        settings=HarnessSettings(harness="claude_code", sandbox_provider={"docker": {}}),
        model_server_url="http://m",
        **kw,
    )


def _mcp(server_name: str, token: str) -> dict:
    return {
        "server_name": server_name,
        "url_path": "/mcp",
        "transport": "http",
        "headers": {"X-NeMo-Gym-Session-Token": token},
    }


class _FakeResp:
    def __init__(self, data: dict):
        self._data = data

    def raise_for_status(self):
        pass

    async def json(self):
        return self._data


def _install_fake_request(monkeypatch, by_url: dict, calls: list):
    """Patch the module-level ``request`` (the shared aiohttp client) that
    _seed_rollout calls, returning a canned response per URL substring."""

    async def fake_request(method, url, json=None, **kwargs):
        calls.append((url, json))
        for key, data in by_url.items():
            if key in url:
                return _FakeResp(data)
        return _FakeResp({})

    monkeypatch.setattr("responses_api_agents.external_harness.app.request", fake_request)


@pytest.mark.asyncio
async def test_verifier_plus_tool_providers_merge_namespaced(monkeypatch):
    calls: list = []
    verifier_seed = {"sandbox_spec": {"files": {}}, "mcp": _mcp("blackbox_tool_toy", "tok-verify")}
    provider_seed = {"mcp": _mcp("blackbox_echo_tool", "tok-tool")}
    _install_fake_request(monkeypatch, {"verifier-host": verifier_seed, "provider-host": provider_seed}, calls)
    cfg = _cfg(verify_url="http://verifier-host:1", tool_provider_urls=["http://provider-host:2"])

    seed = await _seed_rollout(cfg, "ds.00000.r00.a0", {})

    # Both servers lent, keyed (namespaced) by server name.
    assert set(seed.mcp_servers) == {"blackbox_tool_toy", "blackbox_echo_tool"}
    assert seed.mcp_servers["blackbox_echo_tool"]["url"] == "http://provider-host:2/mcp"
    assert seed.mcp_servers["blackbox_tool_toy"]["headers"]["X-NeMo-Gym-Session-Token"] == "tok-verify"
    # The tool provider was seeded tool_only (no task); the verifier was not.
    provider_calls = [j for (u, j) in calls if "provider-host" in u]
    verifier_calls = [j for (u, j) in calls if "verifier-host" in u]
    assert provider_calls and provider_calls[0].get("tool_only") is True
    assert verifier_calls and "tool_only" not in verifier_calls[0]


@pytest.mark.asyncio
async def test_tool_provider_name_collision_raises(monkeypatch):
    calls: list = []
    # Provider reuses the verifier's server name -> must fail loudly.
    seeds = {"verifier-host": {"mcp": _mcp("same_name", "a")}, "provider-host": {"mcp": _mcp("same_name", "b")}}
    _install_fake_request(monkeypatch, seeds, calls)
    cfg = _cfg(verify_url="http://verifier-host:1", tool_provider_urls=["http://provider-host:2"])
    with pytest.raises(ValueError, match="collides"):
        await _seed_rollout(cfg, "ds.00000.r00.a0", {})


def test_resolve_rollout_id_prefers_reserved_and_is_unique():
    assert resolve_rollout_id({"ng_rollout_id": "ds.00001.r00.a0"}, "ds", 0, 0, 0) == "ds.00001.r00.a0"
    a = resolve_rollout_id({}, "ds", 0, 0, 0)
    b = resolve_rollout_id({}, "ds", 0, 0, 0)
    assert a != b  # unique per rollout (no collision)
    with pytest.raises(ValueError):
        resolve_rollout_id({"ng_rollout_id": "bad id!"}, "ds", 0, 0, 0)
