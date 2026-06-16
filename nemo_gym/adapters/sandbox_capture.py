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
"""Sandbox-bound capture proxy.

``start_adapter_proxy`` is model-bound: one proxy fronting an upstream model URL,
started once by the agent server. For a blackbox agent running *inside* a sandbox
we need a proxy bound to one rollout's ``session_id``, started per run, with the
``capture`` interceptor wired to a durable per-session store. The in-box agent's
``*_BASE_URL`` is then pointed at this proxy so every model call is recorded
under that session.

This is a thin wrapper over ``start_adapter_proxy`` that installs the capture
interceptor and returns the local proxy handle. The in-box agent reaches it via
the provider's reverse tunnel (e.g. ECS Fargate) or an advertised routable URL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nemo_gym.adapters.proxy import ProxyHandle, start_adapter_proxy


@dataclass
class SandboxCaptureProxy:
    """A per-rollout, session-keyed capture proxy handle."""

    handle: ProxyHandle
    session_id: str
    store_dir: str

    def stop(self, timeout: float = 5.0) -> None:
        self.handle.stop(timeout=timeout)


def start_capture_proxy(
    *,
    model_base_url: str,
    session_id: str,
    store_dir: str,
    host: str = "127.0.0.1",
    port: int = 0,
    inject_extra_body: dict[str, Any] | None = None,
    request_timeout: float = 600.0,
    translate_anthropic: bool = False,
    translate_model_override: str | None = None,
    upstream_api_key: str | None = None,
) -> SandboxCaptureProxy:
    """Start a proxy bound to ``session_id`` that captures model traffic.

    ``host`` defaults to localhost; pass a routable bind address (with the
    underlying ``unsafe_allow_remote``) when the sandbox must reach the proxy over
    the network. The in-box URL itself is resolved by the agent.

    Set ``translate_anthropic`` for agents that speak the Anthropic Messages API
    (Claude Code): the translation runs *before* capture, so the recorded
    exchange — and therefore the token-ids and trajectory — stays OpenAI-shaped,
    while the agent still receives an Anthropic response.
    """
    adapters: list[dict[str, Any]] = []
    if translate_anthropic:
        adapters.append(
            {"name": "translate_anthropic", "config": {"model_override": translate_model_override}}
        )
    adapters.append(
        {
            "name": "capture",
            "config": {
                "store_dir": store_dir,
                "session_id": session_id,
                "inject_extra_body": inject_extra_body or {},
                "upstream_api_key": upstream_api_key,
            },
        }
    )
    handle = start_adapter_proxy(
        upstream_url=model_base_url,
        adapters=adapters,
        host=host,
        port=port,
        request_timeout=request_timeout,
        unsafe_allow_remote=host not in ("127.0.0.1", "localhost"),
    )

    return SandboxCaptureProxy(handle=handle, session_id=session_id, store_dir=store_dir)
