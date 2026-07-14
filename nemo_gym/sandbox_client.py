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

"""Client-side helpers for talking to a sandbox server.

This is the single bridge between the low-level ``nemo_gym.sandbox`` layer and
the server framework (``nemo_gym.server_utils``): it wires Gym's global aiohttp
client into the remote sandbox provider's transport seam. The sandbox layer
depends only on the transport protocol, and ``server_utils`` does not import the
sandbox layer, so the dependency direction stays one-way (server layer -> both).

Gym servers that operate a sandbox server (the external_harness agent, a
resources server that attaches to or spins up eval boxes) use these helpers
instead of constructing the remote provider directly.
"""

from __future__ import annotations

from typing import Any

from nemo_gym.sandbox import AsyncSandbox, SandboxRef
from nemo_gym.sandbox.providers.remote import RemoteSandboxProvider
from nemo_gym.server_utils import raise_for_status, request


class GymSandboxHttpTransport:
    """SandboxHttpTransport backed by Gym's global aiohttp client."""

    async def request(self, method: str, url: str, **kwargs: Any) -> Any:
        return await request(method, url, **kwargs)

    async def raise_for_status(self, response: Any) -> None:
        await raise_for_status(response)


_TRANSPORT = GymSandboxHttpTransport()


def gym_sandbox_transport() -> GymSandboxHttpTransport:
    """The shared Gym-backed sandbox HTTP transport."""
    return _TRANSPORT


def make_remote_provider(server_url: str, *, api_key: str | None = None) -> RemoteSandboxProvider:
    """Build a remote sandbox provider wired to Gym's HTTP client."""
    return RemoteSandboxProvider(server_url=server_url, transport=_TRANSPORT, api_key=api_key)


async def attach_sandbox(ref: SandboxRef, *, api_key: str | None = None) -> AsyncSandbox:
    """Attach to a server-owned sandbox by reference, wired to Gym's HTTP client."""
    return await AsyncSandbox.attach(ref, transport=_TRANSPORT, api_key=api_key)
