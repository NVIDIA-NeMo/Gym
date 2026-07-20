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

This is the one bridge between the ``nemo_gym.sandbox`` library and the server
framework (``nemo_gym.server_utils``): it wires Gym's global aiohttp client into
the remote provider's transport. The sandbox library depends only on the
transport interface, and ``server_utils`` does not import the sandbox library,
so the dependency runs one way (server framework -> sandbox library).

A Gym server that operates a sandbox server (an agent that creates a box, a
resources server that reattaches to one or spins up its own) uses these helpers
rather than constructing the remote provider directly.
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


async def connect_sandbox(ref: SandboxRef, *, api_key: str | None = None) -> AsyncSandbox:
    """Reattach to a server-owned sandbox by reference, wired to Gym's HTTP client."""
    provider = make_remote_provider(ref.server_url, api_key=api_key)
    return await AsyncSandbox.connect(ref, provider=provider)
