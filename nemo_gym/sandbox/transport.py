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

"""HTTP transport interface for the remote sandbox provider.

The remote provider needs to make async HTTP calls, but ``nemo_gym.sandbox`` is
a library that must not depend on the server framework (``nemo_gym.server_utils``).
So the provider depends only on this small transport interface, and the concrete
aiohttp-backed transport is injected from the server layer (see
``nemo_gym.sandbox_client``).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SandboxHttpTransport(Protocol):
    """Minimal async HTTP surface the remote sandbox provider needs.

    Implemented in the server layer over Gym's global aiohttp client, so the
    sandbox library never imports it directly.
    """

    async def request(self, method: str, url: str, **kwargs: Any) -> Any:
        """Perform an HTTP request and return a response exposing ``status`` and
        an awaitable ``json()`` (aiohttp ``ClientResponse`` shape)."""
        ...

    async def raise_for_status(self, response: Any) -> None:
        """Raise if the response carries an error status."""
        ...
