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
"""Backend-neutral contracts for the interactive-browser environment.

`BrowserBackend` is the whole surface the resources server depends on: how a
browser is obtained, driven and released is entirely the backend's business.
`BrowserSessionProvider` is the second, narrower seam — it exists only for
remote backends, which need someone to hand them a CDP endpoint and take it
back afterwards (see `remote_cdp.py`).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class Element:
    """One interactive element exposed to the policy, addressed by stable `id`."""

    id: int
    role: str  # "link" | "button" | "textbox" | ...
    name: str  # accessible name / visible text (truncated)


@dataclass
class Observation:
    """Compact, token-cheap view of the page handed to the policy each step."""

    url: str
    title: str
    elements: list[Element] = field(default_factory=list)
    # True when the backend stopped collecting at its element budget, so the
    # policy knows the element list is incomplete rather than exhaustive.
    truncated: bool = False

    def render(self, max_elements: int = 50) -> str:
        lines = [f"URL: {self.url}", f"TITLE: {self.title}", "ELEMENTS:"]
        for el in self.elements[:max_elements]:
            lines.append(f"  [{el.id}] {el.role}: {el.name}")
        if self.truncated or len(self.elements) > max_elements:
            lines.append(f"  ... (truncated at {max_elements} elements)")
        return "\n".join(lines)


class BrowserBackend(abc.ABC):
    """Per-episode browser contract. One instance == one isolated context/page.

    Implementations MUST be safe to instantiate many times concurrently (one per
    rollout) and MUST release all resources in `close()`.
    """

    @abc.abstractmethod
    async def open(self, initial_url: str) -> None: ...

    @abc.abstractmethod
    async def goto(self, url: str) -> None: ...

    @abc.abstractmethod
    async def click(self, element_id: int) -> None: ...

    @abc.abstractmethod
    async def type(self, element_id: int, text: str) -> None: ...

    @abc.abstractmethod
    async def observe(self, max_elements: int = 50) -> Observation:
        """Snapshot the page.  Implementations MUST stop collecting elements at
        `max_elements` (probing every interactive node costs a round-trip each)
        and set `Observation.truncated` when they do."""
        ...

    @abc.abstractmethod
    async def current_url(self) -> str: ...

    @abc.abstractmethod
    async def text(self) -> str:
        """Full visible page text (for dom_contains scoring)."""
        ...

    @abc.abstractmethod
    async def close(self) -> None: ...


@dataclass(frozen=True)
class BrowserSessionSpec:
    """What a remote backend tells a provider about the session it needs.

    Deliberately thin: the environment does not know what a provider charges
    for, so anything provider-shaped travels in `provider_options` (from config)
    and anything run-shaped travels in `metadata` (for provider-side tagging and
    observability, e.g. correlating a leaked cloud session back to a rollout).
    """

    metadata: dict[str, str] = field(default_factory=dict)
    provider_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class BrowserSessionHandle:
    """Provider-neutral handle to one live remote browser.

    `raw` is provider-owned opaque state (an SDK session object, a lease token,
    ...). The backend passes it back to the provider on release and never
    inspects it.
    """

    cdp_url: str
    session_id: Optional[str] = None
    provider_name: Optional[str] = None
    raw: Any = None


class BrowserSessionError(RuntimeError):
    """Raised when a provider cannot hand out a usable browser session."""


@runtime_checkable
class BrowserSessionProvider(Protocol):
    """Supplies remote browsers to `RemoteCDPBackend`, one per rollout.

    Implementations MUST return only once the session is reachable over CDP, and
    MUST make `release()` safe to call on a session that was never used or was
    already released — a rollout can die between the two calls.
    """

    name: str

    async def acquire(self, spec: BrowserSessionSpec) -> BrowserSessionHandle:
        """Return a handle whose `cdp_url` accepts a CDP connection now."""
        ...

    async def release(self, handle: BrowserSessionHandle) -> None:
        """Give the session back. Must not raise on an already-released handle."""
        ...
