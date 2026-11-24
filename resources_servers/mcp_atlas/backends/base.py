# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The contract between the MCP-Atlas resources server and the system that serves a task's tools.

A backend starts one tool environment per rollout, proxies the task's tool calls to it, classifies the
errors those calls return, and stops it. The resources server keeps everything that does not depend on where
the tools run: the session bookkeeping, the tool allowlist and call budget, the claim judge, the reward and
the masking policy. The backend's name is recorded in every prepared row, verify response and provenance
record, and a row prepared for one backend is refused by another.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Protocol


class BackendConfigurationError(ValueError):
    """The task cannot run with this backend as configured, for example missing credentials or a row prepared
    for another backend. A seed that raises it is rejected (HTTP 422) and never scored."""


class BackendStartError(RuntimeError):
    """The backend could not bring up the task's tools: an infrastructure failure. The rollout runs without
    tools and its verify response is masked as ``sandbox_start_failed``."""


class BackendTransportError(RuntimeError):
    """A tool call did not reach the backend, or its reply was unusable: an infrastructure failure, masked as
    ``sandbox_unreachable``. ``kind`` names the underlying error type for the model-visible message."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind


class EnvironmentDestroyedError(BackendTransportError):
    """The tools stopped answering while the backend confirms their environment is still allocated, after the
    model ran a program or command in it. The model sees the error and the rollout is scored, as upstream
    scores every tool failure; only a loss the provider itself reports stays an infrastructure failure."""


class ToolOutcome(StrEnum):
    """How the server treats a tool error the backend returned."""

    MODEL_RESULT = "model_result"
    """The model sees the error text and the rollout is scored."""
    SERVICE_ERROR = "service_error"
    """A service behind the tool failed (quota, TLS, server error); masked when ``mask_service_errors`` is set."""
    AUTHENTICATION_FAILURE = "authentication_failure"
    """The configured credentials were rejected; the rollout is refused without a reward."""


@dataclass(frozen=True)
class BackendTask:
    """What a backend needs to serve one task: its tools, the servers behind them and its own metadata."""

    task_id: str
    tools: tuple[str, ...]
    servers: tuple[str, ...]
    # ``verifier_metadata.backend_metadata``: opaque to the server, read only by the backend that wrote it.
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """The HTTP status and decoded body a backend returned for one tool call."""

    status: int
    body: Any


class BackendHandle(Protocol):
    """A started tool environment. ``missing_tools`` lists the task tools the backend could not serve."""

    missing_tools: list[str]


def backend_task(meta: Mapping[str, Any], *, backend: str) -> BackendTask:
    """Read a row's ``verifier_metadata`` into a task, refusing a row prepared for another backend.

    Raises BackendConfigurationError for a backend mismatch or malformed tool and server lists, so the seed is
    rejected before any provider allocation.
    """
    declared = meta.get("backend")
    if declared is not None and declared != backend:
        raise BackendConfigurationError(
            f"task {meta.get('task_id', '')!r} was prepared for backend {declared!r}; this server runs {backend!r}"
        )
    servers = meta.get("servers")
    if not isinstance(servers, list) or not servers or not all(isinstance(server, str) for server in servers):
        raise BackendConfigurationError("enabled servers must be a nonempty list of server names")
    tools = meta.get("enabled_tools")
    if not isinstance(tools, list) or not all(isinstance(tool, str) for tool in tools):
        raise BackendConfigurationError("enabled_tools must be a list of tool names")
    metadata = meta.get("backend_metadata") or {}
    if not isinstance(metadata, Mapping):
        raise BackendConfigurationError("backend_metadata must be an object")
    return BackendTask(
        task_id=str(meta.get("task_id", "")), tools=tuple(tools), servers=tuple(servers), metadata=dict(metadata)
    )


class MCPAtlasBackend(ABC):
    """Serve a task's MCP tools for one rollout at a time per session.

    A backend is built once from the resources server's config and validates its own settings then, so a
    misconfigured server fails at startup. Every method takes the ``session_id`` or the handle ``start``
    returned, never shared state, so rollouts stay independent.
    """

    name: ClassVar[str]

    @abstractmethod
    def __init__(self, *, config: Any, global_config_dict: Mapping[str, Any]) -> None: ...

    @abstractmethod
    async def start(self, session_id: str, task: BackendTask) -> BackendHandle:
        """Bring up the task's tools. Raises BackendConfigurationError (rejected) or BackendStartError (masked)."""

    async def reset(self, handle: BackendHandle, task: BackendTask) -> None:
        """Restore the task's initial state before the agent's first call. A no-op for live services."""

    @abstractmethod
    async def call_tool(
        self, handle: BackendHandle, task: BackendTask, tool_name: str, arguments: dict[str, Any], *, timeout_s: float
    ) -> ToolResult:
        """Call one task tool. Raises asyncio.TimeoutError past ``timeout_s`` and BackendTransportError when the
        call did not reach the tools; a tool's own error comes back as a result with a non-200 status."""

    @abstractmethod
    def classify_tool_error(
        self, handle: BackendHandle, task: BackendTask, tool_name: str, status: int, diagnostic: str
    ) -> ToolOutcome:
        """Decide whether a non-200 tool result is the model's to see, a service failure or a credential failure."""

    def provenance(self, handle: BackendHandle | None) -> dict[str, Any]:
        """What a report needs to interpret this rollout's rewards; always includes ``backend``."""
        return {"backend": self.name}

    @abstractmethod
    async def stop(self, session_id: str) -> bool:
        """Stop the session's tools; False when it has none. Raises when they did not stop, so the caller retries."""

    async def reap_expired(self) -> int:
        """Stop tool environments left unused past their time to live; returns how many stopped."""
        return 0

    @abstractmethod
    async def close(self) -> None:
        """Stop every tool environment at server shutdown. Raises naming the sessions whose tools did not stop."""
