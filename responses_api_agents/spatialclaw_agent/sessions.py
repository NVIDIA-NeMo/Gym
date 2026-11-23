# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native harness state and a connection to the benchmark-owned kernel."""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from nemo_gym.agents import AgentSessionState
from nemo_gym.client import get_response_json, raise_for_status, request
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import DirectHTTPToolAccess
from nemo_gym.trajectory import AgentObservationBundle


@dataclass
class SpatialClawSessionState(AgentSessionState):
    access: DirectHTTPToolAccess
    task: asyncio.Task[NeMoGymResponse] | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    observations: AgentObservationBundle | None = None


async def resource_call(access: DirectHTTPToolAccess, path: str, payload: dict[str, Any]) -> Any:
    """Use the exact episode URL, headers, and cookie jar granted by Resources."""
    response = await request(
        method="POST",
        url=f"{str(access.base_url).rstrip('/')}/{path.lstrip('/')}",
        json=payload,
        cookies=access.cookies,
        headers=access.headers,
        _max_num_tries=1,
    )
    await raise_for_status(response)
    return await get_response_json(response)


class RemoteKernel:
    """Native manager protocol; the Resources Server owns the actual kernel."""

    def __init__(self, access: DirectHTTPToolAccess):
        self.access = access
        self.is_running = True
        self._init_code = ""
        self._injection_code = ""
        self._km = SimpleNamespace(kernel_id="resource-owned")
        self.failure: str | None = None

    async def call(self, operation: str, **kwargs: Any) -> Any:
        if self.failure is not None:
            raise RuntimeError(self.failure)
        try:
            reply = await resource_call(self.access, "/kernel", {"operation": operation, **kwargs})
            return reply["result"]
        except Exception as error:
            # Native nodes catch execution exceptions. Keep infrastructure failure
            # evidence outside the graph so it cannot become a scored fallback.
            self.failure = f"Resource kernel {operation} failed: {type(error).__name__}: {error}"
            raise

    def set_init_code(self, code: str) -> None:
        self._init_code = code

    def set_injection_code(self, code: str) -> None:
        self._injection_code = code

    async def start(self) -> str:
        result = await self.call("start")
        self.is_running = True
        return result

    async def shutdown(self) -> None:
        await self.call("shutdown")
        self.is_running = False

    async def reset_namespace(self) -> None:
        await self.call("reset_namespace")

    async def execute(self, code: str, timeout: float | None = None) -> SimpleNamespace:
        kwargs = {"code": code}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return SimpleNamespace(**await self.call("execute", **kwargs))

    async def get_variables(self) -> dict[str, Any]:
        return await self.call("get_variables")

    async def check_sentinel(self, sentinel_name: str = "_return_answer_result") -> dict[str, Any] | None:
        return await self.call("check_sentinel", name=sentinel_name)

    async def clear_sentinel(self, sentinel_name: str = "_return_answer_result") -> None:
        await self.call("clear_sentinel", name=sentinel_name)


class RemoteKernelPool:
    def __init__(self, access: DirectHTTPToolAccess):
        self.kernel = RemoteKernel(access)

    async def acquire(self) -> RemoteKernel:
        return self.kernel

    async def release(self, kernel: RemoteKernel) -> None:
        pass

    async def shutdown_all(self) -> None:
        # Disconnect only. Environment Server closes Resources after verification.
        pass
