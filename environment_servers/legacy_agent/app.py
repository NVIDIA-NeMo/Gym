# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility environment server for agent servers that still own their episode through `/run`.

`single_agent_turn_legacy` translates a legacy row for a migrated pairing. This one translates nothing:
it relays `/run` to an agent that has not been migrated, so the agent's own contract still applies.
"""

import asyncio
import warnings
from collections.abc import Iterable
from typing import Any

from fastapi import Body, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ConfigDict, PositiveFloat, PositiveInt, model_validator
from typing_extensions import Self

from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import AgentServerRef, AggregateMetrics, AggregateMetricsRequest
from nemo_gym.server_runtime import (
    BaseEnvironmentServer,
    BaseEnvironmentServerConfig,
    BaseEpisodeRequest,
    BaseEpisodeResponse,
    CleanupContext,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


_UNUSED_LIMITS = frozenset(
    {
        "max_concurrent_episodes",
        "queue_timeout_seconds",
        "cleanup_timeout_seconds",
    }
)

# Headers that describe a connection or a body, not the payload. Each hop frames its own, and
# `cookie` is carried separately so aiohttp does not send it twice.
_SKIPPED_HEADERS = frozenset(
    {
        "connection",
        "content-encoding",
        "content-length",
        "cookie",
        "host",
        "keep-alive",
        "transfer-encoding",
        "upgrade",
    }
)


def _relayed_headers(headers: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(name, value) for name, value in headers if name.lower() not in _SKIPPED_HEADERS]


class LegacyAgentEnvironmentServerConfig(BaseEnvironmentServerConfig):
    """Bind one agent server."""

    model_config = ConfigDict(extra="forbid")

    agent_server: AgentServerRef

    # The episode deadline bounds the relayed `/run` call when set; unset, the agent's own limits are the only
    # ones. A relay enforces none of the other limits. They carry defaults only because the base declares them,
    # so configs need not repeat values that do nothing.
    max_concurrent_episodes: PositiveInt | None = None
    queue_timeout_seconds: PositiveFloat | None = None
    default_episode_timeout_seconds: PositiveFloat | None = None
    cleanup_timeout_seconds: PositiveFloat = 180

    @model_validator(mode="after")
    def warn_on_unused_limits(self) -> Self:
        set_limits = sorted(_UNUSED_LIMITS & self.model_fields_set)
        if set_limits:
            warnings.warn(
                f"{self.name} sets {', '.join(set_limits)}, which a relay does not enforce. "
                "This parameter has no effect.",
                stacklevel=2,
            )
        return self


class LegacyAgentEnvironmentServer(BaseEnvironmentServer):
    """Relay `/run` to one agent server without reading either side's contract."""

    config: LegacyAgentEnvironmentServerConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.post("/run")(self.run_legacy)
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        return app

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Forward to the agent server, which aggregates its own rollouts today."""
        response = await self.server_client.post(
            server_name=self.config.agent_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))

    async def run(self, request: BaseEpisodeRequest[Any], cleanup: CleanupContext) -> BaseEpisodeResponse[Any]:
        raise RuntimeError(
            "legacy_agent relays /run to its agent server and has no typed episode protocol. "
            "Reaching this means /run was bound to the base lifecycle instead of run_legacy."
        )

    async def run_legacy(self, request: Request) -> Response:
        """Relay one rollout-collection `/run` call to the agent server, and its answer back.

        This is an opaque relay that connects unmigrated agent servers to the
        new environment server infrastructure. An episode that outlives ``default_episode_timeout_seconds``
        ends as the same retryable failure ``single_agent_turn_legacy`` returns when its deadline fires.
        """
        data = await request.body()
        deadline = asyncio.timeout(self.config.default_episode_timeout_seconds)
        try:
            async with deadline:
                upstream = await self.server_client.post(
                    server_name=self.config.agent_server.name,
                    url_path="/run",
                    data=data,
                    headers=dict(_relayed_headers(request.headers.items())),
                    cookies=request.cookies,
                )
                body = await upstream.read()
        except TimeoutError:
            if not deadline.expired():
                raise
            return JSONResponse(
                {
                    "_ng_failure_class": "environment_server_failed",
                    "_ng_failure_terminal": False,
                    "_ng_failure_message": "Episode timed out",
                    "_ng_failure_stage": None,
                    "agent_ref": {"name": self.config.agent_server.name},
                }
            )
        response = Response(content=body, status_code=upstream.status)
        # raw_headers rather than `headers=`, so repeated fields such as Set-Cookie survive.
        response.raw_headers = [
            (name.lower().encode("latin-1"), value.encode("latin-1"))
            for name, value in _relayed_headers(upstream.headers.items())
        ]
        response.raw_headers.append((b"content-length", str(len(body)).encode("latin-1")))
        return response


if __name__ == "__main__":
    LegacyAgentEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = LegacyAgentEnvironmentServer.run_webserver()  # noqa: F401
