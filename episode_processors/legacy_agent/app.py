# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility processor for agent servers that still own their episode through `/run`."""

from collections.abc import Iterable
from typing import Any

from fastapi import FastAPI, Request, Response
from pydantic import ConfigDict

from nemo_gym.base_episode_processor import BaseEpisodeProcessor, BaseEpisodeProcessorConfig, EpisodeContext
from nemo_gym.config_types import AgentServerRef
from nemo_gym.episode import BaseEpisodeRequest, BaseEpisodeResponse


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


def _relayed(headers: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(name, value) for name, value in headers if name.lower() not in _SKIPPED_HEADERS]


class LegacyAgentEpisodeProcessorConfig(BaseEpisodeProcessorConfig):
    """Bind one agent server."""

    model_config = ConfigDict(extra="forbid")

    agent_server: AgentServerRef


class LegacyAgentEpisodeProcessor(BaseEpisodeProcessor):
    """Relay `/run` to one agent server without reading either side's contract."""

    config: LegacyAgentEpisodeProcessorConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.post("/run")(self.run_legacy)
        return app

    async def process(self, request: BaseEpisodeRequest[Any], context: EpisodeContext) -> BaseEpisodeResponse[Any]:
        raise RuntimeError(
            "legacy_agent episode processor relays /run to its agent server and has no typed episode protocol. "
            "Reaching this means /run was bound to the base lifecycle instead of run_legacy."
        )

    async def run_legacy(self, request: Request) -> Response:
        """Relay one rollout-collection `/run` call to the agent server, and its answer back.

        Both bodies stay opaque. The row shape belongs to the collector, the result shape to
        the agent and the resources server behind it, and neither is this server's to validate.

        `/run` is not idempotent, so this must reach the agent at most once.
        TODO(#3462): `ServerClient` retries connection failures without bound.
        """
        upstream = await self.server_client.post(
            server_name=self.config.agent_server.name,
            url_path="/run",
            data=await request.body(),
            headers=dict(_relayed(request.headers.items())),
            cookies=request.cookies,
        )
        body = await upstream.read()
        response = Response(content=body, status_code=upstream.status)
        # raw_headers rather than `headers=`, so repeated fields such as Set-Cookie survive.
        response.raw_headers = [
            (name.lower().encode("latin-1"), value.encode("latin-1"))
            for name, value in _relayed(upstream.headers.items())
        ]
        response.raw_headers.append((b"content-length", str(len(body)).encode("latin-1")))
        return response


if __name__ == "__main__":
    LegacyAgentEpisodeProcessor.run_webserver()
