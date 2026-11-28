# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool-only BrowseComp HTTP process. No seed, verify, model, judge or aggregation routes."""

import json
import os
from secrets import compare_digest

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from omegaconf import OmegaConf
from pydantic import JsonValue

from nemo_gym.client import ServerClient
from nemo_gym.config import BaseServerConfig
from resources_servers.browsecomp_advanced_harness.app import (
    TavilySearchMetrics,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
)


class BrowsecompToolsServer(TavilySearchResourcesServer):
    """Reuse search/page handling and pocketshell inside the tool container, without the resources protocol."""

    def setup_webserver(self) -> FastAPI:
        app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
        token = os.environ["BROWSECOMP_TOOL_TOKEN"]
        if not token:
            raise ValueError("BROWSECOMP_TOOL_TOKEN must be set")

        def authorize(authorization: str = Header(default="")) -> None:
            if not compare_digest(authorization.encode(), f"Bearer {token}".encode()):
                raise HTTPException(403, "Invalid tool grant")

        grant = [Depends(authorize)]
        self.setup_session_middleware(app)
        self.setup_liveness(app)
        app.post("/search", dependencies=grant)(self.search)
        app.post("/browse", dependencies=grant)(self.browse)
        if self.config.workspace == "per_session":
            app.post("/bash_command", dependencies=grant)(self.bash_command)
        app.get("/_metrics", dependencies=grant)(self.tool_metrics)
        return app

    async def tool_metrics(self) -> dict[str, JsonValue]:
        """One pod serves one agent; combine any session-local provider call records.

        ``exclude_domains_sha256`` lets the outside verifier refuse records searched under a different exclusion list.
        """
        metrics = TavilySearchMetrics(
            async_tavily_calls=[
                call for metrics in self._session_id_to_metrics.values() for call in metrics.async_tavily_calls
            ]
        )
        return {**metrics.model_dump(mode="json"), "exclude_domains_sha256": self._exclude_domains_sha256}


def main() -> None:
    """Serve only this rollout's tool routes on the sibling sandbox interface."""
    settings = json.loads(os.environ.pop("BROWSECOMP_TOOL_SETTINGS"))
    config = TavilySearchResourcesServerConfig(
        **settings,
        name="browsecomp_tools",
        host="0.0.0.0",
        entrypoint="tools_sidecar.py",
        use_judge=False,
        sandbox_provider=None,
        allow_host_execution=True,
    )
    client = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=1), global_config_dict=OmegaConf.create({})
    )
    tools = BrowsecompToolsServer(config=config, server_client=client)
    uvicorn.run(tools.setup_webserver(), host=config.host, port=config.port, proxy_headers=False, access_log=False)


if __name__ == "__main__":
    main()
