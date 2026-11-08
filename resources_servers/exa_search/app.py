# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exa web search as a tool server.

An Environment Server lists this server in ``tool_servers`` and grants its ``exa_search`` tool to the session agent
over MCP, so any harness that takes MCP grants searches with Exa without its own Exa integration. The server keeps no
session state and grades nothing: the benchmark's own resources server verifies the episode.
"""

import asyncio
import json
import logging
from typing import Any

from aiohttp import ClientConnectionError, ClientPayloadError, ClientTimeout
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, SecretStr

from nemo_gym.client import RETRY_ERROR_CODES, request
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


LOG = logging.getLogger(__name__)

MAX_QUERY_CHARS = 1000
MAX_NUM_RESULTS = 10


class ExaSearchResourcesServerConfig(BaseResourcesServerConfig):
    exa_api_key: SecretStr
    exa_base_url: str = "https://api.exa.ai"
    # Results per search when the call names no number, and the page text kept per result.
    num_results: int = Field(default=5, ge=1, le=MAX_NUM_RESULTS)
    max_characters: int = Field(default=4000, ge=1)
    max_attempts: int = Field(default=3, ge=1)
    timeout_s: float = Field(default=60, gt=0)


class ExaSearchRequest(BaseModel):
    query: str | None = Field(default=None, description="Web search query.")
    num_results: int | None = Field(default=None, description=f"Results to return, 1 to {MAX_NUM_RESULTS}.")


class ExaSearchResponse(BaseModel):
    results_string: str


class ExaSearchResourcesServer(SimpleResourcesServer):
    config: ExaSearchResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/exa_search")(self.exa_search)
        return app

    def mcp_allowed_tools_for_session(self, seed_body: dict[str, Any]) -> list[str]:
        return ["exa_search"]

    async def exa_search(self, body: ExaSearchRequest) -> ExaSearchResponse:
        """Search the web with Exa and return each result's title, URL, date and page text."""
        query = (body.query or "").strip()
        if not query:
            return ExaSearchResponse(results_string="Exa search needs a non-empty query.")
        if len(query) > MAX_QUERY_CHARS:
            return ExaSearchResponse(results_string=f"Exa search queries are limited to {MAX_QUERY_CHARS} characters.")
        num_results = min(max(body.num_results or self.config.num_results, 1), MAX_NUM_RESULTS)
        try:
            payload = await self._search(query, num_results)
        except RuntimeError as error:
            # The agent sees why the search failed and can retry or answer without it.
            return ExaSearchResponse(results_string=str(error))
        results = [
            {
                "title": result.get("title"),
                "url": result.get("url"),
                "published_date": result.get("publishedDate"),
                "author": result.get("author"),
                "text": result.get("text"),
            }
            for result in payload.get("results") or []
            if isinstance(result, dict)
        ]
        if not results:
            return ExaSearchResponse(results_string="Exa search found no results.")
        return ExaSearchResponse(results_string=json.dumps(results, ensure_ascii=False))

    async def _search(self, query: str, num_results: int) -> dict[str, Any]:
        body = {
            "query": query,
            "numResults": num_results,
            "contents": {"text": {"maxCharacters": self.config.max_characters}},
        }
        headers = {"content-type": "application/json", "x-api-key": self.config.exa_api_key.get_secret_value()}
        for attempt in range(1, self.config.max_attempts + 1):
            response = None
            try:
                response = await request(
                    "POST",
                    url=f"{self.config.exa_base_url.rstrip('/')}/search",
                    headers=headers,
                    json=body,
                    timeout=ClientTimeout(total=self.config.timeout_s),
                    _max_connection_retries=0,
                )
                if response.status == 200:
                    payload = await response.json()
                    if not isinstance(payload, dict):
                        raise RuntimeError("Exa search returned an unexpected reply.")
                    return payload
                # The provider's error body may echo the key; report only the status.
                error = f"Exa search failed with HTTP {response.status}."
                if response.status not in RETRY_ERROR_CODES:
                    raise RuntimeError(error)
            except (ClientConnectionError, ClientPayloadError, TimeoutError) as transport_error:
                error = f"Exa search failed to reach Exa ({type(transport_error).__name__})."
            finally:
                if response is not None:
                    response.release()
            LOG.warning("%s Attempt %d of %d.", error, attempt, self.config.max_attempts)
            if attempt < self.config.max_attempts:
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(error)

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        raise HTTPException(
            status_code=400,
            detail="exa_search is a tool server and grades nothing; verify with the benchmark's resources server",
        )


if __name__ == "__main__":
    ExaSearchResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ExaSearchResourcesServer.run_webserver()  # noqa: F401
