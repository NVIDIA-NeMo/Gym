# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BIRD with Auto Ontology's catalog tools: ``bird_sql``'s grading, plus tools the agent calls in a catalog sandbox.

Auto Ontology (https://github.com/NVIDIA/auto-ontology) keeps a catalog of a database: its tables and columns, the
business terms and definitions compiled over them, and embeddings to search both. This server gives an agent three of
Auto Ontology's tools over that catalog, as ``direct_http`` routes:

- ``retrieve`` searches the catalog;
- ``run_sql`` checks a query against the catalog, then runs it read-only;
- ``submit`` checks and runs the final query. An accepted submission ends the episode's tools.

Each episode gets its own sandbox from a catalog image, which holds Auto Ontology, the BIRD databases and their
catalog (the image contract is in README.md). At the seed the server starts the sandbox, the catalog's Postgres and
the episode's tool process (``catalog_tool_server.py``). Each tool call runs in that process, which keeps one Auto
Ontology ``ToolContext`` for the episode. Verification stops the sandbox.

``verify`` grades the SQL of the last accepted ``submit`` call in the rollout's output, with ``bird_sql``'s rules.
It reads only the rollout, so a saved rollout can be graded again without the sandbox.
"""

import asyncio
import json
import shlex
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Optional

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxSpec,
    create_provider,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_runtime import (
    TOOL_ERROR_HEADER,
    BaseSeedSessionRequest,
    ResourcesSessionLostError,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.bird_sql.app import (
    BirdSqlResourcesServer,
    BirdSqlResourcesServerConfig,
    BirdSqlVerifyRequest,
    BirdSqlVerifyResponse,
)
from resources_servers.bird_sql.catalog_tools import ACCEPTED, RetrieveArguments, RunSqlArguments, SubmitArguments


TOOL_SERVER_SOURCE = Path(__file__).with_name("catalog_tool_server.py")
# Where the sandbox gets the tool process and the episode's settings, through SandboxSpec.files.
SANDBOX_DIR = "/tmp/nemo-gym-bird-catalog"
SANDBOX_TOOL_SERVER = f"{SANDBOX_DIR}/catalog_tool_server.py"
SANDBOX_EPISODE = f"{SANDBOX_DIR}/episode.json"
SANDBOX_SOCKET = f"{SANDBOX_DIR}/tools.sock"
SANDBOX_TOOL_LOG = f"{SANDBOX_DIR}/tools.log"


class BirdSqlCatalogResourcesServerConfig(BirdSqlResourcesServerConfig):
    """``bird_sql``'s settings, plus the catalog sandbox's. See README.md for the catalog image."""

    # A sandbox provider block of the global config, and the sandbox to start with it: at least the catalog image.
    sandbox_provider: str
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    # In the image: the Python that has Auto Ontology, the folder whose .env names the source databases, and the
    # command that starts the catalog's Postgres.
    catalog_python: str = "/opt/auto-ontology-eval/.venv/bin/python"
    catalog_workdir: str = "/opt/auto-ontology-eval"
    catalog_start_command: str = "/opt/auto-ontology-eval/start-catalog"
    # Environment variables for Auto Ontology in the sandbox, such as the key of its embedding endpoint
    # (DEFAULT_MODELS_API_KEY), which retrieval calls for each search.
    catalog_environment: dict[str, str] = Field(default_factory=dict)
    # Tool calls an episode may make, as Auto Ontology's ReAct loop allows. Past it, only submit is served.
    max_tool_calls: int = Field(default=24, gt=0)
    setup_timeout_s: float = Field(default=300, gt=0)
    tool_timeout_s: float = Field(default=120, gt=0)
    cleanup_timeout_s: float = Field(default=60, gt=0)
    sandbox_ttl_s: float = Field(default=7200, gt=0)


class BirdSqlCatalogSeedRequest(BaseSeedSessionRequest):
    """The task fields the episode's tools need."""

    model_config = ConfigDict(extra="allow")

    question: str
    db_id: str


@dataclass
class _Episode:
    sandbox: Optional[AsyncSandbox] = None


class BirdSqlCatalogResourcesServer(BirdSqlResourcesServer):
    config: BirdSqlCatalogResourcesServerConfig
    seed_request_model: ClassVar[type[BaseSeedSessionRequest]] = BirdSqlCatalogSeedRequest
    _episodes: dict[str, _Episode] = PrivateAttr(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/retrieve")(self.retrieve)
        app.post("/run_sql")(self.run_sql)
        app.post("/submit")(self.submit)
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(inner_app: FastAPI):
            try:
                async with parent_lifespan(inner_app) as value:
                    yield value
            finally:
                # Stop every sandbox a session still holds; the provider's TTL covers a stop that fails here.
                await asyncio.gather(
                    *(self.release_session_state(key) for key in list(self._episodes)), return_exceptions=True
                )

        app.router.lifespan_context = lifespan
        return app

    # ----- session state ----- #

    def has_session_state(self, session_id: str) -> bool:
        return session_id in self._episodes

    async def initialize_session_state(self, session_id: str, body: BirdSqlCatalogSeedRequest) -> None:
        episode = _Episode()
        self._episodes[session_id] = episode  # published before the first await, so release can stop the sandbox
        config = self.config
        global_config = self.server_client.global_config_dict
        provider_config = resolve_provider_config(config.sandbox_provider, global_config)
        sandbox_config = dict(config.sandbox_config)
        metadata = (
            resolve_provider_metadata(config.sandbox_provider, global_config)
            | dict(sandbox_config.pop("metadata", {}))
            | {"bird-db-id": body.db_id[:63]}
        )
        sandbox_config.setdefault("ttl_s", config.sandbox_ttl_s)
        files = {
            SANDBOX_TOOL_SERVER: TOOL_SERVER_SOURCE.read_text(),
            SANDBOX_EPISODE: json.dumps(
                {"database": body.db_id, "question": body.question, "max_tool_calls": config.max_tool_calls}
            ),
        }
        spec = SandboxSpec(**sandbox_config, metadata=metadata, files=files)
        episode.sandbox = AsyncSandbox(create_provider(provider_config))
        await episode.sandbox.start(spec)
        await self._start_catalog(episode.sandbox)
        await self._start_tools(episode.sandbox)

    async def release_session_state(self, session_id: str) -> None:
        episode = self._episodes.get(session_id)
        if episode is None:
            return
        if episode.sandbox is not None:
            async with asyncio.timeout(self.config.cleanup_timeout_s):
                await episode.sandbox.stop()
            episode.sandbox = None
        self._episodes.pop(session_id, None)

    async def _start_catalog(self, sandbox: AsyncSandbox) -> None:
        """Start the catalog's Postgres."""
        result = await sandbox.exec(self.config.catalog_start_command, timeout_s=self.config.setup_timeout_s)
        if result.error_type is not None or result.return_code != 0:
            raise RuntimeError(
                f"The catalog did not start ({result.error_type or f'exit {result.return_code}'}): "
                f"{(result.stderr or result.stdout or '')[-2000:]}"
            )

    async def _start_tools(self, sandbox: AsyncSandbox) -> None:
        """Start the episode's tool process in the background, and wait until it answers."""
        config = self.config
        python, server = shlex.quote(config.catalog_python), shlex.quote(SANDBOX_TOOL_SERVER)
        env_file = shlex.quote(f"{config.catalog_workdir}/.env")
        serve = (
            f"setsid nohup {python} {server} serve {shlex.quote(SANDBOX_SOCKET)} {shlex.quote(SANDBOX_EPISODE)} "
            f"{env_file} > {shlex.quote(SANDBOX_TOOL_LOG)} 2>&1 < /dev/null &"
        )
        started = await sandbox.exec(serve, cwd=config.catalog_workdir, env=config.catalog_environment, timeout_s=30)
        if started.error_type is not None or started.return_code != 0:
            raise RuntimeError(f"The catalog tools did not start: {started.error_type or started.return_code}")
        wait = f"{python} {server} wait {shlex.quote(SANDBOX_SOCKET)} {config.setup_timeout_s:g}"
        ready = await sandbox.exec(wait, timeout_s=config.setup_timeout_s + 30)
        if ready.error_type is not None or ready.return_code != 0:
            log = await sandbox.exec(f"tail -c 4000 {shlex.quote(SANDBOX_TOOL_LOG)}", timeout_s=30)
            raise RuntimeError(f"The catalog tools did not become ready: {(log.stdout or '').strip()}")

    # ----- tools ----- #

    async def retrieve(self, request: Request, body: RetrieveArguments) -> PlainTextResponse:
        """Search the database's catalog."""
        return await self._call_tool(request, "retrieve", body.model_dump(exclude_none=True))

    async def run_sql(self, request: Request, body: RunSqlArguments) -> PlainTextResponse:
        """Check a query against the catalog, then run it read-only."""
        return await self._call_tool(request, "run_sql", body.model_dump())

    async def submit(self, request: Request, body: SubmitArguments) -> PlainTextResponse:
        """Check and run the final query; an accepted one ends the episode's tools."""
        return await self._call_tool(request, "submit", body.model_dump())

    async def _call_tool(self, request: Request, tool: str, arguments: dict[str, Any]) -> PlainTextResponse:
        session_id = request.session[SESSION_ID_KEY]
        async with self.session_lock(session_id):
            episode = self._episodes.get(session_id)
            if episode is None or episode.sandbox is None:
                raise ResourcesSessionLostError("No seeded BIRD catalog session for this tool call")
            payload = json.dumps({"tool": tool, "args": arguments})
            command = shlex.join([self.config.catalog_python, SANDBOX_TOOL_SERVER, "call", SANDBOX_SOCKET, payload])
            result = await episode.sandbox.exec(command, timeout_s=self.config.tool_timeout_s)
        if result.error_type is not None or result.return_code != 0:
            detail = result.error_type or f"exit {result.return_code}: {(result.stderr or '')[-500:]}"
            return PlainTextResponse(f"{tool} failed: {detail}", headers={TOOL_ERROR_HEADER: "true"})
        try:
            reply = json.loads(result.stdout or "")
        except json.JSONDecodeError:
            return PlainTextResponse(f"{tool} failed: unreadable reply", headers={TOOL_ERROR_HEADER: "true"})
        # An accepted submission says exactly ACCEPTED, which verify looks for.
        return PlainTextResponse(ACCEPTED if reply.get("accepted") else str(reply.get("output", "")))

    # ----- grading ----- #

    def answer_sql(self, body: BirdSqlVerifyRequest) -> Optional[str]:
        """The SQL of the last ``submit`` call whose reply was ``ACCEPTED``, or None without one."""
        outputs = {
            item.call_id: item.output
            for item in body.response.output
            if getattr(item, "type", None) == "function_call_output"
        }
        answer = None
        for item in body.response.output:
            if getattr(item, "type", None) != "function_call" or item.name != "submit":
                continue
            output = outputs.get(item.call_id)
            if not isinstance(output, str) or output.strip() != ACCEPTED:
                continue
            try:
                answer = SubmitArguments.model_validate_json(item.arguments).sql.strip() or None
            except ValueError:
                continue
        return answer

    async def verify(self, request: Request, body: BirdSqlVerifyRequest) -> BirdSqlVerifyResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        if not session_id:
            return await super().verify(body)
        # Grading reads only the rollout; leaving verifying() stops the sandbox.
        async with self.verifying(session_id, allow_missing=True):
            return await super().verify(body)


if __name__ == "__main__":
    BirdSqlCatalogResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = BirdSqlCatalogResourcesServer.run_webserver()  # noqa: F401
