# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OfficeEnv resources server: connected SaaS apps simulated per rollout.

The environment is the ``officeenv`` package at the release requirements.txt pins. This server
routes Gym's endpoints to OfficeEnv's framework-neutral episode service
(``officeenv.service.EpisodeService``) and adds only Gym's own response fields.

A row (one JSONL line, written by ``officeenv emit`` or ``officeenv.emit.gym_row``):

    {"responses_create_params": {"input": [...], "tools": [...]},
     "officeenv": {<TaskSpec>}, "officeenv_version": "1.9.0", "task_id": "..."}

An Agent's ``/run`` (the simple agent) POSTs the row to ``/seed_session``, each function call to
``POST /<tool_name>`` with the call's arguments as the body, and the row plus the model's response
to ``/verify``. An Environment Server seeds with a ``ResourcesSeedSessionRequest`` whose
``task_data`` holds the row's other fields, so the session cookie names its
``resources_session_id``; its agent reaches the same tool routes (or the MCP endpoint below) with
that cookie, ``/verify`` grades the episode as for an Agent, and ``/close_session`` ends it.

With ``expose_tools_over_mcp: true``, Gym's MCP endpoint serves every tool of OfficeEnv's
``tools`` exposure under its tool name, and each episode's token admits only that episode's tools.
MCP clients read the tool definitions from that endpoint rather than from the row, so a seed whose
tools differ from those definitions (a row in the ``api_search``, ``cli`` or ``mcp`` exposure, or
with another tool-name separator) is refused.

A row may name its TaskSpec's large strings by SHA-256 instead of carrying them (OfficeEnv's
artifact references, ``officeenv.artifacts``). The stores in ``artifacts_dirs`` (such as a
corpus's ``artifacts/`` folder beside its rows file), searched in order, go to the episode
service, which reads the references back at seed, each checked against its hash, and refuses a
row whose artifacts cannot be read (400).

With ``sandbox_provider`` set, each Environment Server episode also gets a sandbox from Gym's
sandbox providers, lent to the harness (``SandboxAccess``) and stopped before the episode is graded.
The sandbox gets the episode's file API in ``OFFICEENV_API_URL`` and ``OFFICEENV_API_TOKEN``
(``OFFICEENV.md`` in the workspace documents it), so file content moves between the workspace and
the simulated apps over HTTP instead of through the model's context. A sandbox can reach this
server, so sandbox mode guards it: seeding, closing, verification, the app catalog and the tool
routes refuse callers other than this node and ``trusted_networks``, which leaves sandboxes the MCP
endpoint and the file API; and session cookies and MCP tokens are signed with a per-process secret
(``get_session_secret``) instead of Gym's key, which anyone can derive from the server's name.

For training, ``length_penalty`` (off by default) deducts AutomationBench's training length
penalty from the reward and reports the two terms as ``reward_components``, so a trainer can
schedule the penalty on its own (NeMo RL ``grpo.phasic_reward_components``).
"""

import asyncio
import base64
import binascii
import contextvars
import ipaddress
import json
import logging
import os
import shlex
import socket
import weakref
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from secrets import token_urlsafe
from typing import TYPE_CHECKING, Any, ClassVar

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from officeenv.apps import available, get_app_class
from officeenv.artifacts import ArtifactStore
from officeenv.exposure import Catalog, ToolsExposure
from officeenv.service import EpisodeService, Reply
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import get_global_config_dict
from nemo_gym.core import EpisodeId, NeMoGymResponse, TaskId
from nemo_gym.sandbox import AsyncSandbox, DirectSandboxConnection, SandboxAccess, SandboxConfig
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.server_runtime import (
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ClosedResourcesSessionRecords,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionLostError,
    ReverifyMode,
    SimpleResourcesServer,
    lost_session_verification,
)


if TYPE_CHECKING:
    from nemo_gym.server_runtime import MCPTool


LOGGER = logging.getLogger(__name__)


class LengthPenaltyConfig(BaseModel):
    """Optional training reward shaping, AutomationBench's (benchmarks/automationbench): a
    rollout whose task reward is above ``reward_threshold`` loses
    ``min(max_penalty, turns / turn_scale + tokens / token_scale)``, where turns are the model's
    responses and tokens the completion tokens it generated over all of them (reasoning
    included). Off by default; turn it on for training only, so evaluation rewards stay
    OfficeEnv's."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    enabled: bool = False
    reward_threshold: float = Field(default=0.5, ge=0)
    max_penalty: float = Field(default=0.5, ge=0)
    turn_scale: float = Field(default=200.0, gt=0)
    token_scale: float = Field(default=200000.0, gt=0)


class OfficeEnvSandboxConfig(SandboxConfig):
    """The sandbox lent to each Environment Server episode when ``sandbox_provider`` is set."""

    image: str | None = Field(default=None, description="OCI image, e.g. one built from sandbox/Dockerfile")
    workdir: str = "/workspace"
    host_relay: str | None = Field(
        default=None,
        description="Host alias the sandbox forwards this server's and the model servers' loopback ports to "
        "(colima: host.lima.internal), for servers that bind loopback on a workstation; off by default",
    )


class OfficeEnvResourcesServerConfig(BaseResourcesServerConfig):
    # the reward depends on the session's state, which verify consumes
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED

    length_penalty: LengthPenaltyConfig = Field(default_factory=LengthPenaltyConfig)
    artifacts_dirs: list[str] = Field(
        default_factory=list,
        description=(
            "Stores of the artifacts a row's TaskSpec names by SHA-256 instead of carrying them "
            "(officeenv.artifacts), searched in order: absolute paths, such as a corpus's artifacts/ folder "
            "beside its rows file. A row that names artifacts no store holds is refused (400)."
        ),
    )

    @field_validator("artifacts_dirs")
    @classmethod
    def _stores_are_absolute(cls, stores: list[str]) -> list[str]:
        # the server runs in its own directory, not where its config was written
        relative = [store for store in stores if not os.path.isabs(store)]
        if relative:
            raise ValueError(f"artifact stores must be absolute paths: {relative}")
        return stores

    sandbox_provider: str | None = Field(
        default=None,
        description="Top-level sandbox config block (e.g. `sandbox`); set it to lend each episode a sandbox",
    )
    sandbox_config: OfficeEnvSandboxConfig = Field(default_factory=OfficeEnvSandboxConfig)
    files_api_url: str | None = Field(
        default=None,
        description="Base URL a sandbox reaches this server at (default: this server's configured host and port)",
    )
    trusted_networks: list[str] = Field(
        default_factory=list,
        description="In sandbox mode, networks (CIDR) besides this node whose callers reach every route",
    )


class OfficeEnvSeedSessionRequest(BaseSeedSessionRequest):
    """An Agent's seed: the row."""

    model_config = ConfigDict(extra="allow")

    officeenv: dict[str, Any] = Field(description="The TaskSpec for this episode")


class OfficeEnvSeedSessionResponse(BaseSeedSessionResponse):
    model_config = ConfigDict(extra="allow")

    task_id: str
    tools: list[str]
    apps: list[str]
    officeenv_server_version: str
    tool_drift: list[str] = Field(
        default_factory=list,
        description="Tools the row shows the model differently from what this version serves",
    )


class OfficeEnvVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class OfficeEnvVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class OfficeEnvLengthPenaltyVerifyResponse(BaseMultiRewardVerifyResponse):
    """The verify response with the length penalty on: ``reward_components`` holds the reward's
    terms, which sum to ``reward``: ``task_reward`` (OfficeEnv's) and ``length_penalty`` (minus the
    deduction)."""

    model_config = ConfigDict(extra="allow")


_MODEL_OUTPUT_TYPES = frozenset({"reasoning", "message", "function_call"})


def model_turns(response: NeMoGymResponse) -> int:
    """The model's responses in an episode: the runs of model output items in its output,
    which the outputs of the tools each response called separate."""
    turns, in_turn = 0, False
    for item in response.output:
        own = item.type in _MODEL_OUTPUT_TYPES and getattr(item, "role", "assistant") == "assistant"
        turns += own and not in_turn
        in_turn = own
    return turns


def length_penalty_fields(reward: float, response: NeMoGymResponse, config: LengthPenaltyConfig) -> dict[str, Any]:
    """The verify fields the length penalty sets: the penalized ``reward``, its terms
    (``reward_components``), and as metrics the raw ``task_reward``, the ``length_penalty``
    deducted, ``model_turns`` and ``generated_tokens`` (the response's accumulated usage, which
    the penalty needs whenever the task reward is above the threshold)."""
    turns = model_turns(response)
    tokens = None if response.usage is None else response.usage.output_tokens
    penalty = 0.0
    if reward > config.reward_threshold:
        if tokens is None:
            raise ValueError("The length penalty needs the response's usage (its generated tokens)")
        penalty = min(config.max_penalty, turns / config.turn_scale + tokens / config.token_scale)
    fields: dict[str, Any] = {
        "reward": reward - penalty,
        "reward_components": {"task_reward": reward, "length_penalty": -penalty},
        "task_reward": reward,
        "length_penalty": penalty,
        "model_turns": float(turns),
    }
    if tokens is not None:
        fields["generated_tokens"] = float(tokens)
    return fields


@cache
def tools_exposure_catalog() -> dict[str, dict[str, Any]]:
    """Every tool of OfficeEnv's ``tools`` exposure with all of its apps connected, by name: the
    tool definitions the MCP endpoint serves."""
    apps = {name: get_app_class(name)() for name in available()}
    return {tool["name"]: tool for tool in ToolsExposure(Catalog(apps), {}).tools()}


# The tools of the session the current /seed_session request seeded. Gym mints the episode's MCP
# token after the seed handler returns, and reads them then (mcp_allowed_tools_for_session).
_SEEDED_TOOLS: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "officeenv_seeded_tools", default=None
)

# File transfer between an episode's sandbox and an app's files: the ops each verb maps to.
FILE_OPS: dict[str, dict[str, str]] = {
    "gdrive": {"download": "download_file", "create": "create_file", "update": "update_file"},
}
MAX_UPLOAD_BYTES = 20_000_000

WORKSPACE_GUIDE = f"""# OfficeEnv workspace

Business apps are tools on your MCP server. File content moves between this workspace and
Google Drive over HTTP, without going through the conversation, byte for byte: text files
(CSV, TSV, JSON, Markdown) and binary ones (XLSX workbooks, PDFs, images) alike.

    # download a Drive file's content
    curl -sf -H "Authorization: Bearer $OFFICEENV_API_TOKEN" \\
      "$OFFICEENV_API_URL/officeenv/files/gdrive/<file_id>" -o <local path>

    # upload a local file as a new Drive file (parent folder id optional)
    curl -sf -X POST -H "Authorization: Bearer $OFFICEENV_API_TOKEN" --data-binary @<local path> \\
      "$OFFICEENV_API_URL/officeenv/files?app=gdrive&name=<file name>&parent=<folder id>"

    # replace an existing Drive file's content
    curl -sf -X PUT -H "Authorization: Bearer $OFFICEENV_API_TOKEN" --data-binary @<local path> \\
      "$OFFICEENV_API_URL/officeenv/files/gdrive/<file_id>"

The file name's extension sets the file type (.csv, .xlsx, .pdf, ...); add &mime_type=<type> to
the upload URL for another. Uploads are limited to {MAX_UPLOAD_BYTES // 1_000_000} MB. Find file and
folder ids with the Drive tools (list or search files). A request that fails returns the app's
error as JSON.
"""

RELAY_SOURCE = (Path(__file__).parent / "sandbox" / "relay.py").read_text(encoding="utf-8")
RELAY_PATH = "/opt/officeenv/relay.py"
RELAY_LOG = "/tmp/officeenv-relay.log"


def _error_status(body: Any) -> int | None:
    """The HTTP status an app's error output stands for, or None when the output is no error."""
    error = body.get("error") if isinstance(body, dict) else None
    if error is None:
        return None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, int) and 400 <= code < 600 else 400


def relay_ports(named_configs: Mapping[str, Any], *, own_port: int) -> list[int]:
    """This server's port and every model server's: what a sandboxed harness connects to."""
    ports = {own_port}
    for block in named_configs.values():
        models = block.get("responses_api_models") if isinstance(block, Mapping) else None
        for server in models.values() if isinstance(models, Mapping) else ():
            port = server.get("port") if isinstance(server, Mapping) else None
            if isinstance(port, int):
                ports.add(port)
    return sorted(ports)


async def start_relay(sandbox: AsyncSandbox, host: str, ports: list[int]) -> None:
    """Start the loopback relay in the sandbox and wait until it listens."""
    command = " ".join(shlex.quote(part) for part in ["python3", RELAY_PATH, host, *map(str, ports)])
    started = await sandbox.exec(f"nohup {command} > {RELAY_LOG} 2>&1 &", timeout_s=30)
    if started.error_type is not None or started.return_code != 0:
        raise RuntimeError(f"Could not start the sandbox relay: {started.error_type} {started.stderr or ''}")
    ready = await sandbox.exec(
        f"for i in $(seq 100); do grep -q 'relay ready' {RELAY_LOG} && exit 0; sleep 0.1; done; cat {RELAY_LOG}; exit 1",
        timeout_s=30,
    )
    if ready.error_type is not None or ready.return_code != 0:
        raise RuntimeError(f"The sandbox relay did not start: {ready.error_type} {ready.stdout or ''}")


def node_addresses() -> frozenset[str]:
    """This host's own addresses: Gym's servers on this node call each other from one of them."""
    addresses = {"127.0.0.1", "::1"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            addresses.add(str(info[4][0]))
    except OSError:
        pass
    return frozenset(addresses)


def is_trusted_caller(
    host: str | None,
    *,
    local: frozenset[str],
    networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...],
) -> bool:
    """Whether a caller may reach every route: this node itself, or an address in ``networks``."""
    try:
        address = ipaddress.ip_address(host or "")
    except ValueError:
        return False
    if address.version == 6 and address.ipv4_mapped is not None:  # an IPv4 caller of a dual-stack socket
        address = address.ipv4_mapped
    return address.is_loopback or str(address) in local or any(address in network for network in networks)


@dataclass
class _Episode:
    """An Environment Server episode, kept by its ``resources_session_id`` until it is closed."""

    episode_id: EpisodeId
    task_id: TaskId
    tools: list[str]
    # In sandbox mode: the episode's file API token, its sandbox and the access lent to the harness, and whether
    # its verification has started, from which point its tools are closed.
    token: str | None = None
    sandbox: AsyncSandbox | None = None
    access: SandboxAccess | None = None
    grading: bool = False


class OfficeEnvResourcesServer(SimpleResourcesServer):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: OfficeEnvResourcesServerConfig
    # Episodes by Gym's session ID, which an Environment Server's seed sets to its resources_session_id.
    service: EpisodeService = Field(default_factory=EpisodeService)
    _episodes: dict[str, _Episode] = PrivateAttr(default_factory=dict)
    _closed: ClosedResourcesSessionRecords[EpisodeId] = PrivateAttr()
    # Sandbox mode only: episodes by file API token, a lock per episode for the awaits on its sandbox (kept while a
    # request holds or awaits it, so a closed episode, or an ID never seeded, leaves none), whether shutdown has
    # begun, and what get_session_secret and _require_gym_caller read.
    _tokens: dict[str, str] = PrivateAttr(default_factory=dict)
    _locks: weakref.WeakValueDictionary[str, asyncio.Lock] = PrivateAttr(default_factory=weakref.WeakValueDictionary)
    _shutting_down: bool = PrivateAttr(default=False)
    _session_secret: str | None = PrivateAttr(default=None)
    _node_addresses: frozenset[str] = PrivateAttr(default=frozenset())
    _trusted_networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = PrivateAttr(default=())

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._closed = ClosedResourcesSessionRecords(self.config.session_close_retry_window_seconds)
        if self.config.artifacts_dirs:  # the service reads a row's artifact references back from them at seed
            self.service = EpisodeService(artifacts=ArtifactStore(self.config.artifacts_dirs))
        if self.config.sandbox_provider is None:
            return
        if self.config.num_workers not in (None, 1):
            raise ValueError("OfficeEnv sandbox episodes are process-local and require num_workers=1")
        # One process holds every episode, so one secret signs all of its sessions.
        self._session_secret = token_urlsafe(32)
        self._node_addresses = node_addresses()
        self._trusted_networks = tuple(ipaddress.ip_network(network) for network in self.config.trusted_networks)

    def get_session_secret(self) -> str:
        """The key that signs session cookies and MCP tokens: a per-process secret in sandbox mode, where a
        sandbox can reach this server; Gym's name-derived key, which anyone can compute, otherwise."""
        return self._session_secret or super().get_session_secret()

    def _require_gym_caller(self, request: Request) -> None:
        """In sandbox mode, refuse a caller from another host: seeding, closing, verification, the app catalog
        and the tool routes answer Gym's servers on this node (and ``trusted_networks``) only. A check in the
        handlers, not middleware, because Gym's MCP exposure dispatches tool calls straight to them (as from
        loopback) and refuses servers with middleware of their own."""
        if self.config.sandbox_provider is None:
            return
        host = request.client.host if request.client else None
        if not is_trusted_caller(host, local=self._node_addresses, networks=self._trusted_networks):
            raise HTTPException(status_code=403, detail="This route answers Gym's servers only")

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.get("/officeenv/apps")(self.list_apps)
        if self.config.sandbox_provider is not None:
            self._setup_sandbox_routes(app)
        app.post("/{tool_name}")(self.tool_call)  # registered last: fixed routes win
        return app

    def _setup_sandbox_routes(self, app: FastAPI) -> None:
        """The file API, and a lifespan that stops the sandboxes of open episodes at shutdown."""
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        app.get("/officeenv/files/{app_name}/{file_id}")(self.download_file)
        app.put("/officeenv/files/{app_name}/{file_id}")(self.replace_file)
        app.post("/officeenv/files")(self.upload_file)

    async def list_apps(self, request: Request) -> dict[str, Any]:
        self._require_gym_caller(request)
        return self.service.catalog()

    async def seed_session(
        self, request: Request, body: ResourcesSeedSessionRequest | OfficeEnvSeedSessionRequest
    ) -> ResourcesSeedSessionResponse | OfficeEnvSeedSessionResponse:
        self._require_gym_caller(request)
        if isinstance(body, ResourcesSeedSessionRequest):
            if self.config.sandbox_provider is not None:
                return await self._seed_sandboxed_episode(request, body)
            return self._seed_episode(request, body)
        reply = self._seed(request.session[SESSION_ID_KEY], body.model_dump())
        _SEEDED_TOOLS.set(reply.body["tools"])
        return OfficeEnvSeedSessionResponse(**reply.body)

    def _seed_episode(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        """Seed an Environment Server episode under its ``resources_session_id``. Seeding the same ID
        for the same episode again answers as the first seed did; a closed ID is never seeded again."""
        session_id = body.resources_session_id
        # Nothing below awaits, so no other request runs between the checks and the updates.
        if session_id in self._closed:
            raise ResourcesSessionConflictError(f"Resources session {session_id} is already closed")
        episode = self._episodes.get(session_id)
        if episode is None:
            reply = self._seed(session_id, body.task_data)
            episode = self._episodes[session_id] = _Episode(body.episode_id, body.task_id, reply.body["tools"])
        elif (episode.episode_id, episode.task_id) != (body.episode_id, body.task_id):
            raise ResourcesSessionConflictError("resources_session_id is already bound to another episode or task")
        # Gym's session routing names the cookie session after the ID too; this keeps the two the same without it.
        request.session[SESSION_ID_KEY] = session_id
        _SEEDED_TOOLS.set(episode.tools)
        return ResourcesSeedSessionResponse(resources_session_id=session_id)

    async def _seed_sandboxed_episode(
        self, request: Request, body: ResourcesSeedSessionRequest
    ) -> ResourcesSeedSessionResponse:
        """Seed an Environment Server episode and lend it a sandbox. Seeding the same ID for the same
        episode again lends the same sandbox; a sandbox that fails to start leaves no world behind, so
        the seed can be retried. Once shutdown has begun, a seed starts no sandbox (503)."""
        session_id = body.resources_session_id
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            if self._shutting_down:  # shutdown stops the sandboxes it finds, and would not find this one
                raise HTTPException(status_code=503, detail="The server is shutting down")
            new = session_id not in self._episodes
            self._seed_episode(request, body)
            episode = self._episodes[session_id]
            if new:
                try:
                    await self._lend_sandbox(session_id, episode, task_id=body.task_id.task_id)
                except BaseException:
                    del self._episodes[session_id]
                    self._discard(session_id)
                    raise
        return ResourcesSeedSessionResponse(resources_session_id=session_id, sandbox_access=episode.access)

    async def _lend_sandbox(self, session_id: str, episode: _Episode, *, task_id: str) -> None:
        """Start the episode's sandbox with ``OFFICEENV.md`` and the episode's file API in its environment."""
        config = self.config.sandbox_config
        if not config.image:
            raise ValueError("Set sandbox_config.image to the episode sandbox image (see sandbox/Dockerfile)")
        named_configs = get_global_config_dict()
        files = {f"{config.workdir.rstrip('/')}/OFFICEENV.md": WORKSPACE_GUIDE}
        if config.host_relay:
            files[RELAY_PATH] = RELAY_SOURCE
        spec = config.spec(
            self.config.sandbox_provider,
            image=config.image,
            workdir=config.workdir,
            metadata={"officeenv-task": task_id[:63]},
            files=files,
            named_configs=named_configs,
        )
        token = token_urlsafe(32)
        files_api_url = self.config.files_api_url or f"http://{self.config.host}:{self.config.port}"
        spec = replace(
            spec, env={**spec.env, "OFFICEENV_API_URL": files_api_url.rstrip("/"), "OFFICEENV_API_TOKEN": token}
        )
        sandbox = AsyncSandbox(resolve_provider_config(self.config.sandbox_provider, named_configs))
        try:
            if config.host_relay:
                ports = relay_ports(named_configs, own_port=self.config.port)
                await sandbox.start_with_setup(spec, lambda box: start_relay(box, config.host_relay, ports))
            else:
                await sandbox.start(spec)
            access = SandboxAccess(
                connection=DirectSandboxConnection(
                    provider_config_ref=self.config.sandbox_provider, descriptor=await sandbox.serialize()
                ),
                workdir=config.workdir,
            )
        except BaseException:
            await sandbox.stop()  # also closes the provider of a sandbox that never started
            raise
        episode.token, episode.sandbox, episode.access = token, sandbox, access
        self._tokens[token] = session_id

    async def _release(self, episode: _Episode) -> None:
        """Stop the episode's sandbox and revoke its file API token; a failed stop can be retried."""
        self._tokens.pop(episode.token, None)
        if episode.sandbox is not None:
            await episode.sandbox.stop()
            episode.sandbox = None

    async def shutdown(self) -> None:
        """Stop the sandboxes of open episodes. A seed still starting one holds its episode's lock, so the stop waits
        for it, and a seed that comes later starts none."""
        self._shutting_down = True
        for session_id in set(self._episodes) | set(self._locks):  # a seed awaiting its lock has no episode yet
            async with self._locks.setdefault(session_id, asyncio.Lock()):
                episode = self._episodes.get(session_id)
                if episode is None:
                    continue
                try:
                    await self._release(episode)
                except Exception:
                    LOGGER.exception("Failed to stop the sandbox of OfficeEnv session %s", session_id)

    def _seed(self, session_id: str, row: dict[str, Any]) -> Reply:
        """Build the session ``row`` describes. A malformed task, an unknown app or a row from
        another OfficeEnv major version fails the seed with OfficeEnv's status and error code;
        nothing is stored, so the seed can be retried."""
        reply = self.service.seed(session_id, row)
        if reply.status != 200:
            raise HTTPException(status_code=reply.status, detail=reply.body)
        if self.config.expose_tools_over_mcp:
            catalog = tools_exposure_catalog()
            tools = self.service.store.get(session_id).tools()
            unserved = sorted(tool["name"] for tool in tools if catalog.get(tool["name"]) != tool)
            if unserved:
                self._discard(session_id)
                message = (
                    f"The MCP endpoint serves the tools exposure's tools, not {unserved[:5]}: "
                    "emit the row with exposure mode 'tools' and the default separator"
                )
                raise HTTPException(
                    status_code=400, detail={"error": {"code": "unsupported_exposure", "message": message}}
                )
        return reply

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        """Free the episode's world unless verification already has, and stop its sandbox; a retried
        close, or the close of a session that was never seeded, succeeds."""
        self._require_gym_caller(request)
        if self.config.sandbox_provider is None:
            return self._close(body)
        # A seed still starting the episode's sandbox holds the lock, so the close stops that sandbox too.
        async with self._locks.setdefault(body.resources_session_id, asyncio.Lock()):
            episode = self._episodes.get(body.resources_session_id)
            if episode is not None and episode.episode_id == body.episode_id:
                await self._release(episode)  # raises before the episode is forgotten, so a retried close stops it
            return self._close(body)

    def _close(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        session_id = body.resources_session_id
        # Nothing below awaits, so no other request runs between the checks and the updates.
        closed = self._closed.get(session_id)
        if closed is None:
            episode = self._episodes.get(session_id)
            if episode is not None and episode.episode_id != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the seeded resources session")
            self._episodes.pop(session_id, None)
            self._discard(session_id)
            self._closed.remember(session_id, body.episode_id)
        elif closed != body.episode_id:
            raise ResourcesSessionConflictError("episode_id does not match the closed resources session")
        return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def _stop_before_grading(self, session_id: str) -> None:
        """In sandbox mode, close the episode's tools and file API and stop its sandbox before it is graded, so a
        process the harness left running in the sandbox cannot change what is graded. A failed stop is logged and
        grading goes on; the episode keeps the sandbox, so close stops it."""
        episode = self._episodes.get(session_id) if self.config.sandbox_provider is not None else None
        if episode is None:
            return
        episode.grading = True
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            try:
                await self._release(episode)
            except Exception:
                LOGGER.exception("Could not stop the sandbox of OfficeEnv session %s before grading", session_id)

    async def retire_session_state(self, session_id: str) -> None:
        # A seed still starting the episode's sandbox holds the lock, so retiring stops that sandbox too.
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            episode = self._episodes.get(session_id)
            if episode is not None:
                await self._release(episode)
        self._discard(session_id)

    def _discard(self, session_id: str) -> None:
        """Free a session's world without grading it."""
        self.service.store.sessions.pop(session_id, None)
        self.service.notes.pop(session_id, None)

    async def tool_call(self, request: Request, tool_name: str) -> Response:
        # app-level errors are tool outputs the model sees (200); only a body that is no JSON
        # object (400) and a call without a session (404) are integration failures
        self._require_gym_caller(request)
        session_id = request.session[SESSION_ID_KEY]
        if self.service.store.get(session_id) is None:
            raise ResourcesSessionLostError("This session has ended.")
        episode = self._episodes.get(session_id)
        if episode is not None and episode.grading:
            raise HTTPException(
                status_code=409, detail="This episode's verification has started; its tools are closed"
            )
        reply = self.service.call(session_id, tool_name, await request.body())
        return Response(content=reply.text(), media_type="application/json", status_code=reply.status)

    async def verify(
        self, request: Request, body: OfficeEnvVerifyRequest
    ) -> OfficeEnvVerifyResponse | OfficeEnvLengthPenaltyVerifyResponse:
        self._require_gym_caller(request)
        session_id = request.session[SESSION_ID_KEY]
        await self._stop_before_grading(session_id)
        result = self.service.verify(session_id)
        if result.get("officeenv_failure") == "no_session":
            # Never seeded, verified already, closed, or gone with a restart: the world went with
            # it, so the reward would measure the environment, not the policy.
            result = result | lost_session_verification(result["failure_reason"])
        if not self.config.length_penalty.enabled:
            return OfficeEnvVerifyResponse(**{**body.model_dump(), **result})
        penalty = length_penalty_fields(result["reward"], body.response, self.config.length_penalty)
        return OfficeEnvLengthPenaltyVerifyResponse(**{**body.model_dump(), **result, **penalty})

    def mcp_tools(self, harvested: list["MCPTool"], catchall: Any | None) -> list["MCPTool"] | None:
        """Every tool of the ``tools`` exposure, dispatched through the tool route; no other route is a tool."""
        if catchall is None:
            return []
        catalog = tools_exposure_catalog()
        return [catchall.tool(name, tool["parameters"], tool["description"]) for name, tool in catalog.items()]

    def mcp_allowed_tools_for_session(self, seed_body: dict[str, Any]) -> list[str] | None:
        """The tools of the episode this request seeded: its MCP token admits only those."""
        tools = _SEEDED_TOOLS.get()
        if tools is None:
            raise ValueError("No OfficeEnv session was seeded in this request, so no MCP tools can be granted")
        return tools

    # The file API, in sandbox mode: each request names its episode by the episode's bearer token.

    def _file_session(self, request: Request) -> str:
        scheme, _, token = (request.headers.get("authorization") or "").partition(" ")
        session_id = self._tokens.get(token.strip()) if scheme.lower() == "bearer" else None
        if session_id is None:
            raise HTTPException(status_code=401, detail="Send the episode's token as 'Authorization: Bearer <token>'")
        return session_id

    def _refuse_once_grading(self, session_id: str) -> None:
        """Refuse a file API write once the episode's verification has begun. The token is revoked then, but an
        upload it admitted earlier may still be streaming in, and must not change what is graded."""
        episode = self._episodes.get(session_id)
        if episode is not None and episode.grading:
            raise HTTPException(status_code=409, detail="The task has ended: its files are closed")

    def _file_call(self, session_id: str, app_name: str, verb: str, args: dict[str, Any]) -> dict[str, Any]:
        ops = FILE_OPS.get(app_name)
        if ops is None:
            raise HTTPException(status_code=404, detail=f"File transfer serves {sorted(FILE_OPS)}, not {app_name!r}")
        reply = self.service.call(session_id, f"{app_name}_{ops[verb]}", json.dumps(args))
        body = reply.body if isinstance(reply.body, dict) else json.loads(reply.body)
        status = reply.status if reply.status != 200 else _error_status(body)
        if status is not None:
            raise HTTPException(status_code=status, detail=body)
        return body

    @staticmethod
    async def _content_args(request: Request) -> dict[str, str]:
        """The request body as the app's upload arguments: UTF-8 text as ``content``, any other
        bytes base64-encoded as ``content_base64``, so a binary file keeps every byte. The body is
        read in chunks and refused as soon as it passes the upload limit."""
        data = bytearray()
        async for chunk in request.stream():
            data += chunk
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"Uploads are limited to {MAX_UPLOAD_BYTES} bytes")
        try:
            return {"content": data.decode("utf-8")}
        except UnicodeDecodeError:
            return {"content_base64": base64.b64encode(data).decode("ascii")}

    async def download_file(self, request: Request, app_name: str, file_id: str) -> Response:
        body = self._file_call(self._file_session(request), app_name, "download", {"file_id": file_id})
        content, encoded = body.get("content"), body.get("contentBase64")
        if isinstance(content, str):  # a text file: its UTF-8 bytes
            data, media_type = content.encode("utf-8"), "text/plain"
        elif isinstance(encoded, str):  # a binary file: its bytes as the app keeps them
            try:
                data = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError):
                raise HTTPException(status_code=502, detail="The app returned malformed file content") from None
            media_type = "application/octet-stream"
        else:
            raise HTTPException(status_code=502, detail="The app returned no file content")
        return Response(content=data, media_type=body.get("mimeType") or media_type)

    async def upload_file(
        self, request: Request, app: str, name: str, parent: str | None = None, mime_type: str | None = None
    ) -> JSONResponse:
        session_id = self._file_session(request)
        self._refuse_once_grading(session_id)
        args: dict[str, Any] = {"name": name, **await self._content_args(request)}
        self._refuse_once_grading(session_id)  # verification may have begun while the body streamed in
        if parent:
            args["parents"] = [parent]
        if mime_type:
            args["mime_type"] = mime_type
        return JSONResponse(self._file_call(session_id, app, "create", args))

    async def replace_file(
        self, request: Request, app_name: str, file_id: str, mime_type: str | None = None
    ) -> JSONResponse:
        session_id = self._file_session(request)
        self._refuse_once_grading(session_id)
        args: dict[str, Any] = {"file_id": file_id, **await self._content_args(request)}
        self._refuse_once_grading(session_id)  # verification may have begun while the body streamed in
        if mime_type:
            args["mime_type"] = mime_type
        return JSONResponse(self._file_call(session_id, app_name, "update", args))


if __name__ == "__main__":
    OfficeEnvResourcesServer.run_webserver()
