# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OfficeEnv's episodes for sandboxed harnesses: the OfficeEnv resources server, with a sandbox lent to each episode.

``OfficeEnvSandboxResourcesServer`` subclasses the OfficeEnv server (``resources_servers/officeenv``) and keeps its rows,
routes, grading and options, which that server's README documents with the OfficeEnv release to install. Each episode
session here also gets a sandbox from Gym's sandbox providers (``sandbox_provider``), lent to the harness through Gym's
``LentSandboxes`` (an ``operate`` descriptor: the harness uses the sandbox, and only this server stops it) and stopped
before the episode is graded; an agent-run seed (the simple agent's ``/run``) gets none and runs as on the OfficeEnv
server.

The sandbox gets the episode's file API (``/v1/files``) in ``COMPANY_API_URL`` and
``COMPANY_API_TOKEN`` (``README.md`` in the workspace documents it), so file content moves between
the workspace and the file apps over HTTP instead of through the model's context. Every such
episode runs in OfficeEnv's workspace mode from its seed (``officeenv.files``,
``EpisodeService.enable_workspace``), so the apps' own calls keep content out too
(``sandbox_config.inline_file_bytes``): a download is saved in the workspace and answered with its
path, and an upload or an attachment names a workspace file by its path. This server is the mode's
host: it reads the files a call names from the sandbox before the call runs, with its own path,
link and size checks (``EpisodeService.workspace_reads``), writes the files a call saved there
before it answers (``Reply.workspace_writes``), and serves OfficeEnv's file call as the file API
(``EpisodeService.file_download``, ``file_create``, ``file_replace``).

A sandboxed episode's row exposure decides how its harness reaches the apps (``SANDBOX_EXPOSURES``):
one MCP tool per op (``tools``); one command per app in the sandbox (``cli``: ``slack post-message
--channel=...``, which sends its arguments to ``/v1/cli``), with no MCP tools, so any harness with a
shell and curl runs it; or ``api_search`` and its documentation tool over MCP, which find endpoints
the harness calls with curl at ``/v1/<app>/<op>``. The documentation tool goes by the name the
episode's exposure gives it: ``api_fetch``, or ``api_docs`` with the exposure option ``docs_tool``
(OfficeEnv 1.11.0, ``DOCS_TOOLS``). The file API, the content kept out of the conversation and the
workspace guide serve every exposure.

A row may say what its task needs in the sandbox beyond the shell, curl and Python with pandas and
openpyxl (``sandbox_needs``: ``documents``, ``office``, ``TOOL_SETS``). Its episode gets the first
configured image that provides them all (``sandbox_config.image`` and ``provides``, then
``sandbox_config.images``), whose tools the workspace guide names; a row no configured image serves
is refused before its world is built (400 ``unsupported_needs``).

What a sandbox reads of this server names a company's API, never the environment behind it or Gym:
the variables, routes, guide, answers and session cookie (``get_session_cookie_name``), and with the
config's ``mcp_server_name`` and ``mcp_bearer_token`` the MCP endpoint too. A sandbox can reach this
server, so the server guards it: seeding, closing, verification, metrics, the reverify mode, the app
catalog, the task state and the tool routes answer callers other than this node and ``trusted_networks``
as routes that do not exist (404), before reading their bodies, which leaves sandboxes the MCP endpoint,
the file API and their exposure's routes, and no generated docs are served; and session cookies and MCP tokens are signed with a per-process secret
(``get_session_secret``) instead of Gym's key, which anyone can derive from the server's name. With
``sandbox_config.block_internet`` a sandbox reaches nothing else: OpenSandbox's egress policy refuses
every outbound connection except to this server and the model servers.
"""

import asyncio
import functools
import hashlib
import ipaddress
import json
import logging
import os
import posixpath
import shlex
import socket
import stat
import sys
import tempfile
import time
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from secrets import token_urlsafe
from typing import TYPE_CHECKING, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from nemo_gym.agents import restricted_network_policy
from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest, get_global_config_dict
from nemo_gym.config.global_config import SERVER_SHUTDOWN_TIMEOUT_SECONDS_KEY_NAME
from nemo_gym.core import PROVIDER_UNAVAILABLE, TaskId
from nemo_gym.sandbox import AsyncSandbox, SandboxAccess, SandboxConfig
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.server_runtime import (
    LentSandboxes,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionLostError,
    ReverifyMode,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.officeenv.app import (
    OfficeEnvLengthPenaltyVerifyResponse,
    OfficeEnvResourcesServer,
    OfficeEnvResourcesServerConfig,
    OfficeEnvSeedSessionRequest,
    OfficeEnvSeedSessionResponse,
    OfficeEnvTaskState,
    OfficeEnvVerifyRequest,
    OfficeEnvVerifyResponse,
    _Episode,
    tools_exposure_catalog,
)


if TYPE_CHECKING:
    from nemo_gym.server_runtime import MCPTool

# The OfficeEnv server's module, imported above, checks OfficeEnv's install first: without OfficeEnv, or with a release
# older than its README names, the import stops there, naming the step that installs it.
from officeenv.artifacts import write_artifact
from officeenv.exposure import api_search as api_search_exposure
from officeenv.files import FileApp, FileCallError, Unreadable, UploadHint, Workspace, WorkspaceWrite
from officeenv.service import Reply, parse_arguments


LOGGER = logging.getLogger(__name__)
# One line per seed, the seeds ahead of it, its wait for its turn and its world's build, timestamped for a
# CPU sample of this process to be read against (_show_seed_lines)
SEED_LOGGER = logging.getLogger(f"{__name__}.seeds")


class _CurrentStderr(logging.StreamHandler):
    """Writes to whatever ``sys.stderr`` is when a record comes (a test runner swaps it per test)."""

    @property
    def stream(self) -> Any:
        return sys.stderr

    @stream.setter
    def stream(self, value: Any) -> None:
        pass


def _show_seed_lines() -> None:
    """Gym's servers configure no logging, and Python's last resort prints only warnings: the seed lines get a handler."""
    if not any(isinstance(handler, _CurrentStderr) for handler in SEED_LOGGER.handlers):
        handler = _CurrentStderr()
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        SEED_LOGGER.addHandler(handler)
    SEED_LOGGER.setLevel(logging.INFO)


# The most bytes of text a sandboxed episode's answers carry inline, and its calls take pasted, by default
# (sandbox_config.inline_file_bytes); and in place of none: as much text as the file API takes, while other files still
# move through the workspace, which the episode's trace needs
DEFAULT_INLINE_FILE_BYTES = 8192
UNLIMITED_INLINE_FILE_BYTES = 20_000_000
# At shutdown, of Gym's run-wide grace (server_shutdown_timeout_seconds: how long its run command waits for the servers
# it signalled to stop before it kills them, 1 s unless the config sets it), the part kept for uvicorn's own wait
# (0.5 s) and the exit; the sandboxes get the rest (LentSandboxes.release_all)
SHUTDOWN_MARGIN_S = 2.0
# In sandbox mode, before each world build the event loop runs what is ready for this long, so the requests in flight
# finish before the build holds the loop: a harness's MCP setup is a few requests of a few dozen steps of the loop each,
# and a training cap's burst of setups and tool calls needs many of them. A fixed gap: under uvloop nothing tells when
# the loop has run out of work. While seeds queue it costs a seed rate (at 0.5 s builds, 1.63 a second with 10 ms, 1.41
# with this); tool calls, a hundred or so per episode against one build, are answered in the gap they arrive in.
SEED_TURN_YIELD_S = 0.1

# What an episode image may have besides the shell, curl and Python 3.12 with pandas and openpyxl (sandbox/Dockerfile),
# by the names sandbox_config.provides and a row's sandbox_needs use, as the workspace guide names it to the model
# (sandbox/Dockerfile.documents installs both).
ToolSet = Literal["documents", "office"]
TOOL_SETS: dict[str, str] = {
    "documents": "python-docx, python-pptx, pypdf, pdfplumber and reportlab for Python, and Poppler's pdftotext, "
    "pdfinfo and pdftoppm",
    "office": "LibreOffice without a display, which converts documents (soffice --headless --convert-to pdf <file>, "
    "or docx, xlsx, pptx) and recalculates a workbook's formulas as it converts it",
}
INSTALLED = "bash, curl and Python 3.12 with pandas and openpyxl"
NEEDS_KEY = "sandbox_needs"  # the row's list of TOOL_SETS its task needs in its sandbox


class SandboxImage(BaseModel):
    """An episode image and what it has besides the shell, curl and Python with pandas and openpyxl (TOOL_SETS)."""

    image: str = Field(description="The image's reference, as the sandbox provider pulls it")
    provides: list[ToolSet] = Field(
        default_factory=list,
        description="The tool sets the image has (TOOL_SETS), which a row's sandbox_needs are matched against",
    )


class OfficeEnvSandboxConfig(SandboxConfig):
    """The sandbox lent to each episode session."""

    image: str | None = Field(default=None, description="OCI image, e.g. one built from sandbox/Dockerfile")
    provides: list[ToolSet] = Field(
        default_factory=list,
        description="What `image` has besides the shell, curl and Python with pandas and openpyxl: documents "
        "(python-docx, python-pptx, pypdf, pdfplumber, reportlab and Poppler's tools) and office (LibreOffice without "
        "a display); sandbox/Dockerfile.documents adds both to an image",
    )
    images: list[SandboxImage] = Field(
        default_factory=list,
        description="More images and what each provides, for rows whose sandbox_needs `image` does not provide: such "
        "a row's episode gets the first of these that provides them all, and a row none provides is refused "
        "(400 unsupported_needs)",
    )
    workdir: str = "/workspace"
    host_relay: str | None = Field(
        default=None,
        description="Host alias the sandbox forwards this server's and the model servers' loopback ports to "
        "(colima: host.lima.internal), for servers that bind loopback on a workstation; off by default",
    )
    block_internet: bool = Field(
        default=False,
        description="Refuse the sandbox every outbound connection except to this server and the model servers "
        "(OpenSandbox's egress policy, which this sets as provider_options.network_policy); OpenSandbox only",
    )
    inline_file_bytes: int | None = Field(
        default=DEFAULT_INLINE_FILE_BYTES,
        ge=0,
        description="Workspace mode's inline limit (officeenv.files.Workspace.inline_bytes): file content stays out "
        "of the conversation, a download saved in the workspace and answered with its path, carrying the content "
        "only for a text file of at most this many bytes, and a call that pastes binary content, or text over this "
        "many bytes, is refused with the way to send the file by its path. None: text of up to 20 MB goes inline; "
        "other files still move through the workspace",
    )
    mcp_tool_prefix: str = Field(
        default="",
        description="How the harness names this server's MCP tools, for the workspace guide: Pi lists them as "
        "'<mcp_server_name>_<tool>', Hermes as 'mcp_<mcp_server_name>_<tool>', Codex as "
        "'mcp__<mcp_server_name>__<tool>'",
    )

    @model_validator(mode="after")
    def _one_egress_policy(self) -> "OfficeEnvSandboxConfig":
        if self.block_internet and "network_policy" in self.provider_options:
            raise ValueError("block_internet sets provider_options.network_policy; set one or the other")
        if self.block_internet and self.host_relay:
            raise ValueError("block_internet is OpenSandbox's egress policy; host_relay serves a workstation's Docker")
        return self

    def image_for(self, needs: list[str]) -> SandboxImage | None:
        """The first configured image that provides all of ``needs``, ``image`` before ``images``; None if none does."""
        first = [SandboxImage(image=self.image, provides=self.provides)] if self.image else []
        return next((each for each in first + self.images if set(needs) <= set(each.provides)), None)


class OfficeEnvSandboxResourcesServerConfig(OfficeEnvResourcesServerConfig):
    """The OfficeEnv server's options, and the sandbox each episode session gets."""

    sandbox_provider: str = Field(
        description="Top-level sandbox config block (e.g. `sandbox`) the sandboxes come from"
    )
    sandbox_config: OfficeEnvSandboxConfig = Field(default_factory=OfficeEnvSandboxConfig)
    files_api_url: str | None = Field(
        default=None,
        description="Base URL a sandbox reaches this server at (default: this server's configured host and port)",
    )
    trusted_networks: list[str] = Field(
        default_factory=list,
        description="Networks (CIDR) besides this node whose callers reach every route",
    )
    workspace_trace_store: str | None = Field(
        default=None,
        description=(
            "Where the bytes the server hands a sandboxed episode are kept (the files its calls read, and the files "
            "its file API calls send), by SHA-256 in officeenv.artifacts' store format, so its trace "
            "(officeenv_workspace) replays without a sandbox: an absolute path. A write that fails is logged and "
            "counted (sandbox_trace_failures), and the call goes on. None keeps none."
        ),
    )

    @field_validator("workspace_trace_store")
    @classmethod
    def _trace_store_is_absolute(cls, store: str | None) -> str | None:
        if store is not None and not os.path.isabs(store):
            raise ValueError(f"the workspace trace store must be an absolute path: {store}")
        return store


MAX_UPLOAD_BYTES = 20_000_000


# What a sandbox reads of this server names a company's API, never the environment behind it or Gym: the variables
# that name the episode's API, its routes, and the guide.
API_URL_ENV, API_TOKEN_ENV = "COMPANY_API_URL", "COMPANY_API_TOKEN"
API_PREFIX = "/v1"
FILES_ROUTE = f"{API_PREFIX}/files"
CLI_ROUTE = f"{API_PREFIX}/cli"
GUIDE_NAME = "README.md"

# How a sandboxed episode reaches its apps, by the row's exposure: one MCP tool per op (`tools`), one command per app
# (`cli`, through CLI_ROUTE), or endpoints found over MCP and called with curl (`api_search`, through
# /v1/<app>/<op>). Grading reads the apps' final state, so it is the same for all three.
SANDBOX_EXPOSURES = ("tools", "cli", "api_search")

EXPOSURE_GUIDES = {
    "tools": """The company's apps ({apps}) are your tools, one per operation of an app
({prefix}<app>_<operation>). Call them like any other tool.""",
    "cli": """The company's apps are commands here, one per connected app: {apps}.

    <app> <command> [--parameter=value ...]   # e.g. slack post-message --channel='#sales' --text='Ready to review'
    <app> --help                              # the app's commands
    <app> <command> --help                    # a command's parameters, with examples

A command prints its result as JSON and exits 0, or prints the app's error on stderr and exits 1.
Arrays and objects go as JSON under their own flag (--data='[{{"range":"A1","values":[["x"]]}}]');
--flag=null sends null, and a JSON-quoted value stays text (--value='"false"').""",
    "api_search": f"""The company's apps ({{apps}}) are an HTTP API. Find an endpoint with your {{prefix}}api_search
tool, read its parameters with your {{prefix}}{{docs}} tool (it gives the exact command), and call it:

    curl -sS -X POST "${API_URL_ENV}{API_PREFIX}/<app>/<operation>" \\
      -H "Authorization: Bearer ${API_TOKEN_ENV}" -H 'Content-Type: application/json' \\
      -d '<the parameters as a JSON object>'

An endpoint answers with the app's JSON; a call that fails answers with the app's error and an
HTTP error status.""",
}
FIND_FILES = {"tools": "the apps' tools", "cli": "the apps' commands", "api_search": "the apps' endpoints"}

# The API's URL and token as a shell reads them, for a harness whose shell does not inherit the sandbox's environment
# (Codex starts its shell with PATH, HOME, the locale and the XDG folders alone): the apps' commands read the file
# when the variables are unset, and the guide says how to read it.
CREDENTIALS_PATH = "/etc/company-api.env"
CREDENTIALS_GUIDE = f"""The API's URL and token are in the environment variables {API_URL_ENV} and {API_TOKEN_ENV}; a shell
started without them gets them with `. {CREDENTIALS_PATH}`."""


def api_credentials(url: str, token: str) -> str:
    """The file at CREDENTIALS_PATH: the episode's API URL and token, set and exported by a shell that sources it."""
    return f"{API_URL_ENV}={shlex.quote(url)}\n{API_TOKEN_ENV}={shlex.quote(token)}\nexport {API_URL_ENV} {API_TOKEN_ENV}\n"


FILES_GUIDE = f"""File content moves between this workspace and the file apps over HTTP, without going through the
conversation, byte for byte: text files (CSV, TSV, JSON, Markdown) and binary ones (XLSX workbooks,
PDFs, images) alike. <app> is {{codes}}.

    # download a file's content
    curl -sf -H "Authorization: Bearer ${API_TOKEN_ENV}" \\
      "${API_URL_ENV}{FILES_ROUTE}/<app>/<file id>" -o <local path>

    # upload a local file as a new file in a folder
    curl -sf -X POST -H "Authorization: Bearer ${API_TOKEN_ENV}" --data-binary @<local path> \\
      "${API_URL_ENV}{FILES_ROUTE}?app=<app>&name=<file name>&parent=<folder id>"

    # replace an existing file's content
    curl -sf -X PUT -H "Authorization: Bearer ${API_TOKEN_ENV}" --data-binary @<local path> \\
      "${API_URL_ENV}{FILES_ROUTE}/<app>/<file id>"

The file name's extension sets the file type (.csv, .xlsx, .pdf, ...). An upload's URL takes a
parent as &parent=<folder id> and a mime type as &mime_type=<type>, as each app allows:
{{notes}}
Find file and folder ids with {{find}} (list or search files). Uploads are limited to
{MAX_UPLOAD_BYTES // 1_000_000} MB. A request that fails returns the app's error as JSON.
"""

# How the model sends a workspace file with the file API instead of pasting its content into a call: the guide's
# upload commands, as a refusal fills them in for the call's target (UploadHint).
_SEND = f'-H "Authorization: Bearer ${API_TOKEN_ENV}" --data-binary @<local path>'
UPLOAD_HINT = UploadHint(
    replace=f'curl -sf -X PUT {_SEND} "${API_URL_ENV}{FILES_ROUTE}/{{app}}/{{file_id}}"',
    create=f'curl -sf -X POST {_SEND} "${API_URL_ENV}{FILES_ROUTE}?app={{app}}&{{query}}"',
)

CONTENT_GUIDE = """Downloading a file with {find} saves it in this workspace (the answer's savedTo):

{folders}

The answer also carries the content only for a text file of at most {limit} bytes. An upload with
{find} takes a workspace file by its path (content_path), and pasted content only as text of at
most {limit} bytes.
"""


def workspace_guide(
    content_limit: int = DEFAULT_INLINE_FILE_BYTES,
    apps: tuple[str, ...] = (),
    *,
    exposure: str = "tools",
    prefix: str = "",
    tools: tuple[str, ...] = (),
    offline: bool = False,
    docs_tool: str | None = None,
    file_apps: Sequence[FileApp] = (),
) -> str:
    """The workspace's README.md for an episode that connects ``apps``: how its apps are reached (``exposure``,
    ``prefix``, how the harness names this server's MCP tools, and on ``api_search`` the documentation tool as the
    episode's exposure names it, ``docs_tool``, the exposure's default name if None), and, when it connects any, how
    files move between the workspace and the apps the file API reaches (``file_apps``, as EpisodeService.file_apps
    lists them: each one's label, the folder its downloads land in and a line on its file calls), with workspace
    mode's inline limit (``content_limit``). Last, for an image with more than the shell, curl and Python (``tools``,
    TOOL_SETS' names) or a sandbox without the internet (``offline``), what is installed, and whether nothing else can
    be."""
    intro = EXPOSURE_GUIDES[exposure].format(apps=", ".join(apps), prefix=prefix, docs=docs_tool or DOCS_TOOLS[0])
    guide = f"# Company apps\n\n{intro}\n\n{CREDENTIALS_GUIDE}\n"
    if file_apps:
        find = FIND_FILES[exposure]
        guide += "\n" + FILES_GUIDE.format(
            codes=", ".join(f"{spec.app} ({spec.label})" for spec in file_apps),
            notes="".join(f"{spec.label}: {spec.note}.\n" for spec in file_apps),
            find=find,
        )
        width = max(len(f"{spec.folder}/<file id>/<name>") for spec in file_apps) + 3
        folders = "\n".join(f"    {f'{spec.folder}/<file id>/<name>':<{width}}{spec.label}" for spec in file_apps)
        guide += "\n" + CONTENT_GUIDE.format(find=find, folders=folders, limit=content_limit)
    if tools or offline:
        installed = "\n".join(f"- {text}" for text in (INSTALLED, *(TOOL_SETS[name] for name in tools)))
        guide += f"\nInstalled here:\n\n{installed}\n"
        if offline:
            guide += "\nThe sandbox has no internet access, so nothing else can be installed.\n"
    return guide


# This server's caps on the workspace files one call reads (a message's attachments, an upload's content_path): the
# files of one call, their bytes together (Gmail's limit for a message) and each file's (MAX_UPLOAD_BYTES). The
# episode's Workspace carries them, so each call's ReadPlan holds the stricter of them and the op's own, and
# OfficeEnv's refusals name the number.
MAX_ATTACHMENTS = 20
MAX_ATTACHMENT_BYTES = 25_000_000
# Each path is checked in the sandbox before anything is copied out: its kind, a final link checked as a link and not
# followed, and a regular file's size, one line per path ("missing|0" for nothing there), so only a regular file of a
# bounded size is ever downloaded. POSIX test and wc only, which every shell has (GNU's, BusyBox's, and macOS's for the
# local provider), where stat's options differ.
STAT_SCRIPT = (
    'for p; do if [ -L "$p" ]; then echo "link|0"; elif [ -f "$p" ]; then echo "regular file|$(wc -c < "$p" | tr -d " ")"; '
    'elif [ -e "$p" ]; then echo "other|0"; else echo "missing|0"; fi; done'
)


# An api_search episode finds endpoints over MCP and calls them over HTTP: the finding tools, its search and its
# documentation tool, with the argument shapes of OfficeEnv's api_search exposure (a public contract), and its call
# (api_call) made over HTTP. The exposure names the documentation tool api_fetch, or api_docs with its option docs_tool
# (OfficeEnv 1.11.0, which lists the names; api_fetch alone before): MCP serves it under each name, and an episode's
# token admits the one its exposure lists (finding_tools). Descriptions are formatted with the guide's path.
SEARCH_TOOL, CALL_TOOL = "api_search", "api_call"
DOCS_TOOLS: tuple[str, ...] = tuple(getattr(api_search_exposure, "DOCS_TOOLS", ("api_fetch",)))
DOCS_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "endpoint": {
            "type": "string",
            "description": "An endpoint id from search results, e.g. 'slack.post_message'",
        }
    },
    "required": ["endpoint"],
}
DOCS_TOOL_DESCRIPTION = (
    "Fetch an endpoint's documentation: description, parameter schema, examples, and the curl command that calls it."
)
SEARCH_TOOLS: tuple[tuple[str, dict[str, Any], str], ...] = (
    (
        SEARCH_TOOL,
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What you want to do, in words (e.g. 'post a message to a slack channel')",
                },
                "app": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "description": "Restrict results to one connected app",
                },
            },
            "required": ["query"],
        },
        "Search the connected apps' API for endpoints matching what you want to do. Returns endpoint ids and "
        "their HTTP paths; read an endpoint's parameters before calling it with curl (see {guide}).",
    ),
    *((name, DOCS_TOOL_SCHEMA, DOCS_TOOL_DESCRIPTION) for name in DOCS_TOOLS),
)


def finding_tools(tools: Sequence[str]) -> list[str]:
    """The tools an api_search episode reaches over MCP, as its exposure lists them: all but the call, which the
    sandbox makes over HTTP."""
    return [name for name in tools if name != CALL_TOOL]


CURL_CALL = (
    f'curl -sS -X POST "${API_URL_ENV}{{path}}" -H "Authorization: Bearer ${API_TOKEN_ENV}" '
    "-H 'Content-Type: application/json' -d {data}"
)


def endpoint_path(endpoint: str) -> str:
    """The HTTP path an api_search endpoint id (``app.op``) is called at."""
    app, _, op = endpoint.partition(".")
    return f"{API_PREFIX}/{app}/{op}"


def with_http_calls(tool_name: str, text: str, *, docs_tool: str = DOCS_TOOLS[0]) -> str:
    """api_search's results with each endpoint's HTTP path, and the documentation tool's documentation (``docs_tool``,
    as the episode's exposure names it) with the curl command that calls the endpoint (its first example as the body):
    in a sandbox, endpoints are called over HTTP. What an endpoint does with file content there is OfficeEnv's to say,
    in workspace mode's own words."""
    if tool_name not in (SEARCH_TOOL, docs_tool):
        return text
    try:
        body = json.loads(text)
    except ValueError:
        return text
    if not isinstance(body, dict) or "error" in body:
        return text
    if tool_name == SEARCH_TOOL:
        for result in body.get("results") or []:
            if isinstance(result, dict) and isinstance(result.get("endpoint"), str):
                result["path"] = endpoint_path(result["endpoint"])
    elif isinstance(body.get("endpoint"), str):
        path = endpoint_path(body["endpoint"])
        examples = body.get("examples") or []
        example = examples[0] if examples and isinstance(examples[0], dict) else {}
        data = shlex.quote(json.dumps(example, ensure_ascii=False))
        body["http"] = {"method": "POST", "path": path, "curl": CURL_CALL.format(path=path, data=data)}
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"))


def media_type_of(data: bytes) -> str:
    """The media type the file API answers a download with when its app gives none: text/plain for UTF-8 text,
    application/octet-stream for any other bytes."""
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return "application/octet-stream"
    return "text/plain"


def copied_file(local: Path, limit: int) -> bytes | Unreadable:
    """A file a sandbox provider copied to this host: its bytes when it is a regular file of at most ``limit``
    bytes, else why not: too_large, or missing for nothing there, a folder, a device, a pipe or a link, whose
    copy would be read here, not in the sandbox (``docker cp`` copies a link as a link)."""
    try:
        info = local.lstat()
    except OSError:
        return Unreadable("missing")
    if not stat.S_ISREG(info.st_mode):
        return Unreadable("missing")
    return Unreadable("too_large") if info.st_size > limit else local.read_bytes()


async def sandbox_file_sizes(sandbox: Any, paths: list[str]) -> dict[str, int | None]:
    """The size of each of ``paths`` that holds a regular file in the sandbox, None for a link, a folder, a device or
    nothing (``STAT_SCRIPT``, one run for all of them): what a later download copies is bounded before it starts. A
    check that does not run raises: a path it could not check is not missing."""
    done = await sandbox.exec(shlex.join(["sh", "-c", STAT_SCRIPT, "sh", *paths]), cwd="/", timeout_s=60)
    lines = (done.stdout or "").splitlines()
    if done.error_type is not None or done.return_code != 0 or len(lines) != len(paths):
        raise RuntimeError(f"stat: {done.error_type} exit {done.return_code} {(done.stderr or '')[-300:]}")
    sizes: dict[str, int | None] = {}
    for path, line in zip(paths, lines):  # stat prints no name, so a path holding a newline shifts no line
        kind, _, size = line.rpartition("|")
        sizes[path] = int(size) if kind in ("regular file", "regular empty file") and size.isdigit() else None
    return sizes


def _stricter(cap: int | None, own: int) -> int:
    """The stricter of a ReadPlan's cap, which holds this server's own already, and this server's own: a plan without
    one (None) still meets it."""
    return own if cap is None else min(cap, own)


# What /verify reports of a sandboxed episode's workspace from OfficeEnv's counters (EpisodeService.workspace_stats),
# as sandbox_<name>; the files saved there and their bytes are this server's own count, of the writes that landed
WORKSPACE_STATS = ("uploads_refused", "files_attached", "bytes_attached", "files_uploaded", "bytes_uploaded")


RELAY_SOURCE = (Path(__file__).parent / "sandbox" / "relay.py").read_text(encoding="utf-8")
RELAY_PATH = "/usr/local/lib/net-relay/relay.py"
RELAY_LOG = "/tmp/net-relay.log"


def _without_comments(script: str) -> str:
    """A script as the sandbox gets it: its comment lines dropped (the shebang kept), so reading a command shows what
    it runs, not a description of the server behind it."""
    lines = script.splitlines()
    return "\n".join(line for n, line in enumerate(lines) if n == 0 or not line.lstrip().startswith("#")) + "\n"


# The command every app of a `cli` episode is linked to (sandbox/app_command.sh), and the User-Agent it sends, without
# which CLI_ROUTE refuses the call.
APP_COMMAND_SOURCE = _without_comments(
    (Path(__file__).parent / "sandbox" / "app_command.sh").read_text(encoding="utf-8")
)
APP_COMMAND_PATH = "/usr/local/lib/company-cli/app"
APP_COMMAND_DIR = "/usr/local/bin"
CLI_USER_AGENT = "{app}-cli/1.0"


def link_app_commands(apps: tuple[str, ...]) -> str:
    """The command that makes a `cli` episode's apps commands: the app command, made executable, linked under each
    app's name. `ln` refuses a name the image already has there, so a sandbox where an app would not be its command
    fails to start."""
    links = " && ".join(f"ln -s {APP_COMMAND_PATH} {shlex.quote(f'{APP_COMMAND_DIR}/{app}')}" for app in apps)
    return f"chmod 755 {APP_COMMAND_PATH} && mkdir -p {APP_COMMAND_DIR} && {links}"


async def run_setup(sandbox: AsyncSandbox, command: str) -> None:
    """Run one setup command in a started sandbox before it is lent, failing the seed if it fails."""
    done = await sandbox.exec(command, cwd="/", timeout_s=60)
    if done.error_type is not None or done.return_code != 0:
        raise RuntimeError(f"Sandbox setup failed: {done.error_type} {(done.stderr or '')[-500:]}")


def _model_servers(named_configs: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Every model server's config in Gym's merged config: the servers a sandboxed harness calls for completions."""
    servers = []
    for block in named_configs.values():
        models = block.get("responses_api_models") if isinstance(block, Mapping) else None
        servers += [
            server
            for server in (models.values() if isinstance(models, Mapping) else ())
            if isinstance(server, Mapping)
        ]
    return servers


def relay_ports(named_configs: Mapping[str, Any], *, own_port: int) -> list[int]:
    """This server's port and every model server's: what a sandboxed harness connects to."""
    ports = {own_port} | {
        server["port"] for server in _model_servers(named_configs) if isinstance(server.get("port"), int)
    }
    return sorted(ports)


def _reachable_from_a_sandbox(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host.lower().rstrip(".") != "localhost"
    return not (address.is_loopback or address.is_unspecified)


def egress_urls(own_urls: list[str], named_configs: Mapping[str, Any]) -> list[str]:
    """What a sandbox without the internet may still reach: this server at ``own_urls`` (MCP and the file API) and
    the model servers, read from Gym's config. A model server on loopback is left out: no sandbox reaches it."""
    urls = list(own_urls)
    for server in _model_servers(named_configs):
        host = server.get("host")
        if isinstance(host, str) and _reachable_from_a_sandbox(host):
            urls.append(f"http://{host}:{server.get('port')}")
    return urls


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


def binds(address: str) -> bool:
    """Whether a socket here binds to ``address``."""
    try:
        family = socket.AF_INET6 if ipaddress.ip_address(address).version == 6 else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.bind((address, 0))
    except (OSError, ValueError):
        return False
    return True


@functools.lru_cache(maxsize=4096)
def is_own_address(address: str) -> bool:
    """Whether ``address`` is one of this node's own, held by one of its interfaces: a socket binds to it. A sandbox's
    address is not, on whatever network its provider gives it, unless the sandbox shares this node's network."""
    return binds(address)


def binds_only_own_addresses() -> bool:
    """Whether a socket here binds only this node's own addresses, so ``is_own_address`` can tell them: not on a host
    that lets sockets bind any address (Linux's ``net.ipv4.ip_nonlocal_bind``)."""
    return not binds("192.0.2.1")  # TEST-NET-1, which no interface holds


def is_trusted_caller(
    host: str | None,
    *,
    local: frozenset[str],
    networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...],
    own_addresses: bool = False,
) -> bool:
    """Whether a caller may reach every route: this node itself (loopback, ``local``, and with ``own_addresses`` any
    address of its own: ``is_own_address``), or an address in ``networks``."""
    try:
        address = ipaddress.ip_address(host or "")
    except ValueError:
        return False
    if address.version == 6 and address.ipv4_mapped is not None:  # an IPv4 caller of a dual-stack socket
        address = address.ipv4_mapped
    if address.is_loopback or str(address) in local or any(address in network for network in networks):
        return True
    return own_addresses and is_own_address(str(address))


async def gym_caller(request: Request) -> None:
    """The server's caller guard (``_require_gym_caller``) as a FastAPI dependency: a handler with a body depends on it,
    so FastAPI runs it before it validates the body, and a caller the guard refuses learns nothing of its schema."""
    request.app.state.require_gym_caller(request)


@dataclass
class _SandboxEpisode(_Episode):
    """An episode with its sandbox, kept by its ``resources_session_id`` until it is closed."""

    # How its harness reaches its apps (SANDBOX_EXPOSURES), the apps it connects, a `cli` episode's run_command tool,
    # which its apps' commands call, and an `api_search` episode's documentation tool, as its exposure names it
    # (DOCS_TOOLS)
    exposure: str = "tools"
    apps: tuple[str, ...] = ()
    command_tool: str | None = None
    docs_tool: str | None = None
    # Its file API token, its sandbox (lent through LentSandboxes, which stops it) and the access lent to the harness,
    # and whether its verification has started, from which point its tools are closed.
    token: str | None = None
    sandbox: AsyncSandbox | None = None
    access: SandboxAccess | None = None
    grading: bool = False
    provider_failure: str | None = None  # the first sandbox operation the provider failed (_provider_failed)
    # The files its calls saved in the workspace that landed there (_save_downloads), and their bytes: the rest of what
    # moved through its workspace instead of the conversation is OfficeEnv's to count (WORKSPACE_STATS)
    files_saved: int = 0
    bytes_saved: int = 0
    # What the episode sent through its exposure's routes
    commands: int = 0
    commands_refused: int = 0  # calls to CLI_ROUTE made by hand rather than by an app's command
    endpoint_calls: int = 0
    trace_failures: int = 0  # the files it was handed that the workspace trace store could not keep (_keep_in_trace)

    def usage(self) -> dict[str, float]:
        """What the episode sent beside MCP, through its exposure's routes, and the files saved in its workspace, as
        verify metrics."""
        return {
            "sandbox_commands": float(self.commands),
            "sandbox_commands_refused": float(self.commands_refused),
            "sandbox_endpoint_calls": float(self.endpoint_calls),
            "sandbox_files_saved": float(self.files_saved),
            "sandbox_bytes_saved": float(self.bytes_saved),
        }


class OfficeEnvSandboxResourcesServer(OfficeEnvResourcesServer):
    config: OfficeEnvSandboxResourcesServerConfig
    # Episodes by file API token, a lock per episode for the awaits on its sandbox (kept while a request holds or awaits
    # it, so a closed episode, or an ID never seeded, leaves none), whether shutdown has begun, the lent sandboxes
    # (made at the first lend: _lending), and what get_session_secret and _require_gym_caller read.
    _tokens: dict[str, str] = PrivateAttr(default_factory=dict)
    _locks: weakref.WeakValueDictionary[str, asyncio.Lock] = PrivateAttr(default_factory=weakref.WeakValueDictionary)
    _shutting_down: bool = PrivateAttr(default=False)
    _lent: LentSandboxes | None = PrivateAttr(default=None)
    _seeding: asyncio.Semaphore = PrivateAttr(default_factory=lambda: asyncio.Semaphore(1))  # a seed's turn
    _seeds_queued: int = PrivateAttr(default=0)  # seeds waiting for or taking their turn
    _session_secret: str | None = PrivateAttr(default=None)
    _node_addresses: frozenset[str] = PrivateAttr(default=frozenset())
    _own_addresses: bool = PrivateAttr(default=False)
    _trusted_networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = PrivateAttr(default=())

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("OfficeEnv sandbox episodes are process-local and require num_workers=1")
        _show_seed_lines()
        # One process holds every episode, so one secret signs all of its sessions.
        self._session_secret = token_urlsafe(32)
        self._node_addresses = node_addresses()
        self._own_addresses = binds_only_own_addresses()
        if not self._own_addresses:
            LOGGER.warning(
                "This host lets sockets bind any address, so the OfficeEnv sandbox server cannot tell its own "
                "addresses: only loopback, the hostname's addresses and trusted_networks reach Gym's routes"
            )
        self._trusted_networks = tuple(ipaddress.ip_network(network) for network in self.config.trusted_networks)

    def get_session_secret(self) -> str:
        """The key that signs session cookies and MCP tokens: a per-process secret, since a sandbox can reach this
        server and Gym's name-derived key is one anyone can compute."""
        return self._session_secret or super().get_session_secret()

    def get_session_cookie_name(self) -> str:
        """A sandbox's calls get the session cookie too (``curl -v`` shows it), so its name says nothing of the
        environment: "sid_" and a hash of Gym's name for it, which keeps it unique per server."""
        return f"sid_{hashlib.sha256(self.get_session_middleware_key().encode()).hexdigest()[:12]}"

    def _require_gym_caller(self, request: Request) -> None:
        """Answer a caller from another host as a route that does not exist: seeding, closing, verification, metrics,
        the reverify mode, the app catalog, the task state and the tool routes answer Gym's servers on this node (and
        ``trusted_networks``) only. A check the handlers run (``gym_caller`` before reading a body), not middleware,
        because Gym's MCP exposure dispatches tool calls straight to them (as from loopback) and refuses servers with
        middleware of their own.

        A caller on this node calls from one of the node's own addresses (``is_own_address``), whatever its hostname
        resolves to: Debian and Ubuntu map it to 127.0.1.1, and a run with ``+default_host`` (a Docker bridge's
        address, say) is reached from the node's primary address. A sandbox calls from its own network."""
        host = request.client.host if request.client else None
        trusted = is_trusted_caller(
            host, local=self._node_addresses, networks=self._trusted_networks, own_addresses=self._own_addresses
        )
        if not trusted:
            raise HTTPException(status_code=404, detail="Not Found")

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.state.require_gym_caller = self._require_gym_caller  # gym_caller, which handlers with a body depend on
        self._setup_sandbox_routes(app)
        # A sandbox reaches this server, and generated docs would show it every route: none are served.
        docs = {app.openapi_url, app.docs_url, app.swagger_ui_oauth2_redirect_url, app.redoc_url} - {None}
        app.router.routes[:] = [route for route in app.router.routes if getattr(route, "path", None) not in docs]
        return app

    def _setup_sandbox_routes(self, app: FastAPI) -> None:
        """The file API, the `cli` and `api_search` exposures' routes, and a lifespan that stops the sandboxes of open
        episodes at shutdown. None of them is a single path segment, so the tool route (``/{tool_name}``) takes none
        of their calls."""
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        app.get(f"{FILES_ROUTE}/{{app_name}}/{{file_id}}")(self.download_file)
        app.put(f"{FILES_ROUTE}/{{app_name}}/{{file_id}}")(self.replace_file)
        app.post(FILES_ROUTE)(self.upload_file)
        app.post(CLI_ROUTE)(self.run_command_line)
        # A Starlette route, not FastAPI's: Gym's MCP exposure takes any parameterized POST route of FastAPI's for the
        # tool catch-all, and allows one (the tool route).
        app.add_route(f"{API_PREFIX}/{{app_name}}/{{op}}", self.call_endpoint, methods=["POST"])

    async def list_apps(self, request: Request) -> dict[str, Any]:
        self._require_gym_caller(request)
        return await super().list_apps()

    async def aggregate_metrics(
        self, body: AggregateMetricsRequest, _gym: None = Depends(gym_caller)
    ) -> AggregateMetrics:
        return await super().aggregate_metrics(body)

    async def get_reverify_mode(self, request: Request) -> ReverifyMode:
        self._require_gym_caller(request)
        return await super().get_reverify_mode()

    async def task_state(self, request: Request) -> OfficeEnvTaskState:
        self._require_gym_caller(request)
        return await super().task_state(request)

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest | OfficeEnvSeedSessionRequest,
        _gym: None = Depends(gym_caller),
    ) -> ResourcesSeedSessionResponse | OfficeEnvSeedSessionResponse:
        self._require_gym_caller(request)  # a call made without FastAPI, as MCP's direct dispatch makes one
        if isinstance(body, ResourcesSeedSessionRequest):
            return await self._seed_sandboxed_episode(request, body)
        return await super().seed_session(request, body)

    def _new_episode(self, session_id: str, body: ResourcesSeedSessionRequest) -> _SandboxEpisode:
        """A new episode, built in workspace mode (``_seed_in_workspace``), with how its harness reaches its apps:
        by its row's exposure (SANDBOX_EXPOSURES). Its MCP token admits the row's tools, none for a `cli` episode,
        whose apps are commands, and for an `api_search` one the finding tools its exposure lists, whose documentation
        tool is named as the row's exposure names it."""
        reply = self._seed_in_workspace(session_id, body.task_data)
        session = self.service.store.get(session_id)
        tools = list(reply.body["tools"])
        exposure = session.describe()["exposure"]["mode"]
        finding = finding_tools(tools)
        return _SandboxEpisode(
            body.episode_id,
            body.task_id,
            grant={"tools": tools, "cli": [], "api_search": finding}[exposure],
            exposure=exposure,
            apps=tuple(sorted(session.apps)),
            command_tool=tools[0] if exposure == "cli" else None,
            # OfficeEnv's api_search exposure lists its documentation tool beside api_search, and a row that would not
            # is refused before this (_sandbox_unreachable)
            docs_tool=next((name for name in finding if name != SEARCH_TOOL), None)
            if exposure == "api_search"
            else None,
        )

    def _seed_in_workspace(self, session_id: str, row: dict[str, Any]) -> Reply:
        """Build the session ``row`` describes in workspace mode from the start (``_workspace``), so its trace holds
        every call. A seed fails as on the OfficeEnv server, and so does a row whose apps the episode cannot reach
        (``_sandbox_unreachable``); nothing is stored, so the seed can be retried."""
        workspace = self._workspace()
        reply = self.service.seed(session_id, row)
        if reply.status != 200:
            raise HTTPException(status_code=reply.status, detail=reply.body)
        try:
            self.service.enable_workspace(session_id, workspace)
        except BaseException:
            self._discard(session_id)
            raise
        self._refuse_unreachable(session_id, self._sandbox_unreachable(session_id))
        return reply

    def _sandbox_unreachable(self, session_id: str) -> str | None:
        """Why a sandboxed episode's apps cannot be reached, if they cannot. Over MCP its harness reaches the ``tools``
        exposure's tools as workspace mode describes them (the OfficeEnv server's ``_unreachable``); it may reach its
        apps as commands (``cli``) or as endpoints it finds over MCP (``api_search``, with finding tools MCP serves)
        instead, and no other way."""
        session = self.service.store.get(session_id)
        mode = session.describe()["exposure"]["mode"]
        if mode == "api_search":
            served = [name for name, _, _ in SEARCH_TOOLS]
            listed = finding_tools([tool["name"] for tool in session.tools()])
            unserved = [name for name in listed if name not in served]
            if unserved:
                return f"The MCP endpoint serves api_search's finding tools as {served}, not {unserved}"
            if not any(name != SEARCH_TOOL for name in listed):
                return f"An api_search episode finds endpoints with {SEARCH_TOOL} and a documentation tool, not alone"
            return None
        if mode == "cli":
            return None
        if mode not in SANDBOX_EXPOSURES:
            return f"A sandboxed episode reaches its apps through {', '.join(SANDBOX_EXPOSURES)}, not {mode}"
        return self._unreachable(session_id, self._workspace())

    def _sandbox_image(self, row: Mapping[str, Any]) -> SandboxImage:
        """The image a sandboxed row's episode gets: the first configured one that provides all its sandbox_needs
        (TOOL_SETS). A row that names what no sandbox has, or what no configured image provides, is refused before
        its world is built."""
        config = self.config.sandbox_config
        if not config.image and not config.images:
            raise ValueError("Set sandbox_config.image to the episode sandbox image (see sandbox/Dockerfile)")
        needs = row.get(NEEDS_KEY) or []
        known = ", ".join(TOOL_SETS)
        if not isinstance(needs, list) or not all(isinstance(need, str) for need in needs):
            message = f"{NEEDS_KEY} lists what the task needs in its sandbox: some of {known}"
        elif unknown := sorted(set(needs) - set(TOOL_SETS)):
            message = f"{NEEDS_KEY} names {', '.join(unknown)}; a sandbox can provide {known}"
        elif (image := config.image_for(needs)) is not None:
            return image
        else:
            message = (
                f"No configured sandbox image provides {', '.join(sorted(set(needs)))} (sandbox_config.provides and "
                "sandbox_config.images; sandbox/Dockerfile.documents builds one)"
            )
        raise HTTPException(status_code=400, detail={"error": {"code": "unsupported_needs", "message": message}})

    async def _seed_sandboxed_episode(
        self, request: Request, body: ResourcesSeedSessionRequest
    ) -> ResourcesSeedSessionResponse:
        """Seed an episode session and lend it a sandbox, from the image its row needs. Seeding the
        same ID for the same episode again lends the same sandbox; a sandbox that fails to start leaves no
        world behind, so the seed can be retried. Once shutdown has begun, a seed starts no sandbox (503)."""
        session_id = body.resources_session_id
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            new = session_id not in self._episodes
            self._admit()
            image = self._sandbox_image(body.task_data)
            if new:
                await self._build_in_turn(request, body)
            else:
                self._seed_episode(request, body)  # answers as the first seed did: nothing is built
            episode = self._episodes[session_id]
            if new:
                try:
                    await self._lend_sandbox(session_id, episode, image, task_id=body.task_id)
                except BaseException:
                    del self._episodes[session_id]
                    self._discard(session_id)
                    raise
        return ResourcesSeedSessionResponse(resources_session_id=session_id, sandbox_access=episode.access)

    def _admit(self) -> None:
        if self._shutting_down:  # shutdown stops the sandboxes it finds, and would not find this one
            raise HTTPException(status_code=503, detail="The server is shutting down")

    async def _build_in_turn(self, request: Request, body: ResourcesSeedSessionRequest) -> None:
        """Seed a new episode in its turn. A seed builds its world synchronously (a second or two of CPU for a large
        one) on the event loop that serves every episode, and one iteration of the loop runs every task that is ready:
        seeds left to themselves run back to back, while what arrived meanwhile, a harness's MCP initialize among it,
        waits for all of them. So seeds take turns: one at a time, each after the loop has had SEED_TURN_YIELD_S for
        what is ready, which lets a request in flight finish first. The turn is taken before that yield, so no other
        seed starts in it. A seed is admitted again when its turn comes, as shutdown may have begun while it waited."""
        ahead = self._seeds_queued
        self._seeds_queued += 1
        arrived = time.monotonic()
        try:
            async with self._seeding:
                self._admit()
                await asyncio.sleep(SEED_TURN_YIELD_S)
                turn = time.monotonic()
                self._seed_episode(request, body)
                built = time.monotonic()
        finally:
            self._seeds_queued -= 1
        SEED_LOGGER.info(
            "Seed of OfficeEnv session %s: %d ahead, waited %.2f s, built in %.2f s",
            body.resources_session_id,
            ahead,
            turn - arrived,
            built - turn,
        )

    def _workdir(self) -> str:
        return self.config.sandbox_config.workdir.rstrip("/")

    def _guide_path(self) -> str:
        return f"{self._workdir()}/{GUIDE_NAME}"

    def _workspace(self) -> Workspace:
        """Workspace mode as every sandboxed episode runs it (``EpisodeService.enable_workspace``): its files in the
        sandbox's workdir, text inline up to ``inline_file_bytes`` (UNLIMITED_INLINE_FILE_BYTES for None), a refused
        upload naming the file API's command for its target (UPLOAD_HINT), and this server's caps on the files one
        call reads, which each call's ReadPlan, and OfficeEnv's refusals, then carry."""
        limit = self.config.sandbox_config.inline_file_bytes
        return Workspace(
            folder=self._workdir(),
            inline_bytes=UNLIMITED_INLINE_FILE_BYTES if limit is None else limit,
            upload_hint=UPLOAD_HINT,
            max_files=MAX_ATTACHMENTS,
            max_total_bytes=MAX_ATTACHMENT_BYTES,
            max_file_bytes=MAX_UPLOAD_BYTES,
        )

    async def _lend_sandbox(
        self, session_id: str, episode: _SandboxEpisode, image: SandboxImage, *, task_id: TaskId
    ) -> None:
        """Lend the episode a sandbox started from ``image`` (``LentSandboxes.lend``) with the workspace guide, for a
        `cli` episode its apps' commands, and the episode's API (its URL and token) in its environment and at
        CREDENTIALS_PATH. Its start runs to its end once begun, and a seed cancelled meanwhile lends nothing and stops
        what it started (LentSandboxes). Once shutdown has begun, a seed lends nothing (503), though it waited in its
        turn or built its world meanwhile: LentSandboxes refuses once its release has begun, and none is made after
        (``_lending``)."""
        config = self.config.sandbox_config
        named_configs = get_global_config_dict()
        token = token_urlsafe(32)
        files_api_url = self._files_api_url()
        guide = workspace_guide(
            self._workspace().inline_bytes,
            episode.apps,
            exposure=episode.exposure,
            prefix=config.mcp_tool_prefix,
            tools=tuple(image.provides),
            offline=config.block_internet,
            docs_tool=episode.docs_tool,
            # the episode's file apps, or for an episode that connects none, every app the file API serves
            file_apps=self.service.file_apps(session_id) or self.service.file_apps(),
        )
        files = {self._guide_path(): guide, CREDENTIALS_PATH: api_credentials(files_api_url, token)}
        if episode.exposure == "cli":
            files[APP_COMMAND_PATH] = APP_COMMAND_SOURCE
        if config.host_relay:
            files[RELAY_PATH] = RELAY_SOURCE
        needs_setup = episode.exposure == "cli" or config.host_relay
        starting = time.monotonic()
        access = await self._lending().lend(
            session_id,
            task_id,
            files=files,
            image=image.image,
            env={API_URL_ENV: files_api_url, API_TOKEN_ENV: token},
            setup=(lambda box: self._prepare_sandbox(box, episode, named_configs)) if needs_setup else None,
        )
        sandbox = self._lending().get(session_id)
        episode.token, episode.sandbox, episode.access = token, sandbox, access
        self._tokens[token] = session_id
        # The harness's and the provider's logs name the sandbox, this server's the session: this line joins them.
        SEED_LOGGER.info(
            "Sandbox %s of OfficeEnv session %s (task %s) started in %.2f s",
            sandbox.sandbox_id,
            session_id,
            task_id.task_id,
            time.monotonic() - starting,
        )

    def _files_api_url(self) -> str:
        """The URL a sandbox reaches this server's file API at, without a final slash."""
        return (self.config.files_api_url or f"http://{self.config.host}:{self.config.port}").rstrip("/")

    def _lending(self) -> LentSandboxes:
        """The sandboxes lent to this server's episodes (Gym's ``LentSandboxes``), made at the first lend. With
        ``block_internet`` each starts with an egress policy that allows only this server and the model servers; a
        provider without network policies, or a server no sandbox can reach, fails the lend, and a later one tries
        again."""
        if self._lent is None:
            self._admit()  # no lending begins once shutdown has: it would not stop what it lent
            config = self.config.sandbox_config
            if config.block_internet:
                named_configs = get_global_config_dict()
                provider = next(iter(resolve_provider_config(self.config.sandbox_provider, named_configs)))
                own_urls = [self._files_api_url(), f"http://{self.config.host}:{self.config.port}"]
                policy = restricted_network_policy(provider, egress_urls(own_urls, named_configs))
                config = config.model_copy(
                    update={"provider_options": {**config.provider_options, "network_policy": policy}}
                )
            self._lent = LentSandboxes(self.config.sandbox_provider, config, image=None, workdir=config.workdir)
        return self._lent

    async def _prepare_sandbox(self, sandbox: AsyncSandbox, episode: _SandboxEpisode, named_configs: Mapping) -> None:
        """Make a started sandbox ready to lend: a `cli` episode's apps become commands, then the host relay starts."""
        config = self.config.sandbox_config
        if episode.exposure == "cli":
            await run_setup(sandbox, link_app_commands(episode.apps))
        if config.host_relay:
            await start_relay(sandbox, config.host_relay, relay_ports(named_configs, own_port=self.config.port))

    async def _release(self, session_id: str, episode: _SandboxEpisode | None) -> None:
        """Revoke the episode's file API token and stop its sandbox, or the one a failed seed of the session left
        (``LentSandboxes.release``, which goes on even while the request that asked for it is cancelled). A failed stop
        raises, and the session's next release tries it again."""
        if episode is not None:
            self._tokens.pop(episode.token, None)
        if self._lent is not None:
            await self._lent.release(session_id)
        if episode is not None:
            episode.sandbox = None

    async def shutdown(self) -> None:
        """Stop every sandbox within the time Gym's run command leaves its servers once it signals them to stop
        (``server_shutdown_timeout_seconds``, then it kills them): a sandbox left running stays on the cell until its
        TTL. ``LentSandboxes.release_all`` stops the lent ones and those failed seeds left, all at once, and those still
        starting once their starts return; from then on no seed starts one (503). What may still run at the deadline is
        logged for a reaper."""
        self._shutting_down = True
        if self._lent is None:
            return
        grace = get_global_config_dict().get(SERVER_SHUTDOWN_TIMEOUT_SECONDS_KEY_NAME)
        await self._lent.release_all(timeout_s=max(0.5, (1.0 if grace is None else float(grace)) - SHUTDOWN_MARGIN_S))

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest, _gym: None = Depends(gym_caller)
    ) -> ResourcesCloseSessionResponse:
        """Stop the episode's sandbox, or the one its failed seed left, then release its world as the OfficeEnv server
        does; a retried close, or the close of a session that was never seeded, succeeds."""
        self._require_gym_caller(request)  # a call made without FastAPI
        # A seed still starting the episode's sandbox holds the lock, so the close stops that sandbox too.
        async with self._locks.setdefault(body.resources_session_id, asyncio.Lock()):
            episode = self._episodes.get(body.resources_session_id)
            if episode is None or episode.episode_id == body.episode_id:
                # raises before the episode is forgotten, so a retried close stops it
                await self._release(body.resources_session_id, episode)
            return self._close(body)

    async def _stop_before_grading(self, session_id: str) -> None:
        """Close the episode's tools and file API and stop its sandbox before it is graded, so a process the harness
        left running in the sandbox cannot change what is graded. A failed stop is logged and grading goes on; the
        episode keeps the sandbox, so close stops it."""
        episode = self._episodes.get(session_id)
        if episode is None:
            return
        episode.grading = True
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            try:
                await self._release(session_id, episode)
            except Exception:
                LOGGER.exception("Could not stop the sandbox of OfficeEnv session %s before grading", session_id)

    async def retire_session_state(self, session_id: str) -> None:
        # A seed still starting the episode's sandbox holds the lock, so retiring stops that sandbox too. The OfficeEnv
        # server's retire then forgets the episode, since Gym never closes a retired session.
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            await self._release(session_id, self._episodes.get(session_id))
        await super().retire_session_state(session_id)

    async def tool_call(self, request: Request, tool_name: str) -> Response:
        # app-level errors are tool outputs the model sees (200); only a body that is no JSON
        # object (400) and a call without a session (404) are integration failures
        self._require_gym_caller(request)
        session_id = request.session[SESSION_ID_KEY]
        if self.service.store.get(session_id) is None:
            raise ResourcesSessionLostError("This session has ended.")
        episode = self._episodes.get(session_id)
        self._require_open(episode)
        body = await request.body()
        reply = await self._episode_call(session_id, episode, tool_name, body)
        text = reply.text()
        if episode is not None and episode.exposure == "api_search" and reply.status == 200:
            # the endpoints the finding tools find are called over HTTP
            text = with_http_calls(tool_name, text, docs_tool=episode.docs_tool)
        return Response(content=text, media_type="application/json", status_code=reply.status)

    @staticmethod
    def _require_open(episode: _SandboxEpisode | None) -> None:
        """Refuse a call of an episode whose verification has started, from which point its tools are closed."""
        if episode is not None and episode.grading:
            # neutral words: a process the harness left running in the sandbox may read them
            raise HTTPException(status_code=409, detail="The task has ended: its tools are closed")

    async def _episode_call(
        self,
        session_id: str,
        episode: _SandboxEpisode | None,
        tool_name: str,
        body: bytes | str,
        run: Callable[[Mapping[str, bytes | Unreadable]], Any] | None = None,
    ) -> Any:
        """Call ``tool_name`` with ``body`` (the service's call, or ``run`` given the files read for the call) under
        the episode's lock, with the workspace files it names read from the sandbox before it (``_read_workspace``,
        and kept in the trace store: ``_keep_in_trace``) and the files it saved in the workspace (its reply's
        ``workspace_writes``) written there after it (``_save_downloads``). Verification stops the sandbox under the
        same lock: a call under way when verification begins finishes first, and one that comes later is refused. A
        call that raises writes nothing. The task state the call leaves is published once the whole call is over, and
        with ``close_tools_on_task_end`` a call after a published end is refused, as on the OfficeEnv server."""
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            try:
                self._require_open(episode)
                self._refuse_after_task_end(session_id)
                reads = await self._read_workspace(session_id, episode, tool_name, body)
                self._require_open(episode)  # verification may have begun while the call's files were read
                self._keep_in_trace(session_id, episode, reads.values())
                if run is not None:
                    reply = run(reads)
                elif reads:
                    reply = self.service.call(session_id, tool_name, body, workspace_reads=reads)
                else:
                    reply = self.service.call(session_id, tool_name, body)
                await self._save_downloads(session_id, episode, reply.workspace_writes)
            finally:
                self._publish_task_state(session_id)
        return reply

    async def _read_workspace(
        self, session_id: str, episode: _SandboxEpisode | None, tool_name: str, body: bytes | str
    ) -> dict[str, bytes | Unreadable]:
        """The workspace files a call names (``EpisodeService.workspace_reads``: a message's attachments, an upload's
        content_path), read from the episode's sandbox before the call runs: path -> its bytes, or why there are none.
        The caps are the call's ReadPlan's, which hold this server's own (the episode's Workspace: MAX_ATTACHMENTS
        files, MAX_UPLOAD_BYTES each, MAX_ATTACHMENT_BYTES together) and the op's where stricter; a plan without one
        still meets this server's. Each path is checked in the sandbox first (``sandbox_file_sizes``): only a regular
        file within the cap is downloaded, and nothing when the call names more files, or more bytes together, than
        the caps allow. The copy is checked again (``copied_file``: it may have changed since), and a download that
        fails reads nothing. A file gone since the check is missing, as the model left it; any other failed download,
        or a check that does not run, is the provider's, and fails the call (``_provider_failed``)."""
        if episode is None or episode.sandbox is None:
            return {}
        plan = self.service.workspace_reads(session_id, tool_name, body)
        paths = list(plan.paths)
        if not paths:
            return {}
        most_files = _stricter(plan.max_files, MAX_ATTACHMENTS)
        most_each = _stricter(plan.max_file_bytes, MAX_UPLOAD_BYTES)
        most_bytes = _stricter(plan.max_total_bytes, MAX_ATTACHMENT_BYTES)
        if len(paths) > most_files:
            return dict.fromkeys(paths, Unreadable("too_many"))
        sizes = await self._sandbox_file_sizes(session_id, episode, paths)
        read: dict[str, bytes | Unreadable] = {}
        wanted: list[str] = []
        for path in paths:
            size = sizes.get(path)
            if size is None:
                read[path] = Unreadable("missing")
            elif size > most_each:
                read[path] = Unreadable("too_large")
            else:
                wanted.append(path)
        if sum(sizes[path] or 0 for path in wanted) > most_bytes:
            return dict.fromkeys(paths, Unreadable("too_much"))
        total = 0
        for path in wanted:
            with tempfile.TemporaryDirectory(prefix="officeenv-attachment-") as folder:
                local = Path(folder) / "file"
                try:
                    await episode.sandbox.download(path, local)
                except Exception as exc:  # nothing partial is read
                    if (await self._sandbox_file_sizes(session_id, episode, [path]))[path] is not None:
                        raise self._provider_failed(session_id, episode, f"read {path}", exc) from exc
                    LOGGER.info("No file at %s in OfficeEnv session %s: %s", path, session_id, exc)
                    read[path] = Unreadable("missing")  # gone since the check
                    continue
                read[path] = copied_file(local, most_each)
            total += len(read[path]) if isinstance(read[path], bytes) else 0
            if total > most_bytes:  # a file grew since the check
                return dict.fromkeys(paths, Unreadable("too_much"))
        return read

    async def _sandbox_file_sizes(
        self, session_id: str, episode: _SandboxEpisode, paths: list[str]
    ) -> dict[str, int | None]:
        """``sandbox_file_sizes`` in the episode's sandbox: a check that does not run fails the call."""
        try:
            return await sandbox_file_sizes(episode.sandbox, paths)
        except Exception as exc:
            raise self._provider_failed(session_id, episode, "check the files", exc) from exc

    async def verify(
        self, request: Request, body: OfficeEnvVerifyRequest, _gym: None = Depends(gym_caller)
    ) -> OfficeEnvVerifyResponse | OfficeEnvLengthPenaltyVerifyResponse:
        self._require_gym_caller(request)  # a call made without FastAPI
        return await super().verify(request, body)

    async def _verification(self, session_id: str) -> dict[str, Any]:
        """The OfficeEnv server's verification, once the episode's sandbox has stopped (``_stop_before_grading``),
        with what the episode sent beside MCP and moved through its workspace (``sandbox_*``), and masked when the
        sandbox provider failed one of its calls."""
        await self._stop_before_grading(session_id)
        episode = self._episodes.get(session_id)
        # what the episode moved through its workspace instead of the conversation, read before verification releases
        # the session (its trace, officeenv_workspace, comes with the result)
        moved = self.service.workspace_stats(session_id) if episode is not None else {}
        result = await super()._verification(session_id)
        if episode is not None:  # what it sent beside MCP, and moved through its workspace
            counted = {f"sandbox_{name}": float(moved.get(name, 0)) for name in WORKSPACE_STATS}
            result = result | episode.usage() | counted
            if self.config.workspace_trace_store is not None:  # a trace that lacks these files does not replay
                result = result | {"sandbox_trace_failures": float(episode.trace_failures)}
            if episode.provider_failure is not None:  # the reward would measure the sandbox, not the policy
                result = result | {
                    "mask_sample": True,
                    "failure_kind": PROVIDER_UNAVAILABLE,
                    "failure_reason": episode.provider_failure,
                }
        return result

    def mcp_tools(self, harvested: list["MCPTool"], catchall: Any | None) -> list["MCPTool"] | None:
        """Every tool of the ``tools`` exposure as workspace mode describes it (``_workspace``), and the finding tools
        of ``api_search`` episodes, the documentation tool under each name an exposure may give it, dispatched through
        the tool route; no other route is a tool. An episode's grant admits only its own."""
        if catchall is None:
            return []
        catalog = tools_exposure_catalog(self._workspace())
        tools = [catchall.tool(name, tool["parameters"], tool["description"]) for name, tool in catalog.items()]
        for name, schema, description in SEARCH_TOOLS:
            tools.append(catchall.tool(name, schema, description.format(guide=self._guide_path())))
        return tools

    # The file API: each request names its episode by the episode's bearer token.

    def _file_session(self, request: Request) -> str:
        scheme, _, token = (request.headers.get("authorization") or "").partition(" ")
        session_id = self._tokens.get(token.strip()) if scheme.lower() == "bearer" else None
        if session_id is None:
            raise HTTPException(status_code=401, detail="Missing or invalid bearer token")
        return session_id

    def _refuse_once_closed(self, session_id: str) -> None:
        """Refuse a file API call once the episode's files are closed, as its tools are: when its verification has
        begun (the token is revoked then, but an upload it admitted earlier may still be streaming in, and must not
        change what is graded), and with ``close_tools_on_task_end`` once a call has returned with its task ended."""
        episode = self._episodes.get(session_id)
        grading = episode is not None and episode.grading
        if grading or (self.config.close_tools_on_task_end and self._task_ends.get(session_id, False)):
            raise HTTPException(status_code=409, detail="The task has ended: its files are closed")

    def _file_answer(self, session_id: str, call: Callable[[], Any]) -> Any:
        """What a file API call (OfficeEnv's file call, the agent's own action) answers, or the app's error with its
        HTTP status and body: a file it does not have, a name its folder holds, an app it has no file API for, a fault
        the task injects. A file call runs the app's op through the same pipeline as a tool call (its injections among
        it), so it may end the task too: the task state it leaves is published once it is over, whatever it answers,
        as a tool call's is (``/task_state``, and ``close_tools_on_task_end`` for the calls after it)."""
        try:
            return call()
        except FileCallError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.body) from None
        finally:
            self._publish_task_state(session_id)

    def _keep_in_trace(
        self, session_id: str, episode: _SandboxEpisode | None, values: Iterable[bytes | Unreadable]
    ) -> None:
        """Keep the bytes this server hands an episode (the files a call read, the content a file API call sends) in
        the workspace trace store, by SHA-256, so the episode's trace (officeenv_workspace) replays without a sandbox.
        The store is a record kept beside the episode: a write that fails (a full disk, a store gone) is logged and
        counted (``sandbox_trace_failures``), and the call goes on as it would without a store, so the reward still
        measures the policy."""
        store = self.config.workspace_trace_store
        if store is None:
            return
        for value in values:
            if not isinstance(value, bytes):  # a file that could not be read hands nothing
                continue
            try:
                write_artifact(store, value)
            except Exception as exc:
                LOGGER.error(
                    "Could not keep %d bytes of OfficeEnv session %s in the workspace trace store: %s: %s",
                    len(value),
                    session_id,
                    type(exc).__name__,
                    exc,
                )
                if episode is not None:
                    episode.trace_failures += 1

    @staticmethod
    def _provider_failed(session_id: str, episode: _SandboxEpisode, doing: str, exc: BaseException) -> HTTPException:
        """The answer (502) to a call whose sandbox operation ``doing`` the provider failed: it raised, or could not
        run its command, whatever the workspace holds. The model reads none of the provider's words: they go to the
        server's log and the episode's failure_reason. The episode's verification is masked as provider_unavailable,
        since its reward would measure the sandbox, not the policy."""
        LOGGER.error("Could not %s in the sandbox of OfficeEnv session %s", doing, session_id, exc_info=exc)
        if episode.provider_failure is None:
            episode.provider_failure = f"Could not {doing} in the sandbox: {type(exc).__name__}: {exc}"[:500]
        return HTTPException(status_code=502, detail=f"Could not {doing} in the workspace")

    async def _save_downloads(
        self, session_id: str, episode: _SandboxEpisode | None, writes: Sequence[WorkspaceWrite]
    ) -> None:
        """Write into the episode's sandbox the files a call saved in its workspace (its reply's ``workspace_writes``),
        before the route answers that they are there, as mode 644, and count those that landed. A file that cannot be
        written fails the call (502): because of what the workspace holds, or because the provider failed
        (``_provider_failed``)."""
        if episode is None or not writes:
            return
        for write in writes:
            target, data = write.path, write.data
            try:
                # Made for every save, even into a folder made before: the model may have removed it, or put a file
                # in its place, and the provider's copy would then fail as if the provider had.
                folder = posixpath.dirname(target)
                made = await episode.sandbox.exec(f"mkdir -p {shlex.quote(folder)}", cwd="/", timeout_s=60)
                if made.error_type is not None:
                    raise RuntimeError(f"mkdir: {made.error_type} {(made.stderr or '')[-300:]}")
                if made.return_code != 0:  # the workspace is in the way: a file where a folder goes
                    raise HTTPException(
                        status_code=502,
                        detail=f"Could not save {target} in the workspace: its folder could not be made",
                    )
                # a folder at the file's path would take the copy inside it, and savedTo would name the folder
                free = await episode.sandbox.exec(f"test ! -d {shlex.quote(target)}", cwd="/", timeout_s=60)
                if free.error_type is not None:
                    raise RuntimeError(f"test: {free.error_type} {(free.stderr or '')[-300:]}")
                if free.return_code != 0:
                    raise HTTPException(
                        status_code=502, detail=f"Could not save {target} in the workspace: a folder is there"
                    )
                with tempfile.TemporaryDirectory() as staging:  # named as the file, wherever a copy lands
                    staged = Path(staging) / posixpath.basename(target)
                    staged.write_bytes(data)
                    os.chmod(staged, 0o644)  # readable whoever the agent runs as (a copy may keep the mode)
                    await episode.sandbox.upload(str(staged), target)
            except HTTPException:
                raise
            except Exception as exc:
                raise self._provider_failed(session_id, episode, f"save {target}", exc) from exc
            episode.files_saved += 1
            episode.bytes_saved += len(data)
        # An upload need not keep the staged file's mode (OpenSandbox's writes 755): a downloaded file is data,
        # readable by anyone and executable by no one, as Docker's copy leaves it.
        paths = " ".join(shlex.quote(write.path) for write in writes)
        try:
            done = await episode.sandbox.exec(f"chmod 644 {paths}", cwd="/", timeout_s=60)
            problem = None if done.error_type is None and done.return_code == 0 else (done.stderr or "")[-300:]
        except Exception as exc:  # the files are there: their mode is no reason to fail the call
            problem = f"{type(exc).__name__}: {exc}"
        if problem is not None:
            LOGGER.warning("Could not chmod %s in the sandbox of OfficeEnv session %s: %s", paths, session_id, problem)

    @staticmethod
    async def _bounded_body(request: Request) -> bytes:
        """A request's body, read in chunks and refused as soon as it passes the upload limit."""
        data = bytearray()
        async for chunk in request.stream():
            data += chunk
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"Request bodies are limited to {MAX_UPLOAD_BYTES} bytes")
        return bytes(data)

    @classmethod
    async def _content_args(cls, request: Request) -> bytes:
        """The content a file API upload sends: the request's body, refused as soon as it passes the upload limit.
        The file calls take it as bytes, whatever the app's own content parameters."""
        return await cls._bounded_body(request)

    async def download_file(self, request: Request, app_name: str, file_id: str) -> Response:
        """A file's bytes, under the media type its app gives it, or for an app that gives none (Box), text/plain for
        UTF-8 text and application/octet-stream for any other bytes."""
        session_id = self._file_session(request)
        self._refuse_once_closed(session_id)
        downloaded = self._file_answer(session_id, lambda: self.service.file_download(session_id, app_name, file_id))
        return Response(content=downloaded.data, media_type=downloaded.mime_type or media_type_of(downloaded.data))

    async def upload_file(
        self, request: Request, app: str, name: str, parent: str | None = None, mime_type: str | None = None
    ) -> JSONResponse:
        session_id = self._file_session(request)
        self._refuse_once_closed(session_id)
        data = await self._content_args(request)
        self._refuse_once_closed(session_id)  # verification may have begun while the body streamed in
        self._keep_in_trace(session_id, self._episodes.get(session_id), [data])
        created = self._file_answer(
            session_id,
            lambda: self.service.file_create(session_id, app, name, data, parent=parent, mime_type=mime_type),
        )
        return JSONResponse(created)

    async def replace_file(
        self, request: Request, app_name: str, file_id: str, mime_type: str | None = None
    ) -> JSONResponse:
        session_id = self._file_session(request)
        self._refuse_once_closed(session_id)
        data = await self._content_args(request)
        self._refuse_once_closed(session_id)  # verification may have begun while the body streamed in
        self._keep_in_trace(session_id, self._episodes.get(session_id), [data])
        replaced = self._file_answer(
            session_id, lambda: self.service.file_replace(session_id, app_name, file_id, data, mime_type=mime_type)
        )
        return JSONResponse(replaced)

    # The `cli` and `api_search` exposures: each request names its episode by the episode's bearer
    # token, as the file API's do.

    def _exposure_episode(self, request: Request, exposure: str) -> tuple[str, _SandboxEpisode]:
        """The bearer's episode, if its apps are reached through ``exposure``; to any other episode the route does
        not exist. An episode being verified has its tools closed."""
        session_id = self._file_session(request)
        episode = self._episodes.get(session_id)
        if episode is None or episode.exposure != exposure:
            raise HTTPException(status_code=404, detail="Not Found")
        self._require_open(episode)
        return session_id, episode

    async def run_command_line(self, request: Request) -> Response:
        """A command an app runs in a `cli` episode's sandbox (sandbox/app_command.sh): its arguments, the app's
        name first, NUL-separated. They run as OfficeEnv's CLI exposure reads a command line; the answer is the
        command's output as text, stdout for exit 0 and stderr otherwise, with the exit code in X-Exit-Code.

        The commands are the way in: a call that does not come from one (no CLI_USER_AGENT naming the app it runs,
        as a request made by hand after reading the command lacks) is refused with a pointer to the command, and
        counted. A model that copies the User-Agent has rebuilt the command, and gains nothing by it: this route
        takes exactly the command line the command does."""
        session_id, episode = self._exposure_episode(request, "cli")
        try:
            argv = (await self._bounded_body(request)).decode("utf-8").split("\0")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Arguments must be UTF-8") from None
        if argv and not argv[-1]:
            argv.pop()  # the terminator after the last argument
        if not argv or request.headers.get("user-agent") != CLI_USER_AGENT.format(app=argv[0]):
            episode.commands_refused += 1
            app = argv[0] if argv and argv[0] in episode.apps else "<app>"
            text = (
                f"error: run the app's command in the shell: {app} <command> [--parameter=value ...] ({app} --help)\n"
            )
            return Response(content=text, media_type="text/plain; charset=utf-8", status_code=403)
        episode.commands += 1
        body = json.dumps({"command": shlex.join(argv)})
        reply = await self._episode_call(session_id, episode, episode.command_tool, body)
        try:
            envelope = json.loads(reply.text())
        except ValueError:
            envelope = None
        if isinstance(envelope, dict) and {"exit_code", "stdout", "stderr"} <= set(envelope):
            code = int(envelope["exit_code"])
            text = envelope["stdout"] if code == 0 else envelope["stderr"]
        else:  # no command answer (an op that crashed, a session gone): its body is the command's error
            code, text = 1, reply.text()
        if text and not text.endswith("\n"):
            text += "\n"
        return Response(
            content=text,
            media_type="text/plain; charset=utf-8",
            status_code=reply.status,
            headers={"X-Exit-Code": str(code)},
        )

    async def call_endpoint(self, request: Request) -> Response:
        """An endpoint an `api_search` episode calls with curl, ``/v1/<app>/<op>``: the body is the op's parameters
        as a JSON object. It runs as OfficeEnv's api_call does and answers with the app's result, or with its error
        under the status the app gives it, as a vendor's API would: a 404, a 429 with its retry hint, or below 400
        where the vendor answers an error so (Snipe-IT's 200, Gorgias's 301; ``Reply.is_error`` tells such an answer
        from a result). The status is the call's own, never this server's."""
        app_name, op = request.path_params["app_name"], request.path_params["op"]
        session_id, episode = self._exposure_episode(request, "api_search")
        try:
            params = parse_arguments(await self._bounded_body(request))
            if not isinstance(params, dict):
                raise ValueError("send the endpoint's parameters as a JSON object")
        except (ValueError, RecursionError) as exc:
            body = {"error": {"code": "invalid_arguments", "message": f"invalid JSON body: {exc}"}}
            return JSONResponse(body, status_code=400)
        episode.endpoint_calls += 1
        call = {"endpoint": f"{app_name}.{op}", "params": params}
        # The service's endpoint call, not its call, which answers every app error with 200 (a tool output the model
        # reads): an HTTP API answers with the status the app gives, and an op the session could not run raises.
        try:
            reply = await self._episode_call(
                session_id,
                episode,
                CALL_TOOL,
                json.dumps(call),
                lambda reads: self.service.endpoint_call(session_id, CALL_TOOL, call, workspace_reads=reads),
            )
        except HTTPException:  # the call refused, or the sandbox failed it, as any exposure answers
            raise
        except Exception:
            LOGGER.exception("The %s.%s endpoint failed in OfficeEnv session %s", app_name, op, session_id)
            # the details stay in the server's log: the model reads no exception text
            body = {"error": {"code": "internal_error", "message": "The request could not be completed"}}
            return JSONResponse(body, status_code=500)
        return Response(content=reply.text(), media_type="application/json", status_code=reply.status)


if __name__ == "__main__":
    OfficeEnvSandboxResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OfficeEnvSandboxResourcesServer.run_webserver()  # noqa: F401
