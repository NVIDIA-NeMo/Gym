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
with another tool-name separator) is refused, but for a sandboxed episode's surfaces below.

A row may name its TaskSpec's large strings by SHA-256 instead of carrying them (OfficeEnv's
artifact references, ``officeenv.artifacts``). The stores in ``artifacts_dirs`` (such as a
corpus's ``artifacts/`` folder beside its rows file), searched in order, go to the episode
service, which reads the references back at seed, each checked against its hash, and refuses a
row whose artifacts cannot be read (400).

With ``sandbox_provider`` set, each Environment Server episode also gets a sandbox from Gym's
sandbox providers, lent to the harness (``SandboxAccess``) and stopped before the episode is graded.
The sandbox gets the episode's file API (``/v1/files``) in ``COMPANY_API_URL`` and
``COMPANY_API_TOKEN`` (``README.md`` in the workspace documents it), so file content moves between
the workspace and the simulated file apps (Google Drive, Box, SharePoint, OneDrive) over HTTP
instead of through the model's context. Their own content tools keep it out too
(``sandbox_config.inline_file_bytes``): a download saves the file in the workspace
(``<app folder>/<file id>/<name>``, ``FILE_APPS``) and answers with its path, carrying the content
only for short text, and an upload that pastes binary content or long text is refused with the file
API command that sends the file instead. Gmail's attachments move through the workspace as well:
``get_attachment`` saves one there, and a message or draft attaches a workspace file by its path
(``{"path": ...}``), which the server reads from the sandbox before the call runs.

A sandboxed episode's row exposure decides how its harness reaches the apps (``SANDBOX_SURFACES``):
one MCP tool per op (``tools``); one command per app in the sandbox (``cli``: ``slack post-message
--channel=...``, which sends its arguments to ``/v1/cli``), with no MCP tools, so any harness with a
shell and curl runs it; or ``api_search`` and ``api_fetch`` over MCP, which find endpoints the
harness calls with curl at ``/v1/<app>/<op>``. The file API, the content kept out of the
conversation and the workspace guide serve every surface.

A row may say what its task needs in the sandbox beyond the shell, curl and Python with pandas and
openpyxl (``sandbox_needs``: ``documents``, ``office``, ``TOOL_SETS``). Its episode gets the first
configured image that provides them all (``sandbox_config.image`` and ``provides``, then
``sandbox_config.images``), whose tools the workspace guide names; a row no configured image serves
is refused before its world is built (400 ``unsupported_needs``).

What a sandbox reads of this server names a company's API, never the environment behind it or Gym:
the variables, routes, guide, answers and session cookie (``get_session_cookie_name``), and with the
config's ``mcp_server_name`` and ``mcp_bearer_token`` the MCP endpoint too. A sandbox can reach this
server, so sandbox mode guards it: seeding, closing, verification, the app catalog and the tool
routes answer callers other than this node and ``trusted_networks`` as routes that do not exist
(404), which leaves sandboxes the MCP endpoint, the file API and their surface's routes, and no
generated docs are served;
and session cookies and MCP tokens are signed with a per-process secret (``get_session_secret``)
instead of Gym's key, which anyone can derive from the server's name. With
``sandbox_config.block_internet`` a sandbox reaches nothing else: OpenSandbox's egress policy refuses
every outbound connection except to this server and the model servers.

For training, ``length_penalty`` (off by default) deducts AutomationBench's training length
penalty from the reward and reports the two terms as ``reward_components``, so a trainer can
schedule the penalty on its own (NeMo RL ``grpo.phasic_reward_components``).
"""

import asyncio
import base64
import binascii
import contextvars
import hashlib
import ipaddress
import json
import logging
import os
import posixpath
import re
import shlex
import socket
import stat
import tempfile
import weakref
from collections.abc import Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from secrets import token_urlsafe
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from officeenv.apps import available, get_app_class
from officeenv.artifacts import ArtifactStore
from officeenv.errors import InvalidRequest, NotFound
from officeenv.exposure import Catalog, ToolsExposure
from officeenv.exposure.base import Invocation
from officeenv.service import EpisodeService, Reply, parse_arguments
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import get_global_config_dict
from nemo_gym.core import PROVIDER_UNAVAILABLE, EpisodeId, NeMoGymResponse, TaskId
from nemo_gym.sandbox import AsyncSandbox, DirectSandboxConnection, SandboxAccess, SandboxConfig
from nemo_gym.sandbox.agent_tools import restricted_network_policy
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


# The most bytes of a text file a sandboxed episode's Drive answer carries inline (sandbox_config.inline_file_bytes)
DEFAULT_INLINE_FILE_BYTES = 8192

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
    """The sandbox lent to each Environment Server episode when ``sandbox_provider`` is set."""

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
        description="Keep the file apps' content (Google Drive, Box, SharePoint, OneDrive) out of the conversation: "
        "a download saves the file in the workspace under the app's folder and answers with its path, carrying the "
        "content only for a text file of at most this many bytes; an upload that pastes binary content, or text over "
        "this many bytes, is refused with the file API command that sends the file. Gmail's attachments too: "
        "get_attachment saves one in the workspace, and a message or draft attaches a workspace file by its path. "
        "None answers and takes content in full, as without a sandbox",
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

MAX_UPLOAD_BYTES = 20_000_000


@dataclass(frozen=True)
class FileApp:
    """An app whose files move between a sandbox's workspace and the app: the workspace folder its downloads land
    in, and its ops that answer a file's content (``downloads``) and that take content to store (``uploads``)."""

    label: str
    folder: str
    downloads: tuple[str, ...]
    uploads: tuple[str, ...]
    note: str  # the guide's line on the app's file API calls


FILE_APPS: dict[str, FileApp] = {
    "gdrive": FileApp(
        "Google Drive",
        "drive",
        downloads=("download_file", "export_file", "get_revision"),
        uploads=("create_file", "update_file"),
        note="parent optional; add &mime_type=<type> to the upload URL for another file type",
    ),
    "box": FileApp(
        "Box",
        "box",
        downloads=("get_file_content",),
        uploads=("upload_file", "upload_file_version"),
        note="parent optional (the root, 0); a name the folder already holds is refused",
    ),
    "sharepoint": FileApp(
        "SharePoint",
        "sharepoint",
        downloads=("get_item_content",),
        uploads=("upload_content",),
        note="parent required (a folder's id, or a library root's); a name the folder holds gets a new version",
    ),
    "onedrive": FileApp(
        "OneDrive",
        "onedrive",
        downloads=("get_content", "get_version_content"),
        uploads=("upload_file",),
        note="parent optional (your drive's root); a name the folder holds gets a new version",
    ),
}

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
SANDBOX_SURFACES = ("tools", "cli", "api_search")

SURFACE_GUIDES = {
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
tool, read its parameters with your {{prefix}}api_fetch tool (it gives the exact command), and call it:

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


GET_ATTACHMENT = {
    "tools": "{prefix}gmail_get_attachment",
    "cli": "gmail get-attachment",
    "api_search": "the gmail.get_attachment endpoint",
}

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

The file name's extension sets the file type (.csv, .xlsx, .pdf, ...).
{{notes}}
Find file and folder ids with {{find}} (list or search files). Uploads are limited to
{MAX_UPLOAD_BYTES // 1_000_000} MB. A request that fails returns the app's error as JSON.
"""

CONTENT_GUIDE = """Downloading a file with {find} saves it in this workspace (the answer's savedTo):

{folders}

The answer also carries the content only for a text file of at most {limit} bytes. An upload with
{find} takes pasted content only as text of at most {limit} bytes: send any other file
with the commands above.
"""

MAIL_GUIDE = """Mail attachments move through this workspace too. Getting one ({get}) saves it at
gmail/<message id>/<attachment id>/<name> (the answer's savedTo). A message or draft attaches a
workspace file by its path, as the attachment {{"path": "{workdir}/totals.xlsx"}} (named after the
file unless you give a filename); pasted attachment content is taken only as text of at most {limit}
bytes.
"""


def workspace_guide(
    content_limit: int | None = DEFAULT_INLINE_FILE_BYTES,
    apps: tuple[str, ...] = (),
    *,
    surface: str = "tools",
    prefix: str = "",
    workdir: str = "/workspace",
    tools: tuple[str, ...] = (),
    offline: bool = False,
) -> str:
    """The workspace's README.md for an episode that connects ``apps``: how its apps are reached (``surface``, and
    ``prefix``, how the harness names this server's MCP tools), how files move between the workspace and its file apps
    (those of ``apps`` in FILE_APPS, all of them for an episode that connects none), where downloads land when file
    content stays out of the conversation (``content_limit``, see ContentInWorkspace; None: it does not), and then,
    for an episode that connects Gmail, how its attachments move too. Last, for an image with more than the shell,
    curl and Python (``tools``, TOOL_SETS' names) or a sandbox without the internet (``offline``), what is installed,
    and whether nothing else can be."""
    intro = SURFACE_GUIDES[surface].format(apps=", ".join(apps), prefix=prefix)
    names = [app for app in FILE_APPS if app in apps] or list(FILE_APPS)
    shown = [FILE_APPS[app] for app in names]
    find = FIND_FILES[surface]
    files = FILES_GUIDE.format(
        codes=", ".join(f"{app} ({FILE_APPS[app].label})" for app in names),
        notes="".join(f"{spec.label}: {spec.note}.\n" for spec in shown),
        find=find,
    )
    if content_limit is not None:
        width = max(len(f"{spec.folder}/<file id>/<name>") for spec in shown) + 3
        folders = "\n".join(f"    {f'{spec.folder}/<file id>/<name>':<{width}}{spec.label}" for spec in shown)
        files += "\n" + CONTENT_GUIDE.format(find=find, folders=folders, limit=content_limit)
        if "gmail" in apps:
            get = GET_ATTACHMENT[surface].format(prefix=prefix)
            files += "\n" + MAIL_GUIDE.format(get=get, workdir=workdir, limit=content_limit)
    guide = f"# Company apps\n\n{intro}\n\n{CREDENTIALS_GUIDE}\n\n{files}"
    if tools or offline:
        installed = "\n".join(f"- {text}" for text in (INSTALLED, *(TOOL_SETS[name] for name in tools)))
        guide += f"\nInstalled here:\n\n{installed}\n"
        if offline:
            guide += "\nThe sandbox has no internet access, so nothing else can be installed.\n"
    return guide


# The file apps' ops that answer a file's content, and those that take content to store. In a sandboxed episode the
# content moves through the workspace instead (ContentInWorkspace): bytes in the conversation cost the model their
# tokens on every later request, and a model that copies them by hand gets them wrong.
CONTENT_DOWNLOADS = frozenset((app, op) for app, spec in FILE_APPS.items() for op in spec.downloads)
CONTENT_UPLOADS = frozenset((app, op) for app, spec in FILE_APPS.items() for op in spec.uploads)
# Gmail keeps attachments in its messages, with no file API: they move through the workspace too
# (ContentInWorkspace). get_attachment saves one there, at gmail/<message id>/<attachment id>/<name>, and a message
# or draft attaches a workspace file by its path ({"path": ...}), which the tool route reads from the sandbox before
# the call runs (OfficeEnvResourcesServer._read_attachments) and the call carries as the attachment's content.
MAIL_FOLDER = "gmail"
ATTACHMENT_DOWNLOADS = frozenset({("gmail", "get_attachment")})
ATTACHMENT_UPLOADS = frozenset({("gmail", "send_message"), ("gmail", "create_draft"), ("gmail", "update_draft")})
DOWNLOAD_FOLDERS = {app: spec.folder for app, spec in FILE_APPS.items()} | {"gmail": MAIL_FOLDER}
CONTENT_KEYS = frozenset({"content", "content_base64", "contentBase64"})  # an attachment's content, as Gmail takes it
UNREADABLE_PATHS = ("/dev/", "/proc/", "/sys/")  # no attachment is read from these
MAX_ATTACHMENTS = 20  # workspace files one message may attach
MAX_ATTACHMENT_BYTES = 25_000_000  # their bytes together, Gmail's own limit for a message
# Each attachment's path is checked in the sandbox before anything is copied out: stat prints its type and size without
# following a final link (one line per path, "missing" for nothing there), so only a regular file of a bounded size is
# ever downloaded.
STAT_SCRIPT = 'for p; do LC_ALL=C stat -c "%F|%s" -- "$p" 2>/dev/null || echo "missing|0"; done'
PREVIEW_CHARS = 2000  # the start of a text file too long to answer inline
EXPORT_EXTENSIONS = {
    "text/plain": ".txt",
    "text/html": ".html",
    "text/markdown": ".md",
    "text/csv": ".csv",
    "text/tab-separated-values": ".tsv",
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
}

_SAVED = (
    "the answer gives the file's path (savedTo) and size (savedBytes), and carries the text itself (content) only for "
    "a text file of at most {limit} bytes; a longer text file comes with its start (preview), a binary file with its "
    "path alone."
)
_PASTED = (
    "Content goes in this call only as text of at most {limit} bytes: send any other file from the workspace with "
    "the file API ({guide})."
)
# What the file apps' content ops do in a sandboxed episode (ContentInWorkspace), said in place of OfficeEnv's own
# descriptions, which tell of content carried in the answer or in the call (contentBase64, content_base64). Literal
# braces are doubled: the text is formatted with the limit and the guide's path.
CONTENT_DESCRIPTIONS: dict[tuple[str, str], str] = {
    ("gdrive", "download_file"): "Download a file's content (alt=media) into the workspace: " + _SAVED,
    ("gdrive", "export_file"): (
        "Export a Google Doc/Sheet/Slides as text/plain, text/csv, text/html, application/pdf… into the workspace: "
        "the answer gives the exported file's path (savedTo) and size (savedBytes), and carries the text itself "
        "(content) only when it is text of at most {limit} bytes."
    ),
    ("gdrive", "get_revision"): (
        "A revision's metadata, with its content saved in the workspace: the answer gives the file's path (savedTo) "
        "and size (savedBytes), and carries the text itself (content) only for a text file of at most {limit} bytes."
    ),
    ("gdrive", "create_file"): (
        "Create a file or, with the folder mimeType, a folder; you become its owner. Content goes in this call only "
        "as text of at most {limit} bytes: upload any other file from the workspace with the file API ({guide})."
    ),
    ("gdrive", "update_file"): (
        "Update a file's metadata (rename, describe, star, trash/restore), move it with add_parents/remove_parents, "
        "or upload new content: content goes in this call only as text of at most {limit} bytes; send any other file "
        "from the workspace with the file API ({guide})."
    ),
    ("box", "get_file_content"): (
        "Download a file's content (GET /files/{{id}}/content), optionally a given version, into the workspace: "
        + _SAVED
    ),
    ("box", "upload_file"): (
        "Upload a new file (POST /files/content) into parent_id ('0' = root); same name → 409 item_name_in_use. "
        "Returns {{total_count, entries: [file]}}. " + _PASTED
    ),
    ("box", "upload_file_version"): (
        "Upload a new version of a file (POST /files/{{id}}/content); version_number increases and file_version "
        "changes; optionally rename. " + _PASTED
    ),
    ("sharepoint", "get_item_content"): (
        "A file's content (GET /drives/{{drive-id}}/items/{{item-id}}/content), saved in the workspace: " + _SAVED
    ),
    ("sharepoint", "upload_content"): (
        "Upload a file's content by path, or by parent folder + name (PUT /drives/{{drive-id}}/root:/path:/content); "
        "an existing file gets a new version (conflict_behavior replace, the default), fails (fail) or the upload is "
        "renamed (rename). " + _PASTED
    ),
    ("onedrive", "get_content"): (
        "A file's content (GET /me/drive/items/{{item-id}}/content, or by path), saved in the workspace: " + _SAVED
    ),
    ("onedrive", "get_version_content"): (
        "The content of one version of a file (GET /me/drive/items/{{item-id}}/versions/{{version-id}}/content), "
        "saved in the workspace: " + _SAVED
    ),
    ("onedrive", "upload_file"): (
        "Upload a file's content (PUT /me/drive/root:/path:/content, PUT /me/drive/items/{{parent-id}}:/{{name}}:/"
        "content): by path, by parent folder + name, or item_id for an existing file. A file already there gets a new "
        "version (conflict_behavior replace, the default), or the call fails (fail) or the upload is renamed "
        "(rename). " + _PASTED
    ),
    ("gmail", "get_attachment"): (
        "Get an attachment of a message in your mailbox (or of your draft), saved in the workspace: " + _SAVED
    ),
}

_ATTACHED = (
    "{{path, filename?, mimeType?}} attaches a file from the workspace (its path, e.g. {workdir}/totals.xlsx; the "
    "filename defaults to the file's name), and {{filename, content, mimeType?}} text of at most {limit} bytes; any "
    "other file goes by its path. Unknown fields are refused."
)
_TEXT_PARAM = (
    f"Text of at most {{limit}} bytes; a longer or binary file goes from the workspace with the file API "
    f"({{workdir}}/{GUIDE_NAME})"
)
_BYTES_PARAM = (
    "Not taken here: text goes in content (at most {limit} bytes), and a binary file (an XLSX workbook, a PDF, an "
    f"image) goes from the workspace with the file API ({{workdir}}/{GUIDE_NAME}), never pasted here"
)
# Gmail's attachment parameters, and the file apps' upload content, as a sandboxed episode reads them
# (ContentInWorkspace), in place of OfficeEnv's, which take content or content_base64 in the call. Formatted with the
# limit and the workspace, as CONTENT_DESCRIPTIONS.
PARAM_DESCRIPTIONS: dict[tuple[str, str, str], str] = {
    ("gmail", "send_message", "attachments"): "Attachments: " + _ATTACHED,
    ("gmail", "create_draft", "attachments"): "Attachments: " + _ATTACHED,
    ("gmail", "update_draft", "attachments"): "New attachments (replaces): " + _ATTACHED,
    **{(app, op, "content"): _TEXT_PARAM for app, op in CONTENT_UPLOADS},
    **{(app, op, "content_base64"): _BYTES_PARAM for app, op in CONTENT_UPLOADS},
}


def content_description(app: str, op: str, description: str, limit: int | None, *, guide: str) -> str:
    """A tool's description as a sandboxed episode reads it: the file apps' content ops and Gmail's get_attachment
    say what they do there (CONTENT_DESCRIPTIONS, ``guide`` the workspace guide's path); every other op, and every
    op when content moves through the conversation (``limit`` None), keeps OfficeEnv's. The tool's app label
    (``[Google Drive] ``) stays."""
    text = CONTENT_DESCRIPTIONS.get((app, op))
    if text is None or limit is None:
        return description
    label = description[: description.index("] ") + 2] if description.startswith("[") and "] " in description else ""
    return label + text.format(limit=limit, guide=guide)


def content_schema(
    app: str, op: str, schema: Mapping[str, Any], limit: int | None, *, workdir: str = "/workspace"
) -> Mapping[str, Any]:
    """An op's parameter schema as a sandboxed episode reads it: Gmail's attachments and the uploads' content
    described as they move there (PARAM_DESCRIPTIONS, ``workdir`` the workspace); every other schema, and every one
    when content moves through the conversation (``limit`` None), is OfficeEnv's. The schema given is not changed."""
    changed = {param: text for (a, o, param), text in PARAM_DESCRIPTIONS.items() if (a, o) == (app, op)}
    properties = schema.get("properties")
    if limit is None or not changed or not isinstance(properties, Mapping):
        return schema
    properties = dict(properties)
    for param, text in changed.items():
        if isinstance(properties.get(param), Mapping):
            properties[param] = {**properties[param], "description": text.format(limit=limit, workdir=workdir)}
    return {**schema, "properties": properties}


def example_attachments(items: Any, *, workdir: str = "/workspace") -> Any:
    """An example's attachments as a sandboxed episode sends them: one that names a file without its content names
    the workspace file instead (``{"path": "<workdir>/<name>"}``)."""
    if not isinstance(items, list):
        return items
    out = []
    for item in items:
        name = item if isinstance(item, str) else None
        if isinstance(item, Mapping) and not (CONTENT_KEYS | {"path"}) & set(item):
            name = item.get("filename") or item.get("name")
        out.append({"path": posixpath.join(workdir, str(name).strip())} if name else item)
    return out


# A command's --help example of an attachment parameter, as OfficeEnv's command line quotes it
_CLI_ATTACHMENTS = re.compile(r"'--attachments=(\[[^']*\])'")


def _cli_example_attachments(match: re.Match, workdir: str) -> str:
    try:
        items = json.loads(match.group(1))
    except ValueError:
        return match.group(0)
    attached = example_attachments(items, workdir=workdir)
    return shlex.quote("--attachments=" + json.dumps(attached, separators=(",", ":")))


# --help as OfficeEnv's CLI prints it: an app's command list ("  <command>  <description>"), and a command's head line
# ("<app> <command>: <description>") and options ("  --<flag>=VALUE  <type> <description>[ [default: ...]]").
_HELP_COMMAND = re.compile(r"^(?P<lead>  (?P<command>[a-z0-9-]+) +)(?P<description>\S.*)$")
_HELP_HEAD = re.compile(r"^(?P<app>[a-z0-9_-]+) (?P<command>[a-z0-9-]+): (?P<description>.*)$")
_HELP_OPTION = re.compile(
    r"^(?P<lead>  --(?P<flag>[a-z0-9-]+)=VALUE +\S+ )(?P<description>.*?)(?P<default> \[default: .*\])?$"
)


def with_content_descriptions(
    text: str, limit: int | None, argv: Sequence[str], *, guide: str, workdir: str = "/workspace"
) -> str:
    """A command's --help as a sandboxed episode reads it: each line that describes an op moving content
    (CONTENT_DESCRIPTIONS) or one of its parameters (PARAM_DESCRIPTIONS) has its whole description replaced, keyed by
    app, op and parameter, and Gmail's examples attach a workspace file (``example_attachments``). Only help is
    rewritten (``argv`` names the app alone, or asks for --help): what an op answers is the app's data, left as it is."""
    if limit is None or not argv or not (len(argv) == 1 or {"--help", "-h"} & set(argv[1:])):
        return text
    app = argv[0]
    command = argv[1] if len(argv) > 1 and not argv[1].startswith("-") else None
    op = command.replace("-", "_") if command else None
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if op is None:  # the app's command list
            found = _HELP_COMMAND.match(line)
            key = (app, found["command"].replace("-", "_")) if found else None
            if key in CONTENT_DESCRIPTIONS:
                lines[i] = found["lead"] + content_description(*key, found["description"], limit, guide=guide)
        elif (found := _HELP_HEAD.match(line)) and (found["app"], found["command"]) == (app, command):
            if (app, op) in CONTENT_DESCRIPTIONS:
                lines[i] = f"{app} {command}: " + content_description(
                    app, op, found["description"], limit, guide=guide
                )
        elif (found := _HELP_OPTION.match(line)) and (app, op, found["flag"].replace("-", "_")) in PARAM_DESCRIPTIONS:
            described = PARAM_DESCRIPTIONS[(app, op, found["flag"].replace("-", "_"))]
            lines[i] = found["lead"] + described.format(limit=limit, workdir=workdir) + (found["default"] or "")
    text = "\n".join(lines)
    if app == "gmail" and op is not None:
        text = _CLI_ATTACHMENTS.sub(lambda match: _cli_example_attachments(match, workdir), text)
    return text


# An api_search episode finds endpoints over MCP and calls them over HTTP: the two finding tools, with the argument
# shapes of OfficeEnv's api_search exposure (a public contract). Descriptions are formatted with the guide's path.
SEARCH_TOOLS: tuple[tuple[str, dict[str, Any], str], ...] = (
    (
        "api_search",
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
    (
        "api_fetch",
        {
            "type": "object",
            "properties": {
                "endpoint": {
                    "type": "string",
                    "description": "An endpoint id from search results, e.g. 'slack.post_message'",
                }
            },
            "required": ["endpoint"],
        },
        "Fetch an endpoint's documentation: description, parameter schema, examples, and the curl command that "
        "calls it.",
    ),
)

CURL_CALL = (
    f'curl -sS -X POST "${API_URL_ENV}{{path}}" -H "Authorization: Bearer ${API_TOKEN_ENV}" '
    "-H 'Content-Type: application/json' -d {data}"
)


def endpoint_path(endpoint: str) -> str:
    """The HTTP path an api_search endpoint id (``app.op``) is called at."""
    app, _, op = endpoint.partition(".")
    return f"{API_PREFIX}/{app}/{op}"


def with_http_calls(
    tool_name: str, text: str, content_limit: int | None = None, *, guide: str, workdir: str = "/workspace"
) -> str:
    """api_search's results with each endpoint's HTTP path, and api_fetch's documentation with the curl command that
    calls the endpoint (its first example as the body): in a sandbox, endpoints are called over HTTP. An endpoint
    that moves file content is described as it behaves here (``content_description``), and so are Gmail's
    attachments, in the parameters and the examples (``content_schema``, ``example_attachments``)."""
    if tool_name not in ("api_search", "api_fetch"):
        return text
    try:
        body = json.loads(text)
    except ValueError:
        return text
    if not isinstance(body, dict) or "error" in body:
        return text
    if tool_name == "api_search":
        for result in body.get("results") or []:
            if isinstance(result, dict) and isinstance(result.get("endpoint"), str):
                result["path"] = endpoint_path(result["endpoint"])
                app, _, op = result["endpoint"].partition(".")
                if isinstance(result.get("description"), str):
                    result["description"] = content_description(
                        app, op, result["description"], content_limit, guide=guide
                    )
    elif isinstance(body.get("endpoint"), str):
        path = endpoint_path(body["endpoint"])
        app, _, op = body["endpoint"].partition(".")
        if content_limit is not None and isinstance(body.get("parameters"), dict):
            body["parameters"] = content_schema(app, op, body["parameters"], content_limit, workdir=workdir)
        if content_limit is not None and (app, op) in ATTACHMENT_UPLOADS and isinstance(body.get("examples"), list):
            body["examples"] = [
                {**example, "attachments": example_attachments(example["attachments"], workdir=workdir)}
                if isinstance(example, dict) and "attachments" in example
                else example
                for example in body["examples"]
            ]
        examples = body.get("examples") or []
        example = examples[0] if examples and isinstance(examples[0], dict) else {}
        data = shlex.quote(json.dumps(example, ensure_ascii=False))
        body["http"] = {"method": "POST", "path": path, "curl": CURL_CALL.format(path=path, data=data)}
        if isinstance(body.get("description"), str):
            body["description"] = content_description(app, op, body["description"], content_limit, guide=guide)
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"))


def _path_part(name: Any, fallback: str) -> str:
    """A file's name or id as one segment of a workspace path: '/' (a file name is no path) and NUL replaced, at
    most 200 bytes, the extension kept."""
    text = str(name if name is not None else "").replace("/", "_").replace("\0", "").strip()
    if text in ("", ".", ".."):
        text = fallback
    if len(text.encode("utf-8")) > 200:
        stem, ext = posixpath.splitext(text)
        ext = ext if len(ext.encode("utf-8")) <= 20 else ""
        text = stem.encode("utf-8")[: 200 - len(ext.encode("utf-8"))].decode("utf-8", "ignore") + ext
    return text


def _preview(text: str) -> str:
    """The start of a text file: at most PREVIEW_CHARS characters, ending at a line break when it holds one."""
    head = text[:PREVIEW_CHARS]
    if len(text) > PREVIEW_CHARS and "\n" in head:
        head = head[: head.rindex("\n") + 1]
    return head


def _version_folder(app: str, op: str, params: Mapping[str, Any], answer: Mapping[str, Any]) -> str | None:
    """The folder under a file's own that keeps the download of one of its revisions or versions, if it is one, or of
    a mail attachment, under its message's."""
    if (app, op) == ("gmail", "get_attachment"):  # a message may hold two attachments of one name
        return _path_part(answer.get("attachmentId") or params.get("attachment_id"), "attachment")
    if (app, op) == ("gdrive", "get_revision"):
        return f"revisions/{_path_part(answer.get('version'), 'revision')}"
    if (app, op) == ("box", "get_file_content") and params.get("version_id"):
        return f"versions/{_path_part(answer.get('version_number') or params['version_id'], 'version')}"
    if (app, op) == ("onedrive", "get_version_content"):
        return f"versions/{_path_part(answer.get('id') or params.get('version_id'), 'version')}"
    return None


def saved_download(
    app: str,
    op: str,
    params: Mapping[str, Any],
    answer: Mapping[str, Any],
    *,
    folder: str,
    limit: int,
    file_id: str,
    name: str | None,
) -> tuple[dict[str, Any], str, bytes] | None:
    """A file app's answer carrying a file's content (``app``'s ``op``), as a sandboxed episode gets it, with where
    the file is saved and its bytes; None for an answer without content. The answer gives the file's path in the
    workspace (``savedTo``, under ``folder``: ``<file id>/<name>``, a revision or version under its own folder) and
    its size (``savedBytes``) in place of the content, which stays only when it is text of at most ``limit`` bytes;
    longer text gets its start (``preview``), binary content nothing but the path."""
    encoded = answer.get("contentBase64", answer.get("content_base64"))  # Box's key is content_base64
    if isinstance(encoded, str):
        try:
            data, text = base64.b64decode(encoded, validate=True), None
        except (binascii.Error, ValueError):
            return None
    elif isinstance(answer.get("content"), str):
        text = answer["content"]
        data = text.encode("utf-8")
    else:
        return None
    base = f"{folder}/{_path_part(file_id, 'file')}"
    name = _path_part(name, _path_part(file_id, "file"))
    if (app, op) == (
        "gdrive",
        "export_file",
    ):  # a Docs editor file as another type: its name gets the type's extension
        extension = EXPORT_EXTENSIONS.get(str(answer.get("mimeType")), "")
        name = name if name.lower().endswith(extension) else name + extension
    version = _version_folder(app, op, params, answer)
    target = f"{base}/{version}/{name}" if version else f"{base}/{name}"
    out = {key: value for key, value in answer.items() if key not in ("content", "contentBase64", "content_base64")}
    out["savedTo"] = target
    out["savedBytes"] = len(data)
    if text is not None and len(data) <= limit:
        out["content"] = text
    else:
        held = "binary content" if text is None else f"{len(data)} bytes of text, over the {limit} shown inline"
        out["note"] = f"The content is not in this answer ({held}): the file is at savedTo. Work on it with code."
        if text is not None:
            out["preview"] = _preview(text)
    return out, target, data


def _rows(world: Any, app: str, collection: str) -> Mapping[str, dict[str, Any]]:
    """An app's rows of one collection in a session's world, or none when the app is not connected."""
    try:
        return world.collection(app, collection).rows
    except Exception:  # an app the episode does not connect has no collections
        return {}


def _named_drives(world: Any, app: str, drive_id: Any, site: Any = None) -> set[str] | None:
    """The drives a call names: ``drive_id``, an id or, as SharePoint takes it, a library's name (within ``site``,
    an id or a name, when given); None when the call names no drive."""
    if not drive_id:
        return None
    drives = _rows(world, app, "drives")
    if str(drive_id) in drives:
        return {str(drive_id)}
    wanted = str(drive_id).strip().lower()
    names = {"documents", "shared documents"} if wanted in ("documents", "shared documents") else {wanted}
    sites = _rows(world, app, "sites")

    def in_site(row: Mapping[str, Any]) -> bool:
        own = sites.get(row.get("siteId"), {})
        return not site or str(site).strip().lower() in {
            str(value).lower() for value in (row.get("siteId"), own.get("name"), own.get("displayName")) if value
        }

    return {rid for rid, row in drives.items() if str(row.get("name") or "").lower() in names and in_site(row)}


def drive_item_at(world: Any, app: str, path: Any, *, drive_id: Any = None, site: Any = None) -> dict[str, Any] | None:
    """The live SharePoint or OneDrive item at ``path`` (``/Folder/name``, ``/`` for a root), in the drive a call
    names (``drive_id``, ``site``) when it names one, or None when no single item is there."""
    rows = _rows(world, app, "driveItems")
    drives = _named_drives(world, app, drive_id, site)
    want = [part.lower() for part in str(path or "").strip("/").split("/") if part]
    found = []
    for row in rows.values():
        if row.get("deleted") or (drives is not None and row.get("driveId") not in drives):
            continue
        names, current = [], row
        while current is not None and current.get("parent") is not None:
            names.append(str(current.get("name") or "").lower())
            current = rows.get(current["parent"])
        if names[::-1] == want:
            found.append(row)
    return found[0] if len(found) == 1 else None


def download_identity(
    world: Any, app: str, op: str, params: Mapping[str, Any], answer: Mapping[str, Any]
) -> tuple[str, str | None]:
    """The id and name of the file a content download is of: the answer's own, but for a revision or version, whose
    answer names neither, the file's row in the session's world, and for a mail attachment its message."""
    if (app, op) == ("gmail", "get_attachment"):
        return str(params.get("message_id") or "message"), answer.get("filename")
    if (app, op) == ("gdrive", "get_revision"):
        file_id = str(params.get("file_id") or "")
        return file_id, _rows(world, "gdrive", "files").get(file_id, {}).get("name")
    if (app, op) == ("onedrive", "get_version_content"):
        row = _rows(world, "onedrive", "driveItems").get(str(params.get("item_id") or "")) or drive_item_at(
            world, "onedrive", params.get("path"), drive_id=params.get("drive_id")
        )
        if row is None:  # the version answered, so the item is there; a path it cannot place keeps its own name
            return "file", posixpath.basename(str(params.get("path") or "")) or None
        return str(row.get("id")), row.get("name")
    own = params.get("file_id") if app == "gdrive" else None
    return str(own or answer.get("id") or params.get("file_id") or params.get("item_id") or "file"), answer.get("name")


def _upload_target(world: Any, app: str, op: str, params: Mapping[str, Any]) -> tuple[str | None, dict[str, str]]:
    """Where a pasted upload's content goes, said as the file API says it: the file it replaces (an id, sent with
    PUT), or the query of a new file (POST: name, parent, and Drive's mime_type)."""
    if (app, op) in (("gdrive", "update_file"), ("box", "upload_file_version")):
        return str(params.get("file_id") or "<file_id>"), {}
    if (app, op) == ("onedrive", "upload_file") and params.get("item_id"):
        return str(params["item_id"]), {}
    path = str(params.get("path") or "")
    query = {"name": str(params.get("name") or posixpath.basename(path) or "<file_name>")}
    if app == "gdrive":
        parents = params.get("parents")
        if isinstance(parents, list) and parents:
            query["parent"] = str(parents[0])
        if params.get("mime_type"):
            query["mime_type"] = str(params["mime_type"])
    elif app == "box":
        if params.get("parent_id") not in (None, "", "0"):
            query["parent"] = str(params["parent_id"])
    elif params.get("parent_id"):
        query["parent"] = str(params["parent_id"])
    elif path.strip("/").count("/") or app == "sharepoint":  # a folder's path, or SharePoint, which needs a folder
        folder = drive_item_at(
            world,
            app,
            posixpath.dirname("/" + path.strip("/")),
            drive_id=params.get("drive_id"),
            site=params.get("site"),
        )
        query["parent"] = str(folder["id"]) if folder is not None else "<folder_id>"
    return None, query


def pasted_content_refusal(
    app: str, op: str, params: Mapping[str, Any], limit: int, *, guide: str, world: Any = None
) -> str | None:
    """Why a sandboxed episode's upload may not carry its content in the call (binary content, or text over
    ``limit`` bytes), with the file API command that sends the file instead; None when it may."""
    if params.get("content_base64"):
        held = "binary content"
    elif isinstance(params.get("content"), str) and len(params["content"].encode("utf-8")) > limit:
        held = f"text over {limit} bytes"
    else:
        return None
    send = f'-H "Authorization: Bearer ${API_TOKEN_ENV}" --data-binary @<local path>'
    file_id, query = _upload_target(world, app, op, params)
    if file_id is not None:
        command = f'curl -sf -X PUT {send} "${API_URL_ENV}{FILES_ROUTE}/{app}/{quote(file_id, safe="<>_")}"'
    else:
        args = "&".join(f"{key}={quote(value, safe='<>')}" for key, value in query.items())
        command = f'curl -sf -X POST {send} "${API_URL_ENV}{FILES_ROUTE}?app={app}&{args}"'
    after = (
        " (then update_file without content for any other change)" if (app, op) == ("gdrive", "update_file") else ""
    )
    return (
        f"File content is not pasted into a call in this sandbox ({held}). Write the file in the workspace and send "
        f"it with the file API ({guide}){after}:\n  {command}"
    )


def attachment_path(value: Any, workdir: str) -> str | None:
    """The sandbox path an attachment's ``path`` names, normalized (a relative one is in the workdir); None for one
    that names nothing a file can be read from (empty, or under /dev, /proc or /sys)."""
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    path = "/" + posixpath.normpath(text if text.startswith("/") else f"{workdir}/{text}").lstrip("/")
    return None if path.startswith(UNREADABLE_PATHS) else path


def attachment_candidate(item: Any) -> str | None:
    """What a mail attachment names as a file to read: its ``path``, or the attachment itself when it is a bare
    string (``--attachments=/workspace/totals.xlsx`` on a command line)."""
    if isinstance(item, Mapping):
        return item.get("path")
    return item if isinstance(item, str) and item.strip() else None


def named_file(item: str, read: Mapping[str, bytes | str], workdir: str) -> bool:
    """Whether a bare string attachment is to be taken as a file's path: one that writes a path (it holds a ``/``),
    or a name the workspace holds a file under."""
    if "/" in item:
        return True
    return isinstance(read.get(attachment_path(item, workdir) or ""), bytes)


def copied_file(local: Path, limit: int) -> bytes | str:
    """A file a sandbox provider copied to this host: its bytes when it is a regular file of at most ``limit``
    bytes, else why not: "too_large", or "missing" for nothing there, a folder, a device, a pipe or a link, whose
    copy would be read here, not in the sandbox (``docker cp`` copies a link as a link)."""
    try:
        info = local.lstat()
    except OSError:
        return "missing"
    if not stat.S_ISREG(info.st_mode):
        return "missing"
    return "too_large" if info.st_size > limit else local.read_bytes()


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


def mail_attachments(
    items: Any, *, limit: int, read: Mapping[str, bytes | str], workdir: str, guide: str
) -> tuple[list[Any], int, int]:
    """A sandboxed episode's mail attachments as Gmail takes them, with how many workspace files they carry and their
    bytes. An attachment that names a workspace file by its path carries the file's content (UTF-8 text as
    ``content``, other bytes as ``content_base64``), read beforehand by the route (``read``: path -> its bytes, or
    why there are none); so does a bare string that writes a path, or names a file the workspace holds. One that
    pastes binary content, or text over ``limit`` bytes, is refused (``upload_from_workspace``), and so is one that
    names a file without its content (``attach_from_workspace``): each refusal gives the attachment that names the
    workspace file instead (``guide`` the workspace guide's path)."""
    out: list[Any] = []
    files = size = 0
    for item in items if isinstance(items, list) else [items]:
        if isinstance(item, str) and named_file(item, read, workdir):
            item = {"path": item}
        fields = item if isinstance(item, Mapping) else {}
        if "path" in fields:
            if CONTENT_KEYS & set(fields):
                raise InvalidRequest(
                    "attachments: give a file's path or its content, not both", code="invalidArgument"
                )
            path = attachment_path(fields["path"], workdir)
            data = read.get(path) if path is not None else None
            if data == "too_large":
                raise InvalidRequest(
                    f"attachments: {path} is over {MAX_UPLOAD_BYTES // 1_000_000} MB", code="invalidArgument"
                )
            if data == "too_many":
                raise InvalidRequest(
                    f"attachments: a message attaches at most {MAX_ATTACHMENTS} workspace files",
                    code="invalidArgument",
                )
            if data == "too_much":
                raise InvalidRequest(
                    f"attachments: the workspace files together are over {MAX_ATTACHMENT_BYTES // 1_000_000} MB",
                    code="invalidArgument",
                )
            if not isinstance(data, bytes):
                raise NotFound(f"attachments: there is no file at {path or fields['path']} in the workspace")
            entry = {key: value for key, value in fields.items() if key != "path"}
            if not (entry.get("filename") or entry.get("name")):
                entry["filename"] = posixpath.basename(path)
            try:
                entry["content"] = data.decode("utf-8")
            except UnicodeDecodeError:
                entry["content_base64"] = base64.b64encode(data).decode("ascii")
            out.append(entry)
            files, size = files + 1, size + len(data)
            continue
        name = item if isinstance(item, str) else fields.get("filename") or fields.get("name") or "<file>"
        content = fields.get("content")
        use = json.dumps({"path": posixpath.join(workdir, str(name).strip())})
        if fields.get("content_base64", fields.get("contentBase64")) is not None:
            held = "binary content"
        elif isinstance(content, str) and len(content.encode("utf-8")) > limit:
            held = f"text over {limit} bytes"
        elif content is None:
            raise InvalidRequest(
                f"attachments: {name!r} carries no content. Attach a file from the workspace by its path ({guide}): "
                f"{use}",
                code="attach_from_workspace",
            )
        else:
            out.append(item)
            continue
        raise InvalidRequest(
            f"Attachment content is not pasted into a call in this sandbox ({held}). Write the file in the workspace "
            f"and attach it by its path ({guide}): {use}",
            code="upload_from_workspace",
        )
    return out, files, size


class FileApiCalls:
    """A sandboxed episode's exposure, answering the file API's own calls (``_Episode.direct``) as the tools exposure
    does whatever the row's surface: ``<app>_<op>`` parsed on the session's catalog, the result as plain JSON, an
    error as the app's body, its HTTP status kept for the file API to answer with (``direct_status``: Box's and
    Microsoft Graph's error bodies carry no numeric code). Every other call is the session's own exposure, which it
    wraps. A ``cli`` episode's only tool is ``run_command`` and an ``api_search`` episode's are the finding tools and
    ``api_call``, so without it the file API would reach no app on those surfaces."""

    def __init__(self, inner: Any, episode: "_Episode") -> None:
        self._inner = inner
        self._episode = episode
        self._direct = ToolsExposure(inner.catalog, {})

    def __getattr__(self, name: str) -> Any:  # tools, describe, catalog: the session's exposure
        if name.startswith("__") or name in ("_inner", "_episode", "_direct"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def _exposure(self) -> Any:
        return self._direct if self._episode.direct else self._inner

    def parse(self, tool_name: str, args: dict[str, Any]) -> Any:
        return self._exposure().parse(tool_name, args)

    def format_result(self, inv: Any, result: dict[str, Any]) -> Any:
        return self._exposure().format_result(inv, result)

    def format_error(self, inv: Any, err: Any) -> Any:
        if self._episode.direct:  # the file API answers with the app's HTTP status
            self._episode.direct_status = getattr(err, "status", None)
        return self._exposure().format_error(inv, err)


class ContentInWorkspace:
    """A sandboxed episode's exposure with the file apps' content, and Gmail's attachments, kept out of the
    conversation; everything else is the session's own exposure, which it wraps. A download saves the file in the
    workspace and answers with its path (``saved_download``), and an upload that pastes its content is refused with
    the file API command that sends the file (``pasted_content_refusal``, an app error in every surface's own form).
    A message's attachment that names a workspace file by its path carries that file (``mail_attachments``); pasted
    attachment content is refused the same way. Moving bytes is the route's: ``format_result`` queues a download on
    the episode, which the route writes into the sandbox before it answers (``OfficeEnvResourcesServer.
    _save_downloads``), and the route reads the files a call attaches before it runs (``attachment_paths``,
    ``_Episode.attached``). The file API's own calls (``_Episode.direct``) pass as they are, to the
    ``FileApiCalls`` it wraps. With no ``limit``, content moves through the conversation."""

    def __init__(
        self, inner: Any, episode: "_Episode", *, limit: int | None, folder: str, guide: str, world: Any
    ) -> None:
        self._inner = inner
        self._episode = episode
        self._limit = limit
        self._folder = folder
        self._guide = guide
        self._world = world

    def __getattr__(self, name: str) -> Any:  # tools, describe: the session's exposure
        if name.startswith("__") or name in ("_inner", "_episode", "_limit", "_folder", "_guide", "_world"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def attachment_paths(self, tool_name: str, args: Any) -> list[str]:
        """The workspace files a call's mail attachments name by path, for the route to read before the call runs;
        none for any other call, one that does not parse, or when content moves through the conversation. Parsing is
        the session exposure's own, which has no effects (refusals and counts are this wrapper's, in ``parse``)."""
        raw = args if isinstance(args, (bytes, str)) else json.dumps(args)
        # no attachments at all (a command line holds their JSON quoted)
        if self._limit is None or (b"attachments" if isinstance(raw, bytes) else "attachments") not in raw:
            return []
        if isinstance(args, (bytes, str)):
            try:
                args = parse_arguments(args)
            except (ValueError, RecursionError):
                return []
        if not isinstance(args, dict):
            return []
        try:
            parsed = self._inner.parse(tool_name, args)
        except Exception:  # the call itself answers with the error
            return []
        if not isinstance(parsed, Invocation) or (parsed.app, parsed.op) not in ATTACHMENT_UPLOADS:
            return []
        items = parsed.params.get("attachments")
        paths = []
        for item in items if isinstance(items, list) else [items]:
            named = attachment_candidate(item)
            path = attachment_path(named, self._folder) if named is not None else None
            if path is not None and path not in paths:
                paths.append(path)
        return paths

    def parse(self, tool_name: str, args: dict[str, Any]) -> Any:
        parsed = self._inner.parse(tool_name, args)
        if self._limit is None or not isinstance(parsed, Invocation) or self._episode.direct:
            return parsed
        if (parsed.app, parsed.op) in CONTENT_UPLOADS:
            refusal = pasted_content_refusal(
                parsed.app, parsed.op, parsed.params, self._limit, guide=self._guide, world=self._world
            )
            if refusal is not None:
                self._episode.uploads_refused += 1
                raise InvalidRequest(refusal, code="upload_from_workspace")
        elif (parsed.app, parsed.op) in ATTACHMENT_UPLOADS and parsed.params.get("attachments") is not None:
            try:
                attachments, files, size = mail_attachments(
                    parsed.params["attachments"],
                    limit=self._limit,
                    read=self._episode.attached,
                    workdir=self._folder,
                    guide=self._guide,
                )
            except InvalidRequest as exc:
                if exc.code == "upload_from_workspace":
                    self._episode.uploads_refused += 1
                raise
            self._episode.files_attached += files
            self._episode.bytes_attached += size
            parsed = replace(parsed, params={**parsed.params, "attachments": attachments})
        return parsed

    def format_result(self, inv: Any, result: dict[str, Any]) -> Any:
        if (
            self._limit is not None
            and (inv.app, inv.op) in CONTENT_DOWNLOADS | ATTACHMENT_DOWNLOADS
            and isinstance(result, dict)
            and not self._episode.direct
        ):
            file_id, name = download_identity(self._world, inv.app, inv.op, inv.params, result)
            saved = saved_download(
                inv.app,
                inv.op,
                inv.params,
                result,
                folder=f"{self._folder}/{DOWNLOAD_FOLDERS[inv.app]}",
                limit=self._limit,
                file_id=file_id,
                name=name,
            )
            if saved is not None:
                result, target, data = saved
                self._episode.saves.append((target, data))
        return self._inner.format_result(inv, result)


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


def _error_status(body: Any) -> int | None:
    """The HTTP status an app's error output stands for, or None when the output is no error."""
    error = body.get("error") if isinstance(body, dict) else None
    if error is None:
        return None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, int) and 400 <= code < 600 else 400


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
    # How its harness reaches its apps (SANDBOX_SURFACES; `tools` without a sandbox), the apps it connects, the tools
    # its MCP token admits, and a `cli` episode's run_command tool, which its apps' commands call
    surface: str = "tools"
    apps: tuple[str, ...] = ()
    grant: list[str] = field(default_factory=list)
    command_tool: str | None = None
    # In sandbox mode: the episode's file API token, its sandbox and the access lent to the harness, and whether
    # its verification has started, from which point its tools are closed.
    token: str | None = None
    sandbox: AsyncSandbox | None = None
    access: SandboxAccess | None = None
    grading: bool = False
    provider_failure: str | None = None  # the first sandbox operation the provider failed (_provider_failed)
    # The file apps' content and Gmail's attachments kept out of the conversation (ContentInWorkspace): files saved in
    # the workspace instead, uploads refused for pasting their content, and workspace files attached to mail
    files_saved: int = 0
    bytes_saved: int = 0
    uploads_refused: int = 0
    files_attached: int = 0
    bytes_attached: int = 0
    saves: list[tuple[str, bytes]] = field(default_factory=list)  # (path, bytes) the current call saved
    folders: set[str] = field(default_factory=set)  # folders made in the sandbox for them
    attached: dict[str, bytes | str] = field(default_factory=dict)  # the current call's attached files, or why not
    direct: bool = False  # a file API call is under way: its content moves as it is
    direct_status: int | None = None  # the HTTP status of the error that call met, if any
    # What the episode sent through its surface's routes
    commands: int = 0
    commands_refused: int = 0  # calls to CLI_ROUTE made by hand rather than by an app's command
    endpoint_calls: int = 0

    def usage(self) -> dict[str, float]:
        """What the episode sent beside MCP, and moved through its workspace instead of the conversation, as verify
        metrics."""
        return {
            "sandbox_commands": float(self.commands),
            "sandbox_commands_refused": float(self.commands_refused),
            "sandbox_endpoint_calls": float(self.endpoint_calls),
            "sandbox_files_saved": float(self.files_saved),
            "sandbox_bytes_saved": float(self.bytes_saved),
            "sandbox_uploads_refused": float(self.uploads_refused),
            "sandbox_files_attached": float(self.files_attached),
            "sandbox_bytes_attached": float(self.bytes_attached),
        }


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

    def get_session_cookie_name(self) -> str:
        """In sandbox mode a sandbox's calls get the session cookie too (``curl -v`` shows it), so its name says
        nothing of the environment: "sid_" and a hash of Gym's name for it, which keeps it unique per server."""
        key = self.get_session_middleware_key()
        if self.config.sandbox_provider is None:
            return key
        return f"sid_{hashlib.sha256(key.encode()).hexdigest()[:12]}"

    def _require_gym_caller(self, request: Request) -> None:
        """In sandbox mode, answer a caller from another host as a route that does not exist: seeding, closing,
        verification, the app catalog and the tool routes answer Gym's servers on this node (and
        ``trusted_networks``) only. A check in the handlers, not middleware, because Gym's MCP exposure
        dispatches tool calls straight to them (as from loopback) and refuses servers with middleware of their
        own."""
        if self.config.sandbox_provider is None:
            return
        host = request.client.host if request.client else None
        if not is_trusted_caller(host, local=self._node_addresses, networks=self._trusted_networks):
            raise HTTPException(status_code=404, detail="Not Found")

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.get("/officeenv/apps")(self.list_apps)
        if self.config.sandbox_provider is not None:
            self._setup_sandbox_routes(app)
        app.post("/{tool_name}")(self.tool_call)  # registered last: fixed routes win
        if self.config.sandbox_provider is not None:
            # A sandbox reaches this server, and generated docs would show it every route: none are served.
            docs = {app.openapi_url, app.docs_url, app.swagger_ui_oauth2_redirect_url, app.redoc_url} - {None}
            app.router.routes[:] = [route for route in app.router.routes if getattr(route, "path", None) not in docs]
        return app

    def _setup_sandbox_routes(self, app: FastAPI) -> None:
        """The file API, the `cli` and `api_search` surfaces' routes, and a lifespan that stops the sandboxes of open
        episodes at shutdown."""
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

    def _seed_episode(
        self, request: Request, body: ResourcesSeedSessionRequest, *, sandboxed: bool = False
    ) -> ResourcesSeedSessionResponse:
        """Seed an Environment Server episode under its ``resources_session_id``. Seeding the same ID
        for the same episode again answers as the first seed did; a closed ID is never seeded again."""
        session_id = body.resources_session_id
        # Nothing below awaits, so no other request runs between the checks and the updates.
        if session_id in self._closed:
            raise ResourcesSessionConflictError(f"Resources session {session_id} is already closed")
        episode = self._episodes.get(session_id)
        if episode is None:
            reply = self._seed(session_id, body.task_data, sandboxed=sandboxed)
            episode = self._episodes[session_id] = self._new_episode(session_id, body, reply, sandboxed=sandboxed)
        elif (episode.episode_id, episode.task_id) != (body.episode_id, body.task_id):
            raise ResourcesSessionConflictError("resources_session_id is already bound to another episode or task")
        # Gym's session routing names the cookie session after the ID too; this keeps the two the same without it.
        request.session[SESSION_ID_KEY] = session_id
        _SEEDED_TOOLS.set(episode.grant)
        return ResourcesSeedSessionResponse(resources_session_id=session_id)

    def _new_episode(
        self, session_id: str, body: ResourcesSeedSessionRequest, reply: Reply, *, sandboxed: bool
    ) -> _Episode:
        """The episode a seed made, with how its harness reaches its apps: a sandboxed episode by its row's exposure
        (SANDBOX_SURFACES), any other by its tools. Its MCP token admits the row's tools, none for a `cli` episode,
        whose apps are commands, and the finding tools for an `api_search` one."""
        session = self.service.store.get(session_id)
        tools = list(reply.body["tools"])
        surface = session.describe()["exposure"]["mode"] if sandboxed else "tools"
        grant = {"tools": tools, "cli": [], "api_search": [name for name, _, _ in SEARCH_TOOLS]}[surface]
        return _Episode(
            body.episode_id,
            body.task_id,
            surface=surface,
            apps=tuple(sorted(session.apps)),
            grant=grant,
            command_tool=tools[0] if surface == "cli" else None,
        )

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
        """Seed an Environment Server episode and lend it a sandbox, from the image its row needs. Seeding the
        same ID for the same episode again lends the same sandbox; a sandbox that fails to start leaves no
        world behind, so the seed can be retried. Once shutdown has begun, a seed starts no sandbox (503)."""
        session_id = body.resources_session_id
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            if self._shutting_down:  # shutdown stops the sandboxes it finds, and would not find this one
                raise HTTPException(status_code=503, detail="The server is shutting down")
            new = session_id not in self._episodes
            image = self._sandbox_image(body.task_data)
            self._seed_episode(request, body, sandboxed=True)
            episode = self._episodes[session_id]
            if new:
                try:
                    self._keep_content_in_workspace(session_id, episode)
                    await self._lend_sandbox(session_id, episode, image, task_id=body.task_id.task_id)
                except BaseException:
                    del self._episodes[session_id]
                    self._discard(session_id)
                    raise
        return ResourcesSeedSessionResponse(resources_session_id=session_id, sandbox_access=episode.access)

    def _workdir(self) -> str:
        return self.config.sandbox_config.workdir.rstrip("/")

    def _guide_path(self) -> str:
        return f"{self._workdir()}/{GUIDE_NAME}"

    def _keep_content_in_workspace(self, session_id: str, episode: _Episode) -> None:
        """Move the episode's file content and mail attachments through its workspace (ContentInWorkspace), unless
        ``inline_file_bytes`` is None; either way the file API reaches the file apps whatever the surface and answers
        with their HTTP status (FileApiCalls)."""
        session = self.service.store.get(session_id)
        session.exposure = ContentInWorkspace(
            FileApiCalls(session.exposure, episode),
            episode,
            limit=self.config.sandbox_config.inline_file_bytes,
            folder=self._workdir(),
            guide=self._guide_path(),
            world=session.world,
        )

    async def _lend_sandbox(self, session_id: str, episode: _Episode, image: SandboxImage, *, task_id: str) -> None:
        """Start the episode's sandbox from ``image`` with the workspace guide, for a `cli` episode its apps'
        commands, and the episode's API (its URL and token) in its environment and at CREDENTIALS_PATH."""
        config = self.config.sandbox_config
        named_configs = get_global_config_dict()
        token = token_urlsafe(32)
        files_api_url = self.config.files_api_url or f"http://{self.config.host}:{self.config.port}"
        guide = workspace_guide(
            config.inline_file_bytes,
            episode.apps,
            surface=episode.surface,
            prefix=config.mcp_tool_prefix,
            workdir=self._workdir(),
            tools=tuple(image.provides),
            offline=config.block_internet,
        )
        files = {self._guide_path(): guide, CREDENTIALS_PATH: api_credentials(files_api_url.rstrip("/"), token)}
        if episode.surface == "cli":
            files[APP_COMMAND_PATH] = APP_COMMAND_SOURCE
        if config.host_relay:
            files[RELAY_PATH] = RELAY_SOURCE
        spec = config.spec(
            self.config.sandbox_provider,
            image=image.image,
            workdir=config.workdir,
            metadata={"officeenv-task": task_id[:63]},
            files=files,
            named_configs=named_configs,
        )
        spec = replace(spec, env={**spec.env, API_URL_ENV: files_api_url.rstrip("/"), API_TOKEN_ENV: token})
        provider_config = resolve_provider_config(self.config.sandbox_provider, named_configs)
        if config.block_internet:
            # Fails closed for a provider without network policies, or a server a sandbox cannot reach.
            own_urls = [files_api_url, f"http://{self.config.host}:{self.config.port}"]
            policy = restricted_network_policy(next(iter(provider_config)), egress_urls(own_urls, named_configs))
            spec = replace(spec, provider_options={**spec.provider_options, "network_policy": policy})
        sandbox = AsyncSandbox(provider_config)
        try:
            if episode.surface == "cli" or config.host_relay:
                await sandbox.start_with_setup(spec, lambda box: self._prepare_sandbox(box, episode, named_configs))
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

    async def _prepare_sandbox(self, sandbox: AsyncSandbox, episode: _Episode, named_configs: Mapping) -> None:
        """Make a started sandbox ready to lend: a `cli` episode's apps become commands, then the host relay starts."""
        config = self.config.sandbox_config
        if episode.surface == "cli":
            await run_setup(sandbox, link_app_commands(episode.apps))
        if config.host_relay:
            await start_relay(sandbox, config.host_relay, relay_ports(named_configs, own_port=self.config.port))

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

    def _seed(self, session_id: str, row: dict[str, Any], *, sandboxed: bool = False) -> Reply:
        """Build the session ``row`` describes. A malformed task, an unknown app or a row from
        another OfficeEnv major version fails the seed with OfficeEnv's status and error code, and so
        does a row whose apps the episode cannot reach (``_unreachable``); nothing is stored, so the
        seed can be retried."""
        reply = self.service.seed(session_id, row)
        if reply.status != 200:
            raise HTTPException(status_code=reply.status, detail=reply.body)
        message = self._unreachable(session_id, sandboxed=sandboxed)
        if message is not None:
            self._discard(session_id)
            raise HTTPException(
                status_code=400, detail={"error": {"code": "unsupported_exposure", "message": message}}
            )
        return reply

    def _unreachable(self, session_id: str, *, sandboxed: bool) -> str | None:
        """Why a seeded session's apps cannot be reached, if they cannot. Over MCP a harness reaches the ``tools``
        exposure's tools under their own names; a sandboxed episode's harness may reach its apps as commands
        (``cli``) or as endpoints it finds over MCP (``api_search``) instead, and no other way."""
        session = self.service.store.get(session_id)
        mode = session.describe()["exposure"]["mode"]
        if sandboxed and mode in SANDBOX_SURFACES and mode != "tools":
            return None
        if sandboxed and mode not in SANDBOX_SURFACES:
            return f"A sandboxed episode reaches its apps through {', '.join(SANDBOX_SURFACES)}, not {mode}"
        if not self.config.expose_tools_over_mcp:
            return None
        catalog = tools_exposure_catalog()
        unserved = sorted(tool["name"] for tool in session.tools() if catalog.get(tool["name"]) != tool)
        if not unserved:
            return None
        return (
            f"The MCP endpoint serves the tools exposure's tools, not {unserved[:5]}: "
            "emit the row with exposure mode 'tools' and the default separator"
        )

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
        self._require_open(episode)
        body = await request.body()
        reply = await self._episode_call(
            session_id, episode, tool_name, body, lambda: self.service.call(session_id, tool_name, body)
        )
        text = reply.text()
        if episode is not None and episode.surface == "api_search" and reply.status == 200:
            # the endpoints the finding tools find are called over HTTP
            limit = self.config.sandbox_config.inline_file_bytes
            text = with_http_calls(tool_name, text, limit, guide=self._guide_path(), workdir=self._workdir())
        return Response(content=text, media_type="application/json", status_code=reply.status)

    @staticmethod
    def _require_open(episode: _Episode | None) -> None:
        """Refuse a call of an episode whose verification has started, from which point its tools are closed."""
        if episode is not None and episode.grading:
            # neutral words: a process the harness left running in the sandbox may read them
            raise HTTPException(status_code=409, detail="The task has ended: its tools are closed")

    async def _episode_call(
        self, session_id: str, episode: _Episode | None, tool_name: str, body: bytes | str, call: Callable[[], Any]
    ) -> Any:
        """Run ``call``, the service's (synchronous) call of ``tool_name`` with ``body``, under the episode's lock,
        with the workspace files it attaches read from the sandbox before it (``_read_attachments``) and the files its
        downloads saved written there after it (``_save_downloads``). Verification stops the sandbox under the same
        lock: a call under way when verification begins finishes first, and one that comes later is refused. A call
        that raises saves nothing."""
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            self._require_open(episode)
            attached = await self._read_attachments(session_id, episode, tool_name, body)
            self._require_open(episode)  # verification may have begun while the call's attachments were read
            try:
                reply = self._attaching(episode, attached, call)
            except Exception:
                if episode is not None:
                    episode.saves.clear()
                raise
            await self._save_downloads(session_id, episode)
        return reply

    async def _read_attachments(
        self, session_id: str, episode: _Episode | None, tool_name: str, body: bytes | str
    ) -> dict[str, bytes | str]:
        """The workspace files a call's mail attachments name by path (ContentInWorkspace.attachment_paths), read from
        the episode's sandbox before the call runs: path -> its bytes, or why there are none. Each path is checked in
        the sandbox first (``sandbox_file_sizes``): only a regular file of at most MAX_UPLOAD_BYTES is downloaded, and
        nothing when the call names more than MAX_ATTACHMENTS files or more than MAX_ATTACHMENT_BYTES together. The copy
        is checked again (``copied_file``: it may have changed since), and a download that fails attaches nothing. A
        file gone since the check is missing, as the model left it; any other failed download, or a check that does
        not run, is the provider's, and fails the call (``_provider_failed``)."""
        session = self.service.store.get(session_id)
        exposure = session.exposure if session is not None else None
        if episode is None or episode.sandbox is None or not isinstance(exposure, ContentInWorkspace):
            return {}
        paths = exposure.attachment_paths(tool_name, body)
        if not paths:
            return {}
        if len(paths) > MAX_ATTACHMENTS:
            return dict.fromkeys(paths, "too_many")
        sizes = await self._sandbox_file_sizes(session_id, episode, paths)
        read: dict[str, bytes | str] = {}
        wanted: list[str] = []
        for path in paths:
            size = sizes.get(path)
            if size is None:
                read[path] = "missing"
            elif size > MAX_UPLOAD_BYTES:
                read[path] = "too_large"
            else:
                wanted.append(path)
        if sum(sizes[path] or 0 for path in wanted) > MAX_ATTACHMENT_BYTES:
            return dict.fromkeys(paths, "too_much")
        total = 0
        for path in wanted:
            with tempfile.TemporaryDirectory(prefix="officeenv-attachment-") as folder:
                local = Path(folder) / "file"
                try:
                    await episode.sandbox.download(path, local)
                except Exception as exc:  # nothing partial attaches
                    if (await self._sandbox_file_sizes(session_id, episode, [path]))[path] is not None:
                        raise self._provider_failed(session_id, episode, f"read {path}", exc) from exc
                    LOGGER.info("No attachment at %s in OfficeEnv session %s: %s", path, session_id, exc)
                    read[path] = "missing"  # gone since the check
                    continue
                read[path] = copied_file(local, MAX_UPLOAD_BYTES)
            total += len(read[path]) if isinstance(read[path], bytes) else 0
            if total > MAX_ATTACHMENT_BYTES:  # a file grew since the check
                return dict.fromkeys(paths, "too_much")
        return read

    async def _sandbox_file_sizes(self, session_id: str, episode: _Episode, paths: list[str]) -> dict[str, int | None]:
        """``sandbox_file_sizes`` in the episode's sandbox: a check that does not run fails the call."""
        try:
            return await sandbox_file_sizes(episode.sandbox, paths)
        except Exception as exc:
            raise self._provider_failed(session_id, episode, "check the attachments", exc) from exc

    @staticmethod
    def _attaching(episode: _Episode | None, attached: dict[str, bytes | str], call: Callable[[], Any]) -> Any:
        """Run ``call``, the service's (synchronous) call, with the files ``_read_attachments`` read for it in reach of
        the episode's exposure; no await comes between, so another call of the episode sees none of them."""
        if episode is None:
            return call()
        episode.attached = attached
        try:
            return call()
        finally:
            episode.attached = {}

    async def verify(
        self, request: Request, body: OfficeEnvVerifyRequest
    ) -> OfficeEnvVerifyResponse | OfficeEnvLengthPenaltyVerifyResponse:
        self._require_gym_caller(request)
        session_id = request.session[SESSION_ID_KEY]
        await self._stop_before_grading(session_id)
        result = self.service.verify(session_id)
        episode = self._episodes.get(session_id) if self.config.sandbox_provider is not None else None
        if episode is not None:  # what the sandboxed episode moved through its workspace instead of the conversation
            result = result | episode.usage()
            if episode.provider_failure is not None:  # the reward would measure the sandbox, not the policy
                result = result | {
                    "mask_sample": True,
                    "failure_kind": PROVIDER_UNAVAILABLE,
                    "failure_reason": episode.provider_failure,
                }
        if result.get("officeenv_failure") == "no_session":
            # Never seeded, verified already, closed, or gone with a restart: the world went with
            # it, so the reward would measure the environment, not the policy.
            result = result | lost_session_verification(result["failure_reason"])
        if not self.config.length_penalty.enabled:
            return OfficeEnvVerifyResponse(**{**body.model_dump(), **result})
        penalty = length_penalty_fields(result["reward"], body.response, self.config.length_penalty)
        return OfficeEnvLengthPenaltyVerifyResponse(**{**body.model_dump(), **result, **penalty})

    def mcp_tools(self, harvested: list["MCPTool"], catchall: Any | None) -> list["MCPTool"] | None:
        """Every tool of the ``tools`` exposure, and in sandbox mode an ``api_search`` episode's two finding tools,
        dispatched through the tool route; no other route is a tool. An episode's grant admits only its own."""
        if catchall is None:
            return []
        catalog = tools_exposure_catalog()
        limit = self.config.sandbox_config.inline_file_bytes if self.config.sandbox_provider is not None else None
        tools = []
        for name, tool in catalog.items():
            app, _, op = name.partition("_")
            description = content_description(app, op, tool["description"], limit, guide=self._guide_path())
            schema = content_schema(app, op, tool["parameters"], limit, workdir=self._workdir())
            tools.append(catchall.tool(name, schema, description))
        if self.config.sandbox_provider is not None:
            for name, schema, description in SEARCH_TOOLS:
                tools.append(catchall.tool(name, schema, description.format(guide=self._guide_path())))
        return tools

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
            raise HTTPException(status_code=401, detail="Missing or invalid bearer token")
        return session_id

    def _refuse_once_grading(self, session_id: str) -> None:
        """Refuse a file API write once the episode's verification has begun. The token is revoked then, but an
        upload it admitted earlier may still be streaming in, and must not change what is graded."""
        episode = self._episodes.get(session_id)
        if episode is not None and episode.grading:
            raise HTTPException(status_code=409, detail="The task has ended: its files are closed")

    def _file_op(
        self,
        session_id: str,
        app: str,
        verb: str,
        *,
        file_id: str | None = None,
        name: str | None = None,
        parent: str | None = None,
        mime_type: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """The app's op and arguments, content aside, for a file API call: ``verb`` download or update (the file
        ``file_id``) or create (``name`` in the folder ``parent``). Files and folders are named by id alone; where
        an op also wants the item's library, folder or name (SharePoint, OneDrive), they are read from the world.
        ``mime_type`` is Drive's alone."""
        if app == "gdrive":
            typed = {"mime_type": mime_type} if mime_type else {}
            if verb == "download":
                return "download_file", {"file_id": file_id}
            if verb == "update":
                return "update_file", {"file_id": file_id, **typed}
            return "create_file", {"name": name, **({"parents": [parent]} if parent else {}), **typed}
        if app == "box":
            if verb == "download":
                return "get_file_content", {"file_id": file_id}
            if verb == "update":
                return "upload_file_version", {"file_id": file_id}
            return "upload_file", {"name": name, "parent_id": parent or "0"}
        session = self.service.store.get(session_id)
        rows = _rows(session.world, app, "driveItems") if session is not None else {}
        row = rows.get(parent if verb == "create" else file_id or "")
        if row is None and (app == "sharepoint" or (verb == "create" and parent)):
            if verb == "create" and not parent:
                message = (
                    "Name the folder to upload into: add &parent=<folder id> (a folder's id, or a library root's)"
                )
                raise HTTPException(status_code=400, detail={"error": {"code": "invalid_request", "message": message}})
            missing = parent if verb == "create" else file_id
            raise HTTPException(
                status_code=404, detail={"error": {"code": "not_found", "message": f"No item {missing}"}}
            )
        drive = {"drive_id": row["driveId"]} if row is not None else {}
        if app == "sharepoint":
            if verb == "download":
                return "get_item_content", {**drive, "item_id": file_id}
            if verb == "update":  # SharePoint uploads name a file by its folder and name: a new version of it
                return "upload_content", {**drive, "parent_id": row["parent"], "name": row["name"]}
            return "upload_content", {**drive, "parent_id": parent, "name": name}
        if verb == "download":
            return "get_content", {**drive, "item_id": file_id}
        if verb == "update":
            return "upload_file", {**drive, "item_id": file_id}
        return "upload_file", {**drive, "parent_id": parent, "name": name} if parent else {"path": f"/{name}"}

    def _file_call(
        self, session_id: str, app_name: str, verb: str, content: Mapping[str, str] | None = None, **where: Any
    ) -> dict[str, Any]:
        if app_name not in FILE_APPS:
            raise HTTPException(status_code=404, detail="Not Found")  # files move for FILE_APPS' apps only
        op, args = self._file_op(session_id, app_name, verb, **where)
        episode = self._episodes.get(session_id)
        if episode is not None:  # the file API moves the content itself
            episode.direct, episode.direct_status = True, None
        try:
            reply = self.service.call(session_id, f"{app_name}_{op}", json.dumps({**args, **(content or {})}))
        finally:
            status = None
            if episode is not None:
                episode.direct, status = False, episode.direct_status
        body = reply.body if isinstance(reply.body, dict) else json.loads(reply.body)
        status = reply.status if reply.status != 200 else status or _error_status(body)
        if status is not None:
            raise HTTPException(status_code=status, detail=body)
        return body

    @staticmethod
    def _provider_failed(session_id: str, episode: _Episode, doing: str, exc: BaseException) -> HTTPException:
        """The answer (502) to a call whose sandbox operation ``doing`` the provider failed: it raised, or could not
        run its command, whatever the workspace holds. The model reads none of the provider's words: they go to the
        server's log and the episode's failure_reason. The episode's verification is masked as provider_unavailable,
        since its reward would measure the sandbox, not the policy."""
        LOGGER.error("Could not %s in the sandbox of OfficeEnv session %s", doing, session_id, exc_info=exc)
        if episode.provider_failure is None:
            episode.provider_failure = f"Could not {doing} in the sandbox: {type(exc).__name__}: {exc}"[:500]
        return HTTPException(status_code=502, detail=f"Could not {doing} in the workspace")

    async def _save_downloads(self, session_id: str, episode: _Episode | None) -> None:
        """Write into the episode's sandbox the files its call's downloads saved (ContentInWorkspace), before
        the route answers that they are there, as mode 644. The queue is taken before the first await, so a call
        writes only its own files. A file that cannot be written fails the call (502): because of what the
        workspace holds, or because the provider failed (``_provider_failed``)."""
        if episode is None or not episode.saves:
            return
        saves, episode.saves = episode.saves, []
        for target, data in saves:
            try:
                folder = posixpath.dirname(target)
                if folder not in episode.folders:
                    made = await episode.sandbox.exec(f"mkdir -p {shlex.quote(folder)}", cwd="/", timeout_s=60)
                    if made.error_type is not None:
                        raise RuntimeError(f"mkdir: {made.error_type} {(made.stderr or '')[-300:]}")
                    if made.return_code != 0:  # the workspace is in the way: a file where a folder goes
                        raise HTTPException(
                            status_code=502,
                            detail=f"Could not save {target} in the workspace: its folder could not be made",
                        )
                    episode.folders.add(folder)
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
        paths = " ".join(shlex.quote(target) for target, _ in saves)
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
    async def _content_args(cls, request: Request) -> dict[str, str]:
        """The request body as the app's upload arguments: UTF-8 text as ``content``, any other
        bytes base64-encoded as ``content_base64``, so a binary file keeps every byte."""
        data = await cls._bounded_body(request)
        try:
            return {"content": data.decode("utf-8")}
        except UnicodeDecodeError:
            return {"content_base64": base64.b64encode(data).decode("ascii")}

    async def download_file(self, request: Request, app_name: str, file_id: str) -> Response:
        body = self._file_call(self._file_session(request), app_name, "download", file_id=file_id)
        content, encoded = body.get("content"), body.get("contentBase64", body.get("content_base64"))
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
        content = await self._content_args(request)
        self._refuse_once_grading(session_id)  # verification may have begun while the body streamed in
        return JSONResponse(
            self._file_call(session_id, app, "create", content, name=name, parent=parent, mime_type=mime_type)
        )

    async def replace_file(
        self, request: Request, app_name: str, file_id: str, mime_type: str | None = None
    ) -> JSONResponse:
        session_id = self._file_session(request)
        self._refuse_once_grading(session_id)
        content = await self._content_args(request)
        self._refuse_once_grading(session_id)  # verification may have begun while the body streamed in
        return JSONResponse(
            self._file_call(session_id, app_name, "update", content, file_id=file_id, mime_type=mime_type)
        )

    # The `cli` and `api_search` surfaces, in sandbox mode: each request names its episode by the episode's bearer
    # token, as the file API's do.

    def _surface_episode(self, request: Request, surface: str) -> tuple[str, _Episode]:
        """The bearer's episode, if its apps are reached through ``surface``; to any other episode the route does
        not exist. An episode being verified has its tools closed."""
        session_id = self._file_session(request)
        episode = self._episodes.get(session_id)
        if episode is None or episode.surface != surface:
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
        session_id, episode = self._surface_episode(request, "cli")
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
        tool = episode.command_tool
        reply = await self._episode_call(
            session_id, episode, tool, body, lambda: self.service.call(session_id, tool, body)
        )
        try:
            envelope = json.loads(reply.text())
        except ValueError:
            envelope = None
        if isinstance(envelope, dict) and {"exit_code", "stdout", "stderr"} <= set(envelope):
            code = int(envelope["exit_code"])
            text = envelope["stdout"] if code == 0 else envelope["stderr"]
        else:  # no command answer (an op that crashed, a session gone): its body is the command's error
            code, text = 1, reply.text()
        limit = self.config.sandbox_config.inline_file_bytes
        text = with_content_descriptions(text, limit, argv, guide=self._guide_path(), workdir=self._workdir())
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
        under the error's HTTP status (a 404, a 429 with its retry hint), as a vendor's API would."""
        app_name, op = request.path_params["app_name"], request.path_params["op"]
        session_id, episode = self._surface_episode(request, "api_search")
        try:
            params = parse_arguments(await self._bounded_body(request))
            if not isinstance(params, dict):
                raise ValueError("send the endpoint's parameters as a JSON object")
        except (ValueError, RecursionError) as exc:
            body = {"error": {"code": "invalid_arguments", "message": f"invalid JSON body: {exc}"}}
            return JSONResponse(body, status_code=400)
        episode.endpoint_calls += 1
        call = {"endpoint": f"{app_name}.{op}", "params": params}
        # The store's call, not the service's: the service answers every app error with 200 (a tool output the
        # model reads), and an HTTP API answers with the error's status.
        try:
            result = await self._episode_call(
                session_id,
                episode,
                "api_call",
                json.dumps(call),
                lambda: self.service.store.call(session_id, "api_call", call),
            )
        except HTTPException:  # the call refused, or the sandbox failed it, as any surface answers
            raise
        except Exception:
            LOGGER.exception("The %s.%s endpoint failed in OfficeEnv session %s", app_name, op, session_id)
            # the details stay in the server's log: the model reads no exception text
            body = {"error": {"code": "internal_error", "message": "The request could not be completed"}}
            return JSONResponse(body, status_code=500)
        status = 200
        if result.is_error:
            status = result.status if 400 <= result.status < 600 else 400
        return Response(content=result.output, media_type="application/json", status_code=status)


if __name__ == "__main__":
    OfficeEnvResourcesServer.run_webserver()
