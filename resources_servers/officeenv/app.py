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
The sandbox gets the episode's file API (``/v1/files``) in ``COMPANY_API_URL`` and
``COMPANY_API_TOKEN`` (``README.md`` in the workspace documents it), so file content moves between
the workspace and the simulated file apps (Google Drive, Box, SharePoint, OneDrive) over HTTP
instead of through the model's context. Their own content tools keep it out too
(``sandbox_config.inline_file_bytes``): a download saves the file in the workspace
(``<app folder>/<file id>/<name>``, ``FILE_APPS``) and answers with its path, carrying the content
only for short text, and an upload that pastes binary content or long text is refused with the file
API command that sends the file instead.

What a sandbox reads of this server names a company's API, never the environment behind it or Gym:
the variables, routes, guide, answers and session cookie (``get_session_cookie_name``), and with the
config's ``mcp_server_name`` and ``mcp_bearer_token`` the MCP endpoint too. A sandbox can reach this
server, so sandbox mode guards it: seeding, closing, verification, the app catalog and the tool
routes answer callers other than this node and ``trusted_networks`` as routes that do not exist
(404), which leaves sandboxes the MCP endpoint and the file API, and no generated docs are served;
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
import shlex
import socket
import tempfile
import weakref
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from secrets import token_urlsafe
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from officeenv.apps import available, get_app_class
from officeenv.artifacts import ArtifactStore
from officeenv.errors import InvalidRequest
from officeenv.exposure import Catalog, ToolsExposure
from officeenv.exposure.base import Invocation
from officeenv.service import EpisodeService, Reply
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


class OfficeEnvSandboxConfig(SandboxConfig):
    """The sandbox lent to each Environment Server episode when ``sandbox_provider`` is set."""

    image: str | None = Field(default=None, description="OCI image, e.g. one built from sandbox/Dockerfile")
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
        "this many bytes, is refused with the file API command that sends the file. None answers and takes content "
        "in full, as without a sandbox",
    )

    @model_validator(mode="after")
    def _one_egress_policy(self) -> "OfficeEnvSandboxConfig":
        if self.block_internet and "network_policy" in self.provider_options:
            raise ValueError("block_internet sets provider_options.network_policy; set one or the other")
        if self.block_internet and self.host_relay:
            raise ValueError("block_internet is OpenSandbox's egress policy; host_relay serves a workstation's Docker")
        return self


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
# that name the episode's file API, its route, and the guide.
API_URL_ENV, API_TOKEN_ENV = "COMPANY_API_URL", "COMPANY_API_TOKEN"
FILES_ROUTE = "/v1/files"
GUIDE_NAME = "README.md"

FILES_GUIDE = f"""# Company apps

The company's apps are your tools, one per operation of an app. File content moves between this
workspace and the file apps over HTTP, without going through the conversation, byte for byte: text
files (CSV, TSV, JSON, Markdown) and binary ones (XLSX workbooks, PDFs, images) alike. <app> is
{{codes}}.

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
Find file and folder ids with the apps' tools (list or search files). Uploads are limited to
{MAX_UPLOAD_BYTES // 1_000_000} MB. A request that fails returns the app's error as JSON.
"""

CONTENT_GUIDE = """Downloading a file with the apps' tools saves it in this workspace (the answer's savedTo):

{folders}

The answer also carries the content only for a text file of at most {limit} bytes. An upload with
the apps' tools takes pasted content only as text of at most {limit} bytes: send any other file
with the commands above.
"""


def workspace_guide(
    content_limit: int | None = DEFAULT_INLINE_FILE_BYTES, apps: tuple[str, ...] = tuple(FILE_APPS)
) -> str:
    """The workspace guide for an episode whose file ``apps`` (FILE_APPS) are connected: how files move, and where
    downloads land when file content stays out of the conversation (``content_limit``, see ContentInWorkspace; None:
    it does not)."""
    shown = [FILE_APPS[app] for app in apps]
    guide = FILES_GUIDE.format(
        codes=", ".join(f"{app} ({FILE_APPS[app].label})" for app in apps),
        notes="".join(f"{spec.label}: {spec.note}.\n" for spec in shown),
    )
    if content_limit is None:
        return guide
    width = max(len(f"{spec.folder}/<file id>/<name>") for spec in shown) + 3
    folders = "\n".join(f"    {f'{spec.folder}/<file id>/<name>':<{width}}{spec.label}" for spec in shown)
    return guide + "\n" + CONTENT_GUIDE.format(folders=folders, limit=content_limit)


# The file apps' ops that answer a file's content, and those that take content to store. In a sandboxed episode the
# content moves through the workspace instead (ContentInWorkspace): bytes in the conversation cost the model their
# tokens on every later request, and a model that copies them by hand gets them wrong.
CONTENT_DOWNLOADS = frozenset((app, op) for app, spec in FILE_APPS.items() for op in spec.downloads)
CONTENT_UPLOADS = frozenset((app, op) for app, spec in FILE_APPS.items() for op in spec.uploads)
PREVIEW_CHARS = 2000  # the start of a text file too long to answer inline
EXPORT_EXTENSIONS = {
    "text/plain": ".txt",
    "text/html": ".html",
    "text/markdown": ".md",
    "text/csv": ".csv",
    "text/tab-separated-values": ".tsv",
    "application/pdf": ".pdf",
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
}


def content_description(app: str, op: str, description: str, limit: int | None, *, guide: str) -> str:
    """A tool's description as a sandboxed episode reads it: the file apps' content ops say what they do there
    (CONTENT_DESCRIPTIONS, ``guide`` the workspace guide's path); every other op, and every op when content moves
    through the conversation (``limit`` None), keeps OfficeEnv's. The tool's app label (``[Google Drive] ``) stays."""
    text = CONTENT_DESCRIPTIONS.get((app, op))
    if text is None or limit is None:
        return description
    label = description[: description.index("] ") + 2] if description.startswith("[") and "] " in description else ""
    return label + text.format(limit=limit, guide=guide)


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
    """The folder under a file's own that keeps the download of one of its revisions or versions, if it is one."""
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
    answer names neither, the file's row in the session's world."""
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


class ContentInWorkspace:
    """A sandboxed episode's exposure with the file apps' content kept out of the conversation; everything else is
    the session's own exposure, which it wraps. A download saves the file in the workspace and answers with its
    path (``saved_download``), and an upload that pastes its content is refused with the file API command that
    sends the file (``pasted_content_refusal``, an app error the session answers as any other). Saving is the
    route's: ``format_result`` queues the file on the episode, and the tool route writes it into the sandbox before
    it answers (``OfficeEnvResourcesServer._save_downloads``). The file API's own calls (``_Episode.direct``) pass
    as they are, and an error they meet is kept with its HTTP status for the file API to answer with. With no
    ``limit``, content moves through the conversation and only the file API's errors are kept."""

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

    def parse(self, tool_name: str, args: dict[str, Any]) -> Any:
        parsed = self._inner.parse(tool_name, args)
        if (
            self._limit is not None
            and isinstance(parsed, Invocation)
            and (parsed.app, parsed.op) in CONTENT_UPLOADS
            and not self._episode.direct
        ):
            refusal = pasted_content_refusal(
                parsed.app, parsed.op, parsed.params, self._limit, guide=self._guide, world=self._world
            )
            if refusal is not None:
                self._episode.uploads_refused += 1
                raise InvalidRequest(refusal, code="upload_from_workspace")
        return parsed

    def format_result(self, inv: Any, result: dict[str, Any]) -> Any:
        if (
            self._limit is not None
            and (inv.app, inv.op) in CONTENT_DOWNLOADS
            and isinstance(result, dict)
            and not self._episode.direct
        ):
            file_id, name = download_identity(self._world, inv.app, inv.op, inv.params, result)
            saved = saved_download(
                inv.app,
                inv.op,
                inv.params,
                result,
                folder=f"{self._folder}/{FILE_APPS[inv.app].folder}",
                limit=self._limit,
                file_id=file_id,
                name=name,
            )
            if saved is not None:
                result, target, data = saved
                self._episode.saves.append((target, data))
        return self._inner.format_result(inv, result)

    def format_error(self, inv: Any, err: Any) -> Any:
        if self._episode.direct:  # the file API answers with the app's HTTP status
            self._episode.direct_status = getattr(err, "status", None)
        return self._inner.format_error(inv, err)


RELAY_SOURCE = (Path(__file__).parent / "sandbox" / "relay.py").read_text(encoding="utf-8")
RELAY_PATH = "/usr/local/lib/net-relay/relay.py"
RELAY_LOG = "/tmp/net-relay.log"


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
    tools: list[str]
    # In sandbox mode: the episode's file API token, its sandbox and the access lent to the harness, and whether
    # its verification has started, from which point its tools are closed.
    token: str | None = None
    sandbox: AsyncSandbox | None = None
    access: SandboxAccess | None = None
    grading: bool = False
    provider_failure: str | None = None  # the first sandbox operation the provider failed (_provider_failed)
    # Drive content kept out of the conversation (ContentInWorkspace): files saved in the workspace instead, and
    # uploads refused for pasting their content
    files_saved: int = 0
    bytes_saved: int = 0
    uploads_refused: int = 0
    saves: list[tuple[str, bytes]] = field(default_factory=list)  # (path, bytes) the current call saved
    folders: set[str] = field(default_factory=set)  # folders made in the sandbox for them
    direct: bool = False  # a file API call is under way: its content moves as it is
    direct_status: int | None = None  # the HTTP status of the error that call met, if any

    def usage(self) -> dict[str, float]:
        """What the episode moved through its workspace instead of the conversation, as verify metrics."""
        return {
            "sandbox_files_saved": float(self.files_saved),
            "sandbox_bytes_saved": float(self.bytes_saved),
            "sandbox_uploads_refused": float(self.uploads_refused),
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
        app.get(f"{FILES_ROUTE}/{{app_name}}/{{file_id}}")(self.download_file)
        app.put(f"{FILES_ROUTE}/{{app_name}}/{{file_id}}")(self.replace_file)
        app.post(FILES_ROUTE)(self.upload_file)

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
                    self._keep_content_in_workspace(session_id, episode)
                    await self._lend_sandbox(session_id, episode, task_id=body.task_id.task_id)
                except BaseException:
                    del self._episodes[session_id]
                    self._discard(session_id)
                    raise
        return ResourcesSeedSessionResponse(resources_session_id=session_id, sandbox_access=episode.access)

    def _guide_path(self) -> str:
        return f"{self.config.sandbox_config.workdir.rstrip('/')}/{GUIDE_NAME}"

    def _keep_content_in_workspace(self, session_id: str, episode: _Episode) -> None:
        """Move the episode's file content through its workspace (ContentInWorkspace), unless ``inline_file_bytes``
        is None; either way the file API's calls answer with the app's HTTP status."""
        session = self.service.store.get(session_id)
        session.exposure = ContentInWorkspace(
            session.exposure,
            episode,
            limit=self.config.sandbox_config.inline_file_bytes,
            folder=self.config.sandbox_config.workdir.rstrip("/"),
            guide=self._guide_path(),
            world=session.world,
        )

    async def _lend_sandbox(self, session_id: str, episode: _Episode, *, task_id: str) -> None:
        """Start the episode's sandbox with the workspace guide and the episode's file API in its environment."""
        config = self.config.sandbox_config
        if not config.image:
            raise ValueError("Set sandbox_config.image to the episode sandbox image (see sandbox/Dockerfile)")
        named_configs = get_global_config_dict()
        connected = tuple(app for app in FILE_APPS if any(tool.startswith(f"{app}_") for tool in episode.tools))
        files = {self._guide_path(): workspace_guide(config.inline_file_bytes, connected or tuple(FILE_APPS))}
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
        spec = replace(spec, env={**spec.env, API_URL_ENV: files_api_url.rstrip("/"), API_TOKEN_ENV: token})
        provider_config = resolve_provider_config(self.config.sandbox_provider, named_configs)
        if config.block_internet:
            # Fails closed for a provider without network policies, or a server a sandbox cannot reach.
            own_urls = [files_api_url, f"http://{self.config.host}:{self.config.port}"]
            policy = restricted_network_policy(next(iter(provider_config)), egress_urls(own_urls, named_configs))
            spec = replace(spec, provider_options={**spec.provider_options, "network_policy": policy})
        sandbox = AsyncSandbox(provider_config)
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
        body = await request.body()
        # Verification stops the sandbox under the episode's lock, which a call holds until its downloads are in the
        # sandbox: a call under way when verification begins finishes first, and one that comes later is refused.
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            if episode is not None and episode.grading:
                # neutral words: a process the harness left running in the sandbox may read them
                raise HTTPException(status_code=409, detail="The task has ended: its tools are closed")
            reply = self.service.call(session_id, tool_name, body)
            await self._save_downloads(session_id, episode)
        return Response(content=reply.text(), media_type="application/json", status_code=reply.status)

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
        """Every tool of the ``tools`` exposure, dispatched through the tool route; no other route is a tool."""
        if catchall is None:
            return []
        catalog = tools_exposure_catalog()
        limit = self.config.sandbox_config.inline_file_bytes if self.config.sandbox_provider is not None else None
        tools = []
        for name, tool in catalog.items():
            app, _, op = name.partition("_")
            description = content_description(app, op, tool["description"], limit, guide=self._guide_path())
            tools.append(catchall.tool(name, tool["parameters"], description))
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


if __name__ == "__main__":
    OfficeEnvResourcesServer.run_webserver()
