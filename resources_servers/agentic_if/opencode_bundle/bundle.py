# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage the plugin bundle into a task sandbox at seed, and read back what it recorded at verify.

The Resources Server owns the sandbox; it passes a ``SandboxIO`` (its bound sandbox or an adapter). Every sandbox
operation is bounded by a timeout. Staging failures raise; collection never raises for a missing or unreadable part:
it reports a named gap instead, so evidence is never silently empty.
"""

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from shlex import quote
from typing import Protocol

from pydantic import JsonValue

from nemo_gym.sandbox.providers import SandboxExecResult as ExecResult
from resources_servers.agentic_if.if_grading.contracts import SessionInfo
from resources_servers.agentic_if.opencode_bundle.boundary import compile_boundary
from resources_servers.agentic_if.opencode_bundle.layout import (
    BOUNDARY_CONFIG,
    BOUNDARY_PLUGIN,
    BUNDLE_ROOT,
    CALL_RECORDS,
    EVENT_RECEIPTS,
    EXPORT_DIR,
    EXPORT_INVENTORY,
    HOOK_RECEIPTS,
    PLANTS_CONFIG,
    PROFILE_CONFIG,
    PROFILE_PLUGIN,
    RECEIPTS_DIR,
    SOURCE_DIR,
)
from resources_servers.agentic_if.opencode_bundle.profile import compile_plants, compile_profile
from resources_servers.agentic_if.opencode_bundle.records import boundary_records, parse_jsonl
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec


LOG = logging.getLogger(__name__)

_EXPORT_NAME = re.compile(r"[A-Za-z0-9_-]+\.json")


class SandboxIO(Protocol):
    """The sandbox operations the bundle needs; ``user=None`` runs as the sandbox's default identity."""

    async def exec(self, command: str, *, timeout: float, user: str | int | None = None) -> ExecResult:
        """Run a POSIX shell command."""
        ...

    async def upload_bytes(self, path: str, data: bytes) -> None:
        """Write ``data`` to the absolute sandbox ``path``."""
        ...

    async def download_bytes(self, path: str) -> bytes:
        """Read the absolute sandbox ``path``."""
        ...


@dataclass(frozen=True)
class BundleReceipt:
    """What ``stage_bundle`` staged; the load receipts collected at verify must match it."""

    profile_sha256: str
    """The compiled profile's hash, which the profile plugin's load receipt and every tool receipt carry."""
    boundary_sha256: str
    """SHA-256 of the staged ``boundary.json`` bytes, which the boundary's load receipt carries."""
    files: dict[str, str]
    """Absolute staged path -> SHA-256 of its bytes (both plugins and the three configuration files)."""
    owner: str
    """``uid:gid`` owning ``receipts/`` and ``exports/``: the identity OpenCode runs as."""


@dataclass(frozen=True)
class BundleOutputs:
    """Everything the bundle holds after the agent closed, in the formats the evidence builder consumes."""

    exports: dict[str, dict[str, JsonValue]]
    """Raw ``opencode export`` documents keyed by the session ID inside each export (``info.id``)."""
    sessions: list[SessionInfo]
    """The agent's session inventory (``exports/sessions.json``), in its order; empty when it is missing."""
    hook_receipts: list[dict[str, JsonValue]]
    """``receipts/hook-applied.jsonl``: profile load receipt, tool definitions, tool and presentation receipts."""
    event_records: list[dict[str, JsonValue]]
    """``receipts/hook-events-fired.jsonl``: the planted events' ``loaded``/``error`` line and their firings."""
    call_records: list[dict[str, JsonValue]]
    """The boundary's observed model calls (``records.BoundaryRecords.observations``), in request order."""
    boundary_issues: list[str]
    """Exception type names of the model calls the boundary refused or could not complete; any issue makes the
    episode an infrastructure error."""
    boundary_receipts: list[dict[str, JsonValue]] = field(default_factory=list)
    """The boundary's ``loaded``/``load_error`` and ``configured``/``config_error`` lines, one per plugin instance."""
    gaps: list[str] = field(default_factory=list)
    """Missing, unreadable or mismatched parts, and one ``model_boundary_issue:<type>`` per boundary issue type."""
    files: dict[str, bytes] = field(default_factory=dict)
    """Raw bytes of every file read, keyed by its path relative to ``BUNDLE_ROOT``, for the artifacts directory."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bundle_files(spec: RuntimeSpec) -> dict[str, bytes]:
    """The five files of one row's bundle, by file name; raises for unsupported or invalid rows."""
    return {
        PROFILE_PLUGIN: (SOURCE_DIR / PROFILE_PLUGIN).read_bytes(),
        BOUNDARY_PLUGIN: (SOURCE_DIR / BOUNDARY_PLUGIN).read_bytes(),
        # json.dumps defaults: the seed record keeps each staged file's digest (boundary.json's is also in the load
        # receipt), and fixtures/profile-goldens.json pins the profile's bytes.
        PROFILE_CONFIG: json.dumps(compile_profile(spec)).encode(),
        PLANTS_CONFIG: json.dumps(compile_plants(spec)).encode(),
        BOUNDARY_CONFIG: json.dumps(compile_boundary(spec)).encode(),
    }


async def _run(io: SandboxIO, command: str, *, timeout: float, user: str | int | None, what: str) -> str:
    async with asyncio.timeout(timeout):
        result = await io.exec(command, timeout=timeout, user=user)
    if result.return_code != 0 or result.error_type:
        detail = (result.stderr or result.stdout or "")[-1000:]
        raise RuntimeError(
            f"cannot {what} (return_code={result.return_code}, error_type={result.error_type}): {detail}"
        )
    return result.stdout or ""


async def _owner(io: SandboxIO, agent_user: str | int | None, *, timeout: float) -> str:
    """The numeric ``uid:gid`` of the identity OpenCode runs as: ``agent_user``, or the sandbox default."""
    output = await _run(
        io, "id -u && id -g", timeout=timeout, user=agent_user, what="resolve the identity OpenCode runs as"
    )
    numbers = output.split()
    if len(numbers) != 2 or not all(number.isascii() and number.isdigit() for number in numbers):
        raise RuntimeError(f"unexpected identity of the OpenCode user: {output[:200]!r}")
    return f"{numbers[0]}:{numbers[1]}"


async def stage_bundle(
    io: SandboxIO, *, spec: RuntimeSpec, agent_user: str | int | None, timeout: float = 60.0
) -> BundleReceipt:
    """Write the plugins and their compiled configuration into ``BUNDLE_ROOT``.

    Creates empty ``receipts/`` and ``exports/`` owned by the identity OpenCode runs as (``agent_user``, or the
    sandbox default when it is None), makes the files world-readable, and reads every staged file back to check its
    digest. Leaves anything else in ``BUNDLE_ROOT`` (such as ``prefix-export.json``) untouched. Raises
    ``UnsupportedFeatureError``/``ValueError`` for rows the bundle cannot run, ``RuntimeError`` for sandbox failures.
    """
    files = bundle_files(spec)
    owner = await _owner(io, agent_user, timeout=timeout)
    root, receipts, exports = quote(BUNDLE_ROOT), quote(RECEIPTS_DIR), quote(EXPORT_DIR)
    await _run(
        io,
        f"set -eu; mkdir -p {root}; rm -rf {receipts} {exports}; mkdir {receipts} {exports}",
        timeout=timeout,
        user="root",
        what="create the plugin bundle directories",
    )
    for name, data in files.items():
        async with asyncio.timeout(timeout):
            await io.upload_bytes(f"{BUNDLE_ROOT}/{name}", data)
    staged = " ".join(quote(f"{BUNDLE_ROOT}/{name}") for name in files)
    await _run(
        io,
        f"set -eu; chmod 0755 {root}; chmod 0644 {staged}; chown {owner} {receipts} {exports}; "
        f"chmod 0755 {receipts} {exports}",
        timeout=timeout,
        user="root",
        what="set the plugin bundle's ownership and modes",
    )
    digests = {}
    for name, data in files.items():
        path = f"{BUNDLE_ROOT}/{name}"
        async with asyncio.timeout(timeout):
            copy = await io.download_bytes(path)
        if _sha256(copy) != _sha256(data):
            raise RuntimeError(f"the staged copy of {path} does not match the bundle")
        digests[path] = _sha256(data)
    profile = json.loads(files[PROFILE_CONFIG])
    return BundleReceipt(
        profile_sha256=profile["profile_sha256"],
        boundary_sha256=_sha256(files[BOUNDARY_CONFIG]),
        files=digests,
        owner=owner,
    )


class _Reader:
    """Bounded reads that turn failures into named gaps."""

    def __init__(self, io: SandboxIO, timeout: float) -> None:
        self.io, self.timeout = io, timeout
        self.files: dict[str, bytes] = {}
        self.gaps: list[str] = []

    async def read(self, relative: str, *, gap: str) -> bytes | None:
        try:
            async with asyncio.timeout(self.timeout):
                data = await self.io.download_bytes(f"{BUNDLE_ROOT}/{relative}")
        # Deliberate boundary: a missing or unreadable part is reported, never fatal and never silently empty.
        except Exception as error:
            LOG.warning("cannot read %s from the plugin bundle: %s", relative, type(error).__name__)
            self.gaps.append(f"{gap}:{type(error).__name__}")
            return None
        self.files[relative] = data
        return data

    async def jsonl(self, relative: str, *, gap: str) -> list[dict[str, JsonValue]]:
        data = await self.read(relative, gap=f"{gap}_unavailable")
        if data is None:
            return []
        rows, gaps = parse_jsonl(data, name=gap)
        self.gaps.extend(gaps)
        return rows


async def _exports(io: SandboxIO, reader: _Reader, timeout: float) -> tuple[dict[str, dict], list[SessionInfo]]:
    """The agent's inventory (``sessions.json``) first, then each listed session's export.

    A missing or invalid inventory, and a listed session whose export is missing or unreadable, are named gaps.
    Exports are keyed by the session ID inside each document.
    """
    sessions: list[SessionInfo] = []
    inventory = await reader.read(f"exports/{EXPORT_INVENTORY}", gap="export_inventory_unavailable")
    if inventory is not None:
        try:
            rows = json.loads(inventory)
            if not isinstance(rows, list):
                raise ValueError("the session inventory is not a list")
            sessions = [SessionInfo(id=row["id"], parent_id=row.get("parent_id")) for row in rows]
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            reader.gaps.append(f"export_inventory_invalid:{type(error).__name__}")
            sessions = []
    exports: dict[str, dict] = {}
    for session in sessions:
        name = f"{session.id}.json"
        if not _EXPORT_NAME.fullmatch(name):
            reader.gaps.append(f"export_name_invalid:{session.id[:80]!r}")
            continue
        data = await reader.read(f"exports/{name}", gap=f"export_missing:{session.id}")
        if data is None:
            continue
        try:
            document = json.loads(data)
            identity = document["info"]["id"]
            if not isinstance(identity, str) or not identity:
                raise ValueError("the export names no session")
        except (ValueError, TypeError, KeyError):
            reader.gaps.append(f"export_unreadable:{session.id}")
            continue
        if identity != session.id:
            reader.gaps.append(f"export_session_mismatch:{session.id}")
        if identity in exports:
            reader.gaps.append(f"export_duplicate:{identity}")
            continue
        exports[identity] = document
    await _event_logs(io, reader, sessions, timeout)
    return exports, sessions


async def _event_logs(io: SandboxIO, reader: _Reader, sessions: list[SessionInfo], timeout: float) -> None:
    """Keep the root's OpenCode event log(s) for the artifacts; the evidence builder does not read them."""
    roots = [
        session.id
        for session in sessions
        if session.parent_id is None and _EXPORT_NAME.fullmatch(session.id + ".json")
    ]
    if len(roots) != 1:
        return
    listing = (
        f"cd {quote(EXPORT_DIR)} || exit 0; "
        f'for f in {quote(roots[0])}.*events.jsonl; do if [ -f "$f" ]; then printf "%s\\n" "$f"; fi; done'
    )
    try:
        names = (await _run(io, listing, timeout=timeout, user="root", what="list the event logs")).splitlines()
    # Deliberate boundary: event logs are kept for audit only, so a failure here is logged, not a gap.
    except Exception as error:
        LOG.warning("cannot list the OpenCode event logs: %s", type(error).__name__)
        return
    for name in names:
        if name.startswith(roots[0] + ".") and name.endswith("events.jsonl") and "/" not in name:
            try:
                async with asyncio.timeout(timeout):
                    reader.files[f"exports/{name}"] = await io.download_bytes(f"{EXPORT_DIR}/{name}")
            # Deliberate boundary, as above.
            except Exception as error:
                LOG.warning("cannot read the OpenCode event log %s: %s", name, type(error).__name__)


def _check_receipts(
    expected: Mapping[str, JsonValue],
    hook_receipts: list[dict[str, JsonValue]],
    loads: list[dict[str, JsonValue]],
    configurations: list[dict[str, JsonValue]],
) -> list[str]:
    gaps = []
    files = expected.get("files") or {}
    profile_plugin = files.get(f"{BUNDLE_ROOT}/{PROFILE_PLUGIN}")
    boundary_plugin = files.get(f"{BUNDLE_ROOT}/{BOUNDARY_PLUGIN}")
    if any(row.get("kind") == "load_error" for row in hook_receipts):
        gaps.append("profile_load_error")
    if not any(
        row.get("kind") == "loaded" and row.get("profile_sha256") == expected.get("profile_sha256")
        for row in hook_receipts
    ):
        gaps.append("profile_load_receipt_missing")
    plugin_rows = [row for row in hook_receipts if row.get("kind") == "plugin" and row.get("plugin") == "if-profile"]
    if not plugin_rows or any(row.get("plugin_sha256") != profile_plugin for row in plugin_rows):
        gaps.append("profile_plugin_mismatch")
    if any(row.get("kind") == "load_error" for row in loads):
        gaps.append("boundary_load_error")
    loaded = [row for row in loads if row.get("kind") == "loaded"]
    if not loaded:
        gaps.append("boundary_load_receipt_missing")
    elif any(
        row.get("config_sha256") != expected.get("boundary_sha256") or row.get("plugin_sha256") != boundary_plugin
        for row in loaded
    ):
        gaps.append("boundary_load_receipt_mismatch")
    if any(row.get("kind") == "config_error" for row in configurations):
        gaps.append("boundary_config_error")
    if not any(row.get("kind") == "configured" for row in configurations):
        gaps.append("boundary_not_configured")
    return gaps


async def collect_bundle(
    io: SandboxIO,
    *,
    expected: Mapping[str, JsonValue],
    timeout: float = 120.0,
) -> BundleOutputs:
    """Read the exports, hook receipts, event records and boundary call records left in ``BUNDLE_ROOT``.

    Call it after the agent closed and before the task backend touches the sandbox. ``expected`` is the seed's
    ``BundleReceipt`` (or its dict form); without it the load receipts are checked against the files still staged.
    Never raises for missing or unreadable parts: each is a named entry of ``gaps``.
    """
    reader = _Reader(io, timeout)
    exports, sessions = await _exports(io, reader, timeout)
    hook_receipts = await reader.jsonl(f"receipts/{HOOK_RECEIPTS}", gap="hook_receipts")
    event_records = await reader.jsonl(f"receipts/{EVENT_RECEIPTS}", gap="event_records")
    call_rows = await reader.jsonl(f"receipts/{CALL_RECORDS}", gap="call_records")
    records = boundary_records(call_rows)
    gaps = reader.gaps + records.gaps
    gaps += _check_receipts(expected, hook_receipts, records.loads, records.configurations)
    gaps += [f"model_boundary_issue:{issue}" for issue in dict.fromkeys(records.issues)]
    return BundleOutputs(
        exports=exports,
        sessions=sessions,
        hook_receipts=hook_receipts,
        event_records=event_records,
        call_records=records.observations,
        boundary_issues=records.issues,
        boundary_receipts=records.loads + records.configurations,
        gaps=list(dict.fromkeys(gaps)),
        files=reader.files,
    )


def presentation_gaps(outputs: BundleOutputs, *, system: str | None) -> list[str]:
    """Check at verify that the model saw the row's literal system text; only verify has the row's messages.

    ``system`` is the text of the row's system message, or None when it has none. The boundary reads it from the
    instructions file the agent writes, so a drift in the agent's configuration (its own ``system_prompt``, or no
    instructions file) would otherwise go unnoticed. Returns named gaps; an empty list means every configured boundary
    used ``system`` and every actor call carried it as the only system message.
    """
    gaps = []
    expected = None if system is None else _sha256(system.encode())
    for row in outputs.boundary_receipts:
        if row.get("kind") == "configured" and row.get("system_sha256") != expected:
            gaps.append("boundary_system_mismatch")
    if system is not None:
        for record in outputs.call_records:
            if record.get("source") != "model":
                continue
            request = record.get("request") if isinstance(record.get("request"), dict) else {}
            messages = request.get("messages") if isinstance(request.get("messages"), list) else []
            roles = [message.get("role") for message in messages if isinstance(message, dict)]
            if (
                not messages
                or messages[0] != {"role": "system", "content": system}
                or roles.count("system") + roles.count("developer") != 1
            ):
                gaps.append(
                    f"system_prompt_not_presented:{record.get('session_id')}:{record.get('assistant_message_id')}"
                )
    return list(dict.fromkeys(gaps))
