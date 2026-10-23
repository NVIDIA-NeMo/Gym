# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run ``if-boundary.mjs`` under Node through ``boundary_runner.mjs`` and read back what it recorded."""

import base64
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from resources_servers.agentic_if.opencode_bundle.layout import BOUNDARY_PLUGIN, CALL_RECORDS, SOURCE_DIR
from resources_servers.agentic_if.opencode_bundle.records import BoundaryRecords, boundary_records, parse_jsonl


NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is required to run the OpenCode plugins")
RUNNER = Path(__file__).with_name("boundary_runner.mjs")
BASE_URL = "http://model.test/ng-rollout/capture/v1"


def sse_wire(events: list[dict]) -> bytes:
    """Upstream SSE bytes for these events: one CRLF-terminated ``data:`` event each, then ``data: [DONE]``."""
    wire = "".join("data: " + json.dumps(event) + "\r\n\r\n" for event in events) + "data: [DONE]\r\n\r\n"
    return wire.encode()


def upstream(body: bytes, *, status: int = 200, content_type: str = "application/json", chunk: int = 7) -> dict:
    """One fake model-server reply for the runner."""
    return {
        "status": status,
        "content_type": content_type,
        "body_b64": base64.b64encode(body).decode(),
        "chunk": chunk,
    }


def call(
    body: dict | str,
    reply: dict,
    *,
    session: str = "root",
    assistant: str | None = "msg-1",
    agent: str = "build",
    url: str | None = None,
    cancel: bool = False,
) -> dict:
    """One model call: the request OpenCode sends and the identity its hooks see."""
    text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    entry = {"session": session, "agent": agent, "assistant": assistant, "body": text, "upstream": reply}
    if url is not None:
        entry["url"] = url
    if cancel:
        entry["cancel"] = True
    return entry


@dataclass
class BoundaryRun:
    """The runner's view of each call, plus the records the plugin wrote."""

    calls: list[dict]
    permission: dict | None
    provider_fetch_installed: bool
    rows: list[dict]
    records: BoundaryRecords
    global_probe: dict | None = None
    bundle: Path = field(default_factory=Path)

    def returned(self, index: int) -> bytes | None:
        """The bytes OpenCode received for call ``index``."""
        encoded = self.calls[index]["body_b64"]
        return None if encoded is None else base64.b64decode(encoded)


def make_bundle(tmp_path: Path, boundary: dict | bytes | None) -> Path:
    """A bundle directory holding the boundary plugin, its configuration and an empty receipts directory."""
    bundle = tmp_path / "bundle"
    (bundle / "receipts").mkdir(parents=True)
    shutil.copyfile(SOURCE_DIR / BOUNDARY_PLUGIN, bundle / BOUNDARY_PLUGIN)
    if boundary is not None:
        data = boundary if isinstance(boundary, bytes) else json.dumps(boundary).encode()
        (bundle / "boundary.json").write_bytes(data)
    return bundle


def run_boundary(
    tmp_path: Path,
    *,
    boundary: dict | bytes | None,
    calls: list[dict],
    system: str | None = None,
    instructions: list[str] | None = None,
    permission: dict | None = None,
    register_profile: bool = True,
    profile_plugin: Path | None = None,
    global_probe: bool = False,
    bundle: Path | None = None,
    provider_id: str | None = None,
) -> BoundaryRun:
    """Run the plugin over ``calls``; ``system`` is written to the instructions file the agent would write."""
    bundle = bundle or make_bundle(tmp_path, boundary)
    entries = list(instructions or [])
    if system is not None:
        session_dir = tmp_path / "agent-session"
        session_dir.mkdir(exist_ok=True)
        (session_dir / "instructions.md").write_text(system, encoding="utf-8")
        entries.append(str(session_dir / "instructions.md"))
    case = {
        "plugin": str(bundle / BOUNDARY_PLUGIN),
        "base_url": BASE_URL,
        "instructions": entries,
        "permission": permission,
        "register_profile": register_profile,
        "profile_plugin": None if profile_plugin is None else str(profile_plugin),
        "directory": str(tmp_path),
        "global_probe": global_probe,
        "calls": calls,
    }
    if provider_id is not None:
        case["provider_id"] = provider_id
    case_file, result_file = tmp_path / "case.json", tmp_path / "result.json"
    case_file.write_text(json.dumps(case))
    finished = subprocess.run(
        [NODE, str(RUNNER), str(case_file), str(result_file)], capture_output=True, text=True, timeout=60
    )
    assert finished.returncode == 0, finished.stdout + finished.stderr
    result = json.loads(result_file.read_text())
    records_file = bundle / "receipts" / CALL_RECORDS
    rows, gaps = parse_jsonl(records_file.read_bytes() if records_file.exists() else b"", name="call_records")
    assert gaps == []
    return BoundaryRun(
        calls=result["calls"],
        permission=result["permission"],
        provider_fetch_installed=result["provider_fetch_installed"],
        rows=rows,
        records=boundary_records(rows),
        global_probe=result.get("global_probe"),
        bundle=bundle,
    )
