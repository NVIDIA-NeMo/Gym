# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The episode's adapter to the plugin bundle: the sandbox I/O it lends, and the outputs it converts."""

import hashlib
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from resources_servers.agentic_if import opencode_bundle
from resources_servers.agentic_if.episode import bundle
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec
from resources_servers.agentic_if.tests.support import Transport, bundle_outputs, final_call_record, finished_session


@dataclass(frozen=True)
class Receipt:
    profile_sha256: str | None
    files: dict


@pytest.mark.asyncio
async def test_the_sandbox_io_runs_commands_and_moves_bytes_through_the_borrowed_transport():
    transport = Transport()
    io = bundle.BundleSandboxIO(transport)
    await io.exec("mkdir -p /tmp/x", timeout=30, user="agent")
    assert transport.calls[-1] == {"command": "mkdir -p /tmp/x", "timeout_s": 30, "user": "agent"}
    await io.upload_bytes("/tmp/x/file", b"\x00bytes\xff")
    assert transport.uploads["/tmp/x/file"] == b"\x00bytes\xff"
    assert await io.download_bytes("/tmp/x/file") == b"\x00bytes\xff"


@pytest.mark.asyncio
async def test_stage_requires_the_profile_hash_the_load_receipt_will_carry(monkeypatch):
    staged = []

    async def stage_bundle(io, *, spec, agent_user):
        staged.append((io, spec, agent_user))
        return Receipt(profile_sha256=profile, files={"profile.json": "0" * 64})

    monkeypatch.setattr(opencode_bundle, "stage_bundle", stage_bundle)
    spec = RuntimeSpec(harness="opencode", task_backend="swe")
    profile = "f" * 64
    result = await bundle.stage(Transport(), spec=spec, agent_user="agent")
    assert result.profile_sha256 == profile and result.receipt == {
        "profile_sha256": profile,
        "files": {"profile.json": "0" * 64},
    }
    assert staged[0][1:] == (spec, "agent") and isinstance(staged[0][0], bundle.BundleSandboxIO)
    profile = None
    with pytest.raises(RuntimeError, match="profile hash"):
        await bundle.stage(Transport(), spec=spec, agent_user=None)


@pytest.mark.asyncio
async def test_collect_checks_against_what_seed_staged_and_converts_the_outputs(monkeypatch):
    child = finished_session(session="ses_child")
    child["info"]["parentID"] = "ses_root"
    outputs = bundle_outputs([finished_session(), child], gaps=["receipts_unreadable"], boundary_issues=["TypeError"])
    calls = []

    async def collect_bundle(io, *, expected=None):
        calls.append((io, expected))
        return outputs

    monkeypatch.setattr(opencode_bundle, "collect_bundle", collect_bundle)
    staged = {"profile_sha256": "p" * 64, "boundary_sha256": "b" * 64, "files": {}}
    collected = await bundle.collect(Transport(), expected=staged, system=None)
    assert calls[0][1] == staged and isinstance(calls[0][0], bundle.BundleSandboxIO)
    assert set(collected.exports) == {"ses_root", "ses_child"}
    assert [(s.id, s.parent_id) for s in collected.sessions] == [("ses_root", None), ("ses_child", "ses_root")]
    assert collected.hook_receipts == outputs.hook_receipts and collected.gaps == ["receipts_unreadable"]
    assert collected.boundary_issues == ["TypeError"] and collected.files == outputs.files
    assert collected.presentation_gaps == []


def _sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured_with", "call_system", "gaps"),
    [
        ("Be terse.", "Be terse.", []),
        ("Be verbose.", "Be terse.", ["boundary_system_mismatch"]),
        (None, "Be terse.", ["boundary_system_mismatch"]),
        ("Be terse.", None, ["system_prompt_not_presented:ses_root:a1"]),
        ("Be terse.", "Be verbose.", ["system_prompt_not_presented:ses_root:a1"]),
    ],
)
async def test_collect_names_where_the_model_was_not_given_the_rows_system_prompt(
    monkeypatch, configured_with, call_system, gaps
):
    messages = [{"role": "user", "content": "Fix it"}]
    if call_system is not None:
        messages.insert(0, {"role": "system", "content": call_system})
    outputs = bundle_outputs(
        [finished_session()],
        call_records=[final_call_record() | {"request": {"messages": messages}}],
        boundary_receipts=[
            {"kind": "configured", "system_sha256": None if configured_with is None else _sha256(configured_with)}
        ],
    )

    async def collect_bundle(io, *, expected=None):
        return outputs

    monkeypatch.setattr(opencode_bundle, "collect_bundle", collect_bundle)
    collected = await bundle.collect(Transport(), expected=None, system="Be terse.")
    assert collected.presentation_gaps == gaps


@pytest.mark.asyncio
async def test_a_row_without_a_system_message_needs_a_boundary_configured_without_one(monkeypatch):
    outputs = bundle_outputs(
        [finished_session()],
        call_records=[final_call_record()],
        boundary_receipts=[{"kind": "configured", "system_sha256": _sha256("An agent's own prompt")}],
    )

    async def collect_bundle(io, *, expected=None):
        return outputs

    monkeypatch.setattr(opencode_bundle, "collect_bundle", collect_bundle)
    assert (await bundle.collect(Transport(), expected=None, system=None)).presentation_gaps == [
        "boundary_system_mismatch"
    ]
    outputs.boundary_receipts = [{"kind": "configured", "system_sha256": None}]
    assert (await bundle.collect(Transport(), expected=None, system=None)).presentation_gaps == []


@pytest.mark.asyncio
async def test_the_prefix_is_staged_where_the_harness_imports_it(monkeypatch):
    monkeypatch.setattr(opencode_bundle, "SESSION_IMPORT_PATH", "/tmp/nemo-gym-agentic-if/prefix-export.json")
    transport = Transport()
    prefix = {"info": {"id": "ses_prefix"}, "messages": []}
    path = await bundle.stage_session_import(transport, prefix)
    assert path == "/tmp/nemo-gym-agentic-if/prefix-export.json"
    assert json.loads(transport.uploads[path]) == prefix
    assert transport.calls[-1] == {"command": f"chmod 0644 {path}", "user": "root", "timeout_s": 30}
    transport.fail["chmod"] = SimpleNamespace(return_code=1, error_type=None, stdout="", stderr="denied")
    with pytest.raises(RuntimeError, match="prefix"):
        await bundle.stage_session_import(transport, prefix)


def test_the_bundle_package_exports_the_contract_api():
    for name in ("BUNDLE_ROOT", "PLUGIN_PATHS", "EXPORT_DIR", "SESSION_IMPORT_PATH", "RECEIPTS_DIR"):
        assert getattr(opencode_bundle, name)
    assert opencode_bundle.SESSION_IMPORT_PATH == opencode_bundle.BUNDLE_ROOT + "/prefix-export.json"
    assert callable(opencode_bundle.stage_bundle) and callable(opencode_bundle.collect_bundle)
