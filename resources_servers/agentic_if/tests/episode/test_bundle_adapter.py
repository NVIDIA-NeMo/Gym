# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The episode's adapter to the plugin bundle: the sandbox I/O it lends, and the system-prompt check at verify."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from resources_servers.agentic_if import opencode_bundle
from resources_servers.agentic_if.episode import bundle
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec
from resources_servers.agentic_if.tests.support import Transport, bundle_outputs, final_call_record, finished_session


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
async def test_stage_and_collect_lend_the_bundle_the_borrowed_transport(monkeypatch):
    calls = []
    receipt, outputs = object(), bundle_outputs([finished_session()])

    async def stage_bundle(io, *, spec, agent_user):
        calls.append((io, spec, agent_user))
        return receipt

    async def collect_bundle(io, *, expected):
        calls.append((io, expected))
        return outputs

    monkeypatch.setattr(opencode_bundle, "stage_bundle", stage_bundle)
    monkeypatch.setattr(opencode_bundle, "collect_bundle", collect_bundle)
    spec = RuntimeSpec(harness="opencode", task_backend="swe")
    staged = {"profile_sha256": "p" * 64, "boundary_sha256": "b" * 64, "files": {}}
    assert await bundle.stage(Transport(), spec=spec, agent_user="agent") is receipt
    assert await bundle.collect(Transport(), expected=staged) is outputs
    assert calls[0][1:] == (spec, "agent") and calls[1][1] == staged
    assert all(isinstance(call[0], bundle.BundleSandboxIO) for call in calls)


def _sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


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
def test_verify_names_where_the_model_was_not_given_the_rows_system_prompt(configured_with, call_system, gaps):
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
    assert opencode_bundle.presentation_gaps(outputs, system="Be terse.") == gaps


def test_a_row_without_a_system_message_needs_a_boundary_configured_without_one():
    outputs = bundle_outputs(
        [finished_session()],
        call_records=[final_call_record()],
        boundary_receipts=[{"kind": "configured", "system_sha256": _sha256("An agent's own prompt")}],
    )
    assert opencode_bundle.presentation_gaps(outputs, system=None) == ["boundary_system_mismatch"]
    outputs.boundary_receipts = [{"kind": "configured", "system_sha256": None}]
    assert opencode_bundle.presentation_gaps(outputs, system=None) == []


@pytest.mark.asyncio
async def test_the_prefix_is_staged_where_the_harness_imports_it():
    transport = Transport()
    prefix = {"info": {"id": "ses_prefix"}, "messages": []}
    path = await bundle.stage_session_import(transport, prefix)
    assert path == "/tmp/nemo-gym-agentic-if/prefix-export.json"
    assert json.loads(transport.uploads[path]) == prefix
    assert transport.calls[-1] == {"command": f"chmod 0644 {path}", "user": "root", "timeout_s": 30}
    transport.fail["chmod"] = SimpleNamespace(return_code=1, error_type=None, stdout="", stderr="denied")
    with pytest.raises(RuntimeError, match="prefix"):
        await bundle.stage_session_import(transport, prefix)
