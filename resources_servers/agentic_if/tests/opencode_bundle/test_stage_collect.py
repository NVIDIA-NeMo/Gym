# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``stage_bundle`` and ``collect_bundle`` over a fake ``SandboxIO`` whose sandbox is a local directory.

The fake runs the real shell commands, with ``BUNDLE_ROOT`` mapped to a temporary directory, so the staging script,
modes and ownership are exercised as written.
"""

import asyncio
import dataclasses
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from nemo_gym.sandbox.providers import SandboxExecResult as ExecResult
from resources_servers.agentic_if.opencode_bundle import (
    BUNDLE_ROOT,
    BundleReceipt,
    UnsupportedFeatureError,
    bundle_files,
    collect_bundle,
    presentation_gaps,
    stage_bundle,
)
from resources_servers.agentic_if.opencode_bundle.layout import CALL_RECORDS, HOOK_RECEIPTS
from resources_servers.agentic_if.opencode_bundle.profile import compile_profile
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec

from .harness import call, needs_node, run_boundary, sse_wire, upstream
from .specs import private_like_spec


class LocalSandbox:
    """A ``SandboxIO`` whose ``BUNDLE_ROOT`` is ``root`` on this machine; every command runs in a real shell."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.commands: list[tuple[str, str | int | None]] = []
        self.fail_on: str | None = None
        self.corrupt: set[str] = set()

    def local(self, path: str) -> Path:
        assert path.startswith(BUNDLE_ROOT + "/"), path
        return self.root / path[len(BUNDLE_ROOT) + 1 :]

    async def exec(self, command: str, *, timeout: float, user: str | int | None = None) -> ExecResult:
        self.commands.append((command, user))
        if self.fail_on is not None and self.fail_on in command:
            return ExecResult(stdout="", stderr="simulated failure", return_code=1)
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            command.replace(BUNDLE_ROOT, str(self.root)),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        return ExecResult(stdout=stdout.decode(), stderr=stderr.decode(), return_code=process.returncode)

    async def upload_bytes(self, path: str, data: bytes) -> None:
        self.local(path).write_bytes(data)

    async def download_bytes(self, path: str) -> bytes:
        data = self.local(path).read_bytes()
        return data + b" " if path in self.corrupt else data


OWNER = f"{os.getuid()}:{os.getgid()}"


@pytest.fixture
def sandbox(tmp_path: Path) -> LocalSandbox:
    return LocalSandbox(tmp_path / "sandbox-bundle")


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.mark.asyncio
async def test_stage_writes_the_compiled_bundle_for_the_opencode_identity(sandbox: LocalSandbox) -> None:
    spec = private_like_spec()
    receipt = await stage_bundle(sandbox, spec=spec, agent_user="agent")
    files = bundle_files(spec)
    for name, data in files.items():
        assert (sandbox.root / name).read_bytes() == data
        assert mode(sandbox.root / name) == 0o644
    for directory in ("receipts", "exports"):
        assert list((sandbox.root / directory).iterdir()) == [] and mode(sandbox.root / directory) == 0o755
    assert mode(sandbox.root) == 0o755
    assert receipt == BundleReceipt(
        profile_sha256=json.loads(files["profile.json"])["profile_sha256"],
        boundary_sha256=hashlib.sha256(files["boundary.json"]).hexdigest(),
        files={f"{BUNDLE_ROOT}/{name}": hashlib.sha256(data).hexdigest() for name, data in files.items()},
        owner=OWNER,
    )
    # The identity is resolved as the agent user; staging runs as root.
    assert sandbox.commands[0] == ("id -u && id -g", "agent")
    assert all(user == "root" for _, user in sandbox.commands[1:])
    assert any(
        f"chown {OWNER} {BUNDLE_ROOT}/receipts {BUNDLE_ROOT}/exports" in command for command, _ in sandbox.commands
    )


@pytest.mark.asyncio
async def test_stage_resets_receipts_and_exports_but_keeps_the_prefix_export(sandbox: LocalSandbox) -> None:
    for name in ("receipts", "exports"):
        (sandbox.root / name).mkdir(parents=True)
        (sandbox.root / name / "stale.json").write_text("{}")
    (sandbox.root / "prefix-export.json").write_text('{"info": {}}')
    await stage_bundle(sandbox, spec=private_like_spec(), agent_user=None)
    assert (sandbox.root / "prefix-export.json").read_text() == '{"info": {}}'
    assert (
        not (sandbox.root / "receipts" / "stale.json").exists()
        and not (sandbox.root / "exports" / "stale.json").exists()
    )
    assert sandbox.commands[0] == ("id -u && id -g", None)


@pytest.mark.asyncio
async def test_stage_refuses_unsupported_rows_before_touching_the_sandbox(sandbox: LocalSandbox) -> None:
    spec = RuntimeSpec(harness="opencode", task_backend="swe", tools={"tool_names": {"read": "ReadSource"}})
    with pytest.raises(UnsupportedFeatureError):
        await stage_bundle(sandbox, spec=spec, agent_user=None)
    assert sandbox.commands == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fail_on, corrupt, message",
    [
        ("id -u", None, "cannot resolve the identity OpenCode runs as"),
        ("mkdir", None, "cannot create the plugin bundle directories"),
        ("chown", None, "cannot set the plugin bundle's ownership and modes"),
        (None, f"{BUNDLE_ROOT}/boundary.json", "does not match the bundle"),
    ],
)
async def test_stage_failures_raise(
    sandbox: LocalSandbox, fail_on: str | None, corrupt: str | None, message: str
) -> None:
    sandbox.fail_on = fail_on
    if corrupt is not None:
        sandbox.corrupt.add(corrupt)
    with pytest.raises(RuntimeError, match=message):
        await stage_bundle(sandbox, spec=private_like_spec(), agent_user=None)


@pytest.mark.asyncio
async def test_collect_names_every_missing_part(sandbox: LocalSandbox) -> None:
    receipt = await stage_bundle(sandbox, spec=private_like_spec(), agent_user=None)
    outputs = await collect_bundle(sandbox, expected=dataclasses.asdict(receipt))
    assert (outputs.exports, outputs.sessions, outputs.hook_receipts, outputs.event_records, outputs.call_records) == (
        {},
        [],
        [],
        [],
        [],
    )
    assert outputs.gaps == [
        "export_inventory_unavailable:FileNotFoundError",
        "hook_receipts_unavailable:FileNotFoundError",
        "event_records_unavailable:FileNotFoundError",
        "call_records_unavailable:FileNotFoundError",
        "profile_load_receipt_missing",
        "profile_plugin_mismatch",
        "boundary_load_receipt_missing",
        "boundary_not_configured",
    ]


def session_export(session: str, parent: str | None) -> dict:
    return {
        "info": {"id": session, "parentID": parent},
        "messages": [{"info": {"id": "m", "role": "user"}, "parts": []}],
    }


def write_exports(root: Path, sessions: list[tuple[str, str | None]], *, skip: str | None = None) -> None:
    exports = root / "exports"
    (exports / "sessions.json").write_text(
        json.dumps([{"id": session, "parent_id": parent} for session, parent in sessions])
    )
    for session, parent in sessions:
        if session != skip:
            (exports / f"{session}.json").write_text(json.dumps(session_export(session, parent)))


@needs_node
@pytest.mark.asyncio
async def test_staged_plugins_run_and_their_records_come_back_in_the_evidence_formats(
    sandbox: LocalSandbox, tmp_path: Path
) -> None:
    receipt = await stage_bundle(sandbox, spec=private_like_spec(), agent_user=None)
    events = [
        {"id": "c", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Done"}}]},
        {"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    # The schemas OpenCode presents once the profile plugin installed the row's interfaces.
    interfaces = compile_profile(private_like_spec())["tools"]
    body = {
        "stream": True,
        "messages": [{"role": "user", "content": "Fix it"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": interfaces[name]["parameters"] if name in interfaces else {"type": "object"},
                },
            }
            for name in ["bash", "edit", "glob", "grep", "read", "task", "todowrite", "write"]
        ],
    }
    run = run_boundary(
        tmp_path,
        boundary=None,
        bundle=sandbox.root,
        profile_plugin=sandbox.root / "if-profile.mjs",
        calls=[call(body, upstream(sse_wire(events), content_type="text/event-stream"), session="ses_root")],
        system="Literal system",
    )
    assert run.calls[0]["error"] is None
    write_exports(sandbox.root, [("ses_root", None), ("ses_child", "ses_root")])
    (sandbox.root / "exports" / "ses_root.events.jsonl").write_text('{"type": "step_start"}\n')
    for expected in (dataclasses.asdict(receipt),):
        outputs = await collect_bundle(sandbox, expected=expected)
        assert outputs.gaps == []
        assert outputs.exports == {
            "ses_root": session_export("ses_root", None),
            "ses_child": session_export("ses_child", "ses_root"),
        }
        assert [(session.id, session.parent_id) for session in outputs.sessions] == [
            ("ses_root", None),
            ("ses_child", "ses_root"),
        ]
        assert outputs.hook_receipts[0] == {"kind": "loaded", "profile_sha256": receipt.profile_sha256}
        assert outputs.event_records == [{"kind": "loaded", "plants": 1}]
        assert outputs.call_records == run.records.observations
        assert outputs.call_records[0]["request"]["messages"][0] == {"role": "system", "content": "Literal system"}
        assert outputs.boundary_issues == []
        assert [row["kind"] for row in outputs.boundary_receipts] == ["loaded", "configured"]
        # Only verify knows the row's system text: it must be what the boundary presented.
        assert presentation_gaps(outputs, system="Literal system") == []
        assert presentation_gaps(outputs, system="Another system") == [
            "boundary_system_mismatch",
            "system_prompt_not_presented:ses_root:msg-1",
        ]
        assert presentation_gaps(outputs, system=None) == ["boundary_system_mismatch"]
        assert set(outputs.files) >= {
            "exports/sessions.json",
            "exports/ses_root.json",
            "exports/ses_child.json",
            "exports/ses_root.events.jsonl",
            f"receipts/{HOOK_RECEIPTS}",
            f"receipts/{CALL_RECORDS}",
        }


@pytest.mark.asyncio
async def test_collect_names_bad_or_missing_exports(sandbox: LocalSandbox) -> None:
    receipt = dataclasses.asdict(await stage_bundle(sandbox, spec=private_like_spec(), agent_user=None))
    write_exports(
        sandbox.root, [("ses_root", None), ("ses_child", "ses_root"), ("ses_other", "ses_root")], skip="ses_child"
    )
    (sandbox.root / "exports" / "ses_other.json").write_text(json.dumps(session_export("ses_renamed", "ses_root")))
    outputs = await collect_bundle(sandbox, expected=receipt)
    assert set(outputs.exports) == {"ses_root", "ses_renamed"}
    assert "export_missing:ses_child:FileNotFoundError" in outputs.gaps
    assert "export_session_mismatch:ses_other" in outputs.gaps
    (sandbox.root / "exports" / "sessions.json").write_text('{"id": "ses_root"}')
    outputs = await collect_bundle(sandbox, expected=receipt)
    assert outputs.exports == {} and "export_inventory_invalid:ValueError" in outputs.gaps


@pytest.mark.asyncio
async def test_collect_checks_load_receipts_and_reports_boundary_issues(sandbox: LocalSandbox) -> None:
    receipt = await stage_bundle(sandbox, spec=private_like_spec(), agent_user=None)
    plugin_digest = receipt.files[f"{BUNDLE_ROOT}/if-profile.mjs"]
    rows = [
        {"kind": "loaded", "profile_sha256": receipt.profile_sha256},
        {"kind": "plugin", "plugin": "if-profile", "plugin_sha256": plugin_digest},
    ]
    (sandbox.root / "receipts" / HOOK_RECEIPTS).write_text("".join(json.dumps(row) + "\n" for row in rows))
    (sandbox.root / "receipts" / "hook-events-fired.jsonl").write_text('{"kind": "loaded", "plants": 1}\n')
    calls = [
        {
            "kind": "loaded",
            "plugin": "if-boundary",
            "boundary_version": 1,
            "config_sha256": "0" * 64,
            "plugin_sha256": receipt.files[f"{BUNDLE_ROOT}/if-boundary.mjs"],
            "pid": 1,
        },
        {"kind": "configured", "provider": "nemo_gym"},
        {
            "kind": "request",
            "call": "1:1",
            "observation": {
                "source": "model",
                "session_id": "s",
                "assistant_message_id": "a",
                "request": {},
                "execution_tools": {},
            },
        },
        {"kind": "issue", "call": "1:1", "error": "ValueError", "message": "duplicate tool call identity"},
        {"kind": "issue", "call": "1:2", "error": "ValueError", "message": "again"},
        {"kind": "response", "call": "9:9", "response": {}},
    ]
    (sandbox.root / "receipts" / CALL_RECORDS).write_text(
        "".join(json.dumps(row) + "\n" for row in calls) + "not json\n"
    )
    write_exports(sandbox.root, [("ses_root", None)])
    outputs = await collect_bundle(sandbox, expected=dataclasses.asdict(receipt))
    assert outputs.boundary_issues == ["ValueError", "ValueError"]
    assert outputs.call_records == [calls[2]["observation"]]
    assert outputs.gaps == [
        "call_records_unreadable:line_7",
        "model_call_records_invalid:response",
        "boundary_load_receipt_mismatch",
        "model_boundary_issue:ValueError",
    ]
