# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exercise session attribution and copied-history rejection."""

import copy
import json

import pytest

from environments.skill_trigger.codex import CodexSessions, ModelRouteBoundary, PendingSession
from environments.skill_trigger.scoring import EvidenceError


def row(kind, **payload):
    return {"type": kind, "payload": payload}


def fixture(*, fork=False, selected_model=None):
    root = [
        row("session_meta", id="root", source="exec", cli_version="0.156.1"),
        row("turn_context", model="test-model"),
        row("response_item", type="function_call", name="exec_command", call_id="old", arguments="{}"),
        row("event_msg", type="exec_command_end", call_id="old", exit_code=0),
        row("response_item", type="function_call_output", call_id="old", output="Parent's earlier read"),
    ]
    copied = copy.deepcopy(root)
    args = {"message": "Read the skill", "fork_context": fork}
    if selected_model:
        args["model"] = selected_model
    root += [
        row("response_item", type="function_call", name="spawn_agent", call_id="spawn", arguments=json.dumps(args)),
        row("response_item", type="function_call_output", call_id="spawn", output='{"agent_id":"child"}'),
    ]
    child = [
        row(
            "session_meta",
            id="child",
            source={"subagent": "thread_spawn"},
            parent_thread_id="root",
            cli_version="0.156.1",
        )
    ]
    if fork:
        child[0]["payload"].update(forked_from_id="root", subagent_history_start_ordinal=len(copied) + 1)
        child += copied
    child += [
        row("turn_context", model=selected_model or "test-model"),
        row("response_item", type="function_call", name="exec_command", call_id="new", arguments="{}"),
        row("event_msg", type="exec_command_end", call_id="new", exit_code=0),
        row("response_item", type="function_call_output", call_id="new", output="New child's read"),
    ]
    return root, child


def collect(root, child=()):
    result = CodexSessions(model="test-model")
    result.select_root({"type": "thread.started", "thread_id": "root"})
    for name, records in [("root.jsonl", root), ("child.jsonl", child)]:
        for index, event in enumerate(records):
            result.append(path=name, event=event, at=index / 10)
    return result


@pytest.mark.parametrize("fork", [False, True])
def test_child_owns_only_new_actions(fork):
    traces = collect(*fixture(fork=fork))
    child = traces.resolve_request({"client_metadata": {"thread_id": "child"}, "model": "test-model"})
    assert child.spawn.parent_id == "root" and child.spawn.call_id == "spawn"
    assert child.session.inherited_call_ids == (["old"] if fork else [])
    outputs = traces.outputs("child", skill_paths={})
    assert [(out.call_id, out.text) for out in outputs] == [("new", "New child's read")]
    assert outputs[0].at == child.own[-1].at


def test_header_claim_without_canonical_parent_link_cannot_forward():
    root, child = fixture()
    traces = collect(root[:-1], child)
    with pytest.raises(PendingSession, match="parent spawn"):
        traces.resolve_request({"client_metadata": {"thread_id": "child"}, "model": "test-model"})
    traces.append(path="root.jsonl", event=root[-1], at=1)
    assert (
        traces.resolve_request({"client_metadata": {"thread_id": "child"}, "model": "test-model"}).session.id
        == "child"
    )


@pytest.mark.parametrize(
    "mutation", ["owner", "copied", "late-copy", "boundary", "fork", "model", "duplicate", "root"]
)
def test_broken_lineage_is_not_accepted(mutation):
    root, child = fixture(fork=True)
    if mutation == "owner":
        child[0]["payload"]["parent_thread_id"] = "child"
    if mutation == "copied":
        child[2]["payload"]["model"] = "altered-copy"
    if mutation == "late-copy":
        child[5]["payload"]["output"] = "Changed copied output after the parent context"
    if mutation == "boundary":
        child[0]["payload"].pop("subagent_history_start_ordinal")
    if mutation == "fork":
        root[-2]["payload"]["arguments"] = '{"fork_context":false}'
    if mutation == "model":
        child[-4]["payload"]["model"] = "other"
    if mutation == "duplicate":
        root += [copy.deepcopy(root[-1])]
    if mutation == "root":
        root[0]["payload"]["source"] = "interactive"
    traces = collect(root, child)
    with pytest.raises(EvidenceError):
        traces.session("child")


def test_proven_model_selection_is_a_distinct_boundary():
    traces = collect(*fixture(selected_model="other"))
    with pytest.raises(ModelRouteBoundary):
        traces.resolve_request({"client_metadata": {"thread_id": "child"}, "model": "other"})
    assert traces.session("root").session.model == "test-model"


def test_forged_request_model_does_not_change_session_identity():
    traces = collect(*fixture())
    with pytest.raises(EvidenceError, match="Request model"):
        traces.resolve_request({"client_metadata": {"thread_id": "root"}, "model": "other"})


def test_reused_or_absent_session_ownership_is_rejected():
    root, child = fixture()
    traces = collect(root, child)
    with pytest.raises(EvidenceError, match="same session"):
        traces.append(path="another.jsonl", event=root[0], at=2)
    with pytest.raises(EvidenceError, match="exactly one"):
        traces.select_root({"type": "thread.started", "thread_id": "child"})
    with pytest.raises(PendingSession):
        traces.session("unknown")


def test_rollout_watcher_waits_for_complete_lines_and_drains_once(tmp_path):
    from environments.skill_trigger.codex_watch import Rollouts

    watcher = Rollouts(tmp_path)
    path = tmp_path / "session.jsonl"
    path.write_bytes(b'{"type":"session_meta"')
    assert watcher.scan() == []
    with path.open("ab") as file:
        file.write(b'}\n{"type":"turn_context"}\n')
    records = watcher.scan(final=True)
    assert [record["sequence"] for record in records] == [0, 1]
    assert [record["event"]["type"] for record in records] == ["session_meta", "turn_context"]
    assert watcher.scan(final=True) == []


@pytest.mark.parametrize("mutation", ["truncate", "modify", "replace", "delete", "partial"])
def test_rollout_watcher_does_not_hide_incomplete_evidence(tmp_path, mutation):
    from environments.skill_trigger.codex_watch import Rollouts

    path = tmp_path / "session.jsonl"
    path.write_text('{"type":"first"}\n')
    watcher = Rollouts(tmp_path)
    assert len(watcher.scan()) == 1
    if mutation == "truncate":
        path.write_text("")
    if mutation == "modify":
        path.write_text('{"type":"other"}\n')
    if mutation == "replace":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    if mutation == "delete":
        path.unlink()
    if mutation == "partial":
        with path.open("a") as file:
            file.write('{"unfinished":')
    with pytest.raises(ValueError):
        watcher.scan(final=True)


def test_rollout_watcher_rejects_symlink(tmp_path):
    from environments.skill_trigger.codex_watch import Rollouts

    target = tmp_path / "target"
    target.write_text("{}\n")
    (tmp_path / "session.jsonl").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        Rollouts(tmp_path).scan()


async def test_live_rollout_watcher_drains_before_stop(tmp_path):
    import asyncio
    import signal
    import sys
    from pathlib import Path

    from environments.skill_trigger import codex_watch

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(codex_watch.__file__)),
        str(tmp_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        ready = json.loads(await asyncio.wait_for(process.stdout.readline(), 5))
        assert ready == {"ready": True, "pid": process.pid}
        (tmp_path / "rollout.jsonl").write_text('{"type":"session_meta"}\n')
        record = json.loads(await asyncio.wait_for(process.stdout.readline(), 5))
        assert record == {"path": "rollout.jsonl", "sequence": 0, "event": {"type": "session_meta"}}
        with (tmp_path / "rollout.jsonl").open("a") as file:
            file.write('{"type":"turn_context"}\n')
        process.send_signal(signal.SIGTERM)
        tail, errors = await asyncio.wait_for(process.communicate(), 5)
        assert process.returncode == 0, errors.decode()
        assert [json.loads(line) for line in tail.splitlines()] == [
            {"path": "rollout.jsonl", "sequence": 1, "event": {"type": "turn_context"}},
            {"rollouts_stopped": True},
        ]
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
