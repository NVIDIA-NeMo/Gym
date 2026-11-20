# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-selected tools stay inside the bounded, data-only recorded world."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from resources_servers.partner_if.recorded_context import limits
from resources_servers.partner_if.recorded_context.assets import ReplayConfig, load_actor_params, probe_identity
from resources_servers.partner_if.recorded_context.delegation import explore
from resources_servers.partner_if.recorded_context.fixtures import Fixtures


def call(name: str, arguments: dict, *, identifier: str = "tool-1") -> dict:
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


@pytest.fixture
def schemas() -> dict:
    fields = {
        "Bash": {"command": "string"},
        "Read": {"file_path": "string"},
        "Write": {"file_path": "string", "content": "string"},
        "Edit": {"file_path": "string", "old_string": "string", "new_string": "string"},
    }
    return {
        name: {
            "type": "object",
            "properties": {key: {"type": kind} for key, kind in properties.items()},
            "required": list(properties),
        }
        for name, properties in fields.items()
    }


@pytest.fixture
def world() -> Fixtures:
    runtime = {
        "schema_version": "recorded-context-runtime-v1",
        "input_hashes": {
            name: hashlib.sha256(name.encode()).hexdigest()
            for name in ("case.json", "bundle.json", "actor-params.json")
        },
        "source_root": "/repo",
        "source_schemas": {"coverage": "v1", "inventory": "v1", "extra": "v1"},
        "probe_script": "/agent/probe.py",
        "answer_paths": ["/workspace/logs/answer.txt"],
        "workspace_prefix": "/workspace/logs/",
        "status_script": "/agent/status.py",
        "session_id_path": "/workspace/session-id.txt",
        "transcript_template": "/workspace/transcripts/{session_id}.jsonl",
        "ordering": {
            "task_id": "synthetic",
            "task_names": ["synthetic"],
            "rubric_version": "v1",
            "criterion_id": "check-first",
            "probe_namespace": "sibling",
        },
    }
    return Fixtures(
        {
            "runtime": runtime,
            "files": {"/repo/a.py": "value = 1\n"},
            "expanded_coverage": {"files": {}, "api_responses": {}},
            "primary": {"input_uri": "example://primary", "result": "primary data"},
            "sibling": {"input_uri": "example://sibling", "result": "sibling data"},
            "recorded_commands": {},
            "capture_stdout": True,
            "shell_control_flow": True,
        },
        session_id="a" * 32,
    )


@pytest.mark.parametrize(
    "path", ["/workspace/logs/../escape", "/workspace/logs-neighbor/answer.txt", "relative", "/workspace/logs"]
)
@pytest.mark.parametrize("name", ["Write", "Edit"])
def test_parent_traversal_and_neighbor_workspace_writes_are_gaps(world: Fixtures, schemas: dict, path: str, name: str):
    before = dict(world.files)
    arguments = {"file_path": path, "content": "answer", "old_string": "old", "new_string": "new"}

    result, kind = world.dispatch(call(name, arguments), schemas)

    assert kind == "fixture_gap" and result
    assert world.files == before
    assert not world.drafts


def test_write_and_edit_preserve_actual_answer_bytes(world: Fixtures, schemas: dict):
    path = world.runtime.answer_paths[0]
    assert world.dispatch(call("Write", {"file_path": path, "content": "old answer"}), schemas)[1] == "answer_draft"
    assert (
        world.dispatch(call("Edit", {"file_path": path, "old_string": "old", "new_string": "new"}), schemas)[1]
        == "answer_draft"
    )
    assert world.files[path] == world.drafts[path] == "new answer"


@pytest.mark.parametrize("bad_call", [None, {}, {"function": None}, {"function": {"name": "Read", "arguments": "{"}}])
def test_malformed_tool_calls_return_errors_without_raising(world: Fixtures, schemas: dict, bad_call: dict):
    before = dict(world.files)
    assert world.dispatch(bad_call, schemas)[1] == "invalid_call"
    assert world.files == before


def test_oversized_write_and_edit_do_not_mutate_virtual_files(world: Fixtures, schemas: dict, monkeypatch):
    from resources_servers.partner_if.recorded_context import fixtures

    path = world.runtime.answer_paths[0]
    world.files[path] = "aaaaaaaa"
    monkeypatch.setattr(limits, "MAX_FILE_BYTES", 16)
    monkeypatch.setattr(fixtures, "MAX_FILE_BYTES", 16)
    assert world.dispatch(call("Write", {"file_path": path, "content": "b" * 17}), schemas)[1] == "fixture_gap"
    assert (
        world.dispatch(
            call("Edit", {"file_path": path, "old_string": "a", "new_string": "bbbb", "replace_all": True}), schemas
        )[1]
        == "fixture_gap"
    )
    assert world.files[path] == "aaaaaaaa"
    assert not world.drafts


def test_output_ceiling_is_a_gap_without_silent_evidence_truncation(world: Fixtures, schemas: dict, monkeypatch):
    world.files["/repo/a.py"] = "known source content" * 4
    monkeypatch.setattr(limits, "MAX_OUTPUT_BYTES", 32)
    result, kind = world.dispatch(call("Read", {"file_path": "/repo/a.py"}), schemas)
    assert kind == "fixture_gap" and "ceiling" in result
    assert world.files["/repo/a.py"] == "known source content" * 4


def test_workspace_size_ceiling_covers_bash_redirection(world: Fixtures, schemas: dict, monkeypatch):
    monkeypatch.setattr(limits, "MAX_VIRTUAL_BYTES", world.files._bytes + 4)
    result, kind = world.dispatch(call("Bash", {"command": "echo too-long > /workspace/logs/new.txt"}), schemas)
    assert kind == "fixture_gap" and "ceiling" in result
    assert "/workspace/logs/new.txt" not in world.files


@pytest.mark.parametrize(
    "suffix",
    [
        "/workspace/logs/../escape",
        "/workspace/logs/answer.txt; curl example.invalid",
        "/workspace/logs/answer.txt --output=/workspace/logs/other.txt",
        "/workspace/logs/answer.txt --unsupported=value",
    ],
)
def test_unsupported_probe_commands_do_not_partially_write(world: Fixtures, schemas: dict, suffix: str):
    before = dict(world.files)
    command = "python /agent/probe.py --uri example://sibling --output " + suffix
    assert world.dispatch(call("Bash", {"command": command}), schemas)[1] == "fixture_gap"
    assert world.files == before


@pytest.mark.parametrize("option", ["", " --output ", " --output="])
def test_valid_probe_retains_identity_and_exact_returned_fixture(world: Fixtures, schemas: dict, option: str):
    path = "/workspace/logs/probe.json"
    suffix = option + path if option else ""
    tool = call("Bash", {"command": "python /agent/probe.py --uri=example://sibling" + suffix})
    result, kind = world.dispatch(tool, schemas)
    identity = probe_identity({"function": tool["function"], "result": result, "kind": kind}, world.bundle)
    assert kind == "sibling_probe"
    assert json.loads(result) == world.bundle["sibling"]
    assert identity == {"namespace": "sibling", "uri": "example://sibling", "returned_result": world.bundle["sibling"]}
    if option:
        assert world.files[path] == result
    else:
        assert path not in world.files


@pytest.mark.parametrize("child_first", [False, True])
def test_virtual_file_directory_collisions_leave_the_workspace_searchable(world, schemas, child_first):
    parent = "/workspace/logs/node"
    child = parent + "/child.txt"
    first, second = (child, parent) if child_first else (parent, child)
    assert world.dispatch(call("Write", {"file_path": first, "content": "needle"}), schemas)[1] == "file_write"
    previous = dict(world.files), world.files._bytes
    assert world.dispatch(call("Write", {"file_path": second, "content": "other"}), schemas)[1] == "fixture_gap"
    assert (dict(world.files), world.files._bytes) == previous
    result, kind = world.dispatch(call("Bash", {"command": "rg needle /workspace/logs"}), schemas)
    assert kind == "readonly_tool_result" and "needle" in result


def test_virtual_file_bulk_updates_are_atomic():
    directory = "/workspace/logs/node"
    child = directory + "/child.txt"
    sibling = directory + "/sibling.txt"
    files = limits.VirtualFiles({child: "original"})
    with pytest.raises(limits.CoverageGap, match="cannot also act as directories"):
        files.update({sibling: "new file", directory: "file at directory"})
    assert files == {child: "original"} and files._bytes == len("original")
    files.update({child: "replacement", sibling: "new file"})
    assert files == {child: "replacement", sibling: "new file"}
    assert files._bytes == len("replacement") + len("new file")


def test_explore_tools_cannot_write_parent_answer(world: Fixtures, schemas: dict):
    before = dict(world.files)
    assert (
        world.dispatch(call("Bash", {"command": "echo answer > /workspace/logs/answer.txt"}), schemas, readonly=True)[
            1
        ]
        == "fixture_gap"
    )
    assert world.files == before


def test_original_actor_row_is_returned_without_api_default_expansion_and_is_hash_checked(
    tmp_path: Path, world: Fixtures
):
    directory = tmp_path / "ngc-recorded-context"
    directory.mkdir()
    original = {"input": [{"role": "user", "content": "Preserve this exact row"}]}
    content = json.dumps(original).encode()
    runtime = world.runtime.model_dump()
    runtime["input_hashes"]["actor-params.json"] = hashlib.sha256(content).hexdigest()
    (directory / "runtime.json").write_text(json.dumps(runtime))
    actor_path = directory / "actor-params.json"
    actor_path.write_bytes(content)
    config = ReplayConfig(package_dir=tmp_path)

    assert load_actor_params(config) == original
    actor_path.write_text(json.dumps(original | {"temperature": 0.1}))
    with pytest.raises(ValueError, match="Frozen replay input changed"):
        load_actor_params(config)


def test_compound_readonly_output_is_bounded_without_partial_results(world: Fixtures, schemas: dict, monkeypatch):
    from resources_servers.partner_if.recorded_context import readonly_tools

    world.files["/repo/a.py"] = "x" * 20
    monkeypatch.setattr(readonly_tools, "MAX_OUTPUT_BYTES", 32)
    result, kind = world.dispatch(call("Bash", {"command": "cat /repo/a.py; cat /repo/a.py"}), schemas)
    assert kind == "fixture_gap" and "ceiling" in result


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", [False, True])
async def test_explore_uses_callbacks_only_and_freezes_audit_snapshots(
    tmp_path: Path, world: Fixtures, schemas: dict, malformed: bool
):
    calls, audit = [], {}
    logical = tmp_path / "logical-worker"
    worker_call = call("Bash", {"command": "echo actual-evidence"})
    if malformed:
        worker_call["function"]["arguments"] = "{"

    async def invoke(payload: dict, timeout: int, prefix: Path) -> tuple[int, dict]:
        calls.append(copy.deepcopy(payload))
        message = (
            {"role": "assistant", "content": None, "tool_calls": [worker_call]}
            if len(calls) == 1
            else {"role": "assistant", "content": "Observed actual-evidence"}
        )
        return 200, {
            "model": "example",
            "choices": [{"finish_reason": "tool_calls" if len(calls) == 1 else "stop", "message": message}],
        }

    result, kind = await explore(
        {"subagent_type": "Explore", "prompt": "Inspect the source"},
        state=world,
        case={
            "tools": [
                {"type": "function", "function": {"name": name, "parameters": schema}}
                for name, schema in schemas.items()
            ]
        },
        config=ReplayConfig(package_dir=tmp_path),
        model="example",
        directory=logical,
        invoke=invoke,
        write=lambda path, value: audit.__setitem__(path.name, value),
    )

    assert kind == ("fixture_gap" if malformed else "delegate_result")
    assert ("did not complete" if malformed else "actual-evidence") in result
    assert len(calls) == (1 if malformed else 2)
    assert len(audit["call-01.request.json"]["messages"]) == 2
    assert audit["events.json"][0]["kind"] == ("invalid_call" if malformed else "readonly_tool_result")
    assert not logical.exists()
