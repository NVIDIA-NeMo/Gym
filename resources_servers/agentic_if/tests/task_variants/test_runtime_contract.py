# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical rows are explicit, literal, and validated before resource allocation."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from resources_servers.agentic_if.task_variants import runtime


CHECK = {"id": "same", "kind": "judge", "rubric": "Be concise."}
EVENT = {"id": "same", "action": "fail", "text": "Command timed out.", "match": {"tool": "bash"}}


def base_row() -> dict:
    return {
        "instance_id": "public-example",
        "image_name": "example/task:test",
        "workdir": "/app",
        "responses_create_params": {
            "input": [
                {"role": "system", "content": "Literal ${tool:read} and {value}."},
                {"role": "user", "content": "Repair the example."},
            ]
        },
    }


def spec(**changes):
    return runtime.RuntimeSpec(harness="opencode", task_backend="swe", **changes)


def test_sealed_row_keeps_literal_messages_and_private_rubric_separate():
    source = base_row()
    before = deepcopy(source)
    config = spec(checks=[{"id": "end", "kind": "judge", "rubric": "End with DONE."}])
    row = runtime.seal_runtime_row(source, config)
    parsed = runtime.validate_runtime(row)
    assert row["responses_create_params"]["input"][0]["content"] == "Literal ${tool:read} and {value}."
    assert row["responses_create_params"]["input"][1]["content"] == "Repair the example."
    assert parsed.checks[0].rubric == "End with DONE."
    assert source == before


@pytest.mark.parametrize(
    "field,change",
    [
        ("another_field", {"user": "Another prompt"}),
        ("schema_version", 2),
        ("harness", "another"),
        ("task_backend", "another"),
        ("continuation", {"mode": "another"}),
    ],
)
def test_unknown_envelope_fields_and_values_are_rejected(field, change):
    row = runtime.seal_runtime_row(base_row(), spec())
    row["agentic_if"][field] = change
    with pytest.raises(ValidationError, match=field):
        runtime.validate_runtime(row)


def test_omitted_empty_and_single_tool_selection_have_distinct_meaning():
    assert runtime.validate_runtime(runtime.seal_runtime_row(base_row(), spec())).tools.tool_names is None
    empty = runtime.validate_runtime(runtime.seal_runtime_row(base_row(), spec(tools={"tool_names": {}})))
    assert empty.tools.tool_names == {}
    single = spec(tools={"tool_names": {"bash": "RunCommand"}, "tool_descriptions": {"bash": "Exactly {this}."}})
    parsed = runtime.validate_runtime(runtime.seal_runtime_row(base_row(), single))
    assert parsed.tools.tool_names == {"bash": "RunCommand"}
    assert parsed.tools.tool_descriptions == {"bash": "Exactly {this}."}


@pytest.mark.parametrize(
    "tools",
    [
        {"tool_names": {"read": "Same", "write": "Same"}},
        {"tool_names": {"bash": "Run Command"}},
        {"tool_names": {"bash": "Run"}, "tool_descriptions": {"read": "Hidden tool"}},
        {"tool_names": {"read": "Read"}, "tool_interfaces": {"read": "cw/pi@2/edit"}},
        {"tool_names": {"bash": "Run", "read_file": "ReadSource"}},
    ],
)
def test_invalid_tool_configuration_is_rejected(tools):
    with pytest.raises(ValidationError):
        spec(tools=tools)


@pytest.mark.parametrize(
    "edit",
    [
        "actor",
        "config",
        "unsealed",
        # Every backend field the task runs with is bound by the receipt, not only its image.
        pytest.param({"image_name": "example/wrong-image:test"}, id="image_name"),
        pytest.param({"agent_user": "root"}, id="agent_user"),
        pytest.param({"task_folder": "/other/task"}, id="task_folder"),
        pytest.param({"patch": "different reference patch"}, id="patch"),
    ],
)
def test_edited_or_unsealed_inputs_are_rejected(edit):
    source = base_row() | {"agent_user": "agent", "task_folder": "/data/task", "patch": "original reference patch"}
    row = runtime.seal_runtime_row(source, spec())
    if edit == "actor":
        row["responses_create_params"]["input"][1]["content"] = "Different instruction"
    elif edit == "config":
        row["agentic_if"]["tools"]["tool_names"] = {"bash": "Run"}
    elif edit == "unsealed":
        row = {**source, "agentic_if": spec().model_dump(mode="json")}
    else:
        row |= edit
    with pytest.raises(ValueError, match="receipt"):
        runtime.validate_runtime(row)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"checks": [CHECK, CHECK]}, "check.*unique"),
        ({"events": [EVENT, EVENT]}, "event.*unique"),
        ({"checks": [CHECK | {"scope": {"activation": "event", "event_id": "not-defined"}}]}, "event"),
        ({"checks": [CHECK | {"scope": {"activation": "continuation_start"}}]}, "continuation"),
        ({"tools": {"tool_names": {"bash": "Run"}}, "events": [EVENT | {"match": {"tool": "read"}}]}, "exposed"),
        (
            {"events": [EVENT | {"match": {"tool": "bash", "arg": {"field": "not_command", "regex": ".*"}}}]},
            "argument",
        ),
        # A failure is the only planted event; any other action is rejected, never ignored.
        ({"events": [EVENT | {"action": "replace"}]}, "action"),
        ({"events": [EVENT | {"action": "append"}]}, "action"),
    ],
)
def test_inconsistent_configuration_is_rejected(changes, match):
    with pytest.raises(ValidationError, match=match):
        spec(**changes)


GUARD = {"edit": {"unread": "Read {{path}} first.", "changed": "Read {{path}} again."}}


@pytest.mark.parametrize(
    "tools,match",
    [
        ({"read_first": {}}, "read_first requires non-empty tool/case objects"),
        ({"read_first": {"edit": {}}}, "read_first requires non-empty tool/case objects"),
        ({"read_first": {"edit": {"unread": "  "}}}, "read_first messages must not be blank"),
        ({"read_first": {"edit": {"unread": "Read {{file}} first."}}}, "read_first messages support only the"),
        # the guard can only be enforced with the read tool and each tool it guards exposed
        (
            {"tool_names": {"edit": "edit", "write": "write"}, "read_first": GUARD},
            r"exposed built-in tools: \['read'\]",
        ),
        (
            {"tool_names": {"read": "read", "bash": "bash"}, "read_first": {"write": GUARD["edit"]}},
            r"exposed built-in tools: \['write'\]",
        ),
    ],
)
def test_read_first_guards_are_validated_before_allocation(tools, match):
    with pytest.raises(ValidationError, match=match):
        spec(tools=tools)


def test_a_read_first_guard_keeps_its_literal_placeholder():
    tools = {"tool_names": {"read": "read", "edit": "edit", "write": "write"}, "read_first": GUARD}
    parsed = runtime.validate_runtime(runtime.seal_runtime_row(base_row(), spec(tools=tools)))
    assert parsed.tools.read_first.root == GUARD


def test_package_task_references_are_rejected():
    row = base_row() | {"task_name": "example", "task_ref": "sha256:abc", "dataset_ref": "dataset"}
    with pytest.raises(ValueError, match="task_ref"):
        runtime.seal_runtime_row(row, spec())


def test_tb21_row_keeps_its_backend_task_name_and_agent_user():
    source = {
        "task_name": "example-terminal",
        "docker_image": "example/task:test",
        "agent_user": "agent",
        "responses_create_params": base_row()["responses_create_params"],
    }
    sealed = runtime.seal_runtime_row(source, runtime.RuntimeSpec(harness="opencode", task_backend="tb21"))
    assert runtime.validate_runtime(sealed).task_backend == "tb21"
    assert sealed["agent_user"] == "agent"


def continuation(**changes):
    from resources_servers.agentic_if.task_variants.builder import digest
    from resources_servers.agentic_if.task_variants.tasks import backend_task_digest

    return {
        "mode": "restore",
        "prefix_export": {
            "info": {"id": "session"},
            "messages": [
                {"info": {"id": "u", "role": "user"}, "parts": [{"type": "text", "text": "Original task"}]},
                {"info": {"id": "m", "role": "assistant"}, "parts": []},
            ],
        },
        "workspace_diff": "",
        "source_task_sha256": backend_task_digest(base_row()),
        "source_tools_sha256": digest(spec().tools.model_dump(mode="json")),
        **changes,
    }


def test_continuation_is_bound_to_task_and_tools_without_duplicating_messages():
    config = spec(continuation=continuation())
    row = runtime.seal_runtime_row(base_row(), config)
    parsed = runtime.validate_runtime(row)
    assert parsed.continuation.mode == "restore"
    assert row["responses_create_params"]["input"] == base_row()["responses_create_params"]["input"]
    with pytest.raises(ValueError, match="source task receipt"):
        runtime.seal_runtime_row(base_row(), spec(continuation=continuation(source_task_sha256="0" * 64)))
    with pytest.raises(ValueError, match="source tools receipt"):
        runtime.seal_runtime_row(base_row(), spec(continuation=continuation(source_tools_sha256="0" * 64)))


@pytest.mark.parametrize(
    "prefix",
    [
        {},
        {"info": {"id": "s"}, "messages": []},
        {"info": {"id": "s"}, "messages": [{"info": {"role": "user"}, "parts": []}]},
        {
            "info": {"id": "s"},
            "messages": [
                {
                    "info": {"role": "assistant"},
                    "parts": [
                        {"type": "tool", "tool": "bash", "state": {"status": "running"}},
                    ],
                }
            ],
        },
    ],
)
def test_unusable_continuation_prefix_is_rejected_before_allocation(prefix):
    with pytest.raises(ValidationError, match="prefix"):
        spec(continuation=continuation(prefix_export=prefix))


@pytest.mark.parametrize(
    "input",
    [
        [],
        [{"role": "assistant", "content": "Not an initial prompt"}],
        [{"role": "user", "content": [{"type": "image", "url": "private"}]}],
    ],
)
def test_unsupported_actor_input_is_not_silently_dropped(input):
    row = base_row()
    row["responses_create_params"]["input"] = input
    with pytest.raises(ValueError):
        runtime.seal_runtime_row(row, spec())


def test_omitted_and_empty_system_prompts_are_distinct_valid_inputs():
    row = base_row()
    row["responses_create_params"]["input"].pop(0)
    omitted = runtime.seal_runtime_row(row, runtime.RuntimeSpec(harness="opencode", task_backend="swe"))
    runtime.validate_runtime(omitted)
    row["responses_create_params"]["input"].insert(0, {"role": "system", "content": ""})
    empty = runtime.seal_runtime_row(row, runtime.RuntimeSpec(harness="opencode", task_backend="swe"))
    runtime.validate_runtime(empty)
    assert omitted != empty
    assert empty["responses_create_params"]["input"][0] == {"role": "system", "content": ""}


def planted_continuation():
    """A restore continuation of a prefix that read a file, with a read failure planted for the live run."""
    prefix = {
        "info": {"id": "session"},
        "messages": [
            {"info": {"id": "u", "role": "user"}, "parts": [{"type": "text", "text": "Original task"}]},
            {
                "info": {"id": "m", "role": "assistant"},
                "parts": [
                    {
                        "type": "tool",
                        "tool": "read",
                        "callID": "read-one",
                        "state": {"status": "completed", "input": {"filePath": "/app/file"}, "output": "contents"},
                    }
                ],
            },
        ],
    }
    return {
        "continuation": continuation(prefix_export=prefix),
        "events": [{"id": "timeout", "match": {"tool": "read"}, "action": "fail", "text": "Read timed out."}],
    }


def test_planted_continuation_seals_with_an_empty_prefix_event_list():
    row = runtime.seal_runtime_row(base_row(), spec(**planted_continuation()))
    assert runtime.validate_runtime(row).events[0].id == "timeout"
    # Sealed receipts hash the continuation with this field.
    assert row["agentic_if"]["continuation"]["prefix_events"] == []


def test_prefix_event_receipts_are_rejected_not_ignored():
    data = planted_continuation()
    data["continuation"]["prefix_events"] = [
        {"id": "timeout", "step_index": 2, "call_id": "read-one", "invocation_id": "session", "details": {}}
    ]
    with pytest.raises(ValidationError, match="prefix_events must be empty"):
        spec(**data)


def test_gym_collation_of_a_sealed_flat_row_keeps_its_content_identity():
    from nemo_gym.task_materialization import materialize_task
    from resources_servers.agentic_if.task_variants.builder import digest
    from resources_servers.agentic_if.task_variants.materialization import sealed_task_input

    sealed = runtime.seal_runtime_row(base_row(), spec())
    collated = materialize_task({"task_id": digest(sealed)} | sealed, taskset="agentic_if:tests")
    # Collation takes the row's own id as the identity, ahead of instance_id, and keeps the field in task_input.
    assert collated["task_id"]["task_id"] == digest(sealed) == collated["task_input"]["task_id"]
    assert sealed_task_input(collated) == sealed
    runtime.validate_runtime(sealed_task_input(collated))
    forged = deepcopy(collated)
    forged["task_input"]["task_id"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        sealed_task_input(forged)
