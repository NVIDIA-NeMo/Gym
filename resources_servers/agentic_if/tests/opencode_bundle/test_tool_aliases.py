# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tools a row renames: the compiled configuration, and ``if-boundary.mjs`` offering them under their aliases.

The model sees each renamed tool under its alias, in tool definitions, history, ``tool_choice`` and compaction calls;
OpenCode runs the built-in tool of a call made under an alias; the records keep what the model saw and wrote.
"""

import json
import re
from copy import deepcopy
from pathlib import Path

import pytest

from nemo_gym.models.token_capture import assistant_fingerprint
from resources_servers.agentic_if.opencode_bundle import UnsupportedFeatureError, bundle_files
from resources_servers.agentic_if.opencode_bundle.boundary import compile_boundary
from resources_servers.agentic_if.opencode_bundle.profile import compile_profile
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec

from .harness import call, needs_node, run_boundary, sse_wire, upstream
from .specs import ALIASES, SPECS, aliased_spec


SYSTEM = "Literal row system prompt: work with the four tools you are given."
SELECTED = ["bash", "edit", "read", "write"]
# The names the model sees, in OpenCode's tool order, and the sorted list a refusal shows.
VISIBLE = ["shell", "edit", "view_file", "create_file"]
AVAILABLE = "create_file, edit, shell, view_file"
OK = upstream(
    json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}).encode()
)


def opencode_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"built-in {name}",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            },
        }
        for name in SELECTED
    ]


def opencode_request(messages: list[dict] | None = None, **extra: object) -> dict:
    """An actor request as OpenCode sends it: built-in tool IDs, in its history too."""
    return {
        "model": "dummy_model",
        "messages": messages or [{"role": "user", "content": "Fix it"}],
        "tools": opencode_tools(),
        "tool_choice": "auto",
        **extra,
    }


def tool_call(identity: str, name: str, arguments: str = '{"command": "ls"}') -> dict:
    return {"id": identity, "type": "function", "function": {"name": name, "arguments": arguments}}


def json_reply(*calls: dict, content: str | None = None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = list(calls)
    return {
        "id": "r",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}],
    }


def replying(body: dict) -> dict:
    return upstream(json.dumps(body).encode())


# ------------------------------------------------------------------------------------------------ compiled configuration


def test_a_row_that_renames_tools_compiles_its_aliases_last_and_keeps_built_in_ids_elsewhere() -> None:
    spec = aliased_spec()
    boundary = compile_boundary(spec)
    assert list(boundary)[-1] == "tool_aliases" and boundary["tool_aliases"] == ALIASES
    # Denies stay per built-in tool: OpenCode's IDs.
    assert "bash" not in boundary["deny"] and "glob" in boundary["deny"]
    profile = compile_profile(spec)
    assert profile["visible_tools"] == {"bash": "shell", "edit": "edit", "read": "view_file", "write": "create_file"}
    assert profile["check_aliases"] is True


@pytest.mark.parametrize("name", sorted(SPECS))
def test_rows_without_aliases_compile_without_the_new_key(name: str) -> None:
    # The keys and their order are evidence format (boundary_sha256); the profile bytes are pinned by the goldens.
    boundary = compile_boundary(SPECS[name]())
    assert list(boundary) == [
        "boundary_version",
        "provider_id",
        "tool_names",
        "targets",
        "deny",
        "descriptions",
        "parameter_descriptions",
        "historical_rejected_calls",
    ]


def test_an_interface_description_describes_a_renamed_row_s_tool() -> None:
    tools = aliased_spec().tools.model_dump(mode="json")
    del tools["tool_descriptions"]["read"]
    tools["tool_interfaces"] = {"read": "cw/zed@3/read"}
    compile_boundary(RuntimeSpec(harness="opencode", task_backend="swe", tools=tools))


def test_a_renamed_row_without_a_description_for_every_tool_is_refused() -> None:
    # OpenCode's own descriptions name the built-in tools.
    tools = {"tool_names": {"bash": "shell", "read": "read"}, "tool_descriptions": {"bash": "Run."}}
    spec = RuntimeSpec(harness="opencode", task_backend="swe", tools=tools)
    for compile_ in (compile_profile, compile_boundary, bundle_files):
        with pytest.raises(UnsupportedFeatureError, match=r"must describe every tool it selects .*\['read'\]"):
            compile_(spec)


# ------------------------------------------------------------------------------------------------ the boundary plugin


@needs_node
def test_requests_offer_renamed_tools_under_their_aliases_in_definitions_history_and_tool_choice(
    tmp_path: Path,
) -> None:
    history = [
        {"role": "user", "content": "Fix it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call("call_ls", "bash"), tool_call("call_cat", "read", '{"command": "a.py"}')],
        },
        {"role": "tool", "tool_call_id": "call_ls", "name": "bash", "content": "a.py\n"},
        {"role": "tool", "tool_call_id": "call_cat", "content": "x = 1"},
    ]
    body = opencode_request(history, tool_choice={"type": "function", "function": {"name": "bash"}})
    run = run_boundary(tmp_path, boundary=compile_boundary(aliased_spec()), calls=[call(body, OK)], system=SYSTEM)
    sent = json.loads(run.calls[0]["sent"])
    tools = [tool["function"] for tool in sent["tools"]]
    assert [tool["name"] for tool in tools] == VISIBLE
    assert [tool["description"] for tool in tools] == [f"Row description of {name}." for name in VISIBLE]
    assert [call["function"]["name"] for call in sent["messages"][2]["tool_calls"]] == ["shell", "view_file"]
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "shell"}}
    assert all("name" not in message for message in sent["messages"] if message["role"] == "tool")


@needs_node
def test_a_reply_reaches_opencode_with_the_tools_it_runs_and_is_recorded_as_the_model_wrote_it(tmp_path: Path) -> None:
    reply = json_reply(
        tool_call("c-alias", "shell"),
        tool_call("c-case", "Shell"),
        tool_call("c-kept", "Edit"),
        tool_call("c-hidden", "bash"),
        tool_call("c-unknown", "missing"),
    )
    run = run_boundary(
        tmp_path,
        boundary=compile_boundary(aliased_spec()),
        calls=[call(opencode_request(), replying(reply))],
        system=SYSTEM,
    )
    # An alias runs its tool. An alias in other capitals reaches OpenCode as the tool's ID in capitals, for OpenCode's
    # own lowercase repair, as a tool that keeps its name does; a hidden ID and an unknown name pass unchanged
    # (the profile plugin refuses the first, OpenCode's invalid tool answers the second).
    forwarded = deepcopy(reply)
    for entry, name in zip(
        forwarded["choices"][0]["message"]["tool_calls"], ["bash", "BASH", "Edit", "bash", "missing"]
    ):
        entry["function"]["name"] = name
    assert json.loads(run.returned(0)) == forwarded
    observation = run.records.observations[0]
    assert observation["response"] == reply
    assert observation["execution_tools"] == {
        "shell": "bash",
        "edit": "edit",
        "view_file": "read",
        "create_file": "write",
        "Shell": "bash",
        "Edit": "edit",
    }


@needs_node
def test_a_streamed_call_reaches_opencode_under_its_tool_in_one_delta(tmp_path: Path) -> None:
    def delta(**fields: object) -> dict:
        return {"id": "s", "choices": [{"index": 0, "delta": fields}]}

    def fragment(**function: str) -> dict:
        return delta(tool_calls=[{"index": 0, "function": function}])

    events = [
        delta(role="assistant", tool_calls=[{"index": 0, "id": "c1", "type": "function", "function": {"name": "sh"}}]),
        fragment(name="ell"),
        fragment(arguments='{"command": '),
        fragment(arguments='"ls"}'),
        {"id": "s", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]
    run = run_boundary(
        tmp_path,
        boundary=compile_boundary(aliased_spec()),
        calls=[call(opencode_request(stream=True), upstream(sse_wire(events), content_type="text/event-stream"))],
        system=SYSTEM,
    )
    chunks = [
        json.loads(frame[len("data: ") :])
        for frame in run.returned(0).decode().split("\n\n")
        if frame.startswith("data: {")
    ]
    sent = [entry for chunk in chunks for entry in chunk["choices"][0]["delta"].get("tool_calls", [])]
    assert sent == [
        {"index": 0, "function": {"name": "bash", "arguments": '{"command": '}, "id": "c1", "type": "function"},
        {"index": 0, "function": {"arguments": '"ls"}'}},
    ]
    recorded = run.records.observations[0]["response"]["choices"][0]["message"]["tool_calls"]
    assert recorded == [
        {"function": {"name": "shell", "arguments": '{"command": "ls"}'}, "id": "c1", "type": "function"}
    ]


@needs_node
def test_rejected_and_completed_calls_are_shown_as_the_model_wrote_them_with_the_aliases_offered(
    tmp_path: Path,
) -> None:
    first = json_reply(
        tool_call("c-missing", "missing", '{"x": 1}'), tool_call("c-shell", "shell"), content="Checking."
    )
    recovery = "Model tried to call unavailable tool 'missing'. Available tools: bash, edit, read, write."
    # OpenCode's next request: its invalid tool answered the unknown name, and the shell call ran as bash.
    history = [
        {"role": "user", "content": "Fix it"},
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [
                tool_call("c-missing", "invalid", json.dumps({"tool": "missing", "error": recovery})),
                tool_call("c-shell", "bash", '{"command":"ls"}'),
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c-missing",
            "content": "The arguments provided to the tool are invalid: " + recovery,
        },
        {"role": "tool", "tool_call_id": "c-shell", "content": "a.py\n"},
    ]
    run = run_boundary(
        tmp_path,
        boundary=compile_boundary(aliased_spec()),
        calls=[
            call(opencode_request(), replying(first)),
            call(opencode_request(history), OK, assistant="msg-2"),
        ],
        system=SYSTEM,
    )
    request, sent = (json.loads(entry["sent"]) for entry in run.calls)
    reply = first["choices"][0]["message"]
    # The model's own reply follows the context it was given, as token capture needs to resolve the continuation.
    assert sent["messages"][: len(request["messages"]) + 1] == request["messages"] + [reply]
    assert sent["tools"] == request["tools"]
    assert assistant_fingerprint(sent["messages"]) == assistant_fingerprint(request["messages"] + [reply])
    assert (
        sent["messages"][3]["content"]
        == f"missing is unavailable or its arguments are invalid. Available tools: {AVAILABLE}."
    )
    # No renamed tool's built-in ID reaches the model, in the tools or the conversation.
    assert not re.search(
        r"\b(?:bash|read|write)\b", json.dumps({"tools": sent["tools"], "messages": sent["messages"]})
    )


@needs_node
def test_compaction_history_names_renamed_tools_under_their_aliases(tmp_path: Path) -> None:
    body = {
        "messages": [
            {"role": "system", "content": "Summarize"},
            {"role": "user", "content": "Fix it"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call("c1", "bash"), tool_call("c2", "edit", '{"command": "a.py"}')],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "a.py\n"},
            {"role": "tool", "tool_call_id": "c2", "content": "Edit applied successfully."},
            {"role": "user", "content": "What did we do so far?"},
        ]
    }
    run = run_boundary(
        tmp_path,
        boundary=compile_boundary(aliased_spec()),
        calls=[call(body, OK, agent="compaction", assistant="msg-s")],
        system=SYSTEM,
    )
    # Only the renamed tool's name changes: compaction keeps OpenCode's own prompt and history otherwise.
    expected = deepcopy(body)
    expected["messages"][2]["tool_calls"][0]["function"]["name"] = "shell"
    assert json.loads(run.calls[0]["sent"]) == expected
    observation = run.records.observations[0]
    assert observation["source"] == "compaction" and observation["request"] == expected


@needs_node
def test_a_row_that_renames_tools_needs_its_system_text(tmp_path: Path) -> None:
    run = run_boundary(tmp_path, boundary=compile_boundary(aliased_spec()), calls=[call(opencode_request(), OK)])
    assert run.records.configurations[0] == {
        "kind": "config_error",
        "message": "a row that renames tools needs its system text: OpenCode's own system prompt names the tools",
    }
    assert run.calls[0]["sent"] is None and "was not configured" in run.calls[0]["error"]["message"]
