# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compiling ``profile.json``, ``plants.json`` and ``boundary.json`` from the RuntimeSpec."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from resources_servers.agentic_if.opencode_bundle import UnsupportedFeatureError, bundle_files, profile_sha256
from resources_servers.agentic_if.opencode_bundle.boundary import (
    OPENCODE_EXTRA_TOOLS,
    compile_boundary,
    parameter_targets,
    prefix_model_calls,
)
from resources_servers.agentic_if.opencode_bundle.profile import compile_plants, compile_profile
from resources_servers.agentic_if.task_variants.catalog import BUILTIN_TOOLS
from resources_servers.agentic_if.task_variants.interfaces import installed_interfaces
from resources_servers.agentic_if.task_variants.runtime import ContinuationSpec, RuntimeSpec

from .specs import SPECS, continuation_spec, private_like_spec, unrestricted_spec


# What the 00be1884 driver staged for each spec (fixtures/profile-goldens.json; derive_fixtures.py documents how).
GOLDENS = json.loads((Path(__file__).with_name("fixtures") / "profile-goldens.json").read_text())["specs"]


@pytest.mark.parametrize("name", sorted(SPECS))
def test_profile_plants_and_denies_are_byte_identical_to_the_environment_server_driver(name: str) -> None:
    spec, golden = SPECS[name](), GOLDENS[name]
    profile = compile_profile(spec)
    assert profile == golden["profile"]
    assert profile["profile_sha256"] == profile_sha256(profile)
    assert hashlib.sha256(json.dumps(profile).encode()).hexdigest() == golden["staged_sha256"]
    assert compile_plants(spec) == golden["plants"]
    assert compile_boundary(spec)["deny"] == golden["deny"]


def test_profile_keeps_execution_interfaces_and_drops_presentation() -> None:
    spec = private_like_spec()
    profile = compile_profile(spec)
    assert list(profile) == [
        "profile_version",
        "id",
        "tools",
        "read_first",
        "visible_tools",
        "check_aliases",
        "literal_errors",
        "profile_sha256",
    ]
    installed = installed_interfaces(spec.tools.tool_interfaces)
    assert set(profile["tools"]) == {"read", "edit", "bash", "task"}
    for tool, definition in profile["tools"].items():
        assert "description" not in definition
        assert definition["parameters"] == installed[tool]["parameters"]
    assert profile["visible_tools"] == {tool: tool for tool in spec.tools.tool_names}
    assert profile["check_aliases"] is False
    assert profile["literal_errors"] == ["Blocked: pushing is not allowed here."]
    assert profile["read_first"] == spec.tools.read_first.model_dump(mode="json")


def test_profile_of_an_unrestricted_row_has_no_selection_and_continuations_keep_their_hashed_keys() -> None:
    unrestricted = compile_profile(unrestricted_spec())
    assert "visible_tools" not in unrestricted and "read_first" not in unrestricted and unrestricted["tools"] == {}
    continuation = compile_profile(continuation_spec())
    assert continuation["prefix_receipts"] == [] and continuation["prefix_mode"] == "restore"


def test_plants_are_the_events_as_the_profile_plugin_reads_them() -> None:
    assert compile_plants(private_like_spec()) == [
        {
            "id": "push",
            "match": {"tool": "bash", "arg": {"field": "command", "regex": "^git push"}},
            "nth": 2,
            "action": "fail",
            "text": "Blocked: pushing is not allowed here.",
        }
    ]
    assert compile_plants(unrestricted_spec()) == []


def test_boundary_of_a_private_shaped_row() -> None:
    spec = private_like_spec()
    boundary = compile_boundary(spec)
    assert boundary["boundary_version"] == 1 and boundary["provider_id"] == "nemo_gym"
    assert boundary["tool_names"] == list(spec.tools.tool_names)
    assert boundary["targets"] == sorted(spec.tools.tool_names)
    assert boundary["deny"] == sorted((BUILTIN_TOOLS | OPENCODE_EXTRA_TOOLS) - set(spec.tools.tool_names))
    assert "apply_patch" in boundary["deny"] and "webfetch" in boundary["deny"]
    # Every selected tool's row description wins over its catalog interface's.
    assert boundary["descriptions"] == {tool: f"Row description of {tool}." for tool in spec.tools.tool_names}
    assert boundary["parameter_descriptions"] == {
        "read": [
            {"target": "/properties/path", "path": ["properties", "path"], "description": "Absolute path of the file."}
        ],
        "bash": [
            {"target": "/properties/command", "path": ["properties", "command"], "description": "One shell command."}
        ],
    }
    assert boundary["historical_rejected_calls"] == {}


def test_catalog_descriptions_apply_where_the_row_sets_none() -> None:
    spec = RuntimeSpec(
        harness="opencode",
        task_backend="swe",
        tools={
            "tool_names": {"read": "read", "bash": "bash"},
            "tool_interfaces": {"read": "cw/zed@3/read"},
            "tool_descriptions": {"bash": "Row bash."},
        },
    )
    descriptions = compile_boundary(spec)["descriptions"]
    assert descriptions == {
        "read": installed_interfaces({"read": "cw/zed@3/read"})["read"]["description"],
        "bash": "Row bash.",
    }


def test_boundary_of_an_unrestricted_row_keeps_the_opencode_tool_inventory() -> None:
    boundary = compile_boundary(unrestricted_spec())
    assert boundary["tool_names"] is None and boundary["deny"] == [] and boundary["targets"] == []
    assert boundary["historical_rejected_calls"] == {}


def test_boundary_carries_the_witnessed_rejected_calls_of_a_prefix() -> None:
    assert compile_boundary(continuation_spec())["historical_rejected_calls"] == {
        "call_bad": {
            "function": {"name": "Missing", "arguments": '{ "x": 1 }'},
            "available_names": ["bash", "edit", "read", "write"],
        }
    }


@pytest.mark.parametrize(
    "key, path",
    [
        ("command", ["properties", "command"]),
        ("/properties/a~1b/items/0", ["properties", "a/b", "items", "0"]),
        ("/$defs/x~0y", ["$defs", "x~y"]),
        ("/", [""]),
    ],
)
def test_parameter_targets_parse_names_and_json_pointers(key: str, path: list[str]) -> None:
    assert parameter_targets({key: "text"}) == [{"target": key, "path": path, "description": "text"}]


def test_parameter_targets_refuse_the_same_target_twice() -> None:
    with pytest.raises(ValueError, match="duplicate parameter target: /properties/command"):
        parameter_targets({"command": "a", "/properties/command": "b"})


def test_prefix_rejected_calls_need_captured_evidence() -> None:
    prefix = deepcopy(continuation_spec().continuation.prefix_export)
    state = prefix["messages"][1]["parts"][1]["state"]
    del state["metadata"]
    with pytest.raises(ValueError, match="requires captured call evidence"):
        prefix_model_calls(
            ContinuationSpec(
                mode="restore",
                prefix_export=prefix,
                workspace_diff="",
                source_task_sha256="a" * 64,
                source_tools_sha256="b" * 64,
            )
        )
    state["metadata"] = {
        "nemo_gym_model_call": {"function": {"name": "Missing", "arguments": {}}, "available_names": []}
    }
    with pytest.raises(ValueError, match="invalid or duplicate"):
        prefix_model_calls(
            ContinuationSpec(
                mode="restore",
                prefix_export=prefix,
                workspace_diff="",
                source_task_sha256="a" * 64,
                source_tools_sha256="b" * 64,
            )
        )


@pytest.mark.parametrize(
    "tools, message",
    [
        ({"tool_names": {"read": "ReadSource"}}, "tool aliases"),
        ({"tool_names": {"apply_patch": "apply_patch"}}, "apply_patch"),
    ],
)
def test_dropped_features_are_refused_when_compiling(tools: dict, message: str) -> None:
    spec = RuntimeSpec(harness="opencode", task_backend="swe", tools=tools)
    for compile_ in (compile_profile, compile_boundary, bundle_files):
        with pytest.raises(UnsupportedFeatureError, match=message):
            compile_(spec)


def test_frozen_error_receipts_in_a_prefix_are_refused() -> None:
    spec = continuation_spec().model_dump(mode="json")
    part = spec["continuation"]["prefix_export"]["messages"][1]["parts"][0]
    part["state"] = {
        "status": "error",
        "input": {"command": "ls"},
        "error": "Read a.py first.",
        "metadata": {"nemo_gym_profile": {"model_error": "Read a.py first."}},
    }
    with pytest.raises(UnsupportedFeatureError, match="frozen error receipts"):
        compile_profile(RuntimeSpec.model_validate(spec))


def test_bundle_files_are_the_two_plugins_and_three_configurations() -> None:
    files = bundle_files(private_like_spec())
    assert list(files) == ["if-profile.mjs", "if-boundary.mjs", "profile.json", "plants.json", "boundary.json"]
    assert json.loads(files["profile.json"]) == compile_profile(private_like_spec())
    assert json.loads(files["plants.json"]) == compile_plants(private_like_spec())
    assert json.loads(files["boundary.json"]) == compile_boundary(private_like_spec())
    assert files["if-boundary.mjs"].startswith(b"// SPDX-FileCopyrightText")
