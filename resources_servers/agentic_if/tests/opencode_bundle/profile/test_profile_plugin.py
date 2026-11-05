# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The profile plugin (``if-profile.mjs``) under Node, staged in a bundle directory as the Resources Server stages it.

Each ``*.test.mjs`` here drives the plugin's hooks directly; this module stages the plugin beside its compiled
``profile.json`` and ``plants.json`` with an empty ``receipts/`` directory, runs the script with a bounded timeout, and
checks its success line.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from resources_servers.agentic_if.opencode_bundle.layout import PROFILE_PLUGIN, SOURCE_DIR
from resources_servers.agentic_if.opencode_bundle.profile import compile_plants, compile_profile
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec


HERE = Path(__file__).parent
FIXTURE = json.loads((HERE / "profile_fixture.json").read_text())
GUARD = {
    "edit": {"unread": "Read {{path}} first.", "changed": "Read {{path}} again."},
    "write": {"unread": "Read {{path}} before overwriting."},
}

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node is required to run the OpenCode plugin")
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is required for workspace diffs")


def stage(tmp_path: Path, *, profile: dict | str | None, plants: list | str | None) -> Path:
    """A bundle directory: the plugin, its configuration files (None: absent) and an empty receipts directory."""
    bundle = tmp_path / "bundle"
    (bundle / "receipts").mkdir(parents=True)
    shutil.copyfile(SOURCE_DIR / PROFILE_PLUGIN, bundle / PROFILE_PLUGIN)
    for name, content in (("profile.json", profile), ("plants.json", plants)):
        if content is not None:
            (bundle / name).write_text(content if isinstance(content, str) else json.dumps(content))
    return bundle


def run_node(script: str, tmp_path: Path, bundle: Path) -> subprocess.CompletedProcess[str]:
    """Run one test script against the staged plugin, isolated from the user's git configuration."""
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    env = os.environ | {
        "IF_PROFILE_PLUGIN": str(bundle / PROFILE_PLUGIN),
        "IF_BUNDLE": str(bundle),
        "IF_TEST_ROOT": str(work),
        "TMPDIR": str(tmp_path),
        "HOME": str(tmp_path),
        "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    # The working directory is not a git repository, so calls without a directory record no workspace diff.
    return subprocess.run(
        ["node", str(HERE / f"{script}.test.mjs")], cwd=work, env=env, capture_output=True, text=True, timeout=60
    )


def assert_passed(result: subprocess.CompletedProcess[str], line: str) -> None:
    assert result.returncode == 0, result.stdout + result.stderr
    assert line in result.stdout


@needs_git
def test_interfaces_map_arguments_record_metadata_and_workspace_diffs(tmp_path: Path) -> None:
    bundle = stage(tmp_path, profile=FIXTURE, plants=[])
    assert_passed(run_node("plugin_hooks", tmp_path, bundle), "plugin offline test: all checks passed")


def test_planted_failures_load_fire_and_fail_closed(tmp_path: Path) -> None:
    bundle = stage(tmp_path, profile=FIXTURE, plants=None)
    assert_passed(run_node("plants_hooks", tmp_path, bundle), "plants offline test: all checks passed")


def test_read_first_guards_and_resumed_session_state(tmp_path: Path) -> None:
    spec = RuntimeSpec(harness="opencode", task_backend="swe", tools={"read_first": GUARD})
    bundle = stage(tmp_path, profile=compile_profile(spec), plants=compile_plants(spec))
    assert_passed(run_node("read_first", tmp_path, bundle), "read-first guard passed")


def test_error_presentation_for_selected_tools(tmp_path: Path) -> None:
    spec = RuntimeSpec(
        harness="opencode",
        task_backend="swe",
        tools={
            "tool_names": {"read": "read", "bash": "bash", "edit": "edit", "write": "write"},
            "tool_interfaces": {"read": "cw/zed@3/read", "edit": "cw/pi@2/edit"},
        },
        events=[{"id": "literal", "match": {"tool": "read"}, "action": "fail", "text": "PLANTED read timeout"}],
    )
    profile = compile_profile(spec)
    assert profile["visible_tools"] == {"read": "read", "bash": "bash", "edit": "edit", "write": "write"}
    assert profile["check_aliases"] is False
    assert profile["literal_errors"] == ["PLANTED read timeout"]
    # No plant is staged here: the script exercises presentation only, and the literal error must stay as written.
    bundle = stage(tmp_path, profile=profile, plants=[])
    assert_passed(run_node("opencode_errors", tmp_path, bundle), "error presentation passed")


@needs_git
def test_interfaces_plants_and_binary_workspace_receipts_compose(tmp_path: Path) -> None:
    spec = RuntimeSpec(
        harness="opencode",
        task_backend="swe",
        tools={
            "tool_names": {"read": "read", "task": "task", "bash": "bash"},
            "tool_interfaces": {"read": "cw/zed@3/read", "task": "cw/zed@3/task"},
            "tool_parameter_descriptions": {"read": {"start_line": "First line, inclusive."}},
        },
    )
    profile = compile_profile(spec)
    assert profile["tools"]["task"]["fixed"] == {"subagent_type": "general"}
    assert profile["tools"]["read"]["args_map"] == {"filePath": "path", "offset": "start_line", "limit": "line_count"}
    bundle = stage(tmp_path, profile=profile, plants=compile_plants(spec))
    assert_passed(run_node("opencode_hooks", tmp_path, bundle), "combined hooks passed")


def test_load_receipt_and_fail_closed_loading(tmp_path: Path) -> None:
    spec = RuntimeSpec(harness="opencode", task_backend="swe", tools={"tool_names": {"bash": "bash"}})
    bundle = stage(tmp_path, profile=compile_profile(spec), plants=[])
    assert_passed(run_node("load_receipt", tmp_path, bundle), "load receipts passed")
