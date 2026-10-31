# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import types

import pytest

from responses_api_agents.osworld_agent import taskgen_plugins
from responses_api_agents.osworld_agent.taskgen_plugins import TaskPluginError, plugin_digest, register_task_plugins


@pytest.fixture
def fake_desktop_env(monkeypatch):
    """Stand-in for desktop_env.evaluators so the test needs no OSWorld install."""

    root = types.ModuleType("desktop_env")
    evaluators = types.ModuleType("desktop_env.evaluators")
    getters = types.ModuleType("desktop_env.evaluators.getters")
    metrics = types.ModuleType("desktop_env.evaluators.metrics")
    getters.get_vm_file = lambda env, config: None  # a built-in
    metrics.compare_csv = lambda result, expected: 1.0  # a built-in
    evaluators.getters, evaluators.metrics = getters, metrics
    root.evaluators = evaluators
    for name, module in (
        ("desktop_env", root),
        ("desktop_env.evaluators", evaluators),
        ("desktop_env.evaluators.getters", getters),
        ("desktop_env.evaluators.metrics", metrics),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(taskgen_plugins, "_REGISTERED", {})
    return getters, metrics


GETTERS = "def get_rows__ab12(env, config):\n    return config['value']\n"
METRICS = (
    "def check_rows__ab12(result, expected, **options):\n    return 1.0 if result == expected['value'] else 0.0\n"
)


def _task(getters_source=GETTERS, metrics_source=METRICS, **overrides):
    modules = {"getters": getters_source, "metrics": metrics_source}
    block = {
        "schema": "taskgen-plugins/1",
        "getters": ["rows__ab12"],
        "metrics": ["check_rows__ab12"],
        "modules": modules,
        "sha256": plugin_digest(modules),
    }
    block.update(overrides)
    return {"id": "ctb-0123456789abcdef", "evaluator": {"func": "check_rows__ab12"}, "taskgen_plugins": block}


def test_registers_declared_functions(fake_desktop_env):
    getters, metrics = fake_desktop_env
    assert register_task_plugins(_task()) == ["get_rows__ab12", "check_rows__ab12"]
    assert getters.get_rows__ab12(None, {"value": 3}) == 3
    assert metrics.check_rows__ab12(3, {"value": 3}) == 1.0
    assert register_task_plugins(_task()) == ["get_rows__ab12", "check_rows__ab12"]  # same plugin again is fine


def test_task_without_plugins_is_untouched(fake_desktop_env):
    assert register_task_plugins({"id": "plain", "evaluator": {"func": "compare_csv"}}) == []


def test_rejects_tampered_or_incomplete_blocks(fake_desktop_env):
    tampered = _task()
    tampered["taskgen_plugins"]["modules"]["metrics"] += "\n# edited\n"
    with pytest.raises(TaskPluginError, match="sha256"):
        register_task_plugins(tampered)
    with pytest.raises(TaskPluginError, match="does not define"):
        register_task_plugins(_task(metrics_source="def something_else():\n    return 0\n"))
    with pytest.raises(TaskPluginError, match="schema|taskgen-plugins/1"):
        register_task_plugins(_task(schema="other"))


def test_never_replaces_osworld_functions(fake_desktop_env):
    getters, metrics = fake_desktop_env
    source = "def compare_csv(result, expected):\n    return 1.0\n"
    with pytest.raises(TaskPluginError, match="replace OSWorld"):
        register_task_plugins(_task(metrics_source=source, metrics=["compare_csv"]))


def test_same_name_from_another_plugin_is_refused(fake_desktop_env):
    register_task_plugins(_task())
    other = METRICS.replace("1.0 if", "0.5 if")
    with pytest.raises(TaskPluginError, match="different plugin"):
        register_task_plugins(_task(metrics_source=other))


def test_remote_resources_path_fails_loudly(fake_desktop_env):
    with pytest.raises(TaskPluginError, match="remote"):
        register_task_plugins(_task(), remote=True)
