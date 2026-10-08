# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import sys
from types import ModuleType

from resources_servers.osworld.eval_task import _register_rlvr_evaluators


def test_register_rlvr_evaluators_loads_only_current_frozen_task(monkeypatch, tmp_path):
    task_root = tmp_path / "tmp_funcs" / "task-1"
    for role in ("getters", "metrics"):
        role_root = task_root / role
        role_root.mkdir(parents=True)
        function_name = "get_custom_task_1" if role == "getters" else "check_custom_task_1"
        (role_root / "custom.py").write_text(f"def {function_name}(value=None):\n    return value\n")

    desktop_env = ModuleType("desktop_env")
    evaluators = ModuleType("desktop_env.evaluators")
    getters = ModuleType("desktop_env.evaluators.getters")
    metrics = ModuleType("desktop_env.evaluators.metrics")
    evaluators.getters = getters
    evaluators.metrics = metrics
    monkeypatch.setitem(sys.modules, "desktop_env", desktop_env)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators", evaluators)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators.getters", getters)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators.metrics", metrics)
    monkeypatch.setenv("OSWORLD_RLVR_SNAPSHOT", str(tmp_path))

    task = {
        "id": "task-1",
        "evaluator": {
            "func": "check_custom_task_1",
            "result": {"type": "custom_task_1"},
        },
    }
    _register_rlvr_evaluators(task)

    assert getattr(getters, f"get_{task['evaluator']['result']['type']}")("getter") == "getter"
    assert getattr(metrics, task["evaluator"]["func"])("metric") == "metric"
