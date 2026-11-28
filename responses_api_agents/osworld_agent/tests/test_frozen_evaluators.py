# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import types

import pytest

from responses_api_agents.osworld_agent.frozen_evaluators import (
    apply_task_reward_contract,
    clear_frozen_evaluators,
    install_run_command_compat,
    register_frozen_evaluators,
)


def builtin_vm_file(env, config):
    return "builtin"


def builtin_compare_csv(result, expected):
    return 1.0


@pytest.fixture
def fake_desktop_env(monkeypatch):
    """Stand-in for the desktop_env packages so the test needs no OSWorld install."""

    root = types.ModuleType("desktop_env")
    evaluators = types.ModuleType("desktop_env.evaluators")
    getters = types.ModuleType("desktop_env.evaluators.getters")
    metrics = types.ModuleType("desktop_env.evaluators.metrics")
    getters.__path__ = ["/osworld/getters"]
    metrics.__path__ = ["/osworld/metrics"]
    getters.get_vm_file = builtin_vm_file
    metrics.compare_csv = builtin_compare_csv
    evaluators.getters, evaluators.metrics = getters, metrics
    root.evaluators = evaluators
    controllers = types.ModuleType("desktop_env.controllers")
    python = types.ModuleType("desktop_env.controllers.python")

    class PythonController:
        http_server = "http://guest:5000"
        retry_times = 2
        retry_interval = 0

    python.PythonController = PythonController
    for name, module in (
        ("desktop_env", root),
        ("desktop_env.evaluators", evaluators),
        ("desktop_env.evaluators.getters", getters),
        ("desktop_env.evaluators.metrics", metrics),
        ("desktop_env.controllers", controllers),
        ("desktop_env.controllers.python", python),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    yield getters, metrics, PythonController
    clear_frozen_evaluators()


def write_task(root, task_id, files):
    for relative, source in files.items():
        path = root / task_id / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)


TASK_FILES = {
    "getters/vm_file.py": "from .file_helper import label\n\ndef get_vm_file(env, config):\n    return label()\n",
    "getters/file_helper.py": "def label():\n    return 'frozen'\n",
    "metrics/rows.py": (
        "from rows_helper import expected_rows\n\n"
        "def check_rows(result, expected):\n    return float(result == expected_rows())\n"
    ),
    "metrics/rows_helper.py": "def expected_rows():\n    return 3\n",
    "metrics/selftest_rows.py": "raise RuntimeError('validation script must not be loaded')\n",
    "metrics/__init__.py": "raise RuntimeError('package marker must not be loaded')\n",
}


def test_task_modules_replace_builtins_and_import_task_helpers(fake_desktop_env, tmp_path):
    getters, metrics, _ = fake_desktop_env
    write_task(tmp_path, "task-a", TASK_FILES)

    loaded = register_frozen_evaluators({"id": "task-a"}, str(tmp_path))

    assert "getters:get_vm_file" in loaded and "metrics:check_rows" in loaded
    assert getters.get_vm_file(None, {}) == "frozen"
    assert metrics.check_rows(3, None) == 1.0
    assert metrics.compare_csv is builtin_compare_csv


def test_clear_restores_packages_import_path_and_module_cache(fake_desktop_env, tmp_path):
    getters, metrics, _ = fake_desktop_env
    write_task(tmp_path, "task-a", TASK_FILES)
    sys_path_before = list(sys.path)
    modules_before = set(sys.modules)

    register_frozen_evaluators({"id": "task-a"}, str(tmp_path))
    clear_frozen_evaluators()
    clear_frozen_evaluators()

    assert getters.get_vm_file is builtin_vm_file
    assert not hasattr(metrics, "check_rows") and not hasattr(metrics, "expected_rows")
    assert not hasattr(getters, "label")
    assert getters.__path__ == ["/osworld/getters"] and metrics.__path__ == ["/osworld/metrics"]
    assert sys.path == sys_path_before
    assert set(sys.modules) - modules_before == set()


def test_same_named_helpers_do_not_leak_between_tasks(fake_desktop_env, tmp_path):
    _, metrics, _ = fake_desktop_env
    write_task(tmp_path, "task-a", TASK_FILES)
    write_task(
        tmp_path,
        "task-b",
        {
            "metrics/rows.py": TASK_FILES["metrics/rows.py"],
            "metrics/rows_helper.py": "def expected_rows():\n    return 7\n",
        },
    )

    register_frozen_evaluators({"id": "task-a"}, str(tmp_path))
    assert metrics.check_rows(3, None) == 1.0
    clear_frozen_evaluators()
    register_frozen_evaluators({"id": "task-b"}, str(tmp_path))

    assert metrics.check_rows(3, None) == 0.0
    assert metrics.check_rows(7, None) == 1.0


def test_task_without_frozen_modules_is_untouched(fake_desktop_env, tmp_path):
    getters, _, _ = fake_desktop_env

    assert register_frozen_evaluators({"id": "task-a"}, str(tmp_path)) == []
    assert getters.get_vm_file is builtin_vm_file
    assert getters.__path__ == ["/osworld/getters"]


def test_task_directory_without_modules_fails_and_still_clears(fake_desktop_env, tmp_path):
    getters, _, _ = fake_desktop_env
    write_task(tmp_path, "task-a", {"metrics/selftest_rows.py": "pass\n"})

    with pytest.raises(FileNotFoundError, match="task-a"):
        register_frozen_evaluators({"id": "task-a"}, str(tmp_path))
    clear_frozen_evaluators()

    assert getters.__path__ == ["/osworld/getters"]
    assert str(tmp_path / "task-a" / "metrics") not in sys.path


def test_module_without_functions_fails(fake_desktop_env, tmp_path):
    write_task(tmp_path, "task-a", {"metrics/constants.py": "THRESHOLD = 3\n"})

    with pytest.raises(ImportError, match="exports no evaluator functions"):
        register_frozen_evaluators({"id": "task-a"}, str(tmp_path))


def test_failed_import_is_cleared(fake_desktop_env, tmp_path):
    getters, _, _ = fake_desktop_env
    write_task(
        tmp_path,
        "task-a",
        {
            "getters/vm_file.py": TASK_FILES["getters/vm_file.py"],
            "getters/file_helper.py": TASK_FILES["getters/file_helper.py"],
            "metrics/broken.py": "import does_not_exist\n",
        },
    )

    with pytest.raises(ImportError):
        register_frozen_evaluators({"id": "task-a"}, str(tmp_path))
    clear_frozen_evaluators()

    assert getters.get_vm_file is builtin_vm_file


@pytest.mark.parametrize("task_id", ["", None, "../outside"])
def test_rejects_missing_or_escaping_task_ids(fake_desktop_env, tmp_path, task_id):
    with pytest.raises(ValueError):
        register_frozen_evaluators({"id": task_id}, str(tmp_path / "functions"))


def test_declared_routes_must_be_loaded(fake_desktop_env, tmp_path):
    write_task(tmp_path, "task-a", TASK_FILES)
    routes = {"metadata": {"evaluator_routes": [{"route": "metrics:check_rows"}, {"route": "getters:get_vm_file"}]}}

    register_frozen_evaluators({"id": "task-a", **routes}, str(tmp_path))
    clear_frozen_evaluators()

    missing = {"metadata": {"evaluator_routes": [{"route": "metrics:check_columns"}]}}
    with pytest.raises(ImportError, match="metrics:check_columns"):
        register_frozen_evaluators({"id": "task-a", **missing}, str(tmp_path))
    clear_frozen_evaluators()

    with pytest.raises(ValueError, match="route record"):
        register_frozen_evaluators({"id": "task-a", "metadata": {"evaluator_routes": ["check_rows"]}}, str(tmp_path))


def test_registration_installs_run_command(fake_desktop_env, tmp_path):
    _, _, controller = fake_desktop_env
    write_task(tmp_path, "task-a", TASK_FILES)

    register_frozen_evaluators({"id": "task-a"}, str(tmp_path))

    assert callable(controller.run_command)


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code, self._payload, self.text = status_code, payload, text

    def json(self):
        return self._payload


def test_run_command_posts_to_guest_execute(fake_desktop_env, monkeypatch):
    _, _, controller = fake_desktop_env
    calls = []

    def post(url, json, timeout):
        calls.append((url, json, timeout))
        return FakeResponse(200, {"status": "success", "output": "ok", "returncode": 0})

    monkeypatch.setattr("requests.post", post)
    install_run_command_compat()

    result = controller().run_command(["ls", "/tmp"], timeout=9)

    assert result["output"] == "ok"
    assert calls == [("http://guest:5000/execute", {"command": ["ls", "/tmp"], "shell": False}, 9)]


def test_run_command_returns_error_after_retries(fake_desktop_env, monkeypatch):
    _, _, controller = fake_desktop_env
    calls = []

    def post(url, json, timeout):
        calls.append(url)
        return FakeResponse(503, text="busy")

    monkeypatch.setattr("requests.post", post)
    install_run_command_compat()

    result = controller().run_command("ls", shell=True)

    assert len(calls) == 2
    assert result == {
        "status": "error",
        "output": "",
        "error": "guest execute returned HTTP 503: busy",
        "returncode": -1,
    }


def test_existing_run_command_is_kept(fake_desktop_env):
    _, _, controller = fake_desktop_env

    def native(self, command, shell=False, timeout=120):
        return {"native": True}

    controller.run_command = native
    install_run_command_compat()

    assert controller.run_command is native


@pytest.mark.parametrize(("score", "reward"), [(1.0, 1.0), (0.999, 0.0), (0.0, 0.0), (1, 1.0)])
def test_binary_final_reward_counts_only_full_scores(score, reward):
    assert apply_task_reward_contract({"metadata": {"binary_final_reward": True}}, score) == reward


@pytest.mark.parametrize("metadata", [None, {}, {"binary_final_reward": False}, {"binary_final_reward": "true"}])
def test_score_passes_through_without_the_contract(metadata):
    assert apply_task_reward_contract({"metadata": metadata}, 0.4) == 0.4


@pytest.mark.parametrize("score", [1.5, -0.1, float("nan"), "1.0"])
def test_binary_final_reward_rejects_invalid_scores(score):
    with pytest.raises((TypeError, ValueError)):
        apply_task_reward_contract({"metadata": {"binary_final_reward": True}}, score)
