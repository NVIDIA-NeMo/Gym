# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for environment servers' task input modules (nemo_gym/data/task_input.py)."""

import ast
import importlib
import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest

from nemo_gym.data.task_input import TaskInputModuleError, find_task_input_contract, load_task_input_module


REPO_ROOT = Path(__file__).resolve().parents[3]
ENVIRONMENT_SERVER_DIRS = sorted(d for d in (REPO_ROOT / "environment_servers").iterdir() if (d / "app.py").is_file())
TASK_INPUT_FILES = sorted((REPO_ROOT / "environment_servers").glob("*/task_input.py"))
# Environment servers whose /run passes a flat row to the agent's /run: their module sets TaskInput = None.
RELAY_ENVIRONMENT_SERVERS = frozenset({"legacy_agent", "misalignment_monitored"})


def _packages_at_or_below_data() -> tuple[str, ...]:
    """The import-linter layers from ``nemo_gym.data`` down, without the supervisor that starts servers."""
    contract = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["tool"]["importlinter"]["contracts"][0]
    layers = contract["layers"]
    below = layers[layers.index("nemo_gym.data") :]
    modules = [module.strip() for layer in below for module in layer.split(":")]
    return tuple(module for module in modules if module != "nemo_gym.server_runtime.supervisor")


def _import_target(module: str, name: str) -> str:
    """The module ``from <module> import <name>`` loads: a submodule, a package export's own module, or ``module``."""
    if not module.startswith("nemo_gym"):
        return module
    if importlib.util.find_spec(f"{module}.{name}") is not None:
        return f"{module}.{name}"
    submodule = getattr(importlib.import_module(module), "_PUBLIC_API", {}).get(name)
    return f"{module}.{submodule}" if submodule is not None else module


def _imported_modules(module_file: Path) -> list[str]:
    """Every module ``module_file`` imports, with each from-import resolved to the module that defines the name."""
    modules = []
    for node in ast.walk(ast.parse(module_file.read_text())):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules.extend(_import_target(node.module or "", alias.name) for alias in node.names)
    return modules


class TestShippedTaskInputModules:
    def test_every_environment_server_ships_one(self):
        missing = sorted(d.name for d in ENVIRONMENT_SERVER_DIRS if not (d / "task_input.py").is_file())
        assert not missing, (
            f"environment servers without a task_input.py: {missing}. Export the TaskInput their episode request "
            "validates, or TaskInput = None for a relay (see nemo_gym/data/task_input.py)."
        )

    @pytest.mark.parametrize("module_file", TASK_INPUT_FILES, ids=lambda p: p.parent.name)
    def test_imports_are_dependency_light(self, module_file):
        allowed_nemo_gym = _packages_at_or_below_data()
        for module in _imported_modules(module_file):
            allowed = (
                module.split(".")[0] in sys.stdlib_module_names
                or module == "pydantic"
                or module.startswith("pydantic.")
                or any(module == layer or module.startswith(f"{layer}.") for layer in allowed_nemo_gym)
                or (module.startswith("resources_servers.") and module.endswith(".task_data"))
            )
            assert allowed, (
                f"{module_file}: import of {module!r} is not allowed in a task input module (the standard "
                "library, pydantic, nemo_gym packages at or below nemo_gym.data, and task_data modules only)"
            )

    def test_a_facade_import_resolves_to_the_module_that_defines_the_name(self):
        allowed_nemo_gym = _packages_at_or_below_data()
        assert _import_target("nemo_gym.server_runtime", "SingleAgentTurnTaskInput") == (
            "nemo_gym.server_runtime.single_agent_turn_types"
        )
        # A facade or parent package must not admit a module that a direct import would not.
        for module, name in (("nemo_gym", "environment"), ("nemo_gym.server_runtime", "supervisor")):
            target = _import_target(module, name)
            assert target == f"{module}.{name}"
            assert not any(target == layer or target.startswith(f"{layer}.") for layer in allowed_nemo_gym)

    @pytest.mark.parametrize("module_file", TASK_INPUT_FILES, ids=lambda p: p.parent.name)
    def test_loads(self, module_file):
        contract = load_task_input_module(module_file.parent)
        assert contract is not None
        assert contract.is_relay is (module_file.parent.name in RELAY_ENVIRONMENT_SERVERS)
        # Only single_agent_turn_legacy materializes a flat row on /run.
        assert contract.accepts_flat_rows is (module_file.parent.name == "single_agent_turn_legacy")


class TestLoader:
    def _server(self, root: Path, name: str, body: str) -> Path:
        server_dir = root / "environment_servers" / name
        server_dir.mkdir(parents=True)
        (server_dir / "task_input.py").write_text(body)
        return server_dir

    def test_a_module_without_task_input_is_refused(self, tmp_path):
        server_dir = self._server(tmp_path, "no_export", "x = 1\n")
        with pytest.raises(TaskInputModuleError, match="does not export `TaskInput`"):
            load_task_input_module(server_dir)

    def test_a_relay_sets_task_input_to_none_and_takes_no_flat_row_flag(self, tmp_path):
        relay = load_task_input_module(self._server(tmp_path, "relay", "TaskInput = None\n"))
        assert relay is not None and relay.is_relay and relay.task is None
        flagged = self._server(tmp_path, "flagged_relay", "TaskInput = None\nACCEPTS_FLAT_ROWS = True\n")
        with pytest.raises(TaskInputModuleError, match="a relay"):
            load_task_input_module(flagged)

    def test_a_server_that_ships_only_the_module_is_found(self, tmp_path, monkeypatch):
        self._server(tmp_path, "module_only", "TaskInput = None\n")
        monkeypatch.chdir(tmp_path)
        lookup = find_task_input_contract("module_only")
        assert lookup.contract is not None and lookup.contract.is_relay

    def test_a_server_without_the_module_is_reported_as_missing(self, tmp_path, monkeypatch):
        (tmp_path / "environment_servers" / "untyped").mkdir(parents=True)
        # A source marker makes the folder a server folder, not just a folder of that name.
        (tmp_path / "environment_servers" / "untyped" / "requirements.txt").write_text("")
        monkeypatch.chdir(tmp_path)
        lookup = find_task_input_contract("untyped")
        assert lookup.contract is None
        assert lookup.missing == "untyped ships no task_input.py"
        assert find_task_input_contract("nowhere_to_be_found").missing == (
            "no environment_servers/nowhere_to_be_found folder is found"
        )

    def test_a_module_that_does_not_import_is_refused(self, tmp_path):
        server_dir = self._server(tmp_path, "bad_import", "import does_not_exist_anywhere\n")
        with pytest.raises(TaskInputModuleError, match="Failed to import"):
            load_task_input_module(server_dir)

    def test_accepts_flat_rows_must_be_a_bool(self, tmp_path):
        body = (
            "from pydantic import BaseModel\n\nclass TaskInput(BaseModel):\n    x: int\n\nACCEPTS_FLAT_ROWS = 'yes'\n"
        )
        server_dir = self._server(tmp_path, "flag", body)
        with pytest.raises(TaskInputModuleError, match="must be a bool"):
            load_task_input_module(server_dir)

    def test_a_component_import_from_another_root_is_refused(self, tmp_path):
        # The import resolves to this checkout's resources server, not one under the task input module's own root.
        body = "from resources_servers.nemo_user_sim.task_data import TaskData as TaskInput\n"
        server_dir = self._server(tmp_path, "other_root", body)
        with pytest.raises(TaskInputModuleError, match="is not under.*--search-dir or NEMO_GYM_EXTRA_ROOTS"):
            load_task_input_module(server_dir)

    def test_a_dotted_import_from_another_root_is_refused(self, tmp_path):
        # The wrapper class lives in the task input module, but one of its fields uses this checkout's type.
        body = (
            "import resources_servers.nemo_user_sim.task_data\n"
            "from pydantic import BaseModel\n\n"
            "class TaskInput(BaseModel):\n"
            "    task_data: resources_servers.nemo_user_sim.task_data.TaskData\n"
        )
        server_dir = self._server(tmp_path, "dotted", body)
        with pytest.raises(TaskInputModuleError, match="validates with TaskData from"):
            load_task_input_module(server_dir)

    def test_a_foreign_import_that_task_input_does_not_use_is_allowed(self, tmp_path):
        body = (
            "import resources_servers.nemo_user_sim.task_data\n"
            "from pydantic import BaseModel\n\n"
            "class TaskInput(BaseModel):\n    question: str\n"
        )
        contract = load_task_input_module(self._server(tmp_path, "unused_import", body))
        assert contract is not None and contract.task_input.__name__ == "TaskInput"

    def test_the_contract_validates_task_rows(self, tmp_path):
        body = (
            "from pydantic import BaseModel, ConfigDict\n\n"
            "class TaskInput(BaseModel):\n    model_config = ConfigDict(extra='forbid')\n    question: str\n"
        )
        contract = load_task_input_module(self._server(tmp_path, "typed", body))
        assert contract is not None and not contract.accepts_flat_rows
        assert not contract.requires_responses_create_params
        contract.task.validate_python({"task_id": {"taskset": "t", "task_id": "1"}, "task_input": {"question": "q"}})
        with pytest.raises(Exception, match="extra"):
            contract.task.validate_python(
                {"task_id": {"taskset": "t", "task_id": "1"}, "task_input": {"question": "q", "typo": 1}}
            )
