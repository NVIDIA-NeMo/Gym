# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for environment servers' task input modules (nemo_gym/data/task_input.py)."""

import ast
import sys
import tomllib
from pathlib import Path

import pytest

from nemo_gym.data.task_input import TaskInputModuleError, load_task_input_module


REPO_ROOT = Path(__file__).resolve().parents[3]
ENVIRONMENT_SERVER_DIRS = sorted(d for d in (REPO_ROOT / "environment_servers").iterdir() if (d / "app.py").is_file())
TASK_INPUT_FILES = sorted((REPO_ROOT / "environment_servers").glob("*/task_input.py"))
# Environment servers whose /run passes a flat row to the agent's /run: they have no episode task input.
RELAY_ENVIRONMENT_SERVERS = frozenset({"legacy_agent", "misalignment_monitored"})


def _packages_at_or_below_data() -> tuple[str, ...]:
    """The import-linter layers from ``nemo_gym.data`` down, without the supervisor that starts servers."""
    contract = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["tool"]["importlinter"]["contracts"][0]
    layers = contract["layers"]
    below = layers[layers.index("nemo_gym.data") :]
    modules = [module.strip() for layer in below for module in layer.split(":")]
    return tuple(module for module in modules if module != "nemo_gym.server_runtime.supervisor")


class TestShippedTaskInputModules:
    def test_every_typed_environment_server_ships_one_and_relays_do_not(self):
        shipping = {d.name for d in ENVIRONMENT_SERVER_DIRS if (d / "task_input.py").is_file()}
        servers = {d.name for d in ENVIRONMENT_SERVER_DIRS}
        assert RELAY_ENVIRONMENT_SERVERS <= servers, "a listed relay no longer exists; drop it from the list"
        assert not shipping & RELAY_ENVIRONMENT_SERVERS, (
            f"{sorted(shipping & RELAY_ENVIRONMENT_SERVERS)} ship a task_input.py but are listed as relays"
        )
        missing = sorted(servers - shipping - RELAY_ENVIRONMENT_SERVERS)
        assert not missing, (
            f"environment servers without a task_input.py: {missing}. Export the TaskInput their episode request "
            "validates (see nemo_gym/data/task_input.py), or list the server as a relay."
        )

    @pytest.mark.parametrize("module_file", TASK_INPUT_FILES, ids=lambda p: p.parent.name)
    def test_imports_are_dependency_light(self, module_file):
        allowed_nemo_gym = _packages_at_or_below_data()
        for node in ast.walk(ast.parse(module_file.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
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

    @pytest.mark.parametrize("module_file", TASK_INPUT_FILES, ids=lambda p: p.parent.name)
    def test_loads(self, module_file):
        contract = load_task_input_module(module_file.parent)
        assert contract is not None
        # Only single_agent_turn_legacy materializes a flat row on /run.
        assert contract.accepts_flat_rows is (module_file.parent.name == "single_agent_turn_legacy")

    @pytest.mark.parametrize("relay", sorted(RELAY_ENVIRONMENT_SERVERS))
    def test_a_relay_has_no_contract(self, relay):
        assert load_task_input_module(REPO_ROOT / "environment_servers" / relay) is None


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
        with pytest.raises(TaskInputModuleError, match="is not under"):
            load_task_input_module(server_dir)

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
