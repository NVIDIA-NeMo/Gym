# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import re
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

import nemo_gym.server_utils
from nemo_gym import PARENT_DIR
from nemo_gym.global_config import NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME
from nemo_gym.server_utils import (
    SimpleServer,
    _connect_server_to_ray,
    _declared_ray_enabled,
    entrypoint_ray_enabled,
)


WITH_CLUSTER = OmegaConf.create({"ray_head_node_address": "10.0.0.1:6379"})
WITHOUT_CLUSTER = OmegaConf.create({})


DIRECT_RAY_COMPONENTS = {
    "resources_servers": {
        "code_fim",
        "code_gen",
        "evalplus",
        "longmt_eval",
        "spider2_lite",
        "swerl_gen",
        "wmt_translation",
    },
    "responses_api_agents": {
        "anyterminal_agent",
        "harbor_agent",
        "harbor_agent_general",
        "mini_swe_agent",
        "mini_swe_agent_2",
        "osworld_agent",
        "stirrup_agent",
        "swe_agents",
    },
    "responses_api_models": {"local_vllm_model"},
    "environment_servers": set(),
}

INHERITED_RAY_DECLARATIONS = {
    ("resources_servers/legal_agent_bench/harbor_bridge.py", "LegalAgentBenchHarborBridge"): True,
    ("responses_api_models/genrm_model/app.py", "GenRMModel"): True,
}


def _component_imports_ray(component_dir: Path) -> bool:
    for path in component_dir.rglob("*.py"):
        relative_path = path.relative_to(component_dir)
        if any(part.startswith(".") or part in {"scripts", "tests"} for part in relative_path.parts):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                alias.name == "ray" or alias.name.startswith("ray.") for alias in node.names
            ):
                return True
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and (node.module == "ray" or node.module.startswith("ray."))
            ):
                return True
    return False


def _server_class_declarations(root: Path = PARENT_DIR) -> list[tuple[Path, ast.ClassDef, bool]]:
    declarations: list[tuple[Path, ast.ClassDef, bool]] = []
    for server_type, ray_backed_components in DIRECT_RAY_COMPONENTS.items():
        component_root = root / server_type
        for path in component_root.glob("*/**/*.py"):
            relative_path = path.relative_to(component_root)
            if any(part.startswith(".") or part in {"scripts", "tests"} for part in relative_path.parts):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                try:
                    tree = ast.parse(path.read_text())
                except (SyntaxError, UnicodeDecodeError):
                    continue
            classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
            invoked_classes = {
                node.func.value.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run_webserver"
                and isinstance(node.func.value, ast.Name)
            }
            component = relative_path.parts[0]
            for name in invoked_classes:
                if name not in classes:
                    continue
                key = (str(path.relative_to(root)), name)
                expected = INHERITED_RAY_DECLARATIONS.get(key, component in ray_backed_components)
                declarations.append((path, classes[name], expected))
    return declarations


class TestConnectServerToRay:
    class RayServer:
        ray_enabled = True

    class NonRayServer:
        ray_enabled = False

    def _patch(self, monkeypatch: pytest.MonkeyPatch, *, launched_by_gym: bool) -> MagicMock:
        initialize_ray = MagicMock()
        monkeypatch.setattr(nemo_gym.server_utils, "initialize_ray", initialize_ray)
        if launched_by_gym:
            monkeypatch.setenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, "{}")
        else:
            monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        return initialize_ray

    def test_joins_the_configured_cluster(self, monkeypatch: pytest.MonkeyPatch) -> None:
        initialize_ray = self._patch(monkeypatch, launched_by_gym=True)
        _connect_server_to_ray(self.RayServer, WITH_CLUSTER)
        initialize_ray.assert_called_once()

    def test_non_ray_server_never_connects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        initialize_ray = self._patch(monkeypatch, launched_by_gym=True)
        _connect_server_to_ray(self.NonRayServer, WITH_CLUSTER)
        initialize_ray.assert_not_called()

    def test_undeclared_server_never_connects(self, monkeypatch: pytest.MonkeyPatch, caplog) -> None:
        class UndeclaredServer(SimpleServer):
            pass

        initialize_ray = self._patch(monkeypatch, launched_by_gym=True)
        _connect_server_to_ray(UndeclaredServer, WITH_CLUSTER)
        initialize_ray.assert_not_called()
        assert caplog.text == ""

    def test_gym_launched_ray_server_without_a_cluster_fails_instead_of_starting_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        initialize_ray = self._patch(monkeypatch, launched_by_gym=True)
        with pytest.raises(RuntimeError, match="no Ray cluster"):
            _connect_server_to_ray(self.RayServer, WITHOUT_CLUSTER)
        initialize_ray.assert_not_called()

    def test_standalone_ray_server_may_start_its_own_cluster(self, monkeypatch: pytest.MonkeyPatch) -> None:
        initialize_ray = self._patch(monkeypatch, launched_by_gym=False)
        _connect_server_to_ray(self.RayServer, WITHOUT_CLUSTER)
        initialize_ray.assert_called_once()


def test_ray_backed_inventory_matches_production_imports() -> None:
    discovered = {
        server_type: {
            component_dir.name
            for component_dir in (PARENT_DIR / server_type).iterdir()
            if component_dir.is_dir() and _component_imports_ray(component_dir)
        }
        for server_type in DIRECT_RAY_COMPONENTS
    }

    assert discovered == DIRECT_RAY_COMPONENTS


def test_inventory_uses_paths_relative_to_the_checkout(tmp_path: Path) -> None:
    root = tmp_path / ".worktrees" / "review"
    component_dir = root / "resources_servers" / "example"
    component_dir.mkdir(parents=True)
    (component_dir / "app.py").write_text(
        "import ray\n"
        "class ExampleServer:\n"
        "    ray_enabled = False\n"
        "if __name__ == '__main__':\n"
        "    ExampleServer.run_webserver()\n"
    )

    assert _component_imports_ray(component_dir) is True
    declarations = _server_class_declarations(root)
    assert [(node.name, expected) for _path, node, expected in declarations] == [("ExampleServer", False)]


def test_shipped_server_classes_declare_ray_usage() -> None:
    problems: list[str] = []
    declarations = _server_class_declarations()
    assert declarations
    for path, class_node, expected in declarations:
        actual = _declared_ray_enabled(class_node)
        if actual is None:
            key = (str(path.relative_to(PARENT_DIR)), class_node.name)
            actual = INHERITED_RAY_DECLARATIONS.get(key)
        if actual is not expected:
            problems.append(
                f"{path.relative_to(PARENT_DIR)}:{class_node.name} expected ray_enabled = {expected}, got {actual}"
            )

    assert not problems, "Shipped server classes must declare Ray usage:\n" + "\n".join(problems)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(
            "class Server:\n    ray_enabled = False\nServer.run_webserver()\n",
            False,
            id="declared-false",
        ),
        pytest.param(
            "class Server:\n    ray_enabled = True\nServer.run_webserver()\n",
            True,
            id="declared-true",
        ),
        pytest.param("class Server:\n    pass\nServer.run_webserver()\n", None, id="undeclared"),
        pytest.param(
            "from elsewhere import Server\nServer.run_webserver()\n",
            None,
            id="imported-class",
        ),
        pytest.param(
            "class Base:\n    ray_enabled = False\nclass Server(Base):\n    pass\nServer.run_webserver()\n",
            None,
            id="inherited-declaration",
        ),
        pytest.param(
            "class A:\n    ray_enabled = False\nclass B:\n    ray_enabled = True\n"
            "A.run_webserver() if flag else B.run_webserver()\n",
            True,
            id="one-of-several-uses-ray",
        ),
        pytest.param(
            "class A:\n    ray_enabled = False\nfrom elsewhere import B\n"
            "A.run_webserver() if flag else B.run_webserver()\n",
            None,
            id="one-of-several-unknown",
        ),
        pytest.param("import uvicorn\nuvicorn.run(app)\n", None, id="no-run-webserver"),
        pytest.param("class Server(:\n", None, id="unparseable"),
    ],
)
def test_entrypoint_ray_enabled_reads_only_literal_declarations(
    tmp_path: Path, source: str, expected: bool | None
) -> None:
    entrypoint = tmp_path / "app.py"
    entrypoint.write_text(source)

    assert entrypoint_ray_enabled(entrypoint) is expected


def test_missing_entrypoint_has_unknown_ray_declaration(tmp_path: Path) -> None:
    assert entrypoint_ray_enabled(tmp_path / "missing.py") is None


def test_entrypoint_detection_matches_shipped_server_declarations() -> None:
    expected_by_path: dict[Path, bool] = {}
    for path, _class_node, expected in _server_class_declarations():
        expected_by_path[path] = expected_by_path.get(path, False) or expected

    mismatches = [
        f"{path.relative_to(PARENT_DIR)}: expected {expected}"
        for path, expected in expected_by_path.items()
        if entrypoint_ray_enabled(path) is not expected
    ]
    assert not mismatches, "Orchestrator Ray detection disagrees with shipped declarations:\n" + "\n".join(mismatches)


_NEMO_GYM_EXTRAS_RE = re.compile(r"nemo[-_]gym\[([^\]]*)\]")


def _requested_nemo_gym_extras(component_dir: Path) -> set[str]:
    extras: set[str] = set()
    for manifest in ("requirements.txt", "pyproject.toml", "setup.py"):
        manifest_path = component_dir / manifest
        if manifest_path.exists():
            for match in _NEMO_GYM_EXTRAS_RE.finditer(manifest_path.read_text()):
                extras.update(extra.strip() for extra in match.group(1).split(","))
    return extras


def test_ray_servers_request_the_ray_extra() -> None:
    # Ray is installed into a server venv only when that server asks for it, so every server that declares
    # ray_enabled = True must request nemo-gym's `ray` extra in its manifest.
    ray_components = {
        Path(*path.relative_to(PARENT_DIR).parts[:2])
        for path, _class_node, expected in _server_class_declarations()
        if expected
    }
    assert ray_components

    missing = sorted(str(c) for c in ray_components if "ray" not in _requested_nemo_gym_extras(PARENT_DIR / c))
    assert not missing, "Servers declaring ray_enabled = True must request nemo-gym[ray]:\n" + "\n".join(missing)
