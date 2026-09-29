# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import warnings
from pathlib import Path

from nemo_gym import PARENT_DIR
from nemo_gym.server_utils import _WARNED_IMPLICIT_RAY_SERVERS, _server_uses_ray


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
}

INHERITED_RAY_DECLARATIONS = {
    ("resources_servers/gpqa_diamond/app.py", "GPQADiamondResourcesServer"): False,
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


def _declared_ray_value(class_node: ast.ClassDef) -> bool | None:
    for item in class_node.body:
        if not isinstance(item, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "ray_enabled" for target in item.targets):
            continue
        if isinstance(item.value, ast.Constant) and isinstance(item.value.value, bool):
            return item.value.value
    return None


def test_omitted_ray_flag_preserves_compatibility(caplog) -> None:
    class LegacyServer:
        ray_enabled = None

    _WARNED_IMPLICIT_RAY_SERVERS.clear()
    assert _server_uses_ray(LegacyServer) is True
    assert "Ray remains enabled for backward compatibility" in caplog.text
    assert "future release will default it to false" in caplog.text


def test_explicit_ray_declarations_do_not_warn(caplog) -> None:
    class RayServer:
        ray_enabled = True

    class NonRayServer:
        ray_enabled = False

    assert _server_uses_ray(RayServer) is True
    assert _server_uses_ray(NonRayServer) is False
    assert caplog.text == ""


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
        actual = _declared_ray_value(class_node)
        if actual is None:
            key = (str(path.relative_to(PARENT_DIR)), class_node.name)
            actual = INHERITED_RAY_DECLARATIONS.get(key)
        if actual is not expected:
            problems.append(
                f"{path.relative_to(PARENT_DIR)}:{class_node.name} expected ray_enabled = {expected}, got {actual}"
            )

    assert not problems, "Shipped server classes must declare Ray usage:\n" + "\n".join(problems)
