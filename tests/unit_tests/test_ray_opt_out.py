# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import warnings
from pathlib import Path

from nemo_gym import PARENT_DIR
from nemo_gym.server_utils import _WARNED_IMPLICIT_RAY_SERVERS, _server_uses_ray


RAY_BACKED_COMPONENTS = {
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
}


def _component_imports_ray(component_dir: Path) -> bool:
    for path in component_dir.rglob("*.py"):
        if any(part.startswith(".") or part in {"scripts", "tests"} for part in path.parts):
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


def _server_class_declarations() -> list[tuple[Path, ast.ClassDef, bool]]:
    declarations: list[tuple[Path, ast.ClassDef, bool]] = []
    for server_type, ray_backed_components in RAY_BACKED_COMPONENTS.items():
        for path in (PARENT_DIR / server_type).glob("*/**/*.py"):
            if any(part.startswith(".") or part in {"scripts", "tests"} for part in path.parts):
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
            component = path.relative_to(PARENT_DIR).parts[1]
            expected = component in ray_backed_components
            declarations.extend((path, classes[name], expected) for name in invoked_classes if name in classes)
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
        for server_type in RAY_BACKED_COMPONENTS
    }

    assert discovered == RAY_BACKED_COMPONENTS


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
