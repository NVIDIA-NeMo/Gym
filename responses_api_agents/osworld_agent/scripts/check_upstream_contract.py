#!/usr/bin/env python3
"""Statically verify the OSWorld APIs consumed by the Gym adapter."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Iterable


DESKTOP_ENV_INIT_ARGS = {
    "provider_name",
    "action_space",
    "screen_size",
    "headless",
    "require_a11y_tree",
    "os_type",
    "client_password",
    "cache_dir",
}
QWEN_INIT_ARGS = {
    "platform",
    "model",
    "max_tokens",
    "top_p",
    "temperature",
    "action_space",
    "observation_type",
    "history_n",
    "coordinate_type",
    "api_backend",
}


def _parse(path: Path) -> ast.Module:
    if not path.is_file():
        raise RuntimeError(f"required upstream file is missing: {path}")
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class(tree: ast.Module, name: str, path: Path) -> ast.ClassDef:
    node = next((item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name), None)
    if node is None:
        raise RuntimeError(f"{path} does not define class {name}")
    return node


def _method(class_node: ast.ClassDef, name: str, path: Path) -> ast.FunctionDef | ast.AsyncFunctionDef:
    node = next(
        (
            item
            for item in class_node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
        ),
        None,
    )
    if node is None:
        raise RuntimeError(f"{path}:{class_node.name} does not define {name}()")
    return node


def _arg_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = function.args
    return {item.arg for item in [*args.posonlyargs, *args.args, *args.kwonlyargs] if item.arg != "self"}


def _require_args(function: ast.FunctionDef | ast.AsyncFunctionDef, required: Iterable[str], label: str) -> None:
    missing = set(required) - _arg_names(function)
    if missing:
        raise RuntimeError(f"{label} is missing adapter arguments: {sorted(missing)}")


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def check_contract(root: Path, expected_commit: str | None = None) -> dict[str, object]:
    root = root.expanduser().resolve()
    commit = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current")
    status = _git(root, "status", "--porcelain")
    if expected_commit and commit != expected_commit:
        raise RuntimeError(f"OSWorld HEAD is {commit}, expected {expected_commit}")
    if branch != "main":
        raise RuntimeError(f"OSWorld checkout is on {branch!r}, expected 'main'")
    if status:
        raise RuntimeError("OSWorld checkout is not clean")

    env_path = root / "desktop_env" / "desktop_env.py"
    env_class = _class(_parse(env_path), "DesktopEnv", env_path)
    _require_args(_method(env_class, "__init__", env_path), DESKTOP_ENV_INIT_ARGS, "DesktopEnv.__init__")
    _require_args(_method(env_class, "reset", env_path), {"task_config"}, "DesktopEnv.reset")
    for method_name in ("step", "evaluate", "close"):
        _method(env_class, method_name, env_path)

    providers_path = root / "desktop_env" / "providers" / "__init__.py"
    provider_source = providers_path.read_text(encoding="utf-8")
    if 'provider_name == "docker"' not in provider_source and "provider_name == 'docker'" not in provider_source:
        raise RuntimeError("upstream provider factory does not expose the Docker provider")

    qwen_path = root / "mm_agents" / "qwen3vl_agent.py"
    qwen_class = _class(_parse(qwen_path), "Qwen3VLAgent", qwen_path)
    _require_args(_method(qwen_class, "__init__", qwen_path), QWEN_INIT_ARGS, "Qwen3VLAgent.__init__")
    for method_name in ("predict", "reset", "call_llm"):
        _method(qwen_class, method_name, qwen_path)

    for relative_path, class_name, methods in (
        ("mm_agents/m3/agent.py", "M3Agent", ("predict", "reset")),
        ("mm_agents/pointer/main.py", "PointerAgent", ("predict", "reset")),
        ("desktop_env/desktop_env_pointer.py", "DesktopEnv", ("step", "evaluate", "close")),
    ):
        path = root / relative_path
        class_node = _class(_parse(path), class_name, path)
        for method_name in methods:
            _method(class_node, method_name, path)

    return {
        "ok": True,
        "root": str(root),
        "branch": branch,
        "commit": commit,
        "working_tree_clean": True,
        "contracts": [
            "DesktopEnv+docker",
            "Qwen3VLAgent",
            "M3Agent",
            "PointerAgent",
            "Pointer DesktopEnv",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osworld-root", type=Path, required=True)
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    print(json.dumps(check_contract(args.osworld_root, args.expected_commit), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
