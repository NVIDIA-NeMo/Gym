# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repo-wide checks that every agent and Resources Server serves sessions through the shared contract.

Servers run in their own environments, so these checks read source instead of importing each server. Behavior is
tested where it lives: the base agent's session bookkeeping in the base agent tests, and a Resources Server's seed and
close through ``nemo_gym.testing.session_conformance`` in that server's tests.
"""

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sources(directory: str) -> list[Path]:
    # Hidden directories hold per-server virtual environments, not repo source.
    return sorted(
        path
        for path in (REPO_ROOT / directory).rglob("*.py")
        if "tests" not in path.parts and not any(part.startswith(".") for part in path.relative_to(REPO_ROOT).parts)
    )


def _classes(path: Path) -> list[ast.ClassDef]:
    return [node for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.ClassDef)]


def _methods(cls: ast.ClassDef) -> dict[str, ast.AST]:
    return {node.name: node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _location(path: Path, node: ast.AST) -> str:
    return f"{path.relative_to(REPO_ROOT)}:{node.lineno}"


def test_agents_serve_sessions_through_the_base_hooks() -> None:
    """The base agent owns seed and close bookkeeping; an agent implements only the state hooks."""
    violations = []
    for path in _sources("responses_api_agents"):
        for cls in _classes(path):
            methods = _methods(cls)
            overridden = sorted({"seed_agent_session", "close_agent_session"} & methods.keys())
            if overridden:
                violations.append(
                    f"{_location(path, cls)} overrides {overridden}; implement _seed_agent_session_state and "
                    "_close_agent_session_state instead"
                )
            if "_seed_agent_session_state" in methods and "_close_agent_session_state" not in methods:
                violations.append(
                    f"{_location(path, cls)} seeds sessions but does not implement _close_agent_session_state"
                )
    assert not violations, "\n".join(violations)


def test_resources_servers_close_through_the_base_route() -> None:
    """The base serves /close_session before subclass routes, so a server's own registration never runs."""
    violations = []
    for path in _sources("resources_servers"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"post", "add_api_route", "api_route"}
                and any(isinstance(arg, ast.Constant) and arg.value == "/close_session" for arg in node.args)
            ):
                violations.append(
                    f"{_location(path, node)} registers /close_session; override close_resources_session instead"
                )
        for cls in _classes(path):
            methods = _methods(cls)
            seed = methods.get("seed_session")
            if (
                seed is not None
                and "ResourcesSeedSessionRequest" in ast.unparse(seed)
                and "close_resources_session" not in methods
            ):
                violations.append(
                    f"{_location(path, cls)} accepts a typed seed but does not override close_resources_session"
                )
    assert not violations, "\n".join(violations)
