# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repo-wide checks that every agent and Resources Server serves sessions through the shared contract.

Servers run in their own environments, so these checks read source instead of importing each server. Behavior is
tested where it lives: the base agent's session bookkeeping in the base agent tests, and a Resources Server's seed and
close through ``nemo_gym.server_runtime.testing.session_conformance`` in that server's tests.
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
    # A shared base class, such as SandboxSessionResourcesServer, can serve the close for its subclasses.
    closing_bases = {
        cls.name
        for path in _sources("resources_servers")
        for cls in _classes(path)
        if "close_resources_session" in _methods(cls)
    }
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
                and not any(isinstance(base, ast.Name) and base.id in closing_bases for base in cls.bases)
            ):
                violations.append(
                    f"{_location(path, cls)} accepts an episode seed but does not override close_resources_session"
                )
    assert not violations, "\n".join(violations)


def _is_plain_409(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
    if name != "HTTPException":
        return False
    status = node.args[0] if node.args else next((kw.value for kw in node.keywords if kw.arg == "status_code"), None)
    return isinstance(status, ast.Constant) and status.value == 409


# Files whose 409s refuse something other than a session request.
_NOT_SESSION_CONFLICTS = {
    "resources_servers/genrm_compare/app.py",  # A comparison cohort or group that received conflicting members.
    "resources_servers/osworld/worker.py",  # A worker reserved by another deployment.
}


def test_session_refusals_raise_the_conflict_errors() -> None:
    """A 409 from a server or agent is a session conflict, so it raises the error that names one."""
    violations = []
    for directory, error in (
        ("resources_servers", "ResourcesSessionConflictError"),
        ("responses_api_agents", "AgentSessionConflictError"),
    ):
        for path in _sources(directory):
            if str(path.relative_to(REPO_ROOT)) in _NOT_SESSION_CONFLICTS:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if _is_plain_409(node):
                    violations.append(f"{_location(path, node)} raises HTTPException(409); raise {error} instead")
    assert not violations, "\n".join(violations)
