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


# Calls that build an HTTP refusal, by name, and the position of their status argument.
_REFUSALS = {"HTTPException": 0, "StarletteHTTPException": 0, "Response": 1, "JSONResponse": 1}


def _is_409(status: ast.expr | None) -> bool:
    if isinstance(status, ast.Attribute) and status.attr == "value":
        status = status.value
    if isinstance(status, ast.Constant):
        return status.value == 409
    # A named status, such as ``status.HTTP_409_CONFLICT`` or ``HTTPStatus.CONFLICT``.
    name = status.attr if isinstance(status, ast.Attribute) else getattr(status, "id", None)
    return name in {"HTTP_409_CONFLICT", "CONFLICT"}


def _plain_409s(path: Path) -> list[tuple[ast.Call, str | None]]:
    """Each call in ``path`` that builds a plain 409 refusal, with the name of the function that encloses it."""
    found = []

    def visit(node: ast.AST, function: str | None) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node.name
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            if name in _REFUSALS:
                position = _REFUSALS[name]
                status = node.args[position] if len(node.args) > position else None
                status = status or next((kw.value for kw in node.keywords if kw.arg == "status_code"), None)
                if _is_409(status):
                    found.append((node, function))
        for child in ast.iter_child_nodes(node):
            visit(child, function)

    visit(ast.parse(path.read_text()), None)
    return found


# 409s that refuse something other than a session request, by file and enclosing function (``None`` for the file).
_NOT_SESSION_CONFLICTS = {
    # A comparison cohort or group that received conflicting members.
    ("resources_servers/genrm_compare/app.py", "verify"),
    ("resources_servers/genrm_compare/app.py", "_resolve_verify_cohort"),
    # A worker reserved by another deployment.
    ("resources_servers/osworld/worker.py", "create_container"),
    # The tool runner inside the task sandbox, which answers its own protocol and cannot import Gym.
    ("resources_servers/partner_if/tool_runner.py", None),
}


def test_session_refusals_raise_the_conflict_errors() -> None:
    """A 409 from a server or agent is a session conflict, so it raises the error that names one.

    The check reads 409s written as a literal or a named status. A status computed at runtime is not checked.
    """
    violations = []
    for directory, error in (
        ("resources_servers", "ResourcesSessionConflictError"),
        ("responses_api_agents", "AgentSessionConflictError"),
    ):
        for path in _sources(directory):
            relative = path.relative_to(REPO_ROOT).as_posix()
            if (relative, None) in _NOT_SESSION_CONFLICTS:
                continue
            for node, function in _plain_409s(path):
                if (relative, function) not in _NOT_SESSION_CONFLICTS:
                    violations.append(f"{_location(path, node)} answers a plain 409; raise {error} instead")
    assert not violations, "\n".join(violations)
