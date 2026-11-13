# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every resources, agent and environment server either runs with several workers or refuses them.

With `num_workers > 1`, `run_webserver` hands uvicorn the import string `<entrypoint module>:app`,
and each worker imports the entrypoint and serves its module-level `app`.
An entrypoint without that `app` makes every worker exit at startup.
A server that cannot run with several workers must say so instead,
by overriding `multiple_workers_unsupported_reason` so that `run_webserver` refuses before uvicorn starts.
A server that refuses only in some configurations needs both.

These checks read source only,
because each server runs in its own virtual environment and its dependencies are not installed here.
"""

import ast
import os
import warnings
from functools import cache
from pathlib import Path

import pytest
import yaml

from nemo_gym import PARENT_DIR


SERVER_KINDS = ("resources_servers", "responses_api_agents", "environment_servers")
REASON_METHOD = "multiple_workers_unsupported_reason"

# Entrypoints that configs name but that start no server, so neither the adapter nor a refusal applies.
NOT_SERVERS = {
    # The base class of Gymnasium-style servers; its config only declares the module so its tests run.
    "resources_servers/gymnasium/base.py",
    # The local SEC search library that finance_sec_search and finance_agent_v2 import; its config says so.
    "resources_servers/sec_local_index/local_edgar_search.py",
}


def _parse(path: Path) -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(path.read_text())


def _config_entrypoints(node: object, found: set[tuple[str, str, str]]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in SERVER_KINDS and isinstance(value, dict):
                for server_dir, block in value.items():
                    if isinstance(block, dict) and isinstance(block.get("entrypoint"), str):
                        found.add((key, server_dir, block["entrypoint"]))
            _config_entrypoints(value, found)
    elif isinstance(node, list):
        for item in node:
            _config_entrypoints(item, found)


def _config_files(root: Path) -> list[Path]:
    # Configs live beside their servers and also under benchmarks/, environments/ and elsewhere.
    # Dot directories (virtual environments among them) and caches are pruned before they are walked.
    found = []
    for directory, subdirectories, files in os.walk(root):
        subdirectories[:] = [
            name for name in subdirectories if not name.startswith(".") and name not in ("node_modules", "cache")
        ]
        found.extend(Path(directory, name) for name in files if name.endswith((".yaml", ".yml")))
    return found


@cache
def configured_entrypoints(root: Path) -> frozenset[tuple[str, str, str]]:
    """Every `(server kind, server directory, entrypoint)` a config under `root` names."""
    found: set[tuple[str, str, str]] = set()
    for config in _config_files(root):
        try:
            documents = list(yaml.safe_load_all(config.read_text()))
        except (yaml.YAMLError, UnicodeDecodeError):
            continue
        for document in documents:
            _config_entrypoints(document, found)
    return frozenset(found)


def entrypoint_modules(root: Path) -> dict[Path, bool]:
    """Every server entrypoint module, mapped to whether a config names it.

    That is each module a config names, and each `app.py` in a server directory,
    since a server can also be started from a config outside the repository.
    """
    root = root.resolve()
    modules = {path: False for kind in SERVER_KINDS for path in (root / kind).glob("*/app.py")}
    for kind, server_dir, entrypoint in configured_entrypoints(root):
        module = (root / kind / server_dir / entrypoint).resolve()
        # Configs that name a server directory or module that does not exist cannot start, so they are skipped.
        if module.suffix == ".py" and module.is_file():
            modules[module] = True
    return {
        module: named for module, named in sorted(modules.items()) if str(module.relative_to(root)) not in NOT_SERVERS
    }


def entrypoints_outside_their_server_directory(root: Path) -> set[Path]:
    """Modules a config names by a path that leaves the server's directory.

    `run_webserver` builds uvicorn's import string from the entrypoint path,
    so a module reached that way cannot be imported by multi-worker uvicorn even if it has the module-level `app`.
    """
    root = root.resolve()
    outside = set()
    for kind, server_dir, entrypoint in configured_entrypoints(root):
        server_root = (root / kind / server_dir).resolve()
        module = (server_root / entrypoint).resolve()
        if module.is_file() and (".." in Path(entrypoint).parts or not module.is_relative_to(server_root)):
            outside.add(module)
    return outside


def _served_class_names(tree: ast.Module) -> set[str]:
    return {
        node.func.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_webserver"
        and isinstance(node.func.value, ast.Name)
    }


def has_adapter(tree: ast.Module) -> bool:
    """Whether the module assigns a module-level `app` under `is_nemo_gym_fastapi_entrypoint(__file__)`."""
    branches = [node for node in tree.body if isinstance(node, ast.If)]
    while branches:
        branch = branches.pop()
        branches.extend(node for node in branch.orelse if isinstance(node, ast.If))
        test = branch.test
        if not (
            isinstance(test, ast.Call)
            and isinstance(test.func, ast.Name)
            and test.func.id == "is_nemo_gym_fastapi_entrypoint"
        ):
            continue
        for statement in branch.body:
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "app" for target in statement.targets
            ):
                return True
    return False


@cache
def _module_tree(path: Path) -> ast.Module:
    return _parse(path)


def _module_path(root: Path, module: str) -> Path | None:
    base = root.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _find_class(root: Path, path: Path, name: str, seen: set[tuple[Path, str]]) -> tuple[Path, ast.ClassDef] | None:
    """The class `name` refers to in the module at `path`, following imports within the repository."""
    if (path, name) in seen:
        return None
    seen.add((path, name))
    tree = _module_tree(path)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return path, node
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module is None or node.level:
            continue
        for alias in node.names:
            if (alias.asname or alias.name) == name:
                source = _module_path(root, node.module)
                if source is not None:
                    return _find_class(root, source, alias.name, seen)
    return None


def reason_override(root: Path, path: Path, name: str) -> ast.FunctionDef | None:
    """The reason method the class, or a parent class among the servers in the repository, overrides.

    Parents in `nemo_gym` do not count: `SimpleServer` defines the method only to return None.
    """
    found = _find_class(root, path, name, set())
    if found is None:
        return None
    class_path, class_node = found
    if not any(class_path.is_relative_to(root / kind) for kind in SERVER_KINDS):
        return None
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == REASON_METHOD:
            return item
    for base in class_node.bases:
        if isinstance(base, ast.Subscript):
            base = base.value
        if isinstance(base, ast.Name):
            override = reason_override(root, class_path, base.id)
            if override is not None:
                return override
    return None


def refuses_in_every_configuration(override: ast.FunctionDef) -> bool:
    """Whether every path through the override returns a reason.

    Only a string literal or an f-string counts as a reason.
    Any other return value, such as a variable or a conditional expression, might be None,
    so an override that returns one, or that can fall off its end, refuses only in some configurations.
    """
    if not override.body or not isinstance(override.body[-1], ast.Return):
        return False
    for node in ast.walk(override):
        if isinstance(node, ast.Return) and not (
            isinstance(node.value, ast.JoinedStr)
            or (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
        ):
            return False
    return True


def contract_violations(root: Path) -> list[str]:
    violations = []
    outside = entrypoints_outside_their_server_directory(root)
    for path, named_by_config in entrypoint_modules(root).items():
        tree = _module_tree(path)
        relative = path.relative_to(root.resolve())
        class_names = _served_class_names(tree)
        if not class_names:
            # An app.py that starts nothing is a library module; a module a config names must start a server.
            if named_by_config:
                violations.append(
                    f"{relative} is a configured entrypoint but calls no `<ServerClass>.run_webserver()`, "
                    "so this check cannot tell which server it starts. Add it to NOT_SERVERS if it starts none."
                )
            continue
        if has_adapter(tree):
            if path in outside:
                violations.append(
                    f"{relative} has the module-level `app`, but a config names it by a path outside its server "
                    "directory, which multi-worker uvicorn cannot import. Refuse several workers instead."
                )
            continue
        for name in sorted(class_names):
            override = reason_override(root.resolve(), path, name)
            if override is not None and not refuses_in_every_configuration(override):
                violations.append(
                    f"{relative}: {name} refuses several workers only in some configurations, "
                    "so it also needs the module-level `app` for the others:\n"
                    f"    elif is_nemo_gym_fastapi_entrypoint(__file__):\n"
                    f"        app = {name}.run_webserver()  # noqa: F401"
                )
            elif override is None:
                violations.append(
                    f"{relative}: {name} has neither the module-level `app` that multi-worker uvicorn imports "
                    f"nor a refusal. If the server is safe with several workers, end the module with\n"
                    f"    elif is_nemo_gym_fastapi_entrypoint(__file__):\n"
                    f"        app = {name}.run_webserver()  # noqa: F401\n"
                    f"Otherwise override `{REASON_METHOD}` on {name} and return why it needs num_workers=1."
                )
    return violations


def test_every_server_runs_with_several_workers_or_refuses_them() -> None:
    violations = contract_violations(PARENT_DIR)
    assert not violations, "\n\n".join(violations)


def test_the_scan_covers_entrypoints_outside_app_py() -> None:
    modules = {str(path.relative_to(PARENT_DIR.resolve())) for path in entrypoint_modules(PARENT_DIR)}

    assert {
        "responses_api_agents/langgraph_agent/rewoo_agent.py",
        "resources_servers/aviary/gsm8k_app.py",
        "resources_servers/aviary/notebook_app.py",
        "resources_servers/legal_agent_bench/harbor_bridge.py",
    } <= modules


def _write(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for kind in (*SERVER_KINDS, "benchmarks"):
        (tmp_path / kind).mkdir()
    return tmp_path


class TestContractChecker:
    def test_server_without_adapter_or_refusal_is_reported(self, repo: Path) -> None:
        _write(repo, "resources_servers/plain/app.py", "class Plain:\n    pass\n\nPlain.run_webserver()\n")

        [violation] = contract_violations(repo)

        assert violation.startswith("resources_servers/plain/app.py: Plain has neither")
        assert "app = Plain.run_webserver()  # noqa: F401" in violation
        assert REASON_METHOD in violation

    def test_adapter_satisfies_the_contract(self, repo: Path) -> None:
        _write(
            repo,
            "responses_api_agents/safe/app.py",
            "class Safe:\n    pass\n\n"
            "if __name__ == '__main__':\n    Safe.run_webserver()\n"
            "elif is_nemo_gym_fastapi_entrypoint(__file__):\n    app = Safe.run_webserver()\n",
        )

        assert contract_violations(repo) == []

    def test_refusal_inherited_from_a_parent_server_in_another_module(self, repo: Path) -> None:
        _write(
            repo,
            "resources_servers/family/app.py",
            "class Parent:\n    def multiple_workers_unsupported_reason(self):\n        return 'state'\n",
        )
        _write(
            repo,
            "resources_servers/family/child_app.py",
            "from resources_servers.family.app import Parent as Base\n\n"
            "class Child(Base[int]):\n    pass\n\nChild.run_webserver()\n",
        )
        _write(repo, "resources_servers/family/configs/family.yaml", _config("family", "child_app.py"))

        assert contract_violations(repo) == []

    def test_a_refusal_on_a_gym_base_class_does_not_count(self, repo: Path) -> None:
        _write(
            repo,
            "nemo_gym/server_runtime/base_server.py",
            "class SimpleServer:\n    def multiple_workers_unsupported_reason(self):\n        return None\n",
        )
        _write(
            repo,
            "environment_servers/env/app.py",
            "from nemo_gym.server_runtime.base_server import SimpleServer\n\n"
            "class Env(SimpleServer):\n    pass\n\nEnv.run_webserver()\n",
        )

        [violation] = contract_violations(repo)

        assert violation.startswith("environment_servers/env/app.py: Env has neither")

    def test_a_configured_entrypoint_that_starts_no_server_is_reported(self, repo: Path) -> None:
        _write(repo, "resources_servers/lib/app.py", "class Library:\n    pass\n")
        _write(repo, "resources_servers/lib/helpers.py", "def helper():\n    pass\n")
        _write(repo, "resources_servers/lib/configs/lib.yaml", _config("lib", "helpers.py"))

        [violation] = contract_violations(repo)

        assert violation.startswith("resources_servers/lib/helpers.py is a configured entrypoint")

    def test_entrypoints_named_by_configs_outside_the_server_directory_are_checked(self, repo: Path) -> None:
        _write(repo, "responses_api_agents/multi/app.py", "class A:\n    pass\n\nA.run_webserver()\n")
        _write(repo, "responses_api_agents/multi/second_agent.py", "class B:\n    pass\n\nB.run_webserver()\n")
        _write(
            repo,
            "benchmarks/bench/config.yaml",
            "bench_agent:\n  responses_api_agents:\n    multi:\n      entrypoint: second_agent.py\n",
        )

        violations = contract_violations(repo)

        assert [violation.split(":")[0] for violation in violations] == [
            "responses_api_agents/multi/app.py",
            "responses_api_agents/multi/second_agent.py",
        ]

    def test_a_refusal_that_can_return_none_also_needs_the_adapter(self, repo: Path) -> None:
        _write(
            repo,
            "resources_servers/sometimes/app.py",
            "class Sometimes:\n"
            "    def multiple_workers_unsupported_reason(self):\n"
            "        if self.config.sidecar:\n"
            "            return 'sidecar'\n"
            "        return None\n\n"
            "Sometimes.run_webserver()\n",
        )

        [violation] = contract_violations(repo)

        assert violation.startswith(
            "resources_servers/sometimes/app.py: Sometimes refuses several workers only in some"
        )

    def test_a_refusal_in_every_configuration_needs_no_adapter(self, repo: Path) -> None:
        _write(
            repo,
            "resources_servers/never/app.py",
            "class Never:\n"
            "    def multiple_workers_unsupported_reason(self):\n"
            "        if self.config.sidecar:\n"
            "            return 'sidecar'\n"
            "        return 'state'\n\n"
            "Never.run_webserver()\n",
        )

        assert contract_violations(repo) == []

    @pytest.mark.parametrize(
        "body",
        [
            "return 'sidecar' if self.config.sidecar else None",
            "return self._reason()",
            "reason = None\n        return reason",
        ],
    )
    def test_a_computed_reason_might_be_none_so_it_also_needs_the_adapter(self, repo: Path, body: str) -> None:
        _write(
            repo,
            "resources_servers/computed/app.py",
            "class Computed:\n"
            f"    def multiple_workers_unsupported_reason(self):\n        {body}\n\n"
            "Computed.run_webserver()\n",
        )

        [violation] = contract_violations(repo)

        assert violation.startswith("resources_servers/computed/app.py: Computed refuses several workers only in some")

    def test_an_f_string_reason_refuses_in_every_configuration(self, repo: Path) -> None:
        _write(
            repo,
            "resources_servers/formatted/app.py",
            "class Formatted:\n"
            "    def multiple_workers_unsupported_reason(self):\n"
            "        return f'{self.config.name} keeps state'\n\n"
            "Formatted.run_webserver()\n",
        )

        assert contract_violations(repo) == []

    def test_an_adapter_uvicorn_cannot_import_is_reported(self, repo: Path) -> None:
        _write(
            repo,
            "resources_servers/bridge/bridge.py",
            "class Bridge:\n    pass\n\n"
            "if __name__ == '__main__':\n    Bridge.run_webserver()\n"
            "elif is_nemo_gym_fastapi_entrypoint(__file__):\n    app = Bridge.run_webserver()\n",
        )
        _write(
            repo,
            "benchmarks/bench/config.yaml",
            "bench:\n  resources_servers:\n    other:\n      entrypoint: ../../resources_servers/bridge/bridge.py\n",
        )

        [violation] = contract_violations(repo)

        assert violation.startswith("resources_servers/bridge/bridge.py has the module-level `app`, but a config")


def _config(server_dir: str, entrypoint: str) -> str:
    return f"{server_dir}:\n  resources_servers:\n    {server_dir}:\n      entrypoint: {entrypoint}\n"
