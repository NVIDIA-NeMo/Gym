# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Only the CLI entrypoint may import ``nemo_gym.orchestration``; anything else belongs in a lower package.

Covers ``nemo_gym/`` and the code built on it (servers with their own tests, benchmarks, scripts). The core suite under
``tests/`` is not scanned: those tests move together with the code they cover. Only files git knows about are scanned
(tracked, or untracked and not ignored), so installed copies of Gym, such as a server's `.venv`, are left out.

``KNOWN_VIOLATIONS`` lists the imports that break the rule today, as (file, imported module). Each goes away when the
imported code moves into the package that owns it. The test fails on an import that is not listed and on a listed
entry that no longer matches an import, so the list can only shrink.
"""

import ast
import functools
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE = REPO_ROOT / "nemo_gym"
ORCHESTRATION = CORE / "orchestration"
SCANNED_ROOTS = [
    REPO_ROOT / name
    for name in (
        "nemo_gym",
        "benchmarks",
        "environment_servers",
        "environments",
        "examples",
        "nemotron_recipes",
        "resources_servers",
        "responses_api_agents",
        "responses_api_models",
        "scripts",
    )
]
ALLOWED_IMPORTERS = {CORE / "cli" / "main.py"}

KNOWN_VIOLATIONS = {
    ("nemo_gym/cli/eval.py", "nemo_gym.orchestration.recovery"),
    ("nemo_gym/cli/utils.py", "nemo_gym.orchestration.executors.resume_script"),
    ("nemo_gym/eval/rollout_reverification.py", "nemo_gym.orchestration.recovery"),
    ("resources_servers/agentic_turn_simulation/tests/test_app.py", "nemo_gym.orchestration.recovery"),
    ("resources_servers/bbq/tests/test_app.py", "nemo_gym.orchestration.recovery"),
    ("resources_servers/deepswe/tests/test_app.py", "nemo_gym.orchestration.recovery"),
    ("resources_servers/deepswe_external1/tests/test_app.py", "nemo_gym.orchestration.recovery"),
    ("resources_servers/gdp_pdf/tests/test_app.py", "nemo_gym.orchestration.recovery"),
    ("resources_servers/math_with_autograder/tests/test_app.py", "nemo_gym.orchestration.recovery"),
    ("resources_servers/physics_judge/tests/test_app.py", "nemo_gym.orchestration.recovery"),
    (
        "responses_api_agents/harbor_agent_general/tests/test_skillsbench_observations.py",
        "nemo_gym.orchestration.recovery",
    ),
    ("responses_api_agents/miniswe_sandboxed_agent/tests/test_server.py", "nemo_gym.orchestration.recovery"),
}


def _is_orchestration(module: str) -> bool:
    return module == "nemo_gym.orchestration" or module.startswith("nemo_gym.orchestration.")


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _orchestration_imports(node: ast.AST, current: str, is_package: bool) -> list[str]:
    """The orchestration modules `node` imports, named by the module it imports from."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names if _is_orchestration(alias.name)]
    if isinstance(node, ast.ImportFrom):
        base = node.module or ""
        if node.level:
            pkg = current.split(".") if is_package else current.split(".")[:-1]
            pkg = pkg[: len(pkg) - (node.level - 1)]
            base = ".".join(pkg + ([node.module] if node.module else []))
        if _is_orchestration(base):
            return [base]
        return [f"{base}.{alias.name}" for alias in node.names if _is_orchestration(f"{base}.{alias.name}")]
    return []


def _python_files(repo_root: Path, scanned_roots: list[Path]) -> list[Path]:
    """The Python files under `scanned_roots` that git tracks, or that it does not ignore."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "*.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    paths = (repo_root / name for name in listed if name)
    return sorted(p for p in paths if p.is_file() and any(p.is_relative_to(root) for root in scanned_roots))


@functools.cache
def _imports() -> dict[tuple[str, str], list[str]]:
    """Every import of orchestration outside it and the CLI entrypoint, as (file, module) to its source lines."""
    found: dict[tuple[str, str], list[str]] = {}
    for path in _python_files(REPO_ROOT, SCANNED_ROOTS):
        if path.is_relative_to(ORCHESTRATION) or path in ALLOWED_IMPORTERS:
            continue
        source = path.read_text()
        lines = source.splitlines()
        current, is_package = _module_name(path), path.name == "__init__.py"
        relative = str(path.relative_to(REPO_ROOT))
        for node in ast.walk(ast.parse(source)):
            for module in _orchestration_imports(node, current, is_package):
                line = f"{relative}:{node.lineno}: {lines[node.lineno - 1].strip()}"
                found.setdefault((relative, module), []).append(line)
    return found


def test_only_cli_entrypoint_imports_orchestration() -> None:
    found = _imports()
    new = [line for key, lines in sorted(found.items()) if key not in KNOWN_VIOLATIONS for line in lines]
    assert not new, (
        "nemo_gym.orchestration is a Gym-aware orchestrator, not a home for core functionality: deleting it "
        "should leave only the CLI wiring in nemo_gym/cli/main.py to change. Move the code being imported "
        "into the package that owns it:\n" + "\n".join(new)
    )


def test_known_violations_are_still_violations() -> None:
    stale = sorted(KNOWN_VIOLATIONS - set(_imports()))
    assert not stale, "These no longer import orchestration; remove them from KNOWN_VIOLATIONS:\n" + "\n".join(
        f"{path}: {module}" for path, module in stale
    )


def test_python_files_skips_ignored_trees_and_unscanned_roots(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text(".venv/\n")
    for relative in ("scripts/a.py", "scripts/.venv/lib/installed.py", "elsewhere/b.py"):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text("")

    assert _python_files(tmp_path, [tmp_path / "scripts"]) == [tmp_path / "scripts" / "a.py"]
