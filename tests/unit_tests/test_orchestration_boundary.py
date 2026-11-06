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
``tests/`` is not scanned: those tests move together with the code they cover.
"""

import ast
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


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imported_modules(node: ast.AST, current: str, is_package: bool) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        base = node.module or ""
        if node.level:
            pkg = current.split(".") if is_package else current.split(".")[:-1]
            pkg = pkg[: len(pkg) - (node.level - 1)]
            base = ".".join(pkg + ([node.module] if node.module else []))
        return [base] + [f"{base}.{alias.name}" for alias in node.names]
    return []


def _violations() -> list[str]:
    found = []
    for path in sorted(p for root in SCANNED_ROOTS if root.is_dir() for p in root.rglob("*.py")):
        if path.is_relative_to(ORCHESTRATION) or path in ALLOWED_IMPORTERS:
            continue
        source = path.read_text()
        lines = source.splitlines()
        current, is_package = _module_name(path), path.name == "__init__.py"
        for node in ast.walk(ast.parse(source)):
            targets = _imported_modules(node, current, is_package)
            if any(t == "nemo_gym.orchestration" or t.startswith("nemo_gym.orchestration.") for t in targets):
                found.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {lines[node.lineno - 1].strip()}")
    return found


def test_only_cli_entrypoint_imports_orchestration() -> None:
    violations = _violations()
    assert not violations, (
        "nemo_gym.orchestration is a Gym-aware orchestrator, not a home for core functionality: deleting it "
        "should leave only the CLI wiring in nemo_gym/cli/main.py to change. Move the code being imported "
        "into the package that owns it:\n" + "\n".join(violations)
    )
