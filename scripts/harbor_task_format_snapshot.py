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
"""Record what Harbor's task format looks like at one Harbor revision, and say when it changed.

Gym reads `task.toml` through its own mirror of Harbor's `TaskConfig` (`nemo_gym/data/tasks/harbor/models.py`) and
carries a copy of two Harbor functions (`nemo_gym/data/tasks/harbor/harbor_compat.py`); it does not import the
`harbor` package. This script reads a Harbor checkout and writes the snapshot those two are tested against
(`tests/unit_tests/data/tasks/harbor/test_harbor_upstream_format.py`):

- the JSON schema of Harbor's `TaskConfig`, without its descriptions and titles;
- a hash of each Harbor function and constant `harbor_compat.py` copies, and whether Harbor still names a task's
  image `hb__<environment id>` through them;
- Harbor's version, commit and `git describe` revision.

Run it on the Harbor revision to compare with:

    git -C <harbor checkout> pull
    uv run python scripts/harbor_task_format_snapshot.py <harbor checkout> --check

`--check` writes nothing. It exits 1 when the checkout's task format differs from the committed snapshot, or when
one of the checkout's own `examples/tasks/**/task.toml` fails to parse in Gym's mirror or has a key the mirror does
not know. Without `--check` the snapshot is rewritten; the unit test then names every field Gym's mirror does not
yet follow.

The schema is rendered by the pydantic this script runs with, so a pydantic upgrade in Gym can change the snapshot
without a change in Harbor. Refresh the snapshot with the upgrade when that happens.
"""

import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
import subprocess
import sys
import tomllib
import types
from pathlib import Path
from typing import Any

from pydantic import ValidationError


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = (
    REPOSITORY_ROOT / "tests" / "unit_tests" / "data" / "tasks" / "harbor" / "fixtures" / "harbor_task_format.json"
)
HARBOR_URL = "https://github.com/harbor-framework/harbor"
TASK_CONFIG_MODULE = "src/harbor/models/task/config.py"
EXAMPLE_TASKS = "examples/tasks"
# The Harbor definitions `harbor_compat.py` copies: the name there, then where Harbor defines it.
VENDORED_DEFINITIONS = {
    "_CONTENT_HASH_IGNORE_NAMES": ("src/harbor/environments/definition.py", "_CONTENT_HASH_IGNORE_NAMES"),
    "environment_content_hash": ("src/harbor/environments/definition.py", "environment_content_hash"),
    "sanitize_docker_image_name": ("src/harbor/environments/docker/docker.py", "_sanitize_docker_image_name"),
}
# How Harbor's Docker environment names a task's image from the two functions above. Gym composes the same name
# (`nemo_gym/data/tasks/harbor/build_compose.py`), so a change here changes which image both look for.
IMAGE_NAME_SOURCE = "src/harbor/environments/docker/docker.py"
IMAGE_NAME_EXPRESSION = '_sanitize_docker_image_name(f"hb__{self.environment_id}")'
# JSON schema annotations that carry Harbor's prose, not the format.
_PROSE_KEYS = ("description", "title")


def definition_hash(source: str, name: str) -> str:
    """SHA-256 of the top-level function or assignment ``name`` in ``source``, comments and layout aside.

    A function is hashed under the name ``_`` so that a copy exposed under another name hashes the same.
    """
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            node.name = "_"
            return hashlib.sha256(ast.unparse(node).encode()).hexdigest()
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return hashlib.sha256(ast.unparse(node).encode()).hexdigest()
    raise LookupError(f"No top-level definition of {name}")


def _without_prose(node: Any, *, names_are_fields: bool = False) -> Any:
    """``node`` without description and title annotations. Keys of a ``properties`` or ``$defs`` object are field
    and model names, so they are kept whatever they are called."""
    if isinstance(node, list):
        return [_without_prose(item) for item in node]
    if not isinstance(node, dict):
        return node
    return {
        key: _without_prose(value, names_are_fields=key in ("properties", "$defs"))
        for key, value in node.items()
        if names_are_fields or key not in _PROSE_KEYS
    }


def task_config_schema(checkout: Path) -> dict[str, Any]:
    """The JSON schema of the checkout's ``TaskConfig``.

    The ``harbor`` package is registered empty, so only the task model's modules are imported from the checkout:
    Harbor's own ``__init__`` needs the package installed. Harbor's module imports ``toml`` to write ``task.toml``
    files; reading the schema calls none of it, so an empty module stands in where ``toml`` is not installed.
    """
    if not (checkout / TASK_CONFIG_MODULE).is_file():
        raise SystemExit(f"{checkout} is not a Harbor checkout: no {TASK_CONFIG_MODULE}")
    for name in [name for name in sys.modules if name == "harbor" or name.startswith("harbor.")]:
        del sys.modules[name]
    package = types.ModuleType("harbor")
    package.__path__ = [str(checkout / "src" / "harbor")]
    sys.modules["harbor"] = package
    stand_in = importlib.util.find_spec("toml") is None
    if stand_in:
        sys.modules["toml"] = types.ModuleType("toml")
    # Importing from the checkout must not leave `__pycache__` folders in it.
    write_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        from harbor.models.task.config import TaskConfig
    except ModuleNotFoundError as exc:
        missing = exc.name or "a module"
        if missing.split(".")[0] == "harbor":
            raise SystemExit(f"Harbor's task model imports `{missing}`, which this checkout does not have")
        raise SystemExit(f"Harbor's task model needs `{missing}`: run this script with `uv run --with {missing}`")
    finally:
        sys.dont_write_bytecode = write_bytecode
        if stand_in:
            del sys.modules["toml"]
    return _without_prose(TaskConfig.model_json_schema())


def _git(checkout: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(checkout), *args], capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(f"{checkout} must be a git checkout of Harbor, so that the snapshot names its commit")
    return result.stdout.strip()


def read_snapshot(checkout: Path) -> dict[str, Any]:
    """The snapshot of ``checkout``, as written to :data:`SNAPSHOT_PATH`."""
    schema = task_config_schema(checkout)
    vendored: dict[str, str] = {}
    for name, (path, upstream_name) in VENDORED_DEFINITIONS.items():
        try:
            vendored[name] = definition_hash((checkout / path).read_text(), upstream_name)
        except (OSError, LookupError) as exc:
            raise SystemExit(
                f"Harbor no longer defines `{upstream_name}` in {path} ({exc}): find where it moved and update "
                "VENDORED_DEFINITIONS and harbor_compat.py"
            )
    return {
        "source": HARBOR_URL,
        "license": "Apache-2.0",
        # Absent when Harbor's version is not written in pyproject.toml; the revision below still names it.
        "version": tomllib.loads((checkout / "pyproject.toml").read_text()).get("project", {}).get("version"),
        "commit": _git(checkout, "rev-parse", "HEAD"),
        # The commit relative to Harbor's latest release tag: the version alone does not say whether it is the tag.
        "revision": _git(checkout, "describe", "--tags", "--always"),
        "vendored_definitions": vendored,
        "image_name_expression": (
            IMAGE_NAME_EXPRESSION if IMAGE_NAME_EXPRESSION in (checkout / IMAGE_NAME_SOURCE).read_text() else None
        ),
        "task_config_schema": schema,
    }


def example_task_problems(checkout: Path) -> list[str]:
    """What Gym's mirror says about each ``task.toml`` under the checkout's example tasks: nothing when it parses
    and every key is known."""
    from nemo_gym.data.tasks.harbor import HarborTaskConfig

    paths = sorted((checkout / EXAMPLE_TASKS).rglob("task.toml"))
    if not paths:
        # Harbor moved its examples: say so, or the check would pass having parsed nothing.
        return [f"no task.toml under {EXAMPLE_TASKS}"]
    problems: list[str] = []
    for path in paths:
        name = path.relative_to(checkout).as_posix()
        try:
            data = tomllib.loads(path.read_text())
        except ValueError as exc:  # not TOML, or not UTF-8
            problems.append(f"{name}: {exc}")
            continue
        problems += [f"{name}: unknown key `{key}`" for key in HarborTaskConfig.unknown_keys(data)]
        try:
            HarborTaskConfig.model_validate(data)
        except ValidationError as exc:
            problems.append(f"{name}: {exc}")
    return problems


def _format(snapshot: dict[str, Any]) -> str:
    return json.dumps(snapshot, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkout", type=Path, help="a Harbor checkout at the revision to record")
    parser.add_argument("--check", action="store_true", help="compare with the committed snapshot; write nothing")
    args = parser.parse_args(argv)
    checkout = args.checkout.resolve()

    snapshot = read_snapshot(checkout)
    problems = example_task_problems(checkout)
    for problem in problems:
        print(f"Gym's mirror does not read Harbor's example tasks: {problem}")
    print(f"Harbor {snapshot['version']} ({snapshot['revision']}, {snapshot['commit']})")

    if not args.check:
        SNAPSHOT_PATH.write_text(_format(snapshot))
        print(f"Wrote {SNAPSHOT_PATH.relative_to(REPOSITORY_ROOT)}")
        return 1 if problems else 0

    committed = json.loads(SNAPSHOT_PATH.read_text())
    # The revision alone moving is not a change of format.
    compared = ("vendored_definitions", "image_name_expression", "task_config_schema")
    changed = {key for key in compared if snapshot[key] != committed.get(key)}
    if changed:
        print(f"Harbor's task format changed since {committed['version']} ({committed['revision']}):")
        diff = difflib.unified_diff(
            _format({key: committed.get(key) for key in sorted(changed)}).splitlines(),
            _format({key: snapshot[key] for key in sorted(changed)}).splitlines(),
            "committed snapshot",
            "this checkout",
            lineterm="",
        )
        print("\n".join(diff))
    else:
        print("Harbor's task format matches the committed snapshot")
    return 1 if changed or problems else 0


if __name__ == "__main__":
    sys.exit(main())
