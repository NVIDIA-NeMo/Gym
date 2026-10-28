# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python verifiers: which module grades a task, and the ``test.sh`` shim that runs it inside the sandbox.

A task may grade with ``tests/verifier.py`` instead of ``tests/test.sh``; a dataset may ship one ``verifier.py``
beside ``dataset.toml`` that grades every task without a ``tests/`` folder. ``tests/test.sh`` always wins when it is
present: many hub datasets ship a ``tests/verifier.py`` that their ``test.sh`` calls as a helper, and such a task is a
plain ``test.sh`` task to Gym (the module is uploaded with ``tests/`` like any other file). The module's ``verify``
function (see :mod:`nemo_gym.verifiers`) never runs on the host. The ``harbor`` environment server uploads a shim folder
over ``/tests`` after the task's own ``tests/``: a ``test.sh`` that finds a Python (3.8 or newer, :data:`PYTHON_FLOOR`;
an older one is refused with the version named, exit 127, no reward), the runner, a vendored copy
of ``nemo_gym.verifiers``, the task's fields and, at run time, the agent's final response, and runs the
verifier with the workspace path. ``gym dataset publish --harbor`` writes the same shim into each expanded
task's ``tests/`` (without the response, which Harbor cannot provide): the task's own ``tests/verifier.py`` stays
where it is and the shim runs it there, a dataset-level module travels under ``tests/.ng/verifier.py``, so the
published folder grades by ``tests/test.sh`` the same way under Harbor in shared verifier mode, and loads back into
Gym as a plain ``test.sh`` task. Publish refuses a Python
verifier in separate mode: Harbor uploads nothing into a separate verifier container, so the shim could not reach it.

The shim clears any ``reward.json`` or ``reward.txt`` already in ``/logs/verifier`` before the runner starts, and the
runner does the same before ``verify`` runs. Harbor does not: in shared mode its trial bind-mounts the verifier log
folder into the agent's container for the whole run and reads ``reward.json`` before ``reward.txt``, so a file the agent
planted there would otherwise stand as the verdict (beside, or instead of, the runner's own). Gym's server empties the
folder before every upload; the published artifact is graded by Harbor, which is why the shim clears it itself.
"""

from __future__ import annotations

import ast
import json
import logging
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import nemo_gym.verifiers as verifiers_package
from nemo_gym.tasks.harbor.errors import HarborTaskError


logger = logging.getLogger(__name__)

VERIFIER_FILE = "verifier.py"
TEST_SCRIPT = "test.sh"
# Inside the uploaded tests folder: the shim's own files, apart from test.sh.
SHIM_DIR = ".ng"
RUNNER_FILE = "run_verifier.py"
TASK_DATA_FILE = "task_data.json"
RESPONSE_FILE = "response.json"
# The in-sandbox tests folder, Harbor's.
TESTS_DIR = "/tests"
REWARD_DIR = "/logs/verifier"
# The oldest Python the runner and the vendored `nemo_gym.verifiers` run under (they keep to it: no `str.removesuffix`,
# no 3.9+ syntax). The shim's test.sh refuses an older interpreter with the version named rather than letting
# verify() die in a SyntaxError or AttributeError with no reward and no hint at the cause.
PYTHON_FLOOR = (3, 8)


class PythonVerifierError(HarborTaskError):
    """A task's Python verifier is not usable (a module the shim cannot wrap)."""


def python_verifier_for(task_folder: Path, dataset_root: Path) -> Path | None:
    """The module that grades ``task_folder``, or ``None`` when ``tests/test.sh`` does.

    ``tests/verifier.py`` when the task has one and no ``tests/test.sh``; else the dataset root's ``verifier.py``
    when the task has no ``tests/`` folder at all. ``tests/test.sh`` wins whenever it exists: on the hub a
    ``tests/verifier.py`` next to it is the helper that ``test.sh`` runs (the shape of whole datasets, such as
    multimedia-terminalbench, osworld-verified or o11y-bench), not a second grader, so the task is a ``test.sh`` task
    and the module an ordinary test file. A ``tests/`` folder holding neither is left to ``test.sh`` grading
    (``None``): Harbor admits that shape when a separate verifier image ships its own ``/tests/test.sh``, and the
    loader kept admitting it; such a task without one fails at verification with no ``test.sh`` to run, as before.
    """
    tests = Path(task_folder) / "tests"
    if tests.is_dir():
        script = tests / TEST_SCRIPT
        module = tests / VERIFIER_FILE
        if script.is_file():
            if module.is_file():
                logger.debug("%s: tests/test.sh grades the task; tests/verifier.py is uploaded with it", task_folder)
            return None
        if module.is_file():
            return module
        return None
    shared = Path(dataset_root) / VERIFIER_FILE
    if shared.is_file():
        return shared
    return None


def declared_arguments(module_path: Path) -> frozenset[str] | None:
    """The parameter names of ``def verify`` in the module, read from its source without importing it.

    ``None`` when the module defines ``verify`` some other way (``verify = partial(...)``), which the host cannot
    resolve without running task code; the in-sandbox runner then decides at run time.
    """
    try:
        tree = ast.parse(Path(module_path).read_text(), filename=str(module_path))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        raise PythonVerifierError(f"{module_path}: {exc}") from exc
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "verify":
            names = [arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
            return frozenset(names)
    return None


def reads_response(module_path: Path) -> bool:
    """Whether ``verify`` declares ``response``: such a verifier cannot be published to Harbor."""
    declared = declared_arguments(module_path)
    return declared is not None and "response" in declared


@dataclass(frozen=True)
class VerifierShim:
    """What the shim needs to know about the task it wraps."""

    verifier: Path
    # Where the verifier module is in the sandbox: `/tests/verifier.py` when the task's own tests/ carries it,
    # `/tests/.ng/verifier.py` when the shim carries a dataset-level one.
    in_task_tests: bool
    workspace: str | None
    task_data: Mapping[str, Any]
    response: Mapping[str, Any] | None = None

    @property
    def remote_verifier(self) -> str:
        return f"{TESTS_DIR}/{VERIFIER_FILE}" if self.in_task_tests else f"{TESTS_DIR}/{SHIM_DIR}/{VERIFIER_FILE}"


def test_script(shim: VerifierShim) -> str:
    """The ``test.sh`` the sandbox runs: clear stale reward files, find a Python of at least :data:`PYTHON_FLOOR`,
    hand the runner the paths, let it write the reward. No Python, or an older one, is exit 127 with the reason on
    stderr and no reward (the missing-reward outcome)."""
    workspace = shlex.quote(shim.workspace) if shim.workspace else '"$PWD"'
    arguments = [
        f"--verifier {shim.remote_verifier}",
        f"--workspace {workspace}",
        f"--task-data {TESTS_DIR}/{SHIM_DIR}/{TASK_DATA_FILE}",
        f"--reward-dir {REWARD_DIR}",
    ]
    if shim.response is not None:
        arguments.append(f"--response {TESTS_DIR}/{SHIM_DIR}/{RESPONSE_FILE}")
    return (
        "#!/usr/bin/env bash\n"
        "# Written by NeMo Gym: runs the task's Python verifier (verify() in verifier.py) inside the sandbox.\n"
        "# The reward lands in /logs/verifier/reward.txt or reward.json, as Harbor expects.\n"
        "set -u\n"
        f"mkdir -p {REWARD_DIR}\n"
        "# Harbor shares /logs/verifier with the agent's container and reads reward.json before reward.txt: a file\n"
        "# planted there during the run must never stand as the verdict, so both go before the verifier runs.\n"
        f"rm -rf {REWARD_DIR}/reward.json {REWARD_DIR}/reward.txt\n"
        'PY=$(command -v python3 2>/dev/null || command -v python 2>/dev/null) || { echo "the image has no python3 or '
        'python for the verifier" >&2; exit 127; }\n'
        f'"$PY" -c \'import sys; sys.exit(sys.version_info < {PYTHON_FLOOR!r})\' 2>/dev/null || {{ echo "the verifier '
        f'needs Python {PYTHON_FLOOR[0]}.{PYTHON_FLOOR[1]} or newer, found $("$PY" --version 2>&1) at $PY" >&2; '
        "exit 127; }\n"
        f'exec "$PY" {TESTS_DIR}/{SHIM_DIR}/{RUNNER_FILE} {" ".join(arguments)}\n'
    )


def write_shim(target: Path, shim: VerifierShim) -> Path:
    """Write the shim folder at ``target`` (created; existing shim files replaced) and return it.

    ``target/test.sh``, ``target/.ng/run_verifier.py``, ``target/.ng/nemo_gym/verifiers/*`` (the installed
    package's modules, with an empty ``nemo_gym/__init__.py`` so the sandbox never imports the real package),
    ``target/.ng/task_data.json``, ``target/.ng/response.json`` when a response is given, and
    ``target/.ng/verifier.py`` when the module is not in the task's own ``tests/``.

    The generated tree is world-readable (``0755`` folders, ``0644`` files, ``test.sh`` ``0755``) whatever umask the
    process runs under. Gym's upload and Harbor's ``docker cp`` both keep the archive's modes and chown nothing, and
    ``test.sh`` runs as the task's ``[verifier].user``: under ``umask 077`` the runner and its helpers would land
    ``0600`` in a ``0700`` folder that a non-root verifier user could not open, and grading would fail before
    ``verify`` ran. Only the shim's own files are set, and ``target`` itself when this call created it (publish's
    ``tests/`` for a task graded by the dataset's module, which Harbor copies into ``/tests`` modes and all); a
    ``tests/`` the task shipped keeps the modes its author gave it.
    """
    target = Path(target)
    created = not target.exists()
    shim_dir = target / SHIM_DIR
    if shim_dir.exists():
        shutil.rmtree(shim_dir)
    (shim_dir / "nemo_gym" / "verifiers").mkdir(parents=True)
    (target / TEST_SCRIPT).write_text(test_script(shim))
    (target / TEST_SCRIPT).chmod(0o755)
    package_dir = Path(verifiers_package.__file__).resolve().parent
    for module in sorted(package_dir.glob("*.py")):
        shutil.copyfile(module, shim_dir / "nemo_gym" / "verifiers" / module.name)
    (shim_dir / "nemo_gym" / "__init__.py").write_text("")
    shutil.copyfile(package_dir / "runner.py", shim_dir / RUNNER_FILE)
    (shim_dir / TASK_DATA_FILE).write_text(json.dumps(dict(shim.task_data), sort_keys=True) + "\n")
    if shim.response is not None:
        (shim_dir / RESPONSE_FILE).write_text(json.dumps(dict(shim.response)) + "\n")
    if not shim.in_task_tests:
        shutil.copyfile(shim.verifier, shim_dir / VERIFIER_FILE)
    for path in (shim_dir, *shim_dir.rglob("*")):
        path.chmod(0o755 if path.is_dir() else 0o644)
    if created:
        target.chmod(0o755)
    return target
