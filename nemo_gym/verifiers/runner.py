# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The in-sandbox runner of a Python verifier: ``python3 run_verifier.py --verifier ... --workspace ...``.

Copied into the verifier sandbox by the ``test.sh`` shim (``nemo_gym.tasks.harbor.python_verifier``) and run
there as a script, so it is standard library only and imports nothing from ``nemo_gym``. It puts its own
``.ng`` folder first on ``sys.path``, where the shim vendored ``nemo_gym/verifiers``, then the verifier module's
own folder, then the tests folder around the shim (a helper module beside the task's ``tests/verifier.py`` stays
importable when the module was published under ``.ng/``), so the verifier's ``from nemo_gym.verifiers import ...``
resolves without an install and always to the vendored copy: a ``nemo_gym`` package in the task's own ``tests/``,
in the image's site-packages or on its ``PYTHONPATH`` never replaces the helpers. That order holds however the
interpreter was started (as ``test.sh`` runs it, ``python3 /tests/.ng/run_verifier.py``, ``sys.path[0]`` is already
this folder; under ``-I`` it is absent): each of the three entries is moved to its place, not merely added when
missing. It loads the verifier module, calls ``verify`` with the arguments the function
declares (``workspace``, ``task_data``, ``response``, ``state``; positional-only parameters are passed by
position), and writes Harbor's reward files: ``reward.txt`` for a number, ``reward.json`` for a dict. An
``async def verify`` is awaited (``asyncio.run``). ``response`` exists only when the shim was given one
(``--response``): the environment server always passes the agent's final response, a published Harbor task
never has one, so a ``verify`` that requires ``response`` without a default is a contract error there rather than
a verifier quietly called with ``None``.

Exit codes: 0 when a reward was written, 2 for a shim or verifier-contract problem (no ``verify``, an
argument it asks for that is not available, a return value that is not a number or a dict of numbers), 1
when ``verify`` raised. No reward file is written on 1 or 2, which the environment server and Harbor score as
a missing reward, with this script's stderr in the verifier log; and none is left either: before ``verify``
runs, any ``reward.json`` or ``reward.txt`` already in the reward folder (a file, a link or a folder of that name)
is removed, as the ``test.sh`` shim removes them before this script starts, so a file the agent planted there
never stands as the verdict under Harbor, which shares the folder with the agent's container and reads
``reward.json`` first.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import json
import math
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any


KNOWN_ARGUMENTS = ("workspace", "task_data", "response", "state")
REWARD_FILES = ("reward.json", "reward.txt")


def clear_rewards(reward_dir: Path) -> None:
    """Remove every reward file already in ``reward_dir`` (a file, a link or a folder of that name), if the folder
    exists: the verdict is what ``verify`` writes in this run, never something found there."""
    for name in REWARD_FILES:
        path = reward_dir / name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def place_on_path(*folders: Path) -> None:
    """Put ``folders`` at the front of ``sys.path`` in the order given, highest priority first, whatever the
    interpreter put there already: an entry already on the path (``sys.path[0]`` is this script's own folder when it
    runs as ``test.sh`` runs it, without ``-I``; ``PYTHONPATH`` may add any other) is moved, so no folder lower in the
    order is ever searched before one higher in it."""
    wanted: list[str] = []
    for folder in folders:
        entry = str(folder)
        if entry not in wanted:
            wanted.append(entry)
    resolved = {str(Path(entry).resolve()) for entry in wanted}
    rest = [entry for entry in sys.path if entry not in wanted and str(Path(entry or ".").resolve()) not in resolved]
    sys.path[:] = wanted + rest


def load_verify(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("ng_task_verifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ng_task_verifier"] = module
    spec.loader.exec_module(module)
    verify = getattr(module, "verify", None)
    if verify is None or not callable(verify):
        raise RuntimeError(f"{path} does not define a callable `verify`")
    return verify


def select_arguments(verify: Any, available: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    """The arguments to call ``verify`` with: the known names it declares, each from ``available``.

    Returns ``(positional, keywords)``: a positional-only parameter (``def verify(workspace, /)``) is passed by
    position, every other one by name. A parameter the runner does not know, or that is not in ``available`` in this
    run (``response`` without ``--response``), is left to its default; one without a default is a contract error.
    ``*args``/``**kwargs`` receive nothing (a ``**kwargs`` verifier gets every available argument).
    """
    try:
        signature = inspect.signature(verify)
    except (TypeError, ValueError):
        return [], {}
    positional: list[Any] = []
    keywords: dict[str, Any] = {}
    accepts_var_kw = False
    for name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_var_kw = True
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if name in available:
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                positional.append(available[name])
            else:
                keywords[name] = available[name]
        elif parameter.default is inspect.Parameter.empty:
            if name == "response":
                raise RuntimeError(
                    "verify() asks for `response`, which this run has no response for (a published Harbor task's "
                    "test.sh sees only the container); give it a default or grade the workspace"
                )
            raise RuntimeError(
                f"verify() asks for `{name}`, which the runner cannot supply; it knows {', '.join(KNOWN_ARGUMENTS)}"
            )
        elif parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional.append(parameter.default)
    if accepts_var_kw:
        for name, value in available.items():
            keywords.setdefault(name, value)
    return positional, keywords


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def write_reward(result: Any, reward_dir: Path) -> None:
    """Harbor's reward files from ``verify``'s return value; raises ``RuntimeError`` for anything else."""
    reward_dir.mkdir(parents=True, exist_ok=True)
    number = _number(result)
    if number is not None:
        (reward_dir / "reward.txt").write_text(f"{number}\n")
        return
    if isinstance(result, dict):
        rewards = {}
        for key, value in result.items():
            value = _number(value)
            if not isinstance(key, str) or value is None:
                raise RuntimeError(f"verify() returned a dict with a non-numeric entry {key!r}: {result!r}")
            rewards[key] = value
        if "reward" not in rewards and len(rewards) != 1:
            raise RuntimeError(f"verify() returned several metrics and none named 'reward': {sorted(rewards)}")
        (reward_dir / "reward.json").write_text(json.dumps(rewards) + "\n")
        return
    raise RuntimeError(f"verify() must return a number or a dict of numbers, got {type(result).__name__}: {result!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Harbor task's Python verifier and write its reward.")
    parser.add_argument("--verifier", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--task-data", type=Path, default=None)
    parser.add_argument("--response", type=Path, default=None)
    parser.add_argument("--reward-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    here = Path(__file__).resolve().parent
    place_on_path(here, args.verifier.resolve().parent, here.parent)
    try:
        clear_rewards(args.reward_dir)
        task_data = json.loads(args.task_data.read_text()) if args.task_data is not None else {}
        verify = load_verify(args.verifier)
        workspace = args.workspace
        available = {"workspace": workspace, "task_data": task_data, "state": {"workspace": workspace}}
        if args.response is not None:
            available["response"] = json.loads(args.response.read_text())
        positional, keywords = select_arguments(verify, available)
    except Exception as exc:
        print(f"verifier shim: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2
    try:
        result = verify(*positional, **keywords)
        if inspect.iscoroutine(result):
            result = asyncio.run(result)
    except Exception:
        print("verify() raised:", file=sys.stderr)
        traceback.print_exc()
        return 1
    try:
        write_reward(result, args.reward_dir)
    except RuntimeError as exc:
        print(f"verifier shim: {exc}", file=sys.stderr)
        return 2
    print(f"verify() returned {result!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
