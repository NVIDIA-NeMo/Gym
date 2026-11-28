# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluator modules frozen with an RLVR task snapshot.

An RLVR snapshot (VeriGen/Astra task bundles) keeps each task's own getters and metrics
as files under ``<root>/<task_id>/{getters,metrics}/*.py``. The modules may import
task-local helpers relatively or by bare name, so they are loaded as files with their
directory on the import path. Every function they define is set on
``desktop_env.evaluators.getters`` / ``.metrics``, replacing a built-in of the same
name as the snapshot's own evaluator did. ``clear_frozen_evaluators`` undoes all of it
before a Ray worker runs another task.

Snapshot tasks may also set ``metadata.binary_final_reward``: only a full score counts
(``apply_task_reward_contract``).
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import numbers
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Mapping, Tuple


_MISSING = object()
_ATTRIBUTES: List[Tuple[ModuleType, str, object]] = []
_PACKAGE_PATHS: List[Tuple[ModuleType, List[str]]] = []
_SYS_PATH_ENTRIES: List[str] = []
_MODULE_NAMES: List[str] = []


def install_run_command_compat() -> None:
    """Give ``PythonController`` the ``run_command`` API frozen evaluators call.

    The pinned OSWorld controller only has specialized helpers such as
    ``execute_python_command``; snapshot evaluators use the older generic call. Both go
    through the guest's ``/execute``.
    """

    import requests
    from desktop_env.controllers.python import PythonController

    if callable(getattr(PythonController, "run_command", None)):
        return

    def run_command(self: Any, command: str | list[str], shell: bool = False, timeout: int = 120) -> Dict[str, Any]:
        payload = {"command": command, "shell": shell}
        last_error = "unknown error"
        retry_times = max(1, int(getattr(self, "retry_times", 3)))
        retry_interval = float(getattr(self, "retry_interval", 5))
        for attempt in range(retry_times):
            try:
                response = requests.post(self.http_server + "/execute", json=payload, timeout=timeout)
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, dict):
                        return result
                    last_error = f"guest returned non-object JSON: {result!r}"
                else:
                    last_error = f"guest execute returned HTTP {response.status_code}: {response.text[:500]}"
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            except (TypeError, ValueError) as exc:
                last_error = f"invalid guest execute response: {exc}"
            if attempt + 1 < retry_times:
                time.sleep(retry_interval)
        return {"status": "error", "output": "", "error": last_error, "returncode": -1}

    PythonController.run_command = run_command


def _load_module(path: Path, package: ModuleType, task_id: str) -> List[str]:
    module_name = f"{package.__name__}._rlvr_{hashlib.sha256(str(path).encode()).hexdigest()[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{task_id}: cannot load evaluator module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    _MODULE_NAMES.append(module_name)
    spec.loader.exec_module(module)
    exported = []
    for name, value in vars(module).items():
        if callable(value) and getattr(value, "__module__", None) == module_name:
            _ATTRIBUTES.append((package, name, getattr(package, name, _MISSING)))
            setattr(package, name, value)
            exported.append(name)
    if not exported:
        raise ImportError(f"{task_id}: {path} exports no evaluator functions")
    return exported


def register_frozen_evaluators(task_config: Mapping[str, Any], root: str) -> List[str]:
    """Install one task's frozen evaluators; the caller clears them in its task-level finally."""

    task_id = task_config.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("RLVR task has no non-empty id")
    functions_root = Path(root).resolve()
    task_root = (functions_root / task_id).resolve()
    if not task_root.is_relative_to(functions_root):
        raise ValueError(f"frozen evaluator path escapes its root: {task_root}")
    if not task_root.is_dir():
        return []

    from desktop_env.evaluators import getters, metrics

    install_run_command_compat()
    packages = {"getters": getters, "metrics": metrics}
    modules_before = set(sys.modules)
    loaded: List[str] = []
    try:
        for role, package in packages.items():
            role_root = task_root / role
            _PACKAGE_PATHS.append((package, list(getattr(package, "__path__", []))))
            package.__path__ = [str(role_root), *getattr(package, "__path__", [])]
            sys.path.insert(0, str(role_root))
            _SYS_PATH_ENTRIES.append(str(role_root))
            # Task bundles may carry validation scripts next to the evaluators.
            paths = sorted(
                path
                for path in role_root.glob("*.py")
                if path.name != "__init__.py" and not path.name.startswith("selftest_")
            )
            for path in paths:
                loaded.extend(f"{role}:{name}" for name in _load_module(path, package, task_id))
    finally:
        for name in set(sys.modules) - modules_before:
            module_file = getattr(sys.modules[name], "__file__", None)
            if module_file and Path(module_file).resolve().is_relative_to(task_root):
                _MODULE_NAMES.append(name)
    if not loaded:
        raise FileNotFoundError(f"{task_id}: frozen evaluator modules not found")

    metadata = task_config.get("metadata")
    routes = metadata.get("evaluator_routes") if isinstance(metadata, Mapping) else None
    for record in routes if isinstance(routes, list) else []:
        route = record.get("route") if isinstance(record, Mapping) else None
        if not isinstance(route, str) or ":" not in route:
            raise ValueError(f"{task_id}: invalid frozen evaluator route record")
        role, symbol = route.split(":", 1)
        package = packages.get(role)
        if package is None or not callable(getattr(package, symbol, None)):
            raise ImportError(f"{task_id}: selected evaluator route was not loaded: {route}")
    return loaded


def clear_frozen_evaluators() -> None:
    """Restore the evaluator packages, import path and module cache; safe to call repeatedly."""

    for package, name, previous in reversed(_ATTRIBUTES):
        if previous is _MISSING:
            if hasattr(package, name):
                delattr(package, name)
        else:
            setattr(package, name, previous)
    for package, path in reversed(_PACKAGE_PATHS):
        package.__path__ = path
    for entry in _SYS_PATH_ENTRIES:
        if entry in sys.path:
            sys.path.remove(entry)
    for name in _MODULE_NAMES:
        sys.modules.pop(name, None)
    _ATTRIBUTES.clear()
    _PACKAGE_PATHS.clear()
    _SYS_PATH_ENTRIES.clear()
    _MODULE_NAMES.clear()


def apply_task_reward_contract(task_config: Mapping[str, Any], score: Any) -> float:
    """Map a score to 1.0 or 0.0 for tasks whose metadata sets ``binary_final_reward``."""

    metadata = task_config.get("metadata")
    if not (isinstance(metadata, Mapping) and metadata.get("binary_final_reward") is True):
        return float(score)
    if not isinstance(score, numbers.Real):
        raise TypeError("binary evaluator must return a real scalar score")
    value = float(score)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("binary evaluator score must be finite and in [0, 1]")
    return float(value == 1.0)
