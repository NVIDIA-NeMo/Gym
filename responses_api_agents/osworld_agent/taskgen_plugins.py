# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluator plugins that an OSWorld task carries in its own config.

Generated tasks (cua-research/taskgen, e.g. from VeriGen) often judge with getters and metrics that OSWorld does
not ship. taskgen lowers them into the task config as one self-contained block::

    "taskgen_plugins": {
        "schema": "taskgen-plugins/1",
        "getters": ["<type>", ...],          # getter types, looked up as get_<type>
        "metrics": ["<function>", ...],      # metric function names
        "modules": {"getters": "<python source>", "metrics": "<python source>"},
        "sha256": "<plugin_digest(modules)>"
    }

DesktopEnv resolves ``evaluator.func`` and getter types by name on ``desktop_env.evaluators.metrics`` and
``desktop_env.evaluators.getters`` when ``reset()`` reads the task, so the declared functions are set on those
modules first. Built-in OSWorld functions are never replaced. DesktopEnv ignores the extra config key.
"""

from __future__ import annotations

import hashlib
import json
import types
from typing import Any, Dict, List, Mapping


SCHEMA = "taskgen-plugins/1"
_REGISTERED: Dict[str, str] = {}  # function name -> digest of the plugin that defined it (this process)


class TaskPluginError(ValueError):
    """The task's evaluator plugin block is malformed, tampered with, or cannot be used here."""


def plugin_digest(modules: Mapping[str, str]) -> str:
    canonical = json.dumps(dict(modules), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def register_task_plugins(task_config: Mapping[str, Any], *, remote: bool = False) -> List[str]:
    """Register the task's plugin functions; returns their names ([] when the task has none)."""

    block = task_config.get("taskgen_plugins")
    if not block:
        return []
    if remote:
        raise TaskPluginError(
            "task carries evaluator plugins, which the remote Resources Server path cannot load yet; "
            "run it with the in-process environment"
        )
    if not isinstance(block, Mapping) or block.get("schema") != SCHEMA:
        raise TaskPluginError(f"taskgen_plugins must be a {SCHEMA} object")
    modules = block.get("modules") or {}
    if not isinstance(modules, Mapping) or set(modules) - {"getters", "metrics"}:
        raise TaskPluginError("taskgen_plugins.modules may only hold 'getters' and 'metrics' sources")
    digest = plugin_digest(modules)
    if digest != block.get("sha256"):
        raise TaskPluginError("taskgen_plugins.sha256 does not match its modules")

    from desktop_env.evaluators import getters as getter_module
    from desktop_env.evaluators import metrics as metric_module

    task_id = str(task_config.get("id") or "task")
    registered: List[str] = []
    for kind, target, prefix in (("getters", getter_module, "get_"), ("metrics", metric_module, "")):
        names = [prefix + str(name) for name in block.get(kind) or []]
        if not names:
            continue
        source = modules.get(kind)
        if not source:
            raise TaskPluginError(f"taskgen_plugins declares {kind} {names} but has no {kind} source")
        namespace = types.ModuleType(f"taskgen_plugin_{kind}_{digest[:12]}")
        exec(compile(source, f"<taskgen plugin {kind} of {task_id}>", "exec"), namespace.__dict__)  # noqa: S102
        for name in names:
            function = getattr(namespace, name, None)
            if not callable(function):
                raise TaskPluginError(f"taskgen plugin {kind} source does not define {name}")
            previous = _REGISTERED.get(name)
            if previous is None and hasattr(target, name):
                raise TaskPluginError(f"taskgen plugin would replace OSWorld's own {name}")
            if previous is not None and previous != digest:
                raise TaskPluginError(f"taskgen plugin {name} is already registered from a different plugin")
            setattr(target, name, function)
            _REGISTERED[name] = digest
            registered.append(name)
    return registered
