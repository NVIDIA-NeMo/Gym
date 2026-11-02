# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A component's default config, and the server classes that its entry point defines."""

import importlib
import inspect
import re
from pathlib import Path
from typing import Any

import yaml

from nemo_gym.agents import SimpleResponsesAPIAgent
from nemo_gym.server_runtime import SimpleResourcesServer

from .engine import AGENT_SERVER, RESOURCES_SERVER, Kind, Unavailable


INTERPOLATION = re.compile(r"\$\{[^}]*\}")


def default_agent_config(component: Path) -> dict[str, Any]:
    """The first agent instance in the component's default config, with a placeholder for every `${...}`."""
    path = component / "configs" / f"{component.name}.yaml"
    if not path.is_file():
        raise Unavailable(f"{path} does not exist, so there is no agent config to run it with")
    for instance, body in (yaml.safe_load(INTERPOLATION.sub("probe", path.read_text())) or {}).items():
        servers = body.get("responses_api_agents") if isinstance(body, dict) else None
        if servers:
            return {"name": instance, **next(iter(servers.values()))}
    raise Unavailable(f"{path} defines no agent instance")


def _base_class(kind: Kind) -> type:
    if kind == AGENT_SERVER:
        return SimpleResponsesAPIAgent
    if kind == RESOURCES_SERVER:
        return SimpleResourcesServer
    raise ValueError(f"Unknown component kind: {kind!r}")


def _entrypoint(component: Path) -> str:
    """The file that the component's default config names as its entry point, or `app.py`."""
    try:
        entrypoint = default_agent_config(component).get("entrypoint")
    except Unavailable:
        return "app.py"
    return entrypoint if isinstance(entrypoint, str) and (component / entrypoint).is_file() else "app.py"


def server_classes(component: Path, kind: Kind) -> tuple[str, list[type]]:
    """The component's entry point file, and the concrete server classes of `kind` that it defines."""
    base = _base_class(kind)
    entrypoint = _entrypoint(component)
    module = importlib.import_module(".".join((*component.parts, Path(entrypoint).stem)))
    classes = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, base)
        and value.__module__ == module.__name__
        and not inspect.isabstract(value)
    ]
    return entrypoint, classes
