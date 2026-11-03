# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fabric worker uploaded into a task sandbox; deliberately has no Gym dependency."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from shlex import quote

from nemo_fabric import Fabric, FabricConfig, RunRequest
from pydantic import BaseModel, ConfigDict, JsonValue


LOG = logging.getLogger(__name__)


class Invocation(BaseModel):
    """One agent activation, with paths already resolved for the task sandbox."""

    model_config = ConfigDict(extra="forbid")
    config: dict[str, JsonValue]
    input: JsonValue
    request_id: str
    workdir: str


def _first_step_timestamp(trajectory: dict[str, JsonValue]) -> str:
    """Relay names its files by session id, so order trajectories by their first step's timestamp."""
    steps = trajectory.get("steps")
    first = steps[0] if isinstance(steps, list) and steps else None
    return str(first.get("timestamp") if isinstance(first, dict) else None)


async def run(directory: Path) -> None:
    """Write Fabric's result atomically, including Relay files before close removes them."""
    invocation = Invocation.model_validate_json((directory / "input.json").read_text())
    # Fabric launches adapters with `ADAPTER_PYTHON -m ...`. Isolate that interpreter
    # too, while keeping the task's Python environment available to its shell tools.
    adapter_python = directory / "adapter_python"
    adapter_python.write_text(f'#!/bin/sh\nexec {quote(sys.executable)} -I -B "$@"\n')
    adapter_python.chmod(0o700)
    os.environ["ADAPTER_PYTHON"] = str(adapter_python)
    result = await Fabric().run(
        FabricConfig.from_mapping(invocation.config),
        base_dir=invocation.workdir,
        request=RunRequest(
            input=invocation.input,
            request_id=invocation.request_id,
            context={"nemo_gym_rollout_id": invocation.request_id},
        ),
    )
    mapping = result.to_mapping()
    trajectories = []
    for artifact in (mapping.get("output") or {}).get("relay_artifacts") or []:
        if not isinstance(artifact, dict) or artifact.get("kind") != "atif" or not artifact.get("path"):
            continue
        try:
            trajectory = json.loads(Path(artifact["path"]).read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            LOG.warning("Relay trajectory unavailable", exc_info=True)
            continue
        # A file of another shape must not stop the worker after Fabric finished, because no result would be written.
        if isinstance(trajectory, dict):
            trajectories.append(trajectory)
    if trajectories:
        mapping["relay_atif"] = sorted(trajectories, key=_first_step_timestamp)
    temporary = directory / "result.tmp"
    temporary.write_text(json.dumps(mapping, default=str))
    temporary.replace(directory / "result.json")


if __name__ == "__main__":
    asyncio.run(run(Path(sys.argv[1])))
