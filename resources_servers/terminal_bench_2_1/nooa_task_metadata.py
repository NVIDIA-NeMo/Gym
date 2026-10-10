# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read explicit startup commands from the canonical task's final Dockerfile stage."""

import hashlib
import json
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field


class CanonicalImageStartup(BaseModel):
    """A source-bound startup command, never an agent-supplied launch override."""

    docker_image: str
    dockerfile_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    command: list[str] = Field(min_length=1)


def read_image_startup(task_folder: Path) -> CanonicalImageStartup | None:
    """Preserve explicit JSON CMD/ENTRYPOINT; reject unsupported shell forms."""
    dockerfile = task_folder / "environment/Dockerfile"
    if not dockerfile.is_file():
        return None
    source = dockerfile.read_bytes()
    directives: dict[str, list[str]] = {}
    for line in source.decode().splitlines():
        instruction, _, argument = line.strip().partition(" ")
        instruction = instruction.upper()
        if instruction == "FROM":
            directives.clear()
        elif instruction in {"CMD", "ENTRYPOINT"}:
            try:
                command = json.loads(argument)
            except json.JSONDecodeError as error:
                raise ValueError(f"Canonical {instruction} must use a single-line JSON array: {dockerfile}") from error
            if not isinstance(command, list) or not command or not all(isinstance(token, str) for token in command):
                raise ValueError(f"Canonical {instruction} must be a nonempty string array: {dockerfile}")
            directives[instruction] = command
    if not directives:
        return None
    task = tomllib.loads((task_folder / "task.toml").read_text())
    return CanonicalImageStartup(
        docker_image=task["environment"]["docker_image"],
        dockerfile_sha256=hashlib.sha256(source).hexdigest(),
        command=directives.get("ENTRYPOINT", []) + directives.get("CMD", []),
    )
