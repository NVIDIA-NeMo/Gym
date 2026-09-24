# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognize Dockerfiles that only name a base image.

Many tasks ship a Dockerfile that is ``FROM <image>`` plus ``WORKDIR``, ``ENV``,
``USER`` or ``LABEL`` lines. Nothing in such a file changes the filesystem, so Gym
can run the base image directly and apply the recorded settings at sandbox creation
instead of building. Anything else (``RUN``, ``COPY``, ``ADD``, multi-stage builds,
``ARG``-templated images) needs a real build, which is not part of this loader yet.
"""

import re
import shlex
from dataclasses import dataclass, field


_ALLOWED_INSTRUCTIONS = {"FROM", "WORKDIR", "ENV", "USER", "LABEL"}
_ENV_PAIR = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=("(?:[^"\\]|\\.)*"|\'[^\']*\'|[^\s]*)')


@dataclass(frozen=True)
class BaseImage:
    """A Dockerfile that reduces to a base image plus creation-time settings."""

    image: str
    workdir: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    user: str | None = None


def _logical_lines(text: str) -> list[str]:
    """Join continuation lines and drop comments and blanks."""
    lines: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        # Docker drops comment and blank lines even inside a continuation.
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
            continue
        lines.append((buffer + stripped).strip())
        buffer = ""
    if buffer.strip():
        lines.append(buffer.strip())
    return lines


def _parse_env(arguments: str) -> dict[str, str]:
    if "=" not in arguments.split(maxsplit=1)[0]:
        # Legacy form: `ENV KEY value with spaces`
        key, _, value = arguments.partition(" ")
        return {key: value.strip()}
    values: dict[str, str] = {}
    for key, value in _ENV_PAIR.findall(arguments):
        if value[:1] in {'"', "'"}:
            value = shlex.split(value)[0] if value else ""
        values[key] = value
    return values


def base_image_only(text: str) -> BaseImage | None:
    """Return the base image and settings, or ``None`` when the file needs a build."""
    image: str | None = None
    workdir: str | None = None
    user: str | None = None
    env: dict[str, str] = {}
    for line in _logical_lines(text):
        instruction, _, arguments = line.partition(" ")
        instruction = instruction.upper()
        arguments = arguments.strip()
        if instruction not in _ALLOWED_INSTRUCTIONS:
            return None
        if instruction == "FROM":
            if image is not None:
                return None  # multi-stage build
            tokens = [token for token in arguments.split() if not token.startswith("--")]
            if not tokens or "$" in tokens[0] or (len(tokens) > 1 and tokens[1].upper() != "AS"):
                return None
            image = tokens[0]
        elif image is None:
            return None  # instructions before FROM
        elif instruction == "WORKDIR":
            workdir = arguments
        elif instruction == "USER":
            user = arguments
        elif instruction == "ENV":
            env.update(_parse_env(arguments))
    if image is None:
        return None
    return BaseImage(image=image, workdir=workdir, env=env, user=user)
