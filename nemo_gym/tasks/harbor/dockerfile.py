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
_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class BaseImage:
    """A Dockerfile that reduces to a base image plus creation-time settings."""

    image: str
    workdir: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    user: str | None = None


def _logical_lines(text: str) -> list[str]:
    """Join continuations without inventing whitespace; drop comments and blanks."""
    lines: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        piece = raw if buffer else raw.lstrip()
        if piece.rstrip().endswith("\\"):
            buffer += piece.rstrip()[:-1]
            continue
        lines.append((buffer + piece).strip())
        buffer = ""
    if buffer.strip():
        lines.append(buffer.strip())
    return lines


def _parse_env(arguments: str) -> dict[str, str] | None:
    if not arguments or any(char in arguments for char in ("$", "\\")) or "<<" in arguments:
        return None
    first, *rest = arguments.split(maxsplit=1)
    if "=" not in first:
        # Keep plain legacy values verbatim. Quoting/escaping needs a real build.
        if not rest or not _ENV_KEY.fullmatch(first) or any(char in rest[0] for char in "\"'"):
            return None
        return {first: rest[0]}
    try:
        pairs = shlex.split(arguments)
    except ValueError:
        return None
    values: dict[str, str] = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator or not _ENV_KEY.fullmatch(key):
            return None
        values[key] = value
    return values


def base_image_only(text: str) -> BaseImage | None:
    """Return faithfully interpreted settings, or ``None`` when a build is needed.

    Variable expansion and relative working directories depend on the base image's
    configuration. Leave those to a real build until that configuration is available.
    """
    if re.search(r"(?im)^\s*#\s*escape\s*=", text):
        return None
    image: str | None = None
    workdir: str | None = None
    user: str | None = None
    env: dict[str, str] = {}
    for line in _logical_lines(text):
        instruction, *parts = line.split(maxsplit=1)
        instruction = instruction.upper()
        arguments = parts[0] if parts else ""
        if instruction not in _ALLOWED_INSTRUCTIONS or not arguments:
            return None
        if instruction == "FROM":
            if image is not None:
                return None
            tokens = arguments.split()
            if len(tokens) != 1 and not (len(tokens) == 3 and tokens[1].upper() == "AS"):
                return None
            if tokens[0].startswith("--") or any(char in tokens[0] for char in "$\\\"'"):
                return None
            image = tokens[0]
            if image == "scratch":
                return None
        elif image is None:
            return None
        elif instruction in ("WORKDIR", "USER"):
            if any(char in arguments for char in "$\\\"'"):
                return None
            if instruction == "WORKDIR":
                if not arguments.startswith("/"):
                    return None
                workdir = arguments
            else:
                if len(arguments.split()) != 1:
                    return None
                user = arguments
        elif instruction == "ENV":
            values = _parse_env(arguments)
            if values is None:
                return None
            env.update(values)
    if image is None:
        return None
    return BaseImage(image=image, workdir=workdir, env=env, user=user)
