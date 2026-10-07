# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognize Dockerfiles Gym can run without a build.

Two shapes are recognized:

- *Pull mode*: ``FROM <image>`` plus ``WORKDIR``, ``ENV``, ``USER`` or ``LABEL`` lines.
  Nothing changes the filesystem, so Gym runs the base image directly and applies the
  recorded settings at sandbox creation (:func:`base_image_only`).
- *Overlay mode*: the same, plus ``RUN`` lines. Gym pulls the base image and runs the
  ``RUN`` lines inside the sandbox at seed, each with the ``WORKDIR``, ``ENV`` and
  ``USER`` in effect at that point of the Dockerfile (:func:`overlay_image`). This is
  the shape of SWE-bench style tasks: a published environment image plus setup commands.

An empty ``ENTRYPOINT []`` or ``CMD []`` is accepted and ignored: the sandbox provider
runs its own entrypoint. Anything else (``COPY``, ``ADD``, multi-stage ``FROM``,
``ARG``-templated images, heredoc ``RUN``, ``RUN --mount``, a real ``ENTRYPOINT`` or
``CMD``, ``HEALTHCHECK``, ``EXPOSE``, ``VOLUME``, ``SHELL``, ``STOPSIGNAL``) needs a
real build, which is not part of this loader yet; :class:`DockerfileNeedsBuild` names
the first instruction that requires it.
"""

import json
import posixpath
import re
import shlex
from dataclasses import dataclass, field


_SETTING_INSTRUCTIONS = {"WORKDIR", "ENV", "USER", "LABEL"}
_ENV_PAIR = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=("(?:[^"\\]|\\.)*"|\'[^\']*\'|[^\s]*)')
# A Dockerfile heredoc (`RUN <<EOF`, `RUN python3 <<-'EOF'`), but not a bash here-string (`<<<`).
_HEREDOC = re.compile(r"(?<!<)<<(?!<)-?\s*['\"]?\w")


class DockerfileNeedsBuild(ValueError):
    """The Dockerfile holds an instruction this loader cannot apply without a build."""


@dataclass(frozen=True)
class BaseImage:
    """A Dockerfile that reduces to a base image plus creation-time settings."""

    image: str
    workdir: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    user: str | None = None


@dataclass(frozen=True)
class OverlayRun:
    """One ``RUN`` line with the Dockerfile state in effect when it runs.

    ``command`` is the shell-form command (exec form is converted). ``workdir``, ``env``
    and ``user`` are what the preceding ``WORKDIR``, ``ENV`` and ``USER`` lines set;
    ``None`` and ``{}`` mean the base image's own.
    """

    command: str
    workdir: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    user: str | None = None


@dataclass(frozen=True)
class OverlayImage(BaseImage):
    """A base image plus the ``RUN`` lines to apply at seed, in Dockerfile order.

    ``workdir``, ``env`` and ``user`` are the final values, as the built image would record them.
    """

    runs: tuple[OverlayRun, ...] = ()


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


def _json_array(arguments: str) -> list | None:
    """The JSON array an exec-form instruction holds, or ``None`` when the arguments are shell form."""
    if not arguments.startswith("["):
        return None
    try:
        parsed = json.loads(arguments)
    except ValueError:
        return None
    return parsed if isinstance(parsed, list) else None


def _run_command(arguments: str) -> str:
    """The shell command a ``RUN`` line executes."""
    if arguments.startswith("--"):
        flag = arguments.split(maxsplit=1)[0].split("=")[0]
        raise DockerfileNeedsBuild(f"RUN {flag} (BuildKit flags)")
    if _HEREDOC.search(arguments):
        raise DockerfileNeedsBuild("RUN with a heredoc (<<EOF)")
    exec_form = _json_array(arguments)
    if exec_form is not None:
        if not exec_form or not all(isinstance(part, str) for part in exec_form):
            raise DockerfileNeedsBuild(f"RUN {arguments}")
        return shlex.join(exec_form)
    if not arguments:
        raise DockerfileNeedsBuild("RUN with no command")
    return arguments


def parse_dockerfile(text: str) -> OverlayImage:
    """Read a pull-mode or overlay-mode Dockerfile; raise :class:`DockerfileNeedsBuild` for anything else."""
    image: str | None = None
    workdir: str | None = None
    user: str | None = None
    env: dict[str, str] = {}
    runs: list[OverlayRun] = []
    for line in _logical_lines(text):
        instruction, _, arguments = line.partition(" ")
        instruction = instruction.upper()
        arguments = arguments.strip()
        if instruction == "FROM":
            if image is not None:
                raise DockerfileNeedsBuild("a second FROM (multi-stage build)")
            tokens = [token for token in arguments.split() if not token.startswith("--")]
            if not tokens:
                raise DockerfileNeedsBuild("FROM without an image")
            if "$" in tokens[0]:
                raise DockerfileNeedsBuild(f"FROM {tokens[0]} (ARG-templated image)")
            if len(tokens) > 1 and tokens[1].upper() != "AS":
                raise DockerfileNeedsBuild(f"FROM {arguments}")
            image = tokens[0]
        elif image is None:
            raise DockerfileNeedsBuild(f"{instruction} before FROM")
        elif instruction == "RUN":
            runs.append(OverlayRun(command=_run_command(arguments), workdir=workdir, env=dict(env), user=user))
        elif instruction == "WORKDIR":
            # A relative WORKDIR is relative to the previous one, as in Docker.
            workdir = (
                arguments if workdir is None or posixpath.isabs(arguments) else posixpath.join(workdir, arguments)
            )
        elif instruction == "USER":
            user = arguments
        elif instruction == "ENV":
            env.update(_parse_env(arguments))
        elif instruction == "LABEL":
            pass
        elif instruction in ("ENTRYPOINT", "CMD") and _json_array(arguments) == []:
            pass  # an emptied entrypoint changes nothing: the sandbox provider runs its own
        else:
            raise DockerfileNeedsBuild(f"{instruction}")
    if image is None:
        raise DockerfileNeedsBuild("no FROM instruction")
    return OverlayImage(image=image, workdir=workdir, env=env, user=user, runs=tuple(runs))


def overlay_image(text: str) -> OverlayImage | None:
    """The base image, settings and ``RUN`` lines, or ``None`` when the file needs a build."""
    try:
        return parse_dockerfile(text)
    except DockerfileNeedsBuild:
        return None


def base_image_only(text: str) -> BaseImage | None:
    """Return the base image and settings, or ``None`` when the file has ``RUN`` lines or needs a build."""
    parsed = overlay_image(text)
    if parsed is None or parsed.runs:
        return None
    return BaseImage(image=parsed.image, workdir=parsed.workdir, env=parsed.env, user=parsed.user)
