# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognize Dockerfiles Gym can run without a build, and resolve them as Docker would.

Two shapes are recognized:

- *Pull mode*: ``FROM <image>`` plus ``WORKDIR``, ``ENV``, ``USER`` or ``LABEL`` lines.
  Nothing changes the filesystem, so Gym runs the base image directly and applies the
  recorded settings at sandbox creation (:func:`base_image_only`).
- *Overlay mode*: the same, plus ``RUN`` lines. Gym pulls the base image and runs the
  ``RUN`` lines inside the sandbox at seed, each with the ``WORKDIR``, ``ENV`` and
  ``USER`` in effect at that point of the Dockerfile (:func:`overlay_image`). This is
  the shape of SWE-bench style tasks: a published environment image plus setup commands.

``ENV``, ``WORKDIR`` and ``USER`` are resolved the way ``docker build`` resolves them: the
base image's own configuration (:class:`BaseImageConfig`, recorded from the registry at
prepare) is the starting state, ``$NAME`` and ``${NAME}`` expand against the accumulated
environment (not inside single quotes; ``\\$`` is a literal dollar), a relative ``WORKDIR``
joins onto the previous one. The resolved values are literal: nothing reaches the sandbox
with a live ``$`` in it. A Dockerfile whose resolution depends on the base image (a ``RUN``
line, a reference to a variable the Dockerfile did not set, a relative first ``WORKDIR``)
raises :class:`ImageConfigRequired` when no configuration is given.

An empty ``CMD []`` is accepted and ignored. An empty ``ENTRYPOINT []`` is accepted and
recorded (``entrypoint_cleared``): the sandbox provider starts the image with its own keepalive
either way, and the Harbor environment server runs the base image's recorded ``Entrypoint``
beside it unless the Dockerfile cleared it. Anything else (``COPY``, ``ADD``, multi-stage ``FROM``,
``ARG``-templated images, heredoc ``RUN``, ``RUN --mount``, a real ``ENTRYPOINT`` or
``CMD``, ``HEALTHCHECK``, ``EXPOSE``, ``VOLUME``, ``SHELL``, ``STOPSIGNAL``, a ``FROM`` flag
such as ``--platform``, an ``# escape=`` parser directive) needs a real build, which is not
part of this loader yet; :class:`DockerfileNeedsBuild` names the first instruction that requires it.
"""

import json
import posixpath
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


# Docker's PATH for a base image whose configuration records none (``system.DefaultPathEnvUnix``).
DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# Docker's shell for a shell-form ``RUN`` when the image records no ``Shell``.
DEFAULT_SHELL = ("/bin/sh", "-c")
# The shape of a ``compose-images.json`` entry. Bumped whenever the recorded fields grow (``Env`` in M16,
# ``Shell`` after it), so a record an older Gym wrote is re-read at prepare and refused at load instead of
# standing in for a known, empty value.
RECORD_VERSION = 2
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Docker's blank run (its `reWhitespace`), which separates the key from the value of a legacy `ENV KEY value`.
_BLANKS = re.compile(r"[ \t\v\f\r]+")
# A continuation: a backslash, optionally followed by blanks, at the end of the line.
_CONTINUATION = re.compile(r"\\[ \t]*$")
# A shell word that opens a Dockerfile heredoc (``<<EOF``, ``<<-EOF``, ``<<'EOF'``, ``<<"EOF"``), as
# BuildKit's parser recognizes them: the whole word, so ``<<<`` here-strings, ``$((1<<2))`` and a
# quoted ``"<<EOF"`` are not heredocs.
_HEREDOC_WORD = re.compile(r"^<<-?(?:\w+|\"[^\"]+\"|'[^']+')$")


class DockerfileNeedsBuild(ValueError):
    """The Dockerfile holds an instruction this loader cannot apply without a build."""


class ImageConfigRequired(ValueError):
    """Resolving the Dockerfile needs the base image's recorded configuration, which was not given."""


@dataclass(frozen=True)
class BaseImageConfig:
    """What a base image's OCI configuration sets before its Dockerfile's first instruction.

    ``env`` is the image's ``Env`` as a mapping; ``workdir`` and ``user`` are ``WorkingDir``
    and ``User`` (``None`` when the image sets none, meaning ``/`` and root); ``shell`` is the
    image's ``Shell``, which runs its shell-form ``RUN`` lines (Docker's ``/bin/sh -c`` when unset).
    """

    env: dict[str, str] = field(default_factory=dict)
    workdir: str | None = None
    user: str | None = None
    # ``Entrypoint`` as recorded (``None`` when absent or empty) and the raw ``Healthcheck`` block, what
    # Docker starts the image with; the Harbor environment server reproduces both beside its keepalive.
    entrypoint: tuple[str, ...] | None = None
    healthcheck: dict[str, Any] | None = None
    shell: tuple[str, ...] = DEFAULT_SHELL

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "BaseImageConfig":
        """Read a ``compose-images.json`` entry (see ``nemo_gym.data.tasks.harbor.image_configs``).

        Raises :class:`ValueError` for an entry written before :data:`RECORD_VERSION`: it lacks fields
        this resolver reads (an absent ``Env`` would read as a known, empty environment), so the
        dataset must be prepared again to re-record it.
        """
        if record.get("record_version") != RECORD_VERSION:
            raise ValueError(
                "the recorded image configuration predates the fields the Dockerfile resolver reads; "
                "prepare the dataset again (`gym eval run` or `gym dataset validate`) to re-record it"
            )
        config = record.get("config") or {}
        env: dict[str, str] = {}
        for item in config.get("Env") or []:
            key, _, value = str(item).partition("=")
            env[key] = value
        entrypoint = config.get("Entrypoint")
        healthcheck = config.get("Healthcheck")
        return cls(
            env=env,
            workdir=config.get("WorkingDir") or None,
            user=config.get("User") or None,
            entrypoint=tuple(str(part) for part in entrypoint)
            if isinstance(entrypoint, list) and entrypoint
            else None,
            healthcheck=dict(healthcheck) if isinstance(healthcheck, Mapping) and healthcheck else None,
            shell=tuple(config.get("Shell") or DEFAULT_SHELL),
        )


@dataclass(frozen=True)
class BaseImage:
    """A Dockerfile that reduces to a base image plus creation-time settings.

    ``env`` holds the keys the Dockerfile's ``ENV`` lines set, with their resolved final
    values; ``workdir`` and ``user`` are the resolved final values (``None`` when neither
    the Dockerfile nor a given base image configuration sets them). ``entrypoint_cleared``
    is whether the Dockerfile wrote ``ENTRYPOINT []``, so the base image's entrypoint must not run.
    """

    image: str
    workdir: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    user: str | None = None
    entrypoint_cleared: bool = False


@dataclass(frozen=True)
class OverlayRun:
    """One ``RUN`` line with the Dockerfile state in effect when it runs, resolved to literals.

    ``command`` is the command text: a shell-form line as written, or an exec-form argument
    vector joined with shell quoting. ``shell`` runs it: the image's ``Shell`` in effect for a
    shell-form line (Docker's ``/bin/sh -c`` unless the image sets one), ``None`` for an exec-form
    line, which runs its arguments directly, without a shell. ``env`` is the whole environment the
    line runs with (the base image's ``Env`` plus the ``ENV`` lines so far), ``workdir`` the
    directory it runs in and ``user`` who runs it (``None`` is root).
    """

    command: str
    workdir: str = "/"
    env: dict[str, str] = field(default_factory=dict)
    user: str | None = None
    shell: tuple[str, ...] | None = DEFAULT_SHELL


@dataclass(frozen=True)
class OverlayWorkdir:
    """One ``WORKDIR`` instruction: the directory to create, owned by the user in effect at it.

    ``before`` is how many ``RUN`` lines precede the instruction, which places it among
    :attr:`OverlayImage.runs`: the directory is made before run ``before`` (counting from 0),
    or after the last run when no ``RUN`` follows it. Docker creates every ``WORKDIR`` whether or
    not a ``RUN`` uses it, so the finished image has them all.
    """

    path: str
    user: str | None = None
    before: int = 0


@dataclass(frozen=True)
class OverlayImage(BaseImage):
    """A base image plus the ``RUN`` lines to apply at seed, in Dockerfile order.

    ``workdirs`` are the ``WORKDIR`` instructions, in order, each with the user in effect at it
    and its place among the runs (see :class:`OverlayWorkdir`).
    """

    runs: tuple[OverlayRun, ...] = ()
    workdirs: tuple[OverlayWorkdir, ...] = ()


@dataclass(frozen=True)
class Dockerfile:
    """The parsed instructions of a pull-mode or overlay-mode Dockerfile, before resolution."""

    image: str
    # ``(INSTRUCTION, arguments)`` for every line after ``FROM``, in order.
    instructions: tuple[tuple[str, str], ...] = ()
    # The Dockerfile wrote ``ENTRYPOINT []``: the image starts without the base image's entrypoint.
    entrypoint_cleared: bool = False

    @property
    def has_runs(self) -> bool:
        return any(instruction == "RUN" for instruction, _ in self.instructions)


def _logical_lines(text: str) -> list[str]:
    """Join continuation lines and drop comments and blanks, as Docker's parser does.

    A trailing backslash is removed together with the newline and the next line is appended
    verbatim (no separator is inserted). Comment and blank lines are dropped even inside a
    continuation. Only the first line of an instruction has its leading blanks removed.
    """
    lines: list[str] = []
    buffer: str | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        piece = raw.lstrip() if buffer is None else raw
        continued = _CONTINUATION.search(piece) is not None
        if continued:
            piece = _CONTINUATION.sub("", piece)
        buffer = piece if buffer is None else buffer + piece
        if not continued:
            lines.append(buffer.strip())
            buffer = None
    if buffer is not None and buffer.strip():
        lines.append(buffer.strip())
    return lines


def split_words(text: str) -> list[str]:
    """Split on unquoted blanks, keeping quotes and escapes in the words (Docker's ``parseWords``)."""
    words: list[str] = []
    current = ""
    quote: str | None = None
    index = 0
    while index < len(text):
        char = text[index]
        if quote is None and char in " \t":
            if current:
                words.append(current)
                current = ""
        elif char == "\\" and quote != "'" and index + 1 < len(text):
            current += text[index : index + 2]
            index += 1
        elif quote is None and char in "'\"":
            quote = char
            current += char
        elif quote == char:
            quote = None
            current += char
        else:
            current += char
        index += 1
    if current:
        words.append(current)
    return words


class _Expander:
    """Docker's shell-word processing: quote removal, escapes and ``$`` expansion against ``env``.

    With ``known=False`` the base image's environment is unknown, so a reference to a
    variable the Dockerfile did not set itself raises :class:`ImageConfigRequired` instead of
    silently expanding to nothing.
    """

    def __init__(self, env: Mapping[str, str], *, known: bool) -> None:
        self.env = env
        self.known = known

    def expand(self, word: str) -> str:
        out = ""
        quote: str | None = None
        index = 0
        while index < len(word):
            char = word[index]
            if quote == "'":
                if char == "'":
                    quote = None
                else:
                    out += char
                index += 1
            elif char == "$":
                value, index = self._dollar(word, index)
                out += value
            elif char == "\\" and quote is None:
                # Unquoted: the next character stands for itself (a trailing backslash is dropped).
                if index + 1 < len(word):
                    out += word[index + 1]
                index += 2
            elif char == "\\" and quote == '"':
                # Double-quoted: only `"`, `$` and `\` can be escaped; other backslashes stay.
                if index + 1 < len(word) and word[index + 1] in '"$\\':
                    out += word[index + 1]
                    index += 2
                else:
                    out += char
                    index += 1
            elif quote is None and char in "'\"":
                quote = char
                index += 1
            elif quote == '"' and char == '"':
                quote = None
                index += 1
            else:
                out += char
                index += 1
        return out

    def _lookup(self, name: str) -> str | None:
        if name in self.env:
            return self.env[name]
        if not self.known:
            raise ImageConfigRequired(f"`${name}` refers to the base image's environment")
        return None

    def _dollar(self, word: str, index: int) -> tuple[str, int]:
        """Expand the ``$`` at ``index``; return the text and the index after the expression."""
        rest = word[index + 1 :]
        if not rest.startswith("{"):
            match = _NAME.match(rest)
            if match is None:
                return "$", index + 1
            return self._lookup(match.group()) or "", index + 1 + match.end()
        match = _NAME.match(rest, 1)
        close = _closing_brace(rest)
        if match is None or close < 0:
            raise DockerfileNeedsBuild(f"unsupported variable expansion {word!r}")
        name = match.group()
        modifier = rest[match.end() : close]
        value = self._lookup(name)
        # The default or alternative is a word of its own and may hold `${...}` itself; it is expanded like one.
        if modifier == "":
            expanded = value or ""
        elif modifier.startswith(":-") or modifier.startswith("-"):
            default = modifier[2:] if modifier.startswith(":-") else modifier[1:]
            unset = value is None or (modifier.startswith(":-") and value == "")
            expanded = self.expand(default) if unset else value
        elif modifier.startswith(":+") or modifier.startswith("+"):
            alternative = modifier[2:] if modifier.startswith(":+") else modifier[1:]
            is_set = value is not None and (not modifier.startswith(":+") or value != "")
            expanded = self.expand(alternative) if is_set else ""
        else:
            raise DockerfileNeedsBuild(f"unsupported variable expansion {word!r}")
        return expanded, index + 2 + close


def _closing_brace(text: str) -> int:
    """Index of the ``}`` closing the ``{`` that opens ``text``, skipping nested ``${...}``; -1 when unterminated."""
    depth, index = 1, 1
    while index < len(text):
        if text.startswith("${", index):
            depth += 1
            index += 2
            continue
        if text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def _env_pairs(arguments: str) -> list[tuple[str, str]]:
    """``(key, raw value)`` pairs of an ``ENV`` line; values keep their quotes for the expander."""
    words = split_words(arguments)
    if not words:
        raise DockerfileNeedsBuild("ENV with no arguments")
    if "=" not in words[0]:
        # Legacy form: `ENV KEY value with spaces`. Docker splits on the first run of blanks (any kind) and
        # keeps the rest of the line as the value; a lone key is an error.
        parts = _BLANKS.split(arguments, maxsplit=1)
        if len(parts) < 2:
            raise DockerfileNeedsBuild(f"ENV {arguments} must have two arguments")
        return [(parts[0], parts[1])]
    pairs: list[tuple[str, str]] = []
    for word in words:
        key, separator, value = word.partition("=")
        if not separator or not key:
            raise DockerfileNeedsBuild(f"ENV {word} (not KEY=value)")
        pairs.append((key, value))
    return pairs


def _json_array(arguments: str) -> list | None:
    """The JSON array an exec-form instruction holds, or ``None`` when the arguments are shell form."""
    if not arguments.startswith("["):
        return None
    try:
        parsed = json.loads(arguments)
    except ValueError:
        return None
    return parsed if isinstance(parsed, list) else None


def has_heredoc(command: str) -> bool:
    """Whether ``command`` opens a Dockerfile heredoc, judged word by word as BuildKit does."""
    try:
        words = shlex.split(command, posix=False)
    except ValueError:
        words = command.split()
    return any(_HEREDOC_WORD.match(word) for word in words)


def dockerfile_has_heredoc(text: str) -> bool:
    """Whether any ``RUN``, ``COPY`` or ``ADD`` instruction of a Dockerfile opens a heredoc.

    Each instruction's arguments are judged as :func:`has_heredoc` judges one line, word by word as BuildKit
    does, so ``RUN cat > /x << 'EOF'`` (a space before the delimiter, which BuildKit does not read as a heredoc
    either) is not one. The body lines of a heredoc are not instructions, but they are only ever reached once
    the opening line has already answered.
    """
    for line in _logical_lines(text):
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].upper() in ("RUN", "COPY", "ADD") and has_heredoc(parts[1]):
            return True
    return False


def _run_command(arguments: str) -> str:
    """The validated arguments of a ``RUN`` line: a shell-form line, or the exec-form JSON array as written."""
    if arguments.startswith("--"):
        flag = arguments.split(maxsplit=1)[0].split("=")[0]
        raise DockerfileNeedsBuild(f"RUN {flag} (BuildKit flags)")
    exec_form = _json_array(arguments)
    if exec_form is not None:
        if not exec_form or not all(isinstance(part, str) for part in exec_form):
            raise DockerfileNeedsBuild(f"RUN {arguments}")
        return arguments
    if not arguments:
        raise DockerfileNeedsBuild("RUN with no command")
    if has_heredoc(arguments):
        raise DockerfileNeedsBuild("RUN with a heredoc (<<EOF)")
    return arguments


def _reject_escape_directive(text: str) -> None:
    """Refuse a Dockerfile whose leading comment block sets ``# escape=``.

    Docker reads parser directives only in the comments before the first instruction, and this loader
    joins continuation lines with a backslash alone (:func:`_logical_lines`), so a file that switches
    the escape character would be read with the wrong line structure (``RUN echo foo \\`` would swallow
    the next instruction). A build honours the directive; a ``# escape=`` after an instruction is a plain
    comment, as it is for Docker. ``# syntax=`` and ``# check=`` do not change how lines are read.
    """
    for line in text.splitlines():
        if not line.strip():
            continue
        if not line.lstrip().startswith("#"):
            return
        if re.match(r"\s*#\s*escape\s*=", line, re.IGNORECASE):
            raise DockerfileNeedsBuild("# escape= directive")


def parse_dockerfile(text: str) -> Dockerfile:
    """Read a pull-mode or overlay-mode Dockerfile; raise :class:`DockerfileNeedsBuild` for anything else."""
    _reject_escape_directive(text)
    image: str | None = None
    instructions: list[tuple[str, str]] = []
    entrypoint_cleared = False
    for line in _logical_lines(text):
        parts = line.split(None, 1)
        instruction = parts[0].upper()
        arguments = parts[1].strip() if len(parts) > 1 else ""
        if instruction == "FROM":
            if image is not None:
                raise DockerfileNeedsBuild("a second FROM (multi-stage build)")
            tokens = arguments.split()
            if tokens and tokens[0].startswith("--"):
                # `--platform` selects an image the registry resolver would not (it records linux/amd64).
                raise DockerfileNeedsBuild(f"FROM {tokens[0].split('=')[0]} (BuildKit flag)")
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
            instructions.append((instruction, _run_command(arguments)))
        elif instruction in ("WORKDIR", "USER"):
            if not arguments:
                raise DockerfileNeedsBuild(f"{instruction} with no argument")
            instructions.append((instruction, arguments))
        elif instruction == "ENV":
            _env_pairs(arguments)  # validate the shape now; values resolve later
            instructions.append((instruction, arguments))
        elif instruction == "LABEL":
            pass
        elif instruction == "ENTRYPOINT" and _json_array(arguments) == []:
            # The base image's entrypoint is not to run; the sandbox provider starts its own keepalive anyway.
            entrypoint_cleared = True
        elif instruction == "CMD" and _json_array(arguments) == []:
            pass  # CMD is replaced by the keepalive command whatever it was
        else:
            raise DockerfileNeedsBuild(f"{instruction}")
    if image is None:
        raise DockerfileNeedsBuild("no FROM instruction")
    return Dockerfile(image=image, instructions=tuple(instructions), entrypoint_cleared=entrypoint_cleared)


def resolve_dockerfile(dockerfile: Dockerfile, base: BaseImageConfig | None = None) -> OverlayImage:
    """Apply the instructions in order from the base image's state, as ``docker build`` would.

    With ``base`` given, the starting environment is its ``Env`` (plus Docker's default
    ``PATH`` when it has none), the working directory its ``WorkingDir`` or ``/`` and the
    user its ``User``. Without it, only a Dockerfile whose values do not depend on the base
    image resolves; anything else raises :class:`ImageConfigRequired`.
    """
    known = base is not None
    env: dict[str, str] = {}
    workdir: str | None = None
    user: str | None = None
    shell: tuple[str, ...] = DEFAULT_SHELL
    if base is not None:
        env = dict(base.env)
        env.setdefault("PATH", DEFAULT_PATH)
        workdir = base.workdir or "/"
        user = base.user
        shell = base.shell
    declared: dict[str, str] = {}
    runs: list[OverlayRun] = []
    workdirs: list[OverlayWorkdir] = []
    for instruction, arguments in dockerfile.instructions:
        expander = _Expander(env, known=known)
        if instruction == "RUN":
            if not known:
                raise ImageConfigRequired(
                    "RUN lines run with the base image's environment, working directory and user"
                )
            argv = _json_array(arguments)
            # Exec form runs its arguments directly; shell form goes through the image's shell.
            command = shlex.join(argv) if argv is not None else arguments
            runs.append(
                OverlayRun(
                    command=command,
                    workdir=workdir or "/",
                    env=dict(env),
                    user=user,
                    shell=None if argv is not None else shell,
                )
            )
        elif instruction == "ENV":
            # Docker expands every pair of one ENV line against the environment before the line, so
            # `ENV A=x B=$A` gives B the previous A; only the next instruction sees the new values.
            resolved = [(key, expander.expand(raw)) for key, raw in _env_pairs(arguments)]
            for key, value in resolved:
                env[key] = value
                declared[key] = value
        elif instruction == "WORKDIR":
            target = expander.expand(arguments)
            if posixpath.isabs(target):
                workdir = posixpath.normpath(target)
            elif workdir is None:
                raise ImageConfigRequired(f"WORKDIR {arguments} is relative to the base image's working directory")
            else:
                workdir = posixpath.normpath(posixpath.join(workdir, target))
            workdirs.append(OverlayWorkdir(path=workdir, user=user, before=len(runs)))
        elif instruction == "USER":
            user = expander.expand(arguments) or None
    return OverlayImage(
        image=dockerfile.image,
        workdir=workdir,
        env=declared,
        user=user,
        entrypoint_cleared=dockerfile.entrypoint_cleared,
        runs=tuple(runs),
        workdirs=tuple(workdirs),
    )


def overlay_image(text: str, base: BaseImageConfig | None = None) -> OverlayImage | None:
    """The base image, settings and ``RUN`` lines, or ``None`` when the file needs a build.

    Raises :class:`ImageConfigRequired` when the file depends on a base image configuration and none is given.
    """
    try:
        return resolve_dockerfile(parse_dockerfile(text), base)
    except DockerfileNeedsBuild:
        return None


def base_image_only(text: str, base: BaseImageConfig | None = None) -> BaseImage | None:
    """Return the base image and settings, or ``None`` when the file has ``RUN`` lines or needs a build."""
    try:
        parsed = parse_dockerfile(text)
    except DockerfileNeedsBuild:
        return None
    if parsed.has_runs:
        return None
    resolved = resolve_dockerfile(parsed, base)
    return BaseImage(
        image=resolved.image,
        workdir=resolved.workdir,
        env=resolved.env,
        user=resolved.user,
        entrypoint_cleared=resolved.entrypoint_cleared,
    )
