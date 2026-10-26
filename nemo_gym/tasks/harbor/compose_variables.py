# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose interpolation for Harbor tasks.

The interpolation engine with Compose's operators, the dotenv reader with Compose's quoting
rules, the inlining of env_file entries into each service, the type restoration healthcheck
fields need after interpolation, the residual interpolation check, and Harbor's variable
precedence for a task: infrastructure variables, then the task's environment table, then the
process environment, then environment/.env, with Compose's empty-string rule and one warning
per unset name. It lives under tasks/harbor because it reads HarborTask and the task digest's
ignore rules. The error type and the two Compose shape helpers it raises and reads live in
nemo_gym.sandbox.compose_fields, which the adapter-side field checks share. The loader and the
build modules both resolve a task's Compose file through it.
"""

import logging
import os
import re
from collections import ChainMap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nemo_gym.sandbox.compose_fields import ComposeSubstitutionError, _environment_mapping
from nemo_gym.tasks.harbor.digest import context_digest, is_ignored
from nemo_gym.tasks.harbor.registry import ImageConfigError
from nemo_gym.tasks.harbor.task import HarborTask


LOGGER = logging.getLogger(__name__)

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _closing_brace(text: str, start: int) -> int:
    """Index of the ``}`` that closes the ``${`` opening at ``start``, nested ``${...}`` included."""
    depth = 0
    index = start
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
    raise ComposeSubstitutionError(text[start:], f"unterminated interpolation {text[start : start + 40]!r}")


def _substitute_text(text: str, variables: dict[str, str]) -> str:
    """Compose interpolation: ``$$`` (a literal dollar, kept doubled for ``_compose_literals``), ``${NAME}``,
    ``${NAME:-default}``, ``${NAME-default}``, ``${NAME:?error}``, ``${NAME?error}``, ``${NAME:+alternative}``,
    ``${NAME+alternative}`` and bare ``$NAME``. Defaults, alternatives and error texts are interpolated too, so
    ``${A:-${B:-x}}`` nests the way Compose documents."""
    out: list[str] = []
    index = 0
    while index < len(text):
        dollar = text.find("$", index)
        if dollar < 0:
            out.append(text[index:])
            break
        out.append(text[index:dollar])
        if text.startswith("$$", dollar):
            out.append("$$")
            index = dollar + 2
            continue
        if text.startswith("${", dollar):
            close = _closing_brace(text, dollar)
            body = text[dollar + 2 : close]
            match = _NAME.match(body)
            if match is None:
                raise ComposeSubstitutionError(body, f"invalid interpolation ${{{body}}}")
            name = match.group(0)
            rest = body[match.end() :]
            operator = ""
            argument = ""
            if rest:
                operator = rest[:2] if rest[:2] in (":-", ":?", ":+") else rest[:1]
                if operator not in ("-", "?", "+", ":-", ":?", ":+"):
                    raise ComposeSubstitutionError(name, f"invalid interpolation ${{{body}}}")
                argument = rest[len(operator) :]
            out.append(_resolve(name, operator, argument, variables))
            index = close + 1
            continue
        match = _NAME.match(text, dollar + 1)
        if match is None:
            # A lone ``$`` with no name after it is literal, as in Compose.
            out.append("$")
            index = dollar + 1
            continue
        out.append(_resolve(match.group(0), "", "", variables))
        index = match.end()
    return "".join(out)


def _resolve(name: str, operator: str, argument: str, variables: dict[str, str]) -> str:
    value = variables.get(name)
    unset = value is None
    empty = unset or (operator.startswith(":") and value == "")
    if operator in ("-", ":-"):
        return _substitute_text(argument, variables) if empty else value
    if operator in ("?", ":?"):
        if empty:
            message = _substitute_text(argument, variables) if argument else "no value"
            raise ComposeSubstitutionError(name, f"${{{name}}} is required: {message}")
        return value
    if operator in ("+", ":+"):
        return "" if empty else _substitute_text(argument, variables)
    if unset:
        raise ComposeSubstitutionError(name)
    return value


def substitute_compose(value: Any, variables: dict[str, str]) -> Any:
    """Interpolate ``${VAR}`` references in every string of a Compose document from ``variables`` alone.

    The process environment is never consulted, so a dataset resolves the same way on every
    deployment. A variable that is neither defined nor given a default raises
    :class:`ComposeSubstitutionError` naming it.
    """
    if isinstance(value, str):
        return _substitute_text(value, variables)
    if isinstance(value, list):
        return [substitute_compose(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: substitute_compose(item, variables) for key, item in value.items()}
    return value


# Escapes a double-quoted env_file value honours; ``\$`` is mapped by the caller's ``escaped_dollar``.
_DOUBLE_QUOTE_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\", '"': '"'}


def _quoted_env_value(text: str, line: int, escaped_dollar: str) -> tuple[str, str, int]:
    """The quoted env_file value opening at ``text[0]``: ``(value, quote, characters consumed)``. The value runs
    to its closing quote, across newlines if need be. Double quotes decode ``\\n``, ``\\r``, ``\\t``, ``\\\\``,
    ``\\"`` and ``\\$`` (the last becoming ``escaped_dollar``) and keep any other ``\\X`` as written; single
    quotes are literal except for ``\\'``, which is an escaped quote in either style."""
    quote = text[0]
    index = 1
    chars: list[str] = []
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            escaped = text[index + 1]
            if escaped == quote:
                chars.append(quote)
                index += 2
                continue
            if quote == '"' and (escaped == "$" or escaped in _DOUBLE_QUOTE_ESCAPES):
                chars.append(escaped_dollar if escaped == "$" else _DOUBLE_QUOTE_ESCAPES[escaped])
                index += 2
                continue
        if char == quote:
            return "".join(chars), quote, index + 1
        chars.append(char)
        index += 1
    raise ComposeSubstitutionError(f"line {line}", f"env_file line {line} has no closing quote")


def env_file_entries(text: str, escaped_dollar: str = "$") -> list[tuple[str, str, str]]:
    """The ``KEY=value`` lines of a Compose env file in file order, as ``(key, value, quote)``. ``quote`` is
    ``'``, ``"`` or ``""`` for an unquoted value, which is how Compose's dotenv reader decides what to do with
    it: a single-quoted value is literal, the other two it interpolates. Blank lines and ``#`` comments are
    skipped and an optional ``export`` prefix dropped. A quoted value ends at its closing quote, which may be on
    a later line, and the rest of that line is a comment; its escapes are decoded as :func:`_quoted_env_value`
    says, a double-quoted ``\\$`` becoming ``escaped_dollar``. An unquoted value ends at the first ``#`` that
    follows whitespace and is returned as written, ``$`` included. A line that is not ``KEY=value`` or a quote
    that never closes raises :class:`ComposeSubstitutionError` naming the line number only, as env files hold
    secrets."""
    entries: list[tuple[str, str, str]] = []
    text = text.lstrip("\ufeff").replace("\r\n", "\n")
    position = 0
    number = 0
    while position < len(text):
        newline = text.find("\n", position)
        end = len(text) if newline < 0 else newline
        raw = text[position:end]
        number += 1
        position = end + 1
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not _NAME.fullmatch(key):
            # The line itself is not echoed: env files hold secrets.
            raise ComposeSubstitutionError(f"line {number}", f"env_file line {number} is not KEY=value")
        if value.lstrip()[:1] not in ("'", '"'):
            entries.append((key, re.split(r"\s+#", value.lstrip(), maxsplit=1)[0].rstrip(), ""))
            continue
        # The quote's offset in the whole text: the first ``=`` of the raw line is the separator (neither the
        # leading whitespace nor ``export`` holds one), and the value may run on past this line.
        after_equals = raw[raw.index("=") + 1 :]
        start = end - len(after_equals) + (len(after_equals) - len(after_equals.lstrip()))
        unquoted, quote, consumed = _quoted_env_value(text[start:], number, escaped_dollar)
        entries.append((key, unquoted, quote))
        closing = start + consumed
        number += text.count("\n", start, closing)
        newline = text.find("\n", closing)
        position = len(text) if newline < 0 else newline + 1
    return entries


def parse_env_file(text: str) -> dict[str, str]:
    """``KEY=value`` lines of a Compose ``env_file``, taken literally.

    Lines are read as :func:`env_file_entries` reads them. Values are not interpolated: a ``$`` in a value
    is escaped as ``$$`` so the document's later interpolation and literal checks leave it as one ``$``.
    """
    return {key: value.replace("$", "$$") for key, value, _ in env_file_entries(text)}


def _env_file_path(base_dir: Path, path: str, service: str) -> Path:
    """``path`` resolved under ``base_dir`` (Harbor's ``${CONTEXT_DIR}/x.env`` is absolute and lands there too).
    A path that resolves outside, or a symlink, is refused: task folders are third-party content and the
    file must be part of what prepare digested. So is a file the task digest skips (``*~``, ``*.swp``,
    ``__pycache__`` and the like, see :func:`nemo_gym.tasks.harbor.digest.is_ignored`): it would change the
    document without changing the identity the server checks at seed."""
    base = Path(base_dir).resolve()
    file = Path(path) if Path(path).is_absolute() else base / path
    escapes = file.is_symlink() or not file.resolve().is_relative_to(base)
    if escapes:
        raise ComposeSubstitutionError(
            path, f"service {service!r} env_file {path!r} must stay under the task's environment folder"
        )
    if is_ignored(file.resolve().relative_to(base.parent).as_posix()):
        raise ComposeSubstitutionError(
            path, f"service {service!r} env_file {path!r} is excluded from the task digest; rename it"
        )
    return file


def inline_env_files(document: dict, base_dir: Path, variables: dict[str, str] | None = None) -> dict:
    """Read each service's ``env_file`` entries (relative to ``base_dir``) into its ``environment`` mapping.

    Every service's ``environment`` becomes a mapping of strings. A service's own entries win over its
    files, as in Compose. A bare ``NAME`` entry in a list-form ``environment`` (or ``NAME:`` with no value in
    the mapping form) takes the task variable of that name; when there is none the variable is left unset,
    as Compose leaves it, so it also removes the ``env_file`` value of that name. A missing required file
    raises :class:`ComposeSubstitutionError` naming it; an entry with ``required: false`` is skipped when
    absent; an entry with a ``format`` is refused, as only the default dotenv format is read. A file that
    does not parse raises with the service, the file and the line number, never the line.
    """
    variables = variables or {}
    services = document.get("services") if isinstance(document, dict) else None
    if not isinstance(services, dict):
        return document
    for name, service in services.items():
        if not isinstance(service, dict):
            continue
        merged: dict[str, str | None] = {}
        if "env_file" in service:
            entries = service.pop("env_file")
            if isinstance(entries, (str, dict)):
                entries = [entries]
            if not isinstance(entries, list):
                raise ComposeSubstitutionError(
                    "env_file", f"service {name!r} env_file must be a path or a list of paths"
                )
            for entry in entries:
                required = True
                path = entry
                if isinstance(entry, dict):
                    path = entry.get("path")
                    required = bool(entry.get("required", True))
                if not isinstance(path, str) or not path:
                    raise ComposeSubstitutionError("env_file", f"service {name!r} has an env_file entry with no path")
                if isinstance(entry, dict) and entry.get("format") is not None:
                    raise ComposeSubstitutionError(
                        path,
                        f"service {name!r} env_file {path!r}: format {entry['format']!r} is not supported "
                        "(only the default dotenv format is)",
                    )
                file = _env_file_path(base_dir, path, name)
                if not file.is_file():
                    if required:
                        raise ComposeSubstitutionError(
                            path, f"service {name!r} env_file {path!r} not found under the task's environment folder"
                        )
                    continue
                try:
                    merged.update(parse_env_file(file.read_text()))
                except ComposeSubstitutionError as exc:
                    raise ComposeSubstitutionError(path, f"service {name!r} env_file {path!r}: {exc}") from exc
        if merged or "environment" in service:
            merged.update(_environment_mapping(name, service.get("environment"), variables))
            service["environment"] = {key: value for key, value in merged.items() if value is not None}
    return document


# Compose healthcheck fields the sandbox adapter reads by type, which interpolation hands back as strings.
_HEALTHCHECK_BOOLEANS = {"true": True, "false": False}


def restore_compose_types(document: dict) -> dict:
    """Give the typed healthcheck fields their types back after interpolation, which turns every scalar it
    touches into a string: ``disable`` becomes a boolean (``${DISABLE:-false}`` must not disable the check by
    being a non-empty string) and ``retries`` an integer. Any other string value raises
    :class:`ComposeSubstitutionError` naming the field. Mutates and returns ``document``."""
    services = document.get("services") if isinstance(document, dict) else None
    if not isinstance(services, dict):
        return document
    for name, service in services.items():
        health = service.get("healthcheck") if isinstance(service, dict) else None
        if not isinstance(health, dict):
            continue
        disable = health.get("disable")
        if isinstance(disable, str):
            if disable.strip().lower() not in _HEALTHCHECK_BOOLEANS:
                raise ComposeSubstitutionError(
                    f"services.{name}.healthcheck.disable",
                    f"service {name!r} healthcheck disable must be true or false, not {disable!r}",
                )
            health["disable"] = _HEALTHCHECK_BOOLEANS[disable.strip().lower()]
        retries = health.get("retries")
        if isinstance(retries, str):
            if not retries.strip().isdigit():
                raise ComposeSubstitutionError(
                    f"services.{name}.healthcheck.retries",
                    f"service {name!r} healthcheck retries must be an integer, not {retries!r}",
                )
            health["retries"] = int(retries.strip())
    return document


def residual_interpolation(value: Any, path: str = "") -> str | None:
    """The key path (``services.api.environment.DB_PASS``, ``services.a.command[1]``) of the first string still
    carrying a ``$NAME`` or ``${...}`` after substitution, or ``None``. The path, not the value: the value next to
    a residual may be a secret the host supplied, and the caller reports this in an error message."""
    if isinstance(value, str):
        match = re.search(r"\$(?:\{|[A-Za-z_])", value.replace("$$", ""))
        return (path or "the document") if match else None
    if isinstance(value, list):
        found = (residual_interpolation(v, f"{path}[{i}]") for i, v in enumerate(value))
        return next((r for r in found if r), None)
    if isinstance(value, dict):
        found = (residual_interpolation(v, f"{path}.{k}" if path else str(k)) for k, v in value.items())
        return next((r for r in found if r), None)
    return None


# In-sandbox paths Harbor's legacy ``${HOST_*_PATH}`` and ``${ENV_*_PATH}`` log-mount variables stand for.
LOG_MOUNT_PATHS = {"VERIFIER_LOGS": "/logs/verifier", "AGENT_LOGS": "/logs/agent", "ARTIFACTS": "/logs/artifacts"}
# Where a task's ``tests/`` folder lands in the sandbox, the value of ``TEST_DIR``. Harbor's terminal-bench mapper
# writes ``TEST_DIR=${TEST_DIR}`` into its Compose files and Harbor never sets the variable, so it is empty under
# Harbor; Gym defines it, a deliberate improvement.
TEST_DIR = "/tests"
# Harbor's ``[environment].env`` template (``_TEMPLATE_PATTERN`` in ``harbor/utils/env.py`` at v0.22.0): a whole
# value ``${VAR}`` or ``${VAR:-default}``, read from the host environment.
HARBOR_ENV_TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")
# The project directory's env file Compose interpolates with, below everything the command's environment sets.
COMPOSE_DOTENV = ".env"
# ``(task id, variable)`` pairs already warned about: the server loads a task's document once per episode.
_WARNED_UNSET: set[tuple[str, str]] = set()


def compose_variables(task: HarborTask, *, unset: set[str] | None = None) -> dict[str, str]:
    """The variables a task's Compose file may interpolate, resolved in the order Harbor's ``docker compose`` sees.

    Harbor runs ``docker compose`` with the process environment, the task's ``[environment].env`` over it and
    its own infrastructure variables over both (``merge_compose_env`` in
    ``harbor/environments/docker/compose_env.py`` and ``_compose_env_vars`` in ``docker.py`` at v0.22.0, called
    with ``include_os_env=True``), and Compose reads the project directory's ``.env`` below everything the
    command's environment sets. Highest first:

    1. Gym's infrastructure set: ``MAIN_IMAGE_NAME`` (the resolved task image), ``CONTEXT_DIR`` (the task's
       ``environment/`` folder), ``TEST_DIR`` (:data:`TEST_DIR`), ``COMPOSE_PROJECT_NAME`` (``hb__<hash>``,
       Harbor's environment id of the context and the ``main`` image name, a stable placeholder: compose-go
       injects the real value from ``--project-name``, the trial session id under Harbor and ``compose-<uuid>``
       per episode in Gym's adapter, which has no Compose-generated names for it to match), ``PREBUILT_IMAGE_NAME``
       when ``docker_image`` is declared, ``CPUS`` and ``MEMORY`` (as ``<n>M``, Harbor's spelling) from the
       task's resources, and the ``HOST_*_PATH`` / ``ENV_*_PATH`` log mounts as the ``/logs`` tree;
    2. the task's ``[environment].env``, each value read as Harbor's ``resolve_env_vars`` reads it
       (``harbor/utils/env.py``): a whole value ``${VAR}`` or ``${VAR:-default}`` comes from the process
       environment, the default standing in when the variable is unset, and one with neither raises
       :class:`ImageConfigError` naming the task and the variable, as Harbor fails the trial; any other value
       is literal;
    3. the process environment;
    4. ``environment/.env``, its unquoted and double-quoted values interpolated against the levels above and
       the lines before them, as Compose's dotenv reader does; single-quoted values are literal.

    A reference none of them sets takes its default (``${VAR:-x}``, ``${VAR-x}``), else it is an empty string,
    as docker compose makes it, with one warning naming the task and the variable (added to ``unset`` when
    given, so the caller warns once for the whole document, else logged here). ``${VAR:?msg}`` and
    ``${VAR?msg}`` raise :class:`ImageConfigError` with the message. The process environment is not part of
    the task's digest: prepare and the environment server must run with the same variables, as Harbor's one
    process does.

    Every value is literal to the document, as it is to Compose (which splices a variable's value in verbatim and
    never re-scans it): a ``$`` in a host, task or infrastructure value is doubled here, the convention
    :func:`parse_env_file` and :func:`_dotenv_variables` keep for a literal dollar, so a password such as
    ``$2b$12$...`` neither trips the document's residual check nor loses a ``$$`` when the sandbox starts.
    """
    environment = task.config.environment
    context = task.path / "environment"
    infra: dict[str, str] = {"MAIN_IMAGE_NAME": task.image or "", "CONTEXT_DIR": str(context), "TEST_DIR": TEST_DIR}
    if context.is_dir():
        # Not the runtime project name (Harbor: the trial session id; Gym: the adapter's compose-<uuid>). A stable
        # per-task value, so files like o11y-bench's `o11y-stack-${COMPOSE_PROJECT_NAME}` alias and URL agree.
        infra["COMPOSE_PROJECT_NAME"] = f"hb__{task.context_id or context_digest(context)}"
    if environment.docker_image:
        infra["PREBUILT_IMAGE_NAME"] = environment.docker_image
    if environment.cpus:
        infra["CPUS"] = str(environment.cpus)
    if environment.memory_mb:
        infra["MEMORY"] = f"{environment.memory_mb}M"
    for suffix, path in LOG_MOUNT_PATHS.items():
        infra[f"HOST_{suffix}_PATH"] = path
        infra[f"ENV_{suffix}_PATH"] = path
    host = {key: _literal(value) for key, value in os.environ.items()}
    above = {
        **host,
        **_task_env(task, environment.env or {}),
        **{key: _literal(value) for key, value in infra.items()},
    }
    collected: set[str] = set() if unset is None else unset
    variables = {**_dotenv_variables(task, context, above, collected), **above}
    if unset is None:
        _warn_unset(task, collected)
    return variables


def _literal(value: str) -> str:
    """``value`` as the document's interpolation and literal checks read a literal: every ``$`` doubled."""
    return value.replace("$", "$$")


def _task_env(task: HarborTask, env: Mapping[str, Any]) -> dict[str, str]:
    """``[environment].env`` as Harbor's ``resolve_env_vars`` hands it to ``docker compose`` (see
    :func:`compose_variables`), each value literal to the document (:func:`_literal`)."""
    resolved: dict[str, str] = {}
    for key, value in env.items():
        match = HARBOR_ENV_TEMPLATE.fullmatch(str(value))
        if match is None:
            resolved[str(key)] = _literal(str(value))
            continue
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            resolved[str(key)] = _literal(os.environ[name])
        elif default is not None:
            resolved[str(key)] = _literal(default)
        else:
            raise ImageConfigError(
                f"{task.task_id}: [environment].env {key} reads ${{{name}}}, which is not set in the process "
                "environment and has no default"
            )
    return resolved


def _dotenv_variables(task: HarborTask, context: Path, above: Mapping[str, str], unset: set[str]) -> dict[str, str]:
    """``environment/.env`` read as Compose's dotenv reader reads it (``compose-go/dotenv``): an unquoted or
    double-quoted value is interpolated against ``above`` and then the lines before it, a single-quoted value is
    literal. A dollar that stays literal is kept doubled, as :func:`parse_env_file` keeps it, so the document's
    literal check leaves it alone."""
    dotenv = context / COMPOSE_DOTENV
    if not dotenv.is_file():
        return {}
    variables: dict[str, str] = {}
    try:
        for key, value, quote in env_file_entries(dotenv.read_text(), escaped_dollar="$$"):
            if quote == "'":
                variables[key] = value.replace("$", "$$")
            else:
                variables[key] = substitute_with_empty(value, ChainMap(above, variables), unset)
    except ComposeSubstitutionError as exc:
        raise ImageConfigError(f"{task.task_id}: environment/{COMPOSE_DOTENV}: {exc}") from exc
    return variables


def substitute_with_empty(value: Any, variables: Mapping[str, str], unset: set[str]) -> Any:
    """:func:`substitute_compose` with Compose's rule for a bare ``${VAR}`` or ``$VAR`` nobody sets: an empty
    string, the name added to ``unset`` for the caller to warn about once. Only the bare form raises the resolver's
    default message; ``${VAR:-x}`` keeps its default, ``${VAR:?msg}`` its refusal, and every other failure passes
    through unchanged. The loader and the Dockerfile build (``nemo_gym.tasks.harbor.image_build``) interpolate a
    task's Compose file through this one rule, so the two never disagree on the same file."""
    if isinstance(value, list):
        return [substitute_with_empty(item, variables, unset) for item in value]
    if isinstance(value, dict):
        return {key: substitute_with_empty(item, variables, unset) for key, item in value.items()}
    if not isinstance(value, str):
        return value
    empty: dict[str, str] = {}
    while True:
        try:
            return substitute_compose(value, ChainMap(variables, empty))
        except ComposeSubstitutionError as exc:
            if str(exc) != str(ComposeSubstitutionError(exc.name)) or exc.name in empty:
                raise
            empty[exc.name] = ""
            unset.add(exc.name)


def _warn_unset(task: HarborTask, unset: set[str]) -> None:
    for name in sorted(unset):
        if (task.task_id, name) in _WARNED_UNSET:
            continue
        _WARNED_UNSET.add((task.task_id, name))
        LOGGER.warning(
            "%s: Compose variable ${%s} has no value from the task, the process environment or environment/.env; "
            "it is an empty string, as docker compose makes it",
            task.task_id,
            name,
        )
