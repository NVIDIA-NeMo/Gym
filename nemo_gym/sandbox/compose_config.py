# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a published Harbor Compose overlay into a document the sandbox Compose adapter can start.

Harbor publishes ``environment/docker-compose.yaml`` as an overlay on its own base file: the
``main`` service is the task image, sidecars are prebuilt images. Starting it without Docker
needs each image's recorded OCI configuration (entrypoint, command, user, exposed ports,
healthcheck), which ``image_configs`` supplies, keyed by image reference.

Adapted from Gym's Terminal Bench 4 server (``resources_servers/terminal_bench_4/compose_config.py``).
"""

import re
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class ComposeSubstitutionError(ValueError):
    """A Compose file names a variable or an ``env_file`` that prepare cannot resolve."""

    def __init__(self, name: str, message: str | None = None) -> None:
        self.name = name
        super().__init__(message or f"unresolved Compose variable ${{{name}}}")


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


def _unquote_env_value(value: str) -> str:
    """One env_file value after ``=``: a quoted value ends at its closing quote and the rest is a comment;
    an unquoted value ends at the first ``#`` that follows whitespace. Double quotes honour the escapes
    ``\\n``, ``\\t``, ``\\"``, ``\\\\`` and ``\\$``; single quotes are literal."""
    value = value.lstrip()
    if value[:1] in ("'", '"'):
        quote = value[0]
        index = 1
        chars: list[str] = []
        while index < len(value):
            char = value[index]
            if quote == '"' and char == "\\" and index + 1 < len(value):
                escaped = value[index + 1]
                chars.append({"n": "\n", "t": "\t"}.get(escaped, escaped))
                index += 2
                continue
            if char == quote:
                return "".join(chars)
            chars.append(char)
            index += 1
        raise ComposeSubstitutionError("value", "env_file value has no closing quote")
    return re.split(r"\s+#", value, maxsplit=1)[0].rstrip()


def parse_env_file(text: str) -> dict[str, str]:
    """``KEY=value`` lines of a Compose ``env_file``, taken literally.

    Blank lines and ``#`` comments are skipped, an optional ``export`` prefix dropped, quotes and inline
    comments handled as :func:`_unquote_env_value` says. Values are not interpolated: a ``$`` in a value
    is escaped as ``$$`` so the document's later interpolation and literal checks leave it as one ``$``.
    """
    values: dict[str, str] = {}
    for number, raw in enumerate(text.lstrip("﻿").splitlines(), start=1):
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
        values[key] = _unquote_env_value(value).replace("$", "$$")
    return values


def _env_file_path(base_dir: Path, path: str, service: str) -> Path:
    """``path`` resolved under ``base_dir`` (Harbor's ``${CONTEXT_DIR}/x.env`` is absolute and lands there too).
    A path that resolves outside, or a symlink, is refused: task folders are third-party content and the
    file must be part of what prepare digested."""
    base = Path(base_dir).resolve()
    file = Path(path) if Path(path).is_absolute() else base / path
    escapes = file.is_symlink() or not file.resolve().is_relative_to(base)
    if escapes:
        raise ComposeSubstitutionError(
            path, f"service {service!r} env_file {path!r} must stay under the task's environment folder"
        )
    return file


def _compose_scalar(value: Any) -> str:
    """An environment value as Compose would emit it: booleans lower-case, ``None`` empty."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _environment_mapping(service_name: str, environment: Any, variables: dict[str, str]) -> dict[str, str]:
    """A service's ``environment`` as a mapping. A bare ``NAME`` in the list form takes the task variable of
    that name and is dropped when there is none, as Compose drops an unset host variable."""
    if environment is None:
        return {}
    if isinstance(environment, list):
        pairs: dict[str, str] = {}
        for item in environment:
            key, separator, value = str(item).partition("=")
            if separator:
                pairs[key] = value
            elif key in variables:
                pairs[key] = variables[key]
        return pairs
    if isinstance(environment, dict):
        return {str(key): _compose_scalar(value) for key, value in environment.items()}
    raise ComposeSubstitutionError("environment", f"service {service_name!r} environment must be a list or a mapping")


def inline_env_files(document: dict, base_dir: Path, variables: dict[str, str] | None = None) -> dict:
    """Read each service's ``env_file`` entries (relative to ``base_dir``) into its ``environment`` mapping.

    Every service's ``environment`` becomes a mapping of strings. A service's own entries win over its
    files, as in Compose. A bare ``NAME`` entry in a list-form ``environment`` takes the task variable of
    that name and is dropped otherwise, as Compose drops a host variable that is not set. A missing
    required file raises :class:`ComposeSubstitutionError` naming it; an entry with ``required: false``
    is skipped when absent.
    """
    variables = variables or {}
    services = document.get("services") if isinstance(document, dict) else None
    if not isinstance(services, dict):
        return document
    for name, service in services.items():
        if not isinstance(service, dict):
            continue
        merged: dict[str, str] = {}
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
                file = _env_file_path(base_dir, path, name)
                if not file.is_file():
                    if required:
                        raise ComposeSubstitutionError(
                            path, f"service {name!r} env_file {path!r} not found under the task's environment folder"
                        )
                    continue
                merged.update(parse_env_file(file.read_text()))
        if merged or "environment" in service:
            service["environment"] = {**merged, **_environment_mapping(name, service.get("environment"), variables)}
    return document


def residual_interpolation(value: Any) -> str | None:
    """The first ``$NAME`` or ``${...}`` still present in a document after substitution, or ``None``."""
    if isinstance(value, str):
        match = re.search(r"\$(?:\{|[A-Za-z_])", value.replace("$$", ""))
        return value if match else None
    if isinstance(value, list):
        return next((r for r in (residual_interpolation(v) for v in value) if r), None)
    if isinstance(value, dict):
        return next((r for r in (residual_interpolation(v) for v in value.values()) if r), None)
    return None


def _compose_literals(value: Any) -> Any:
    if isinstance(value, str):
        escaped = value.replace("$$", "\x00")
        if re.search(r"\$(?:\{|[A-Za-z_])", escaped):
            raise ValueError("Resolve Compose environment substitutions before starting the sandbox collection")
        return escaped.replace("\x00", "$")
    if isinstance(value, list):
        return [_compose_literals(item) for item in value]
    if isinstance(value, dict):
        return {key: _compose_literals(item) for key, item in value.items()}
    return value


def _bytes(value: str | int) -> int:
    if isinstance(value, int):
        return value
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmgt]?)(?:i?b)?", value.lower())
    if match is None:
        raise ValueError(f"Invalid Compose memory size: {value!r}")
    return int(float(match[1]) * 1024 ** " kmgt".index(match[2] or " "))


def resolve_compose(document: dict, main_image: str, image_configs: dict) -> dict:
    """Apply Harbor's prebuilt-image base and normalize the published task overlay."""
    resolved = _compose_literals(deepcopy(document))
    if not isinstance(resolved, dict) or not isinstance(resolved.get("services"), dict):
        raise ValueError("Compose requires a services mapping")
    services = resolved["services"]
    for name, service in services.items():
        # ``main:`` with no body is how Compose spells "the base service, unchanged"; a null
        # override leaves Harbor's base ``main`` in place, so it reads as an empty mapping here.
        if service is None:
            services[name] = {}
        elif not isinstance(service, dict):
            raise ValueError(f"Service {name!r} must be a mapping")
    services["main"] = {
        "image": main_image,
        "command": ["sh", "-c", "sleep infinity"],
        **services.get("main", {}),
    }
    for name, service in services.items():
        image = service.get("image")
        if image not in image_configs:
            raise ValueError(f"Service {name!r} has no recorded OCI startup metadata for {image!r}")
        record = image_configs[image]
        if (record["os"], record["architecture"]) != ("linux", "amd64"):
            raise ValueError(f"Service {name!r} requires a supported Linux/amd64 image")
        config = record["config"]
        service["image"] = record["image"]
        explicit_entrypoint = service.get("entrypoint") is not None
        if not explicit_entrypoint:
            service["entrypoint"] = config.get("Entrypoint") or []
        if service.get("command") is None:
            service["command"] = [] if explicit_entrypoint else config.get("Cmd") or []
        if config.get("WorkingDir"):
            service.setdefault("working_dir", config["WorkingDir"])
        if config.get("User"):
            service.setdefault("user", config["User"])
        service["expose"] = list(dict.fromkeys([*service.get("expose", []), *config.get("ExposedPorts", {})]))
        if isinstance(service.get("environment"), list):
            environment = {}
            for item in service["environment"]:
                if "=" not in item:
                    raise ValueError(f"Service {name!r} has an unresolved environment variable: {item!r}")
                key, value = item.split("=", 1)
                environment[key] = value
            service["environment"] = environment
        if isinstance(service.get("depends_on"), list):
            service["depends_on"] = {
                dependency: {"condition": "service_started"} for dependency in service["depends_on"]
            }
        for key in ("shm_size", "mem_limit"):
            if key in service:
                service[key] = _bytes(service[key])
        if config.get("Healthcheck") and "healthcheck" not in service:
            health = config["Healthcheck"]
            service["healthcheck"] = {"test": health["Test"]}
            for source, target in (("Interval", "interval"), ("Timeout", "timeout"), ("StartPeriod", "start_period")):
                if health.get(source):
                    service["healthcheck"][target] = f"{health[source] / 1_000_000_000}s"
            if health.get("Retries"):
                service["healthcheck"]["retries"] = health["Retries"]
    return resolved


def adapt_non_root_sidecars(document: dict, image_configs: dict) -> dict:
    """Keep a sidecar on its image's default non-root user without host-file injection.

    A non-root process cannot edit ``/etc/hosts``, so peer names are resolved into the
    environment variables that carry ``http://<service>`` URLs instead, and the explicit
    ``user`` copied from image metadata is dropped so the provider does not try to ``su``
    to the user it already runs as. Mutates and returns ``document``.
    """
    services = document["services"]
    by_image = {record["image"]: record for record in image_configs.values()}
    for name, service in services.items():
        if name == "main":
            continue
        record = by_image.get(service.get("image")) or image_configs.get(service.get("image")) or {}
        user = str((record.get("config") or {}).get("User") or "")
        if not user or user in ("root", "0"):
            continue
        service.pop("user", None)
        peers = [
            key
            for key, value in (service.get("environment") or {}).items()
            if isinstance(value, str) and urlsplit(value).scheme and urlsplit(value).hostname in services
        ]
        options = {"hosts": []}
        if peers:
            options["resolve_environment"] = peers
        service["x-sandbox"] = options
    return document


def opensandbox_shm_labels(document: dict, label: str = "nemo.nvidia.com/shm") -> dict:
    """OpenSandbox takes a service's shared-memory size as a label. Mutates and returns ``document``."""
    for service in document["services"].values():
        if service.get("shm_size") is not None:
            service.setdefault("labels", {})[label] = str(service["shm_size"])
    return document
