# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a published Harbor Compose overlay into a document the sandbox Compose adapter can start.

Harbor publishes ``environment/docker-compose.yaml`` as an overlay on its own base file: the
``main`` service is the task image, sidecars are prebuilt images. Starting it without Docker
needs each image's recorded OCI configuration (entrypoint, command, user, exposed ports,
healthcheck), which ``image_configs`` supplies, keyed by image reference.

Adapted from Gym's Terminal Bench 4 server (``resources_servers/terminal_bench_4/compose_config.py``).
"""

import ipaddress
import re
from collections.abc import Mapping
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


def _unquote_env_value(value: str, escaped_dollar: str = "$") -> tuple[str, str]:
    """One env_file value after ``=`` and the quote it came in (``'``, ``"`` or ``""`` for none): a quoted value
    ends at its closing quote and the rest is a comment; an unquoted value ends at the first ``#`` that follows
    whitespace. Double quotes honour the escapes ``\\n``, ``\\t``, ``\\"``, ``\\\\`` and ``\\$`` (the last one
    becoming ``escaped_dollar``); single quotes are literal."""
    value = value.lstrip()
    if value[:1] in ("'", '"'):
        quote = value[0]
        index = 1
        chars: list[str] = []
        while index < len(value):
            char = value[index]
            if quote == '"' and char == "\\" and index + 1 < len(value):
                escaped = value[index + 1]
                chars.append({"n": "\n", "t": "\t", "$": escaped_dollar}.get(escaped, escaped))
                index += 2
                continue
            if char == quote:
                return "".join(chars), quote
            chars.append(char)
            index += 1
        raise ComposeSubstitutionError("value", "env_file value has no closing quote")
    return re.split(r"\s+#", value, maxsplit=1)[0].rstrip(), ""


def env_file_entries(text: str, escaped_dollar: str = "$") -> list[tuple[str, str, str]]:
    """The ``KEY=value`` lines of a Compose env file in file order, as ``(key, value, quote)``. ``quote`` is
    ``'``, ``"`` or ``""`` for an unquoted value, which is how Compose's dotenv reader decides what to do with
    it: a single-quoted value is literal, the other two it interpolates. Blank lines and ``#`` comments are
    skipped, an optional ``export`` prefix dropped, quotes and inline comments handled as
    :func:`_unquote_env_value` says, a double-quoted ``\\$`` becoming ``escaped_dollar``; otherwise values are
    returned as written, ``$`` included. A line that is not ``KEY=value`` raises
    :class:`ComposeSubstitutionError` naming the line number only, as env files hold secrets."""
    entries: list[tuple[str, str, str]] = []
    for number, raw in enumerate(text.lstrip("\ufeff").splitlines(), start=1):
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
        unquoted, quote = _unquote_env_value(value, escaped_dollar)
        entries.append((key, unquoted, quote))
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


# The keys the sandbox Compose adapter (``nemo_gym.sandbox.adapters.docker_compose``) starts. Every other key
# of a document or a service is refused by name: at prepare, per task, and again by the adapter before it
# provisions anything.
SUPPORTED_COMPOSE_FIELDS = frozenset({"name", "services", "networks", "volumes"})
SUPPORTED_SERVICE_FIELDS = frozenset(
    {
        "image",
        "entrypoint",
        "command",
        "environment",
        "working_dir",
        "user",
        "depends_on",
        "healthcheck",
        "expose",
        "ports",
        "networks",
        "volumes",
        "cpus",
        "mem_limit",
        "labels",
        "restart",
        "network_mode",
        "cap_add",
        "shm_size",
        "x-sandbox",
        # The M28 fields, given their meaning by check_compose_fields below.
        "deploy",
        "extra_hosts",
        "pull_policy",
        "platform",
        "post_start",
        # The provider service options (providers.base.COMPOSE_SERVICE_OPTION_FIELDS), checked per provider there.
        "tmpfs",
        "ulimits",
        "sysctls",
        "security_opt",
    }
)


def _unsupported_fields(document: Any) -> list[tuple[str, str]]:
    """``(field, message)`` per offending key of ``document``, in document order: the top level first, then each
    service; ``field`` is the first unknown key of that level. Only mappings are inspected: a document or service
    that is not one fails the adapter's shape checks instead."""
    found: list[tuple[str, str]] = []
    if not isinstance(document, Mapping):
        return found
    unknown = sorted(str(key) for key in set(document) - SUPPORTED_COMPOSE_FIELDS)
    if unknown:
        found.append((unknown[0], f"Unsupported Compose fields: {unknown}"))
    services = document.get("services")
    if not isinstance(services, Mapping):
        return found
    for name, service in services.items():
        if not isinstance(service, Mapping):
            continue
        unknown = sorted(str(key) for key in set(service) - SUPPORTED_SERVICE_FIELDS)
        if unknown:
            found.append((unknown[0], f"Service {name!r}: unsupported Compose fields {unknown}"))
    return found


def unsupported_compose_fields(document: Any) -> list[str]:
    """The refusals the sandbox Compose adapter raises for the keys of ``document``: one message per offender in
    document order (the top level first, then each service), empty when every key is one the adapter starts.

    The key allow-list is the first step of :func:`check_compose_fields`, the one field check prepare and the
    adapter run (a refused key is an :class:`UnsupportedComposeFieldError` there). Pure and key-only, so prepare
    also screens the raw file of a task about to be built with it (``nemo_gym.tasks.harbor.cli``), before
    interpolation gives the values their shapes.
    """
    return [message for _, message in _unsupported_fields(document)]


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


DEFAULT_SANDBOX_PLATFORM = ("linux", "amd64")


def sandbox_platform(provider_options: Mapping[str, Any] | None) -> tuple[str, str]:
    """The ``(os, architecture)`` the sandboxes run: the OpenSandbox ``platform`` provider option
    (``{os, arch}``, see ``nemo_gym.sandbox.providers.opensandbox``) when a deployment sets one, otherwise
    linux/amd64, the default of every provider.
    """
    platform = (provider_options or {}).get("platform")
    if not isinstance(platform, Mapping):
        return DEFAULT_SANDBOX_PLATFORM
    os_name = str(platform.get("os") or DEFAULT_SANDBOX_PLATFORM[0]).lower()
    architecture = str(platform.get("arch") or DEFAULT_SANDBOX_PLATFORM[1]).lower()
    return os_name, architecture


def resolve_compose(
    document: dict, main_image: str, image_configs: dict, platform: tuple[str, str] = DEFAULT_SANDBOX_PLATFORM
) -> dict:
    """Apply Harbor's prebuilt-image base and normalize the published task overlay.

    ``platform`` is the ``(os, architecture)`` the sandboxes run (:func:`sandbox_platform`); a service whose
    recorded image is for another platform is refused, naming both.
    """
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
        recorded = (str(record.get("os") or "").lower(), str(record.get("architecture") or "").lower())
        if recorded != tuple(platform):
            raise ValueError(
                f"Service {name!r}: the recorded image {image!r} is {'/'.join(recorded)}, not "
                f"{'/'.join(platform)}, the platform the sandbox runs; prepare the dataset for that platform"
            )
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


def non_root_sidecar_users(document: dict, image_configs: dict) -> dict[str, str]:
    """The sidecars whose recorded image ``USER`` is not root, with that user.

    These are the services :func:`adapt_non_root_sidecars` keeps on their own user without host injection;
    prepare uses the same predicate so what it accepts is what the group runs. ``image_configs`` is read
    by reference and by pinned image, so the published document and the resolved one both work.
    """
    by_image = {record["image"]: record for record in image_configs.values() if "image" in record}
    users: dict[str, str] = {}
    for name, service in document["services"].items():
        if name == "main" or not isinstance(service, dict):
            continue
        record = by_image.get(service.get("image")) or image_configs.get(service.get("image")) or {}
        user = str((record.get("config") or {}).get("User") or "")
        if user and user not in ("root", "0"):
            users[name] = user
    return users


def check_sidecar_extra_hosts(document: dict, image_configs: dict) -> None:
    """Refuse ``extra_hosts`` on a sidecar the group runs without host injection, naming the service.

    A non-root sidecar gets no hosts file entries (see :func:`adapt_non_root_sidecars`), so host entries it
    declares would be silently lost; the adapter rejects that document at seed. Prepare runs this check once
    the image configurations are recorded, so the task fails there, with the same reason.
    """
    for name, user in non_root_sidecar_users(document, image_configs).items():
        if document["services"][name].get("extra_hosts"):
            raise ComposeFieldError(
                "extra_hosts",
                f"service {name!r} extra_hosts cannot be applied without host injection: the image runs as "
                f"user {user!r}, so the sidecar takes no host entries",
            )


def adapt_non_root_sidecars(document: dict, image_configs: dict) -> dict:
    """Keep a sidecar on its image's default non-root user without host-file injection.

    A non-root process cannot edit ``/etc/hosts``, so peer names are resolved into the
    environment variables that carry ``http://<service>`` URLs instead, and the explicit
    ``user`` copied from image metadata is dropped so the provider does not try to ``su``
    to the user it already runs as. Such a sidecar cannot take ``extra_hosts``
    (:func:`check_sidecar_extra_hosts` raises first). Mutates and returns ``document``.
    """
    check_sidecar_extra_hosts(document, image_configs)
    services = document["services"]
    for name in non_root_sidecar_users(document, image_configs):
        service = services[name]
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


class ComposeFieldError(ValueError):
    """A Compose service field prepare cannot run as written; ``field`` is the Compose key it names."""

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(message)


class UnsupportedComposeFieldError(ComposeFieldError):
    """A key the sandbox Compose adapter does not start at all (:func:`unsupported_compose_fields`), as opposed to a
    supported field written in a way the group cannot run."""


# ``restart`` values Compose defines; the sandbox group restarts nothing, so each is recorded, not applied.
RESTART_POLICIES = ("no", "always", "on-failure", "unless-stopped")
# Compose ``pull_policy`` values that mean "pull when the image is not present" once the age window is ignored.
_PULL_MISSING = ("missing", "if_not_present", "daily", "weekly")
# Fields of a ``deploy`` block the group gives a meaning; the rest name swarm behaviour and fail at prepare.
_DEPLOY_KEYS = frozenset({"resources", "replicas", "restart_policy"})
_NETWORK_KEYS = frozenset({"name", "driver", "external", "attachable", "labels"})
_SERVICE_NETWORK_KEYS = frozenset({"aliases"})
_HOOK_KEYS = frozenset({"command", "user", "working_dir", "environment"})
_ARCHITECTURES = {"amd64": "amd64", "x86_64": "amd64", "x86-64": "amd64", "arm64": "arm64", "aarch64": "arm64"}


def normalize_platform(value: str) -> str:
    """``os/arch[/variant]`` as ``os/arch`` with Docker's architecture aliases folded (``x86_64`` is ``amd64``)."""
    parts = str(value).strip().lower().split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"platform must be os/arch, got {value!r}")
    return f"{parts[0]}/{_ARCHITECTURES.get(parts[1], parts[1])}"


DEFAULT_PLATFORM = "linux/amd64"


def deployment_platform(provider_options: Mapping[str, Any] | None) -> str:
    """The platform a deployment runs, which a service's ``platform`` must match: the sandbox provider options'
    ``platform`` (OpenSandbox's ``{os, arch}``) when set, else linux/amd64, the one prepare records images for.
    Prepare and the server read it from the same ``sandbox_provider_options`` so the two checks agree: the
    ``os/arch`` form of :func:`sandbox_platform`, the reader the image checks use."""
    return normalize_platform("/".join(sandbox_platform(provider_options)))


def _cpus(value: Any, where: str) -> float:
    try:
        cpus = float(value)
    except (TypeError, ValueError):
        raise ComposeFieldError(where, f"{where} must be a number of CPUs, got {value!r}") from None
    if cpus <= 0:
        raise ComposeFieldError(where, f"{where} must be positive, got {value!r}")
    return cpus


def _memory(value: Any, where: str) -> int:
    try:
        return _bytes(value)
    except (TypeError, ValueError):
        raise ComposeFieldError(where, f"{where} must be a memory size such as 512M, got {value!r}") from None


def _devices(devices: Any, where: str) -> int:
    """GPUs requested by ``deploy.resources.reservations.devices``: every entry must ask for the ``gpu``
    capability with a numeric ``count`` or explicit ``device_ids``; ``count: all`` has no meaning in a pod."""
    if not isinstance(devices, list):
        raise ComposeFieldError(where, f"{where} must be a list")
    total = 0
    for index, device in enumerate(devices):
        entry = f"{where}[{index}]"
        if not isinstance(device, dict):
            raise ComposeFieldError(where, f"{entry} must be a mapping")
        unknown = set(device) - {"capabilities", "count", "device_ids", "driver", "options"}
        if unknown:
            raise ComposeFieldError(
                f"{entry}.{sorted(unknown)[0]}", f"{entry}: unsupported field(s) {sorted(unknown)}"
            )
        capabilities = device.get("capabilities")
        if not isinstance(capabilities, list) or "gpu" not in capabilities:
            raise ComposeFieldError(f"{entry}.capabilities", f"{entry}: only the gpu capability is supported")
        if device.get("driver") not in (None, "nvidia"):
            raise ComposeFieldError(f"{entry}.driver", f"{entry}: only the nvidia driver is supported")
        if device.get("options"):
            raise ComposeFieldError(f"{entry}.options", f"{entry}: device options are not supported")
        if "device_ids" in device:
            ids = device["device_ids"]
            if not isinstance(ids, list) or not ids:
                raise ComposeFieldError(f"{entry}.device_ids", f"{entry}: device_ids must be a non-empty list")
            if "count" in device:
                raise ComposeFieldError(f"{entry}.count", f"{entry}: give count or device_ids, not both")
            total += len(ids)
            continue
        count = device.get("count", 1)
        if count == "all":
            raise ComposeFieldError(f"{entry}.count", f"{entry}: count: all is not supported; give a number")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ComposeFieldError(f"{entry}.count", f"{entry}: count must be a positive integer, got {count!r}")
        total += count
    return total


def normalize_deploy(service_name: str, deploy: Any) -> dict[str, Any]:
    """A service's ``deploy`` block in the form the adapter reads.

    ``resources.limits`` and ``resources.reservations`` carry ``cpus`` (a number) and ``memory`` (bytes);
    ``reservations.devices`` becomes ``gpus``, the GPU count. ``replicas`` must be 1. ``restart_policy``
    is kept as written and recorded, like ``restart``. ``limits.pids`` and every other key name swarm
    scheduling and raise :class:`ComposeFieldError`.
    """
    where = f"service {service_name!r} deploy"
    if not isinstance(deploy, dict):
        raise ComposeFieldError("deploy", f"{where} must be a mapping")
    unknown = set(deploy) - _DEPLOY_KEYS
    if unknown:
        field = sorted(unknown)[0]
        raise ComposeFieldError(f"deploy.{field}", f"{where}.{field} is not supported in a sandbox group")
    out: dict[str, Any] = {}
    replicas = deploy.get("replicas", 1)
    if replicas != 1:
        raise ComposeFieldError("deploy.replicas", f"{where}.replicas must be 1, got {replicas!r}")
    if "restart_policy" in deploy:
        if not isinstance(deploy["restart_policy"], dict):
            raise ComposeFieldError("deploy.restart_policy", f"{where}.restart_policy must be a mapping")
        out["restart_policy"] = dict(deploy["restart_policy"])
    resources = deploy.get("resources") or {}
    if not isinstance(resources, dict) or set(resources) - {"limits", "reservations"}:
        raise ComposeFieldError("deploy.resources", f"{where}.resources takes limits and reservations only")
    out_resources: dict[str, Any] = {}
    for kind in ("limits", "reservations"):
        block = resources.get(kind) or {}
        prefix = f"deploy.resources.{kind}"
        if not isinstance(block, dict):
            raise ComposeFieldError(prefix, f"{where}.resources.{kind} must be a mapping")
        # ``gpus`` is this function's own output for ``devices``: the server normalizes once, writes the document,
        # and the adapter normalizes it again, so the normalized form must be a fixed point.
        allowed = {"cpus", "memory"} | ({"devices", "gpus"} if kind == "reservations" else set())
        unknown = set(block) - allowed
        if unknown:
            field = sorted(unknown)[0]
            raise ComposeFieldError(f"{prefix}.{field}", f"{where}.resources.{kind}.{field} is not supported")
        normalized: dict[str, Any] = {}
        if block.get("cpus") is not None:
            normalized["cpus"] = _cpus(block["cpus"], f"{prefix}.cpus")
        if block.get("memory") is not None:
            normalized["memory"] = _memory(block["memory"], f"{prefix}.memory")
        if block.get("devices"):
            if block.get("gpus") is not None:
                raise ComposeFieldError(f"{prefix}.gpus", f"{where}.resources.{kind}: give devices or gpus, not both")
            normalized["gpus"] = _devices(block["devices"], f"{prefix}.devices")
        elif block.get("gpus") is not None:
            gpus = block["gpus"]
            if isinstance(gpus, bool) or not isinstance(gpus, int) or gpus < 1:
                raise ComposeFieldError(
                    f"{prefix}.gpus", f"{where}.resources.{kind}.gpus must be a positive integer, got {gpus!r}"
                )
            normalized["gpus"] = gpus
        if normalized:
            out_resources[kind] = normalized
    if out_resources:
        out["resources"] = out_resources
    return out


def service_resources(service: dict) -> dict[str, Any]:
    """The sandbox resources a normalized service asks for: ``cpu``, ``memory_mib`` and ``gpu`` when set.

    ``deploy.resources.limits`` wins, then ``reservations``, then the legacy ``cpus``/``mem_limit`` keys.
    GPUs come from ``reservations.devices`` only, as Compose defines them.
    """
    resources: dict[str, Any] = {}
    deploy = (service.get("deploy") or {}).get("resources") or {}
    limits, reservations = deploy.get("limits") or {}, deploy.get("reservations") or {}
    cpus = limits.get("cpus", reservations.get("cpus", service.get("cpus")))
    memory = limits.get("memory", reservations.get("memory", service.get("mem_limit")))
    if cpus is not None:
        resources["cpu"] = float(cpus)
    if memory is not None:
        resources["memory_mib"] = (int(memory) + 1048575) // 1048576
    if reservations.get("gpus"):
        resources["gpu"] = int(reservations["gpus"])
    return resources


def service_gpus(service: dict) -> int:
    """GPUs a service requests through ``deploy.resources.reservations.devices`` (0 when none)."""
    return int(service_resources(service).get("gpu") or 0)


def _normalize_restart(service_name: str, value: Any) -> str:
    policy = str(value) if not isinstance(value, bool) else ("always" if value else "no")
    base = policy.split(":", 1)[0]
    if base not in RESTART_POLICIES or (":" in policy and (base != "on-failure" or not policy[11:].isdecimal())):
        raise ComposeFieldError(
            "restart", f"service {service_name!r} restart must be one of {', '.join(RESTART_POLICIES)}, got {value!r}"
        )
    return policy


def normalize_pull_policy(service_name: str, value: Any) -> str:
    """Compose ``pull_policy`` as ``always``, ``missing`` or ``never``.

    ``if_not_present`` is ``missing``; the age forms ``daily``, ``weekly`` and ``every_<duration>`` are also
    ``missing``, because a sandbox group holds no pull history. ``build`` raises: services do not build.
    """
    policy = str(value).strip().lower()
    if policy in ("always", "never"):
        return policy
    if policy in _PULL_MISSING or re.fullmatch(r"every_\d+[smhd]?", policy):
        return "missing"
    if policy == "build":
        raise ComposeFieldError("pull_policy", f"service {service_name!r} pull_policy: build is not supported")
    raise ComposeFieldError("pull_policy", f"service {service_name!r} pull_policy {value!r} is not a Compose value")


def normalize_extra_hosts(service_name: str, value: Any) -> dict[str, str]:
    """``extra_hosts`` as a hostname-to-address mapping, from the list (``host:ip``, ``host=ip``) or mapping form.
    ``host-gateway`` has no meaning in a pod and raises."""
    where = f"service {service_name!r} extra_hosts"
    pairs: list[tuple[str, str]] = []
    if isinstance(value, list):
        for item in value:
            text = str(item)
            separator = "=" if "=" in text and (":" not in text or text.index("=") < text.index(":")) else ":"
            host, found, address = text.partition(separator)
            if not found:
                raise ComposeFieldError("extra_hosts", f"{where}: entry {text!r} is not host:address")
            pairs.append((host.strip(), address.strip().strip("[]")))
    elif isinstance(value, dict):
        pairs = [(str(host), str(address).strip("[]")) for host, address in value.items()]
    else:
        raise ComposeFieldError("extra_hosts", f"{where} must be a list or a mapping")
    hosts: dict[str, str] = {}
    for host, address in pairs:
        if address == "host-gateway":
            raise ComposeFieldError("extra_hosts", f"{where}: host-gateway is not available in a sandbox group")
        try:
            ipaddress.ip_address(address)
        except ValueError:
            raise ComposeFieldError("extra_hosts", f"{where}: {host!r} needs an IP address, got {address!r}") from None
        if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,252})", host):
            raise ComposeFieldError("extra_hosts", f"{where}: invalid hostname {host!r}")
        hosts[host] = address
    return hosts


def normalize_hooks(service_name: str, hooks: Any) -> list[dict[str, Any]]:
    """``post_start`` entries as ``command`` (a string, run through the shell, or an argv list), optional ``user``,
    ``working_dir`` and ``environment`` (a mapping). ``privileged`` raises: no hook escalates."""
    where = f"service {service_name!r} post_start"
    if not isinstance(hooks, list):
        raise ComposeFieldError("post_start", f"{where} must be a list of hooks")
    out = []
    for index, hook in enumerate(hooks):
        if not isinstance(hook, dict) or "command" not in hook:
            raise ComposeFieldError("post_start", f"{where}[{index}] must be a mapping with a command")
        unknown = set(hook) - _HOOK_KEYS
        if unknown:
            field = sorted(unknown)[0]
            raise ComposeFieldError(f"post_start.{field}", f"{where}[{index}].{field} is not supported")
        command = hook["command"]
        if isinstance(command, list):
            command = [str(part) for part in command]
        elif not isinstance(command, str) or not command.strip():
            raise ComposeFieldError("post_start", f"{where}[{index}].command must be a string or a list")
        entry: dict[str, Any] = {"command": command}
        if hook.get("user") not in (None, ""):
            entry["user"] = str(hook["user"])
        if hook.get("working_dir"):
            entry["working_dir"] = str(hook["working_dir"])
        if hook.get("environment") is not None:
            entry["environment"] = _environment_mapping(service_name, hook["environment"], {})
        out.append(entry)
    return out


def normalize_service_options(service_name: str, service: dict) -> dict[str, Any]:
    """``tmpfs``, ``ulimits``, ``sysctls`` and ``security_opt`` in one shape each: a list of mounts, a mapping of
    name to an integer or ``{soft, hard}``, a mapping of key to string, and a list of options."""
    out: dict[str, Any] = {}
    where = f"service {service_name!r}"
    if "tmpfs" in service:
        mounts = service["tmpfs"]
        mounts = [mounts] if isinstance(mounts, str) else mounts
        if not isinstance(mounts, list) or not all(isinstance(m, str) and m.startswith("/") for m in mounts):
            raise ComposeFieldError("tmpfs", f"{where} tmpfs must be absolute container paths")
        out["tmpfs"] = list(mounts)
    if "ulimits" in service:
        limits = service["ulimits"]
        if not isinstance(limits, dict):
            raise ComposeFieldError("ulimits", f"{where} ulimits must be a mapping")
        normalized: dict[str, Any] = {}
        for name, limit in limits.items():
            if isinstance(limit, dict):
                if set(limit) != {"soft", "hard"} or not all(isinstance(limit[k], int) for k in ("soft", "hard")):
                    raise ComposeFieldError("ulimits", f"{where} ulimits.{name} needs integer soft and hard values")
                normalized[str(name)] = {"soft": limit["soft"], "hard": limit["hard"]}
            elif isinstance(limit, int) and not isinstance(limit, bool):
                normalized[str(name)] = limit
            else:
                raise ComposeFieldError("ulimits", f"{where} ulimits.{name} must be an integer or soft/hard")
        out["ulimits"] = normalized
    if "sysctls" in service:
        sysctls = service["sysctls"]
        if isinstance(sysctls, list):
            pairs = [str(item).partition("=") for item in sysctls]
            if any(not found for _, found, _ in pairs):
                raise ComposeFieldError("sysctls", f"{where} sysctls entries must be key=value")
            sysctls = {key: value for key, _, value in pairs}
        if not isinstance(sysctls, dict):
            raise ComposeFieldError("sysctls", f"{where} sysctls must be a mapping or a list of key=value")
        out["sysctls"] = {str(key): _compose_scalar(value) for key, value in sysctls.items()}
    if "security_opt" in service:
        options = service["security_opt"]
        if not isinstance(options, list) or not all(isinstance(o, str) for o in options):
            raise ComposeFieldError("security_opt", f"{where} security_opt must be a list of strings")
        out["security_opt"] = list(options)
    return out


def normalize_networks(document: dict) -> dict[str, dict[str, Any]]:
    """Top-level ``networks`` in the shape the adapter reads; every one maps onto the group's single pod network.

    ``name``, ``driver``, ``external``, ``attachable`` and ``labels`` are accepted and have no further meaning:
    services sharing a network reach each other by name, as on any Compose network. ``internal``, ``ipam``
    with a configuration, ``driver_opts`` and ``enable_ipv6`` ask for isolation or addressing the pod does
    not provide and raise :class:`ComposeFieldError` naming them.
    """
    declared = document.get("networks")
    if declared is None:
        return {}
    if not isinstance(declared, dict):
        raise ComposeFieldError("networks", "top-level networks must be a mapping")
    out: dict[str, dict[str, Any]] = {}
    for name, options in declared.items():
        options = dict(options or {})
        if not options.get("ipam"):
            options.pop("ipam", None)
        unknown = set(options) - _NETWORK_KEYS
        if unknown:
            field = sorted(unknown)[0]
            raise ComposeFieldError(
                f"networks.{name}.{field}",
                f"network {name!r}: {field} needs network isolation the sandbox group lacks",
            )
        out[str(name)] = options
    return out


def _service_networks(service_name: str, value: Any, declared: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """A service's ``networks`` as a mapping of network name to options; each must be declared (``default`` is
    implicit). Only ``aliases`` has a meaning; static addresses and the rest raise."""
    where = f"service {service_name!r} networks"
    if value is None:
        return {"default": {}}
    if isinstance(value, list):
        value = {str(name): {} for name in value}
    if not isinstance(value, dict):
        raise ComposeFieldError("networks", f"{where} must be a list or a mapping")
    out: dict[str, dict[str, Any]] = {}
    for name, options in value.items():
        options = dict(options or {})
        if name != "default" and name not in declared:
            raise ComposeFieldError("networks", f"{where}: {name!r} is not declared under top-level networks")
        unknown = set(options) - _SERVICE_NETWORK_KEYS
        if unknown:
            field = sorted(unknown)[0]
            raise ComposeFieldError(f"networks.{field}", f"{where}.{name}.{field} is not supported")
        aliases = options.get("aliases") or []
        if not isinstance(aliases, list) or not all(isinstance(alias, str) for alias in aliases):
            raise ComposeFieldError("networks.aliases", f"{where}.{name}.aliases must be a list of names")
        out[str(name)] = {"aliases": list(aliases)} if aliases else {}
    return out or {"default": {}}


def check_compose_fields(
    document: dict,
    *,
    platform: str = DEFAULT_PLATFORM,
    provider: str | None = None,
    service_option_fields: frozenset[str] | None = None,
    gpu_provider: str | None = None,
    gpu_service_option_fields: frozenset[str] | None = None,
) -> dict:
    """Give every M28 service field its defined meaning, or raise :class:`ComposeFieldError` naming the field.

    The one field check: prepare runs it per task (``nemo_gym.tasks.harbor.image_configs.check_task_compose_fields``)
    and the adapter again at start. A key outside the allow-list (:func:`unsupported_compose_fields`: a sidecar
    ``build:``, ``privileged``, ``logging``, ...) is an :class:`UnsupportedComposeFieldError` before any value is
    read. Mutates and returns ``document``, and applied to its own output returns it unchanged: the server normalizes
    once and the adapter again, from the written file. ``deploy`` is normalized (see :func:`normalize_deploy`);
    ``restart`` is validated and kept (recorded by the adapter, the group restarts nothing); ``pull_policy``
    becomes ``always``, ``missing`` or ``never``; ``platform`` must match ``platform``, the one the deployment
    runs (see :func:`deployment_platform`); ``networks`` (top level and per service) and ``extra_hosts`` take
    their mapping forms; ``post_start`` hooks are checked; ``tmpfs``, ``ulimits``, ``sysctls`` and
    ``security_opt`` are normalized and, when ``provider`` is given, each must be in ``service_option_fields``
    (the provider's ``SERVICE_OPTION_FIELDS``) or the error names the field and the provider. A sidecar that
    reserves GPUs through ``deploy.resources`` is checked against ``gpu_provider`` and
    ``gpu_service_option_fields`` instead when ``gpu_provider`` is given, since that is the provider it runs on
    (``main`` follows the task, not its ``deploy`` block). Without a provider name the capability check is left
    to the adapter, which runs it against the live provider of each service before any sandbox.
    """
    services = document.get("services") if isinstance(document, dict) else None
    if not isinstance(services, dict):
        return document
    # The key allow-list first: a key the group does not start is refused before any value is read. ``pre_stop``
    # is named in its own words, as the only hook kind Compose defines that the group has no moment for.
    for name, service in services.items():
        if isinstance(service, dict) and "pre_stop" in service:
            raise ComposeFieldError("pre_stop", f"service {name!r} pre_stop hooks are not supported")
    for field, message in _unsupported_fields(document):
        raise UnsupportedComposeFieldError(field, message)
    declared = normalize_networks(document)
    if declared:
        document["networks"] = declared
    expected = normalize_platform(platform)
    for name, service in services.items():
        if not isinstance(service, dict):
            continue
        if "deploy" in service:
            service["deploy"] = normalize_deploy(name, service["deploy"])
        if "restart" in service:
            service["restart"] = _normalize_restart(name, service["restart"])
        if "pull_policy" in service:
            service["pull_policy"] = normalize_pull_policy(name, service["pull_policy"])
        if "platform" in service:
            try:
                declared_platform = normalize_platform(service["platform"])
            except ValueError as exc:
                raise ComposeFieldError("platform", f"service {name!r} {exc}") from None
            if declared_platform != expected:
                raise ComposeFieldError(
                    "platform",
                    f"service {name!r} platform {service['platform']!r} does not match the deployment platform "
                    f"{expected}",
                )
            service["platform"] = declared_platform
        if "networks" in service:
            service["networks"] = _service_networks(name, service["networks"], declared)
        if "extra_hosts" in service:
            service["extra_hosts"] = normalize_extra_hosts(name, service["extra_hosts"])
        if "post_start" in service:
            service["post_start"] = normalize_hooks(name, service["post_start"])
        options = normalize_service_options(name, service)
        checked_against, supported = provider, service_option_fields
        if gpu_provider is not None and name != "main" and service_gpus(service):
            checked_against, supported = gpu_provider, gpu_service_option_fields
        for field, value in options.items():
            if checked_against is not None and field not in (supported or frozenset()):
                raise ComposeFieldError(
                    field, f"service {name!r} {field} is not supported by the {checked_against} sandbox provider"
                )
            service[field] = value
    return document
