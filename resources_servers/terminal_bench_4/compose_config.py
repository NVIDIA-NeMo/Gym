# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve published Harbor Compose inputs using recorded OCI startup metadata."""

import re
from collections.abc import Collection, Mapping
from copy import deepcopy
from typing import Any


_VARIABLE = re.compile(r"\$\$|\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:?-)([^}]*))?\}")


def _interpolate(value: Any, variables: Mapping[str, str]) -> Any:
    """Apply Compose ``${NAME}``, ``${NAME:-default}`` and ``${NAME-default}`` substitutions."""
    if isinstance(value, str):

        def substitute(match: re.Match) -> str:
            name, separator, default = match.groups()
            current = variables.get(name) if name else None
            if name is None or (current is None and separator is None):
                # $$ escapes and unknown variables stay for _compose_literals.
                return match[0]
            return default if current is None or (separator == ":-" and current == "") else current

        return _VARIABLE.sub(substitute, value)
    if isinstance(value, list):
        return [_interpolate(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _interpolate(item, variables) for key, item in value.items()}
    return value


def prebuilt_compose(
    document: Any, images: Mapping[str, str], variables: Mapping[str, str], ignored: Collection[str] = ()
) -> Any:
    """Run Harbor Compose services that build from Dockerfiles from prebuilt images.

    Applies Harbor's Compose variables (task CPUs and memory), replaces each
    sidecar ``build`` with its configured image (``main`` uses the task image),
    and removes the ``ignored`` service settings, which a deployment cannot
    enforce. Ignoring ``networks`` puts every service on the default network.
    """
    document = _interpolate(deepcopy(document), variables)
    if not isinstance(document, dict) or not isinstance(document.get("services"), dict):
        return document
    if "networks" in ignored:
        document.pop("networks", None)
    for name, service in document["services"].items():
        if not isinstance(service, dict):
            continue
        if service.pop("build", None) is not None:
            service.pop("pull_policy", None)
            if name != "main":
                if name not in images:
                    raise ValueError(f"Service {name!r} builds from a Dockerfile and has no configured prebuilt image")
                service["image"] = images[name]
        for key in ignored:
            service.pop(key, None)
    return document


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


def resolve_compose(document: dict, main_image: str, image_configs: dict, task: str | None = None) -> dict:
    """Apply Harbor's prebuilt-image base and normalize the published task overlay.

    Image metadata is acquired and digest-checked upstream. No image registry or
    Docker installation is needed while executing the benchmark. Images that a
    deployment builds from a task's Dockerfiles have no published name; their
    records are keyed ``<task>/<service>`` and keep the configured image.
    """
    resolved = _compose_literals(deepcopy(document))
    if not isinstance(resolved, dict) or not isinstance(resolved.get("services"), dict):
        raise ValueError("Compose requires a services mapping")
    services = resolved["services"]
    # This is Harbor's docker-compose-prebuilt.yaml base command, including for
    # images with a service-starting ENTRYPOINT that subsequently execs it.
    services["main"] = {
        "image": main_image,
        "command": ["sh", "-c", "sleep infinity"],
        **services.get("main", {}),
    }
    for name, service in services.items():
        image = service.get("image")
        record = image_configs.get(image)
        if record is None and task is not None:
            record = image_configs.get(f"{task}/{name}")
        if record is None:
            raise ValueError(f"Service {name!r} has no recorded OCI startup metadata for {image!r}")
        if (record["os"], record["architecture"]) != ("linux", "amd64"):
            raise ValueError(f"Service {name!r} requires a supported Linux/amd64 image")
        config = record["config"]
        service["image"] = record.get("image", image)
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
