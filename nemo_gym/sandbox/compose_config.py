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
from typing import Any
from urllib.parse import urlsplit


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
