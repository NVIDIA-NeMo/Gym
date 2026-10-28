# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Compose services through the sandbox API.

Input YAML is resolved upstream; unsupported runtime semantics are rejected
before provisioning services. Service fields get their meaning from
``nemo_gym.sandbox.compose_config.check_compose_fields``: ``deploy.resources``
sizes each service's sandbox, ``restart`` and ``pull_policy`` are recorded (and
the pull policy applied where the provider can), ``networks`` scope which
services see each other by name, ``extra_hosts`` add hostnames, ``post_start``
hooks run once the service is up, and ``tmpfs``, ``ulimits``, ``sysctls`` and
``security_opt`` need a provider that implements
``SupportsSandboxServiceOptions``. A service may run on its own provider
(``service_providers``), the way a GPU sidecar lands on a GPU deployment, and
then resolves its external volumes through that provider's own mapping
(``service_volume_sources``).
"""

import asyncio
import logging
import re
import shlex
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from graphlib import CycleError, TopologicalSorter
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import yaml

from nemo_gym.data.tasks.harbor.digest import is_ignored
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.compose_fields import (
    ComposeFieldError,
    UnsupportedComposeFieldError,
    check_compose_fields,
    service_resources,
)
from nemo_gym.sandbox.compose_volumes import LOGS_TREE, first_link, lexical_target
from nemo_gym.sandbox.providers import SandboxSpec, create_provider
from nemo_gym.sandbox.providers.base import (
    COMPOSE_SERVICE_OPTION_FIELDS,
    SupportsSandboxEndpoint,
    SupportsSandboxNetwork,
    SupportsSandboxPortForwarding,
    SupportsSandboxPullPolicy,
    SupportsSandboxRuntimeRequirements,
    SupportsSandboxServiceOptions,
    SupportsSandboxSharedStorage,
)


LOGGER = logging.getLogger(__name__)
# Printed by the shell of ``_prepare_logs`` when it removed a non-empty target.
_HIDDEN_MARK = "gym-logs-target-hidden"


def _provider_name(provider) -> str:
    return getattr(provider, "name", type(provider).__name__)


def _merge_provider_options(*layers: Mapping[str, Any]) -> dict[str, Any]:
    """Provider options from several sources: lists (``run_args``, ``volumes``) concatenate, mappings
    (``extensions``) merge with later layers winning, scalars are replaced."""
    merged: dict[str, Any] = {}
    for layer in layers:
        for key, value in layer.items():
            current = merged.get(key)
            if isinstance(current, list) and isinstance(value, (list, tuple)):
                merged[key] = [*current, *value]
            elif isinstance(current, Mapping) and isinstance(value, Mapping):
                merged[key] = {**current, **value}
            else:
                merged[key] = list(value) if isinstance(value, tuple) else value
    return merged


def _seconds(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    parts = re.findall(r"(\d+(?:\.\d+)?)(ns|us|ms|s|m|h)", value)
    if not parts or "".join(number + unit for number, unit in parts) != value:
        raise ValueError(f"Invalid Compose duration: {value!r}")
    return sum(
        float(number) * {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1, "m": 60, "h": 3600}[unit]
        for number, unit in parts
    )


class AsyncSandboxCompose:
    """Start services from a Compose YAML file and own their sandbox lifecycle.

    Deployment settings stay in the provider and optional ``service_specs``.
    Compose fields override the corresponding spec fields, except ``environment``: a spec's ``env`` is
    the caller's host-resolved override for that service and wins over the service's Compose
    ``environment`` (see ``_prepare``). ``service_providers`` names another provider (instance or
    single-key config) for particular services; every provider must reach the others' sandboxes on
    the network. ``platform`` is the one the deployment runs, which a service's ``platform`` must
    match. ``restart_policies`` and ``pull_policies`` record what each service declared.


    ``x-sandbox.uploads`` entries (files or folders copied into a service once it starts) are read in
    this process, so every source must lie under ``upload_root``, the one folder the caller vouches for
    (the task folder the planner validated against); a document with uploads and no root is refused.
    ``x-sandbox.logs`` entries name the paths of Harbor's ``/logs`` tree a service bind-mounted: each
    source is created, world-writable, before the service's command, hooks and healthcheck, and a target
    elsewhere is linked to it, the way Docker creates both bind targets at container create.

    ``volume_sources`` maps the external volume names and foreign bind paths of the collection's provider to
    its shared storage; ``service_volume_sources`` gives a service on its own provider that provider's
    mapping, which its volumes resolve through instead. ``service_setup`` names, per service, a coroutine
    function the group awaits with that service's sandbox once it is created and its runtime is configured,
    before its uploads, its command, its hooks and its healthcheck: the caller's own image setup (the Harbor
    server runs an overlay Dockerfile's ``RUN`` lines there, so they precede the mounts and the command, as
    a build precedes ``compose up``). An exception from it fails the start and stops the group.
    """

    def __init__(
        self,
        provider,
        compose_file: str | Path | None,
        *,
        service_specs: Mapping[str, SandboxSpec] | None = None,
        timeout_s: float = 1200,
        poll_interval_s: float = 0.5,
        volume_init_image: str = "alpine:3.22",
        volume_sources: Mapping[str, str] | None = None,
        service_providers: Mapping[str, Any] | None = None,
        platform: str = "linux/amd64",
        upload_root: str | Path | None = None,
        service_volume_sources: Mapping[str, Mapping[str, str]] | None = None,
        service_setup: Mapping[str, Callable[[AsyncSandbox], Awaitable[None]]] | None = None,
    ):
        self.provider = create_provider(provider) if isinstance(provider, Mapping) else provider
        self.service_providers = {
            name: create_provider(value) if isinstance(value, Mapping) else value
            for name, value in (service_providers or {}).items()
        }
        self.platform = platform
        self.restart_policies: dict[str, str] = {}
        self.pull_policies: dict[str, str] = {}
        self._service_options: dict[str, dict[str, Any]] = {}
        self.compose_file = Path(compose_file).resolve() if compose_file is not None else None
        self.document: dict[str, Any] = {}
        self.service_specs = dict(service_specs or {})
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        if timeout_s <= 0 or poll_interval_s <= 0:
            raise ValueError("Compose timeout_s and poll_interval_s must be positive")
        self.volume_init_image = volume_init_image
        self.volume_sources = dict(volume_sources or {})
        self.service_volume_sources = {name: dict(mapping) for name, mapping in (service_volume_sources or {}).items()}
        self.service_setup = dict(service_setup or {})
        self.upload_root = Path(upload_root).resolve() if upload_root is not None else None
        self.project = "compose-" + uuid.uuid4().hex
        self.services: dict[str, AsyncSandbox] = {}
        self._plans: dict[str, dict[str, Any]] = {}
        self._runtime_metadata: dict[str, dict[str, str]] = {}
        self._started = False
        self._ready = False
        self._closed = False
        self._volume_helpers: dict[int, AsyncSandbox] = {}
        self._stop_task: asyncio.Task | None = None
        self._processes: dict[str, asyncio.Task] = {}
        self._seeds: list[AsyncSandbox] = []

    def _load(self):
        return yaml.safe_load(self.compose_file.read_text(encoding="utf-8"))

    def _provider_for(self, name: str):
        """The provider a service runs on: its own from ``service_providers``, else the collection's."""
        return self.service_providers.get(name, self.provider)

    def _volume_sources_for(self, name: str) -> dict[str, str]:
        """The shared-storage mapping a service's volumes resolve through: its own provider's from
        ``service_volume_sources``, else the collection's ``volume_sources``."""
        return self.service_volume_sources.get(name, self.volume_sources)

    def _providers(self) -> list[Any]:
        """Every distinct provider of the collection, the default first."""
        distinct: list[Any] = [self.provider]
        for provider in self.service_providers.values():
            if all(provider is not seen for seen in distinct):
                distinct.append(provider)
        return distinct

    def _validate(self) -> list[str]:
        if not isinstance(self.document, Mapping):
            raise ValueError("Compose requires a mapping")
        services = self.document.get("services")
        if not isinstance(services, Mapping) or not services:
            raise ValueError("Compose requires a non-empty services mapping")
        for field in ("networks", "volumes"):
            if not isinstance(self.document.get(field) or {}, Mapping):
                raise ValueError(f"Compose {field} requires a mapping")
        for name, service in services.items():
            if not isinstance(service, Mapping):
                raise ValueError(f"Service {name!r} requires a mapping")
        try:
            # The one field check, shared with prepare (``nemo_gym.data.tasks.harbor.image_configs``), which reports a
            # refused field per task before any sandbox exists; here it is the last line before provisioning. A key
            # the group does not start is refused first, then every M28 field takes its normalized shape, so the
            # checks below read one form each.
            check_compose_fields(self.document, platform=self.platform)
        except UnsupportedComposeFieldError as error:
            raise NotImplementedError(str(error)) from error
        except ComposeFieldError as error:
            raise ValueError(str(error)) from error
        for field, mapping in (
            ("service_providers", self.service_providers),
            ("service_volume_sources", self.service_volume_sources),
            ("service_setup", self.service_setup),
        ):
            unknown_services = set(mapping) - set(services)
            if unknown_services:
                raise ValueError(f"{field} names unknown services: {sorted(unknown_services)}")
        networks = self.document.get("networks") or {}
        dependencies = {}
        for name, service in services.items():
            for field in ("environment", "depends_on", "networks", "healthcheck", "labels", "x-sandbox"):
                if not isinstance(service.get(field) or {}, Mapping):
                    raise ValueError(f"Service {name!r}: resolve {field} to a mapping upstream")
            for field in ("ports", "volumes"):
                if not isinstance(service.get(field, []), list) or any(
                    not isinstance(item, Mapping) for item in service.get(field, [])
                ):
                    raise ValueError(f"Service {name!r}: resolve {field} to a list of mappings upstream")
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name):
                raise ValueError(f"Invalid Compose service name: {name!r}")
            provider = self._provider_for(name)
            # Recorded, not applied: the group starts each service once and reports an exit through the
            # dependency gates. Compose's restart loop has no counterpart in a sandbox.
            self.restart_policies[name] = str(service.get("restart", "no"))
            if not service.get("image"):
                raise ValueError(f"Service {name!r} requires a prebuilt image in the Compose YAML")
            runtime = {"cap_add": tuple(service.get("cap_add") or ()), "shm_size": service.get("shm_size")}
            if runtime["cap_add"] or runtime["shm_size"] is not None:
                if not isinstance(provider, SupportsSandboxRuntimeRequirements):
                    raise NotImplementedError("Sandbox provider does not support runtime requirements")
                self._runtime_metadata[name] = provider.validate_runtime_requirements(**runtime) or {}
            declared_options = {field: service[field] for field in COMPOSE_SERVICE_OPTION_FIELDS if field in service}
            if declared_options:
                supported = (
                    provider.SERVICE_OPTION_FIELDS if isinstance(provider, SupportsSandboxServiceOptions) else set()
                )
                for field in sorted(set(declared_options) - set(supported)):
                    raise NotImplementedError(
                        f"Service {name!r}: {field} is not supported by the {_provider_name(provider)} sandbox provider"
                    )
                self._service_options[name] = provider.service_options(
                    tmpfs=tuple(declared_options.get("tmpfs", ())),
                    ulimits=declared_options.get("ulimits"),
                    sysctls=declared_options.get("sysctls"),
                    security_opt=tuple(declared_options.get("security_opt", ())),
                )
            if "pull_policy" in service:
                policy = service["pull_policy"]
                self.pull_policies[name] = policy
                if isinstance(provider, SupportsSandboxPullPolicy):
                    self._service_options[name] = _merge_provider_options(
                        self._service_options.get(name, {}), provider.pull_policy_options(policy)
                    )
            options = service.get("x-sandbox") or {}
            if set(options) - {"hosts", "resolve_environment", "uploads", "logs"}:
                raise NotImplementedError(f"Service {name!r}: unsupported x-sandbox options")
            logs = options.get("logs", [])
            if not isinstance(logs, list) or any(
                not isinstance(entry, Mapping) or set(entry) != {"source", "target"} for entry in logs
            ):
                raise ValueError(f"Service {name!r}: x-sandbox.logs requires a list of source/target mappings")
            for entry in logs:
                source, target = entry["source"], entry["target"]
                if not isinstance(source, str) or not (source == LOGS_TREE or source.startswith(LOGS_TREE + "/")):
                    raise ValueError(f"Service {name!r}: x-sandbox.logs source {source!r} is not under {LOGS_TREE}")
                if not isinstance(target, str) or not target.startswith("/") or ".." in target.split("/"):
                    raise ValueError(f"Service {name!r}: x-sandbox.logs target {target!r} must be an absolute path")
                if not PurePosixPath(target).name:
                    # `/.` and `/././` name the root as `/` does: the planner's rule, for a document it did not write.
                    raise ValueError(f"Service {name!r}: x-sandbox.logs target {target!r} is the container root")
                # Compared in the lexical form ``_prepare_logs`` links (`/logs/.` and `//logs` are `/logs`): the
                # spelling must not get a nesting target past this check (M22 review).
                source_path, target_path = lexical_target(source) + "/", lexical_target(target) + "/"
                if source_path != target_path and (
                    source_path.startswith(target_path) or target_path.startswith(source_path)
                ):
                    # The target is replaced by a link to the source: one inside the other would delete the
                    # source or link it to itself.
                    raise ValueError(f"Service {name!r}: x-sandbox.logs target {target!r} and source {source!r} nest")
            uploads = options.get("uploads", [])
            if not isinstance(uploads, list) or any(
                not isinstance(entry, Mapping) or set(entry) != {"source", "target"} for entry in uploads
            ):
                raise ValueError(f"Service {name!r}: x-sandbox.uploads requires a list of source/target mappings")
            if uploads and self.upload_root is None:
                raise ValueError(f"Service {name!r}: x-sandbox.uploads need an upload_root to read sources from")
            for entry in uploads:
                source, target = entry["source"], entry["target"]
                if not isinstance(source, str) or not Path(source).exists():
                    raise ValueError(f"Service {name!r}: upload source {source!r} does not exist")
                # Sources are read in this process: only the folder the caller vouches for, links resolved.
                if not Path(source).resolve().is_relative_to(self.upload_root):
                    raise ValueError(
                        f"Service {name!r}: upload source {source!r} lies outside the upload root {self.upload_root}"
                    )
                if not isinstance(target, str) or not target.startswith("/") or ".." in target.split("/"):
                    raise ValueError(f"Service {name!r}: upload target {target!r} must be an absolute container path")
                if not PurePosixPath(target).name:
                    raise ValueError(f"Service {name!r}: upload target {target!r} is the container root")
            selected = options.get("resolve_environment", [])
            if not isinstance(selected, list) or any(not isinstance(key, str) for key in selected):
                raise ValueError("x-sandbox.resolve_environment requires a list of environment variable names")
            if "hosts" in options and options["hosts"] != []:
                raise ValueError("x-sandbox.hosts only supports [] to explicitly disable host injection")
            if options.get("hosts") == [] and service.get("extra_hosts"):
                raise ValueError(f"Service {name!r}: extra_hosts cannot be applied without host injection")
            dependencies[name] = dict(service.get("depends_on") or {})
            for dependency, config in dependencies[name].items():
                if not isinstance(config, Mapping):
                    raise ValueError(f"Service {name!r}: resolve dependency {dependency!r} options upstream")
                if dependency not in services:
                    raise ValueError(f"Service {name!r}: unknown dependency {dependency!r}")
                if (
                    set(config) - {"condition", "required", "restart"}
                    or config.get("restart")
                    or config.get("required") is False
                ):
                    raise NotImplementedError(f"Service {name!r}: unsupported depends_on options")
                if config.get("condition", "service_started") not in {
                    "service_started",
                    "service_healthy",
                    "service_completed_successfully",
                }:
                    raise ValueError(f"Service {name!r}: invalid dependency condition")
            mode = service.get("network_mode")
            if mode:
                target = mode.removeprefix("service:")
                if not mode.startswith("service:") or target not in services or target == name:
                    raise NotImplementedError(f"Unsupported Compose network_mode: {mode!r}")
                if service.get("networks") or service.get("ports"):
                    raise NotImplementedError(
                        "Forwarded service network mode cannot declare networks or published ports"
                    )
                if not isinstance(provider, SupportsSandboxPortForwarding):
                    raise NotImplementedError("Sandbox provider does not support service network forwarding")
                provider.validate_port_forwarding()
                dependencies[name].setdefault(target, {"condition": "service_started"})
            for port in service.get("ports", []):
                if set(port) - {"target", "published", "host_ip", "protocol", "mode"}:
                    raise NotImplementedError(f"Service {name!r}: unsupported ports options")
                if port.get("published") not in (None, "0", 0) or port.get("host_ip"):
                    raise NotImplementedError("Fixed published ports/host_ip cannot be honored by sandbox endpoints")
                if port.get("protocol", "tcp") != "tcp" or port.get("mode", "ingress") != "ingress":
                    raise NotImplementedError("Only TCP sandbox endpoint ports are supported")
                if not isinstance(provider, SupportsSandboxEndpoint):
                    raise NotImplementedError("Sandbox provider does not support service endpoints")
            for volume in service.get("volumes", []):
                # Volumes are seeded and mounted through the provider that runs the service (see _prepare_volumes).
                if not isinstance(provider, SupportsSandboxSharedStorage):
                    raise NotImplementedError(
                        f"Service {name!r}: the {_provider_name(provider)} sandbox provider does not support shared "
                        "volumes"
                    )
                if not isinstance(volume, Mapping) or volume.get("type") not in {"bind", "volume"}:
                    raise NotImplementedError("Only normalized bind and named volumes are supported")
                if set(volume) - {"type", "source", "target", "read_only", "bind", "volume"}:
                    raise NotImplementedError("Unsupported Compose volume options")
                if not volume.get("source") or not str(volume.get("target", "")).startswith("/"):
                    raise ValueError("Volumes require a source and absolute target")
                if set(volume.get("bind") or {}) - {"create_host_path"} or set(volume.get("volume") or {}) - {
                    "nocopy"
                }:
                    raise NotImplementedError("Unsupported bind/volume options")
                # Resolved through the mapping of the provider that runs the service (see _prepare_volumes).
                volume_sources = self._volume_sources_for(name)
                if volume["type"] == "bind" and volume["source"] not in volume_sources:
                    raise NotImplementedError("Remote bind mounts require a volume_sources mapping to shared storage")
                if volume["type"] == "volume":
                    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", volume["source"]):
                        raise ValueError("Named volumes require a valid Compose volume name")
                    declared = self.document.get("volumes") or {}
                    if volume["source"] not in declared:
                        raise ValueError(f"Undeclared Compose volume: {volume['source']!r}")
                    # A bodiless declaration (`volumes:\n  data:`) is valid Compose and loads as None.
                    declaration = declared[volume["source"]] or {}
                    if set(declaration) - {"name", "external"}:
                        raise NotImplementedError("Unsupported named volume driver/options")
                    if declaration.get("external") and volume["source"] not in volume_sources:
                        raise NotImplementedError("External volumes require a volume_sources mapping")
                source = volume_sources.get(volume["source"], f"{self.project}/{volume['source']}")
                provider.shared_volume_options(source, volume["target"], read_only=volume.get("read_only", False))
        # Forwarding chains (a -> service:b -> service:c) resolve to one namespace; a cycle is named here, before
        # the dependency sort reports it anonymously.
        for name in services:
            self._forwarding_root(name)
        # Services reach each other by name on the networks they share; a custom network is a name scope on the
        # group's one pod network, nothing more. Aliases must be unambiguous per network: a service's name is
        # reserved on the networks it is on (a forwarded service on its chain root's, as ``_memberships`` says),
        # not on every network, so ``main`` may alias ``db`` on a network ``db`` is not on. An alias that names a
        # service the aliasing service shares any network with is ambiguous too: ``_visible_hosts`` would hand
        # that service two addresses for the one name. So is a name two services hold on different networks once a
        # third service is on both (``a`` on ``front`` and ``back``, ``b`` aliasing ``x`` on ``front``, ``c`` aliasing
        # ``x`` on ``back``, or ``c`` aliasing ``b`` on ``back`` while ``b`` is on ``front``): Docker's DNS would
        # answer ``a`` with both addresses, a hosts file holds one. After the forwarding roots: ``_memberships``
        # walks the chains.
        memberships = self._memberships()
        aliases_seen: dict[str, set[str]] = {
            network: {name for name, joined in memberships.items() if network in joined}
            for network in {"default", *networks}
        }
        # name or alias -> {network: the service holding it there}; a service holds its name where it is on.
        alias_holders: dict[str, dict[str, str]] = {
            name: dict.fromkeys(joined, name) for name, joined in memberships.items()
        }
        for name, service in services.items():
            for network, network_options in (service.get("networks") or {"default": {}}).items():
                for alias in (network_options or {}).get("aliases", []):
                    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", alias):
                        raise ValueError(f"Invalid network alias: {alias!r}")
                    if alias != name and (
                        alias in aliases_seen[network] or memberships.get(alias, set()) & memberships[name]
                    ):
                        raise ValueError(f"Ambiguous Compose service alias: {alias!r}")
                    for other, holder in alias_holders.get(alias, {}).items():
                        if holder != name and any({network, other} <= joined for joined in memberships.values()):
                            raise ValueError(
                                f"Ambiguous Compose service alias: {alias!r} names {holder!r} on {other!r} and "
                                f"{name!r} on {network!r}, and a service is on both networks"
                            )
                    aliases_seen[network].add(alias)
                    alias_holders.setdefault(alias, {})[network] = name
        try:
            order = list(TopologicalSorter(dependencies).static_order())
        except CycleError as error:
            raise ValueError("Compose dependency cycle") from error
        for provider in self._providers():
            if not isinstance(provider, SupportsSandboxNetwork):
                raise NotImplementedError("Sandbox provider does not support networking between sandboxes")
            provider.validate_networking()
        return order

    async def _prepare(self):
        for name, service in self.document["services"].items():
            image = service["image"]
            entrypoint = service.get("entrypoint") or []
            command = service.get("command") or []
            if isinstance(entrypoint, str):
                entrypoint = shlex.split(entrypoint)
            if isinstance(command, str):
                command = shlex.split(command)
            argv = list(entrypoint) + list(command or [])
            if not argv:
                raise ValueError(f"Service {name!r}: resolve entrypoint or command upstream before starting Compose")
            env = {key: str(value) for key, value in (service.get("environment") or {}).items() if value is not None}
            for key in (service.get("x-sandbox") or {}).get("resolve_environment", []):
                url = urlsplit(env.get(key, ""))
                if not url.scheme or url.hostname not in self.document["services"]:
                    raise ValueError(f"Service {name!r}: environment {key!r} must be a URL naming a Compose service")
                # Validate the port before provisioning any sandboxes.
                _ = url.port
            health = service.get("healthcheck") or {}
            if set(health) - {"test", "disable", "interval", "timeout", "retries", "start_period", "start_interval"}:
                raise NotImplementedError(f"Service {name!r}: unsupported healthcheck options")
            test = health.get("test") or ["NONE"]
            if health.get("disable"):
                test = ["NONE"]
            if not isinstance(test, list) or test[0] not in {"CMD", "CMD-SHELL", "NONE"}:
                raise ValueError(f"Service {name!r}: invalid healthcheck test")
            if (test[0] == "CMD-SHELL" and len(test) != 2) or (test[0] == "CMD" and len(test) < 2):
                raise ValueError(f"Service {name!r}: healthcheck requires a command")
            for field in ("timeout", "interval", "start_interval", "start_period"):
                if field in health and _seconds(health[field]) < (0 if field == "start_period" else 1e-9):
                    raise ValueError(f"Service {name!r}: invalid healthcheck {field}")
            if not isinstance(health.get("retries", 3), int) or health.get("retries", 3) < 1:
                raise ValueError(f"Service {name!r}: healthcheck retries must be positive")
            ports = [int(str(port).removesuffix("/tcp")) for port in service.get("expose", [])]
            ports += [int(port["target"]) for port in service.get("ports", [])]
            spec = self.service_specs.get(name, SandboxSpec(ttl_s=self.timeout_s + 3600))
            # deploy.resources limits, then reservations, then the legacy cpus/mem_limit keys, over the spec's.
            resources = {**dict(vars(spec.resources)), **service_resources(service)}
            user = service.get("user")
            if user == "":
                user = None
            if isinstance(user, str) and ":" in user:
                raise NotImplementedError(f"Service {name!r}: user:group overrides are unsupported by the sandbox API")
            if isinstance(user, str) and user.isdecimal():
                user = int(user)
            runtime = f"/tmp/{self.project}/{name}"
            self._plans[name] = {
                "command": shlex.join(argv),
                "user": user,
                "health": {**health, "test": test},
                "shell": ["/bin/sh", "-c"],
                "runtime": runtime,
                "hooks": list(service.get("post_start") or []),
                "spec": replace(
                    spec,
                    image=image,
                    # The spec's env is the caller's host-resolved override and wins over the service's Compose
                    # ``environment``. This is Harbor's order for ``main``: its Docker environment appends the
                    # task's Compose file and then its own env override file, ``{services: {main: {environment:
                    # _startup_env()}}}`` (harbor 0.22.0 ``environments/docker/docker.py`` 363-370,
                    # ``environments/docker/__init__.py`` 20-25), and ``_startup_env`` is the task's
                    # ``[environment].env`` resolved on the host (``environments/base.py`` 278-285). Compose
                    # merges later files over earlier ones per key, so a ``${X:-}`` in the task file that
                    # interpolated to empty never hides the host value. The Harbor server passes ``main`` a
                    # spec env already layered image ENV < this file's ``environment`` < ``[environment].env``,
                    # so an image ENV it carries does not climb above the file. Harbor's override names ``main``
                    # only; callers pass an empty env for sidecars, which keep their Compose ``environment``.
                    env={**env, **spec.env},
                    workdir=service.get("working_dir", spec.workdir),
                    resources=resources,
                    ports=tuple(dict.fromkeys(ports)),
                    entrypoint=["/bin/sh", "-c", "while :; do sleep 3600; done"],
                    metadata={
                        **spec.metadata,
                        **(service.get("labels") or {}),
                        **self._runtime_metadata.get(name, {}),
                    },
                    provider_options=_merge_provider_options(
                        spec.provider_options, self._service_options.get(name, {})
                    ),
                ),
            }
        for name, service in self.document["services"].items():
            if service.get("network_mode"):
                target = service["network_mode"].removeprefix("service:")
                if not self._plans[target]["spec"].ports:
                    raise ValueError(f"Service {target!r} must expose TCP ports for forwarding")
            for dep, options in (service.get("depends_on") or {}).items():
                if options.get("condition") == "service_healthy" and self._plans[dep]["health"]["test"][0] == "NONE":
                    raise ValueError(f"Service {name!r} requires {dep!r} healthy, but it has no healthcheck")

    @property
    def _volume_helper(self) -> AsyncSandbox | None:
        """The shared-storage init helper on the collection's provider (one per provider that mounts volumes)."""
        return self._volume_helpers.get(id(self.provider))

    @_volume_helper.setter
    def _volume_helper(self, helper: AsyncSandbox | None) -> None:
        if helper is None:
            self._volume_helpers.pop(id(self.provider), None)
        else:
            self._volume_helpers[id(self.provider)] = helper

    async def _volume_helper_for(self, provider, root: str) -> AsyncSandbox:
        """The init helper on ``provider``'s shared storage, started on first use. A service on its own provider
        mounts that provider's storage, so its directories are made and cleaned there."""
        helper = self._volume_helpers.get(id(provider))
        if helper is None:
            helper = AsyncSandbox(
                provider,
                SandboxSpec(
                    image=self.volume_init_image,
                    ttl_s=self.timeout_s + 3600,
                    entrypoint=["sh", "-c", "while :; do sleep 3600; done"],
                    metadata=provider.shared_volume_metadata(),
                    provider_options=provider.shared_volume_options(None, root),
                ),
                owns_provider=False,
            )
            await helper.start()
            self._volume_helpers[id(provider)] = helper
        return helper

    async def _prepare_volumes(self):
        root = "/tmp/compose-volumes"
        for name, service in self.document["services"].items():
            if not service.get("volumes"):
                continue
            # Everything about a service's volumes happens on the provider that runs it: the directory on its
            # shared storage, the copy-up seed (a sandbox of the service's own spec) and the mount options.
            provider = self._provider_for(name)
            volume_sources = self._volume_sources_for(name)
            metadata = provider.shared_volume_metadata()
            helper = await self._volume_helper_for(provider, root)
            mounts = []
            for volume in service["volumes"]:
                source = volume["source"]
                managed = source not in volume_sources
                relative = volume_sources.get(source, f"{self.project}/{source}")
                if managed:
                    result = await helper.exec(f"mkdir -p {shlex.quote(root + '/' + relative)}", user="root")
                    if result.return_code:
                        raise RuntimeError(f"Shared volume initialization failed: {result.stderr}")
                    if not (volume.get("volume") or {}).get("nocopy"):
                        # The seed reads the service's image, so it keeps the service's provider options, its
                        # pull policy first (``pull_policy_options``, which OpenSandbox reads per spec: a deployment
                        # default of Never would otherwise fail the seed of an uncached ``always`` image before the
                        # service could pull it, and one of Always would pull an image declared ``never``), less
                        # the service's own mounts: it mounts the copy-up storage alone.
                        service_options = self._plans[name]["spec"].provider_options
                        seed_spec = replace(
                            self._plans[name]["spec"],
                            metadata=metadata,
                            provider_options=_merge_provider_options(
                                {key: value for key, value in service_options.items() if key != "volumes"},
                                provider.shared_volume_options(relative, root),
                            ),
                        )
                        seed = AsyncSandbox(provider, seed_spec, owns_provider=False)
                        self._seeds.append(seed)
                        await seed.start()
                        try:
                            target = shlex.quote(volume["target"])
                            result = await seed.exec(
                                f'if test -n "$(ls -A {root})"; then :; '
                                f"elif test -d {target}; then cp -a {target}/. {root}/; "
                                f"elif test -e {target}; then exit 1; fi",
                                user="root",
                                timeout_s=300,
                            )
                            if result.return_code:
                                raise RuntimeError(f"Named volume copy-up failed: {result.stderr}")
                        finally:
                            await seed.stop()
                options = provider.shared_volume_options(
                    relative, volume["target"], read_only=volume.get("read_only", False)
                )
                mounts.extend(options["volumes"])
            if mounts:
                spec = self._plans[name]["spec"]
                self._plans[name]["spec"] = replace(
                    spec,
                    metadata={**metadata, **spec.metadata},
                    provider_options={
                        **spec.provider_options,
                        "volumes": [*spec.provider_options.get("volumes", []), *mounts],
                    },
                )

    def _forwarding_root(self, name: str) -> str:
        """The service whose network namespace ``name`` ends up in: itself when it declares no ``network_mode``,
        else the end of its ``network_mode: service:<target>`` chain. A chain that returns to a service it already
        passed is a cycle, which raises naming it; ``_validate`` runs this before any sandbox exists."""
        services = self.document["services"]
        chain = [name]
        while services[chain[-1]].get("network_mode"):
            target = services[chain[-1]]["network_mode"].removeprefix("service:")
            if target in chain:
                cycle = " -> ".join([*chain[chain.index(target) :], target])
                raise ValueError(f"Service {name!r}: network_mode cycle: {cycle}")
            chain.append(target)
        return chain[-1]

    def _memberships(self) -> dict[str, set[str]]:
        """The networks each service is on. A ``network_mode: service:<target>`` service shares the target's
        network namespace, so it is on the networks of the service at the end of its forwarding chain."""
        services = self.document["services"]
        return {
            name: set(services[self._forwarding_root(name)].get("networks") or {"default": {}}) for name in services
        }

    def _visible_hosts(self, addresses: Mapping[str, str]) -> dict[str, dict[str, str]]:
        """Per service, the hostnames it resolves: itself, every service sharing a network with it (by name
        and by the aliases on those networks) and its own ``extra_hosts``."""
        services = self.document["services"]
        memberships = self._memberships()
        visible: dict[str, dict[str, str]] = {}
        for name in services:
            hosts: dict[str, str] = {}
            for peer, peer_service in services.items():
                shared = memberships[name] & memberships[peer]
                if peer != name and not shared:
                    continue
                hosts[peer] = addresses[peer]
                for network, options in (peer_service.get("networks") or {"default": {}}).items():
                    if network in shared or peer == name:
                        for alias in (options or {}).get("aliases", []):
                            hosts[alias] = addresses[peer]
            hosts.update(services[name].get("extra_hosts") or {})
            visible[name] = hosts
        return visible

    async def _run_hooks(self, name: str) -> None:
        """Run the service's ``post_start`` hooks in order, once its command has started (as Compose does: after
        the container starts, not after it is healthy). A hook that exits non-zero fails the start."""
        plan = self._plans[name]
        for index, hook in enumerate(plan["hooks"]):
            command = hook["command"]
            if isinstance(command, list):
                command = shlex.join(command)
            env = {**plan.get("resolved_env", {}), **hook.get("environment", {})}
            result = await self.services[name].exec(
                command,
                cwd=hook.get("working_dir"),
                env=env or None,
                user=hook.get("user", plan["user"]),
                timeout_s=None,
            )
            if result.return_code:
                detail = (result.stderr or result.stdout or "")[-4096:]
                raise RuntimeError(
                    f"Compose service {name!r} post_start hook {index} exited ({result.return_code}): {detail}"
                )

    async def _exec_privileged(self, sandbox: AsyncSandbox, command: str):
        """Run ``command`` as root, or as the sandbox's default user when it cannot switch users."""
        try:
            result = await sandbox.exec(command, user="root")
        except Exception:
            return await sandbox.exec(command)
        return result if result.return_code == 0 else await sandbox.exec(command)

    async def _upload(self, name: str) -> None:
        """Copy each ``x-sandbox.uploads`` entry (a task file or folder a bind mount named) into the service.

        Runs after the sandbox starts and before the service's command, so the process finds the files
        where the Compose file mounted them. The receiving folders are created first, and the folders
        and files the upload touched (every folder created for it, the folders the files landed in and
        the files; nothing else under the target) are made readable by every user, for an image whose
        process runs as a non-root user. A folder's tree is recreated under the target, empty folders
        included (a program that expects the output or cache folder the task supplies finds it, as under
        a bind mount), over whatever the image already has there (a merge, where a bind mount would hide
        the image's folder). The names the task digest skips (``__pycache__``, ``*.pyc``, editor leftovers:
        :func:`nemo_gym.data.tasks.harbor.digest.is_ignored`) are left out, so what reaches the service is what
        the digest covers, and a link inside the folder is refused, naming it, rather than read through.
        """
        sandbox = self.services[name]
        for entry in (self.document["services"][name].get("x-sandbox") or {}).get("uploads", []):
            source, target = Path(entry["source"]), entry["target"]
            if source.is_dir():
                link = first_link(source)
                if link is not None:
                    raise RuntimeError(f"Service {name!r}: upload source {source} contains a link: {link}")
                remotes: dict[Path, str] = {}
                folders = {target}
                for path in sorted(source.rglob("*")):
                    relative = path.relative_to(source).as_posix()
                    if is_ignored(relative):
                        continue
                    if path.is_dir():
                        folders.add(target + "/" + relative)
                    elif path.is_file():
                        remotes[path] = target + "/" + relative
                        folders.add(remotes[path].rsplit("/", 1)[0])
            else:
                remotes = {source: target}
                folders = {target.rsplit("/", 1)[0] or "/"}
            # Which folders ``mkdir -p`` is about to create, intermediates included: they are created with the
            # root shell's umask, so they are in the chmod below; the folders that already exist above them are not.
            created = await self._missing_folders(sandbox, folders)
            result = await self._exec_privileged(
                sandbox, "mkdir -p " + " ".join(shlex.quote(f) for f in sorted(folders))
            )
            if result.return_code:
                raise RuntimeError(
                    f"Service {name!r}: cannot create {target!r} for upload: {result.stderr or result.stdout}"
                )
            for path, remote in remotes.items():
                await sandbox.upload(path, remote)
            # Never recursive, never ``/``: the target may be a folder the image ships (``/etc``) or hold a
            # shared-storage mount, and what already existed above the created folders is not this upload's.
            touched = [*sorted((folders | created) - {"/"}), *remotes.values()]
            for chunk in range(0, len(touched), 256):
                paths = " ".join(shlex.quote(p) for p in touched[chunk : chunk + 256])
                await self._exec_privileged(sandbox, f"chmod a+rX {paths} 2>/dev/null || true")

    async def _prepare_logs(self, name: str) -> None:
        """Create each ``x-sandbox.logs`` source (a path of Harbor's ``/logs`` tree the service bind-mounted) in
        the service, world-writable, and link a target elsewhere to it.

        Runs after the caller's image setup and before the uploads, the service's command, its ``post_start``
        hooks and its healthcheck, where Harbor's bind sits between the image build and ``docker compose cp``:
        a healthcheck that needs ``/logs/verifier`` passes on a clean image, and an upload under the target lands
        in the source through the link. ``/logs/artifacts:/app/output`` under Harbor is a second mount of one
        host folder that hides whatever the image has at the target, so here the target becomes a link to the
        source unconditionally (``/logs/artifacts`` is where artifacts are collected from): a folder the image
        ships there, or an overlay ``RUN mkdir -p`` created, is removed first, with a warning when it was not
        empty, as a plain folder would keep the task's artifacts out of the collection and show the agent what
        the mount hides.
        """
        sandbox = self.services[name]
        for entry in (self.document["services"][name].get("x-sandbox") or {}).get("logs", []):
            # A target spelled with a trailing separator or dot (`/app/output/`, `/app/output/.`, valid at prepare)
            # names the folder; as the link's name it would make `ln` want an existing directory there, the one
            # just removed, and `rm -rf` refuses a `.` entry outright. The lexical form, the one ``_validate``
            # compared, drops both (M22 review).
            source, target = lexical_target(entry["source"]), lexical_target(entry["target"])
            quoted_source, quoted_target = shlex.quote(source), shlex.quote(target)
            command = f"mkdir -p {quoted_source} && chmod 777 {quoted_source}"
            if target != source:
                parent = target.rsplit("/", 1)[0] or "/"
                # An existing link is replaced by ``ln -sfn``. Anything else at the target goes: ``rmdir`` takes
                # the empty folder quietly, a non-empty one or a file is removed and reported on stdout.
                command += (
                    f" && mkdir -p {shlex.quote(parent)}"
                    f" && if [ -e {quoted_target} ] && [ ! -L {quoted_target} ]; then"
                    f" rmdir {quoted_target} 2>/dev/null || {{ rm -rf {quoted_target} && echo {_HIDDEN_MARK}; }}; fi"
                    f" && ln -sfn {quoted_source} {quoted_target}"
                )
            result = await self._exec_privileged(sandbox, command)
            if result.return_code:
                raise RuntimeError(
                    f"Service {name!r}: cannot create {source!r} for {target!r}: "
                    f"{result.stderr or result.stdout or f'exit status {result.return_code}'}"
                )
            # A provider may report a command that printed nothing as ``stdout=None`` (OpenSandbox does).
            if _HIDDEN_MARK in (result.stdout or "").split():
                LOGGER.warning(
                    "Service %r: %s was not empty before it became a link to %s; its contents are gone, as a bind "
                    "mount of %s there would hide them",
                    name,
                    target,
                    source,
                    source,
                )

    async def _missing_folders(self, sandbox: AsyncSandbox, folders: set[str]) -> set[str]:
        """The paths ``mkdir -p`` would create for ``folders``: each one and its missing ancestors, below ``/``.

        One probe lists every ancestor that does not exist yet; only paths from the candidate set are taken
        from its output, so a name the shell printed oddly is dropped rather than chmod-ed. A probe that
        fails reports nothing created: the chmod is best effort, the ``mkdir -p`` that follows is not.
        """
        candidates: set[str] = set()
        for folder in folders:
            parts = [part for part in folder.split("/") if part]
            candidates.update("/" + "/".join(parts[:depth]) for depth in range(1, len(parts) + 1))
        if not candidates:
            return set()
        probe = " ".join(shlex.quote(path) for path in sorted(candidates))
        try:
            result = await self._exec_privileged(
                sandbox, f'for p in {probe}; do test -e "$p" || printf "%s\\n" "$p"; done'
            )
        except Exception:
            return set()
        if result.return_code:
            return set()
        return {line for line in (result.stdout or "").splitlines() if line in candidates}

    async def _wait(self, name, condition):
        if condition == "service_started":
            return
        plan = self._plans[name]
        health = plan["health"]
        started = plan.get("started_at", asyncio.get_running_loop().time())
        failures = 0
        while True:
            done = await self.services[name].exec(
                f"if test -f {plan['runtime']}/exit; then cat {plan['runtime']}/exit; fi", user=None
            )
            if done.return_code:
                raise RuntimeError(f"Cannot read service {name!r} exit status: {done.stderr}")
            if done.stdout and done.stdout.strip():
                if condition == "service_completed_successfully" and done.stdout.strip() == "0":
                    return
                raise RuntimeError(
                    f"Compose service {name!r} exited ({done.stdout.strip()}); {await self._command_log_note(name)}"
                )
            if condition == "service_completed_successfully":
                await asyncio.sleep(self.poll_interval_s)
                continue
            test = health["test"]
            command = shlex.join(test[1:]) if test[0] == "CMD" else shlex.join([*plan["shell"], test[1]])
            try:
                result = await self.services[name].exec(
                    command,
                    timeout_s=_seconds(health.get("timeout", "30s")),
                    user=plan["user"],
                    env=plan.get("resolved_env"),
                )
                healthy = result.return_code == 0
                detail = result.stderr or result.stdout
            except TimeoutError:
                healthy, detail = False, "healthcheck timed out"
            if healthy:
                return
            elapsed = asyncio.get_running_loop().time() - started
            in_start_period = elapsed < _seconds(health.get("start_period", 0))
            if not in_start_period:
                failures += 1
                if failures >= health.get("retries", 3):
                    # The healthcheck's own output rarely says why (a connection refused is "exit status 1"); the
                    # service's command log, which the run script redirects stdout and stderr to, usually does. A
                    # bounded tail of it travels with the failure, as it does for a service that exited.
                    raise RuntimeError(
                        f"Compose service {name!r} unhealthy: {detail}; {await self._command_log_note(name)}"
                    )
            interval = health.get("start_interval", "5s") if in_start_period else health.get("interval", "30s")
            await asyncio.sleep(_seconds(interval))

    async def _create_workdir(self, name: str) -> None:
        """Create the service's working directory once its sandbox is up, as Docker creates a missing
        ``working_dir`` (or image ``WORKDIR``) at container create.

        Every exec in the service defaults to that directory, this group's bootstrap ones included, and a
        provider may refuse a command whose working directory does not exist (OpenSandbox answers 400): a
        ``main`` started from a base image whose Dockerfile only names a ``WORKDIR`` has nothing else to create
        it. Root from ``/``, else the default user; the caller owns it afterwards as it owns a plain sandbox's.
        """
        workdir = self._plans[name]["spec"].workdir
        if not workdir or workdir == "/":
            return
        command = f"mkdir -p {shlex.quote(workdir)}"
        sandbox = self.services[name]
        try:
            result = await sandbox.exec(command, cwd="/", user="root")
        except Exception:
            result = None
        if result is None or result.return_code:
            result = await sandbox.exec(command, cwd="/")
        if result.return_code:
            raise RuntimeError(
                f"Service {name!r}: cannot create its working directory {workdir!r}: "
                f"{result.stderr or result.stdout or f'exit status {result.return_code}'}"
            )

    async def _command_log_note(self, name: str) -> str:
        """What a failed service's command log says, for the failure's message: a bounded tail of the log the run
        script redirects the command's stdout and stderr to, a note that it wrote nothing, or why the log could not
        be read. The read is a second exec after the verdict that names the service, and the service's sandbox may
        be gone by then (a provider that ends an OOM-killed pod between the final healthcheck and this read), so
        anything it raises goes into the message rather than replacing the verdict."""
        try:
            log = await self.services[name].exec(f"tail -c 4096 {self._plans[name]['runtime']}/log", timeout_s=30)
        except Exception as exc:
            return f"its command log could not be read ({type(exc).__name__}: {exc})"
        output = (log.stdout or "").strip() if log.return_code == 0 else ""
        return f"last output of its command: {output}" if output else "its command wrote no output"

    async def start(self):
        if self._started or self._closed:
            raise RuntimeError("Compose collection already started or closed")
        if self.compose_file is None:
            raise ValueError("Starting a Compose collection requires a YAML file")
        self._started = True
        try:
            async with asyncio.timeout(self.timeout_s):
                self.document = self._load()
                order = self._validate()
                await self._prepare()
                if any(service.get("volumes") for service in self.document["services"].values()):
                    await self._prepare_volumes()
                for name in order:
                    provider = self._provider_for(name)
                    self.services[name] = AsyncSandbox(provider, self._plans[name]["spec"], owns_provider=False)
                    await self.services[name].start()
                    await self._create_workdir(name)
                    service = self.document["services"][name]
                    if service.get("cap_add") or service.get("shm_size") is not None:
                        await provider.configure_runtime(
                            self.services[name]._require_handle(),
                            cap_add=tuple(service.get("cap_add") or ()),
                            shm_size=service.get("shm_size"),
                        )
                    # The caller's image setup (an overlay Dockerfile's RUN lines) comes first, as a build precedes
                    # `compose up`: then the mounts (the logs tree, then the uploads, as Harbor copies task files
                    # into the mounted container), then, below, the command, hooks and healthcheck.
                    if name in self.service_setup:
                        await self.service_setup[name](self.services[name])
                    if (service.get("x-sandbox") or {}).get("logs"):
                        await self._prepare_logs(name)
                    if (service.get("x-sandbox") or {}).get("uploads"):
                        await self._upload(name)
                hosts: dict[str, str] = {}
                if self.services:
                    addresses: dict[str, str] = {}
                    for name, sandbox in self.services.items():
                        addresses[name] = await self._provider_for(name).network_address(sandbox._require_handle())
                    visible = self._visible_hosts(addresses)
                    hosts = {name: addresses[name] for name in self.services}
                    for name, sandbox in self.services.items():
                        options = self.document["services"][name].get("x-sandbox") or {}
                        plan = self._plans[name]
                        plan["resolved_env"] = {}
                        for key in options.get("resolve_environment", []):
                            url = urlsplit(plan["spec"].env[key])
                            if url.hostname not in visible[name]:
                                raise ValueError(
                                    f"Service {name!r}: environment {key!r} names {url.hostname!r}, which shares no "
                                    "network with it"
                                )
                            address = visible[name][url.hostname]
                            authority = f"[{address}]" if ":" in address else address
                            if url.port is not None:
                                authority += f":{url.port}"
                            if "@" in url.netloc:
                                authority = url.netloc.rsplit("@", 1)[0] + "@" + authority
                            plan["resolved_env"][key] = url._replace(netloc=authority).geturl()
                        if options.get("hosts") != []:
                            await self._provider_for(name).set_hosts(sandbox._require_handle(), visible[name])
                for name in order:
                    service = self.document["services"][name]
                    for dep, options in (service.get("depends_on") or {}).items():
                        await self._wait(dep, options.get("condition", "service_started"))
                    plan = self._plans[name]
                    runtime = plan["runtime"]
                    result = await self.services[name].exec(f"mkdir -p {runtime} && chmod 777 {runtime}")
                    if result.return_code:
                        raise RuntimeError(f"Cannot initialize service {name!r}: {result.stderr or result.stdout}")
                    if service.get("network_mode"):
                        target = service["network_mode"].removeprefix("service:")
                        ports = self._plans[target]["spec"].ports
                        ready = runtime + "/forwarding-ready"
                        # The relay runs inside the forwarder's sandbox, so on the provider that owns it.
                        relay = asyncio.create_task(
                            self._provider_for(name).forward_ports(
                                self.services[name]._require_handle(), hosts[target], ports, ready_file=ready
                            )
                        )
                        self._processes[name + ":forwarding"] = relay
                        while (await self.services[name].exec(f"test -f {ready}")).return_code:
                            if relay.done():
                                await relay
                                raise RuntimeError(f"TCP forwarding for {name!r} exited before becoming ready")
                            await asyncio.sleep(self.poll_interval_s)
                    script = f"touch {runtime}/started\n{plan['command']}\nstatus=$?\nprintf '%s' \"$status\" > {runtime}/exit\n"
                    with tempfile.TemporaryDirectory() as temp:
                        path = Path(temp) / "run.sh"
                        path.write_text(script)
                        await self.services[name].upload(path, runtime + "/run.sh")
                    # Keep the exec request alive: some providers reap background
                    # children as soon as their launching shell exits.
                    process = asyncio.create_task(
                        self.services[name].exec(
                            f"sh {runtime}/run.sh >{runtime}/log 2>&1 </dev/null",
                            user=plan["user"],
                            env=plan.get("resolved_env"),
                            timeout_s=None,
                        )
                    )
                    self._processes[name] = process
                    while True:
                        finished = process.done()
                        result = await self.services[name].exec(f"test -f {runtime}/started")
                        if result.return_code == 0:
                            break
                        if finished:
                            outcome = process.result()
                            raise RuntimeError(f"Compose service {name!r} did not start: {outcome.stderr}")
                        await asyncio.sleep(self.poll_interval_s)
                    plan["started_at"] = asyncio.get_running_loop().time()
                    await self._run_hooks(name)
                for name in order:
                    if self._plans[name]["health"]["test"][0] != "NONE":
                        await self._wait(name, "service_healthy")
                for name, process in self._processes.items():
                    if name.endswith(":forwarding") and process.done():
                        await process
                        raise RuntimeError(f"TCP forwarding for {name!r} stopped during startup")
        except BaseException:
            await self.stop()
            raise
        self._ready = True
        return self

    async def serialize(self, *, scope: str | None = None) -> dict[str, Any]:
        """Describe a running collection using provider connection descriptors.

        The creating process must keep the collection alive: it owns service
        and forwarding tasks and managed-volume cleanup. Provider configuration
        and YAML are not included.
        """
        if not self._ready or self._closed:
            raise RuntimeError("Serializing requires a running Compose collection")
        return {
            "services": {name: await sandbox.serialize(scope=scope) for name, sandbox in self.services.items()},
        }

    @classmethod
    async def connect(
        cls, descriptor: Mapping[str, Any], *, provider, service_providers: Mapping[str, Any] | None = None
    ) -> "AsyncSandboxCompose":
        """Connect to all members without provisioning or restarting services.

        Like AsyncSandbox.connect, stop() closes the connected sandboxes using
        provider semantics. The creator retains ownership of managed-volume
        cleanup and running service/forwarding tasks and must also call stop().
        ``service_providers`` names the provider of members that live on another
        one, as it did at start.
        """
        if not isinstance(descriptor, Mapping) or set(descriptor) != {"services"}:
            raise ValueError("Invalid Compose connection descriptor")
        services = descriptor["services"]
        if (
            not isinstance(services, Mapping)
            or not services
            or any(not isinstance(name, str) or not isinstance(value, Mapping) for name, value in services.items())
        ):
            raise ValueError("Invalid Compose connection members")
        collection = cls(provider, None, service_providers=service_providers)
        try:
            for name, member in services.items():
                collection.services[name] = await AsyncSandbox.connect(
                    member, provider=collection._provider_for(name), owns_provider=False
                )
        except BaseException:
            # A partial attach must not destroy a collection owned by another server.
            for member_provider in collection._providers():
                await member_provider.aclose()
            raise
        collection._started = collection._ready = True
        return collection

    async def stop(self):
        if self._closed:
            return
        if self._stop_task is None or self._stop_task.done():
            self._stop_task = asyncio.create_task(self._stop())
        try:
            await asyncio.shield(self._stop_task)
        except asyncio.CancelledError:
            await self._stop_task
            raise

    async def _stop(self):
        for process in self._processes.values():
            process.cancel()
        await asyncio.gather(*self._processes.values(), return_exceptions=True)
        errors = []
        for sandbox in [*reversed(list(self.services.values())), *self._seeds]:
            try:
                await sandbox.stop()
            except Exception as error:
                errors.append(error)
        for key, helper in list(self._volume_helpers.items()):
            try:
                result = await helper.exec(f"rm -rf /tmp/compose-volumes/{self.project}", user="root")
                if result.return_code:
                    raise RuntimeError(f"Shared volume cleanup failed: {result.stderr}")
            except Exception as error:
                errors.append(error)
            try:
                await helper.stop()
                del self._volume_helpers[key]
            except Exception as error:
                errors.append(error)
        for provider in self._providers():
            try:
                await provider.aclose()
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Compose cleanup failed", errors)
        self._closed = True

    async def __aenter__(self):
        if self._ready and not self._closed:
            return self
        return await self.start()

    async def __aexit__(self, *exc):
        await self.stop()
