# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a Compose service may say, and the shape each field takes before the adapter reads it.

The field allow-list of the sandbox Compose adapter and its message per offending key, the
normalisation of deploy resources, devices and gpus, restart, pull_policy, platform, extra_hosts,
hooks, networks and the provider service options, and check_compose_fields, which runs them all
over a document and is a fixed point of its own output. The two Compose scalar and environment
shape helpers, the memory size parser and the interpolation error class live here because the
field checks, the resolver and the interpolation code in nemo_gym.data.tasks.harbor.compose_variables
all read them, and this module sits below both. Nothing here reads a task or a record.
"""

import ipaddress
import re
from collections.abc import Mapping
from typing import Any


def _bytes(value: str | int) -> int:
    if isinstance(value, int):
        return value
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmgt]?)(?:i?b)?", value.lower())
    if match is None:
        raise ValueError(f"Invalid Compose memory size: {value!r}")
    return int(float(match[1]) * 1024 ** " kmgt".index(match[2] or " "))


class ComposeSubstitutionError(ValueError):
    """A Compose file names a variable or an ``env_file`` that prepare cannot resolve."""

    def __init__(self, name: str, message: str | None = None) -> None:
        self.name = name
        super().__init__(message or f"unresolved Compose variable ${{{name}}}")


def _compose_scalar(value: Any) -> str:
    """An environment value as Compose would emit it: booleans lower-case, ``None`` empty."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _environment_mapping(service_name: str, environment: Any, variables: dict[str, str]) -> dict[str, str | None]:
    """A service's ``environment`` as a mapping. A bare ``NAME`` in the list form, or a null value in the mapping
    form, stands for the task variable of that name and is ``None`` when there is none: Compose then leaves the
    variable unset in the container, even where an ``env_file`` sets it. ``NAME=`` and ``NAME: ''`` are empty."""
    if environment is None:
        return {}
    if isinstance(environment, list):
        pairs: dict[str, str | None] = {}
        for item in environment:
            key, separator, value = str(item).partition("=")
            pairs[key] = value if separator else variables.get(key)
        return pairs
    if isinstance(environment, dict):
        return {
            str(key): variables.get(str(key)) if value is None else _compose_scalar(value)
            for key, value in environment.items()
        }
    raise ComposeSubstitutionError("environment", f"service {service_name!r} environment must be a list or a mapping")


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


def unsupported_compose_fields(document: Any) -> list[str]:
    """The refusals the sandbox Compose adapter raises for the keys of ``document``: one message per offender in
    document order (the top level first, then each service), empty when every key is one the adapter starts.

    The key allow-list is the first step of :func:`check_compose_fields`, the one field check prepare and the
    adapter run (a refused key is an :class:`UnsupportedComposeFieldError` there). Pure and key-only, so prepare
    also screens the raw file of a task about to be built with it (``nemo_gym.data.tasks.harbor.prepare``), before
    interpolation gives the values their shapes.
    """
    return [message for _, message in _unsupported_fields(document)]


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
        # A float, a list or a mapping would escape ``_bytes`` as an ``AttributeError``, past the per-task error
        # boundary of prepare; a malformed value of this field is the field's own error like any other.
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise TypeError
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
        # Interpolation hands prepare every scalar it touched as a string (``count: "${GPUS:-1}"`` resolves to
        # ``"1"``; the type restoration covers healthchecks alone), so a decimal string means its integer. The
        # normalized ``reservations.gpus`` is this function's own output and stays an integer.
        if isinstance(count, str) and re.fullmatch(r"[+-]?[0-9]+", count.strip()):
            count = int(count)
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
        resources["memory_mib"] = (_bytes(memory) + 1048575) // 1048576
    if reservations.get("gpus"):
        resources["gpu"] = int(reservations["gpus"])
    return resources


def service_gpus(service: dict) -> int:
    """GPUs a service requests through ``deploy.resources.reservations.devices`` (0 when none).

    Read from the normalized ``reservations.gpus`` alone: the field check calls this on every sidecar before
    :func:`resolve_compose` has converted the legacy ``mem_limit``/``shm_size`` strings, so a CPU-only service
    with ``mem_limit: 512M`` must not be sized here.
    """
    reservations = ((service.get("deploy") or {}).get("resources") or {}).get("reservations") or {}
    return int(reservations.get("gpus") or 0)


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
    """``post_start`` entries as ``command`` (a string, run through the shell, or an argv list), optional ``user``
    (a name, or a numeric id as an integer), ``working_dir`` and ``environment`` (a mapping). ``privileged`` raises:
    no hook escalates."""
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
        user = hook.get("user")
        if user not in (None, ""):
            # A decimal id is an integer, as the adapter makes the service's own ``user``: a provider that switches
            # users by id (OpenSandbox runs an int as ``uid``, a string through ``su <name>``) then runs the hook as
            # uid 0, which an image without an account named ``0`` could not. ``user:group`` has no sandbox form.
            if isinstance(user, bool) or not isinstance(user, int):
                user = str(user).strip()
                user = int(user) if user.isdecimal() else user
            if isinstance(user, str) and ":" in user:
                raise ComposeFieldError(
                    "post_start.user", f"{where}[{index}].user: user:group overrides are not supported"
                )
            entry["user"] = user
        if hook.get("working_dir"):
            entry["working_dir"] = str(hook["working_dir"])
        if hook.get("environment") is not None:
            try:
                entry["environment"] = _environment_mapping(service_name, hook["environment"], {})
            except ComposeSubstitutionError:
                raise ComposeFieldError(
                    "post_start.environment", f"{where}[{index}].environment must be a list or a mapping"
                ) from None
        out.append(entry)
    return out


_PORT_KEYS = frozenset({"target", "published", "host_ip", "protocol", "mode"})


def normalize_ports(service_name: str, value: Any) -> list[dict[str, Any]]:
    """A service's ``ports`` in long syntax, each entry checked the way the sandbox group runs it.

    A short entry (``"8080"``, ``"8080:80"``, ``"127.0.0.1:8080:80/udp"``, a bare integer) takes the mapping Compose
    gives it (``target``, and ``published``, ``host_ip``, ``protocol`` when written), so prepare and the adapter
    read one shape. The group exposes a service's ports as endpoints the provider allocates, so a fixed
    ``published`` port (anything but none or ``0``), a ``host_ip``, a protocol other than TCP, a ``mode`` other than
    ``ingress``, a port range and an unknown option raise :class:`ComposeFieldError` naming the service: the
    refusals the adapter makes at seed, made at prepare, for the short and the long syntax alike.
    """
    where = f"service {service_name!r} ports"
    if value is None:
        return []
    if not isinstance(value, list):
        raise ComposeFieldError("ports", f"{where} must be a list")
    out: list[dict[str, Any]] = []
    for entry in value:
        if isinstance(entry, dict):
            unknown = sorted(str(key) for key in set(entry) - _PORT_KEYS)
            if unknown:
                raise ComposeFieldError("ports", f"{where}: unsupported option {unknown[0]!r}")
            port = dict(entry)
        elif isinstance(entry, bool) or not isinstance(entry, (str, int)):
            raise ComposeFieldError("ports", f"{where}: entry {entry!r} must be a string, a number or a mapping")
        else:
            text = str(entry).strip()
            if "/" in text:
                text, protocol = text.rsplit("/", 1)
            else:
                protocol = None
            parts = text.split(":")
            if len(parts) > 3 or not all(parts):
                raise ComposeFieldError("ports", f"{where}: cannot read {entry!r}")
            port = {"target": parts[-1]}
            if len(parts) >= 2:
                port["published"] = parts[-2]
            if len(parts) == 3:
                port["host_ip"] = parts[0]
            if protocol is not None:
                port["protocol"] = protocol
        target = port.get("target")
        if isinstance(target, bool) or not isinstance(target, (str, int)) or "-" in str(target):
            raise ComposeFieldError("ports", f"{where}: target {target!r} must be one port, not a range")
        try:
            port["target"] = int(str(target))
        except ValueError:
            raise ComposeFieldError("ports", f"{where}: target {target!r} must be a port number") from None
        # The sandbox spec's contract (``SandboxSpec.__post_init__``: 1 through 65535), applied here so a target it
        # would refuse at every seed is this task's reason at prepare. The container target only: ``published: 0``
        # stays the request for an allocated endpoint.
        if not 1 <= port["target"] <= 65535:
            raise ComposeFieldError("ports", f"{where}: target must be between 1 and 65535, got {port['target']}")
        published = port.get("published")
        if published not in (None, 0, "0", ""):
            raise ComposeFieldError(
                "ports",
                f"{where}: fixed published ports cannot be honoured by sandbox endpoints (published {published!r}); "
                "publish none, the group allocates one, or expose the port",
            )
        if published in ("", None):
            port.pop("published", None)
        if port.get("host_ip"):
            raise ComposeFieldError("ports", f"{where}: host_ip cannot be honoured by sandbox endpoints")
        port.pop("host_ip", None)
        if str(port.get("protocol", "tcp")).lower() != "tcp":
            raise ComposeFieldError("ports", f"{where}: only TCP ports are supported, not {port['protocol']!r}")
        if "protocol" in port:
            # Canonical, as the adapter compares it (``"TCP"`` and ``"Tcp"`` passed here and were refused at seed).
            port["protocol"] = "tcp"
        if port.get("mode", "ingress") != "ingress":
            raise ComposeFieldError("ports", f"{where}: only ingress mode is supported, not {port['mode']!r}")
        out.append(port)
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
    services sharing a network reach each other by name, as on any Compose network. ``internal: true``, ``ipam``
    with a configuration, ``driver_opts`` and ``enable_ipv6: true`` ask for isolation or addressing the pod does
    not provide and raise :class:`ComposeFieldError` naming them. Compose's defaults spelled out (``internal:
    false``, ``enable_ipv6: false``, an empty ``ipam`` block) ask for nothing and are dropped.
    """
    declared = document.get("networks")
    if declared is None:
        return {}
    if not isinstance(declared, dict):
        raise ComposeFieldError("networks", "top-level networks must be a mapping")
    out: dict[str, dict[str, Any]] = {}
    for name, options in declared.items():
        if options is not None and not isinstance(options, dict):
            raise ComposeFieldError(f"networks.{name}", f"network {name!r} must be a mapping, got {options!r}")
        options = dict(options or {})
        # A false value is Compose's default written out (every kumo task declares ``internal: false``), as an
        # empty ``ipam`` block is: neither asks for anything, so neither names what the pod lacks.
        for option in ("ipam", "internal", "enable_ipv6"):
            if option in options and not options[option]:
                options.pop(option)
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
        if name != "default" and name not in declared:
            raise ComposeFieldError("networks", f"{where}: {name!r} is not declared under top-level networks")
        if options is not None and not isinstance(options, dict):
            raise ComposeFieldError("networks", f"{where}.{name} must be a mapping, got {options!r}")
        options = dict(options or {})
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

    The one field check: prepare runs it per task (``nemo_gym.data.tasks.harbor.image_configs.check_task_compose_fields``)
    and the adapter again at start. A key outside the allow-list (:func:`unsupported_compose_fields`: a sidecar
    ``build:``, ``privileged``, ``logging``, ...) is an :class:`UnsupportedComposeFieldError` before any value is
    read. Mutates and returns ``document``, and applied to its own output returns it unchanged: the server normalizes
    once and the adapter again, from the written file. ``deploy`` is normalized (see :func:`normalize_deploy`);
    ``restart`` is validated and kept (recorded by the adapter, the group restarts nothing); ``pull_policy``
    becomes ``always``, ``missing`` or ``never``; ``platform`` must match ``platform``, the one the deployment
    runs (see :func:`deployment_platform`); ``networks`` (top level and per service) and ``extra_hosts`` take
    their mapping forms; ``post_start`` hooks are checked; ``ports`` take their long syntax, with the group's
    endpoint rules applied (see :func:`normalize_ports`); ``tmpfs``, ``ulimits``, ``sysctls`` and
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
        if "ports" in service:
            service["ports"] = normalize_ports(name, service["ports"])
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
