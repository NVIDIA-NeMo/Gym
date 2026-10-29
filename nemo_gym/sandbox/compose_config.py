# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a published Harbor Compose overlay into a document the sandbox Compose adapter can start.

Harbor publishes ``environment/docker-compose.yaml`` as an overlay on its own base file: the
``main`` service is the task image, sidecars are prebuilt images. Starting it without Docker
needs each image's recorded OCI configuration (entrypoint, command, user, exposed ports,
healthcheck), which ``image_configs`` supplies, keyed by image reference.

Adapted from Gym's Terminal Bench 4 server (``resources_servers/terminal_bench_4/compose_config.py``).
"""

import hashlib
import logging
import os
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from nemo_gym.data.tasks.harbor.digest import is_covered
from nemo_gym.sandbox.compose_fields import ComposeFieldError, _bytes, normalize_platform, service_gpus
from nemo_gym.sandbox.compose_volumes import (
    LOGS_TREE,
    ComposeVolumeError,
    _bind_source,
    absolute_link,
    first_link,
    lexical_target,
    parse_volume,
)


log = logging.getLogger(__name__)


def plan_compose_volumes(
    document: dict,
    base_dir: Path,
    *,
    root: Path | None = None,
    volume_sources: dict[str, str] | None = None,
    gpu_volume_sources: dict[str, str] | None = None,
    shared_storage: bool | None = None,
    gpu_shared_storage: bool | None = None,
    warn_read_only: bool = False,
) -> dict[str, list[dict[str, str]]]:
    """Decide how each service's ``volumes`` run in a sandbox group. Mutates and returns per-service uploads.

    A bind mount whose source is a file or folder of the task (under ``root``, default ``base_dir``;
    relative paths resolve against ``base_dir``, as Compose resolves them against the Compose file)
    becomes an upload into the service once it starts: the entry leaves ``volumes`` and joins
    ``x-sandbox.uploads`` as ``{"source": <host path>, "target": <container path>}``. With a task ``root``
    given, such a source must be one the task digest covers (:func:`~nemo_gym.data.tasks.harbor.digest.is_covered`:
    under ``environment/``, ``tests/``, ``solution/`` or ``steps/``, or one of the top-level task files, and
    not a name the digest skips), or it raises naming the path: an upload the digest does not see could change
    without changing the task's identity. Named volumes declared at top level stay, in long syntax, for the provider's shared
    storage; a volume declared ``external: true``, and a bind source outside the task, resolve through
    ``volume_sources`` (the ``sandbox`` block's mapping of volume name or host path to a shared-storage path)
    and raise :class:`ComposeVolumeError` naming the volume when it is missing. A sidecar that reserves GPUs
    through ``deploy.resources`` resolves through ``gpu_volume_sources`` instead when that is given (the
    ``gpu_sandbox_provider`` block's mapping, the block the server runs it on; ``main`` follows the task, as
    :func:`check_compose_fields` routes the capability check). A bind of Harbor's ``/logs`` tree onto ``main``
    leaves ``volumes`` and joins ``x-sandbox.logs`` as ``{"source": <logs path>, "target": <container path>}``:
    the adapter creates the source in the agent's sandbox before the service's command, hooks and healthcheck,
    and links a target elsewhere to it, as Docker creates both of Harbor's bind targets at container create.
    A sidecar that binds a path of the tree shares it with ``main`` the way Harbor's one host folder is shared:
    the path becomes a named volume on the provider's shared storage (declared as ``nemo-gym-<path>``, with
    ``nocopy``; a second path that spells the same name, ``/logs/a/b`` beside ``/logs/a-b``, gets a digest of its
    spelling appended), mounted at the sidecar's target and at the path itself in ``main``, so what the sidecar
    writes there is what the verifier collects. A logs path with a ``..`` segment (``/logs/../etc``) raises naming the
    path, on ``main`` or a sidecar: it is used as written. ``shared_storage`` says whether the deployment's provider has shared
    storage (``gpu_shared_storage`` for the services ``gpu_volume_sources`` routes): ``False`` refuses every
    named volume, mapped source and shared logs bind at prepare, naming the setting, instead of at every seed;
    ``None`` leaves that to the adapter. A source the task does not contain, or a symlink that leaves it, raises naming the
    path; so does a bound folder that contains a link, and a source reached through a link with an absolute
    target (see :func:`absolute_link`). A service that already carries ``x-sandbox`` raises naming the field:
    that key is this planner's output, not task input. A null top-level declaration (``volumes: {pgdata:
    null}``) is normalised to ``{}``, the form the adapter reads.
    """
    base = Path(base_dir).resolve()
    task_root = Path(root).resolve() if root is not None else None
    root = task_root if task_root is not None else base
    base_sources = dict(volume_sources or {})
    services = document.get("services") if isinstance(document, dict) else None
    if not isinstance(services, dict):
        return {}
    declared = document.get("volumes")
    if declared is None:
        declared = {}
    if not isinstance(declared, dict):
        raise ComposeVolumeError("volumes", "top-level volumes must be a mapping")
    # ``x-sandbox`` is what this planner (and the sidecar adaptation) write for the Compose adapter, never task
    # input: an entry in the task file could name any host file as an upload source. Refused before anything
    # else, so prepare fails with the field name.
    for name, service in services.items():
        if isinstance(service, dict) and "x-sandbox" in service:
            raise ComposeVolumeError(
                "x-sandbox", f"service {name!r} declares x-sandbox; it is written by Gym at prepare, not by the task"
            )
    # ``volumes:\n  pgdata:`` (a null declaration) is the canonical empty one; the adapter reads a mapping. Any
    # other non-mapping (``pgdata: 123``) is refused here, naming the volume, whether or not a service mounts it
    # (Compose rejects the file either way), as this planner's own error: prepare reports the task and goes on.
    for volume_name, declaration in declared.items():
        if declaration is None:
            declared[volume_name] = {}
        elif not isinstance(declaration, dict):
            raise ComposeVolumeError(
                volume_name, f"volume {volume_name!r}: declaration must be a mapping, got {declaration!r}"
            )
    uploads: dict[str, list[dict[str, str]]] = {}
    logs: dict[str, list[dict[str, str]]] = {}
    # Paths of the logs tree a sidecar binds, with every service that binds each: shared through one volume below.
    shared_logs: dict[str, list[tuple[str, str, bool]]] = {}
    # Whether the block each service runs on has shared storage, and that block's name for the refusal.
    storage_of: dict[str, tuple[bool | None, str]] = {}
    for name, service in services.items():
        if not isinstance(service, dict) or "volumes" not in service:
            continue
        entries = service.pop("volumes")
        if entries is None:
            continue
        if not isinstance(entries, list):
            raise ComposeVolumeError("volumes", f"service {name!r} volumes must be a list")
        # The mapping of the block this service runs on: a GPU-reserving sidecar lives on the GPU block.
        sources, storage, block = base_sources, shared_storage, _TASK_BLOCK
        if (gpu_volume_sources, gpu_shared_storage) != (None, None) and name != "main" and service_gpus(service):
            sources, storage, block = dict(gpu_volume_sources or {}), gpu_shared_storage, _GPU_BLOCK
        storage_of[name] = (storage, block)
        kept: list[dict[str, Any]] = []
        for entry in entries:
            volume = parse_volume(entry, name)
            source = volume["source"]
            if volume["type"] == "volume":
                if source not in declared:
                    raise ComposeVolumeError(source, f"service {name!r} mounts undeclared volume {source!r}")
                declaration = declared[source]
                unsupported = set(declaration) - {"name", "external"}
                if unsupported:
                    field = sorted(unsupported)[0]
                    raise ComposeVolumeError(field, f"volume {source!r}: unsupported option {field!r}")
                if declaration.get("external") and source not in sources:
                    raise ComposeVolumeError(
                        source, f"external volume {source!r} needs a `volume_sources` entry in the sandbox block"
                    )
                kept.append(volume)
                continue
            host = _bind_source(source, base, f"service {name!r} volumes")
            if str(host) == LOGS_TREE or str(host).startswith(LOGS_TREE + "/"):
                if ".." in host.parts:
                    # The path is used as written: the adapter's ``mkdir -p`` in ``main``, and for a sidecar the name
                    # and ``main``'s target of the shared volume, which the provider refuses at every seed
                    # (``Shared storage target must be an absolute container path without traversal``).
                    raise ComposeVolumeError(
                        source,
                        f"service {name!r} binds {source!r}, a logs path with a `..` segment; write the path it means",
                    )
                if name == "main":
                    # The adapter creates the logs path in the agent's sandbox before the command, and links a
                    # target elsewhere (``/logs/artifacts:/app/output``) to it. A target inside the path, or the
                    # path inside the target (``/logs:/logs/child``, ``/logs/verifier:/logs``), would delete the
                    # path or link it to itself: the adapter refuses that document at every seed, so prepare
                    # refuses the task here, naming the field, as the README promises. The target is compared in
                    # the lexical form the adapter links (`/logs/.` is `/logs`) and recorded as written.
                    source_path, target_path = str(host).rstrip("/") + "/", lexical_target(volume["target"]) + "/"
                    if source_path != target_path and (
                        source_path.startswith(target_path) or target_path.startswith(source_path)
                    ):
                        raise ComposeVolumeError(
                            "target",
                            f"service {name!r} binds the logs path {str(host)!r} at {volume['target']!r}, which nests "
                            "it; the target becomes a link to the path, so bind it beside the path or at itself",
                        )
                    logs.setdefault(name, []).append({"source": str(host), "target": volume["target"]})
                    continue
                # Under Harbor the sidecar and ``main`` bind one host folder (kumo's verifier sidecar appends to
                # ``/logs/verifier/actions.jsonl``, which ``main``'s test reads): shared through the provider's
                # storage, planned once every service's binds are known.
                shared_logs.setdefault(str(host), []).append((name, volume["target"], volume["read_only"]))
                continue
            resolved = host.resolve()
            if resolved.is_relative_to(root):
                if not host.exists():
                    raise ComposeVolumeError(
                        source, f"service {name!r} binds {source!r}, which the task folder does not contain"
                    )
                link = absolute_link(host, root)
                if link is not None:
                    # Fine here, but on the server's copy of the task the link still points at this folder.
                    raise ComposeVolumeError(
                        source,
                        f"service {name!r} binds {source!r} through a link with an absolute target: {link}; "
                        "make the link relative",
                    )
                link = first_link(resolved) if resolved.is_dir() else None
                if link is not None:
                    # The folder is uploaded entry by entry (a copy, not a mount); a link would be read through.
                    raise ComposeVolumeError(
                        source,
                        f"service {name!r} binds {source!r}, which contains a link: {link}; links are not uploaded",
                    )
                if task_root is not None and not is_covered(resolved.relative_to(task_root).as_posix()):
                    # The server uploads the bytes of its task copy, which it checks by the task digest alone: a
                    # source the digest does not cover could change without changing the task's identity.
                    raise ComposeVolumeError(
                        source,
                        f"service {name!r} binds {source!r}, which the task digest does not cover; keep upload "
                        "sources under environment/, tests/, solution/ or steps/ and free of names the digest skips",
                    )
                if volume["read_only"] and warn_read_only:
                    # The upload is a writable copy, where Harbor's `:ro` bind is a mount the kernel keeps read-only:
                    # an agent that can write it can change reference data before the verifier reads it, in either
                    # verifier mode. Said by prepare alone (``warn_read_only``): the server plans the same document
                    # again at every seed and stays quiet. A real read-only mount is tracked in issue #5062.
                    log.warning(
                        "service %r binds %s read-only (:ro) at %s, which the sandbox uploads as a writable copy: "
                        "`ro` is not enforced, so the agent can change it before the verifier reads it; keep data "
                        "test.sh must trust under tests/, which is uploaded into /tests after the agent's turn",
                        name,
                        source,
                        volume["target"],
                    )
                uploads.setdefault(name, []).append({"source": str(resolved), "target": volume["target"]})
                continue
            if Path(os.path.normpath(host)).is_relative_to(root):
                # The source names a path inside the task folder (its ``..`` collapsed lexically, no link followed)
                # that resolves outside it: a link inside the task that points out, not part of what prepare
                # digested. An ordinary ``../../shared`` from ``environment/`` collapses to a path outside the task
                # instead: a foreign source, looked up in ``volume_sources`` below, as Docker would bind it.
                raise ComposeVolumeError(
                    source, f"service {name!r} binds {source!r}, a link that leaves the task folder"
                )
            if source not in sources:
                raise ComposeVolumeError(
                    source,
                    f"service {name!r} binds {source!r} from outside the task; map it under `volume_sources` in the "
                    "sandbox block",
                )
            kept.append(volume)
        if kept and storage is False:
            raise ComposeVolumeError(
                kept[0]["source"],
                f"service {name!r} mounts {kept[0]['source']!r}, which needs the sandbox provider's shared storage; "
                f"{_no_shared_storage(block, name)}",
            )
        if kept:
            service["volumes"] = kept
    # The volume names this pass gives the shared logs paths, with the path each stands for.
    generated: dict[str, str] = {}
    for host, binders in shared_logs.items():
        mounts = list(binders)
        if services.get("main") is None:
            # An overlay that declares only the sidecar, or ``main:`` with no body, is Harbor's base ``main``
            # unchanged (:func:`resolve_compose` reads both as ``{}``): it receives the share like a written one, or
            # the ``main`` the resolver makes would hold a private ``/logs/verifier`` the sidecar's files never reach.
            services["main"] = {}
        if isinstance(services["main"], dict):
            # ``main`` holds the folder at the logs path itself, whether or not it binds it: that is where the
            # verifier collects from, and its own binds of the path (created and linked by the adapter) sit on it.
            # A ``main`` that is not a mapping is left for the resolver's shape error.
            mounts.append(("main", host, False))
        # The volume is mounted through the block of every service that receives it, so each needs the storage:
        # ``main``'s block (the task's, whatever it binds itself) and each binder's, which may be the GPU block.
        for service_name, _target, _read_only in mounts:
            storage, block = (shared_storage, _TASK_BLOCK) if service_name == "main" else storage_of[service_name]
            if storage is False:
                raise ComposeVolumeError(
                    host,
                    f"service {binders[0][0]!r} binds the logs tree {host!r}, which it shares with `main` through "
                    f"the sandbox provider's shared storage; {_no_shared_storage(block, service_name)}",
                )
        volume_name = "nemo-gym" + host.replace("/", "-")
        if generated.get(volume_name, host) != host:
            # ``/logs/a-b`` and ``/logs/a/b`` spell one name: the path planned second takes a digest of its own
            # spelling, the same at every prepare, so the two stay separate shares instead of the task being
            # refused as though it had declared Gym's name itself.
            volume_name += "-" + hashlib.sha256(host.encode()).hexdigest()[:8]
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", volume_name):
            # The adapter mounts a named volume only under a Compose volume name (``Named volumes require a valid
            # Compose volume name``), at every seed: a logs path with another character (a space, ``+``, ``@``) is
            # refused here instead.
            raise ComposeVolumeError(
                host,
                f"service {binders[0][0]!r} binds the logs tree {host!r}, whose name for the shared volume "
                f"({volume_name!r}) is not a Compose volume name; use letters, digits, `_`, `.` and `-` in the path",
            )
        if volume_name in declared:
            raise ComposeVolumeError(
                volume_name, f"volume {volume_name!r} is the name Gym gives the shared logs path {host!r}; rename it"
            )
        declared[volume_name] = {}
        generated[volume_name] = host
        document["volumes"] = declared
        for service_name, target, read_only in mounts:
            service = services[service_name]
            service.setdefault("volumes", []).append(
                {
                    "type": "volume",
                    "source": volume_name,
                    "target": target,
                    "read_only": read_only,
                    "volume": {"nocopy": True},
                }
            )
            if service_name != "main" and target == host:
                # The adapter makes the path world-writable in the sidecar too, before its command.
                logs.setdefault(service_name, []).append({"source": host, "target": target})
    for name, service in services.items():
        if not isinstance(service, dict):
            continue
        options: dict[str, Any] = {}
        if name in uploads:
            options["uploads"] = uploads[name]
        if name in logs:
            options["logs"] = logs[name]
        if options:
            service["x-sandbox"] = options
    return uploads


_TASK_BLOCK = "`sandbox:`"
_GPU_BLOCK = "`gpu_sandbox_provider`"


def _no_shared_storage(block: str, service_name: str) -> str:
    """The refusal's reason: the block ``service_name`` runs on has no shared storage, and where that is set."""
    return (
        f"the {block} block, where {service_name!r} runs, configures none "
        "(`shared_storage.host_path` in the OpenSandbox provider block)"
    )


def document_volume_sources(document: Mapping[str, Any], volume_sources: Mapping[str, str]) -> dict[str, str]:
    """The entries of ``volume_sources`` a planned Compose document resolves through it, and nothing else.

    The adapter treats every volume name it finds in its mapping as externally managed (one shared path,
    no per-group directory, no copy-up), so a deployment-wide mapping handed over whole would make an
    ordinary ``volumes: {pgdata: null}`` of one task share the directory another task maps as external. Only
    the names this document declares ``external: true`` and mounts, and the host paths of its remaining
    (foreign) bind mounts, are kept; a key the document never mounts, or mounts as an ordinary named volume,
    is dropped, and that volume stays private to the group, as Harbor's per-trial Compose project keeps it.
    """
    services = document.get("services") if isinstance(document, Mapping) else None
    declared = document.get("volumes") if isinstance(document, Mapping) else None
    if not isinstance(services, Mapping):
        return {}
    if not isinstance(declared, Mapping):
        declared = {}
    used: set[str] = set()
    for service in services.values():
        for volume in (service.get("volumes") or []) if isinstance(service, Mapping) else []:
            if not isinstance(volume, Mapping):
                continue
            if volume.get("type") == "bind":
                used.add(volume.get("source"))
            elif volume.get("type") == "volume" and (declared.get(volume.get("source")) or {}).get("external"):
                used.add(volume.get("source"))
    return {key: value for key, value in volume_sources.items() if key in used}


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
        # The provider starts an image as its own default user, so the record's ``User`` is only
        # copied where the adapter can act on it: never onto ``main`` (whose overlay never asks for an
        # override) and never in the ``user:group`` form the sandbox API cannot set.
        if name != "main" and config.get("User") and ":" not in str(config["User"]):
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


def _peer_url(value: Any, services: dict) -> bool:
    """Whether ``value`` is exactly one URL whose host is a Compose service, the shape ``resolve_environment`` rewrites.

    A compound value (Kafka's comma-separated ``KAFKA_ADVERTISED_LISTENERS``) parses to a hostname too, but
    its port text is not a number and the adapter would fail reading it; it is left as written.
    """
    if not isinstance(value, str) or "," in value or any(char.isspace() for char in value):
        return False
    parts = urlsplit(value)
    if not parts.scheme or parts.hostname not in services:
        return False
    try:
        parts.port
    except ValueError:
        return False
    return True


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
        peers = [key for key, value in (service.get("environment") or {}).items() if _peer_url(value, services)]
        options = {**(service.get("x-sandbox") or {}), "hosts": []}
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


def deployment_platform(provider_options: Mapping[str, Any] | None) -> str:
    """The platform a deployment runs, which a service's ``platform`` must match: the sandbox provider options'
    ``platform`` (OpenSandbox's ``{os, arch}``) when set, else linux/amd64, the one prepare records images for.
    Prepare and the server read it from the same ``sandbox_provider_options`` so the two checks agree: the
    ``os/arch`` form of :func:`sandbox_platform`, the reader the image checks use."""
    return normalize_platform("/".join(sandbox_platform(provider_options)))
