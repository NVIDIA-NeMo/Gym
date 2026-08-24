# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Translating a ``SandboxSpec`` into IdeGYM's start-server request.

``image``
    Passed through minus a ``docker://`` scheme. It has to be an IdeGYM *server*
    image; map benchmark images onto one with ``sandbox_spec.image_rewrites``.
``resources``
    Becomes Kubernetes *limits*, with ``provider_options.resource_requests`` as the
    *requests*. IdeGYM's quota accounting reads limits first.
``env``
    IdeGYM only imports pod environment from ConfigMaps and Secrets, so plain values
    are exported per command by :mod:`~nemo_gym.sandbox.providers.idegym.shell`.
``metadata``
    Cannot become Kubernetes labels — IdeGYM has no per-server label API — so
    selected keys name the pod instead.
``ttl_s`` / ``ports`` / ``entrypoint``
    Unsupported: servers live until stopped, expose only their own service port, and
    own their container entrypoint.
"""

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from nemo_gym.sandbox.providers.base import SandboxResources, SandboxSpec
from nemo_gym.sandbox.providers.idegym.config import IdeGymCreateConfig
from nemo_gym.sandbox.providers.idegym.errors import IdeGymCreateError
from nemo_gym.sandbox.providers.idegym.naming import generate_server_name


LOGGER = logging.getLogger(__name__)

IMAGE_SCHEME_PREFIX = "docker://"
# Mirrors `idegym.api.type.OCIImageName` (1-383 characters, lowercase only). Checked
# here so a bad image fails with an explanation rather than as a pydantic error deep
# inside the SDK.
_OCI_IMAGE = re.compile(r"[a-z0-9._/:@-]{1,383}")


def normalize_image(image: str | None) -> str:
    """Return ``image`` as an IdeGYM OCI image tag."""
    if not image:
        raise IdeGymCreateError(
            "spec.image is required by the idegym provider: the sandbox is a Kubernetes pod running an "
            "IdeGYM server image."
        )
    if image.startswith(IMAGE_SCHEME_PREFIX):
        image = image[len(IMAGE_SCHEME_PREFIX) :]
    if not _OCI_IMAGE.fullmatch(image):
        raise IdeGymCreateError(
            f"spec.image {image!r} is not a valid OCI image reference for IdeGYM, which accepts only "
            f"lowercase [a-z0-9._/:@-]. Use sandbox_spec.image_rewrites to map it onto an IdeGYM server image."
        )
    return image


def format_cpu(cores: float) -> str:
    """Format a CPU count as a Kubernetes quantity: ``2``, or ``500m`` below whole cores."""
    millicores = round(cores * 1000)
    if millicores <= 0:
        raise IdeGymCreateError(f"CPU request must be greater than zero, got {cores!r}")
    return str(millicores // 1000) if millicores % 1000 == 0 else f"{millicores}m"


def format_mib(mib: int) -> str:
    """Format a memory size as a Kubernetes quantity: ``8192Mi``."""
    if mib <= 0:
        raise IdeGymCreateError(f"Memory request must be greater than zero, got {mib!r}")
    return f"{int(mib)}Mi"


def format_gib(gib: int) -> str:
    """Format a disk size as a Kubernetes quantity: ``30Gi``."""
    if gib <= 0:
        raise IdeGymCreateError(f"Disk request must be greater than zero, got {gib!r}")
    return f"{int(gib)}Gi"


def resource_quantities(resources: SandboxResources) -> dict[str, str]:
    """Render one neutral resource request as a Kubernetes quantity mapping."""
    quantities: dict[str, str] = {}
    if resources.cpu is not None:
        quantities["cpu"] = format_cpu(resources.cpu)
    if resources.memory_mib is not None:
        quantities["memory"] = format_mib(resources.memory_mib)
    if resources.disk_gib is not None:
        # Pod-local scratch space; IdeGYM has no separate volume-size knob.
        quantities["ephemeral-storage"] = format_gib(resources.disk_gib)
    return quantities


@dataclass(frozen=True)
class IdeGymProviderOptions:
    """Per-sandbox IdeGYM options carried in ``SandboxSpec.provider_options``.

    These are a validated pass-through of the IdeGYM start-server request fields
    that have no neutral equivalent. Validating here means a typo fails before a
    pod is allocated.
    """

    # Scheduling requests, paired with `spec.resources` as the limits.
    resource_requests: SandboxResources | None = None
    runtime_class_name: str | None = None
    run_as_root: bool | None = None
    node_selector: dict[str, str] = field(default_factory=dict)
    volumes: tuple[Mapping[str, Any], ...] = ()
    volume_mounts: tuple[Mapping[str, Any], ...] = ()
    env_from: tuple[Mapping[str, Any], ...] = ()
    service_account_name: str | None = None
    pod_overrides: Mapping[str, Any] | None = None
    reuse_strategy: str | None = None
    server_kind: str | None = None
    snapshot: Mapping[str, Any] | None = None
    max_restarts: int | None = None
    server_name: str | None = None
    service_port: int | None = None
    container_port: int | None = None

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any] | None) -> "IdeGymProviderOptions":
        if not options:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(
                f"Unknown idegym provider_options keys: {sorted(unknown)}. Allowed keys: {sorted(allowed)}"
            )
        # The structured fields are validated and removed one by one; whatever
        # remains is scalar pass-through the dataclass can take as-is.
        scalars = dict(options)
        requests = scalars.pop("resource_requests", None)
        node_selector = _mapping(scalars.pop("node_selector", None), "node_selector") or {}
        volumes = _mapping_tuple(scalars.pop("volumes", None), "volumes")
        volume_mounts = _mapping_tuple(scalars.pop("volume_mounts", None), "volume_mounts")
        env_from = _mapping_tuple(scalars.pop("env_from", None), "env_from")
        pod_overrides = _mapping(scalars.pop("pod_overrides", None), "pod_overrides")
        snapshot = _mapping(scalars.pop("snapshot", None), "snapshot")
        return cls(
            resource_requests=SandboxResources.from_mapping(requests) if requests is not None else None,
            node_selector={str(key): str(value) for key, value in node_selector.items()},
            volumes=volumes,
            volume_mounts=volume_mounts,
            env_from=env_from,
            pod_overrides=pod_overrides,
            snapshot=snapshot,
            **scalars,
        )


def _mapping(value: Any, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"provider_options[{name!r}] must be a mapping, got {type(value).__name__}")
    return dict(value)


def _mapping_tuple(value: Any, name: str) -> tuple[Mapping[str, Any], ...]:
    if not value:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"provider_options[{name!r}] must be a list of mappings, got {type(value).__name__}")
    entries: list[Mapping[str, Any]] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise TypeError(f"provider_options[{name!r}][{index}] must be a mapping, got {type(entry).__name__}")
        entries.append(dict(entry))
    return tuple(entries)


def _or_default(option: Any, default: Any) -> Any:
    """Return ``option``, falling back to ``default`` only when it was left unset."""
    return default if option is None else option


class ServerRequestTranslator:
    """Builds IdeGYM start-server requests from neutral sandbox specs."""

    def __init__(self, create: IdeGymCreateConfig) -> None:
        self._create = create
        # A benchmark translates one spec per task, so an unsupported field would
        # otherwise warn on every sandbox in the run.
        self._warned: set[str] = set()

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            LOGGER.warning(message)

    def server_name(self, spec: SandboxSpec) -> str:
        """Generate the RFC-1035 server name for one sandbox.

        The name carries the configured metadata hints so a pod is traceable back to
        its task. It is not made unique: IdeGYM appends its own server id, which is
        what Kubernetes sees and what the database keeps unique.
        """
        hints = [
            str(value)
            for key in self._create.server_name_metadata_keys
            if (value := spec.metadata.get(key)) not in (None, "")
        ]
        return generate_server_name(self._create.server_name_prefix, hints)

    def translate(self, spec: SandboxSpec, options: IdeGymProviderOptions) -> dict[str, Any]:
        """Return the plain-dict start-server request for ``spec``.

        Raises:
            IdeGymCreateError: If the spec asks for something IdeGYM cannot express.
        """
        self._reject_unsupported(spec)
        create = self._create
        request: dict[str, Any] = {
            "image_tag": normalize_image(spec.image),
            "server_name": options.server_name or self.server_name(spec),
            "run_as_root": _or_default(options.run_as_root, create.run_as_root),
            "service_port": _or_default(options.service_port, create.service_port),
            "container_port": _or_default(options.container_port, create.container_port),
            "reuse_strategy": options.reuse_strategy or create.reuse_strategy,
            "max_restarts": _or_default(options.max_restarts, create.max_restarts),
            # How long the SDK waits before retrying its own 429; it takes whole seconds.
            "retry_delay_in_seconds": max(1, round(create.busy_retry_delay_s)),
        }
        if resources := self._resources(spec, options):
            request["resources"] = resources
        for key, value in (
            ("runtime_class_name", options.runtime_class_name),
            ("node_selector", options.node_selector or None),
            ("volumes", list(options.volumes) or None),
            ("volume_mounts", list(options.volume_mounts) or None),
            ("env_from", list(options.env_from) or None),
            ("service_account_name", options.service_account_name),
            ("pod_overrides", options.pod_overrides),
            ("server_kind", options.server_kind),
            ("snapshot", options.snapshot),
        ):
            if value is not None:
                request[key] = value
        return request

    def _resources(self, spec: SandboxSpec, options: IdeGymProviderOptions) -> dict[str, Any]:
        limits = resource_quantities(spec.resources)
        requests = resource_quantities(options.resource_requests) if options.resource_requests is not None else {}
        self._warn_about_gpu(spec.resources, options.resource_requests)
        resources: dict[str, Any] = {}
        if requests:
            resources["requests"] = requests
        if limits:
            resources["limits"] = limits
        return resources

    def _warn_about_gpu(self, *blocks: SandboxResources | None) -> None:
        if any(block is not None and (block.gpu or block.gpu_type) for block in blocks):
            self._warn_once(
                "gpu",
                "The idegym provider cannot map GPU resource requests: IdeGYM's resource model covers "
                "cpu, memory, and ephemeral-storage only. Request accelerators through "
                "provider_options.node_selector or provider_options.pod_overrides instead.",
            )

    def _reject_unsupported(self, spec: SandboxSpec) -> None:
        if spec.entrypoint:
            raise IdeGymCreateError(
                "spec.entrypoint is not supported by the idegym provider: the container's entrypoint starts "
                "the IdeGYM server that the sandbox API talks to."
            )
        if spec.ttl_s is not None:
            self._warn_once(
                "ttl_s",
                "spec.ttl_s is not enforced by the idegym provider; servers live until close(). The IdeGYM "
                "orchestrator's own watcher reaps servers whose client stops heartbeating.",
            )
        if spec.ports:
            self._warn_once(
                "ports",
                f"spec.ports {list(spec.ports)!r} is ignored by the idegym provider: an IdeGYM server pod "
                "exposes only its own API port, and the orchestrator forwards requests rather than routing "
                "raw TCP. sandbox.endpoint() is unavailable on this provider.",
            )
