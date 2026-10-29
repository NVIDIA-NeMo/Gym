# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provision the sandboxes an episode runs in.

Building the SandboxSpec from the task and the server configuration, clamping resources to
what the provider allows, routing GPU tasks to the GPU provider, creating the sandbox inside
the ready deadline with the seed grace, resolving and caching the task's Compose document and
starting the Compose group, and creating the separate verifier sandbox. The seed grace
constant and the seed timeout error live here with the deadline that uses them. Everything in
the mixin resolves the provider configuration, the global config and the sandbox classes from
this module's globals, which is where tests patch them.
"""

import asyncio
import json
import logging
import math
import os
import random
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml
from fastapi import HTTPException

from environment_servers.harbor.network_policy import phase_network_policy
from environment_servers.harbor.overlay import HarborOverlayError
from environment_servers.harbor.session import _task_sandbox_env, resolve_env_templates
from environment_servers.harbor.users import _creation_user, _is_root
from environment_servers.harbor.verifier import _verifier_image, _verifier_key, _verifier_runs_task_image
from nemo_gym.agents import sandbox_server_url
from nemo_gym.client import apply_rollout_prefix
from nemo_gym.config import get_global_config_dict
from nemo_gym.sandbox import (
    ORPHAN_REAP_TIMEOUT_S,
    AsyncSandbox,
    AsyncSandboxCompose,
    SandboxCreateError,
    SandboxSpec,
    adapt_non_root_sidecars,
    clamp_resource_requests,
    cpu_cap_env,
    deployment_platform,
    document_volume_sources,
    opensandbox_shm_labels,
    resolve_compose,
    resolve_provider_config,
    resolve_provider_metadata,
    resolve_provider_volume_sources,
    sandbox_platform,
    service_gpus,
    service_resources,
)
from nemo_gym.tasks.harbor import HarborTask
from nemo_gym.tasks.harbor.compose_overlay import overlay_files
from nemo_gym.tasks.harbor.dockerfile import RECORD_VERSION
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.image_configs import (
    check_task_compose_fields,
    check_task_compose_services,
    check_task_sidecar_extra_hosts,
    load_compose_document,
    plan_task_compose_volumes,
    reusable_record,
    task_declares_gpus,
    without_restated_main_build,
)
from nemo_gym.tasks.harbor.models import HarborEnvironment
from nemo_gym.tasks.harbor.records import MAIN_SERVICE
from nemo_gym.tasks.harbor.registry import ImageConfigError


LOGGER = logging.getLogger(__name__)

# Added to `sandbox_ready_timeout_s` for this server's own deadline on a create. The provider caps its create
# request at the ready timeout, so a pod that never schedules fails the request by then; the grace is room for
# the provider to find and delete what that create left behind (its orphan-reap allowance) and report what it
# found, before this server gives up on it. A provider attempt budget configured above the ready timeout
# (`create.timeout_s`) is not covered: the deadline here is this server's promise.
SEED_CREATE_GRACE_S = ORPHAN_REAP_TIMEOUT_S + 15.0


class HarborSeedTimeoutError(RuntimeError):
    """The sandbox was not ready within `sandbox_ready_timeout_s`: a provider-side seed failure, retried."""


COMPOSE_IMAGES_FILE = "compose-images.json"


def _sandbox_resources(environment: HarborEnvironment, *, request_gpu_type: bool = False) -> dict[str, Any]:
    resources: dict[str, Any] = {}
    if environment.cpus is not None:
        resources["cpu"] = environment.cpus
    if environment.memory_mb is not None:
        resources["memory_mib"] = environment.memory_mb
    if environment.storage_mb is not None:
        resources["disk_gib"] = max(1, math.ceil(environment.storage_mb / 1024))
    if environment.gpus:
        resources["gpu"] = environment.gpus
        if environment.gpu_types and request_gpu_type:
            resources["gpu_type"] = environment.gpu_types[0]
    return resources


def _needs_gpu(task: HarborTask) -> bool:
    # The same predicate prepare routes the Compose capability check by (image_configs.check_task_compose_fields).
    return task_declares_gpus(task)


# Image pull failures the server retries itself before it reports a retryable seed failure.
IMAGE_PULL_FAILURE_MARKERS = (
    "errimagepull",
    "imagepullbackoff",
    "failed to pull",
    "error pulling",
    "image pull",
    "pulling image",
)
# Retried starts queue behind this gate so a registry hiccup cannot turn into a pull storm.
PULL_RETRY_CONCURRENCY = 4
PULL_RETRY_BASE_DELAY_S = 5.0
PULL_RETRY_MAX_DELAY_S = 60.0


def _is_image_pull_failure(error: BaseException) -> bool:
    """Whether a failed sandbox start names an image pull as its cause.

    Walks the cause/context chain and the members of exception groups: a Compose group whose cleanup
    after a failed pull failed too surfaces as an ``ExceptionGroup`` of the cleanup errors, raised while
    the pull failure was being handled.
    """
    seen: set[int] = set()
    pending: list[BaseException] = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        message = str(current).lower()
        if any(marker in message for marker in IMAGE_PULL_FAILURE_MARKERS):
            return True
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        chained = current.__cause__ or current.__context__
        if chained is not None:
            pending.append(chained)
    return False


def _pull_retry_delay(attempt: int) -> float:
    """Exponential backoff with jitter before retrying a failed image pull; ``attempt`` is the one that failed."""
    base = min(PULL_RETRY_MAX_DELAY_S, PULL_RETRY_BASE_DELAY_S * 2 ** (attempt - 1))
    return base * random.uniform(0.5, 1.5)


def _provider_name(provider_config: Any) -> str | None:
    """The registry name of a resolved single-key provider config; ``None`` for a provider instance (tests)."""
    return next(iter(provider_config)) if isinstance(provider_config, Mapping) else None


def _discard_private_document(done: "asyncio.Task[tuple[dict[str, Any], Path]]") -> None:
    """Remove the private Compose document a write finished after its seed was cancelled (``_create_compose``)."""
    if done.cancelled() or done.exception() is not None:
        return
    _, path = done.result()
    shutil.rmtree(path.parent, ignore_errors=True)


class HarborSandboxMixin:
    """Provision the sandboxes an episode runs in.

    Building the SandboxSpec from the task and the server configuration, clamping resources to
    what the provider allows, routing GPU tasks to the GPU provider, creating the sandbox inside
    the ready deadline with the seed grace, resolving and caching the task's Compose document and
    starting the Compose group, and creating the separate verifier sandbox. The seed grace
    constant and the seed timeout error live here with the deadline that uses them. Everything in
    the mixin resolves the provider configuration, the global config and the sandbox classes from
    this module's globals, which is where tests patch them.

    """

    def _sandbox_spec(
        self,
        task: HarborTask,
        workdir: str | None,
        *,
        environment: HarborEnvironment | None = None,
        image: str | None = None,
        role: str = "agent",
        ttl: float | None = None,
        network_policy: Mapping[str, Any] | None = None,
    ) -> SandboxSpec:
        """The agent's sandbox by default; pass the verifier's environment for a separate verifier.

        ``network_policy`` is the provider's egress policy for this sandbox (the separate verifier's
        ``[verifier].network_mode``, see :meth:`_verifier_network_policy`); it replaces the deployment's
        ``sandbox_provider_options.network_policy`` for that sandbox (never widening a deny-by-default one, which
        :meth:`_verifier_network_policy` refuses), as Harbor's explicit phase policy replaces
        the environment baseline.
        """
        global_config_dict = get_global_config_dict()
        if environment is None:
            # The agent's sandbox: Dockerfile ENV recorded by the loader, under the host-resolved `[environment.env]`.
            environment, task_env = task.config.environment, _task_sandbox_env(task)
        else:
            # A separate verifier in its own image: only its `[verifier.environment].env`, host-resolved.
            task_env = resolve_env_templates(environment.env)
        metadata = (
            resolve_provider_metadata(self._provider_ref(task), global_config_dict)
            | self.config.sandbox_metadata
            | {"nemo_gym_environment_server": self.config.name, "harbor_task": task.task_id[:63], "harbor_role": role}
        )
        if ttl is None:
            ttl = self._episode_budget_s(task) + self.config.cleanup_timeout_seconds + self.config.sandbox_ttl_slack_s
        # The override replaces only the keys it names, so a GPU task routed to the GPU provider keeps `gpu`.
        resources = _sandbox_resources(environment, request_gpu_type=self.config.request_gpu_type) | (
            self.config.sandbox_resources_override or {}
        )
        # Every sandbox the server creates (the agent's, a Compose group's `main`, which starts from this spec,
        # and a separate verifier's, from its own environment) tells its processes the CPU limit through the
        # CPU-count hints (`OMP_NUM_THREADS`, `PYTHON_CPU_COUNT` and the other `CPU_CAP_ENV_VARS`), always:
        # many clusters show a sandbox the host's cores, so pools sized from `nproc` fan out past a small
        # limit. They are hints, the cgroup limit stays the provider's, and whatever the task declares
        # (`[environment.env]`, the image's `ENV`, the verifier environment's `env`) wins on a key clash.
        env = cpu_cap_env(resources.get("cpu")) | task_env
        # The default for every exec in the task's own image: the Dockerfile's final USER (or [agent].user)
        # for the agent, the image's own USER for a separate verifier in that image, independent of
        # [agent].user. A separate verifier in another image keeps that image's own default. An overlay task
        # starts without it, because its RUN lines may be what creates the user, and a numeric uid is never
        # the facade's raw default; `_set_default_user` installs the normalised default once the sandbox runs.
        default = task.entrypoint_user if role == "verifier" else task.user
        runs_task_image = image is None or image == task.image
        if runs_task_image:
            # A `name:group` default is Docker's alone: refused here, before the sandbox exists, for an overlay task
            # too (it installs the default after its RUN lines), as Daytona would create the sandbox for that user.
            self._refuse_unsupported_group(task, default, global_config_dict)
            # So is the user of every overlay RUN line (`USER root:1234` then `RUN ...`): the line would run with
            # the group lost or be refused by `su` half-way through the overlay, so the task is refused whole.
            for step in task.overlay:
                self._refuse_unsupported_group(task, step.user, global_config_dict)
        in_task_image = runs_task_image and not task.overlay
        user = _creation_user(task, default) if in_task_image else None
        provider_options = self._provider_options_within(resources, task, role)
        if network_policy is not None:
            provider_options["network_policy"] = dict(network_policy)
        return SandboxSpec(
            image=image or task.image,
            ttl_s=ttl,
            ready_timeout_s=self.config.sandbox_ready_timeout_s,
            workdir=workdir,
            env=env,
            metadata=metadata,
            resources=resources,
            provider_options=provider_options,
            user=user,
        )

    def _verifier_network_policy(self, task: HarborTask) -> dict[str, Any] | None:
        """The separate verifier's egress policy from ``[verifier].network_mode`` and ``allowed_hosts``; ``None`` when public.

        Harbor switches the environment's policy for the verify phase and back (``_phase_network_policy``);
        OpenSandbox cannot change a running sandbox's ``defaultAction`` (its egress API only patches rules
        under the policy the sandbox was created with), so the restricted policy is applied to the separate
        verifier's sandbox at creation, for the whole of its life. The loader admits the phase policy for
        separate-mode tasks only; here the provider is checked (:func:`phase_network_policy` fails closed for
        every provider but OpenSandbox) and the entries are turned into egress rules. Raises ``ValueError``.
        """
        verifier = task.config.verifier
        if verifier.network_mode in (None, "public") and not verifier.allowed_hosts:
            return None  # the common case: no provider lookup for a public verifier
        policy = phase_network_policy(self._task_provider_name(task), verifier.network_mode, verifier.allowed_hosts)
        # The task's policy replaces the deployment's for the verifier's sandbox, so under a deny-by-default
        # deployment an allowlist would open hosts the operator closed. Refused rather than intersected (no host
        # matching to get subtly wrong), whatever the deployment's own rules allow: such a task needs an
        # allow-by-default deployment; `no-network` (and an empty allowlist, the same policy) only narrows and
        # passes. OpenSandbox reads a policy that names no `defaultAction` as deny (its NetworkPolicy default),
        # so an empty mapping `{}` is a present deny-all policy (the 0.1.15 SDK), not an absent one; only no
        # policy at all is allow-all.
        deployment = self.config.sandbox_provider_options.get("network_policy")
        if deployment is not None and not isinstance(deployment, Mapping):
            raise ValueError(
                f"sandbox_provider_options.network_policy must be a mapping, got {type(deployment).__name__}"
            )
        denies = deployment is not None and deployment.get("defaultAction", "deny") == "deny"
        if denies and policy is not None and policy["egress"]:
            hosts = ", ".join(rule["target"] for rule in policy["egress"])
            raise ValueError(
                f"allowed_hosts {hosts} would widen this deployment's deny-by-default "
                "sandbox_provider_options.network_policy for the verifier; run the task on an allow-by-default "
                "deployment (its deny rules still apply to the agent's sandbox)"
            )
        return policy

    def _provider_options_within(self, resources: Mapping[str, Any], task: HarborTask, role: str) -> dict[str, Any]:
        """The configured provider options, with scheduling requests capped at the task's declared limits.

        A deployment's ``resource_requests`` (say ``disk_gib: 10`` for every sandbox) can exceed what a
        task declares (``storage = "4G"``); Kubernetes never schedules that pod and the provider would only
        see it pending. The provider clamps too; the warning here names the task.
        """
        options = dict(self.config.sandbox_provider_options)
        requests = options.get("resource_requests")
        if not isinstance(requests, Mapping):
            return options
        clamped, changed = clamp_resource_requests(requests, resources)
        for key, (request, limit) in changed.items():
            LOGGER.warning(
                "sandbox_provider_options.resource_requests.%s=%s exceeds the %s limit %s task %s declares; "
                "clamping the request to the limit so the sandbox can be scheduled",
                key,
                request,
                key,
                limit,
                f"{task.task_id} ({role})",
            )
        if changed:
            options["resource_requests"] = clamped
        return options

    def _provider_ref(self, task: HarborTask) -> str:
        """Which top-level sandbox block runs this task: the GPU one when the task declares GPUs."""
        if self.config.gpu_sandbox_provider and _needs_gpu(task):
            return self.config.gpu_sandbox_provider
        return self.config.sandbox_provider

    def _task_provider_name(self, task: HarborTask, global_config_dict: Mapping[str, Any] | None = None) -> str | None:
        """The registry name of the provider that runs ``task`` (``docker``, ``opensandbox``, ...): :func:`_provider_name`
        of its resolved config, else the own ``name`` of a provider instance in place of the config (tests), else
        ``None``."""
        if global_config_dict is None:
            global_config_dict = get_global_config_dict()
        provider_config = resolve_provider_config(self._provider_ref(task), global_config_dict)
        return _provider_name(provider_config) or getattr(provider_config, "name", None)

    async def _create_sandbox(self, task: HarborTask, workdir: str | None, *, episode_key: str) -> AsyncSandbox:
        """Start the agent's sandbox, with the in-server image pull retries."""
        provider_config = resolve_provider_config(self._provider_ref(task), get_global_config_dict())
        spec = self._sandbox_spec(task, workdir)
        return await self._start_with_pull_retries(
            task,
            key=episode_key,
            what=repr(task.image),
            make=lambda: AsyncSandbox(provider_config),
            start=lambda sandbox: sandbox.start(spec),
        )

    async def _create_sandbox_within_deadline(self, task: HarborTask, episode_key: str) -> AsyncSandbox:
        """The agent's ``_create_sandbox`` under the ready deadline; the timeout is a seed failure, retried.

        An overlay task starts in the base image's own working directory: a provider creates the spec's
        directory as root before anything runs, and the overlay's ``WORKDIR`` steps (then ``_prepare_workdir``)
        would find it existing and leave it root-owned. Created by the overlay instead, it belongs to the
        user in effect at that ``WORKDIR``, as in the build. The ``main`` service of a Compose group starts
        the same way (:meth:`_create_compose`).
        """
        workdir = None if task.overlay else task.workdir
        return await self._within_ready_deadline(self._create_sandbox(task, workdir, episode_key=episode_key))

    async def _within_ready_deadline[T](self, create: Awaitable[T]) -> T:
        """A sandbox create bounded by the configured ready timeout, whatever the provider's retry budget.

        The provider retries a create on its own schedule and its request may block as long as the cluster
        takes to allocate the pod; `sandbox_ready_timeout_s` is this server's promise, so it is enforced
        here too, on the agent's sandbox and on the separate verifier's alike. The timeout raises
        ``HarborSeedTimeoutError``; the cancelled provider attempt deletes the sandbox it may have started.
        """
        deadline_s = self.config.sandbox_ready_timeout_s + SEED_CREATE_GRACE_S
        deadline = asyncio.timeout(deadline_s)
        try:
            async with deadline:
                return await create
        except TimeoutError as exc:
            if not deadline.expired():
                raise
            raise HarborSeedTimeoutError(
                f"the sandbox was not ready within {deadline_s:g}s "
                f"(sandbox_ready_timeout_s={self.config.sandbox_ready_timeout_s:g} plus {SEED_CREATE_GRACE_S:g}s grace)"
            ) from exc

    def _uses_compose(self, task: HarborTask) -> bool:
        """Whether the episode starts a Compose group: the task ships a Compose file, or the job-level overlay names
        Compose files, which turn a plain image task into Harbor's one-service group plus the overlay's services."""
        return task.needs_compose or bool(overlay_files(self.config.compose_overlay))

    def _compose_image_configs(self, task: HarborTask) -> dict[str, Any]:
        """The image configurations `gym dataset fetch` recorded next to the task folders."""
        path = task.path.parent / COMPOSE_IMAGES_FILE
        if not path.is_file():
            raise RuntimeError(
                f"Compose task {task.task_id!r} needs recorded image configurations at {path}; "
                "prepare the dataset again to record them"
            )
        return json.loads(path.read_text())

    def _check_compose_images(self, task: HarborTask, document: Mapping[str, Any]) -> dict[str, Any]:
        """Refuse, before anything is provisioned, a Compose group whose images prepare did not record for this
        deployment: the record file missing next to the task folders, or a service of the merged document (an overlay's,
        or ``main`` with the task's image) without an entry this deployment can use, by prepare's own rule
        (:func:`~nemo_gym.tasks.harbor.image_configs.reusable_record`: the current record version and the platform the
        sandboxes run; ``resolve_compose`` refuses a record of another platform at the group's start). A dataset
        prepared without the ``compose_overlay`` the server now runs with, or for another platform (prepared on x86,
        served on arm64), is the usual cause; it is a configuration error, so the seed is refused terminally (422)
        naming the services and what was recorded, where the group's start would raise and be retried for every
        episode. Returns the records, for the checks that read them (``_check_compose_group``)."""
        path = task.path.parent / COMPOSE_IMAGES_FILE
        if not path.is_file():
            raise HTTPException(
                422,
                f"Compose group of task {task.task_id!r} needs the image configurations prepare records at {path}; "
                "prepare the dataset again, with this compose_overlay when one is set",
            )
        recorded = json.loads(path.read_text())
        platform = sandbox_platform(self.config.sandbox_provider_options)
        missing: list[str] = []
        for name, service in (document.get("services") or {}).items():
            image = service.get("image") if isinstance(service, dict) else None
            if image is None and name == MAIN_SERVICE:
                image = task.image
            record = recorded.get(image)
            if reusable_record(record, platform):
                continue
            if isinstance(record, dict):
                # Recorded, but not for this deployment: say what the record is, so the operator knows which prepare.
                was = (
                    f"{str(record.get('os') or '?').lower()}/{str(record.get('architecture') or '?').lower()}, record "
                    f"version {record.get('record_version')!r}"
                )
                missing.append(
                    f"service {name!r} ({image!r} is recorded for {was}; this deployment runs {'/'.join(platform)} and "
                    f"reads record version {RECORD_VERSION})"
                )
            else:
                missing.append(f"service {name!r} ({image!r})")
        if missing:
            raise HTTPException(
                422,
                f"Compose group of task {task.task_id!r}: no image configuration this deployment can use recorded in "
                f"{path} for {', '.join(missing)}; prepare the dataset again with this compose_overlay, for this "
                "platform, so its images are recorded. An image reference, or a variable in one, that only a seed-time "
                "compose_variables value sets resolves to its default at prepare, so set the value in prepare's process "
                "environment too: image references and the variables they use must resolve at preparation",
            )
        return recorded

    def _compose_preflight(self, task: HarborTask, episode_key: str) -> None:
        """The seed's checks on the task's Compose group, before anything is provisioned: blocking file work (the
        task's Compose file, the overlay files, their env files and the image records are read here), so ``_seed``
        runs it in a worker thread, off the loop that serves every other episode. The merged document must load with
        this deployment's values (a variable no ``compose_variables`` entry sets, an overlay file that does not merge
        onto the task's: a configuration error, refused and not retried), its images must be recorded for this
        deployment (``_check_compose_images``) and the deployment must be able to run it (``_check_compose_group``:
        every check prepare ran, on the document the group would start, rather than inside ``_create_compose`` where
        the failure would be retried for every episode; the overlay is a run-config value, so it changes between
        prepare and serve more easily than a task file). The records read for the image check are handed to the group
        check, so the record file is read once here. Raises ``HTTPException`` (422) naming the task and the reason."""
        try:
            document = load_compose_document(
                task,
                operator_variables=self._compose_variable_values(episode_key),
                overlay=self.config.compose_overlay,
            )
        except ImageConfigError as exc:
            raise HTTPException(422, f"Compose file of task {task.task_id!r}: {exc}") from exc
        recorded = self._check_compose_images(task, document)
        try:
            self._check_compose_group(task, document, recorded)
        except (ImageConfigError, HarborTaskError) as exc:
            raise HTTPException(422, f"Compose group of task {task.task_id!r}: {exc}") from exc

    def _check_compose_group(
        self, task: HarborTask, document: dict[str, Any], image_configs: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """The checks prepare ran on the task's Compose document, with this deployment's inputs: ``main``'s restated
        build dropped, the field check against the providers the services run on, the volume planner against the
        blocks' ``volume_sources``, the services the task's artifacts and collect hooks name present in the merged
        document, and no ``extra_hosts`` on a sidecar whose recorded image runs as a non-root user (``image_configs``
        is the record file the seed already read, or it is read here). Mutates and returns ``document`` (its fields
        in their normalized forms, its bind mounts planned). The seed runs it before anything is provisioned, so a
        document the deployment cannot run (a job-level overlay file with a field the group cannot run, a bind
        source no ``volume_sources`` entry maps, a service reset away that a collect hook names, host entries on a
        non-root sidecar; the overlay being a run-config value that changes between prepare and serve) is refused
        terminally there, with the reason, and not raised from the group's start and retried for every episode;
        ``_compose_document`` runs it again on the document the group starts. Raises :class:`ImageConfigError` or
        :class:`HarborTaskError` naming the task, the files and the field."""
        global_config_dict = get_global_config_dict()
        provider = resolve_provider_config(self._provider_ref(task), global_config_dict)
        if task.built_image is None:
            # A pull or overlay task's main build: (and the pull_policy: build that goes with it) that only restates
            # Harbor's own build of the Dockerfile is dropped; a key that would change the image is the task's
            # error, which prepare already reported. Every other pull_policy stays for the adapter to apply.
            document, reason = without_restated_main_build(task, document)
            if reason is not None:
                raise HarborTaskError(f"{task.task_id}: {reason}")
        # The same field check prepare ran, with the same inputs: deploy.resources, networks, extra_hosts, post_start
        # and the rest take their normalized forms, each service is checked against the provider it runs on (a GPU
        # sidecar against the GPU block's) and `platform` against this deployment's, so a field the deployment cannot
        # apply fails here, before a sandbox. A provider instance in place of the config (tests) has no registry
        # name; the adapter checks it live.
        gpu_provider = (
            resolve_provider_config(self.config.gpu_sandbox_provider, global_config_dict)
            if self.config.gpu_sandbox_provider
            else None
        )
        check_task_compose_fields(
            task,
            document,
            provider=_provider_name(provider),
            gpu_provider=_provider_name(gpu_provider),
            platform=self._compose_platform(),
            overlay=self.config.compose_overlay,
        )
        # The same planner prepare used: task-file bind mounts become uploads, named volumes stay for shared
        # storage, external ones resolve through the block each service runs on (a GPU-routed sidecar through the
        # GPU block's, as `_create_compose` hands the adapter). Sources point into the server's task copy.
        plan_task_compose_volumes(
            task,
            document,
            volume_sources=self._volume_sources(task),
            gpu_volume_sources=self._gpu_volume_sources(global_config_dict),
            overlay=self.config.compose_overlay,
        )
        # The two checks prepare runs after those, in its order: the services the task's artifacts and collect hooks
        # name are in the merged document (an overlay file can reset one away), and a sidecar whose recorded image
        # runs as a non-root user declares no extra_hosts (the group starts it without host injection;
        # `adapt_non_root_sidecars` would raise the same inside the group's start).
        check_task_compose_services(task, document, overlay=self.config.compose_overlay)
        if image_configs is None:
            image_configs = self._compose_image_configs(task)
        check_task_sidecar_extra_hosts(task, document, image_configs, overlay=self.config.compose_overlay)
        return document

    def _compose_document(self, task: HarborTask, episode_key: str | None = None) -> dict[str, Any]:
        """The published overlay (with the job-level overlay files merged on) resolved against recorded image
        metadata, ready for the Compose adapter."""
        image_configs = self._compose_image_configs(task)
        global_config_dict = get_global_config_dict()
        provider = resolve_provider_config(self._provider_ref(task), global_config_dict)
        # The same reader prepare used: env_file inlined, ${VAR} resolved from the task, not the process, and the
        # images prepare built standing in for the sidecars it built (and main when the task runs from its built
        # image), their build: and a pull_policy: build dropped. The record was read once by load_task: no hashing here.
        # The deployment's compose_variables fill the names the task leaves to it (a name still unset fails here).
        document = load_compose_document(
            task, operator_variables=self._compose_variable_values(episode_key), overlay=self.config.compose_overlay
        )
        # The checks the seed already ran on this document (`_check_compose_group`), on the one the group starts.
        document = self._check_compose_group(task, document, image_configs)
        # Every service of the group, GPU-routed or not, starts with the one configured set of provider options,
        # so the platform they request (OpenSandbox ``platform: {os, arch}``; linux/amd64 when unset) is the one
        # each recorded image has to be for (and the one `_compose_platform` spelled for the field check above).
        platform = sandbox_platform(self.config.sandbox_provider_options)
        document = resolve_compose(document, task.image, image_configs, platform=platform)
        # A non-root sidecar runs without host injection and so takes no extra_hosts: prepare and the check above
        # refused that already.
        adapt_non_root_sidecars(document, image_configs)
        if "opensandbox" in provider:
            opensandbox_shm_labels(document)
        # Checked above too (`check_task_compose_services`, before the resolution): the last line before the adapter.
        sidecars = {a.service for a in task.config.artifacts} | {h.service for h in task.config.verifier.collect}
        missing = sidecars - {None, "main"} - set(document["services"])
        if missing:
            raise RuntimeError(
                f"Artifacts or collect hooks name Compose services that do not exist: {sorted(missing)}"
            )
        if overlay_files(self.config.compose_overlay):
            # The merged group is the run's, not the task's: say what the overlay made of this task once per
            # resolution (per episode, since the document is private), so the run log shows the attached services.
            LOGGER.info(
                "%s: Compose group with the job-level overlay (%d file(s)): services %s; MCP servers %s",
                task.task_id,
                len(overlay_files(self.config.compose_overlay)),
                list(document["services"]),
                [server.name for server in self._task_mcp_servers(task)],
            )
        return document

    def _write_compose_document(
        self, task: HarborTask, episode_key: str | None = None, *, private: bool = False
    ) -> tuple[dict[str, Any], Path]:
        """Resolve the task's Compose document and write it where the adapter reads it (blocking file work).

        ``private`` puts it in a folder of its own outside ``artifacts_dir`` (the caller removes it once the group
        has started): a document with deployment values in it may hold credentials.
        """
        # The shared document (one per task digest) has no episode; the private one carries the episode's values.
        document = self._compose_document(task, episode_key) if private else self._compose_document(task)
        if private:
            folder = Path(tempfile.mkdtemp(prefix="nemo-gym-compose-"))
        else:
            # Episode IDs are opaque caller input, never host paths: the document gets a fresh folder.
            self._artifacts_dir.mkdir(parents=True, exist_ok=True)
            folder = Path(tempfile.mkdtemp(prefix="compose-", dir=self._artifacts_dir))
        path = folder / "compose.yaml"
        try:
            path.write_text(yaml.safe_dump(document, sort_keys=False))
        except BaseException:
            if private:
                # A write that failed after the folder was made leaves no partial document with the values in it.
                shutil.rmtree(folder, ignore_errors=True)
            raise
        return document, path

    async def _compose_document_for(self, task: HarborTask) -> tuple[dict[str, Any], Path]:
        """The resolved document and its compose.yaml, parsed and written once per task digest.

        The document is the same for every episode of a task (the published overlay, the recorded image
        metadata and the provider block do not change under a running server), so the YAML and JSON
        reads and the write happen once, in a thread; concurrent first episodes share the one resolution,
        and a failed one is forgotten so the next episode tries again.
        """
        key = (task.path, task.digest)
        pending = self._compose_documents.get(key)
        if pending is None:
            pending = asyncio.create_task(asyncio.to_thread(self._write_compose_document, task))
            self._compose_documents[key] = pending

            def forget_failed(done: asyncio.Task[tuple[dict[str, Any], Path]]) -> None:
                if done.cancelled() or done.exception() is not None:
                    self._compose_documents.pop(key, None)

            pending.add_done_callback(forget_failed)
        return pending.result() if pending.done() else await asyncio.shield(pending)

    async def _create_compose(self, task: HarborTask, episode_key: str) -> AsyncSandboxCompose:
        """Start the task's Compose group; the ``main`` service is the agent's sandbox.

        With ``compose_variables`` or a job-level Compose overlay the document is the episode's (a model server URL
        carries the rollout prefix) and may hold credentials: the adapter reads it once, at start, from a private
        folder removed right after, never under ``artifacts_dir``. Without them it is the task's, resolved and
        written once per digest.
        """
        # A job-level overlay file is the operator's and commonly carries the deployment's keys (a memory
        # sidecar's model key), so its document stays private too, like one with compose_variables.
        private = bool(self.config.compose_variables) or bool(overlay_files(self.config.compose_overlay))
        if private:
            # The write runs in a worker thread, which a cancelled await does not stop: the write stays owned past the
            # cancellation (an episode deadline, a shutdown), so a document the worker finishes after it is removed
            # as soon as it is done, instead of staying behind with the deployment's values in it.
            writing = asyncio.create_task(
                asyncio.to_thread(self._write_compose_document, task, episode_key, private=True)
            )
            try:
                document, path = await asyncio.shield(writing)
            except asyncio.CancelledError:
                writing.add_done_callback(_discard_private_document)
                raise
        else:
            document, path = await self._compose_document_for(task)
        try:
            return await self._start_compose_group(task, episode_key, document, path)
        finally:
            # The private document goes with every exit, a refusal before any group exists (`_sandbox_spec` refusing
            # an agent user the provider cannot run as, a sidecar's provider block) as much as the start's return.
            if private:
                shutil.rmtree(path.parent, ignore_errors=True)

    async def _start_compose_group(
        self, task: HarborTask, episode_key: str, document: dict[str, Any], path: Path
    ) -> AsyncSandboxCompose:
        """Start the group ``document`` (written at ``path``, where the adapter reads it) describes, for ``_create_compose``:
        the service specs, providers and volume mappings from the task and the document, the overlay's RUN lines on
        ``main`` for an overlay task, then the start with its pull retries."""
        # An overlay task's main service starts without the task workdir, as its single sandbox would (see
        # `_create_sandbox_within_deadline`); the sidecars derive from it and start in their images' own.
        main_spec = self._sandbox_spec(task, None if task.overlay else task.workdir)
        # Harbor's order for main: image ENV (baked in, or the Dockerfile ENV the loader folded into task.env for a
        # pull or overlay task) < the task's Compose file < Harbor's env override file, which holds only the
        # host-resolved [environment].env (harbor 0.22.0 environments/docker/docker.py 363-370, environments/base.py
        # 278-285). _sandbox_spec's env is task.env, the image layer with [environment].env already over it; the
        # file goes between the two here, and the adapter applies the whole spec env over the file (``{**env,
        # **spec.env}``), which leaves exactly those three layers in that order.
        compose_env = {
            key: str(value)
            for key, value in ((document["services"].get("main") or {}).get("environment") or {}).items()
            if value is not None
        }
        main_spec = replace(
            main_spec,
            env={**main_spec.env, **compose_env, **resolve_env_templates(task.config.environment.env or {})},
        )
        global_config_dict = get_global_config_dict()
        task_ref = self._provider_ref(task)
        # Sidecars keep their own image's default user; the task's USER belongs to the main image only. A sidecar
        # that reserves GPUs runs on the GPU block, per service, the way the main service does for the task, and
        # resolves its external volumes through that block's mapping. Each mapping is narrowed to the names this
        # document resolves through it, so an ordinary named volume never takes a mapped name's shared path.
        service_specs: dict[str, SandboxSpec] = {}
        service_providers: dict[str, Any] = {}
        service_volume_sources: dict[str, dict[str, str]] = {}
        self._warn_unsized_sidecars(task, document)
        for name, service in document["services"].items():
            if name == "main":
                service_specs[name] = main_spec
                continue
            service_specs[name] = self._sidecar_spec(task, main_spec, service)
            ref = self._sidecar_provider_ref(task, service)
            if ref != task_ref:
                service_providers[name] = resolve_provider_config(ref, global_config_dict)
                service_volume_sources[name] = document_volume_sources(
                    document, resolve_provider_volume_sources(ref, global_config_dict)
                )
        volume_sources = document_volume_sources(document, self._volume_sources(task))
        provider_config = resolve_provider_config(task_ref, global_config_dict)
        platform = self._compose_platform()
        # An overlay task's Dockerfile RUN lines run inside the group's start, on main once it exists and before
        # its bind-mount uploads, its command and its healthcheck: Harbor's order (a build, then `compose up` with
        # the mounts). The overlay keeps its own budget on top of the start's, and the names it resolved wait for
        # the seed under the episode key.
        service_setup: dict[str, Callable[[AsyncSandbox], Awaitable[None]]] = {}
        timeout_s = self.config.sandbox_ready_timeout_s
        overlay_failure: list[HarborOverlayError] = []
        if task.overlay:
            timeout_s += task.config.environment.build_timeout_sec

            async def run_overlay(sandbox: AsyncSandbox) -> None:
                try:
                    self._overlay_names[episode_key] = await self._apply_overlay(sandbox, task)
                except HarborOverlayError as exc:
                    overlay_failure.append(exc)
                    raise

            service_setup["main"] = run_overlay
        elif task.built_image is None and task.entrypoint_workdir and not _is_root(task.entrypoint_user):

            async def own_workdir(sandbox: AsyncSandbox) -> None:
                # `synthesized_group` writes the Dockerfile's final WORKDIR as main's working_dir, which the adapter
                # has just made as root, as Docker makes a missing working_dir at create. The single sandbox makes
                # that directory the entrypoint user's before the entrypoint starts (`_start_entrypoint`, through
                # `overlay_workdir_command`: a WORKDIR under that USER, as the build makes it), so it is handed over
                # here, in the adapter's slot between the create and main's command, which starts the entrypoint in
                # it; the seed's `_prepare_workdir` then hands the task workdir to `task.user`, as in a single sandbox.
                # A built image carries its WORKDIR's owner itself, and an overlay task's is the overlay's (above).
                await self._own_workdir(sandbox, task.entrypoint_workdir, task.entrypoint_user)

            service_setup["main"] = own_workdir
        # Ownership starts before start(): a group that fails half-way is still stopped whole, and a pull
        # failure of any service is retried with a fresh group, like a single sandbox's. A private document
        # is read by every attempt and removed by `_create_compose` once the last one has returned.
        try:
            return await self._start_with_pull_retries(
                task,
                key=episode_key,
                what="Compose group",
                make=lambda: AsyncSandboxCompose(
                    provider_config,
                    path,
                    service_specs=service_specs,
                    timeout_s=timeout_s,
                    service_providers=service_providers,
                    platform=platform,
                    volume_sources=volume_sources,
                    service_volume_sources=service_volume_sources,
                    service_setup=service_setup,
                    # Upload sources may only come from the server's own copy of the task, the one the planner saw.
                    upload_root=task.path,
                ),
                start=lambda compose: compose.start(),
            )
        except ExceptionGroup as exc:
            # The group's cleanup after a failed overlay failed too: the overlay is still the seed's reason.
            self._overlay_names.pop(episode_key, None)
            if overlay_failure:
                raise overlay_failure[-1] from exc
            raise
        except BaseException:
            self._overlay_names.pop(episode_key, None)
            raise

    async def _create_verifier_sandbox(self, task: HarborTask, *, episode_key: str | None = None) -> AsyncSandbox:
        """A separate verifier runs in its own sandbox, sized by ``[verifier.environment]``.

        Without a ``[verifier.environment]`` and a built verifier image the verifier runs in the task's own prebuilt
        image, so it is created like the agent's sandbox: the Dockerfile's resolved ``ENV`` under ``task.toml``'s.

        Its lifetime is the episode-sized default the agent's sandbox gets, not ``timeout_sec`` plus
        slack: the sandbox also has to come up, receive ``/tests`` and the artifacts, run ``test.sh``
        with its grace and hand back the reward, and a provider's renew interval must fit under it.
        """
        environment = task.config.verifier.environment
        if _verifier_runs_task_image(task, True):
            # The task image keeps the task's working directory, as the agent had it, and the agent's resolved
            # environment (Dockerfile ENV under task.toml's). An overlay task starts without it, as the agent's
            # sandbox does: the overlay's WORKDIR step makes the directory owned by the user in effect, where a
            # provider pre-creating `task.workdir` would hand the USER's RUN lines a root-owned one.
            workdir = None if task.overlay else task.workdir
        elif environment is None:
            # Built from tests/Dockerfile under Harbor's copy of [environment]: that block's settings and
            # `workdir` (else the image's own WORKDIR), not the agent image's Dockerfile ENV and WORKDIR.
            environment = task.config.environment
            workdir = environment.workdir
        else:
            # An explicit verifier image starts in its `[verifier.environment].workdir`, else its own WORKDIR.
            workdir = environment.workdir
        spec = self._sandbox_spec(
            task,
            workdir,
            environment=environment,
            image=_verifier_image(task, True),
            role="verifier",
            network_policy=self._verifier_network_policy(task),
        )
        provider_config = resolve_provider_config(self._provider_ref(task), get_global_config_dict())
        if episode_key is None:
            sandbox = AsyncSandbox(provider_config)
            await self._within_ready_deadline(sandbox.start(spec))
            return sandbox
        # Owned from before start(), like the agent's sandbox, so a failed or interrupted stop is retried;
        # a start that failed on an image pull is retried like the agent's, under the same ready deadline.
        return await self._within_ready_deadline(
            self._start_with_pull_retries(
                task,
                key=_verifier_key(episode_key),
                what=f"verifier {spec.image!r}",
                make=lambda: AsyncSandbox(provider_config),
                start=lambda sandbox: sandbox.start(spec),
            )
        )

    async def _start_with_pull_retries(
        self,
        task: HarborTask,
        *,
        key: str,
        what: str,
        make: Callable[[], Any],
        start: Callable[[Any], Awaitable[None]],
    ) -> Any:
        """Start a sandbox or Compose group, retrying a start that failed on an image pull `image_pull_retries` times.

        ``make`` builds a fresh object (neither accepts a second ``start``), owned under ``key`` from before
        ``start`` so that a failed or half-started one is stopped whole. Only pull failures are retried here
        (a provider that rejects the request, or a sandbox that never becomes ready, is reported at once);
        the retries back off with jitter and queue behind a per-server gate, so one registry hiccup cannot
        multiply into a pull storm.
        """
        attempts = self.config.image_pull_retries + 1
        for attempt in range(1, attempts + 1):
            sandbox = make()
            self._sandboxes[key] = sandbox
            try:
                if attempt == 1:
                    await start(sandbox)
                else:
                    async with self._pull_retry_gate:
                        await start(sandbox)
                return sandbox
            except (SandboxCreateError, RuntimeError, OSError, ExceptionGroup) as exc:
                # A Compose group whose cleanup failed too raises an ExceptionGroup over the pull failure.
                if attempt >= attempts or not _is_image_pull_failure(exc):
                    raise
                LOGGER.warning(
                    f"{task.task_id}: image pull failed starting {what} (attempt {attempt}/{attempts}): {exc}"
                )
                try:
                    await self._stop_sandbox(key)
                except Exception:
                    # Still owned: shutdown retries the stop. Do not start a second sandbox under the same key.
                    LOGGER.exception(f"{task.task_id}: could not stop the {what} of a failed image pull")
                    raise exc from None
                await asyncio.sleep(_pull_retry_delay(attempt))
        raise AssertionError("unreachable")  # pragma: no cover

    def _warn_unsized_sidecars(self, task: HarborTask, document: dict[str, Any]) -> None:
        """Say once per task which sidecars run at the provider's default sandbox size.

        A sidecar's size is its ``deploy.resources`` (:func:`_sidecar_spec` hands the adapter no task resources);
        without one it gets whatever the provider gives a sandbox that asks for nothing (the OpenSandbox SDK's
        defaults), where Harbor's ``docker compose`` gives it the host's resources. A sidecar that loads a large
        catalog then dies at start, so the choice is logged, with the two places to size it; the default itself
        is not changed here.
        """
        unsized = sorted(
            name
            for name, service in document["services"].items()
            if name != "main" and not {"cpu", "memory_mib"} & set(service_resources(service))
        )
        if not unsized or task.digest in self._unsized_sidecars_warned:
            return
        self._unsized_sidecars_warned.add(task.digest)
        LOGGER.warning(
            f"{task.task_id}: Compose service(s) {', '.join(unsized)} declare no deploy.resources, so each runs at the "
            "sandbox provider's default size (under Harbor's docker compose a sidecar has the host's resources); size a "
            "sidecar with deploy.resources.limits (cpus, memory) in the task's Compose file"
        )

    def _sidecar_provider_ref(self, task: HarborTask, service: dict[str, Any]) -> str:
        """Which sandbox block runs a Compose sidecar: the GPU one when the service reserves GPUs through
        ``deploy.resources`` and a GPU block is set, as ``_provider_ref`` routes the main service."""
        if self.config.gpu_sandbox_provider and service_gpus(service):
            return self.config.gpu_sandbox_provider
        return self._provider_ref(task)

    def _sidecar_spec(self, task: HarborTask, main_spec: SandboxSpec, service: dict[str, Any]) -> SandboxSpec:
        """A sidecar's spec: the main spec's deployment settings, no task resources, env or user (the sidecar keeps
        its own image's defaults; its ``deploy.resources`` are applied by the adapter). A GPU sidecar carries
        the task's ``gpu_types[0]`` when ``request_gpu_type`` is on, as the main service does, and the GPU
        block's ``default_metadata``. A sidecar keeps its own image's working directory (the adapter applies the
        recorded WorkingDir); main's workdir is a path that exists only in main's image."""
        resources: dict[str, Any] = {}
        metadata = dict(main_spec.metadata)
        if service_gpus(service):
            gpu_types = task.config.environment.gpu_types
            if gpu_types and self.config.request_gpu_type:
                resources["gpu_type"] = gpu_types[0]
            metadata = (
                resolve_provider_metadata(self._sidecar_provider_ref(task, service), get_global_config_dict())
                | self.config.sandbox_metadata
                | {key: metadata[key] for key in ("nemo_gym_environment_server", "harbor_task", "harbor_role")}
            )
        return replace(
            main_spec,
            resources=resources,
            env={},
            user=None,
            workdir=None,
            metadata=metadata,
            provider_options=dict(self.config.sandbox_provider_options),
        )

    def _compose_platform(self) -> str:
        """The platform this deployment runs, for a service's ``platform`` to match: the provider options'
        ``platform`` (OpenSandbox's ``{os, arch}``) when set, else linux/amd64, the one prepare records images for.
        Prepare reads the same ``sandbox_provider_options`` (``compose_platform_from_overrides``), so both agree."""
        return deployment_platform(self.config.sandbox_provider_options)

    def _volume_sources(self, task: HarborTask) -> dict[str, str]:
        """The task's provider block's ``volume_sources``: external volumes and foreign bind sources on shared
        storage, the whole deployment-wide mapping. The planner checks the document against it; the adapter gets
        only the entries the planned document resolves through it (:func:`document_volume_sources`, in
        `_create_compose`), so an ordinary named volume that happens to share a mapped name stays the group's own."""
        return resolve_provider_volume_sources(self._provider_ref(task), get_global_config_dict())

    def _gpu_volume_sources(self, global_config_dict: dict[str, Any]) -> dict[str, str] | None:
        """The GPU block's ``volume_sources`` (empty when it has none), or ``None`` without a GPU block: what a
        GPU-routed service resolves its external volumes through, as `_sidecar_provider_ref` routes it."""
        if not self.config.gpu_sandbox_provider:
            return None
        return resolve_provider_volume_sources(self.config.gpu_sandbox_provider, global_config_dict)

    def _compose_variable_values(self, episode_key: str | None) -> dict[str, str]:
        """The deployment's ``compose_variables`` resolved for one episode.

        A ``model_server`` value is that server's sandbox-reachable root URL with the episode's rollout prefix
        (no training-capture segment), so a sidecar's model calls are kept with the rollout's evaluation
        capture and never become training tokens. A missing environment variable or an unreachable model
        server is a configuration error (422).
        """
        values: dict[str, str] = {}
        for name, source in self.config.compose_variables.items():
            if source.value is not None:
                values[name] = source.value
            elif source.env is not None:
                if source.env not in os.environ:
                    raise HTTPException(422, f"compose_variables.{name}: environment variable {source.env!r} is unset")
                values[name] = os.environ[source.env]
            else:
                try:
                    base_url = sandbox_server_url(source.model_server, require_reachable=True)
                except Exception as exc:
                    raise HTTPException(422, f"compose_variables.{name}: {type(exc).__name__}: {exc}") from exc
                values[name] = apply_rollout_prefix(base_url, episode_key)
        return values
