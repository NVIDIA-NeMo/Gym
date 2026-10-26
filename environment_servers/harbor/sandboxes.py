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
import random
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from environment_servers.harbor.overlay import HarborOverlayError
from environment_servers.harbor.session import _task_sandbox_env, resolve_env_templates
from environment_servers.harbor.users import _creation_user
from environment_servers.harbor.verifier import _verifier_image, _verifier_key, _verifier_runs_task_image
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, AsyncSandboxCompose, SandboxSpec, clamp_resource_requests
from nemo_gym.sandbox.compose_config import (
    adapt_non_root_sidecars,
    deployment_platform,
    document_volume_sources,
    opensandbox_shm_labels,
    resolve_compose,
    sandbox_platform,
)
from nemo_gym.sandbox.compose_fields import service_gpus
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata, resolve_provider_volume_sources
from nemo_gym.sandbox.providers.base import ORPHAN_REAP_TIMEOUT_S, SandboxCreateError
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.tasks.harbor import HarborTask
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.image_configs import (
    check_task_compose_fields,
    load_compose_document,
    plan_task_compose_volumes,
    task_declares_gpus,
    without_restated_main_build,
)
from nemo_gym.tasks.harbor.models import HarborEnvironment


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
    ) -> SandboxSpec:
        """The agent's sandbox by default; pass the verifier's environment for a separate verifier."""
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
        return SandboxSpec(
            image=image or task.image,
            ttl_s=ttl,
            ready_timeout_s=self.config.sandbox_ready_timeout_s,
            workdir=workdir,
            env=env,
            metadata=metadata,
            resources=resources,
            provider_options=self._provider_options_within(resources, task, role),
            user=user,
        )

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

    def _compose_image_configs(self, task: HarborTask) -> dict[str, Any]:
        """The image configurations `gym dataset fetch` recorded next to the task folders."""
        path = task.path.parent / COMPOSE_IMAGES_FILE
        if not path.is_file():
            raise RuntimeError(
                f"Compose task {task.task_id!r} needs recorded image configurations at {path}; "
                "prepare the dataset again to record them"
            )
        return json.loads(path.read_text())

    def _compose_document(self, task: HarborTask) -> dict[str, Any]:
        """The published overlay resolved against recorded image metadata, ready for the Compose adapter."""
        image_configs = self._compose_image_configs(task)
        global_config_dict = get_global_config_dict()
        provider = resolve_provider_config(self._provider_ref(task), global_config_dict)
        # The same reader prepare used: env_file inlined, ${VAR} resolved from the task, not the process, and the
        # images prepare built standing in for the sidecars it built (and main when the task runs from its built
        # image), their build: and a pull_policy: build dropped. The record was read once by load_task: no hashing here.
        document = load_compose_document(task)
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
        )
        # The same planner prepare used: task-file bind mounts become uploads, named volumes stay for shared
        # storage, external ones resolve through the block each service runs on (a GPU-routed sidecar through the
        # GPU block's, as `_create_compose` hands the adapter). Sources point into the server's task copy.
        plan_task_compose_volumes(
            task,
            document,
            volume_sources=self._volume_sources(task),
            gpu_volume_sources=self._gpu_volume_sources(global_config_dict),
        )
        # Every service of the group, GPU-routed or not, starts with the one configured set of provider options,
        # so the platform they request (OpenSandbox ``platform: {os, arch}``; linux/amd64 when unset) is the one
        # each recorded image has to be for (and the one `_compose_platform` spelled for the field check above).
        platform = sandbox_platform(self.config.sandbox_provider_options)
        document = resolve_compose(document, task.image, image_configs, platform=platform)
        # A non-root sidecar runs without host injection and so takes no extra_hosts: prepare refused that already.
        adapt_non_root_sidecars(document, image_configs)
        if "opensandbox" in provider:
            opensandbox_shm_labels(document)
        sidecars = {a.service for a in task.config.artifacts} | {h.service for h in task.config.verifier.collect}
        missing = sidecars - {None, "main"} - set(document["services"])
        if missing:
            raise RuntimeError(
                f"Artifacts or collect hooks name Compose services that do not exist: {sorted(missing)}"
            )
        return document

    def _write_compose_document(self, task: HarborTask) -> tuple[dict[str, Any], Path]:
        """Resolve the task's Compose document and write it where the adapter reads it (blocking file work)."""
        document = self._compose_document(task)
        # Episode IDs are opaque caller input, never host paths: the document gets a fresh folder.
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = Path(tempfile.mkdtemp(prefix="compose-", dir=self._artifacts_dir)) / "compose.yaml"
        path.write_text(yaml.safe_dump(document, sort_keys=False))
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
        """Start the task's Compose group; the ``main`` service is the agent's sandbox."""
        document, path = await self._compose_document_for(task)
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
        # Ownership starts before start(): a group that fails half-way is still stopped whole, and a pull
        # failure of any service is retried with a fresh group, like a single sandbox's.
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
            task, workdir, environment=environment, image=_verifier_image(task, True), role="verifier"
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
