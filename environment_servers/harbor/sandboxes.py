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
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Awaitable

import yaml
from fastapi import HTTPException

from environment_servers.harbor.session import _task_sandbox_env, resolve_env_templates
from environment_servers.harbor.users import _creation_user
from environment_servers.harbor.verifier import _verifier_image, _verifier_key
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, AsyncSandboxCompose, SandboxSpec, clamp_resource_requests
from nemo_gym.sandbox.agent_tools import sandbox_server_url
from nemo_gym.sandbox.compose_config import adapt_non_root_sidecars, opensandbox_shm_labels, resolve_compose
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.providers.base import ORPHAN_REAP_TIMEOUT_S
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import apply_rollout_prefix
from nemo_gym.tasks.harbor import HarborTask
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.image_configs import (
    load_compose_document,
    without_main_build,
    without_pull_policy,
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
    verifier_environment = task.config.verifier.environment
    return bool(task.config.environment.gpus or (verifier_environment is not None and verifier_environment.gpus))


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
            ttl = (
                self.config.default_episode_timeout_seconds
                + self.config.cleanup_timeout_seconds
                + self.config.sandbox_ttl_slack_s
            )
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
        """The registry name of the provider that runs ``task`` (``docker``, ``opensandbox``, ...); ``None`` for a
        provider instance in place of the config (tests) that names itself no better."""
        if global_config_dict is None:
            global_config_dict = get_global_config_dict()
        provider_config = resolve_provider_config(self._provider_ref(task), global_config_dict)
        if isinstance(provider_config, Mapping):
            return next(iter(provider_config))
        return getattr(provider_config, "name", None)

    async def _create_sandbox(self, task: HarborTask, workdir: str | None, *, episode_key: str) -> AsyncSandbox:
        provider_config = resolve_provider_config(self._provider_ref(task), get_global_config_dict())
        sandbox = AsyncSandbox(provider_config)
        self._sandboxes[episode_key] = sandbox
        await sandbox.start(self._sandbox_spec(task, workdir))
        return sandbox

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

    def _compose_variable_values(self, episode_key: str) -> dict[str, str]:
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

    def _compose_document(
        self, task: HarborTask, deployment_variables: Mapping[str, str] | None = None
    ) -> dict[str, Any]:
        """The published overlay resolved against recorded image metadata, ready for the Compose adapter."""
        image_configs = self._compose_image_configs(task)
        # The same reader prepare used, plus the deployment's compose_variables for this episode.
        document = load_compose_document(task, deployment_variables=deployment_variables)
        # The sandbox has no pull step to configure, so ``pull_policy`` is dropped from every service. Prepare
        # ran the adapter's field check on this same document (``nemo_gym.tasks.harbor.cli``).
        document = without_pull_policy(document)
        if task.built_image is not None:
            # Prepare built main from this file; the recorded image stands in for its build: section.
            document = without_main_build(document)
        else:
            # A pull or overlay task's main build: that only restates Harbor's own build of the Dockerfile is
            # dropped; a key that would change the image is the task's error, which prepare already reported.
            document, reason = without_restated_main_build(task, document)
            if reason is not None:
                raise HarborTaskError(f"{task.task_id}: {reason}")
        document = resolve_compose(document, task.image, image_configs)
        adapt_non_root_sidecars(document, image_configs)
        provider = resolve_provider_config(self._provider_ref(task), get_global_config_dict())
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

    def _write_episode_compose_document(self, task: HarborTask, episode_key: str) -> tuple[dict[str, Any], Path]:
        """The document with this episode's ``compose_variables``, in a private folder (they may be credentials)."""
        document = self._compose_document(task, self._compose_variable_values(episode_key))
        path = Path(tempfile.mkdtemp(prefix="nemo-gym-compose-")) / "compose.yaml"
        path.write_text(yaml.safe_dump(document, sort_keys=False))
        return document, path

    async def _create_compose(self, task: HarborTask, episode_key: str) -> AsyncSandboxCompose:
        """Start the task's Compose group; the ``main`` service is the agent's sandbox.

        With ``compose_variables`` the document differs per episode (a model server's URL carries the rollout
        prefix), so it is resolved for this episode and its private folder is removed once the group starts.
        """
        private = bool(self.config.compose_variables)
        if private:
            document, path = await asyncio.to_thread(self._write_episode_compose_document, task, episode_key)
        else:
            document, path = await self._compose_document_for(task)
        # An overlay task's main service starts without the task workdir, as its single sandbox would (see
        # `_create_sandbox_within_deadline`); the sidecars derive from it and start in their images' own.
        main_spec = self._sandbox_spec(task, None if task.overlay else task.workdir)
        # Sidecars keep their own image's default user; the task's USER belongs to the main image only.
        sidecar_spec = replace(
            main_spec, resources={}, env={}, user=None, provider_options=dict(self.config.sandbox_provider_options)
        )
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
        compose = AsyncSandboxCompose(
            resolve_provider_config(self._provider_ref(task), get_global_config_dict()),
            path,
            service_specs={name: main_spec if name == "main" else sidecar_spec for name in document["services"]},
            timeout_s=self.config.sandbox_ready_timeout_s,
        )
        # Ownership starts before start(): a group that fails half-way is still stopped whole.
        self._sandboxes[episode_key] = compose
        try:
            await compose.start()
        finally:
            if private:
                shutil.rmtree(path.parent, ignore_errors=True)
        return compose

    async def _create_verifier_sandbox(self, task: HarborTask, *, episode_key: str | None = None) -> AsyncSandbox:
        """A separate verifier runs in its own sandbox, sized by ``[verifier.environment]``.

        Without a ``[verifier.environment]`` the verifier runs in the task's own image, so it is
        created like the agent's sandbox: the Dockerfile's resolved ``ENV`` under ``task.toml``'s.

        Its lifetime is the episode-sized default the agent's sandbox gets, not ``timeout_sec`` plus
        slack: the sandbox also has to come up, run the overlay when it gets one, receive ``/tests``
        and the artifacts, run ``test.sh`` with its grace and hand back the reward, and a provider's
        renew interval must fit under it. The default covers the overlay for the agent, so it does
        for the verifier too.
        """
        environment = task.config.verifier.environment
        spec = self._sandbox_spec(
            task,
            # The task image keeps the task's working directory, as the agent had it; an explicit
            # verifier image starts in its `[verifier.environment].workdir`, else its own WORKDIR.
            # An overlay task's verifier starts without the directory, as the agent's sandbox does: the
            # overlay's WORKDIR step makes it owned by the user in effect, where a provider pre-creating
            # `task.workdir` would hand the USER's RUN lines a root-owned one.
            (None if task.overlay else task.workdir) if environment is None else environment.workdir,
            environment=environment,
            image=_verifier_image(task),
            role="verifier",
        )
        provider_config = resolve_provider_config(self._provider_ref(task), get_global_config_dict())
        sandbox = AsyncSandbox(provider_config)
        if episode_key is not None:
            # Owned from before start(), like the agent's sandbox, so a failed or interrupted stop is retried.
            self._sandboxes[_verifier_key(episode_key)] = sandbox
        await self._within_ready_deadline(sandbox.start(spec))
        return sandbox
