# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Resolve the canonical per-task execution-environment contract."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nemo_gym.sandbox.providers.base import SandboxSpec

_IMAGE_DIGEST_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$", re.IGNORECASE)


class TaskEnvironment(BaseModel):
    """Dataset-owned declaration of the task execution environment.

    Only these fields are accepted from a dataset row. Provider selection,
    credentials, secrets, networking, resources, and provider options remain
    operator-owned.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str | None = None
    image: str | None = None
    workdir: str | None = None

    @field_validator("task_id", "image", "workdir")
    @classmethod
    def _non_empty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("task_environment values must not be empty")
        return value.strip() if value is not None else None

    @field_validator("image")
    @classmethod
    def _immutable_image(cls, value: str | None) -> str | None:
        if value is not None and not _IMAGE_DIGEST_RE.fullmatch(value):
            raise ValueError(
                "task_environment.image must use an immutable OCI digest "
                "(for example registry.example/task@sha256:<64 hex characters>)"
            )
        return value


class ResolvedTaskEnvironment(BaseModel):
    """Auditable result of resolving a dataset task environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str | None = None
    image: str | None = None
    workdir: str | None = None

    def sandbox_spec(self, operator_spec: SandboxSpec | None = None) -> SandboxSpec:
        """Apply the resolved dataset fields to an operator-owned SandboxSpec."""
        base = operator_spec or SandboxSpec()
        metadata = dict(base.metadata)
        if self.task_id is not None:
            metadata["task_id"] = self.task_id
        if self.image is not None:
            metadata["task_image"] = self.image
        return SandboxSpec(
            image=self.image if self.image is not None else base.image,
            ttl_s=base.ttl_s,
            ready_timeout_s=base.ready_timeout_s,
            workdir=self.workdir if self.workdir is not None else base.workdir,
            env=dict(base.env),
            files=dict(base.files),
            metadata=metadata,
            resources=base.resources,
            entrypoint=list(base.entrypoint) if base.entrypoint is not None else None,
            provider_options=dict(base.provider_options),
            ports=tuple(base.ports),
        )


def _manifest_entry(
    task_id: str,
    manifest: Mapping[str, Any],
) -> tuple[str, str | None]:
    try:
        entry = manifest[task_id]
    except KeyError as exc:
        raise ValueError(f"Unknown task_environment.task_id: {task_id!r}") from exc

    if isinstance(entry, str):
        return entry, None
    if not isinstance(entry, Mapping):
        raise ValueError(f"Manifest entry for {task_id!r} must be a string or mapping")

    allowed = {"image", "workdir"}
    unknown = set(entry) - allowed
    if unknown:
        raise ValueError(
            f"Manifest entry for {task_id!r} contains unsupported keys: "
            f"{', '.join(sorted(unknown))}"
        )
    image = entry.get("image")
    if not isinstance(image, str):
        raise ValueError(f"Manifest entry for {task_id!r} must define an image")
    return image, entry.get("workdir")


def resolve_task_environment(
    row: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
    operator_spec: SandboxSpec | None = None,
) -> ResolvedTaskEnvironment:
    """Resolve ``row.task_environment`` into a provider-neutral environment.

    A row may specify an immutable image, a manifest-backed task_id, or both.
    When both are present they must resolve to the same image. Dataset input
    can only affect task_id, image, and workdir.
    """
    raw = row.get("task_environment")
    if raw is None:
        return ResolvedTaskEnvironment(
            image=operator_spec.image if operator_spec else None,
            workdir=operator_spec.workdir if operator_spec else None,
        )
    if not isinstance(raw, Mapping):
        raise ValueError("task_environment must be an object")

    task_env = TaskEnvironment.model_validate(raw)
    manifest_image = None
    manifest_workdir = None

    if task_env.task_id is not None:
        if manifest is None:
            raise ValueError(
                "task_environment.task_id requires a trusted task manifest"
            )
        manifest_image, manifest_workdir = _manifest_entry(task_env.task_id, manifest)
        if not _IMAGE_DIGEST_RE.fullmatch(manifest_image):
            raise ValueError(
                f"Manifest image for {task_env.task_id!r} must be an immutable OCI digest"
            )
        if task_env.image is not None and task_env.image != manifest_image:
            raise ValueError(
                f"task_environment.image does not match the manifest image for "
                f"{task_env.task_id!r}"
            )

    image = task_env.image or manifest_image
    workdir = task_env.workdir or manifest_workdir

    return ResolvedTaskEnvironment(
        task_id=task_env.task_id,
        image=image,
        workdir=workdir,
    )
