# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-class Task model for the ``swe_bench`` Environment.

A **Task** (τ) is one problem instance from a benchmark's task distribution — not the
Environment (``swe_bench`` resources server) and not the published benchmark name alone
(e.g. *SWE-bench Verified*).

Terminology:

* ``task_id`` / ``instance_id`` — unique instance key (``django__django-13741``)
* ``dataset_name`` — published benchmark product (HuggingFace id)
* ``harness_family`` / ``benchmark`` — harness registry key inside this Environment
  (``swe-bench``, ``r2e-gym``, …)
* ``problem_statement`` — initial observation (user message) for the agent
* ``metadata`` — privileged grading fields (``instance_dict``, etc.); Environment-only
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


ENVIRONMENT_NAME = "swe_bench"

_HARNESS_FAMILY_ALIASES: list[tuple[str, str]] = [
    ("R2E-Gym", "r2e-gym"),
    ("SWE-bench_Multilingual", "swe-bench-multilingual"),
    ("SWE-bench", "swe-bench"),
]


class TaskRunBody(Protocol):
    """Minimal run/seed/verify request shape carrying task fields."""

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming | None
    verifier_metadata: dict[str, Any] | None


class TaskPublic(BaseModel):
    """Agent-visible task identity returned from ``seed_session``."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    environment: str = ENVIRONMENT_NAME
    dataset_name: str = ""
    harness_family: str = ""
    split: str = "test"


class TaskSubmission(BaseModel):
    """Agent-produced artifact graded at ``verify`` (Environment-owned scoring)."""

    model_config = ConfigDict(extra="forbid")

    model_patch: str = ""


@dataclass
class SweTask:
    """One SWE Environment task instance — provisioning + grading input.

    This is the Environment-internal task value. Harnesses consume ``SweTask``;
    HTTP callers supply dataset rows that parse into this type.
    """

    instance_id: str
    image: str | None = None
    base_commit: str | None = None
    repo_workdir: str = "/testbed"
    test_command: str = ""
    test_framework: str = ""
    model_patch: str = ""
    test_patch: str = ""
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    benchmark: str = "swe-bench-ext"
    split: str = "test"
    dataset_name: str = ""
    problem_statement: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        return self.instance_id

    @property
    def harness_family(self) -> str:
        return self.benchmark

    def public_view(self, *, environment: str = ENVIRONMENT_NAME) -> TaskPublic:
        """Return the agent-visible task identity (no privileged metadata)."""
        return TaskPublic(
            task_id=self.task_id,
            environment=environment,
            dataset_name=self.dataset_name,
            harness_family=self.harness_family,
            split=self.split,
        )

    def privileged_verifier_metadata(self, *, flat_eval: bool) -> dict[str, Any]:
        """Privileged fields the Environment needs on verify (not for agent logic)."""
        return {
            "instance_id": self.instance_id,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "benchmark": self.benchmark,
            "harness_family": self.harness_family,
            "problem_statement": self.problem_statement,
            "flat_eval": flat_eval,
            "instance_dict": self.metadata.get("instance_dict"),
        }

    def with_submission(self, submission: TaskSubmission | None) -> SweTask:
        """Return a copy with the agent's graded submission applied."""
        patch = (submission.model_patch if submission else "") or ""
        return replace(self, model_patch=patch)


def harness_family_key(dataset_name: str) -> str:
    """Map a HuggingFace dataset name to a harness registry key."""
    for needle, key in _HARNESS_FAMILY_ALIASES:
        if needle in dataset_name:
            return key
    return "swe-bench"


def instance_image(container_formatter: Any, instance_id: str) -> str:
    fmt = container_formatter[0] if isinstance(container_formatter, list) else container_formatter
    fmt = fmt or "swebench/sweb.eval.x86_64.{instance_id}"
    if fmt.endswith(".sif") or fmt.startswith(("/", ".")):
        return fmt.format(instance_id=instance_id)
    if fmt.startswith("docker://"):
        fmt = fmt[len("docker://") :]
    tag = instance_id.replace("__", "_1776_").lower()
    image = fmt.format(instance_id=tag)
    if ":" not in image.rsplit("/", 1)[-1]:
        image += ":latest"
    return image


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            return list(json.loads(value))
        except (json.JSONDecodeError, TypeError):
            return [value] if value else []
    return list(value or [])


def merge_row_metadata(
    verifier_metadata: dict[str, Any] | None,
    responses_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge dataset row fields from verifier and responses metadata."""
    return _merge_row_metadata(verifier_metadata, responses_metadata)


def _merge_row_metadata(
    verifier_metadata: dict[str, Any] | None,
    responses_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if responses_metadata:
        info.update(responses_metadata)
    if verifier_metadata:
        info.update(verifier_metadata)
    return info


def _initial_observation(row: dict[str, Any], responses_metadata: dict[str, Any] | None) -> str:
    if row.get("problem_statement"):
        return str(row["problem_statement"])
    params = row.get("responses_create_params")
    if isinstance(params, dict):
        raw_input = params.get("input")
    elif responses_metadata is not None:
        raw_input = None
    else:
        raw_input = None
    if raw_input is None and hasattr(row.get("responses_create_params"), "input"):
        raw_input = row["responses_create_params"].input  # type: ignore[union-attr]
    if isinstance(raw_input, str):
        return raw_input
    if isinstance(raw_input, list) and raw_input:
        first = raw_input[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    return ""


def build_task(
    row: dict[str, Any],
    *,
    container_formatter: str,
    flat_eval: bool = True,
    responses_metadata: dict[str, Any] | None = None,
) -> SweTask:
    """Build a ``SweTask`` from merged dataset / verifier metadata."""
    inst_raw = row.get("instance_dict")
    inst = json.loads(inst_raw) if isinstance(inst_raw, str) else dict(inst_raw or {})
    dataset_name = str(row.get("dataset_name", ""))
    instance_id = row["instance_id"]
    image = instance_image(row.get("container_formatter") or container_formatter, instance_id)

    return SweTask(
        instance_id=instance_id,
        image=image,
        base_commit=inst.get("base_commit"),
        repo_workdir="/testbed",
        test_patch=inst.get("test_patch", ""),
        fail_to_pass=_as_list(inst.get("FAIL_TO_PASS") or inst.get("fail_to_pass")),
        pass_to_pass=_as_list(inst.get("PASS_TO_PASS") or inst.get("pass_to_pass")),
        benchmark=harness_family_key(dataset_name),
        split=str(row.get("split", "test")),
        dataset_name=dataset_name,
        problem_statement=_initial_observation(row, responses_metadata),
        metadata={"instance_dict": inst, "flat_eval": flat_eval, "dataset_name": dataset_name},
    )


def parse_task_from_request(
    body: TaskRunBody,
    *,
    container_formatter: str,
    flat_eval: bool = True,
    environment: str = ENVIRONMENT_NAME,
) -> SweTask:
    """Parse a first-class Task from an agent ``/run`` or Environment HTTP body."""
    responses_metadata = (body.responses_create_params.metadata or {}) if body.responses_create_params else {}
    row = merge_row_metadata(body.verifier_metadata, responses_metadata)
    if "instance_id" not in row:
        raise ValueError(
            "Task requires verifier_metadata.instance_id (or responses_create_params.metadata.instance_id)"
        )
    return build_task(
        row,
        container_formatter=container_formatter,
        flat_eval=flat_eval,
        responses_metadata=responses_metadata,
    )


def parse_submission(verifier_metadata: dict[str, Any] | None) -> TaskSubmission:
    """Extract the agent submission from verify request metadata."""
    meta = dict(verifier_metadata or {})
    patch = meta.get("model_patch") or meta.get("git_patch") or ""
    return TaskSubmission(model_patch=patch if isinstance(patch, str) else str(patch))
