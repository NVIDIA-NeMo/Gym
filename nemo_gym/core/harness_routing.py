# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark attribution and deterministic harness selection for a prompt group."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


HARNESS_NAMES = ("opencode", "codex", "hermes")
HARNESS_ROUTING_VERSION = "benchmark-harness-v1"
HarnessName = Literal["opencode", "codex", "hermes"]


class HarnessRoutingMetadata(BaseModel):
    """The declared task identity, selected route and actual responding bridge."""

    model_config = ConfigDict(extra="forbid")

    benchmark: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    env: str = Field(min_length=1)
    task_type: Literal["agentic", "non_agentic"]
    routing_version: Literal["benchmark-harness-v1"] = HARNESS_ROUTING_VERSION
    routing_seed: int = Field(default=0, strict=True, ge=0)
    group_index: int | None = Field(default=None, strict=True, ge=0)
    group_id: str | None = Field(default=None, min_length=1)
    selected_harness: HarnessName | None = None
    executed_harness: HarnessName | None = None

    @model_validator(mode="after")
    def consistent_harnesses(self) -> HarnessRoutingMetadata:
        if self.task_type == "non_agentic":
            if self.selected_harness is not None or self.executed_harness is not None:
                raise ValueError("Direct non-agentic tasks cannot name an external harness")
            return self
        if self.group_index is None:
            raise ValueError("Agentic tasks require a prepared within-benchmark group_index")
        expected = HARNESS_NAMES[(self.group_index + self.routing_seed) % len(HARNESS_NAMES)]
        if self.selected_harness is not None and self.selected_harness != expected:
            raise ValueError("Selected harness differs from the prepared prompt-group route")
        if self.executed_harness is not None and self.executed_harness != self.selected_harness:
            raise ValueError("The executed bridge differs from the selected harness")
        return self

    def select(self) -> HarnessRoutingMetadata:
        """Select once from the prepared benchmark ordinal, independent of replicas and retries."""
        if self.task_type == "non_agentic":
            return self.model_copy()
        assert self.group_index is not None
        selected = HARNESS_NAMES[(self.group_index + self.routing_seed) % len(HARNESS_NAMES)]
        return self.model_copy(update={"selected_harness": selected})

    def with_execution(self, executed: str | None) -> HarnessRoutingMetadata:
        """Validate the identity returned by the bridge that handled the activation."""
        return type(self).model_validate(self.model_dump() | {"executed_harness": executed})


def annotate_prompt_groups(
    rows: Sequence[Mapping[str, Any]],
    *,
    sources: Mapping[str, Mapping[str, str]],
    routing_seed: int,
) -> list[dict[str, Any]]:
    """Attach explicit source attribution without replicating or changing task/model inputs."""
    ordinals: Counter[str] = Counter()
    result: list[dict[str, Any]] = []
    for row in rows:
        source = row["task_source"] if "task_source" in row else row["task_id"]["taskset"]
        declared = sources[source]
        benchmark = declared["benchmark"]
        agentic = declared["task_type"] == "agentic"
        route = HarnessRoutingMetadata(
            benchmark=benchmark,
            dataset=declared["dataset"],
            env=declared["env"],
            task_type=declared["task_type"],
            routing_seed=routing_seed,
            group_index=ordinals[benchmark] if agentic else None,
        ).select()
        if agentic:
            ordinals[benchmark] += 1
        copied = dict(row)
        metadata = route.model_dump(mode="json")
        if "harness_routing" in row and row["harness_routing"] != metadata:
            raise ValueError("Prepared source already carries different harness routing")
        copied["harness_routing"] = metadata
        if "task_input" in row:
            task_input = dict(row["task_input"])
            if "harness_routing" in task_input and task_input["harness_routing"] != metadata:
                raise ValueError("Materialized task already carries different harness routing")
            task_input["harness_routing"] = metadata
            copied["task_input"] = task_input
        result.append(copied)
    return result
