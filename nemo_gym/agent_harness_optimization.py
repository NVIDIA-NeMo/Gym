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
"""Reference implementation of the agent-harness optimization seam.

The seam deliberately preserves NeMo Gym's native contracts:

* input examples are ordinary rollout dataset rows;
* the evaluator dispatches prepared rows to Agent ``/run`` through
  ``RolloutCollectionHelper.run_examples``; and
* results are native ``/run`` response dictionaries augmented with collection
  identity and candidate provenance.

Candidate types are adapter-specific. The shared seam is generic over the
candidate type so prompts, skills, and other harness-config optimization targets can
reuse their native representations.
"""

from __future__ import annotations

import hashlib
from asyncio import Semaphore
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.config_types import BaseServerConfig
from nemo_gym.global_config import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.prompt import PromptConfig, apply_prompt_to_row
from nemo_gym.rollout_collection import RolloutCollectionHelper


AGENT_MODULE_REFS_KEY_NAME = "agent_module_refs"

AgentModuleType = Literal["working_memory", "long_term_memory", "skill_library"]
Cohort = Literal["train", "validation"]
RolloutRow = dict[str, Any]


def canonicalize_system_prompt(content: str) -> str:
    """Return canonical system-prompt content."""

    return content.replace("\r\n", "\n").rstrip()


def system_prompt_sha256(content: str) -> str:
    return hashlib.sha256(canonicalize_system_prompt(content).encode()).hexdigest()


class SystemPromptCandidate(BaseModel):
    """Candidate type owned by the system-prompt config adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    system: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_system(
        cls,
        *,
        system: str,
        parent_sha256: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SystemPromptCandidate:
        canonical = canonicalize_system_prompt(system)
        return cls(
            system=canonical,
            sha256=system_prompt_sha256(canonical),
            parent_sha256=parent_sha256,
            metadata=metadata or {},
        )

    @model_validator(mode="after")
    def validate_content_digest(self) -> SystemPromptCandidate:
        expected = system_prompt_sha256(self.system)
        if self.sha256 != expected:
            raise ValueError(f"sha256 does not match canonical system prompt: expected {expected}")
        return self


class AgentModuleRef(BaseModel):
    """Provenance for one behavior-shaping artifact active during a rollout."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    type: AgentModuleType
    name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    path: str | None = None
    uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HarnessConfigAdapter[CandidateT, ArtifactT](Protocol):
    """Apply and freeze one declared harness-config optimization target."""

    def apply(self, candidate: CandidateT, source_row: Mapping[str, Any]) -> RolloutRow: ...

    def freeze(self, candidate: CandidateT) -> ArtifactT: ...


class HarnessOptimizer[CandidateT](Protocol):
    """Optimize one candidate type against a fixed Gym evaluator."""

    async def optimize(
        self,
        initial_candidate: CandidateT,
        evaluator: HarnessEvaluator[CandidateT, Any],
    ) -> CandidateT: ...


class SystemPromptConfigAdapter:
    """Replace the system field of a fixed PromptConfig for each source row."""

    def __init__(
        self,
        *,
        module_name: str,
        base_prompt: PromptConfig,
    ) -> None:
        self.module_name = module_name
        self.base_prompt = base_prompt

    def apply(self, candidate: SystemPromptCandidate, source_row: Mapping[str, Any]) -> RolloutRow:
        prompt = PromptConfig(user=self.base_prompt.user, system=candidate.system)
        row = apply_prompt_to_row(dict(source_row), prompt)
        candidate_ref = AgentModuleRef(
            type="working_memory",
            name=self.module_name,
            sha256=candidate.sha256,
        )
        row[AGENT_MODULE_REFS_KEY_NAME] = [
            candidate_ref.model_dump(mode="json"),
            *row.get(AGENT_MODULE_REFS_KEY_NAME, []),
        ]
        return row

    def freeze(self, candidate: SystemPromptCandidate) -> PromptConfig:
        """Return a native PromptConfig containing the selected candidate."""

        return PromptConfig(user=self.base_prompt.user, system=candidate.system)


class HarnessEvaluator[CandidateT, ArtifactT]:
    """Evaluate candidates against fixed Gym cohorts and freeze the winner."""

    def __init__(
        self,
        *,
        train_rows: Sequence[Mapping[str, Any]],
        config_adapter: HarnessConfigAdapter[CandidateT, ArtifactT],
        validation_rows: Sequence[Mapping[str, Any]] | None = None,
        num_repeats: int = 1,
        agent_name: str | None = None,
        head_server_config: BaseServerConfig | None = None,
        max_concurrency: int | None = None,
        helper: RolloutCollectionHelper | None = None,
    ) -> None:
        if not train_rows:
            raise ValueError("at least one training row is required")
        if validation_rows is not None and not validation_rows:
            raise ValueError("validation_rows cannot be empty when provided")
        if num_repeats < 1:
            raise ValueError("num_repeats must be at least 1")
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")

        self._cohorts: dict[Cohort, tuple[RolloutRow, ...]] = {
            "train": tuple(deepcopy(dict(row)) for row in train_rows),
        }
        if validation_rows is not None:
            self._cohorts["validation"] = tuple(deepcopy(dict(row)) for row in validation_rows)
        self.config_adapter = config_adapter
        self.num_repeats = num_repeats
        self.agent_name = agent_name
        self.head_server_config = head_server_config
        self.max_concurrency = max_concurrency
        self.helper = helper or RolloutCollectionHelper()

    def examples(self, cohort: Cohort = "train") -> tuple[RolloutRow, ...]:
        """Return copies of the fixed native rows available to an optimizer."""

        return tuple(deepcopy(row) for row in self._cohort_rows(cohort))

    def freeze(self, candidate: CandidateT) -> ArtifactT:
        """Freeze a selected candidate into its native target artifact."""

        return self.config_adapter.freeze(deepcopy(candidate))

    async def evaluate(
        self,
        candidate: CandidateT,
        *,
        cohort: Cohort = "train",
        example_indices: Sequence[int] | None = None,
    ) -> list[RolloutRow]:
        candidate = deepcopy(candidate)
        selected_rows = self._select_rows(cohort, example_indices)
        materialized: list[RolloutRow] = []
        for task_index, source_row in selected_rows:
            candidate_row = self.config_adapter.apply(candidate, source_row)
            for rollout_index in range(self.num_repeats):
                row = deepcopy(candidate_row)
                if self.agent_name is not None:
                    row.setdefault(AGENT_REF_KEY_NAME, {"name": self.agent_name})
                agent_ref = row.get(AGENT_REF_KEY_NAME)
                if not isinstance(agent_ref, dict) or not agent_ref.get("name"):
                    raise ValueError(f"row {task_index} requires agent_ref or evaluator agent_name")
                row[TASK_INDEX_KEY_NAME] = task_index
                row[ROLLOUT_INDEX_KEY_NAME] = rollout_index
                materialized.append(row)

        identities = [(row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME]) for row in materialized]
        semaphore = Semaphore(self.max_concurrency) if self.max_concurrency is not None else None
        results: dict[tuple[int, int], RolloutRow] = {}
        for future in self.helper.run_examples(
            materialized,
            head_server_config=self.head_server_config,
            semaphore=semaphore,
        ):
            row, raw_result = await future
            identity = (row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME])
            if identity in results:
                raise ValueError(f"rollout evaluator returned duplicate identity: {identity}")

            result = deepcopy(raw_result)
            result[TASK_INDEX_KEY_NAME] = identity[0]
            result[ROLLOUT_INDEX_KEY_NAME] = identity[1]
            result[AGENT_REF_KEY_NAME] = row[AGENT_REF_KEY_NAME]
            if AGENT_MODULE_REFS_KEY_NAME in row:
                result[AGENT_MODULE_REFS_KEY_NAME] = row[AGENT_MODULE_REFS_KEY_NAME]
            results[identity] = result

        missing = set(identities) - results.keys()
        if missing:
            raise ValueError(f"rollout evaluator returned no result for identities: {sorted(missing)}")
        return [results[identity] for identity in identities]

    def _cohort_rows(self, cohort: Cohort) -> tuple[RolloutRow, ...]:
        try:
            return self._cohorts[cohort]
        except KeyError:
            raise ValueError(f"{cohort} cohort is not configured") from None

    def _select_rows(
        self,
        cohort: Cohort,
        example_indices: Sequence[int] | None,
    ) -> list[tuple[int, RolloutRow]]:
        rows = self._cohort_rows(cohort)
        indices = list(range(len(rows))) if example_indices is None else list(example_indices)
        if len(indices) != len(set(indices)):
            raise ValueError("example_indices must not contain duplicates")
        invalid = [index for index in indices if index < 0 or index >= len(rows)]
        if invalid:
            raise ValueError(f"example_indices out of range for {cohort} cohort: {invalid}")
        if not indices:
            raise ValueError("at least one example index is required")
        return [(index, deepcopy(rows[index])) for index in indices]
