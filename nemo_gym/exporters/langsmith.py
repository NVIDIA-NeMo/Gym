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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar, Optional
from uuid import NAMESPACE_URL, UUID, uuid5

from langsmith import Client
from langsmith.utils import LangSmithConflictError
from omegaconf import DictConfig

from nemo_gym.config_types import LangSmithConfig
from nemo_gym.exporters.base import BaseExporter


_LANGSMITH_RUNNER_METADATA = {
    "ls_runner": "nemo-gym",
    "source": "nemo-gym",
}

_ROLLOUT_PAYLOAD_KEYS = frozenset(
    {
        "responses_create_params",
        "response",
        "reward",
    }
)


@dataclass(frozen=True)
class _MappedRollout:
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    metadata: dict[str, Any]
    reward: float


def _map_rollout(rollout: dict[str, Any]) -> _MappedRollout:
    metadata = {key: value for key, value in rollout.items() if key not in _ROLLOUT_PAYLOAD_KEYS}
    metadata.update(_LANGSMITH_RUNNER_METADATA)

    return _MappedRollout(
        inputs=rollout["responses_create_params"],
        outputs=rollout["response"],
        metadata=metadata,
        reward=float(rollout["reward"]),
    )


def _stable_uuid(*parts: Any) -> UUID:
    normalized = ":".join(str(part) for part in parts)
    return uuid5(
        NAMESPACE_URL,
        f"nemo-gym-langsmith:{normalized}",
    )


def _example_metadata(mapped: _MappedRollout) -> dict[str, Any]:
    metadata = {key: mapped.metadata[key] for key in ("_ng_task_index", "task_source") if key in mapped.metadata}
    metadata.update(_LANGSMITH_RUNNER_METADATA)
    return metadata


class LangSmithExporter(BaseExporter):
    """Export Gym evaluation results to LangSmith."""

    name: ClassVar[str] = "langsmith"

    def __init__(self, global_config_dict: DictConfig) -> None:
        super().__init__(global_config_dict)
        self.config = LangSmithConfig.model_validate(global_config_dict)
        self.client: Optional[Client] = None
        self.dataset_id: Optional[str] = None
        self.experiment_id: Optional[str] = None

    def setup(self) -> None:
        self.client = Client(
            api_url=self.config.langsmith_endpoint,
            api_key=self.config.langsmith_api_key,
            workspace_id=self.config.langsmith_workspace_id,
        )
        dataset_name = self.config.langsmith_dataset_name

        if self.client.has_dataset(dataset_name=dataset_name):
            dataset = self.client.read_dataset(dataset_name=dataset_name)
        else:
            dataset = self.client.create_dataset(
                dataset_name,
                description="Evaluation dataset exported by NeMo Gym.",
                metadata=_LANGSMITH_RUNNER_METADATA,
            )

        self.dataset_id = str(dataset.id)

        # LangSmith's SDK calls evaluation experiments projects.
        experiment = self.client.create_project(
            self.config.langsmith_experiment_name,
            upsert=True,
            reference_dataset_id=self.dataset_id,
            metadata=_LANGSMITH_RUNNER_METADATA,
        )
        self.experiment_id = str(experiment.id)

    def teardown(self) -> None:
        if self.client is None:
            return

        client = self.client

        try:
            if self.experiment_id is not None:
                client.update_project(
                    self.experiment_id,
                    end_time=datetime.now(timezone.utc),
                )
        finally:
            try:
                client.close()
            finally:
                self.client = None
                self.dataset_id = None
                self.experiment_id = None

    def _log_config(self, _config_dict: DictConfig) -> None:
        pass

    def _log_metrics(self, _metrics: dict[str, Any], _step: Optional[int] = None) -> None:
        pass

    def _log_rollouts(self, rollouts: list[dict[str, Any]]) -> None:
        if self.client is None or self.dataset_id is None or self.experiment_id is None:
            raise RuntimeError("LangSmith exporter is not set up")

        mapped_rollouts = [_map_rollout(rollout) for rollout in rollouts]
        examples_by_task: dict[tuple[Any, Any], _MappedRollout] = {}

        for mapped in mapped_rollouts:
            task_key = (
                mapped.metadata.get("task_source"),
                mapped.metadata["_ng_task_index"],
            )
            examples_by_task.setdefault(task_key, mapped)

        examples: list[dict[str, Any]] = []
        example_ids: dict[tuple[Any, Any], UUID] = {}

        for task_key, mapped in examples_by_task.items():
            example_id = _stable_uuid(
                self.dataset_id,
                "example",
                *task_key,
            )
            example_ids[task_key] = example_id
            examples.append(
                {
                    "id": example_id,
                    "inputs": mapped.inputs,
                    "metadata": _example_metadata(mapped),
                }
            )

        if examples:
            self.client.create_examples(
                dataset_id=self.dataset_id,
                examples=examples,
            )

        for mapped in mapped_rollouts:
            task_source = mapped.metadata.get("task_source")
            task_index = mapped.metadata["_ng_task_index"]
            rollout_index = mapped.metadata["_ng_rollout_index"]
            task_key = (task_source, task_index)

            agent_ref = mapped.metadata.get("agent_ref")
            agent_name = agent_ref.get("name") if isinstance(agent_ref, dict) else agent_ref

            run_id = _stable_uuid(
                self.experiment_id,
                "run",
                task_source,
                task_index,
                rollout_index,
                mapped.metadata.get("_ng_attempt_index"),
                mapped.metadata.get("_ng_rollout_id"),
                agent_name,
            )
            timestamp = datetime.now(timezone.utc)

            self.client.create_run(
                name=f"task-{task_index}-rollout-{rollout_index}",
                run_type="chain",
                inputs=mapped.inputs,
                outputs=mapped.outputs,
                project_name=self.config.langsmith_experiment_name,
                reference_example_id=example_ids[task_key],
                id=run_id,
                trace_id=run_id,
                tags=["nemo-gym"],
                extra={"metadata": mapped.metadata},
                start_time=timestamp,
                end_time=timestamp,
            )

            feedback_id = _stable_uuid(
                run_id,
                "feedback",
                "reward",
            )
            try:
                self.client.create_feedback(
                    key="reward",
                    score=mapped.reward,
                    run_id=run_id,
                    trace_id=run_id,
                    session_id=self.experiment_id,
                    feedback_id=_stable_uuid(
                        run_id,
                        "feedback",
                        "reward",
                    ),
                    source_info=_LANGSMITH_RUNNER_METADATA,
                )
            except LangSmithConflictError:
                self.client.update_feedback(
                    feedback_id,
                    score=mapped.reward,
                )
