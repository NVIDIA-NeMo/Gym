# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CAD dataset hooks shared by training frameworks and standalone Gym consumers."""

from typing import Any

from pydantic import BaseModel

from resources_servers.deepcad.dataset.lazy_inputs import materialize_examples as materialize_cad_examples
from resources_servers.deepcad.dataset.source_operation_sampler import (
    SourceOperationSamplingConfig,
    source_operation_dataloader_kwargs,
    validate_source_operation_sampling,
)


class CADTrainingConfig(BaseModel, extra="forbid"):
    source_operation_sampling: SourceOperationSamplingConfig | None = None


def validate_data_config(data_config: dict[str, Any]) -> None:
    """Reject incompatible shuffling or balancing before dataset construction."""
    CADTrainingConfig.model_validate(data_config["gym_dataset_hooks"]["config"])
    validate_source_operation_sampling(data_config)


def dataloader_kwargs(dataset: Any, data_config: dict[str, Any], batch_size: int) -> dict[str, Any]:
    """Return an index sampler; no processed dataset item is read or rendered."""
    return source_operation_dataloader_kwargs(dataset, data_config, batch_size)


def materialize_examples(examples: list[dict[str, Any]], *, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Render only the sampled batch before image processing and rollout repetition."""
    CADTrainingConfig.model_validate(config)
    return materialize_cad_examples(examples)
