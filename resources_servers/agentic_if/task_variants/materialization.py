# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The task input of a sealed row, and its content-digest identity, as Gym's episode routing carries them."""

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.agentic_if.task_variants.builder import digest
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec


class AgenticIFTaskInput(BaseModel):
    """Literal actor input, IF configuration, and the resource-owned fields of the backend task."""

    model_config = ConfigDict(extra="allow")
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    agentic_if: RuntimeSpec


def sealed_task_input(task: Mapping[str, object]) -> object:
    """Return a materialized task's sealed content.

    Gym collation of a flat row (``nemo_gym.task_materialization.materialize_task``) takes the row's
    ``task_id`` as the task identity and keeps that field inside ``task_input``. The sealed content, and
    therefore the content digest, is everything else.
    """
    content = task.get("task_input")
    if isinstance(content, Mapping) and "task_id" in content:
        identity = task.get("task_id")
        if not isinstance(identity, Mapping) or content["task_id"] != identity.get("task_id"):
            raise ValueError("materialized task identity receipt does not match its content")
        return {key: value for key, value in content.items() if key != "task_id"}
    return content


def validate_task_identity(task: Mapping[str, object]) -> None:
    """Content-derived task identities must also hold at the execution boundary."""
    identity = task.get("task_id")
    if isinstance(identity, Mapping) and identity.get("task_id") != digest(sealed_task_input(task)):
        raise ValueError("materialized task identity receipt does not match its content")
