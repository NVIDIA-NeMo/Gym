# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PinchBench's prepared rows. The pinned task definition owns the prompts and fixtures."""

from pydantic import BaseModel, Field

from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming


class TaskSelector(BaseModel):
    task_id: str = Field(pattern=r"^task_[a-z0-9_]+$")


class TaskInput(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    verifier_metadata: TaskSelector
