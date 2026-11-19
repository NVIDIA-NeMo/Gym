# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SWE-Together's task input: the ``task_input`` its episode request validates.

Data tooling loads this module without the server's dependencies.
So it imports only the standard library, pydantic, and ``nemo_gym`` packages at or below ``nemo_gym.data`` (see ``nemo_gym/data/task_input.py``).
"""

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming


SOURCE_REVISION = "891d19eb4b3a64a47c3d49bbd066a311e0133254"  # pragma: allowlist secret


class TaskData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    image: str
    image_digest: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    workdir: str = Field(default="/workspace", pattern=r"^/")
    source_revision: str = SOURCE_REVISION


class TaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_data: TaskData
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming | None = None
