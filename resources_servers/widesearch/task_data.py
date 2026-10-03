# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from pydantic import BaseModel, ConfigDict


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    instance_id: str
    query: str
    evaluation: dict[str, Any]
    gold_answer: list[dict[str, Any]]
    language: str
