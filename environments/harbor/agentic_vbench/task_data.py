# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The typed task field of every Agentic-VBench task: its family (``task_data.json``, written by the prepare script)."""

from typing import Literal

from pydantic import BaseModel


class TaskData(BaseModel):
    family: Literal["repair", "assembly", "sequencing", "repurpose"]
