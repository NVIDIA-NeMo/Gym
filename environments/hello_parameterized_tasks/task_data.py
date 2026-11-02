# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shape of every row of tasks.jsonl, validated at prepare, at seed and at collate.

A row whose fields do not satisfy this model (a missing key, a number where a string is expected, a key the model
does not declare) refuses the dataset at prepare with the task named; the environment server refuses the episode
and collate refuses the result the same way. Keep it the complete description of a task's fields.
"""

from pydantic import BaseModel


class TaskData(BaseModel):
    question: str
    expected_answer: str
