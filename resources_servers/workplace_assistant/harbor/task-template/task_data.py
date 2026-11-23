# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fields of one workplace_assistant task (a row of tasks.jsonl), checked at prepare, at seed and at collate.

``system_prompt`` and ``prompt`` are the native row's system and user messages; the instruction is both, in that
order.
``ground_truth`` is the native row's list of ``{name, arguments}`` calls, ``arguments`` being JSON text; it reaches only
the verifier (tests/ground_truth.json) and the oracle (solution/calls.json), never the agent's sandbox.
"""

from pydantic import BaseModel


class TaskData(BaseModel):
    system_prompt: str
    prompt: str
    ground_truth: list[dict[str, str]]
    id: int
    category: str
    environment_name: str
