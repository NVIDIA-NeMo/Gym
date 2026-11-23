# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fields of every tasks.jsonl row: the native user message, and the three fields reasoning_gym's verify reads."""

from typing import Any, Optional

from pydantic import BaseModel


class TaskData(BaseModel):
    # The native row's user message (question plus answer-format line), which the instruction repeats after the
    # folded system prompt.
    user_message: str
    # Hidden from the agent: only tests/verifier.py reads these three, as the native verify request carries them.
    question: str
    answer: Optional[str]
    metadata: dict[str, Any]
