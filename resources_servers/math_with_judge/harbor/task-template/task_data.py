# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fields of every tasks.jsonl row: the two fields math_with_judge's verify request carries."""

from pydantic import BaseModel


class TaskData(BaseModel):
    # The native row's user message, which the instruction repeats after the folded system prompt.
    question: str
    # Hidden from the agent: only tests/verifier.py reads it.
    expected_answer: str
