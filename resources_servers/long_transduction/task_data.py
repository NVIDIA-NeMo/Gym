# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flat task fields for long-transduction datasets.

Mirrors LongTransductionRunRequest's optional fields and permissive extras.
Expressions carry strings, numbers, or UUID lists depending on the task type.
Task fields are accepted at the row top level; no legacy-location markers apply.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """Reference payloads, difficulty knobs, and generation metadata."""

    model_config = ConfigDict(extra="allow")

    type: str | None = Field(default=None, description="Task variant; omitted values use unnumbered_streaming_sum.")
    target_tokens: int | None = Field(default=None, description="Input-token budget used for context-length metrics.")
    max_operands: int | None = None
    n_expressions: int | None = None
    expressions: list[dict[str, object]] | None = Field(
        default=None, description="Per-item reference expressions and answers, with task-dependent value types."
    )
    uuid_lines: list[list[str]] | None = None
    uuids_per_line: int | None = None
    n_lines: int | None = None
    expected_output: str | None = None
    n_rows: int | None = None
    n_cols: int | None = None
    perm_fraction: float | None = None
    vocab_fraction: float | None = None
    n_vocab: int | None = None
    n_variables: int | None = None
    question: str | None = Field(default=None, description="Source prompt before Responses input materialization.")
    approx_prompt_tokens: int | None = None
