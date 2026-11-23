# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the equivalence_rule server.

Required-expected_answer QA family heir: extends hotpotqa_qa's ``ExpectedAnswerTaskDataCore``.
``expected_answer`` is the only field ``verify()`` consumes (graded by the config-selected rule:
exact / seq_match / weighted_seq_match on normalized strings); question/dataset/source/hashid/
context_length are long-context (mrcr-style) provenance passthrough surviving via
``extra="allow"``. Required-ness mirrors ``EquivalenceRuleRunRequest`` (app.py):
``expected_answer`` is the sole wire-declared field.
"""

from typing import Optional, Union

from pydantic import Field

from resources_servers.hotpotqa_qa.task_data import ExpectedAnswerTaskDataCore


class TaskData(ExpectedAnswerTaskDataCore):
    question: Optional[str] = Field(
        default=None,
        description="Task instruction/question text; provenance only (the prompt lives in the input messages).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    dataset: Optional[str] = Field(
        default=None,
        description="Source dataset slice label (e.g. 'mrcr.sdg_ns_out+S.0-8k'); provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    source: Optional[str] = Field(
        default=None,
        description="Source benchmark name (e.g. 'mrcr'); provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    hashid: Optional[str] = Field(
        default=None,
        description="Stable per-task hash identifier; provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    context_length: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "Context length of the task input; a stringified integer (e.g. '1562') in committed data, "
            "typed open to int for other feeds. Provenance only."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
