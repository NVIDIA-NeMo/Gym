# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the opencode_search server.

Two required top-level strings, mirroring ``SearchRunRequest`` (app.py) exactly; there is no
verifier_metadata. ``question`` and ``ground_truth`` fill the BrowseComp judge prompt, a fixed template
imported from browsecomp_advanced_harness/judge_prompt.py; only the judge model comes from server
config. There is no exact-match path: a row whose final answer is empty scores 0.0 without a judge
call, and every other row is graded by the judge.

The agent builds its run from ``responses_create_params`` (framework-owned, not typed here): the
user message, plus ``temperature``/``top_p`` when the row sets them. Other top-level columns pass
through (``extra="allow"``, matching the request model); upstream BrowseComp rows have none.

``id`` is declared because the committed example rows carry it. Undeclared keys make a validation
report unclean, so without it both the committed-rows drift gate (tests/unit_tests/data/test_task_data.py)
and ``gym dataset collate +mode=example_validation`` fail.
"""

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[Union[int, str]] = Field(
        default=None,
        description="Ride-along row identifier; never read by verify() or metrics.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    question: str = Field(
        description=(
            "The research question, interpolated into the LLM-judge prompt. The agent itself sees the "
            "question through the user message in responses_create_params, not this field."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    ground_truth: str = Field(
        description=(
            "Reference answer the LLM judge compares the agent's extracted final answer against; "
            "also echoed on the verify response."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
