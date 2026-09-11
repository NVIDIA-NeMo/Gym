# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the browsecomp_advanced_harness server.

Two required top-level strings, mirroring ``TavilySearchRunRequest`` as redeclared in this
server's app.py. Rows additionally carry tools inside ``responses_create_params``
(framework-owned, not typed here). verify() branches on server config.use_judge — LLM judge
against ground_truth, or (when use_judge=false) exact string equality between ground_truth and
the span extracted from the last assistant message by a fixed "Answer: ... Confidence:" pattern;
ground_truth is never interpreted as a regex.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    question: str = Field(
        description="The search question posed to the agent; also fed to the LLM judge prompt.",
        json_schema_extra={"consumed_by": ["verify", "prompt"]},
    )
    ground_truth: str = Field(
        description=(
            "Reference answer. Judge target when config.use_judge=true; otherwise compared for "
            "exact string equality against the span extracted (by a fixed 'Answer: ... Confidence:' "
            "pattern) from the last assistant message — never interpreted as a regex."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
