# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the tavily_search server (parent of the search-QA pair).

Two required top-level strings, mirroring ``TavilySearchRunRequest`` (app.py) exactly; there is
no verifier_metadata. Whether verification runs an LLM judge or an exact string match is chosen
by server config (``use_judge``), not by row data — both paths consume the same two fields. When
``use_judge=false``, verify() extracts the span between "Answer:" and "Confidence:" in the last
assistant message (a fixed extraction regex; ground_truth itself is never a regex) and requires
exact string equality with ground_truth.
browsecomp_advanced_harness shares this schema verbatim (its app.py redeclares the identical
request model) and imports it.
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
