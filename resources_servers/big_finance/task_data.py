# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Task-data schema for the big_finance server.

Rows from ``benchmarks/big_finance/prepare.py`` carry every task field at the top level (no
``verifier_metadata``). Required-ness mirrors ``BigFinanceRunRequest`` (app.py), where every field
has a default: an unlabeled row is valid for ``reward_mode=passthrough``. ``license`` and
``provenance`` are written by ``prepare.py`` but are not declared on the wire model, so they are
dropped at request validation; they are declared here so tooling sees them.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class RubricLine(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str = Field(description="One analyst step the judge checks the trace for.")
    points: int = Field(default=1, ge=1, description="Weight of this step in rubric_points scoring.")


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(
        default="",
        description="Upstream question id, echoed into the verify response.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    query: str = Field(
        default="",
        description="The question. Shown to both judges; the policy sees it in responses_create_params.input.",
        json_schema_extra={"consumed_by": ["verify", "prompt"]},
    )
    reference_answer: str = Field(
        default="",
        description="Authoritative answer compared with the final answer in every judged reward mode.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    rubric: list[RubricLine] = Field(
        default_factory=list,
        description="Point-weighted analyst steps graded against the trace; empty for answer_match rows.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    evaluation_only: Optional[bool] = Field(
        default=None,
        description="Upstream flag: the item is for evaluation only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    do_not_train: Optional[bool] = Field(
        default=None,
        description="Upstream flag: the item must not be used for training.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    benchmark_canary: Optional[str] = Field(
        default=None,
        description="Upstream canary string for detecting benchmark contamination.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Upstream per-item source references.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    as_of_date: Optional[str] = Field(
        default=None,
        description=(
            "ISO date. With an offline SEC corpus, edgar_search and fetch_url hide filings made after it; "
            "ignored otherwise. Read at seed_session for the tools, not by verify."
        ),
    )
    license: Optional[str] = Field(
        default=None,
        description="Dataset license recorded by prepare.py. Not declared on the wire model.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    provenance: Optional[dict] = Field(
        default=None,
        description=(
            "Pinned source repository, commit, dataset URL, and attribution recorded by prepare.py. "
            "Not declared on the wire model."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
