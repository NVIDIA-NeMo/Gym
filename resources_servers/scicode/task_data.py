# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the scicode server.

Task fields ride at the row top level; there is no verifier_metadata. Mirrors
``ScicodeRunRequest`` (app.py) minus ``solutions``: that field is populated by the agent at
runtime ({"<problem_id>.<step>": accumulated_code}) and flows run -> verify — it is a verify-time
injected field, not a dataset column, so it stays declared on the request model in app.py only.
Likewise the gold test data (test_data.h5) comes from server config, not the row.

Wire required-ness here covers the whole rollout path, not just the resources server:
``ScicodeRunRequest`` omits ``required_dependencies``, but the wired agent's
``ScicodeAgentRunRequest`` (responses_api_agents/scicode_agent/app.py) declares it ``str`` with no
default and reads it unconditionally, and scicode's config wires that agent — so a row missing the
field would pass a resources-server-only schema yet 422 at the agent /run before any rollout.
It is therefore required below.

``sub_steps`` items stay untyped dicts to mirror the wire's ``List[dict]``; verify() consumes
only ``test_cases`` (list[str]) and ``step_number`` (str) per item, the rest is agent-facing
prompt material.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    problem_id: str = Field(
        description=(
            "SciCode problem identifier (e.g. '10'); verify() keys into the agent-built solutions "
            "dict as '<problem_id>.<step_index + 1>'."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    sub_steps: List[Dict[str, Any]] = Field(
        description=(
            "Ordered sub-problem dicts, each shaped {step_number: str, step_description_prompt: "
            "str, step_background: str, ground_truth_code: str, function_header: str, test_cases: "
            "list[str], return_line: str}. verify() consumes len(sub_steps) plus test_cases and "
            "step_number per item (via sanitize_test/build_test_program); the other subkeys drive "
            "the per-step agent prompts. Wire type is List[dict]."
        ),
        json_schema_extra={"consumed_by": ["verify", "prompt"]},
    )
    problem_name: Optional[str] = Field(
        default=None,
        description="Human-readable problem slug (e.g. 'ewald_summation'); never read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    required_dependencies: str = Field(
        description=(
            "Newline-joined import statements the problem's code depends on (e.g. 'import numpy as "
            "np'); agent-facing prompt material, never read by verify(). Required: the wired "
            "agent's ScicodeAgentRunRequest declares it `str` with no default, so a row missing it "
            "fails the agent /run with a 422 before any rollout."
        ),
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    uuid: Optional[str] = Field(
        default=None,
        description="Row identifier (mirrors problem_id in committed data); never read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
