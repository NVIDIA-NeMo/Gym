# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the self-contained agentdojo_agent (no resources server).

A row selects one official AgentDojo execution. The suite and user task say which task to run;
adding an injection task and an attack turns that same task into its attacked counterpart, which
is why the clean and attacked conditions share a schema rather than being two row types.

This is the official `ethz-spylab/agentdojo` backend, so the suites are the original four.
AgentDyn's suites (`shopping`, `github`, `dailylife`) come from a different upstream that installs
the same top-level `agentdojo` package, so they are not accepted here.

Required-ness mirrors `AgentDojoRunRequest` in `app.py`. Only `suite` and `user_task_id` are
required there; everything else has a wire default.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    # `extra="allow"` matches the run request: rows may carry fields the agent does not read.
    model_config = ConfigDict(extra="allow")

    suite: Literal["banking", "slack", "travel", "workspace"] = Field(
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    user_task_id: str = Field(
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    # None marks the clean arm of the matrix; a string names the injection task to run alongside
    # the user task. The adapter rejects this without `attack`, and `attack` without this.
    injection_task_id: Optional[str] = Field(
        default=None,
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    attack: Optional[str] = Field(
        default=None,
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    # `agentdojo.agent_pipeline.agent_pipeline.DEFENSES` at the pinned v0.1.35 commit; None runs
    # the undefended pipeline. The AgentDyn-only defenses are not registered by this upstream.
    defense: Optional[
        Literal["tool_filter", "transformers_pi_detector", "spotlighting_with_delimiting", "repeat_user_prompt"]
    ] = Field(
        default=None,
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    benchmark_version: Optional[str] = Field(
        default=None,
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    # Written by `benchmarks/agentdojo/prepare.py` so a materialized file can be split by
    # condition without re-deriving it; the agent itself keys on `injection_task_id`.
    condition: Optional[Literal["clean", "attacked"]] = Field(
        default=None,
        json_schema_extra={"consumed_by": ["provenance"]},
    )
