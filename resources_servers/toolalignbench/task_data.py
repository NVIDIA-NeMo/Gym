# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the toolalignbench server.

One row is one **episode**: a domain, a scenario type, a prompt condition, and the four escalating
documents the model is shown as successive user turns. Benchmark rows store the composed system
prompt and document 1 in ``system_prompt`` and ``first_document``; Gym's prompt renderer places
them in ``responses_create_params.input``. Documents 2-4 ride in ``remaining_documents`` because
the Responses API input can only express the first turn.

Ground truth is the taxonomy, not the row: which tools count as misaligned is a property of the
domain, so a row carries no expected answer -- only the ``domain`` the verifier looks up. Fields
ride top-level rather than under ``verifier_metadata`` so the agent can read the ones it needs.
"""

from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[Union[int, str]] = Field(
        default=None,
        description="Episode identifier, '<domain>-<scenario_type>-<prompt_condition>'.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    domain: str = Field(
        description=(
            "Scenario domain, e.g. 'pharmaceutical-distribution'. Selects the tool taxonomy that "
            "verify() grades against, so a row with an unknown domain cannot be scored."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    scenario_type: Optional[str] = Field(
        default=None,
        description=(
            "'wrongdoing' (the documents evidence real misconduct) or 'safe' (the same documents "
            "with the violations inverted). Used to slice pass@k, not to grade."
        ),
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    prompt_condition: Optional[str] = Field(
        default=None,
        description=(
            "Which of the four system-prompt conditions this row bakes in: bold vs. tame agency "
            "crossed with whether a <scratchpad> reasoning step was requested."
        ),
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    tool_names: List[str] = Field(
        default_factory=list,
        description=(
            "Tools offered for this domain, in prompt order. The agent stub-executes these; a call "
            "to anything else gets a 'not found' result."
        ),
        json_schema_extra={"consumed_by": ["agent"]},
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Composed system prompt, materialized into the initial system turn for benchmark rows.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    first_document: Optional[str] = Field(
        default=None,
        description="Document 1, materialized into the initial user turn for benchmark rows.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    remaining_documents: List[str] = Field(
        default_factory=list,
        description=(
            "Documents 2..N, appended by the agent as successive user turns. At rollout time, "
            "document 1 is already in responses_create_params.input."
        ),
        json_schema_extra={"consumed_by": ["agent"]},
    )
