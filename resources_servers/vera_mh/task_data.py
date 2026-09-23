# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the vera_mh server (VERA-MH simulated suicide-risk conversations).

Fields are top-level row columns written by ``benchmarks/vera_mh/prepare.py``: one row per
(persona, user simulator). The row carries the rendered persona system prompt that the user
simulator plays and the persona sheet fields as slices; the chatbot under evaluation only ever
sees ``responses_create_params.input`` (upstream's generic assistant system prompt). Required-ness
mirrors ``VeraMHAgentRunRequest``: only ``persona_system_prompt`` and ``user_simulator`` are
required on the wire.
"""

from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    persona_system_prompt: str = Field(
        description=(
            "The persona's system prompt, rendered from upstream's `persona_prompt_template.txt` with the "
            "persona sheet row exactly as `load_prompts_from_csv` does. Given to the user simulator, never "
            "to the model under evaluation."
        ),
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    user_simulator: str = Field(
        description=(
            "Key of the user simulator that plays this persona: `gpt-5.2` or `claude-opus-4-5`. Selects the "
            "model server from the agent's `user_model_servers`; compute_metrics slices on it."
        ),
        json_schema_extra={"consumed_by": ["prompt", "metrics"]},
    )
    id: Optional[str] = Field(
        default=None,
        description="`vera_mh__<persona name>__<user simulator>`; unique across the 200 rows.",
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    persona_index: Optional[int] = Field(
        default=None,
        description="1-based row number of the persona in upstream's `data/personas.tsv`.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    persona_name: Optional[str] = Field(
        default=None,
        description="The persona's `Name` column, used in the upstream transcript file names.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    persona: Optional[Dict[str, str]] = Field(
        default=None,
        description="The persona sheet row as published (all columns of `personas.tsv`).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    persona_sha256: Optional[str] = Field(
        default=None,
        description="SHA-256 of the tab-joined persona row, so a rollout names the persona text it ran.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    user_simulator_upstream_model: Optional[str] = Field(
        default=None,
        description=(
            "The model id upstream's recommended pipeline uses for this simulator (`gpt-5.2`, "
            "`claude-opus-4-5-20251101`); informational."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    risk_level: Optional[str] = Field(
        default=None,
        description=(
            "The persona's `Short Current Suicide Risk Level`: None, Low, High or Immediate. Drives the "
            "per-risk-level metrics, as upstream's `score_results_by_risk` does."
        ),
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    disclosure_level: Optional[str] = Field(
        default=None,
        description="The persona's `Disclosure of Suicide Risk` level, without its parenthetical description.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    max_turns: Optional[int] = Field(
        default=None,
        description="Turn budget for the conversation (30 in the recommended profile); the agent rounds it so the chatbot speaks last.",
        json_schema_extra={"consumed_by": ["prompt", "metrics"]},
    )
    upstream: Optional[Dict[str, str]] = Field(
        default=None,
        description="Pinned upstream repository, revision, rubric tag, persona sheet hash and license, so a rollout names the data it scored.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
