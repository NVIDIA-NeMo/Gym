# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the agentic_turn_simulation server.

A row is a recorded agent-harness conversation up to the user message the policy answers, carried in
``responses_create_params``. The server reads every task field from ``verifier_metadata`` (an optional free-form dict
on its run request), so each field carries ``legacy_location`` and all are optional: a row without ``harness`` runs on
the generic harness. The fields mirror the ``verifier_metadata`` keys table in ``docs/data.md``. ``extra="allow"``
because rows may carry keys of their own, which ``metric_breakdown_keys`` can break ``turn_score`` down by.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    harness: str | None = Field(
        default=None,
        description="The harness the row was recorded in (sandbox, claude_code, hermes, openclaw, opencode, ...); "
        "selects how tool calls are simulated and the turn is judged. Missing or unknown: generic. Cline rows also "
        "need protocol cline_xml.",
        json_schema_extra={"consumed_by": ["prompt", "verify", "metrics"], "legacy_location": "verifier_metadata"},
    )
    harness_version: str | None = Field(
        default=None,
        description="The harness release (e.g. '2.1.90'), for tool-result formats that changed between versions.",
        json_schema_extra={"consumed_by": ["prompt"], "legacy_location": "verifier_metadata"},
    )
    protocol: Literal["native", "cline_xml"] | None = Field(
        default=None,
        description="How the agent calls tools: function calls (native, the default) or Cline's XML in the reply "
        "text. cline_xml selects the Cline harness.",
        json_schema_extra={"consumed_by": ["prompt", "verify"], "legacy_location": "verifier_metadata"},
    )
    record_kind: str | None = Field(
        default=None,
        description="interactive (a person wrote the request) or subagent (a parent agent delegated it; the judge "
        "reads the requester as that agent).",
        json_schema_extra={"consumed_by": ["verify", "metrics"], "legacy_location": "verifier_metadata"},
    )
    categories: list[str] | None = Field(
        default=None,
        description="Task categories that turn_score_by_category breaks the score down by.",
        json_schema_extra={"consumed_by": ["metrics"], "legacy_location": "verifier_metadata"},
    )
    corrected_turn_item_range: dict[str, list[int]] | None = Field(
        default=None,
        description='{"steerability": [lo, hi]}: the half-open range of input items holding the earlier agent turn '
        "this user message corrects, which the judge sees as <rejected_turn>. Native-protocol rows only.",
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    reference_turn: list[dict[str, Any]] | None = Field(
        default=None,
        description="The recorded reply to this message as Responses items, with its real tool results; judged "
        "instead of the policy's turn when score_source is reference_turn.",
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    obstacle: Literal[False] | dict[str, Any] | None = Field(
        default=None,
        description="Replaces the row's draw when obstacle_injection is on: false never arms the row, "
        '{"kind": "transient" | "blocked_route" | "empty_search"} arms that kind.',
        json_schema_extra={"consumed_by": ["prompt", "metrics"], "legacy_location": "verifier_metadata"},
    )
    row_id: str | None = Field(
        default=None,
        description="The row's id, which the data-build scripts key rows by.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    session_id: str | None = Field(
        default=None,
        description="The recorded session the row was cut from; the data-build scripts group rows by it.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    source: str | None = Field(
        default=None,
        description="Where the row's recording came from ('example' for the committed example rows).",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
