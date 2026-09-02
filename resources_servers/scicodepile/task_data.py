# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the scicodepile server.

Rows carry task data inside the legacy ``verifier_metadata`` bucket (the wire model
types it only as ``Optional[Dict[str, Any]]``); the schema is written flat with
``legacy_location`` annotations per the protocol. Required-ness follows verify()'s
hard reads: ``test`` and ``entry_point`` are indexed with ``meta[...]`` (KeyError if
absent), while everything else is read via ``.get`` or kept for provenance only.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: Optional[str] = Field(
        default=None,
        description="SciCodePile task identifier, e.g. 'alignment/python/1'; echoed into the verify response.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    test: str = Field(
        description=(
            "Python source defining ``check(candidate)``. Executed in the same namespace as the model's "
            "code; ``check`` is then called with the function named by ``entry_point``."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    entry_point: str = Field(
        description="Name of the function under test; must be defined by the model's own code.",
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    setup_code: str = Field(
        default="",
        description=(
            "Optional preamble executed before the model's code (imports, stubs, fixtures). Present on "
            "105 of the 200 released tasks."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    audit_flags: Optional[list[str]] = Field(
        default=None,
        description=(
            "Upstream stability annotations, e.g. 'env_sensitive' (all 200 tasks), 'globals_patch' (117), "
            "'internal_state_check' (40). Provenance only; verify() does not branch on them."
        ),
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    primary_score_eligible: Optional[bool] = Field(
        default=None,
        description="Upstream flag marking a task as countable toward the headline score. Provenance only.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
