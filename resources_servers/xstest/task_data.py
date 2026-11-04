# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the xstest server.

XSTest over-refusal benchmark graded by an LLM judge. All task fields live inside
``verifier_metadata`` on today's wire (``XSTestRunRequest`` at app.py:55 types the whole bucket
``Optional[dict[str, Any]]``, ``extra="allow"``, with no subkey typing); schemas are written
flat, so every field carries ``legacy_location``. verify() tolerates a missing bucket entirely
(``body.verifier_metadata or {}``) and reads only ``label`` and ``type`` via ``.get`` with
defaults, so all fields stay Optional. ``label`` flips reward polarity: 'safe' rewards
compliance, anything else rewards refusal.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: Optional[str] = Field(
        default=None,
        description=(
            "Safety label ('safe' or 'unsafe'); verify() defaults to 'safe'. Polarity switch: safe rows "
            "reward full compliance, non-safe rows reward refusal."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    type: Optional[str] = Field(
        default=None,
        description=(
            "Prompt-type taxonomy label (e.g. 'homonyms', 'definitions', 'contrast_homonyms'); verify() "
            "defaults to 'unknown' and echoes it as prompt_type in the verify response."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"], "legacy_location": "verifier_metadata"},
    )
    id: Optional[int] = Field(
        default=None,
        description="Source-dataset row id (e.g. 1, 17); unread by verify().",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    focus: Optional[str] = Field(
        default=None,
        description="Keyword the prompt hinges on (e.g. 'kill', 'coke'); unread by verify().",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    note: Optional[str] = Field(
        default=None,
        description="Topic annotation (e.g. 'violence', 'drugs', may be empty); unread by verify().",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
