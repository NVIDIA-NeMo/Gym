# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the ether0 server.

Chemistry reasoning rows whose task fields all live inside ``verifier_metadata`` today (schema
written flat; ``legacy_location`` records that). Required-ness mirrors ``Ether0RunRequest``
(app.py:40): the whole bucket is ``Optional[dict[str, Any]]`` and every read in ``verify()`` is a
defensive ``meta.get(...)`` with a default, so all fields here are Optional. ``solution`` is a
serialized ether0 ``RewardFunctionInfo`` whose inner structure is owned by the external ether0
package — it stays an opaque ``str`` by design. Two task families share the envelope: the base
family (both committed example.jsonl files) and an MCQ family produced by
``scripts/prepare_ether0.py --boxed-letter-format`` that adds the optional ``choices`` map
(code-reachable, absent from committed data) — a single model with optionals, not a union.
"""

from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    solution: Optional[str] = Field(
        default=None,
        description=(
            "Serialized ether0 RewardFunctionInfo ('<fxn_name>!:!<answer_info>!:!<problem_type>'), validated "
            "by RewardFunctionInfo.model_validate inside verify() to pick the eval function and its "
            "arguments. Opaque string owned by the external ether0 package; read as meta.get('solution', '') "
            "and a malformed value scores 0.0 rather than erroring."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    problem_type: Optional[str] = Field(
        default=None,
        description=(
            "Problem family label (e.g. 'functional-group', may contain '/', e.g. "
            "'property-regression-adme/log_mdr1-mdck_er'); read defensively and echoed on the verify "
            "response for reporting."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"], "legacy_location": "verifier_metadata"},
    )
    choices: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "MCQ letter -> choice text map; when present, verify() maps a single-letter model answer back "
            "to the full choice text before scoring. Only emitted by scripts/prepare_ether0.py "
            "--boxed-letter-format; absent from committed example data."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    ideal: Optional[str] = Field(
        default=None,
        description="Human-readable reference answer (e.g. a SMILES string); carried but never read.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    id: Optional[str] = Field(
        default=None,
        description="UUID task identifier; carried but never read.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
