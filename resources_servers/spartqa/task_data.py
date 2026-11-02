# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the spartqa server.

Mirrors ``SpartqaRunRequest`` (app.py): both fields are wire-optional with defaults. On today's
wire the two fields are DELIBERATELY DUPLICATED — once at the row top level and once inside
``verifier_metadata`` with identical values. The duplication is load-bearing: the nemo-evaluator
``gym://...protocol=native`` driver forwards top-level SCALAR fields to /verify but DROPS
list/dict fields (``options`` never arrives that way), while verifier_metadata is forwarded
intact; verify() reads top-level first and falls back to verifier_metadata via .get() with
defaults. Migration/prep tooling must preserve both placements for as long as that driver path
exists.

Both fields therefore carry ``legacy_location: "verifier_metadata"`` plus
``legacy_duplicated_top_level: true`` so the dual placement is machine-readable: migration tooling
consuming the legacy_location reverse map must keep BOTH copies (splicing verifier_metadata up and
dropping it would silently break options delivery on the native-driver path). Core validation
splices the verifier_metadata copy up before checking, where the equal values dedupe harmlessly,
so this one flat schema validates both placements.
"""

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    target: str = Field(
        default="",
        description=(
            "Gold consistency-of-objects label (e.g. 'both of them'); verify() matches the "
            "extracted prediction against it, and compute_metrics() slices accuracy by it via the "
            "rollout's top-level 'target'."
        ),
        json_schema_extra={
            "consumed_by": ["verify", "metrics"],
            "legacy_location": "verifier_metadata",
            "legacy_duplicated_top_level": True,
        },
    )
    options: List[str] = Field(
        default_factory=list,
        description=(
            "The two story-object labels offered by the question; verify() builds the candidate "
            "label set from them (falling back to [target] when empty). As a list this field is "
            "dropped by the native driver's top-level forwarding — the verifier_metadata copy is "
            "the only path that always survives."
        ),
        json_schema_extra={
            "consumed_by": ["verify"],
            "legacy_location": "verifier_metadata",
            "legacy_duplicated_top_level": True,
        },
    )
