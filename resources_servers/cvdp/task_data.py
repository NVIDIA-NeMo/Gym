# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the cvdp server.

CVDP hardware-verification rows carry every task field inside ``verifier_metadata`` (schema written
flat; ``legacy_location`` records today's placement). Field typing and required-ness mirror
``CVDPVerifierMetadata`` (app.py:87), which ``verify()`` ``model_validate``s against the wire's
otherwise-untyped ``Dict[str, Any]`` bucket: only ``task_id`` is required, everything else defaults.
Three task families share this one envelope, differing only in which optionals are populated:
code-generation rows (data/example.jsonl) carry ``target_files`` + ``harness_files``; agentic rows
(data/example_agentic.jsonl) add ``context_files``; code-comprehension rows (categories 6/8/9/10)
leave the file fields empty and carry ``subjective_reference`` instead — so this is a single model
with optionals, not a union (there is no in-band discriminator beyond ``categories[0]``).
``rtl_files`` (CVDPVerifyRequest, app.py:101) is a verify-time-only field injected by the agent and
is deliberately NOT a task field.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        description="CVDP task identifier (e.g. 'cvdp_copilot_skid_buffer_0001'); echoed on the verify response.",
        json_schema_extra={"consumed_by": ["verify", "provenance"], "legacy_location": "verifier_metadata"},
    )
    categories: List[str] = Field(
        default_factory=list,
        description=(
            "Exactly two elements on real rows: [category_id, difficulty], e.g. ['cid003', 'medium']. "
            "verify() hard-indexes both (category routing via int(categories[0][3:]), difficulty from "
            "categories[1]) and echoes them into the CVDP report, but the wire model defaults to []."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"], "legacy_location": "verifier_metadata"},
    )
    difficulty: str = Field(
        default="",
        description=(
            "Declared by CVDPVerifierMetadata but unread: verify() derives difficulty from categories[1] "
            "instead. Empty in example.jsonl, duplicated from categories[1] in example_agentic.jsonl."
        ),
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    target_files: List[str] = Field(
        default_factory=list,
        description=(
            "RTL filenames the model must produce (parse targets for the objective grading path); "
            "empty for code-comprehension categories."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    harness_files: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description=(
            "Filename -> full file content for the testbench harness (docker-compose.yml, src/*.py), "
            "passed to TestbenchRunner on the objective grading path; empty for code-comprehension categories."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    context_files: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Filename -> content of companion RTL needed for compilation (from input.context); present only "
            "on agentic rows (data/example_agentic.jsonl)."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    subjective_reference: Optional[str] = Field(
        default=None,
        description=(
            "Reference answer for code-comprehension categories (6, 8, 9, 10), emitted by "
            "scripts/convert_to_gym.py; read as `meta.subjective_reference or ''` on the subjective "
            "(BLEU/ROUGE/judge) path. Absent from both committed example files."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
