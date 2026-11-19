# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the proof_genselect server (pick the better of two candidate proofs).

Mirrors ``ProofGenSelectVerifyRequest`` (app.py): ``problem``, ``proof_1``, ``proof_2`` and
``correct_index`` are wire-required; ``score_1``/``score_2`` are Optional (prepare_data.py emits
them only when present in the source rows). Scoring compares the parsed ``<best_solution>`` index
against ``correct_index``; the proofs and problem were already baked into the prompt by
prepare_data, so they stay wire-required even though ``verify()`` never reads the proofs and
reads ``problem`` only for optional side-channel JSONL logging (``PROOF_GENSELECT_LOG_JSONL_PATH``).
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    problem: str = Field(
        description=(
            "Problem statement (already baked into the prompt); verify() reads it only for optional "
            "side-channel JSONL logging."
        ),
        json_schema_extra={"consumed_by": ["prompt", "provenance"]},
    )
    proof_1: str = Field(
        description="First candidate proof (already baked into the prompt); pass-through, unread by verify().",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    proof_2: str = Field(
        description="Second candidate proof (already baked into the prompt); pass-through, unread by verify().",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    correct_index: int = Field(
        description="Index of the correct proof; reward is 1.0 iff the parsed <best_solution> index equals it.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    score_1: Optional[float] = Field(
        default=None,
        description="Source-dataset quality score for proof_1; optional per-row, unread by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    score_2: Optional[float] = Field(
        default=None,
        description="Source-dataset quality score for proof_2; optional per-row, unread by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
