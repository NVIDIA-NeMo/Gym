# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VoiceBench subset/scorer bindings and subtype-specific verification metadata."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue, model_validator


Subset = Literal[
    "alpacaeval_full", "commoneval", "wildvoice", "openbookqa", "mmsu", "ifeval", "bbh", "advbench", "sd-qa"
]
ScoringType = Literal["open", "mcq", "ifeval", "bbh", "refusal", "reference_qa"]
SUBSET_SCORERS: dict[str, ScoringType] = {
    "alpacaeval_full": "open",
    "commoneval": "open",
    "wildvoice": "open",
    "openbookqa": "mcq",
    "mmsu": "mcq",
    "ifeval": "ifeval",
    "bbh": "bbh",
    "advbench": "refusal",
    "sd-qa": "reference_qa",
}


class VoiceBenchMetadata(BaseModel):
    """Source metadata stays outside candidate-visible Responses input."""

    model_config = ConfigDict(extra="allow")
    subset: Subset
    sample_id: str
    prompt: str
    reference: str | None = None
    id: str | int | None = None
    instruction_id_list: list[str] | None = None
    kwargs: list[dict[str, JsonValue]] | None = None

    # Source provenance written by the dataset preparer; verify() does not read these.
    audio_filepath: str | None = None
    sample_rate: int | None = None
    key: int | None = None  # ifeval
    category: str | None = None  # mmsu
    src: str | None = None  # mmsu
    question_id: int | None = None  # mmsu
    cot_content: str | None = None  # mmsu
    conversation_hash: str | None = None  # wildvoice

    @model_validator(mode="after")
    def validate_scoring_fields(self) -> "VoiceBenchMetadata":
        """Reject missing references or malformed instruction lists before scoring."""
        if not self.sample_id:
            raise ValueError("sample_id must be nonempty")
        scoring_type = SUBSET_SCORERS[self.subset]
        if scoring_type in ("mcq", "bbh", "reference_qa") and self.reference is None:
            raise ValueError(f"{self.subset} requires verifier_metadata.reference")
        if scoring_type == "ifeval":
            if not self.instruction_id_list or self.kwargs is None:
                raise ValueError("ifeval requires nonempty instruction_id_list and kwargs")
            if len(self.instruction_id_list) != len(self.kwargs):
                raise ValueError("ifeval instruction_id_list and kwargs lengths must match")
        return self


# Rows still nest these fields under verifier_metadata; validation splices them up to this flat schema.
TaskData = VoiceBenchMetadata
