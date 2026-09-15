# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo-Sim task data carried inside the native episode envelope."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


NEMO_SIM_MODEL_ALIASES = frozenset(
    {"user_model", "assistant_model", "api_response_model", "judge_model", "summary_model"}
)
EPISODE_INTERACTION_PROTOCOL = "nemo_sim.ConversationLoop"


class NeMoSimScenario(BaseModel):
    model_config = ConfigDict(extra="allow")

    persona: dict[str, Any]
    probe_type: str = "general_open_ended"
    theme: dict[str, Any] | str
    locale: str = "en_US"


class NeMoSimTaskData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: NeMoSimScenario
    model_responses_create_params: dict[str, NeMoGymResponseCreateParamsNonStreaming] = Field(default_factory=dict)
    simulation_config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_unknown_model_aliases(self) -> "NeMoSimTaskData":
        unknown = set(self.model_responses_create_params) - NEMO_SIM_MODEL_ALIASES
        if unknown:
            raise ValueError(f"model_responses_create_params contains unknown aliases: {sorted(unknown)}")
        return self
