# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo-Sim task, seed, and verification contracts."""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.processors import AgentTurn, EpisodeSeedResponse, EpisodeVerification, EpisodeVerifyRequest


NEMO_SIM_MODEL_ALIASES = frozenset(
    {"user_model", "assistant_model", "api_response_model", "judge_model", "summary_model"}
)
EPISODE_INTERACTION_PROTOCOL = "nemo_sim.ConversationLoop"


class NeMoSimScenario(BaseModel):
    model_config = ConfigDict(extra="allow")

    persona: dict[str, Any]
    probe_type: str = "general_open_ended"
    theme: dict[str, Any] | str
    goal: str = ""
    locale: str = "en_US"


class NeMoSimTheme(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=1)
    goal: str = Field(min_length=1)


class NeMoSimSamplingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    locale: str = Field("en_US", pattern=r"^[A-Za-z0-9_]+$")
    seed: int
    probe_type: str | None = None


class NeMoSimProtocolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_query_attempts: int = Field(3, ge=1)
    max_assistant_attempts: int = Field(1, ge=1)
    enforce_user_language: bool = True
    user_language_min_script_compliance: float = Field(0.6, ge=0, le=1)
    user_language_min_letters: int = Field(8, ge=0)
    incremental_disclosure_ratio: float = Field(0.6, ge=0, le=1)
    persona_grounding_ratio: float = Field(1, ge=0, le=1)
    context_compression: bool = True
    compression_window: int = Field(1, ge=1)
    store_reasoning: bool = True
    random_seed: int | None = None
    verbosity: int = Field(1, ge=0, le=2)


class NeMoSimTaskData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nemo_sim_sampling: NeMoSimSamplingRequest
    model_responses_create_params: dict[str, NeMoGymResponseCreateParamsNonStreaming] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_unknown_model_aliases(self) -> "NeMoSimTaskData":
        unknown = set(self.model_responses_create_params) - NEMO_SIM_MODEL_ALIASES
        if unknown:
            raise ValueError(f"model_responses_create_params contains unknown aliases: {sorted(unknown)}")
        return self


class ResolvedNeMoSimContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    locale: str
    seed: int
    personas_dataset_version: str
    personas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    personas_panel_seed: int


class NeMoSimSeedResponse(EpisodeSeedResponse):
    scenario: NeMoSimScenario
    nemo_sim_context: ResolvedNeMoSimContext


class NeMoSimSimulationResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    conversation_messages: list[dict[str, Any]]
    conversation_status: bool
    simulation_outcome: dict[str, Any] = Field(default_factory=dict)
    conversation_metadata: dict[str, Any] | None = None
    simulation_traces: list[dict[str, Any]] | None = None

    @field_validator(
        "conversation_messages",
        "simulation_outcome",
        "conversation_metadata",
        "simulation_traces",
        mode="before",
    )
    @classmethod
    def decode_json_columns(cls, value: Any) -> Any:
        return json.loads(value) if isinstance(value, str) else value


class NeMoSimEpisodeStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: dict[str, Any] = Field(default_factory=dict)
    terminated: bool = False
    termination_reason: str | None = None


class NeMoSimVerifyRequest(EpisodeVerifyRequest):
    model_config = ConfigDict(extra="forbid")

    scenario: NeMoSimScenario
    nemo_sim_context: ResolvedNeMoSimContext
    nemo_sim_result: NeMoSimSimulationResult
    agent_turns: list[AgentTurn]
    episode_interaction_protocol: str = EPISODE_INTERACTION_PROTOCOL


class NeMoSimVerification(EpisodeVerification):
    """Typed verifier output before it enters EpisodeResponse."""

    scenario_completed: bool
