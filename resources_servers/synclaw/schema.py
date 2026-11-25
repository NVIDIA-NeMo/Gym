# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict, Field, PositiveInt, field_validator
from synclaw.rollout.policy import RolloutPolicyConfig
from synclaw.rollout.prompts import PROMPT_FINGERPRINT

from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
)


SYNCLAW_SUBMISSION_KEY = "synclaw_submission"
# Written by prepare_db.py.
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "synclaw_emolecules.sqlite"


class SynClawResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    db_path: Path = Field(default=DEFAULT_DB_PATH, validate_default=True)
    max_similarity_reward: float = Field(
        0.5,
        gt=0.0,
        le=1.0,
        description="Similarity weight for non-exact accepted routes; exact reconstructions earn 1.",
    )
    hide_target_building_block: bool = Field(
        True,
        description="Exclude stock identical to each task target from all tools, including submission grounding.",
    )
    max_concurrent_tool_calls: PositiveInt = 64
    db_staging_dir: Path | None = Field(
        None, description="Copy the prepared database here before serving; use a job-private node-local directory."
    )

    @field_validator("db_path")
    @classmethod
    def require_db(cls, db_path: Path) -> Path:
        if not db_path.is_file():
            raise ValueError(
                f"BioNeMo-SynClaw database not found at {db_path}. Build the default database with "
                "`python resources_servers/synclaw/prepare_db.py`, or set db_path to an existing one."
            )
        return db_path


class SynClawRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_id: str
    target_smiles: str
    system_prompt: str
    question: str
    prompt_fingerprint: str | None = None

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )

    @field_validator("prompt_fingerprint")
    @classmethod
    def require_current_prompt(cls, fingerprint: str | None) -> str | None:
        if fingerprint is not None and fingerprint != PROMPT_FINGERPRINT:
            raise ValueError(
                f"Taskset prompt fingerprint {fingerprint!r} does not match installed BioNeMo-SynClaw "
                f"({PROMPT_FINGERPRINT}). Regenerate the taskset with `synclaw make-taskset`."
            )
        return fingerprint


class SynClawSeedSessionRequest(BaseSeedSessionRequest):
    system_prompt: str
    question: str
    config: RolloutPolicyConfig


class SynClawSeedSessionResponse(NeMoGymResponseCreateParamsNonStreaming):
    pass


class SynClawVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    target_smiles: str


class SynClawVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    evaluation_completed: bool = Field(
        default=False, description="True only after the verifier computes a score, including a zero score."
    )

    submitted: bool = False
    accepted: bool = False
    similarity: float = 0.0
    exact_reconstruction: bool = False
    target_hidden_from_stock: bool = False
