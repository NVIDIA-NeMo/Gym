# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical models for generated synthetic tool-use seed artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PROTOCOL_VERSION = "1"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_hash(value: Any, *, length: int = 16) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def normalize_domain_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


class StageState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class DomainApplication(BaseModel):
    model_config = ConfigDict(extra="allow")

    function: str = Field(min_length=1)

    @field_validator("function")
    @classmethod
    def strip_function(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("function must not be blank")
        return value


class GenerationMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    base_url: str | None = None
    sampling: dict[str, Any] = Field(default_factory=dict)
    prompt_name: str | None = None
    prompt_sha256: str | None = None


class DomainCandidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    domain_id: str = ""
    name: str = Field(min_length=1)
    normalized_name: str = ""
    applications: list[DomainApplication] = Field(min_length=1)
    generation_profile: str
    request_index: int = Field(ge=0)
    candidate_index: int = Field(ge=0)
    generation: GenerationMetadata
    accepted: bool = True
    rejection_reason: str | None = None

    @model_validator(mode="after")
    def populate_identity(self) -> DomainCandidate:
        self.name = self.name.strip()
        self.normalized_name = self.normalized_name or normalize_domain_name(self.name)
        if not self.normalized_name:
            raise ValueError("name must contain letters or numbers")
        self.domain_id = self.domain_id or canonical_hash(
            {"generation_profile": self.generation_profile, "normalized_name": self.normalized_name}
        )
        return self


class SeedToolSignature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    doc: str = Field(min_length=1)
    params: dict[str, Any]
    returns: dict[str, Any]

    @field_validator("name", "doc")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


class CustomerScenarioArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_persona: str = Field(min_length=1)
    reason_for_contact: str = Field(min_length=1)
    customer_details: str = Field(min_length=1)
    unknown_info: str | None = None
    task_instructions: str = Field(min_length=1)
    representative_domain: str = Field(min_length=1)
    outside_policy_scope: bool

    @field_validator(
        "customer_persona",
        "reason_for_contact",
        "customer_details",
        "task_instructions",
        "representative_domain",
    )
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @property
    def scenario_id(self) -> str:
        return canonical_hash(self.model_dump())


class ModelRoleConfig(BaseModel):
    model: str
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = Field(default=600.0, gt=0)
    provider_attempts: int = Field(default=3, ge=1)
    retry_initial_backoff_seconds: float = Field(default=1.0, ge=0)
    retry_max_backoff_seconds: float = Field(default=30.0, ge=0)
    concurrency: int = Field(default=8, ge=1)
    sampling: dict[str, Any] = Field(default_factory=lambda: {"temperature": 1})


class DomainStageConfig(BaseModel):
    request_count: int = Field(default=10, ge=1)
    semantic_attempts: int = Field(default=2, ge=1)


class PolicyToolsStageConfig(BaseModel):
    semantic_attempts: int = Field(default=3, ge=1)
    refine: bool = True
    judge_enabled: bool = True
    judge_votes: int = Field(default=3, ge=1)
    judge_max_failure_fraction: float = Field(default=0.5, ge=0, le=1)
    golden_comparison_enabled: bool = True
    golden_comparison_count: int = Field(default=4, ge=1)


class ScenarioStageConfig(BaseModel):
    scenarios_per_request: int = Field(default=10, ge=1)
    request_count_per_domain: int = Field(default=10, ge=1)
    outside_policy_scope_fraction: float = Field(default=0.1, ge=0, le=1)
    semantic_attempts: int = Field(default=3, ge=1)
    scenarios_per_file: int = Field(default=3000, ge=1)


class SeedGenerationConfig(BaseModel):
    run_name: str
    output_dir: Path
    generation_profile: str
    source_name: str
    random_seed: int = 1
    code_revision: str | None = None
    domain_model: ModelRoleConfig
    policy_tools_model: ModelRoleConfig
    judge_model: ModelRoleConfig | None = None
    scenario_model: ModelRoleConfig
    domains: DomainStageConfig = Field(default_factory=DomainStageConfig)
    policy_tools: PolicyToolsStageConfig = Field(default_factory=PolicyToolsStageConfig)
    scenarios: ScenarioStageConfig = Field(default_factory=ScenarioStageConfig)


class StageStatus(BaseModel):
    state: StageState = StageState.PENDING
    attempts: int = 0
    updated_at: str = Field(default_factory=utc_now)
    failure_category: str | None = None
    failure_detail: str | None = None


class DomainManifestEntry(BaseModel):
    domain_id: str
    source_index: int = Field(ge=0)
    name: str
    normalized_name: str
    artifact_dir: str
    stages: dict[str, StageStatus] = Field(
        default_factory=lambda: {
            "policy_tools": StageStatus(),
            "scenarios": StageStatus(),
        }
    )
    scenario_count: int = 0


class RunManifest(BaseModel):
    protocol_version: str = PROTOCOL_VERSION
    run_id: str
    run_name: str
    source_name: str
    generation_profile: str
    code_revision: str | None = None
    random_seed: int
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    config: dict[str, Any]
    asset_hashes: dict[str, str] = Field(default_factory=dict)
    domains: list[DomainManifestEntry] = Field(default_factory=list)

    @classmethod
    def from_config(
        cls,
        config: SeedGenerationConfig,
        asset_hashes: dict[str, str] | None = None,
    ) -> RunManifest:
        asset_hashes = dict(sorted((asset_hashes or {}).items()))
        non_secret_config = config.model_dump(mode="json")
        for role_name in ("domain_model", "policy_tools_model", "judge_model", "scenario_model"):
            role = non_secret_config.get(role_name)
            if role:
                role.pop("api_key_env", None)
                role.pop("base_url", None)
        return cls(
            run_id=canonical_hash(
                {
                    "run_name": config.run_name,
                    "source_name": config.source_name,
                    "profile": config.generation_profile,
                    "seed": config.random_seed,
                    "asset_hashes": asset_hashes,
                }
            ),
            run_name=config.run_name,
            source_name=config.source_name,
            generation_profile=config.generation_profile,
            code_revision=config.code_revision,
            random_seed=config.random_seed,
            config=non_secret_config,
            asset_hashes=asset_hashes,
        )
