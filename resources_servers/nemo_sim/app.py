# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic NeMo-Sim scenario initialization backed by managed personas."""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY
from processors.user_assistant.app import UserAssistantVerifyRequest, UserAssistantVerifyResponse


SUPPORTED_PROBES = frozenset({"general_open_ended", "general_educational"})


class ProbeTheme(BaseModel):
    """One deterministic theme and its materialized user objective."""

    topic: str = Field(min_length=1)
    goal: str = Field(min_length=1)


class NeMoSimResourcesServerConfig(BaseResourcesServerConfig):
    personas_dir: Path = Path("~/.data-designer/managed-assets/datasets")
    probe_mix: dict[str, float] = Field(
        default_factory=lambda: {
            "general_open_ended": 0.5,
            "general_educational": 0.5,
        }
    )
    probe_themes: dict[str, list[ProbeTheme]] = Field(
        default_factory=lambda: {
            "general_open_ended": [
                ProbeTheme(
                    topic="local food and dining",
                    goal="Seek a practical recommendation about local food and dining.",
                )
            ],
            "general_educational": [
                ProbeTheme(
                    topic="local ecology",
                    goal="Learn about local ecology by asking focused follow-up questions.",
                )
            ],
        }
    )

    @model_validator(mode="after")
    def validate_probes(self) -> "NeMoSimResourcesServerConfig":
        unknown = set(self.probe_mix) - SUPPORTED_PROBES
        if unknown:
            raise ValueError(f"Unsupported probe types: {sorted(unknown)}")
        if not self.probe_mix or any(weight < 0 for weight in self.probe_mix.values()):
            raise ValueError("probe_mix must contain non-negative weights")
        if sum(self.probe_mix.values()) <= 0:
            raise ValueError("probe_mix weights must sum to more than zero")
        missing_themes = {
            probe for probe, weight in self.probe_mix.items() if weight > 0 and not self.probe_themes.get(probe)
        }
        if missing_themes:
            raise ValueError(f"Missing themes for probe types: {sorted(missing_themes)}")
        return self


class NeMoSimSamplingRequest(BaseModel):
    locale: str = Field("en_US", pattern=r"^[A-Za-z0-9_]+$")
    seed: int
    probe_type: Optional[str] = None

    @model_validator(mode="after")
    def validate_probe_type(self) -> "NeMoSimSamplingRequest":
        if self.probe_type is not None and self.probe_type not in SUPPORTED_PROBES:
            raise ValueError(f"Unsupported probe type: {self.probe_type!r}")
        return self


class NeMoSimSeedSessionRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")

    user_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    nemo_sim_sampling: NeMoSimSamplingRequest


class ResolvedNeMoSimContext(BaseModel):
    locale: str
    seed: int
    probe_type: str
    theme: ProbeTheme
    goal: str
    persona: dict[str, Any]


class NeMoSimSeedSessionResponse(BaseModel):
    user_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    nemo_sim_context: ResolvedNeMoSimContext


class NeMoSimEpisodeStatusResponse(BaseModel):
    terminated: bool = False
    reason: Optional[str] = None
    state: dict[str, Any]


class NeMoSimVerifyResponse(UserAssistantVerifyResponse):
    nemo_sim_context: ResolvedNeMoSimContext
    scenario_completed: bool


def _stable_fraction(*parts: Any) -> float:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest, "big") / (1 << (8 * len(digest)))


def _stable_index(size: int, *parts: Any) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest, "big") % size


class NeMoSimResourcesServer(SimpleResourcesServer):
    """Resolve one replayable persona and general-purpose probe per episode."""

    config: NeMoSimResourcesServerConfig
    session_id_to_context: dict[str, ResolvedNeMoSimContext] = Field(default_factory=dict)
    locale_to_personas: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/episode_status")(self.episode_status)
        return app

    def _personas_path(self, locale: str) -> Path:
        return self.config.personas_dir.expanduser() / f"{locale}.parquet"

    def _load_personas(self, locale: str) -> list[dict[str, Any]]:
        if locale in self.locale_to_personas:
            return self.locale_to_personas[locale]

        path = self._personas_path(locale)
        if not path.is_file():
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Managed persona dataset for locale {locale!r} was not found at {path}. "
                    "Install the Data Designer managed persona assets or configure personas_dir."
                ),
            )
        rows = pq.read_table(path).to_pylist()
        personas = []
        for row in rows:
            raw_persona = row.get("persona", row)
            if isinstance(raw_persona, str):
                try:
                    raw_persona = json.loads(raw_persona)
                except json.JSONDecodeError:
                    continue
            if isinstance(raw_persona, dict) and raw_persona:
                personas.append(raw_persona)
        if not personas:
            raise HTTPException(
                status_code=422,
                detail=f"Managed persona dataset at {path} contains no usable personas",
            )
        self.locale_to_personas[locale] = personas
        return personas

    def _select_probe(self, sampling: NeMoSimSamplingRequest) -> str:
        if sampling.probe_type is not None:
            return sampling.probe_type
        threshold = _stable_fraction(sampling.seed, sampling.locale, "probe")
        total = sum(self.config.probe_mix.values())
        cumulative = 0.0
        for probe, weight in self.config.probe_mix.items():
            cumulative += weight / total
            if threshold < cumulative:
                return probe
        return next(reversed(self.config.probe_mix))

    def _resolve_context(self, sampling: NeMoSimSamplingRequest) -> ResolvedNeMoSimContext:
        personas = self._load_personas(sampling.locale)
        persona = personas[_stable_index(len(personas), sampling.seed, sampling.locale, "persona")]
        probe_type = self._select_probe(sampling)
        themes = self.config.probe_themes[probe_type]
        theme = themes[_stable_index(len(themes), sampling.seed, sampling.locale, probe_type, "theme")]
        return ResolvedNeMoSimContext(
            locale=sampling.locale,
            seed=sampling.seed,
            probe_type=probe_type,
            theme=theme,
            goal=theme.goal,
            persona=persona,
        )

    def _context(self, request: Request) -> ResolvedNeMoSimContext:
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self.session_id_to_context:
            raise RuntimeError("No active NeMo-Sim scenario. Call /seed_session first.")
        return self.session_id_to_context[session_id]

    async def seed_session(
        self,
        request: Request,
        body: NeMoSimSeedSessionRequest,
    ) -> NeMoSimSeedSessionResponse:
        context = await asyncio.to_thread(self._resolve_context, body.nemo_sim_sampling)
        self.session_id_to_context[request.session[SESSION_ID_KEY]] = context
        user_params = body.user_responses_create_params.model_copy(deep=True)
        metadata = dict(user_params.metadata or {})
        metadata["nemo_sim"] = json.dumps(
            {
                "locale": context.locale,
                "goal": context.goal,
                "persona": context.persona,
                "probe_type": context.probe_type,
                "theme": context.theme.model_dump(mode="json"),
                "seed": context.seed,
            },
            ensure_ascii=False,
        )
        user_params = user_params.model_copy(update={"metadata": metadata})
        return NeMoSimSeedSessionResponse(
            user_responses_create_params=user_params,
            nemo_sim_context=context,
        )

    async def episode_status(self, request: Request) -> NeMoSimEpisodeStatusResponse:
        context = self._context(request)
        return NeMoSimEpisodeStatusResponse(
            state={
                "locale": context.locale,
                "seed": context.seed,
                "probe_type": context.probe_type,
                "theme": context.theme.model_dump(mode="json"),
            }
        )

    async def verify(
        self,
        request: Request,
        body: UserAssistantVerifyRequest,
    ) -> NeMoSimVerifyResponse:
        context = self._context(request)
        scenario_completed = bool(body.assistant_trajectory and body.user_trajectory)
        return NeMoSimVerifyResponse(
            **body.model_dump(mode="json"),
            reward=float(scenario_completed),
            nemo_sim_context=context,
            scenario_completed=scenario_completed,
        )


if __name__ == "__main__":
    NeMoSimResourcesServer.run_webserver()
