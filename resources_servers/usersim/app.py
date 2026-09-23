# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic NeMo UserSim scenario initialization backed by managed personas."""

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError, model_validator

from nemo_gym import WORKING_DIR
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    SimpleResourcesServer,
)
from nemo_gym.episode_types import (
    EpisodeId,
    TaskId,
)
from nemo_gym.server_utils import SESSION_ID_KEY
from nemo_gym.usersim_episode_types import (
    ResolvedUserSimContext,
    UserSimEpisodeStatus,
    UserSimSamplingRequest,
    UserSimScenario,
    UserSimSeedResponse,
    UserSimSimulationResult,
    UserSimTaskInput,
    UserSimTheme,
    UserSimVerification,
    UserSimVerifyRequest,
)


SUPPORTED_PROBES = frozenset({"general_open_ended", "general_educational"})
logger = logging.getLogger(__name__)


class UserSimResourcesServerConfig(BaseResourcesServerConfig):
    personas_cache_dir: Path = Path("~/.cache/nemo-gym/usersim/personas")
    personas_dataset_version: str = Field("0.0.2", pattern=r"^[A-Za-z0-9._-]+$")
    personas_locales: list[str] = Field(default_factory=lambda: ["en_US"])
    probe_mix: dict[str, float] = Field(
        default_factory=lambda: {
            "general_open_ended": 0.5,
            "general_educational": 0.5,
        }
    )
    probe_themes: dict[str, list[UserSimTheme]] = Field(
        default_factory=lambda: {
            "general_open_ended": [
                UserSimTheme(
                    topic="local food and dining",
                    goal="Seek a practical recommendation about local food and dining.",
                )
            ],
            "general_educational": [
                UserSimTheme(
                    topic="local ecology",
                    goal="Learn about local ecology by asking focused follow-up questions.",
                )
            ],
        }
    )

    @model_validator(mode="after")
    def validate_probes(self) -> "UserSimResourcesServerConfig":
        if not self.personas_locales:
            raise ValueError("personas_locales must contain at least one locale")
        invalid_locales = [locale for locale in self.personas_locales if not locale.replace("_", "").isalnum()]
        if invalid_locales:
            raise ValueError(f"Invalid persona locales: {invalid_locales}")
        if len(set(self.personas_locales)) != len(self.personas_locales):
            raise ValueError("personas_locales must not contain duplicates")
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


class PreparedPersonaDataset(BaseModel):
    locale: str
    personas_dataset_version: str
    panel_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    panel_size_bytes: int = Field(ge=1)
    panel_rows: int
    generator: str


class UserSimEpisodeState(BaseModel):
    user_context: dict[str, str] = Field(default_factory=dict)
    assistant_context_reads: int = Field(0, ge=0)
    termination_reason: str | None = None


class SeededUserSimEpisode(BaseModel):
    episode_id: EpisodeId
    task_id: TaskId
    seed: UserSimSeedResponse
    state: UserSimEpisodeState = Field(default_factory=UserSimEpisodeState)


class RecordUserContextRequest(BaseModel):
    key: str = Field(min_length=1, max_length=100)
    value: str = Field(min_length=1, max_length=1_000)


class FinishEpisodeRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=500)


def _stable_fraction(*parts: Any) -> float:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest, "big") / (1 << (8 * len(digest)))


def _stable_index(size: int, *parts: Any) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest, "big") % size


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _persona_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    nested_persona = row.get("persona")
    if isinstance(nested_persona, dict) and nested_persona:
        return nested_persona
    if isinstance(nested_persona, str):
        try:
            decoded = json.loads(nested_persona)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict) and decoded:
            return decoded
    return row or None


def _conversation_roles(result: UserSimSimulationResult) -> set[str]:
    messages = result.conversation_messages
    return {
        role for message in messages if isinstance(message, dict) and isinstance((role := message.get("role")), str)
    }


class UserSimResourcesServer(SimpleResourcesServer):
    """Resolve one replayable persona and general-purpose probe per episode."""

    config: UserSimResourcesServerConfig
    session_id_to_seed: dict[str, SeededUserSimEpisode] = Field(default_factory=dict)
    locale_to_personas: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    locale_to_dataset: dict[str, PreparedPersonaDataset] = Field(default_factory=dict)

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        for locale in self.config.personas_locales:
            personas, dataset = self._load_prepared_panel(locale)
            self.locale_to_personas[locale] = personas
            self.locale_to_dataset[locale] = dataset

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/record_user_context")(self.record_user_context)
        app.post("/read_user_context")(self.read_user_context)
        app.post("/finish_episode")(self.finish_episode)
        app.post("/episode_status")(self.episode_status)
        app.post("/close_session")(self.close_session)
        return app

    def _version_dir(self) -> Path:
        cache_dir = self.config.personas_cache_dir.expanduser()
        if not cache_dir.is_absolute():
            cache_dir = WORKING_DIR / cache_dir
        return cache_dir / self.config.personas_dataset_version

    def _panel_path(self, locale: str) -> Path:
        return self._version_dir() / "panels" / f"{locale}.parquet"

    def _manifest_path(self, locale: str) -> Path:
        return self._panel_path(locale).with_suffix(".manifest.json")

    def _load_prepared_panel(self, locale: str) -> tuple[list[dict[str, Any]], PreparedPersonaDataset]:
        panel_path = self._panel_path(locale)
        manifest_path = self._manifest_path(locale)
        if not panel_path.is_file() or not manifest_path.is_file():
            raise RuntimeError(
                f"Prepared NeMo UserSim panel for {locale!r} is missing at {panel_path}. "
                "Run `gym eval prepare --benchmark usersim` before starting the Resources Server."
            )
        try:
            manifest = PreparedPersonaDataset.model_validate_json(manifest_path.read_text())
        except Exception as exc:
            raise RuntimeError(f"Prepared NeMo UserSim panel manifest at {manifest_path} is invalid: {exc}") from exc
        if manifest.locale != locale or manifest.personas_dataset_version != self.config.personas_dataset_version:
            raise RuntimeError(f"Prepared NeMo UserSim panel manifest at {manifest_path} does not match configuration")
        if manifest.panel_size_bytes != panel_path.stat().st_size or manifest.panel_sha256 != _sha256_file(panel_path):
            raise RuntimeError(f"Prepared NeMo UserSim panel at {panel_path} does not match its manifest")
        try:
            panel_rows = pq.read_table(panel_path).to_pylist()
        except Exception as exc:
            raise RuntimeError(f"Prepared NeMo UserSim panel at {panel_path} is not valid Parquet: {exc}") from exc
        personas = [persona for row in panel_rows if (persona := _persona_from_row(row)) is not None]
        if not personas or len(personas) != manifest.panel_rows:
            raise RuntimeError(f"Prepared NeMo UserSim panel at {panel_path} contains invalid persona rows")
        logger.info("Loaded prepared NeMo UserSim panel at %s", panel_path)
        return personas, manifest

    def _load_personas(self, locale: str) -> list[dict[str, Any]]:
        personas = self.locale_to_personas.get(locale)
        if personas is None:
            raise HTTPException(
                status_code=422,
                detail=f"Locale {locale!r} was not initialized; configured locales: {self.config.personas_locales}",
            )
        return personas

    def _select_probe(self, sampling: UserSimSamplingRequest) -> str:
        if sampling.probe_type is not None:
            if sampling.probe_type not in SUPPORTED_PROBES:
                raise HTTPException(status_code=422, detail=f"Unsupported probe type: {sampling.probe_type!r}")
            return sampling.probe_type
        threshold = _stable_fraction(sampling.seed, sampling.locale, "probe")
        total = sum(self.config.probe_mix.values())
        cumulative = 0.0
        for probe, weight in self.config.probe_mix.items():
            cumulative += weight / total
            if threshold < cumulative:
                return probe
        return next(reversed(self.config.probe_mix))

    def _resolve_seed(self, sampling: UserSimSamplingRequest, resources_session_id: str) -> UserSimSeedResponse:
        personas = self._load_personas(sampling.locale)
        dataset = self.locale_to_dataset[sampling.locale]
        persona = personas[_stable_index(len(personas), sampling.seed, sampling.locale, "persona")]
        probe_type = self._select_probe(sampling)
        themes = self.config.probe_themes[probe_type]
        theme = themes[_stable_index(len(themes), sampling.seed, sampling.locale, probe_type, "theme")]
        return UserSimSeedResponse(
            resources_session_id=resources_session_id,
            scenario=UserSimScenario(
                locale=sampling.locale,
                persona=persona,
                probe_type=probe_type,
                theme={
                    "type": theme.topic,
                    "description": theme.goal,
                },
                goal=theme.goal,
            ),
            usersim_context=ResolvedUserSimContext(
                locale=sampling.locale,
                seed=sampling.seed,
                personas_dataset_version=dataset.personas_dataset_version,
                personas_panel_sha256=dataset.panel_sha256,
            ),
        )

    def _seeded_episode(self, request: Request) -> SeededUserSimEpisode:
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self.session_id_to_seed:
            raise RuntimeError("No active NeMo UserSim scenario. Call /seed_session first.")
        return self.session_id_to_seed[session_id]

    @staticmethod
    def _status(seeded: SeededUserSimEpisode) -> UserSimEpisodeStatus:
        return UserSimEpisodeStatus(
            state={
                "user_context": seeded.state.user_context,
                "assistant_context_reads": seeded.state.assistant_context_reads,
            },
            terminated=seeded.state.termination_reason is not None,
            termination_reason=seeded.state.termination_reason,
        )

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest,
    ) -> UserSimSeedResponse:
        try:
            task = UserSimTaskInput.model_validate(body.task_data)
        except ValidationError as error:
            raise HTTPException(status_code=422, detail=error.errors()) from error
        session_id = request.session[SESSION_ID_KEY]
        result = await asyncio.to_thread(self._resolve_seed, task.sampling, session_id)
        self.session_id_to_seed[session_id] = SeededUserSimEpisode(
            episode_id=body.episode_id,
            task_id=body.task_id,
            seed=result,
        )
        return result

    async def record_user_context(
        self,
        request: Request,
        body: RecordUserContextRequest,
    ) -> UserSimEpisodeStatus:
        seeded = self._seeded_episode(request)
        seeded.state.user_context[body.key] = body.value
        return self._status(seeded)

    async def read_user_context(self, request: Request) -> UserSimEpisodeStatus:
        seeded = self._seeded_episode(request)
        seeded.state.assistant_context_reads += 1
        return self._status(seeded)

    async def finish_episode(
        self,
        request: Request,
        body: FinishEpisodeRequest,
    ) -> UserSimEpisodeStatus:
        seeded = self._seeded_episode(request)
        seeded.state.termination_reason = body.reason
        return self._status(seeded)

    async def episode_status(self, request: Request) -> UserSimEpisodeStatus:
        return self._status(self._seeded_episode(request))

    async def verify(
        self,
        request: Request,
        body: UserSimVerifyRequest,
    ) -> UserSimVerification:
        seeded = self._seeded_episode(request)
        verification_input = body.verification_input
        if (
            body.episode_id != seeded.episode_id
            or body.task_id != seeded.task_id
            or verification_input.usersim_context != seeded.seed.usersim_context
            or verification_input.scenario != seeded.seed.scenario
        ):
            raise HTTPException(
                status_code=409,
                detail="Verified NeMo UserSim resolved episode does not match the seeded session",
            )
        participants_completed = {"user", "assistant"} <= _conversation_roles(verification_input.usersim_result)
        status = self._status(seeded)
        shared_state_exercised = bool(seeded.state.user_context) and seeded.state.assistant_context_reads > 0
        tool_scenario_started = (
            bool(seeded.state.user_context) or seeded.state.assistant_context_reads > 0 or status.terminated
        )
        scenario_completed = participants_completed and (
            not tool_scenario_started or (shared_state_exercised and status.terminated)
        )
        return UserSimVerification(
            reward=float(scenario_completed),
            reward_components={
                "participants_completed": float(participants_completed),
                "shared_state_exercised": float(shared_state_exercised),
                "terminated": float(status.terminated),
            },
            scenario_completed=scenario_completed,
            verifier_data={
                "invocations": [invocation.model_dump(mode="json") for invocation in verification_input.invocations],
                "environment_state": status.state,
                "episode_interaction_protocol": verification_input.episode_interaction_protocol,
                "scenario": verification_input.scenario.model_dump(mode="json"),
                "usersim_context": verification_input.usersim_context.model_dump(mode="json"),
                "usersim_result": verification_input.usersim_result.model_dump(mode="json"),
                "scenario_completed": scenario_completed,
                "termination_reason": status.termination_reason,
            },
        )

    async def close_session(
        self,
        request: Request,
        body: ResourcesCloseSessionRequest,
    ) -> ResourcesCloseSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        seeded = self._seeded_episode(request)
        if body.resources_session_id != seeded.seed.resources_session_id or body.episode_id != seeded.episode_id:
            raise HTTPException(status_code=409, detail="Resources session does not match the active episode")
        del self.session_id_to_seed[session_id]
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)


if __name__ == "__main__":
    UserSimResourcesServer.run_webserver()
