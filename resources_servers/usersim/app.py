# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic NeMo UserSim scenario initialization backed by managed personas."""

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pyarrow as pa
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
NEMOTRON_PERSONAS_TEAM = "nvidia/nemotron-personas"
NEMOTRON_PERSONAS_DATASET_PREFIX = "nemotron-personas-dataset-"
logger = logging.getLogger(__name__)


class UserSimResourcesServerConfig(BaseResourcesServerConfig):
    personas_cache_dir: Path = Path("~/.cache/nemo-gym/usersim/personas")
    personas_dataset_version: str = Field("0.0.2", pattern=r"^[A-Za-z0-9._-]+$")
    personas_locales: list[str] = Field(default_factory=lambda: ["en_US"])
    personas_panel_size: int = Field(1_000, ge=1)
    personas_panel_seed: int = 42
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
    resource: str
    version: str
    source_sha256: str
    source_rows: int
    panel_seed: int
    panel_rows: int


class CachedPersonaSource(BaseModel):
    locale: str
    resource: str
    version: str
    sha256: str
    size_bytes: int
    rows: int


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


@contextmanager
def _exclusive_file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


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
            personas, dataset = self._prepare_locale(locale)
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

    def _source_path(self, locale: str) -> Path:
        return self._version_dir() / "source" / f"{locale}.parquet"

    def _panel_path(self, locale: str) -> Path:
        filename = f"{locale}-n{self.config.personas_panel_size}-seed{self.config.personas_panel_seed}.parquet"
        return self._version_dir() / "panels" / filename

    def _source_manifest_path(self, locale: str) -> Path:
        return self._source_path(locale).with_suffix(".manifest.json")

    def _manifest_path(self, locale: str) -> Path:
        return self._panel_path(locale).with_suffix(".manifest.json")

    def _resource(self, locale: str) -> str:
        dataset_name = f"{NEMOTRON_PERSONAS_DATASET_PREFIX}{locale.lower()}"
        return f"{NEMOTRON_PERSONAS_TEAM}/{dataset_name}"

    def _validate_parquet(self, path: Path) -> int:
        try:
            row_count = pq.ParquetFile(path).metadata.num_rows
        except Exception as exc:
            raise RuntimeError(f"Persona dataset at {path} is not valid Parquet: {exc}") from exc
        if row_count < 1:
            raise RuntimeError(f"Persona dataset at {path} contains no rows")
        return row_count

    def _ensure_source(self, locale: str) -> tuple[Path, int, str]:
        source_path = self._source_path(locale)
        manifest_path = self._source_manifest_path(locale)
        lock_path = self._version_dir() / "locks" / f"{locale}.lock"
        with _exclusive_file_lock(lock_path):
            if not source_path.is_file():
                raise RuntimeError(
                    f"Prepared persona dataset {self._resource(locale)}:"
                    f"{self.config.personas_dataset_version} is missing at {source_path}. "
                    "Run `gym eval prepare --benchmark usersim` before starting the Resources Server."
                )
            logger.info("Loading prepared persona dataset at %s", source_path)
            source_rows = self._validate_parquet(source_path)
            if manifest_path.is_file():
                try:
                    manifest = CachedPersonaSource.model_validate_json(manifest_path.read_text())
                except ValueError:
                    manifest = None
                if (
                    manifest is not None
                    and manifest.resource == self._resource(locale)
                    and manifest.version == self.config.personas_dataset_version
                    and manifest.size_bytes == source_path.stat().st_size
                    and manifest.rows == source_rows
                ):
                    logger.info("Reusing cached persona source manifest at %s", manifest_path)
                    return source_path, source_rows, manifest.sha256

            logger.info("Recording persona source checksum for %s", source_path)
            source_sha256 = _sha256_file(source_path)
            manifest = CachedPersonaSource(
                locale=locale,
                resource=self._resource(locale),
                version=self.config.personas_dataset_version,
                sha256=source_sha256,
                size_bytes=source_path.stat().st_size,
                rows=source_rows,
            )
            temporary_manifest = manifest_path.with_suffix(".json.tmp")
            temporary_manifest.write_text(manifest.model_dump_json(indent=2))
            os.replace(temporary_manifest, manifest_path)
        return source_path, source_rows, source_sha256

    def _sample_panel(self, source_path: Path, locale: str) -> list[dict[str, Any]]:
        sample_size = self.config.personas_panel_size
        random_generator = random.Random(
            f"{self.config.personas_dataset_version}:{locale}:{self.config.personas_panel_seed}"
        )
        personas: list[dict[str, Any]] = []
        usable_rows = 0
        for batch in pq.ParquetFile(source_path).iter_batches(batch_size=1_024):
            for row in batch.to_pylist():
                persona = _persona_from_row(row)
                if persona is None:
                    continue
                usable_rows += 1
                if len(personas) < sample_size:
                    personas.append(persona)
                    continue
                replacement_index = random_generator.randrange(usable_rows)
                if replacement_index < sample_size:
                    personas[replacement_index] = persona
        if not personas:
            raise RuntimeError(f"Persona dataset at {source_path} contains no usable personas")
        return personas

    def _write_panel(self, panel_path: Path, personas: list[dict[str, Any]]) -> None:
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = panel_path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(personas), temporary_path)
        os.replace(temporary_path, panel_path)

    def _prepare_locale(self, locale: str) -> tuple[list[dict[str, Any]], PreparedPersonaDataset]:
        source_path, source_rows, source_sha256 = self._ensure_source(locale)
        panel_path = self._panel_path(locale)
        manifest_path = self._manifest_path(locale)
        lock_path = self._version_dir() / "locks" / f"{panel_path.stem}.lock"
        with _exclusive_file_lock(lock_path):
            if panel_path.is_file() and manifest_path.is_file():
                try:
                    manifest = PreparedPersonaDataset.model_validate_json(manifest_path.read_text())
                except ValueError:
                    manifest = None
                manifest_matches = (
                    manifest is not None
                    and manifest.locale == locale
                    and manifest.resource == self._resource(locale)
                    and manifest.version == self.config.personas_dataset_version
                    and manifest.source_sha256 == source_sha256
                    and manifest.panel_seed == self.config.personas_panel_seed
                )
                if not manifest_matches:
                    panel_path.unlink()
                    manifest_path.unlink()
                else:
                    try:
                        panel_rows = pq.read_table(panel_path).to_pylist()
                    except Exception:
                        panel_rows = []
                    personas = [persona for row in panel_rows if (persona := _persona_from_row(row)) is not None]
                    if personas and len(personas) == manifest.panel_rows:
                        logger.info("Reusing prepared persona panel at %s", panel_path)
                        return personas, manifest

            logger.info("Preparing deterministic persona panel at %s", panel_path)
            personas = self._sample_panel(source_path, locale)
            self._write_panel(panel_path, personas)
            dataset = PreparedPersonaDataset(
                locale=locale,
                resource=self._resource(locale),
                version=self.config.personas_dataset_version,
                source_sha256=source_sha256,
                source_rows=source_rows,
                panel_seed=self.config.personas_panel_seed,
                panel_rows=len(personas),
            )
            temporary_manifest = manifest_path.with_suffix(".json.tmp")
            temporary_manifest.write_text(dataset.model_dump_json(indent=2))
            os.replace(temporary_manifest, manifest_path)
            return personas, dataset

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
                personas_dataset_version=dataset.version,
                personas_source_sha256=dataset.source_sha256,
                personas_panel_seed=dataset.panel_seed,
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
