# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic NeMo UserSim scenario initialization backed by managed personas."""

import asyncio
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow.parquet as pq
from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from nemo_gym import WORKING_DIR
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.episode_types import (
    EpisodeId,
    TaskId,
)
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, raise_for_status
from resources_servers.usersim.episode_contracts import (
    ResolvedUserSimContext,
    UserSimSamplingRequest,
    UserSimScenario,
    UserSimSeedResponse,
    UserSimSimulationResult,
    UserSimTaskInput,
    UserSimTheme,
    UserSimVerification,
    UserSimVerifyRequest,
)


THEME_DRIVEN_PROBES = frozenset({"tool_calling", "general_open_ended", "general_educational"})
PROBE_SCORERS = {
    "financial_services": "financial_services",
    "health_decision_support_disclosure": "health_disclosure_concealment",
    "health_general_disclosure": "health_disclosure_concealment",
    "health_therapy_disclosure": "health_disclosure_concealment",
    "health_triage_disclosure": "health_disclosure_concealment",
    "identity_disclosure": "identity_disclosure",
    "safety_agentic": "safety_agentic",
    "safety_chat_pressure": "safety_chat_pressure",
    "sov_ai_dynamic": "sov_ai_dynamic",
    "sov_ai_facts": "sov_ai_facts",
    "sov_ai_multilingual_parity": "sov_ai_multilingual_parity",
    "tool_calling": "tool_use",
}
SUPPORTED_PROBES = frozenset(
    {
        "financial_services",
        "general_educational",
        "general_open_ended",
        "health_decision_support_disclosure",
        "health_general_disclosure",
        "health_therapy_disclosure",
        "health_triage_disclosure",
        "identity_disclosure",
        "safety_agentic",
        "safety_chat_pressure",
        "sov_ai_dynamic",
        "sov_ai_facts",
        "sov_ai_multilingual_parity",
        "tool_calling",
    }
)
ASSISTANT_QUALITY_AXES = ("helpfulness", "accuracy", "coherence")
logger = logging.getLogger(__name__)


class UserSimResourcesServerConfig(BaseResourcesServerConfig):
    personas_cache_dir: Path = Path("~/.cache/nemo-gym/usersim/personas")
    personas_dataset_version: str = Field("0.0.2", pattern=r"^[A-Za-z0-9._-]+$")
    usersim_revision: str = Field(
        "2d9ec0d7c32ac800f2171b5943382a7b1eb96cbc",  # pragma: allowlist secret
        pattern=r"^[0-9a-f]{40}$",
    )
    personas_locales: list[str] = Field(default_factory=lambda: ["en_US"])
    probe_scorer_model: ModelServerRef | None = None
    model_call_timeout_seconds: float = Field(300.0, gt=0)
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
            probe
            for probe, weight in self.probe_mix.items()
            if probe in THEME_DRIVEN_PROBES and weight > 0 and not self.probe_themes.get(probe)
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


class SeededUserSimEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task_id: TaskId
    seed: UserSimSeedResponse


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


class _ResourcesModelFacade:
    """Async UserSim facade backed by a Gym Model Server."""

    def __init__(
        self,
        server: "UserSimResourcesServer",
        model: ModelServerRef,
    ) -> None:
        self.server = server
        self.model = model
        self.model_name = model.name

    async def acompletion(self, messages: Sequence[Any], **kwargs: Any) -> SimpleNamespace:
        unsupported = set(kwargs) - {"max_tokens", "max_completion_tokens", "response_format"}
        if unsupported:
            raise NotImplementedError(f"Unsupported UserSim support-model options: {sorted(unsupported)}")
        max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
        try:
            async with asyncio.timeout(self.server.config.model_call_timeout_seconds):
                return await self._completion(
                    messages,
                    max_tokens=max_tokens,
                    response_format=kwargs.get("response_format"),
                )
        except TimeoutError as error:
            raise TimeoutError(
                f"Timed out after {self.server.config.model_call_timeout_seconds}s waiting for {self.model.name}"
            ) from error

    async def _completion(
        self,
        messages: Sequence[Any],
        *,
        max_tokens: int | None,
        response_format: Mapping[str, Any] | None,
    ) -> SimpleNamespace:
        params: dict[str, Any] = {"input": [_to_responses_input(message) for message in messages]}
        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens
        if response_format is not None:
            json_schema = response_format.get("json_schema")
            if response_format.get("type") != "json_schema" or not isinstance(json_schema, Mapping):
                raise NotImplementedError(f"Unsupported response format: {response_format!r}")
            params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": json_schema["name"],
                    "schema": json_schema["schema"],
                    "strict": json_schema.get("strict", True),
                }
            }
        response = await self.server.server_client.post(
            server_name=self.model.name,
            url_path="/v1/responses",
            json=NeMoGymResponseCreateParamsNonStreaming.model_validate(params),
        )
        await raise_for_status(response)
        gym_response = NeMoGymResponse.model_validate(await get_response_json(response))
        usage = gym_response.usage
        return SimpleNamespace(
            message=SimpleNamespace(content=_response_text(gym_response), reasoning_content=None, tool_calls=None),
            usage=(
                SimpleNamespace(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
                if usage is not None
                else None
            ),
        )


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
                "Run `gym eval prepare --config environments/usersim/config.yaml` before starting the "
                "Resources Server."
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
        themes = self.config.probe_themes.get(probe_type) or [
            UserSimTheme(
                topic=probe_type.replace("_", " "),
                goal=f"Run the {probe_type} UserSim probe.",
            )
        ]
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
                usersim_revision=self.config.usersim_revision,
            ),
        )

    def _seeded_episode(self, request: Request) -> SeededUserSimEpisode:
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self.session_id_to_seed:
            raise RuntimeError("No active NeMo UserSim scenario. Call /seed_session first.")
        return self.session_id_to_seed[session_id]

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
        result = self._resolve_seed(task.sampling, body.resources_session_id)
        result = result.model_copy(
            update={"scenario": result.scenario.model_copy(update={"probe_data": task.probe_data})}
        )
        self.session_id_to_seed[session_id] = SeededUserSimEpisode(
            episode_id=body.episode_id,
            task_id=body.task_id,
            seed=result,
        )
        return result

    async def _score_native_result(
        self,
        seeded: SeededUserSimEpisode,
        native_result: UserSimSimulationResult,
    ) -> tuple[str | None, dict[str, Any] | None, bool]:
        scenario = seeded.seed.scenario
        scorer_name = PROBE_SCORERS.get(scenario.probe_type)
        if (
            scorer_name == "health_disclosure_concealment"
            and scenario.probe_data.get("probe_variant", "default") != "guarded"
        ):
            return None, None, True
        if scorer_name is None:
            return None, None, True

        trajectory = {
            **native_result.model_dump(mode="python"),
            "locale": scenario.locale,
            "persona": scenario.persona,
            "probe_type": scenario.probe_type,
        }
        if self.config.probe_scorer_model is not None:
            scorer_models = {
                "judge_model": _ResourcesModelFacade(
                    self,
                    self.config.probe_scorer_model,
                )
            }
        else:
            scorer_models = {}

        try:
            from usersim.engine.evaluator.scorers import get_scorer

            scores = await get_scorer(scorer_name)(trajectory, scorer_models)
        except Exception as error:
            logger.exception("Native UserSim scorer %s failed", scorer_name)
            scores = {
                "status_proposal": False,
                "error": f"{type(error).__name__}: {error}",
            }
        passed = scores.get("status_proposal") is True and not scores.get("error")
        return scorer_name, scores, passed

    async def _evaluate_assistant_quality(
        self,
        seeded: SeededUserSimEpisode,
        native_result: UserSimSimulationResult,
    ) -> tuple[dict[str, Any], dict[str, float], float | None]:
        from usersim.engine.core.behavioral import get_conversation_language
        from usersim.engine.evaluator.runtime import TrajectoryEvaluatorRuntime
        from usersim.taxonomy.eval_cell import normalize_axis_score, score_from_eval_cell

        if self.config.probe_scorer_model is None:
            return (
                {
                    "axes": {},
                    "scorers": {},
                    "skipped": True,
                    "skipped_reason": "missing_probe_scorer_model",
                },
                {},
                None,
            )

        model = _ResourcesModelFacade(self, self.config.probe_scorer_model)
        evaluator = TrajectoryEvaluatorRuntime(models={"judge_model": model})
        scenario = seeded.seed.scenario
        evaluation = await evaluator.evaluate(
            {
                **native_result.model_dump(mode="python"),
                "persona": scenario.persona,
                "probe_family": scenario.probe_type,
                "probe_variant": scenario.probe_data.get("probe_variant"),
                "locale": scenario.locale,
                "conversation_language": get_conversation_language(scenario.locale),
            }
        )
        normalized_scores: dict[str, float] = {}
        for axis in evaluation.get("envelope", {}).get("axes", []):
            score = score_from_eval_cell(evaluation, axis)
            if score is not None:
                normalized_scores[axis] = normalize_axis_score(axis, score)

        quality_scores = [normalized_scores.get(axis) for axis in ASSISTANT_QUALITY_AXES]
        assistant_quality = (
            sum(score for score in quality_scores if score is not None) / len(ASSISTANT_QUALITY_AXES)
            if all(score is not None for score in quality_scores)
            else None
        )
        return evaluation, normalized_scores, assistant_quality

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
        native_result = verification_input.usersim_result
        native_scorer_name, native_scores, native_scorer_pass = await self._score_native_result(
            seeded,
            native_result,
        )
        try:
            assistant_eval, normalized_axis_scores, assistant_quality = await self._evaluate_assistant_quality(
                seeded,
                native_result,
            )
        except Exception as error:
            logger.exception("Native UserSim trajectory evaluator failed")
            assistant_eval = {
                "axes": {},
                "scorers": {},
                "error": f"{type(error).__name__}: {error}",
            }
            normalized_axis_scores = {}
            assistant_quality = None
        outcome = native_result.simulation_outcome
        native_failure = outcome.get("status") == "failed"
        failure_attribution = outcome.get("failure_attribution")
        evaluator_error = assistant_eval.get("error")
        scorer_error = native_scores.get("error") if native_scores is not None else None
        mask_sample = (native_failure and failure_attribution != "assistant_model") or bool(
            scorer_error or evaluator_error
        )
        failure_kind = outcome.get("failure_class") if native_failure else None
        failure_reason = outcome.get("failure_detail") if native_failure else None
        if scorer_error:
            failure_kind = "probe_scorer_error"
            failure_reason = str(scorer_error)
        elif evaluator_error:
            failure_kind = "trajectory_evaluator_error"
            failure_reason = str(evaluator_error)
        participants_completed = {"user", "assistant"} <= _conversation_roles(native_result)
        scenario_completed = native_result.conversation_status and participants_completed and native_scorer_pass
        reward = assistant_quality if scenario_completed and assistant_quality is not None else 0.0
        return UserSimVerification(
            reward=reward,
            mask_sample=mask_sample,
            failure_kind=failure_kind,
            failure_reason=failure_reason,
            reward_components={
                "participants_completed": float(participants_completed),
                "native_conversation_status": float(native_result.conversation_status),
                "native_scorer_applied": float(native_scorer_name is not None),
                "native_scorer_pass": float(native_scorer_pass),
                "trajectory_evaluator_applied": float(not assistant_eval.get("skipped", False)),
                "assistant_quality": assistant_quality or 0.0,
                **{f"quality.{axis}": score for axis, score in normalized_axis_scores.items()},
            },
            scenario_completed=scenario_completed,
            native_usersim_result=native_result,
            verifier_data={
                "invocations": [invocation.model_dump(mode="json") for invocation in verification_input.invocations],
                "episode_interaction_protocol": verification_input.episode_interaction_protocol,
                "scenario": verification_input.scenario.model_dump(mode="json"),
                "usersim_context": verification_input.usersim_context.model_dump(mode="json"),
                "usersim_result": native_result.model_dump(mode="json"),
                "native_scorer_name": native_scorer_name,
                "native_scores": native_scores,
                "assistant_eval": assistant_eval,
                "normalized_axis_scores": normalized_axis_scores,
                "scenario_completed": scenario_completed,
            },
        )

    async def close_resources_session(
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


def _to_responses_input(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        value = message.model_dump(mode="json", exclude_none=True)
    elif isinstance(message, Mapping):
        value = dict(message)
    else:
        value = {"role": getattr(message, "role"), "content": getattr(message, "content", "")}
    role = getattr(value.get("role"), "value", value.get("role"))
    return {"type": "message", "role": role, "content": value.get("content", "")}


def _output_message_text(message: NeMoGymResponseOutputMessage) -> str:
    chunks: list[str] = []
    for content in message.content:
        text = getattr(content, "text", None) or getattr(content, "refusal", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks)


def _response_text(response: NeMoGymResponse) -> str:
    return "\n".join(
        text
        for item in response.output
        if isinstance(item, NeMoGymResponseOutputMessage)
        if (text := _output_message_text(item))
    )


if __name__ == "__main__":
    UserSimResourcesServer.run_webserver()
