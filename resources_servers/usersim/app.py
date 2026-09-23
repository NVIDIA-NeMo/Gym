# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic NeMo UserSim scenario initialization backed by managed personas."""

import asyncio
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow.parquet as pq
from fastapi import Body, FastAPI, HTTPException, Request
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
    NeMoGymResponseFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, raise_for_status
from nemo_gym.usersim_episode_types import (
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


TOOL_PROBES = frozenset({"tool_calling", "safety_agentic", "financial_services"})
SUPPORTED_PROBES = frozenset({"general_open_ended", "general_educational", *TOOL_PROBES})
logger = logging.getLogger(__name__)


class UserSimResourcesServerConfig(BaseResourcesServerConfig):
    personas_cache_dir: Path = Path("~/.cache/nemo-gym/usersim/personas")
    personas_dataset_version: str = Field("0.0.2", pattern=r"^[A-Za-z0-9._-]+$")
    usersim_revision: str = Field(
        "693865d7b33c3d96283a9742703c8c89413a9d2b",
        pattern=r"^[0-9a-f]{40}$",
    )
    personas_locales: list[str] = Field(default_factory=lambda: ["en_US"])
    api_response_model: ModelServerRef | None = None
    judge_model: ModelServerRef | None = None
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


class SeededUserSimEpisode(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    episode_id: EpisodeId
    task_id: TaskId
    seed: UserSimSeedResponse
    runtime: Any | None = None


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


def _external_probe_transcript(
    messages: list[dict[str, Any]],
    invocations: Sequence[Any],
) -> list[dict[str, Any]]:
    """Replace collapsed Assistant turns with the Agent's full tool transcript."""
    assistant_responses = iter(
        invocation.response for invocation in invocations if invocation.alias == "assistant_model"
    )
    transcript: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant":
            transcript.append(message)
            continue
        response = next(assistant_responses, None)
        if response is None:
            transcript.append(message)
            continue
        converted = _response_output_messages(response)
        transcript.extend(converted or [message])
    return transcript


def _response_output_messages(response: NeMoGymResponse) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    pending_calls: list[dict[str, Any]] = []

    def flush_calls() -> None:
        if pending_calls:
            messages.append({"role": "assistant", "content": "", "tool_calls": list(pending_calls)})
            pending_calls.clear()

    for item in response.output:
        if isinstance(item, NeMoGymResponseFunctionToolCall):
            pending_calls.append(
                {
                    "id": item.call_id,
                    "type": "function",
                    "function": {"name": item.name, "arguments": item.arguments},
                }
            )
            continue
        flush_calls()
        if isinstance(item, NeMoGymResponseFunctionCallOutput):
            messages.append(
                {
                    "role": "tool",
                    "content": item.output if isinstance(item.output, str) else json.dumps(item.output),
                    "tool_call_id": item.call_id,
                }
            )
        elif isinstance(item, NeMoGymResponseOutputMessage):
            messages.append({"role": "assistant", "content": _output_message_text(item)})
    flush_calls()
    return messages


class _ResourcesModelFacade:
    """Synchronous UserSim facade backed by a Gym Model Server."""

    def __init__(
        self,
        server: "UserSimResourcesServer",
        model: ModelServerRef,
        event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.server = server
        self.model = model
        self.model_name = model.name
        self.event_loop = event_loop

    def completion(self, messages: Sequence[Any], **kwargs: Any) -> SimpleNamespace:
        unsupported = set(kwargs) - {"max_tokens", "max_completion_tokens", "response_format"}
        if unsupported:
            raise NotImplementedError(f"Unsupported UserSim support-model options: {sorted(unsupported)}")
        max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
        future = asyncio.run_coroutine_threadsafe(
            self._completion(
                messages,
                max_tokens=max_tokens,
                response_format=kwargs.get("response_format"),
            ),
            self.event_loop,
        )
        try:
            return future.result(timeout=self.server.config.model_call_timeout_seconds)
        except FutureTimeoutError as error:
            future.cancel()
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

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/close_session")(self.close_session)
        app.post("/{tool_name}")(self.invoke_probe_tool)
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
        result = await asyncio.to_thread(self._resolve_seed, task.sampling, session_id)
        result = result.model_copy(
            update={"scenario": result.scenario.model_copy(update={"probe_data": task.probe_data})}
        )
        runtime = None
        if result.scenario.probe_type in TOOL_PROBES:
            runtime = await asyncio.to_thread(
                self._create_probe_runtime,
                result.scenario,
                task,
                asyncio.get_running_loop(),
            )
            scenario = result.scenario
            if scenario.probe_type == "tool_calling":
                scenario = scenario.model_copy(
                    update={"probe_data": {**scenario.probe_data, "tools": runtime.assistant_tools}}
                )
            result = result.model_copy(update={"scenario": scenario, "assistant_tools": runtime.assistant_tools})
        self.session_id_to_seed[session_id] = SeededUserSimEpisode(
            episode_id=body.episode_id,
            task_id=body.task_id,
            seed=result,
            runtime=runtime,
        )
        return result

    def _create_probe_runtime(
        self,
        scenario: UserSimScenario,
        task: UserSimTaskInput,
        event_loop: asyncio.AbstractEventLoop,
    ) -> Any:
        from usersim.engine.config import ConversationSimulatorConfig
        from usersim.engine.core.behavioral import compute_behavioral_profile, get_conversation_language
        from usersim.engine.core.episode_runtime import ProbeEpisodeRuntime

        if scenario.probe_type == "tool_calling" and self.config.api_response_model is None:
            raise ValueError("tool_calling requires resources api_response_model configuration")
        models = {}
        if self.config.api_response_model is not None:
            models["api_response_model"] = _ResourcesModelFacade(
                self,
                self.config.api_response_model,
                event_loop,
            )
        if self.config.judge_model is not None:
            models["judge_model"] = _ResourcesModelFacade(
                self,
                self.config.judge_model,
                event_loop,
            )
        data = {
            **task.probe_data,
            "persona": scenario.persona,
            "probe_type": scenario.probe_type,
            "theme": scenario.theme,
        }
        config = ConversationSimulatorConfig(
            name="gym_probe_episode_runtime",
            locale=scenario.locale,
            random_seed=task.sampling.seed,
            tools_column="tools" if scenario.probe_type == "tool_calling" else None,
            finance_retrieval_mode="golden",
        )
        return ProbeEpisodeRuntime(
            probe_type=scenario.probe_type,
            persona=scenario.persona,
            locale=scenario.locale,
            language=get_conversation_language(scenario.locale),
            models=models,
            config=config,
            data=data,
            profile=compute_behavioral_profile(scenario.persona),
        )

    async def invoke_probe_tool(
        self,
        request: Request,
        tool_name: str,
        body: dict[str, Any] = Body(),
    ) -> Any:
        """Simulate one tool selected for the request's seeded episode."""
        seeded = self._seeded_episode(request)
        if seeded.runtime is None:
            raise HTTPException(status_code=404, detail="This episode does not expose probe tools")
        try:
            payload = await asyncio.to_thread(seeded.runtime.simulate_tool_call, tool_name, body)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {"result": payload}

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
        if seeded.runtime is not None:
            transcript = _external_probe_transcript(
                verification_input.usersim_result.conversation_messages,
                verification_input.invocations,
            )
            native_result = UserSimSimulationResult.model_validate(
                await asyncio.to_thread(seeded.runtime.finalize, transcript)
            )
        native_scores: dict[str, Any] | None = None
        native_scorer_pass = True
        if seeded.runtime is not None:
            from usersim.engine.evaluator.scorers import get_scorer, load_default_scorers

            load_default_scorers()
            scorer_name = (
                "tool_use" if seeded.seed.scenario.probe_type == "tool_calling" else seeded.seed.scenario.probe_type
            )
            trajectory = {
                **native_result.model_dump(mode="python"),
                **seeded.runtime.evidence()["result_extras"],
                "locale": seeded.seed.scenario.locale,
                "persona": seeded.seed.scenario.persona,
                "probe_type": seeded.seed.scenario.probe_type,
            }
            scorer_models = {alias: model for alias, model in seeded.runtime.models.items() if alias == "judge_model"}
            native_scores = await asyncio.to_thread(get_scorer(scorer_name), trajectory, scorer_models)
            native_scorer_pass = bool(native_scores.get("status_proposal", False)) and not native_scores.get("error")
        participants_completed = {"user", "assistant"} <= _conversation_roles(native_result)
        scenario_completed = native_result.conversation_status and participants_completed and native_scorer_pass
        return UserSimVerification(
            reward=float(scenario_completed),
            reward_components={
                "participants_completed": float(participants_completed),
                "native_conversation_status": float(native_result.conversation_status),
                "native_scorer_pass": float(native_scorer_pass),
            },
            scenario_completed=scenario_completed,
            native_usersim_result=native_result if seeded.runtime is not None else None,
            verifier_data={
                "invocations": [invocation.model_dump(mode="json") for invocation in verification_input.invocations],
                "episode_interaction_protocol": verification_input.episode_interaction_protocol,
                "scenario": verification_input.scenario.model_dump(mode="json"),
                "usersim_context": verification_input.usersim_context.model_dump(mode="json"),
                "usersim_result": native_result.model_dump(mode="json"),
                "native_scores": native_scores,
                "scenario_completed": scenario_completed,
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
