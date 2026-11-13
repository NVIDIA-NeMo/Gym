# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-answer coding training with remote rendering and a trusted quality judge."""

import asyncio
import hashlib
import json
import logging
from itertools import islice
from pathlib import Path
from typing import Any, ClassVar, Literal

from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field, PrivateAttr, ValidationError

from nemo_gym.agents import AGENT_OUTCOME_METADATA_KEY, AgentOutcome
from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import ModelServerRef, get_global_config_dict
from nemo_gym.core import (
    VERIFIER_ERROR,
    FailureClass,
    FailureReport,
    NeMoGymChatCompletion,
    NeMoGymResponse,
    NgFailureEvent,
    classify_failure,
)
from nemo_gym.sandbox import SandboxSpec, resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSeedRejectedError,
    ReverifyMode,
    SimpleResourcesServer,
    lost_session_verification,
    observe_llm_judge_grading,
    observe_programmatic_grading,
)
from nemo_gym.utils import explicit_causes, is_nemo_gym_fastapi_entrypoint
from nemo_gym.utils.exception_inspection import qualified_type_name
from nemo_gym.verification import JudgeError, VerifierFixture, call_judge, record_parsed_judge_verdict
from resources_servers.mm_coding_training.extraction import embeds_image_data, extract_answer
from resources_servers.mm_coding_training.quality import Quality, QualityVerifier, ScoringInput, final_reward
from resources_servers.mm_coding_training.remote_renderer import LANGUAGES, RemoteRenderer, image_uri
from resources_servers.mm_coding_training.task_data import CodingTask, TaskData


CONTRACT = json.loads(Path(__file__).with_name("contract.json").read_text())
logger = logging.getLogger(__name__)


def log_verification_failure(error: Exception, *, stage: str) -> None:
    """Log bounded cause types without SDK messages, requests or credentials."""
    causes = " <- ".join(qualified_type_name(cause)[:160] for cause in islice(explicit_causes(error), 8))
    logger.warning("Masked coding verification failure: stage=%s causes=%s", stage, causes)


class CodingConfig(BaseResourcesServerConfig):
    """Deployment settings; providers and judge keys are resolved by Gym."""

    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    reference_root: Path
    sandbox_provider: str | dict[str, Any]
    sandbox_image: str = Field(pattern=r"^\S+@sha256:[a-f0-9]{64}$")
    sandbox_memory_mib: int = Field(default=16384, ge=1024)
    sandbox_cpus: int = Field(default=4, ge=1)
    sandbox_ttl_s: int = Field(default=14400, ge=600)
    sandbox_ready_timeout_s: float = Field(default=600, gt=0)
    execution_timeout_s: float = Field(default=420, gt=0)
    inconclusive_verification_retries: int = Field(default=1, ge=0, le=2)
    verification_concurrency: int = Field(default=8, ge=1)
    judge_concurrency: int = Field(default=8, ge=1)
    judge_model_server: ModelServerRef
    judge_model: str
    judge_api: Literal["responses", "chat_completions"] = "responses"
    judge_max_output_tokens: int = Field(default=2048, ge=1)
    max_source_bytes: int = Field(default=4 * 1024 * 1024, ge=1)


class CodingRequest(TaskData, BaseVerifyRequest):
    """Policy response plus separately carried task-owned grading metadata."""

    # The legacy agent forwards routing/index fields alongside the strict task.
    model_config = ConfigDict(extra="allow")
    evaluation_completed: bool = False
    source_artifact: str | None = Field(default=None, repr=False)
    source_sha256: str | None = None
    source_binding_sha256: str | None = None


class CodingSeedRequest(TaskData, BaseSeedSessionRequest):
    """Validate the verifier-owned task at the public episode seed boundary."""


class CodingResponse(TaskData, BaseVerifyResponse):
    """Training reward and component diagnostics, bound to the graded artifacts."""

    model_config = ConfigDict(extra="allow")

    task_id: str
    evaluation_completed: bool = False
    execution: float = 0.0
    task_fidelity: float | None = None
    visual_fidelity: float | None = None
    judge_reason: str | None = None
    answer_sha256: str | None = None
    source_artifact: str | None = Field(default=None, repr=False)
    source_sha256: str | None = None
    source_binding_sha256: str | None = None
    render_sha256: str | None = None
    reference_sha256: str | None = None


class CodingTrainingServer(SimpleResourcesServer):
    """Stateless verification; every candidate executes in its own disposable sandbox."""

    checkpoint_mode = "stateless"
    checkpoint_verify = "replay"
    config: CodingConfig
    seed_request_model: ClassVar[type[BaseSeedSessionRequest]] = CodingSeedRequest
    _tasks: dict[str, CodingSeedRequest] = PrivateAttr(default_factory=dict)
    _verification_slots: asyncio.Semaphore = PrivateAttr()
    _judge_slots: asyncio.Semaphore = PrivateAttr()

    def has_session_state(self, session_id: str) -> bool:
        return session_id in self._tasks

    async def initialize_session_state(self, session_id: str, body: CodingSeedRequest) -> None:
        try:
            self._reference(body.task)
        except (OSError, ValueError) as error:
            raise ResourcesSeedRejectedError(
                "Coding reference is unavailable or differs from its declared SHA256"
            ) from error
        self._tasks[session_id] = body.model_copy(deep=True)

    async def release_session_state(self, session_id: str) -> None:
        self._tasks.pop(session_id, None)

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        root = self.config.reference_root
        if not root.is_absolute():
            root = Path(__file__).resolve().parents[2] / root
        self.config.reference_root = root.resolve()
        self._verification_slots = asyncio.Semaphore(self.config.verification_concurrency)
        self._judge_slots = asyncio.Semaphore(self.config.judge_concurrency)

    def _reference(self, task: CodingTask) -> bytes | None:
        if task.reference_image is None:
            return None
        root = self.config.reference_root.resolve()
        path = (root / task.reference_image).resolve()
        if not path.is_relative_to(root):
            logger.warning("Reference validation failed: path escapes the configured root")
            raise ValueError("Reference image is outside the configured root")
        if not path.is_file():
            logger.warning("Reference validation failed: configured image is missing")
            raise FileNotFoundError("Configured reference image is missing")
        if path.stat().st_size > 32 * 1024 * 1024:
            logger.warning("Reference validation failed: image exceeds the byte limit")
            raise ValueError("Reference image is too large")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != task.reference_sha256:
            logger.warning("Reference validation failed: image differs from the task SHA256")
            raise ValueError("Reference image differs from the task SHA256")
        image_uri(data)
        return data

    def _renderer(self) -> RemoteRenderer:
        global_config = get_global_config_dict()
        provider = resolve_provider_config(self.config.sandbox_provider, global_config)
        if "local" in provider:
            raise ValueError("Candidate rendering requires an isolated sandbox provider, not host execution")
        metadata = {
            **resolve_provider_metadata(self.config.sandbox_provider, global_config),
            "nemo_gym_server": self.config.name[:63],
        }
        return RemoteRenderer(
            provider,
            SandboxSpec(
                image=self.config.sandbox_image,
                ttl_s=self.config.sandbox_ttl_s,
                resources={"cpu": self.config.sandbox_cpus, "memory_mib": self.config.sandbox_memory_mib},
                metadata=metadata,
            ),
            execution_timeout=self.config.execution_timeout_s,
            ready_timeout=self.config.sandbox_ready_timeout_s,
        )

    async def _judge(self, task: CodingTask, source: str, png: bytes, reference: bytes | None) -> Quality:
        rubric = CONTRACT["rubrics"][task.benchmark]
        evidence: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": json.dumps(
                    {
                        "task": task.prompt,
                        "benchmark": task.benchmark,
                        "semantic_prompt": task.semantic_prompt,
                        "rubric": rubric,
                        "candidate_source": source,
                        "original_source": "",
                        "reference_patch": "",
                        "reference_code_or_patch": task.ground_truth,
                        "reference_render_available": reference is not None,
                    }
                ),
            }
        ]
        if reference is not None:
            evidence += [
                {"type": "input_text", "text": "REFERENCE target rendering"},
                {"type": "input_image", "image_url": image_uri(reference), "detail": "auto"},
            ]
        evidence += [
            {"type": "input_text", "text": "CANDIDATE rendering to grade"},
            {"type": "input_image", "image_url": image_uri(png), "detail": "auto"},
        ]
        payload = {
            "model": self.config.judge_model,
            "instructions": CONTRACT["judge_instructions"] + rubric,
            "input": [{"role": "user", "content": evidence}],
            "max_output_tokens": self.config.judge_max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "coding_quality",
                    "strict": True,
                    "schema": CONTRACT["quality_schema"],
                }
            },
        }
        async with self._judge_slots:
            if self.config.judge_api == "chat_completions":
                content = [
                    {"type": "text", "text": part["text"]}
                    if part["type"] == "input_text"
                    else {"type": "image_url", "image_url": {"url": part["image_url"], "detail": part["detail"]}}
                    for part in evidence
                ]
                chat_completion = await call_judge(
                    self.server_client,
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/chat/completions",
                    json={
                        "model": self.config.judge_model,
                        "messages": [
                            {"role": "system", "content": payload["instructions"]},
                            {"role": "user", "content": content},
                        ],
                        "max_completion_tokens": self.config.judge_max_output_tokens,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                key: value for key, value in payload["text"]["format"].items() if key != "type"
                            },
                        },
                    },
                    response_model=NeMoGymChatCompletion,
                )
                choice = chat_completion.choices[0] if len(chat_completion.choices) == 1 else None
                complete = choice is not None and choice.finish_reason == "stop"
                verdict_text = choice.message.content if choice is not None else None
            else:
                completion = await call_judge(
                    self.server_client,
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/responses",
                    json=payload,
                    response_model=NeMoGymResponse,
                )
                complete = completion.status == "completed"
                verdict_text = completion.output_text
        try:
            if not complete or not isinstance(verdict_text, str):
                raise ValueError("Judge response is incomplete")
            verdict = Quality.model_validate_json(verdict_text)
        except (ValueError, ValidationError) as error:
            raise JudgeError("Judge did not return a complete, consistent quality verdict") from error
        record_parsed_judge_verdict()
        return verdict

    async def verify(self, request: Request, body: CodingRequest) -> CodingResponse:
        session_id = request.session[SESSION_ID_KEY]
        async with self.verifying(session_id, allow_missing=True) as verification:
            if verification.is_lost:
                return self.lost_result(body)
            seeded = self._tasks.get(session_id)
            if seeded is not None and seeded.task != body.task:
                raise HTTPException(409, "Verification does not match the seeded task")
            verification.retain(None)
            return await self._verify_answer(body)

    def new_result(self, body: CodingRequest) -> CodingResponse:
        return CodingResponse(
            **(
                body.model_dump()
                | {"reward": 0.0, "task_id": body.task.task_id, "mask_sample": False, "evaluation_completed": False}
            )
        )

    def lost_result(self, body: CodingRequest) -> CodingResponse:
        return CodingResponse(
            **(
                body.model_dump()
                | lost_session_verification("Coding task session is closed or lost")
                | {"task_id": body.task.task_id, "evaluation_completed": False}
            )
        )

    async def _verify_answer(self, body: CodingRequest) -> CodingResponse:
        result = self.new_result(body)
        if not self.response_is_measurable(body):
            result.mask_sample = True
            result.failure_kind = VERIFIER_ERROR
            result.failure_reason = "Policy response is not a completed measurement or token-budget exhaustion"
            return result
        text = "".join(
            content.text
            for item in body.response.output
            if item.type == "message" and item.role == "assistant"
            for content in item.content
            if content.type == "output_text"
        )
        result.answer_sha256 = hashlib.sha256(text.encode()).hexdigest()
        try:
            if body.response.status == "incomplete":
                raise ValueError("Generation exhausted its budget")
            source = extract_answer(text, LANGUAGES[body.task.benchmark][0])
            if len(source.encode()) > self.config.max_source_bytes:
                raise ValueError("Answer exceeds the source byte budget")
        except ValueError:
            result.failure_kind = "mm_coding_training:invalid_artifact"
            result.evaluation_completed = True
            return observe_programmatic_grading(lambda: result)
        return await self.grade_source(body, source, result)

    def response_is_measurable(self, body: CodingRequest) -> bool:
        """Admit classified model failures while retaining infrastructure and unknown masks."""
        response = body.response
        if response.status in {"completed", "incomplete"}:
            return True
        if response.status != "failed":
            return False
        try:
            outcome = AgentOutcome.model_validate_json((response.metadata or {}).get(AGENT_OUTCOME_METADATA_KEY, ""))
            # The calling boundary selected verification, possibly through a custom outcome policy.
            if (
                classify_failure(failure_kind=outcome.failure_kind, http_status=outcome.status_code)
                != FailureClass.MODEL
            ):
                return False
            for report in (
                (body.model_extra or {}).get("failure_report"),
                (response.model_extra or {}).get("failure_report"),
            ):
                if report is not None:
                    saved = FailureReport.model_validate(report)
                    if saved.failure_id is not None:
                        selected = saved.events[saved.failure_id]
                        if (
                            classify_failure(
                                failure_kind=selected.failure_kind,
                                stage=selected.stage,
                                http_status=selected.http_status,
                            )
                            != FailureClass.MODEL
                        ):
                            return False
        except ValidationError:
            return False
        return True

    async def grade_source(self, body: CodingRequest, source: str, result: CodingResponse) -> CodingResponse:
        """Grade captured source in an independent verifier sandbox, preserving the policy response."""
        result.source_artifact = source
        result.source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        # The shared judge failsafe copies this typed body into its failure-sidecar row.
        body.source_artifact = source
        body.source_sha256 = result.source_sha256
        body.evaluation_completed = False
        if embeds_image_data(source):
            # A rendered copy of the input image is not a solved task.
            result.failure_kind = "mm_coding_training:invalid_artifact"
            result.evaluation_completed = True
            return observe_programmatic_grading(lambda: result)
        async with self._verification_slots:
            stage = "reference"
            try:
                reference = self._reference(body.task)
                stage = "render"
                for attempt in range(self.config.inconclusive_verification_retries + 1):
                    try:
                        png = await self._renderer().render(body.task.benchmark, source)
                        break
                    except (OSError, ValueError, RuntimeError, NgFailureEvent):
                        if attempt == self.config.inconclusive_verification_retries:
                            raise
                if png is None:
                    result.failure_kind = "mm_coding_training:execution_failed"
                    result.evaluation_completed = True
                    return observe_programmatic_grading(lambda: result)
                result.execution = 1.0
                result.render_sha256 = hashlib.sha256(png).hexdigest()
                result.reference_sha256 = body.task.reference_sha256

                async def grade() -> CodingResponse:
                    verdict = await self._judge(body.task, source, png, reference)
                    result.reward = final_reward(ScoringInput(executed=True, verdict=verdict))
                    result.task_fidelity = verdict.task_fidelity
                    result.visual_fidelity = verdict.visual_fidelity
                    result.judge_reason = verdict.reason
                    result.evaluation_completed = True
                    return result

                stage = "judge"
                return await observe_llm_judge_grading(grade)
            except JudgeError as error:
                log_verification_failure(error, stage=stage)
                raise
            except NgFailureEvent as error:
                log_verification_failure(error, stage=stage)
                if stage == "judge":
                    raise
                result.mask_sample = True
                result.failure_kind = error.failure_kind
                result.failure_reason = "Verification dependency failed; retain this answer for reverification"
            except (OSError, ValueError, RuntimeError) as error:
                log_verification_failure(error, stage=stage)
                result.mask_sample = True
                result.failure_kind = VERIFIER_ERROR
                result.failure_reason = "Reference or remote renderer failed; no model-quality measurement"
        return result


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=QualityVerifier,
    request_model=ScoringInput,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    CodingTrainingServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = CodingTrainingServer.run_webserver()
