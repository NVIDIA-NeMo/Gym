# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Open-ended (video) questions graded against each question's weighted-points rubric by a judge model.

The final answer goes to the judge with the question and the rubric (no video, reasoning or reference answer); the
judge reports the points of every criterion and the contradictions it found, and the reward is the checked total /
100 (rubric.py). The judge is any OpenAI-compatible chat model with JSON-schema structured output, reached through a
judge ``ModelServerRef`` and Gym's ``call_judge``.
"""

import asyncio
import json
import logging
import math
import random
from collections import Counter
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NgFailureEvent,
)
from nemo_gym.models import ChatCompletionsConverter
from nemo_gym.server_runtime import (
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    observe_llm_judge_grading,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import (
    JUDGE_FAILURES,
    JudgeError,
    call_judge,
    is_judge_failure,
    record_parsed_judge_verdict,
    record_rejected_judge_reply,
)
from resources_servers.video_mcqa.app import response_failure
from resources_servers.video_mcqa.sandbox import VideoSandboxServer, VideoSandboxServerConfig
from resources_servers.video_mcqa.scoring import final_answer_text
from resources_servers.video_mcqa.task import Setting
from resources_servers.video_qa_rubric.rubric import (
    JUDGMENT_SCHEMA,
    JUDGMENT_SCHEMA_NAME,
    judge_messages,
    judgment_points,
)
from resources_servers.video_qa_rubric.task_data import RubricDict, Text


logger = logging.getLogger(__name__)

Outcome = Literal["graded", "empty_answer", "execution_failure", "infrastructure_failure"]
OUTCOMES: tuple[Outcome, ...] = ("graded", "empty_answer", "execution_failure", "infrastructure_failure")
# Judge replies that fail the same way again: every 4xx except a timeout (408) or rate limit (429).
RETRYABLE_JUDGE_STATUSES = frozenset({408, 429})
# Fields a verification writes; a verified row sent again must not carry them in twice.
VERIFICATION_FIELDS = frozenset(
    {
        "reward",
        "mask_sample",
        "failure_kind",
        "failure_reason",
        "outcome",
        "rubric_score",
        "rubric_points",
        "judgment",
        "judge_usage",
        "judge_attempts",
        "evaluation_completed",
    }
)


class VideoQARubricResourcesServerConfig(VideoSandboxServerConfig):
    """The judge and its retry budget; ``empty_answer_reward`` is a training option. The sandbox settings (video_mcqa's
    sandbox.py) let the agentic setting lend each episode a sandbox that holds the row's video."""

    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    judge_model_server: ModelServerRef
    # Sampling and reasoning settings of the judge's chat request; the server sets messages and response_format.
    judge_chat_completion_create_params: NeMoGymChatCompletionCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymChatCompletionCreateParamsNonStreaming(messages=[])
    )
    judge_endpoint_max_concurrency: int = Field(default=64, ge=1)
    # Attempts per answer: a failed or timed-out call and a reply the rubric's arithmetic rejects each use one.
    judge_max_attempts: int = Field(default=3, ge=1)
    judge_attempt_timeout_seconds: float = Field(default=600, gt=0, allow_inf_nan=False)
    judge_retry_base_delay_seconds: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    judge_retry_max_delay_seconds: float = Field(default=30.0, ge=0, allow_inf_nan=False)
    # Training only, as video_mcqa's option: the reward of an attempt that produced output but no final answer
    # (reasoning only, an unfinished reasoning block, an empty message). A response without any output stays 0: Gym's
    # agents verify a failed run (execution_failure_reward_zero) as an empty response and require exactly 0. None
    # scores attempts without an answer 0.
    empty_answer_reward: float | None = Field(default=None, le=0, allow_inf_nan=False)


class VideoQARubricMetadata(BaseModel):
    """Verifier-only question data: the rubric never reaches the model."""

    model_config = ConfigDict(extra="allow")
    question: Text
    rubric: RubricDict


class VideoQARubricVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    task_id: Text
    setting: Setting
    verifier_metadata: VideoQARubricMetadata


class VideoQARubricVerifyResponse(VideoQARubricVerifyRequest, BaseVerifyResponse):
    """Reward = the rubric score (0-1); 0, or ``empty_answer_reward``, for an attempt without an answer; a failed
    single-turn model call leaves the attempt unmeasured (masked)."""

    outcome: Outcome
    # The rubric score from 0 to 1; None when the attempt was not measured.
    rubric_score: float | None
    rubric_points: float | None = None
    judgment: dict[str, Any] | None = None
    judge_usage: dict[str, Any] | None = None
    judge_attempts: int | None = None
    evaluation_completed: bool


def final_answer(response: NeMoGymResponse) -> str:
    """The final answer: the text of the last assistant message after any reasoning block, never a reasoning item or a
    tool result."""
    for item in reversed(response.output):
        if item.type == "message" and getattr(item, "role", None) == "assistant":
            return final_answer_text("".join(part.text for part in item.content if part.type == "output_text"))
    return ""


def _is_retryable(failure: NgFailureEvent) -> bool:
    """Whether another attempt can succeed: no reply (a timeout or lost connection), a 408/429, or a 5xx."""
    status = failure.http_status
    return status is None or status in RETRYABLE_JUDGE_STATUSES or status >= 500


class VideoQARubricResourcesServer(VideoSandboxServer):
    """Rubric grading of final answers. With ``sandbox_provider`` set, each agentic episode's seed lends a sandbox that
    holds the row's video, as for the video MCQA servers; otherwise the server keeps no sessions."""

    config: VideoQARubricResourcesServerConfig
    checkpoint_verify: ClassVar[Literal["replay"]] = "replay"
    _judge_semaphore: asyncio.Semaphore = PrivateAttr()
    _judge_converter: ChatCompletionsConverter = PrivateAttr(
        default_factory=lambda: ChatCompletionsConverter(return_token_id_information=False, uses_reasoning_parser=True)
    )

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._judge_semaphore = asyncio.Semaphore(self.config.judge_endpoint_max_concurrency)

    async def verify(self, body: VideoQARubricVerifyRequest) -> VideoQARubricVerifyResponse:
        """Grade the final answer; an attempt without one scores 0 (or ``empty_answer_reward``) without a judge call.

        A judge that fails every attempt raises: Gym's judge failsafe masks the row and keeps it, answer included, for
        judge-only recovery. It is never scored as a wrong answer.
        """
        response = body.response
        # video_mcqa's rules, read from the response's own status so a replayed row scores the same: a failed
        # single-turn model call is not measured; a single-turn tool call and a failed or incomplete agent run score 0.
        failure, failure_reason = response_failure(response, setting=body.setting)
        if failure is not None:
            return self._respond(
                body,
                outcome=failure,
                rubric_score=None if failure == "infrastructure_failure" else 0.0,
                failure_kind=f"video_qa_rubric:{failure}",
                failure_reason=failure_reason,
            )
        answer = final_answer(response)
        if not answer:
            penalty = self.config.empty_answer_reward
            reward = 0.0 if penalty is None or not response.output else penalty
            return self._respond(body, outcome="empty_answer", rubric_score=0.0, reward=reward)

        async def grade() -> VideoQARubricVerifyResponse:
            judgment, points, usage, attempts = await self._judge(body.verifier_metadata, answer)
            return self._respond(
                body,
                outcome="graded",
                rubric_score=points / 100,
                rubric_points=float(points),
                judgment=judgment,
                judge_usage=usage,
                judge_attempts=attempts,
            )

        return await observe_llm_judge_grading(grade)

    def _respond(
        self,
        body: VideoQARubricVerifyRequest,
        *,
        rubric_score: float | None,
        reward: float | None = None,
        **fields: Any,
    ) -> VideoQARubricVerifyResponse:
        """The verify response; the reward is the rubric score unless given, and 0 for an unmeasured attempt."""
        if reward is None:
            reward = 0.0 if rubric_score is None else rubric_score
        return VideoQARubricVerifyResponse(
            **body.model_dump(exclude=VERIFICATION_FIELDS),
            reward=reward,
            mask_sample=rubric_score is None,
            evaluation_completed=rubric_score is not None,
            rubric_score=rubric_score,
            **fields,
        )

    async def _judge(
        self, metadata: VideoQARubricMetadata, answer: str
    ) -> tuple[dict[str, Any], float, dict[str, Any] | None, int]:
        """(judgment, points from 0 to 100, the judge's usage, attempts) of the first judgment the rubric accepts."""
        params = self.config.judge_chat_completion_create_params.model_copy(deep=True)
        params.messages = judge_messages(metadata.question, metadata.rubric, answer)
        params.response_format = {
            "type": "json_schema",
            "json_schema": {"name": JUDGMENT_SCHEMA_NAME, "strict": True, "schema": JUDGMENT_SCHEMA},
        }
        attempts = self.config.judge_max_attempts
        for attempt in range(1, attempts + 1):
            try:
                # Queue waiting does not consume the attempt's deadline.
                async with self._judge_semaphore:
                    async with asyncio.timeout(self.config.judge_attempt_timeout_seconds):
                        # This loop owns the retry budget, so the shared client makes one attempt per iteration.
                        reply = await call_judge(
                            self.server_client,
                            server_name=self.config.judge_model_server.name,
                            url_path="/v1/chat/completions",
                            json=params,
                            response_model=NeMoGymChatCompletion,
                            max_attempts=1,
                        )
                try:
                    judgment = json.loads(self._reply_text(reply))
                    points = judgment_points(metadata.rubric, judgment)
                except ValueError:
                    record_rejected_judge_reply()
                    raise
            # JUDGE_FAILURES covers the transport failure call_judge raises as a judge_failed event.
            except (*JUDGE_FAILURES, TimeoutError, ValueError) as error:
                if isinstance(error, NgFailureEvent) and not is_judge_failure(error):
                    raise  # another Gym failure, such as a judge provider stop: not this loop's to retry
                if isinstance(error, NgFailureEvent) and not _is_retryable(error):
                    raise  # an auth or bad-request reply fails the same way again
                logger.warning(
                    "Rubric judge attempt %d/%d failed: %s: %s", attempt, attempts, type(error).__name__, error
                )
                if attempt == attempts:
                    raise JudgeError(
                        f"The rubric judge failed {attempts} attempts: {type(error).__name__}: {error}"
                    ) from error
                await asyncio.sleep(self._retry_delay(attempt))
                continue
            record_parsed_judge_verdict()
            usage = reply.usage.model_dump(mode="json") if reply.usage is not None else None
            return judgment, points, usage, attempt
        raise AssertionError("judge_max_attempts must be positive")

    def _reply_text(self, reply: NeMoGymChatCompletion) -> str:
        """The JSON text of the judge's single completed choice, reasoning removed."""
        if len(reply.choices) != 1 or reply.choices[0].finish_reason != "stop":
            raise ValueError("The judge did not return one completed choice")
        output = self._judge_converter.postprocess_chat_response(reply.choices[0], structured_output=True)
        return "".join(
            part.text
            for item in output
            if item.type == "message"
            for part in item.content
            if part.type == "output_text"
        )

    def _retry_delay(self, attempt: int) -> float:
        """Full-jitter exponential backoff, so concurrent verifies do not retry a rate-limited judge in lockstep."""
        config = self.config
        return random.uniform(
            0, min(config.judge_retry_max_delay_seconds, config.judge_retry_base_delay_seconds * 2 ** (attempt - 1))
        )

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """The shares of measured attempts with full and with at least 0.8 of the points, and the outcomes; Gym adds
        the mean, spread and range of ``rubric_score`` and ``reward`` (masked attempts are left out of all of them)."""
        rows = [row for task in tasks for row in task if "outcome" in row]
        scores = [row["rubric_score"] for row in rows]
        if any(not isinstance(score, (int, float)) or not math.isfinite(score) for score in scores):
            raise ValueError("Measured attempts carry a finite rubric_score")
        outcomes = Counter(row["outcome"] for row in rows)
        metrics: dict[str, Any] = {f"outcomes/{outcome}": outcomes[outcome] for outcome in OUTCOMES}
        if scores:
            metrics["full_score_rate"] = sum(score >= 1 for score in scores) / len(scores)
            metrics["at_least_0.8_rate"] = sum(score >= 0.8 for score in scores) / len(scores)
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        keys = ("mean/rubric_score", "mean/reward", "full_score_rate")
        return {key: agent_metrics[key] for key in keys if key in agent_metrics}


if __name__ == "__main__":
    VideoQARubricResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VideoQARubricResourcesServer.run_webserver()  # noqa: F401
