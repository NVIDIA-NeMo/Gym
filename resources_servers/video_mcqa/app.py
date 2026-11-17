# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixed-cohort verifier for VideoMME-v1 and LVBench accuracy.

Run directly (configs/video_mcqa_single_turn.yaml), it scores the benchmark its cohort names. In the agentic setting
it can also lend each episode a sandbox that holds the video (sandbox.py).
"""

from collections import Counter
from pathlib import Path
from typing import ClassVar, Literal, Self
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from nemo_gym.config import ROLLOUT_INDEX_KEY_NAME, AggregateMetrics, AggregateMetricsRequest
from nemo_gym.core import NeMoGymResponse
from nemo_gym.server_runtime import BaseVerifyRequest, BaseVerifyResponse, ReverifyMode
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.video_mcqa.answer_judge import AnswerJudgeConfig, grade_final_answer
from resources_servers.video_mcqa.dataset import SOURCES, Benchmark, validate_subsets
from resources_servers.video_mcqa.sandbox import VideoSandboxServer, VideoSandboxServerConfig
from resources_servers.video_mcqa.scoring import (
    AnswerPolicy,
    Failure,
    Outcome,
    QuestionScore,
    aggregate_scores,
    grade_answer,
)
from resources_servers.video_mcqa.task import Setting, VerifierMetadata


class AccuracyCohort(BaseModel):
    """Freeze benchmark identity, denominator and category membership before inference."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    benchmark: Benchmark
    dataset_source: str
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    setting: Setting
    answer_policy: AnswerPolicy = "boxed_only_v1"
    task_ids: tuple[str, ...] = Field(min_length=1)
    task_subsets: dict[str, dict[str, tuple[str, ...]]]
    num_repeats: int = Field(default=1, ge=1, strict=True)

    @model_validator(mode="after")
    def valid_cohort(self) -> Self:
        if self.dataset_source != SOURCES[self.benchmark]:
            raise ValueError("Dataset source does not match this benchmark")
        if len(set(self.task_ids)) != len(self.task_ids) or any(
            not identity.startswith(f"{self.benchmark}:") or not identity.split(":", 1)[1].strip()
            for identity in self.task_ids
        ):
            raise ValueError("Expected unique, benchmark-scoped task IDs")
        if set(self.task_subsets) != set(self.task_ids):
            raise ValueError("Every frozen task must have complete subset membership")
        for subsets in self.task_subsets.values():
            validate_subsets(self.benchmark, subsets)
        return self


class VideoMCQAResourcesServerConfig(VideoSandboxServerConfig, AnswerJudgeConfig):
    """The task cohort is a required input, not inferred from returned results."""

    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    cohort_manifest: Path


class VideoMCQAVerifyRequest(BaseVerifyRequest):
    """Keep verifier-only answers outside the model-visible Responses payload."""

    model_config = ConfigDict(extra="allow")
    task_id: str
    setting: Setting
    verifier_metadata: VerifierMetadata

    @model_validator(mode="after")
    def matching_identity(self) -> Self:
        if self.task_id != self.verifier_metadata.task_id:
            raise ValueError("Top-level and verifier task IDs differ")
        return self


class VideoMCQAVerifyResponse(VideoMCQAVerifyRequest, BaseVerifyResponse):
    """Reward plus explicit format, execution and infrastructure diagnoses."""

    answer_policy: AnswerPolicy
    outcome: Outcome
    predicted_answer: str | None
    evaluation_completed: bool


def final_assistant_text(response: NeMoGymResponse) -> str:
    """Never parse a reasoning summary, an earlier answer or a tool response."""
    for item in reversed(response.output):
        if item.type == "message" and item.role == "assistant":
            return "".join(part.text for part in item.content if part.type == "output_text")
    return ""


def response_failure(response: NeMoGymResponse, *, setting: Setting) -> tuple[Failure | None, str | None]:
    """Why a response is not graded on its answer, from its own status, so a replayed row scores the same.

    Single-turn: a failed model call is masked (no answer was measured), and a tool call is a wrong answer. Agentic:
    the Environment Server sends verify only the failed runs that its outcome policy scores (a model 400, 413 or 422,
    an out-of-memory kill, or a run without an outcome), and an incomplete run ended on a time, output or context
    limit; both score 0 even when an earlier text holds an answer.
    """
    if setting == "agentic":
        if response.status == "failed" or response.error is not None:
            return "execution_failure", "The agent run failed; its final answer is not scored."
        if response.status == "incomplete":
            limit = getattr(response.incomplete_details, "reason", None) or "a limit"
            return "execution_failure", f"The agent run stopped at {limit}; its final answer is not scored."
        return None, None
    if response.status == "failed" or response.error is not None:
        return "infrastructure_failure", "The model endpoint failed; no answer was measured."
    if any(item.type == "function_call" for item in response.output):
        return "execution_failure", "Tool calls are not allowed in the single-turn setting."
    return None, None


class VideoMCQAResourcesServer(VideoSandboxServer):
    """Common deterministic verifier; concrete servers fix the benchmark, this one takes the cohort's."""

    config: VideoMCQAResourcesServerConfig
    benchmark: ClassVar[Benchmark | None] = None
    checkpoint_verify: ClassVar[Literal["replay"]] = "replay"
    _cohort: AccuracyCohort = PrivateAttr()

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._cohort = AccuracyCohort.model_validate_json(self.config.cohort_manifest.read_text(encoding="utf-8"))
        if self.benchmark is not None and self._cohort.benchmark != self.benchmark:
            raise ValueError("Cohort belongs to a different benchmark resources server")
        self.config.check_answer_judge(self._cohort.answer_policy)

    def _validate_metadata(self, metadata: VerifierMetadata, *, setting: str) -> None:
        cohort = self._cohort
        if (
            metadata.task_id not in cohort.task_ids
            or metadata.dataset_revision != cohort.dataset_revision
            or metadata.dataset_source != cohort.dataset_source
            or setting != cohort.setting
        ):
            raise ValueError("Result benchmark, task, revision, source or setting differs from the frozen cohort")
        if metadata.option_count != 4 or metadata.level is not None or metadata.group_id is not None:
            raise ValueError("VideoMME-v1/LVBench require four choices and no v2 group or level")

    async def verify(self, body: VideoMCQAVerifyRequest) -> VideoMCQAVerifyResponse:
        """Score final answers only; masked infrastructure failures remain inconclusive."""
        self._validate_metadata(body.verifier_metadata, setting=body.setting)
        failure, reason = response_failure(body.response, setting=body.setting)
        return await grade_final_answer(
            self.server_client,
            self.config,
            body.responses_create_params,
            final_assistant_text(body.response),
            body.verifier_metadata,
            setting=body.setting,
            answer_policy=self._cohort.answer_policy,
            failure=failure,
            respond=lambda score: self._verify_response(body, score, failure, reason),
        )

    def _verify_response(
        self, body: VideoMCQAVerifyRequest, score: QuestionScore, failure: Failure | None, reason: str | None
    ) -> VideoMCQAVerifyResponse:
        payload = body.model_dump(
            exclude={
                "reward",
                "mask_sample",
                "failure_kind",
                "failure_reason",
                "answer_policy",
                "outcome",
                "predicted_answer",
                "evaluation_completed",
            }
        )
        return VideoMCQAVerifyResponse(
            **payload,
            reward=score.reward if score.reward is not None else 0.0,
            mask_sample=not score.evaluation_completed,
            failure_kind=f"{self._cohort.benchmark}:{failure}" if failure else None,
            failure_reason=reason,
            answer_policy=score.answer_policy,
            outcome=score.outcome,
            predicted_answer=score.predicted_answer,
            evaluation_completed=score.evaluation_completed,
        )

    def _attempts(self, body: AggregateMetricsRequest) -> list[list[QuestionScore]]:
        cohort = self._cohort
        repeats: list[list[QuestionScore]] = [[] for _ in range(cohort.num_repeats)]
        for result in body.verify_responses:
            repeat = result.get(ROLLOUT_INDEX_KEY_NAME, 0)
            if type(repeat) is not int or not 0 <= repeat < cohort.num_repeats:
                raise ValueError("Unexpected repeat index for the frozen cohort")
            if "outcome" not in result:
                # A row Gym adds to count a task as zero (count_failure_classes_as_zero,
                # count_missing_rollouts_as_zero) has no verifier fields; the frozen cohort counts it as missing.
                continue
            metadata = VerifierMetadata(**result["verifier_metadata"])
            self._validate_metadata(metadata, setting=result.get("setting"))
            if result.get("task_id") != metadata.task_id or result.get("answer_policy") != cohort.answer_policy:
                raise ValueError("Result task or answer policy differs from the frozen cohort")
            score = (
                grade_answer(
                    None,
                    metadata,
                    setting=cohort.setting,
                    answer_policy=cohort.answer_policy,
                    failure="infrastructure_failure",
                )
                if result.get("mask_sample", False)
                else QuestionScore(
                    task_id=metadata.task_id,
                    dataset_revision=metadata.dataset_revision,
                    setting=result["setting"],
                    answer_policy=result["answer_policy"],
                    outcome=result["outcome"],
                    predicted_answer=result["predicted_answer"],
                    reward=result["reward"],
                    evaluation_completed=result["evaluation_completed"],
                )
            )
            repeats[repeat].append(score)
        return repeats

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """Micro accuracy, with frozen denominators overall and in each category.

        Missing/infra-failed attempts do not shrink the denominator. LVBench
        multi-label questions count once overall and once in each assigned
        capability, exactly as its official scorer does on complete predictions.
        No VideoMME-v2 nonlinear metric is produced by either of these servers.
        """
        cohort = self._cohort
        repeats = self._attempts(body)
        metric = "Video_MME" if cohort.benchmark == "videomme_v1" else "LVBench"
        if cohort.setting == "agentic":
            metric += "_agentic"
        subsets: dict[str, tuple[str, ...]] = {metric: cohort.task_ids}
        for task_id, memberships in cohort.task_subsets.items():
            for axis, labels in memberships.items():
                for label in labels:
                    prefix = f"{metric}/{axis}/{quote(label, safe='')}"
                    subsets[prefix] = (*subsets.get(prefix, ()), task_id)
        measured: dict[str, float | int | bool] = {}
        for prefix, task_ids in subsets.items():
            # Validate all attempts against the full cohort before filtering;
            # duplicate or unexpected tasks must never disappear in a subset.
            summaries = [
                aggregate_scores(
                    scores if prefix == metric else [score for score in scores if score.task_id in task_ids],
                    expected_task_ids=task_ids,
                    dataset_revision=cohort.dataset_revision,
                    setting=cohort.setting,
                    answer_policy=cohort.answer_policy,
                )
                for scores in repeats
            ]
            expected = sum(summary.expected for summary in summaries)
            completed = sum(summary.completed for summary in summaries)
            correct = sum(summary.correct for summary in summaries)
            counts: Counter[str] = Counter()
            for summary in summaries:
                counts.update(summary.outcome_counts)
            measured.update(
                {
                    f"{prefix}/raw_acc": 100 * correct / expected,
                    f"{prefix}/expected": expected,
                    f"{prefix}/correct": correct,
                    f"{prefix}/returned": sum(summary.returned for summary in summaries),
                    f"{prefix}/missing": sum(summary.missing for summary in summaries),
                    f"{prefix}/coverage": completed / expected,
                    f"{prefix}/is_complete": completed == expected,
                    **{f"{prefix}/outcomes/{key}": value for key, value in counts.items()},
                }
            )
        measured[f"mean/{metric}"] = measured[f"{metric}/raw_acc"] / 100
        metrics = await super().aggregate_metrics(body)
        metrics.agent_metrics.update(measured)
        metrics.key_metrics.update(
            {
                key: value
                for key, value in measured.items()
                if key.startswith("mean/") or key.endswith(("/raw_acc", "/coverage", "/is_complete"))
            }
        )
        return metrics


if __name__ == "__main__":
    VideoMCQAResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VideoMCQAResourcesServer.run_webserver()
