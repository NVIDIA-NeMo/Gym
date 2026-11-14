# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateless VideoMME-v2 verification with fixed-cohort coverage accounting."""

from collections import Counter
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.videomme_v2.metrics import (
    MetricSlice,
    QuestionGroup,
    aggregate_benchmark_scores,
    validate_metric_cohort,
)
from resources_servers.videomme_v2.scoring import (
    EXECUTION_FAILURE_METADATA_KEY,
    AnswerPolicy,
    Outcome,
    QuestionScore,
    grade_answer,
)
from resources_servers.videomme_v2.task import Setting, VerifierMetadata


class CohortManifest(BaseModel):
    """Public run identity, never a source of model instructions or gold answers."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1, 2] = 1
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    setting: Setting = "single_turn"
    answer_policy: AnswerPolicy = "boxed_only_v1"
    task_ids: tuple[str, ...] = Field(min_length=1)
    num_repeats: int = Field(default=1, ge=1, strict=True)
    groups: tuple[QuestionGroup, ...] = ()
    task_levels: dict[str, Annotated[int, Field(strict=True, ge=1, le=3)]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        if any(not identity.strip() for identity in self.task_ids) or len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("The frozen cohort must contain nonempty, unique task IDs")
        validate_metric_cohort(self.task_ids, groups=self.groups, task_levels=self.task_levels)
        if self.schema_version == 2 and not self.groups:
            raise ValueError("Cohort schema version 2 requires complete question groups")
        return self


class VideoMMEV2ResourcesServerConfig(BaseResourcesServerConfig):
    """A run must declare its cohort before any results are collected."""

    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    cohort_manifest: Path


class VideoMMEV2VerifyRequest(BaseVerifyRequest):
    """Gold remains beside, not inside, responses_create_params."""

    model_config = ConfigDict(extra="allow")
    task_id: str
    setting: Setting
    verifier_metadata: VerifierMetadata

    @model_validator(mode="after")
    def matching_identity(self) -> Self:
        if self.task_id != self.verifier_metadata.task_id:
            raise ValueError("Top-level and verifier task IDs differ")
        return self


class VideoMMEV2VerifyResponse(VideoMMEV2VerifyRequest, BaseVerifyResponse):
    """Gym's numeric reward plus an explicit answer and completion diagnosis."""

    answer_policy: AnswerPolicy
    outcome: Outcome
    predicted_answer: str | None
    evaluation_completed: bool


def final_assistant_text(response: NeMoGymResponse) -> str:
    """Read only the last assistant message; never grade a reasoning summary."""
    for item in reversed(response.output):
        if item.type == "message" and item.role == "assistant":
            return "".join(part.text for part in item.content if part.type == "output_text")
    return ""


class VideoMMEV2ResourcesServer(SimpleResourcesServer):
    """No tools, process launches, LLM judge, mutable sessions or answer repair."""

    config: VideoMMEV2ResourcesServerConfig
    checkpoint_verify: ClassVar[Literal["replay"]] = "replay"
    _cohort: CohortManifest = PrivateAttr()
    _task_levels: dict[str, int] = PrivateAttr(default_factory=dict)
    _task_groups: dict[str, QuestionGroup] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._cohort = CohortManifest.model_validate_json(self.config.cohort_manifest.read_text(encoding="utf-8"))
        self._task_levels = validate_metric_cohort(
            self._cohort.task_ids, groups=self._cohort.groups, task_levels=self._cohort.task_levels
        )
        self._task_groups = {task_id: group for group in self._cohort.groups for task_id in group.task_ids}

    def _validate_metadata(self, metadata: VerifierMetadata) -> None:
        if metadata.task_id not in self._cohort.task_ids or metadata.dataset_revision != self._cohort.dataset_revision:
            raise ValueError("Result does not belong to the frozen dataset cohort")
        level = self._task_levels.get(metadata.task_id)
        if level is not None and metadata.level is not None and metadata.level != level:
            raise ValueError("Result level differs from its frozen cohort")
        group = self._task_groups.get(metadata.task_id)
        if group is not None and metadata.group_id is not None and metadata.group_id != group.group_id:
            raise ValueError("Result group differs from its frozen cohort")

    async def verify(self, body: VideoMMEV2VerifyRequest) -> VideoMMEV2VerifyResponse:
        """Score a completed answer; failed model responses remain inconclusive."""
        cohort = self._cohort
        self._validate_metadata(body.verifier_metadata)
        if body.setting != cohort.setting:
            raise ValueError("Result setting differs from the frozen cohort")
        failure = None
        failure_reason = None
        if body.response.status == "failed" or body.response.error is not None:
            failure = "infrastructure_failure"
            failure_reason = "The model endpoint reported a failed response; no answer was measured."
        elif body.setting == "single_turn" and any(item.type == "function_call" for item in body.response.output):
            failure = "execution_failure"
            failure_reason = "A tool call is not a valid single-turn answer."
        elif (
            body.setting == "agentic" and (body.response.metadata or {}).get(EXECUTION_FAILURE_METADATA_KEY) == "true"
        ):
            failure = "execution_failure"
            failure_reason = "The agent run failed; its final answer is not scored."
        score = grade_answer(
            final_assistant_text(body.response),
            body.verifier_metadata,
            setting=body.setting,
            answer_policy=cohort.answer_policy,
            failure=failure,
        )
        # Gym requires a numeric reward. Its mask flag carries the inconclusive
        # status; the fixed-cohort aggregation does not silently shrink the denominator.
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
        return VideoMMEV2VerifyResponse(
            **payload,
            reward=score.reward if score.reward is not None else 0.0,
            mask_sample=not score.evaluation_completed,
            failure_kind=f"videomme_v2:{failure}" if failure else None,
            failure_reason=failure_reason,
            answer_policy=score.answer_policy,
            outcome=score.outcome,
            predicted_answer=score.predicted_answer,
            evaluation_completed=score.evaluation_completed,
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """Publish raw accuracy and official nonlinear score on a 0–100 scale.

        Missing attempts and masked infrastructure errors make the result
        provisional. They remain in the frozen denominator for every repeat.
        Four-question groups are scored within each repeat, never across repeats.
        Gym's standard mean/ metric is the 0–1 raw accuracy.
        """
        cohort = self._cohort
        repeats: list[list[QuestionScore]] = [[] for _ in range(cohort.num_repeats)]
        for result in body.verify_responses:
            repeat = result.get(ROLLOUT_INDEX_KEY_NAME, 0)
            if type(repeat) is not int or not 0 <= repeat < cohort.num_repeats:
                raise ValueError("Unexpected repeat index for the frozen cohort")
            metadata = VerifierMetadata(**result["verifier_metadata"])
            self._validate_metadata(metadata)
            if result.get("task_id") != metadata.task_id or result.get("setting") != cohort.setting:
                raise ValueError("Result task identity or setting differs from its cohort")
            if "answer_policy" in result and result["answer_policy"] != cohort.answer_policy:
                raise ValueError("Result answer policy differs from its cohort")
            if result.get("mask_sample", False):
                score = grade_answer(
                    None,
                    metadata,
                    setting=cohort.setting,
                    answer_policy=cohort.answer_policy,
                    failure="infrastructure_failure",
                )
            else:
                score = QuestionScore(
                    task_id=metadata.task_id,
                    dataset_revision=metadata.dataset_revision,
                    setting=result["setting"],
                    answer_policy=result["answer_policy"],
                    outcome=result["outcome"],
                    predicted_answer=result["predicted_answer"],
                    reward=result["reward"],
                    evaluation_completed=result["evaluation_completed"],
                )
            repeats[repeat].append(score)
        benchmarks = [
            aggregate_benchmark_scores(
                scores,
                expected_task_ids=cohort.task_ids,
                dataset_revision=cohort.dataset_revision,
                setting=cohort.setting,
                answer_policy=cohort.answer_policy,
                groups=cohort.groups,
                task_levels=cohort.task_levels,
            )
            for scores in repeats
        ]
        summaries = [benchmark.accuracy for benchmark in benchmarks]
        # Retain generic Gym metrics and runtime observability. Only this named
        # benchmark metric uses the declared full cohort rather than measured rows.
        metrics = await super().aggregate_metrics(body)
        expected = sum(summary.expected for summary in summaries)
        completed = sum(summary.completed for summary in summaries)
        correct = sum(summary.correct for summary in summaries)
        counts: Counter[str] = Counter()
        for summary in summaries:
            counts.update(summary.outcome_counts)
        # Single-turn reports the EF metric Video_MME_v2; agentic has its own name.
        metric = "Video_MME_v2" if cohort.setting == "single_turn" else "Video_MME_v2_agentic"
        benchmark_metrics = {
            f"mean/{metric}": correct / expected,
            f"{metric}/coverage": completed / expected,
            f"{metric}/is_complete": completed == expected,
            f"{metric}/expected": expected,
            f"{metric}/returned": sum(summary.returned for summary in summaries),
            f"{metric}/missing": sum(summary.missing for summary in summaries),
            **{f"{metric}/outcomes/{key}": value for key, value in counts.items()},
        }
        for level in (None, 1, 2, 3):
            prefix = metric if level is None else f"{metric}/L{level}"
            slices = [benchmark.overall if level is None else benchmark.levels[level] for benchmark in benchmarks]
            benchmark_metrics.update(
                {f"{prefix}/{key}": value for key, value in MetricSlice.combine(slices).metrics().items()}
            )
        metrics.agent_metrics.update(benchmark_metrics)
        metrics.key_metrics.update(
            {key: benchmark_metrics[key] for key in (f"mean/{metric}", f"{metric}/coverage", f"{metric}/is_complete")}
        )
        metrics.key_metrics.update(
            {
                key: value
                for key, value in benchmark_metrics.items()
                if key.endswith(("/raw_acc", "/nonlinear_score", "/nonlinear_available", "/group_coverage"))
            }
        )
        return metrics


if __name__ == "__main__":
    VideoMMEV2ResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VideoMMEV2ResourcesServer.run_webserver()
