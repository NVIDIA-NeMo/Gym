# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CG-Bench (full / mini / hard) long_acc fixed-cohort resources server built on VideoMCQAResourcesServer."""

from collections import Counter
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock
from urllib.parse import quote

from pydantic import PrivateAttr

from nemo_gym.client import ServerClient
from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import VerifierFixture
from resources_servers.cgbench.dataset import SLICES, CGBenchCohort
from resources_servers.cgbench.scoring import (
    MAX_OPTIONS,
    CGBenchAnswerPolicy,
    CGBenchQuestionScore,
    grade_cgbench_answer,
)
from resources_servers.video_mcqa.app import (
    VideoMCQAResourcesServer,
    VideoMCQAResourcesServerConfig,
    VideoMCQAVerifyRequest,
    VideoMCQAVerifyResponse,
)
from resources_servers.video_mcqa.sandbox import VideoSandboxServer
from resources_servers.video_mcqa.scoring import Failure, QuestionScore, aggregate_scores
from resources_servers.video_mcqa.task import Setting, VerifierMetadata


class CGBenchVerifyResponse(VideoMCQAVerifyResponse):
    """Verify response supporting both CG-Bench's rule policy and shared video_mcqa policies."""

    answer_policy: CGBenchAnswerPolicy


class CGBenchResourcesServer(VideoMCQAResourcesServer):
    """Fixed-cohort long_acc verifier for CG-Bench (12,129 Q), mini (3,000 Q), and hard (2,417 Q)."""

    verify_response_cls: ClassVar[type[VideoMCQAVerifyResponse]] = CGBenchVerifyResponse
    score_cls: ClassVar[type[QuestionScore]] = CGBenchQuestionScore
    _cohort: CGBenchCohort = PrivateAttr()

    def _load_cohort(self, manifest: Path) -> CGBenchCohort:
        return CGBenchCohort.model_validate_json(manifest.read_text(encoding="utf-8"))

    def _validate_cohort(self) -> None:
        configured = self.server_client.global_config_dict.get("num_repeats")
        if configured is not None and configured != self._cohort.num_repeats:
            raise ValueError(
                f"Configured num_repeats={configured} disagrees with frozen cohort "
                f"num_repeats={self._cohort.num_repeats}; re-prepare with ++num_repeats={configured}"
            )

    def _validate_option_count(self, metadata: VerifierMetadata) -> None:
        if not 2 <= metadata.option_count <= MAX_OPTIONS:
            raise ValueError(f"CG-Bench requires between 2 and {MAX_OPTIONS} choices")

    def _cohort_scope(self) -> str:
        return self._cohort.benchmark

    def _metric_spec(self) -> tuple[str, dict[str, dict[str, tuple[str, ...]]]]:
        metric = SLICES[self._cohort.benchmark].metric_name
        return metric, self._cohort.task_subsets

    def _grade_rule(
        self,
        final_text: str | None,
        metadata: VerifierMetadata,
        *,
        setting: Setting,
        answer_policy: str,
        failure: Failure | None = None,
    ) -> QuestionScore:
        return grade_cgbench_answer(
            final_text, metadata, setting=setting, answer_policy=answer_policy, failure=failure
        )

    async def verify(self, body: VideoMCQAVerifyRequest) -> CGBenchVerifyResponse:
        """Score final assistant text with CG-Bench's rule policy or configured answer judge."""
        result = await super().verify(body)
        assert isinstance(result, CGBenchVerifyResponse)
        return result

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """Aggregate fixed-cohort long_acc overall and across category, sub_category, domain, and duration.

        Uses set membership and list accumulation so the full 12,129-question CG-Bench cohort aggregates in
        linear time across all 36 official breakdown slices while preserving VideoMCQAResourcesServer's exact
        metric schema and validation semantics.
        """
        cohort = self._cohort
        repeats = self._attempts(body)
        metric, task_subsets = self._metric_spec()
        if cohort.setting == "agentic":
            metric += "_agentic"
        subsets: dict[str, list[str]] = {metric: list(cohort.task_ids)}
        for task_id, memberships in task_subsets.items():
            for axis, labels in memberships.items():
                for label in labels:
                    prefix = f"{metric}/{axis}/{quote(label, safe='')}"
                    subsets.setdefault(prefix, []).append(task_id)
        measured: dict[str, float | int | bool] = {}
        for prefix, task_ids in subsets.items():
            id_set = set(task_ids)
            summaries = [
                aggregate_scores(
                    scores if prefix == metric else [score for score in scores if score.task_id in id_set],
                    expected_task_ids=task_ids,
                    dataset_revision=cohort.dataset_revision,
                    setting=cohort.setting,
                    answer_policy=cohort.answer_policy,
                    allowed_policies=self.score_cls.ALLOWED_POLICIES,
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
        metrics = await VideoSandboxServer.aggregate_metrics(self, body)
        metrics.agent_metrics.update(measured)
        metrics.key_metrics.update(
            {
                key: value
                for key, value in measured.items()
                if key.startswith("mean/") or key.endswith(("/raw_acc", "/coverage", "/is_complete"))
            }
        )
        return metrics


def _fixture_server() -> CGBenchResourcesServer:
    """Build a synthetic offline CG-Bench server for ``gym env test`` verifier contract checks."""
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    return CGBenchResourcesServer(
        config=VideoMCQAResourcesServerConfig(
            name="cgbench",
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            cohort_manifest=Path(__file__).parent / "tests" / "verifier_cohort.json",
        ),
        server_client=client,
    )


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=_fixture_server,
    request_model=VideoMCQAVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    CGBenchResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = CGBenchResourcesServer.run_webserver()
