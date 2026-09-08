# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GDP.pdf rubric verifier and benchmark metrics.

The scalar reward is criterion mean-pass so the same environment provides a
useful dense reward for RL. Evaluation additionally reports Artificial
Analysis' headline all-pass metric. Each rubric criterion is judged in an
independent call that sees only the task, candidate answer, and that criterion.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, ClassVar, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import highest_k_metrics


_DEFAULT_JUDGE_PROMPT = Path(__file__).parent / "prompts" / "judge.yaml"
_VERDICT_RE = re.compile(r"(?:^|\b)(PASS|FAIL)(?:\b|$)", re.IGNORECASE)


def extract_response_text(response: NeMoGymResponse) -> str:
    """Return the concatenated assistant output text without reasoning items."""
    parts: list[str] = []
    for item in response.output:
        if getattr(item, "type", None) != "message" or getattr(item, "role", None) != "assistant":
            continue
        content = getattr(item, "content", None)
        if isinstance(content, str):
            parts.append(content)
            continue
        for block in content or []:
            if getattr(block, "type", None) == "output_text":
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(parts).strip()


def parse_judge_verdict(text: str) -> tuple[bool, bool]:
    """Return ``(passed, parsed)``; malformed judge output is a failed criterion."""
    matches = _VERDICT_RE.findall(text.strip())
    if not matches:
        return False, False
    return matches[-1].upper() == "PASS", True


class GdpPdfResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    judge_prompt_path: str = str(_DEFAULT_JUDGE_PROMPT)
    judge_max_parse_attempts: int = Field(default=5, ge=1, le=10)
    judge_endpoint_max_concurrency: Optional[int] = Field(default=64, ge=1)

    # The benchmark config uses this to retain the fixed 100-task denominator
    # when every rollout of a policy task fails before verification.
    expected_task_count: Optional[int] = Field(default=None, ge=1)
    expected_domain_task_counts: dict[str, int] = Field(default_factory=dict)


class GdpPdfVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: dict[str, Any]


class CriterionEvaluation(BaseModel):
    criterion_id: str
    criterion: str
    passed: bool
    verdict_parsed: bool
    judge_output: Optional[str] = None


class GdpPdfVerifyResponse(BaseMultiRewardVerifyResponse):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: dict[str, Any]
    extracted_answer: str
    criterion_evaluations: list[CriterionEvaluation]
    criteria_passed: int
    criteria_total: int
    mean_pass: float
    all_pass: float


class GdpPdfResourcesServer(SimpleResourcesServer):
    config: GdpPdfResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        prompt_data = yaml.safe_load(Path(self.config.judge_prompt_path).read_text(encoding="utf-8"))
        self._judge_prompt = prompt_data["user"]
        self._judge_sem = (
            asyncio.Semaphore(self.config.judge_endpoint_max_concurrency)
            if self.config.judge_endpoint_max_concurrency is not None
            else None
        )
        super().model_post_init(context)

    async def _judge_criterion(self, task_prompt: str, answer: str, criterion: dict[str, Any]) -> CriterionEvaluation:
        criterion_id = str(criterion.get("id", ""))
        criterion_text = str(criterion.get("criterion", "")).strip()
        if not criterion_text:
            raise ValueError(f"empty GDP.pdf rubric criterion {criterion_id!r}")

        params = self.config.judge_responses_create_params.model_copy(deep=True)
        params.input = [
            NeMoGymEasyInputMessage(
                role="user",
                content=self._judge_prompt.format(
                    task_prompt=task_prompt,
                    candidate_answer=answer,
                    criterion=criterion_text,
                ),
            )
        ]

        for _ in range(self.config.judge_max_parse_attempts):
            if self._judge_sem is None:
                response = await call_judge(
                    self.server_client,
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/responses",
                    json=params,
                    response_model=NeMoGymResponse,
                )
            else:
                async with self._judge_sem:
                    response = await call_judge(
                        self.server_client,
                        server_name=self.config.judge_model_server.name,
                        url_path="/v1/responses",
                        json=params,
                        response_model=NeMoGymResponse,
                    )

            judge_output = extract_response_text(response)
            passed, parsed = parse_judge_verdict(judge_output)
            if parsed:
                return CriterionEvaluation(
                    criterion_id=criterion_id,
                    criterion=criterion_text,
                    passed=passed,
                    verdict_parsed=True,
                    judge_output=judge_output,
                )

        raise JudgeError(
            f"judge returned no PASS/FAIL verdict for criterion {criterion_id} after "
            f"{self.config.judge_max_parse_attempts} attempts"
        )

    async def verify(self, body: GdpPdfVerifyRequest) -> GdpPdfVerifyResponse:
        metadata = body.verifier_metadata
        task_prompt = str(metadata.get("task_prompt", "")).strip()
        criteria = metadata.get("rubric_criteria")
        if not task_prompt:
            raise ValueError("GDP.pdf verifier_metadata.task_prompt is required")
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("GDP.pdf verifier_metadata.rubric_criteria must be a non-empty list")

        answer = extract_response_text(body.response)
        if answer:
            evaluations = list(
                await asyncio.gather(
                    *(self._judge_criterion(task_prompt, answer, criterion) for criterion in criteria)
                )
            )
        else:
            # AA v4.3 treats an empty/missing model answer as a zero attempt.
            evaluations = [
                CriterionEvaluation(
                    criterion_id=str(criterion.get("id", "")),
                    criterion=str(criterion.get("criterion", "")),
                    passed=False,
                    verdict_parsed=True,
                )
                for criterion in criteria
            ]

        criteria_passed = sum(evaluation.passed for evaluation in evaluations)
        criteria_total = len(evaluations)
        mean_pass = criteria_passed / criteria_total
        all_pass = float(criteria_passed == criteria_total)
        return GdpPdfVerifyResponse(
            **body.model_dump(),
            reward=mean_pass,
            reward_components={"mean_pass": mean_pass, "all_pass": all_pass},
            extracted_answer=answer,
            criterion_evaluations=evaluations,
            criteria_passed=criteria_passed,
            criteria_total=criteria_total,
            mean_pass=mean_pass,
            all_pass=all_pass,
        )

    @staticmethod
    def _rollout_indexed(task_rollouts: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        indexed: dict[int, dict[str, Any]] = {}
        for fallback_index, rollout in enumerate(task_rollouts):
            rollout_index = rollout.get("_ng_rollout_index", fallback_index)
            indexed[int(rollout_index)] = rollout
        return indexed

    @classmethod
    def _macro_at_k(
        cls,
        tasks: list[list[dict[str, Any]]],
        *,
        k: int,
        score_name: str,
        expected_task_count: Optional[int] = None,
    ) -> float:
        task_means = []
        for rollouts in tasks:
            indexed = cls._rollout_indexed(rollouts)
            task_means.append(sum(float(indexed.get(i, {}).get(score_name, 0.0)) for i in range(k)) / k)
        denominator = expected_task_count or len(task_means)
        if denominator == 0:
            return 0.0
        return 100.0 * sum(task_means) / denominator

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        if not tasks:
            return {}

        max_k = max(max(self._rollout_indexed(task), default=-1) + 1 for task in tasks)
        metrics: dict[str, Any] = {
            "tasks/observed": len(tasks),
            "tasks/expected": self.config.expected_task_count or len(tasks),
        }

        for k in range(1, max_k + 1):
            for score_name in ("all_pass", "mean_pass"):
                metrics[f"pass@1[avg-of-{k}]/{score_name}"] = self._macro_at_k(
                    tasks,
                    k=k,
                    score_name=score_name,
                    expected_task_count=self.config.expected_task_count,
                )

        by_domain: dict[str, list[list[dict[str, Any]]]] = {}
        for rollouts in tasks:
            if not rollouts:
                continue
            metadata = rollouts[0].get("verifier_metadata") or {}
            domain = str(metadata.get("domain", "unknown"))
            by_domain.setdefault(domain, []).append(rollouts)

        for domain, domain_tasks in sorted(by_domain.items()):
            slug = re.sub(r"[^a-z0-9]+", "_", domain.lower()).strip("_")
            expected = self.config.expected_domain_task_counts.get(domain)
            for k in range(1, max_k + 1):
                metrics[f"domain/{slug}/pass@1[avg-of-{k}]/mean_pass"] = self._macro_at_k(
                    domain_tasks,
                    k=k,
                    score_name="mean_pass",
                    expected_task_count=expected,
                )
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        key: dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["all_pass", "mean_pass"]))
        return key


if __name__ == "__main__":
    GdpPdfResourcesServer.run_webserver()
