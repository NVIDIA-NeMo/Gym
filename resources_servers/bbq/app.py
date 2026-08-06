# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NeMo Gym resource server for the BBQ two-judge reward contract."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from util import (
    AnswerJudgment,
    EmptyPolicyResponseError,
    ExpectedValues,
    ExplanationJudgment,
    JudgeCallError,
    JudgeOutputError,
    MetadataValidationError,
    answer_reward,
    explanation_reward,
    extract_last_assistant_text,
    extract_policy_prompt,
    json_for_prompt,
    parse_answer_judgment,
    parse_explanation_judgment,
    render_template,
    validate_answer_judgment,
)

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import (
    compute_pass_majority_metrics,
    compute_subset_metrics,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


logger = logging.getLogger(__name__)


class BBQTwoJudgeConfig(BaseResourcesServerConfig):
    """Runtime configuration for the shared judge endpoint and prompt templates."""

    name: str = "bbq_two_judge"
    judge_model_server: ModelServerRef
    judge_chat_create_params: NeMoGymChatCompletionCreateParamsNonStreaming
    judge_system_message: str | None = None
    judge_prompt_templates_path: str = "configs/verifier_prompt_templates.yaml"
    judge_timeout_seconds: float = Field(default=120.0, gt=0.0, le=600.0)
    # Bounded per-check retry for transient judge failures (transport errors,
    # timeouts, malformed or inconsistent judge JSON). Fail-closed is
    # preserved: after the attempt budget the typed error propagates and the
    # row gets no reward. 1 = single attempt (no retry).
    judge_max_attempts: int = Field(default=3, ge=1, le=8)
    judge_retry_backoff_seconds: float = Field(default=2.0, ge=0.0, le=60.0)
    # Reward for a policy response with no assistant text. An empty response is
    # a policy-behaviour outcome (it answers nothing), not an operational
    # failure, so it is scored rather than raised; a single degenerate
    # generation must not abort a training run, and keeping the row preserves
    # the GRPO group size the advantage estimate depends on. Set to null to
    # restore the strict raise (EmptyPolicyResponseError) for evaluation runs
    # where an empty response should be surfaced instead of scored.
    empty_response_reward: float | None = Field(default=0.0, ge=0.0, le=1.0)


class BBQVerifyRequest(BaseVerifyRequest):
    """Gym verify request with explicit hidden row metadata."""

    model_config = ConfigDict(extra="allow")
    expected_values: dict[str, Any] = Field(default_factory=dict)


class AnswerJudgeEvaluation(BaseModel):
    check: Literal["answer_check"] = "answer_check"
    model_server: str
    prompt_version: str
    attempt_count: int = Field(default=1, ge=1)
    request: NeMoGymChatCompletionCreateParamsNonStreaming
    response: NeMoGymChatCompletion
    raw_output_sha256: str
    judgment: AnswerJudgment


class ExplanationJudgeEvaluation(BaseModel):
    check: Literal["explanation_quality_check"] = "explanation_quality_check"
    model_server: str
    prompt_version: str
    attempt_count: int = Field(default=1, ge=1)
    request: NeMoGymChatCompletionCreateParamsNonStreaming
    response: NeMoGymChatCompletion
    raw_output_sha256: str
    judgment: ExplanationJudgment


class BBQVerifyResponse(BaseVerifyResponse):
    """Successful response. Operational failures raise and contain no reward.

    An empty policy response is a policy-behaviour outcome, not an operational
    failure: when `empty_response_reward` is set it is scored (default 0.0)
    with `empty_policy_response=True` and no judge fields, since there is
    nothing for the judges to read.
    """

    expected_values: dict[str, Any]
    model_response_text: str
    answer_judgment: AnswerJudgment | None = None
    explanation_judgment: ExplanationJudgment | None = None
    answer_judge_evaluation: AnswerJudgeEvaluation | None = None
    explanation_judge_evaluation: ExplanationJudgeEvaluation | None = None
    reward_answer: float
    reward_explanation_quality: float
    empty_policy_response: bool = False


class BBQTwoJudgeResourcesServer(SimpleResourcesServer):
    """Single-turn comparative-QA verifier with two concurrent LLM checks."""

    config: BBQTwoJudgeConfig
    _prompt_templates: dict[str, dict[str, str]] = PrivateAttr(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        self._load_prompt_templates()
        return super().setup_webserver()

    def _load_prompt_templates(self) -> None:
        path = Path(self.config.judge_prompt_templates_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise MetadataValidationError(f"could not load judge prompt templates from {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise MetadataValidationError("judge prompt template file must contain a mapping")
        required = {"answer_check", "explanation_quality_check"}
        if set(payload) != required:
            raise MetadataValidationError(f"judge prompt template file must contain exactly {sorted(required)}")
        for name in sorted(required):
            item = payload[name]
            if not isinstance(item, dict) or set(item) != {"prompt_version", "prompt_template"}:
                raise MetadataValidationError(f"{name} must contain exactly prompt_version and prompt_template")
            if not all(isinstance(item[key], str) and item[key].strip() for key in item):
                raise MetadataValidationError(f"{name} prompt fields must be nonempty strings")
        self._prompt_templates = payload

    def _template(self, name: str) -> dict[str, str]:
        if not self._prompt_templates:
            self._load_prompt_templates()
        return self._prompt_templates[name]

    async def verify(self, body: BBQVerifyRequest) -> BBQVerifyResponse:
        try:
            expected = ExpectedValues.model_validate(body.expected_values)
        except Exception as exc:
            raise MetadataValidationError(f"invalid expected_values: {exc}") from exc

        policy_prompt = extract_policy_prompt(body)
        if not policy_prompt:
            raise MetadataValidationError("responses_create_params contains no policy-facing prompt")
        policy_response = extract_last_assistant_text(body)
        if not policy_response:
            if self.config.empty_response_reward is None:
                raise EmptyPolicyResponseError("policy response contains no assistant text")
            reward = self.config.empty_response_reward
            logger.warning(
                "BBQ verify: empty policy response scored %.3f provenance_id=%s",
                reward,
                expected.provenance_id,
            )
            return BBQVerifyResponse(
                **body.model_dump(),
                reward=reward,
                model_response_text="",
                reward_answer=reward,
                reward_explanation_quality=reward,
                empty_policy_response=True,
            )

        answer_template = self._template("answer_check")
        explanation_template = self._template("explanation_quality_check")
        answer_prompt = render_template(
            answer_template["prompt_template"],
            {
                "context_condition": expected.context_condition,
                "label": expected.label,
                "actor_aliases_json": json_for_prompt(expected.actor_aliases),
                "policy_prompt_json": json_for_prompt(policy_prompt),
                "policy_response_json": json_for_prompt(policy_response),
            },
        )
        explanation_prompt = render_template(
            explanation_template["prompt_template"],
            {
                "category": expected.category,
                "protected_axis": expected.protected_axis,
                "actor_aliases_json": json_for_prompt(expected.actor_aliases),
                "protected_roles_json": json_for_prompt(expected.protected_roles),
                "protected_descriptors_json": json_for_prompt(expected.protected_descriptors),
                "gold_evidence_json": json_for_prompt(expected.gold_evidence),
                "ambiguity_reason_json": json_for_prompt(expected.ambiguity_reason),
                "decisive_detail_json": json_for_prompt(expected.decisive_detail),
                "policy_prompt_json": json_for_prompt(policy_prompt),
                "policy_response_json": json_for_prompt(policy_response),
            },
        )

        answer_task = asyncio.create_task(
            self._evaluate_answer(
                answer_prompt,
                answer_template["prompt_version"],
                expected,
            )
        )
        explanation_task = asyncio.create_task(
            self._evaluate_explanation(
                explanation_prompt,
                explanation_template["prompt_version"],
            )
        )
        tasks = (answer_task, explanation_task)
        try:
            answer_evaluation, explanation_evaluation = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        reward_answer = answer_reward(answer_evaluation.judgment, expected)
        reward_explanation = explanation_reward(explanation_evaluation.judgment)
        reward = reward_answer * reward_explanation

        logger.info(
            "BBQ verify provenance_id=%s category=%s condition=%s answer_class=%s "
            "quality_class=%s rewards=(%.3f, %.3f, %.3f)",
            expected.provenance_id,
            expected.category,
            expected.context_condition,
            answer_evaluation.judgment.answer_class,
            explanation_evaluation.judgment.quality_class,
            reward_answer,
            reward_explanation,
            reward,
        )

        return BBQVerifyResponse(
            **body.model_dump(),
            reward=reward,
            model_response_text=policy_response,
            answer_judgment=answer_evaluation.judgment,
            explanation_judgment=explanation_evaluation.judgment,
            answer_judge_evaluation=answer_evaluation,
            explanation_judge_evaluation=explanation_evaluation,
            reward_answer=reward_answer,
            reward_explanation_quality=reward_explanation,
        )

    @staticmethod
    def _score_fn(result: dict) -> dict[str, float]:
        """Named scores per rollout.

        - correct: reward == 1.0, the keep-band definition (team decision
          2026-08-03): correct answer AND high-quality explanation.
        - correct_lenient: reward > 0, the documented fallback if the
          keep-band yield under `correct` is too low (correct answer AND
          explanation not unacceptable; a 0.5 low-quality explanation still
          counts).
        - answer_accuracy: the Answer Check alone.
        """
        return {
            "correct": float(result.get("reward", 0.0) == 1.0),
            "correct_lenient": float(result.get("reward", 0.0) > 0.0),
            "answer_accuracy": float(result.get("reward_answer", 0.0) == 1.0),
        }

    @staticmethod
    def _flatten_subset_fields(tasks: list[list[dict]]) -> list[list[dict]]:
        """Copy category/context_condition out of expected_values so subset
        grouping (which reads top-level rollout fields) can see them."""
        flattened = []
        for task_rollouts in tasks:
            enriched_rollouts = []
            for rollout in task_rollouts:
                expected = rollout.get("expected_values") or {}
                enriched = dict(rollout)
                enriched.setdefault("category", expected.get("category"))
                enriched.setdefault("context_condition", expected.get("context_condition"))
                enriched_rollouts.append(enriched)
            flattened.append(enriched_rollouts)
        return flattened

    def compute_metrics(self, tasks: list[list[dict]]) -> dict:
        """BBQ metrics: overall pass@k plus per-category and ambig/disambig subsets."""
        flattened = self._flatten_subset_fields(tasks)
        metrics, *_ = compute_pass_majority_metrics(flattened, score_fn=self._score_fn)
        metrics.update(compute_subset_metrics(flattened, "category", self._score_fn))
        metrics.update(compute_subset_metrics(flattened, "context_condition", self._score_fn))
        return metrics

    async def _attempt_with_retry(self, check: str, attempt_once):
        """Run one judge check with a bounded retry budget.

        Retries only the typed, transient failure classes (transport errors,
        timeouts, malformed or inconsistent judge output). After the budget the
        last typed error propagates unchanged; a row never receives a
        fabricated reward.
        """
        last_error: Exception | None = None
        for attempt in range(1, self.config.judge_max_attempts + 1):
            try:
                return attempt, await attempt_once()
            except (JudgeCallError, JudgeOutputError) as exc:
                last_error = exc
                if attempt == self.config.judge_max_attempts:
                    break
                logger.warning(
                    "%s attempt %d/%d failed (%s); retrying",
                    check,
                    attempt,
                    self.config.judge_max_attempts,
                    exc,
                )
                if self.config.judge_retry_backoff_seconds:
                    await asyncio.sleep(self.config.judge_retry_backoff_seconds * attempt)
        assert last_error is not None
        raise last_error

    async def _evaluate_answer(
        self,
        prompt: str,
        prompt_version: str,
        expected: ExpectedValues,
    ) -> AnswerJudgeEvaluation:
        answer_schema = self._answer_schema_for(expected)

        async def attempt_once():
            request, response, raw_output = await self._call_judge(
                check="answer_check",
                prompt=prompt,
                prompt_version=prompt_version,
                output_model=AnswerJudgment,
                output_schema=answer_schema,
            )
            judgment = parse_answer_judgment(raw_output)
            validate_answer_judgment(judgment, expected)
            return request, response, raw_output, judgment

        attempt_count, (request, response, raw_output, judgment) = await self._attempt_with_retry(
            "answer_check", attempt_once
        )
        return AnswerJudgeEvaluation(
            model_server=self.config.judge_model_server.name,
            prompt_version=prompt_version,
            attempt_count=attempt_count,
            request=request,
            response=response,
            raw_output_sha256=hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
            judgment=judgment,
        )

    async def _evaluate_explanation(
        self,
        prompt: str,
        prompt_version: str,
    ) -> ExplanationJudgeEvaluation:
        async def attempt_once():
            request, response, raw_output = await self._call_judge(
                check="explanation_quality_check",
                prompt=prompt,
                prompt_version=prompt_version,
                output_model=ExplanationJudgment,
            )
            judgment = parse_explanation_judgment(raw_output)
            explanation_reward(judgment)
            return request, response, raw_output, judgment

        attempt_count, (request, response, raw_output, judgment) = await self._attempt_with_retry(
            "explanation_quality_check", attempt_once
        )
        return ExplanationJudgeEvaluation(
            model_server=self.config.judge_model_server.name,
            prompt_version=prompt_version,
            attempt_count=attempt_count,
            request=request,
            response=response,
            raw_output_sha256=hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
            judgment=judgment,
        )

    @staticmethod
    def _answer_schema_for(expected: ExpectedValues) -> dict:
        """AnswerJudgment schema with selected_answer constrained to THIS row's
        active actors (+ uncertain). Constrained decoding then makes an
        inactive-actor verdict impossible to emit; per the answer template, an
        unmappable commitment must instead be selected_answer=null with
        answer_class=incorrect_answer. (Observed live: the judge occasionally
        emitted e.g. actor_c on a two-actor row, deterministically at
        temperature 0, which fail-closed every retry.)"""
        schema = AnswerJudgment.model_json_schema()
        allowed = ["uncertain", *sorted(expected.actor_aliases)]
        for branch in schema["properties"]["selected_answer"].get("anyOf", []):
            if branch.get("type") == "string":
                branch["enum"] = allowed
        return schema

    async def _call_judge(
        self,
        *,
        check: Literal["answer_check", "explanation_quality_check"],
        prompt: str,
        prompt_version: str,
        output_model: type[BaseModel],
        output_schema: dict | None = None,
    ) -> tuple[
        NeMoGymChatCompletionCreateParamsNonStreaming,
        NeMoGymChatCompletion,
        str,
    ]:
        params = self.config.judge_chat_create_params.model_copy(deep=True)
        messages: list[dict[str, str]] = []
        if self.config.judge_system_message:
            messages.append({"role": "system", "content": self.config.judge_system_message})
        messages.append({"role": "user", "content": prompt})
        params.messages = messages  # type: ignore[assignment]
        params.response_format = {  # type: ignore[assignment]
            "type": "json_schema",
            "json_schema": {
                "name": f"bbq_{check}_{prompt_version}",
                "strict": True,
                "schema": output_schema if output_schema is not None else output_model.model_json_schema(),
            },
        }

        response = await self._request_judge(params, check)
        if not response.choices:
            raise JudgeOutputError(f"{check} returned no choices")
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise JudgeOutputError(f"{check} returned no textual JSON content")
        return params, response, content

    async def _request_judge(
        self,
        params: NeMoGymChatCompletionCreateParamsNonStreaming,
        check: str,
    ) -> NeMoGymChatCompletion:
        """Make one verifier-level call. No retry or semantic fallback is applied here."""

        try:
            async with asyncio.timeout(self.config.judge_timeout_seconds):
                http_response = await self.server_client.post(
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/chat/completions",
                    json=params,
                )
                await raise_for_status(http_response)
                payload = await get_response_json(http_response)
                return NeMoGymChatCompletion.model_validate(payload)
        except JudgeCallError:
            raise
        except Exception as exc:
            raise JudgeCallError(f"{check} judge request failed: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    BBQTwoJudgeResourcesServer.run_webserver()
