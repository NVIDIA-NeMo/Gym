# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import logging
from typing import Any, Literal

from aiohttp import ClientTimeout
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import Domain, ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json
from resources_servers.arena.arena import (
    _extract_thinking_content,
    _extract_verdict,
    _strip_thinking_blocks,
    _weighted_scores_as_a,
    _weighted_scores_as_b,
)
from resources_servers.arena.metrics import ArenaMetrics
from resources_servers.arena.taxonomy import get_prompt_slices


logger = logging.getLogger(__name__)


class ArenaResourcesServerConfig(BaseResourcesServerConfig):
    model_config = ConfigDict(extra="forbid")

    num_workers: int | None  # Number of resources-server worker processes; null uses framework behavior.
    domain: Domain  # Benchmark domain used by NeMo Gym metadata.
    verified: bool  # Whether the benchmark has completed NeMo Gym verification.
    description: str  # Human-readable benchmark description.
    value: str  # What model capability or metric the benchmark measures.
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    verdict_weight: int = Field(ge=1)  # Strong-verdict upweighting is used by lmarena_v2 only.
    # Fraction of rollouts allowed to fail (gen_answer or gen_judgment API errors) before
    # raising an error.
    max_rollout_failure_rate: float = Field(ge=0, le=1)
    # Max number of concurrent HTTP calls to the judge model. Lower this if the judge
    # endpoint returns 429 (rate limit) errors.
    judge_concurrency: int = Field(ge=1)
    # When true, store generations without calling the judge or reporting score metrics.
    generation_only: bool
    # Explicit aliases for detecting when the policy also supplied the baseline answer.
    policy_model_aliases: list[str]
    # lmarena_v3 treats both-bad as a tie; lmarena_v2 excludes it.
    score_both_bad_as_tie: bool
    # lmarena_v2 uses Bradley-Terry; lmarena_v3 uses reference-length style control.
    style_control_method: Literal["bradley_terry", "reference_length"]
    # Accepted policy/reference token ratio for reference-length style control.
    style_length_ratio_range: tuple[float, float] | None
    # Optional lmarena_v3 short-reference rule; both limits must be set to enable it.
    style_short_reference_max_tokens: int | None
    style_short_response_max_tokens: int | None
    # Optional Bradley-Terry parameters used only by lmarena_v2, keyed by category.
    # Values are lists of 4 floats (converted to np.ndarray at startup).
    style_norm_mean: dict[str, list[float]] | None
    style_norm_std: dict[str, list[float]] | None
    style_coefs: dict[str, list[float]] | None
    # Judge prompt template and system message.
    judge_prompt_template: str
    judge_system_message: str
    # Optional category-specific overrides; otherwise judge_system_message is used.
    judge_system_message_by_category: dict[str, str]
    # Per-request timeout for judge API calls in seconds.
    judge_timeout_secs: float = Field(gt=0)
    tokenizer_model: str
    bootstrap_rounds: int = Field(ge=1)
    bootstrap_seed: int


class ArenaRunRequest(BaseRunRequest):
    """Fields added to every JSONL row (beyond responses_create_params)."""

    # Prompt metadata is consumed by offline reporting, not by the evaluation server.
    model_config = ConfigDict(extra="ignore")

    question_id: str
    question: str  # raw user message content, passed verbatim to the judge
    baseline_answer: str  # the baseline model's answer for pairwise comparison
    baseline_model: str | None = None
    category: str
    # Source metadata is reduced to compact slice labels before saving the rollout.
    metadata: dict[str, Any] | None = Field(default=None, exclude=True)
    prompt_slices: dict[str, list[str]] = Field(default_factory=dict)
    # Whether this lmarena_v3 row is derived from an lmarena_v2 prompt.
    is_lmarena_v2_prompt: bool = False
    # reference token count for style control
    style_reference_token_count: int | None = None
    # Set to True when the baseline answer was provided by the same model being evaluated.
    # When True, verify() returns immediately (no judge call) and compute_metrics() excludes
    # this rollout from both scoring and the task-failure-rate denominator.
    self_comparison: bool = False


class ArenaVerifyRequest(ArenaRunRequest, BaseVerifyRequest):
    pass


class ArenaGame(BaseModel):
    """Result of one judge game (one ordering of policy vs baseline)."""

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    # None if the judge output couldn't be parsed.
    verdict: str | None


class ArenaVerifyResponse(BaseVerifyResponse):
    question_id: str
    question: str
    baseline_answer: str
    baseline_model: str | None = None
    category: str
    style_reference_token_count: int | None = None
    policy_answer: str | None = None
    # Retained in the rollout output; only policy_answer is sent to the judge.
    policy_reasoning: str | None = None
    games: list[ArenaGame] | None = None
    self_comparison: bool = False
    prompt_slices: dict[str, list[str]] = Field(default_factory=dict)
    is_lmarena_v2_prompt: bool = False


class ArenaResourcesServer(SimpleResourcesServer):
    """Pairwise LLM-judge resources server for arena-style chat benchmarks.

    Evaluates the policy model's response against a fixed baseline answer using
    an LLM judge. Two games are played (policy=A/baseline=B, then baseline=A/policy=B)
    to cancel out positional bias. The reward is the average of both game scores.
    """

    config: ArenaResourcesServerConfig
    _judge_semaphore: asyncio.Semaphore = PrivateAttr()
    _metrics: ArenaMetrics = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._judge_semaphore = asyncio.Semaphore(self.config.judge_concurrency)
        # Online and offline evaluation share this scoring implementation.
        self._metrics = ArenaMetrics(self.config)

    async def verify(self, body: ArenaVerifyRequest) -> ArenaVerifyResponse:
        prompt_slices = body.prompt_slices or {
            namespace: sorted(labels)
            for namespace, labels in get_prompt_slices(
                {
                    "metadata": body.metadata,
                    "responses_create_params": body.responses_create_params.model_dump(),
                }
            ).items()
        }
        body.prompt_slices = prompt_slices

        # Automatic model-name matching is enabled only for lmarena_v3.
        if self.config.style_control_method == "reference_length":
            body.self_comparison = body.self_comparison or self._is_self_comparison(body)

        # Skip judging when the evaluated model also supplied the baseline answer.
        if body.self_comparison and not self.config.generation_only:
            return ArenaVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                policy_answer=None,
                games=None,
            )

        policy_answer, policy_reasoning = self._extract_response_parts(body.response)

        # Generation-only runs save the response without invoking the judge.
        if self.config.generation_only:
            return ArenaVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                policy_answer=policy_answer,
                policy_reasoning=policy_reasoning,
                games=None,
            )

        # lmarena_v3 scores truncated responses as zero without spending judge tokens.
        if (
            self.config.style_control_method == "reference_length"
            and body.response.incomplete_details
            and body.response.incomplete_details.reason == "max_output_tokens"
        ):
            return ArenaVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                policy_answer=policy_answer,
                policy_reasoning=policy_reasoning,
                games=None,
            )

        # An empty response cannot be judged and counts as a failed rollout.
        if not policy_answer:
            return ArenaVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                policy_answer=None,
                policy_reasoning=policy_reasoning,
                games=None,
            )

        # Resolve judge prompt template and system message (may vary by category).
        prompt_template = self.config.judge_prompt_template
        system_message = (
            self.config.judge_system_message_by_category.get(body.category or "") or self.config.judge_system_message
        )

        # Run both game orderings concurrently to reduce latency.
        game1, game2 = await asyncio.gather(
            self._run_judge_game(body.question, policy_answer, body.baseline_answer, system_message, prompt_template),
            self._run_judge_game(body.question, body.baseline_answer, policy_answer, system_message, prompt_template),
        )

        # Game 1: policy is A. Game 2: policy is B.
        # Strong verdicts (>>) are counted `verdict_weight` times.
        weight = self.config.verdict_weight
        scores = _weighted_scores_as_a(game1.verdict, weight) + _weighted_scores_as_b(game2.verdict, weight)
        reward = sum(scores) / len(scores)

        return ArenaVerifyResponse(
            **body.model_dump(),
            reward=reward,
            policy_answer=policy_answer,
            policy_reasoning=policy_reasoning,
            games=[game1, game2],
        )

    def _is_self_comparison(self, body: ArenaVerifyRequest) -> bool:
        if not body.baseline_model:
            return False
        baseline = body.baseline_model.strip().casefold().rstrip("/")
        policy_models = [body.response.model, *self.config.policy_model_aliases]
        return baseline in {model.strip().casefold().rstrip("/") for model in policy_models if model}

    @staticmethod
    def _extract_response_parts(response: NeMoGymResponse) -> tuple[str | None, str | None]:
        """Return (policy_answer, policy_reasoning).

        policy_answer:   concatenated output_text content with thinking blocks stripped.
                         This is what is sent to the judge.
        policy_reasoning: thinking block content from output_text, plus summary text from
                          any type='reasoning' output items. Never sent to the judge.
        """
        text_parts: list[str] = []
        reasoning_parts: list[str] = []

        for output_item in response.output:
            if output_item.type == "reasoning":
                # OpenAI o-series style: reasoning content in summary items.
                for s in output_item.summary:
                    if s.text.strip():
                        reasoning_parts.append(s.text.strip())
            elif output_item.type == "message":
                for content_item in output_item.content:
                    if content_item.type == "output_text":
                        text_parts.append(content_item.text)

        if text_parts:
            joined = "".join(text_parts)
            think_content = _extract_thinking_content(joined)
            if think_content:
                reasoning_parts.append(think_content)
            answer = _strip_thinking_blocks(joined) or None
        else:
            answer = None

        reasoning = "\n\n".join(reasoning_parts) or None
        return answer, reasoning

    async def _run_judge_game(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        system_message: str,
        prompt_template: str,
    ) -> ArenaGame:
        """Run one judge game comparing answer_a against answer_b for the given question.

        Acquired under `_judge_semaphore` to cap concurrent calls to the judge endpoint
        and avoid 429 rate-limit errors.
        """
        async with self._judge_semaphore:
            config = self.config
            responses_create_params = config.judge_responses_create_params.model_copy(deep=True)

            judge_prompt = prompt_template.format(
                question=question,
                answer_a=answer_a,
                answer_b=answer_b,
            )
            responses_create_params.input = [
                NeMoGymEasyInputMessage(role="system", content=system_message),
                NeMoGymEasyInputMessage(role="user", content=judge_prompt),
            ]

            try:
                response = await self.server_client.post(
                    server_name=config.judge_model_server.name,
                    url_path="/v1/responses",
                    json=responses_create_params,
                    timeout=ClientTimeout(total=config.judge_timeout_secs),
                )
                judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
            except Exception as exc:
                logger.warning("Judge call failed (question skipped): %s", exc)
                return ArenaGame(
                    responses_create_params=responses_create_params,
                    response=NeMoGymResponse(
                        id="error",
                        created_at=0.0,
                        model="",
                        object="response",
                        output=[],
                        parallel_tool_calls=False,
                        tool_choice="none",
                        tools=[],
                    ),
                    verdict=None,
                )

            verdict: str | None = None
            if judge_response.output:
                last_output = judge_response.output[-1]
                if last_output.type == "message" and last_output.content:
                    last_content = last_output.content[-1]
                    if last_content.type == "output_text":
                        verdict = _extract_verdict(last_content.text)

            return ArenaGame(
                responses_create_params=responses_create_params,
                response=judge_response,
                verdict=verdict,
            )

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """NeMo Gym hook for aggregate rollout metrics."""
        return self._metrics.compute(tasks)

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        """NeMo Gym hook for the compact metrics summary."""
        return self._metrics.get_key_metrics(agent_metrics)


if __name__ == "__main__":
    ArenaResourcesServer.run_webserver()
