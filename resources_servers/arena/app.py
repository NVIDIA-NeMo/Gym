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
from typing import Any

import numpy as np
from aiohttp import ClientTimeout
from pydantic import BaseModel, PrivateAttr, field_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json
from resources_servers.arena.arena import (
    _VERDICT_LABEL_BOTH_BAD,
    _bootstrap_per_category,
    _compute_raw_style_feature,
    _extract_thinking_content,
    _extract_verdict,
    _strip_thinking_blocks,
    _weighted_scores_as_a,
    _weighted_scores_as_b,
)


logger = logging.getLogger(__name__)


# ── Pydantic models ───────────────────────────────────────────────────────────


class ArenaResourcesServerConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    verdict_weight: int = 3  # strong verdicts (>>) count this many times in the reward average
    # Fraction of rollouts allowed to fail (gen_answer or gen_judgment API errors) before
    # raising an error.
    max_rollout_failure_rate: float = 0.01
    # Max number of concurrent HTTP calls to the judge model. Lower this if the judge
    # endpoint returns 429 (rate limit) errors.
    judge_concurrency: int = 16
    # Whether to apply style control when computing win_rate. When True (default), win_rate
    # is a Bradley-Terry win probability after removing length/formatting bias via pre-computed
    # style offsets. When False, win_rate is a bootstrap mean of weighted scores.
    style_control: bool = True
    # Style normalization constants and BT coefficients, keyed by category.
    # Single-category datasets use the key "default".
    # Values are lists of 4 floats (converted to np.ndarray at startup).
    style_norm_mean: dict[str, list[float]] = {}
    style_norm_std: dict[str, list[float]] = {}
    style_coefs: dict[str, list[float]] = {}
    # Judge prompt template and system message.
    judge_prompt_template: str
    judge_system_message: str
    judge_system_message_by_category: dict[str, str] = {}
    # Per-request timeout for judge API calls in seconds. Default 1800 s (30 min) covers
    # extended-thinking judges (e.g. claude-opus-4-6 with budget_tokens). Lower for
    # non-thinking judges where a 5 min hang already signals a problem.
    judge_timeout_secs: float = 1800.0


class ArenaRunRequest(BaseRunRequest):
    """Fields added to every JSONL row (beyond responses_create_params)."""

    question_id: str
    question: str  # raw user message content, passed verbatim to the judge
    baseline_answer: str  # the baseline model's answer for pairwise comparison
    category: str | None = None
    # Set to True when the baseline answer was provided by the same model being evaluated.
    # When True, verify() returns immediately (no judge call) and compute_metrics() excludes
    # this rollout from both scoring and the task-failure-rate denominator.
    self_comparison: bool = False

    @field_validator("baseline_answer", mode="before")
    @classmethod
    def _coerce_baseline_answer(cls, v: object) -> str:
        if isinstance(v, dict):
            return v["answer"]
        return v


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
    category: str | None = None
    policy_answer: str | None = None
    # Reasoning/thinking content extracted from the policy model's response (for debugging only).
    # Populated from <think>/<thinking> blocks and type='reasoning' output items.
    # Never sent to the judge — only the stripped policy_answer is used for judging.
    policy_reasoning: str | None = None
    games: list[ArenaGame] | None = None
    self_comparison: bool = False


# ── Server ────────────────────────────────────────────────────────────────────


class ArenaResourcesServer(SimpleResourcesServer):
    """Pairwise LLM-judge resources server for arena-style chat benchmarks.

    Evaluates the policy model's response against a fixed baseline answer using
    an LLM judge. Two games are played (policy=A/baseline=B, then baseline=A/policy=B)
    to cancel out positional bias. The reward is the average of both game scores.
    """

    config: ArenaResourcesServerConfig
    _judge_semaphore: asyncio.Semaphore = PrivateAttr()
    _style_norm_mean: dict[str, np.ndarray] = PrivateAttr()
    _style_norm_std: dict[str, np.ndarray] = PrivateAttr()
    _style_coefs: dict[str, np.ndarray] = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._judge_semaphore = asyncio.Semaphore(self.config.judge_concurrency)
        self._style_norm_mean = {k: np.array(v) for k, v in self.config.style_norm_mean.items()}
        self._style_norm_std = {k: np.array(v) for k, v in self.config.style_norm_std.items()}
        self._style_coefs = {k: np.array(v) for k, v in self.config.style_coefs.items()}

    def _get_style_constants(self, category: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return (norm_mean, norm_std, coefs) for *category*, or None if not configured.

        Single-category datasets use the key "default" (set by precompute_style_constants.py).
        If the exact category is not found, falls back to "default" — this handles datasets
        where question.jsonl sets a dataset-level category name rather than a meaningful
        category (e.g. lmarena-260311 uses category="lmarena-260311" for all questions).
        Returns None if neither the category nor "default" is present.
        """
        cat = category or "default"
        if cat not in self._style_norm_mean:
            if "default" in self._style_norm_mean:
                cat = "default"
            else:
                if self._style_norm_mean:
                    logger.warning("Unknown style category %r; skipping style control for this rollout.", cat)
                return None
        return self._style_norm_mean[cat], self._style_norm_std[cat], self._style_coefs[cat]

    async def verify(self, body: ArenaVerifyRequest) -> ArenaVerifyResponse:
        if body.self_comparison:
            return ArenaVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                policy_answer=None,
                games=None,
                self_comparison=True,
            )

        policy_answer, policy_reasoning = self._extract_response_parts(body.response)

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

        # Game 1: policy is A. Game 2: policy is B — flip perspective.
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

    @staticmethod
    def _extract_response_parts(response: NeMoGymResponse) -> tuple[str | None, str | None]:
        """Return (policy_answer, policy_reasoning).

        policy_answer:   concatenated output_text content with thinking blocks stripped.
                         This is what is sent to the judge.
        policy_reasoning: thinking block content from output_text, plus summary text from
                          any type='reasoning' output items. For debugging only — never sent
                          to the judge.
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

    # ── Metrics ───────────────────────────────────────────────────────────────

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Compute bootstrapped win rate and judge parse failure rate.

        Two reward metrics are reported (see README for full explanation):

        ``mean/reward``
            Framework metric: per-task mean reward averaged equally over all tasks.
            Best for RL training signal.

        ``win_rate`` (± bootstrap 95% CI)
            When ``style_control=True`` (default): Bradley-Terry win probability after
            removing length/formatting bias via pre-computed style offsets.
            When ``style_control=False``: bootstrap mean of weighted scores.
            [[BB]] and parse failures are excluded. Strong verdicts (>>) are weighted
            ``verdict_weight``-times. Bootstrap 2.5th/97.5th percentile CI.

        Raises:
            ValueError: if the fraction of failed rollouts exceeds ``max_rollout_failure_rate``.
                No score is returned — computed metrics would be unreliable.
        """
        total_games = 0
        total_rollouts = failed_rollouts = self_comparison_rollouts = BB_rollout_count = valid_rollout_count = 0
        weight = self.config.verdict_weight

        # Accumulated per-category per-battle data for aggregate metrics.
        # Keyed by category (or "default" for single-category datasets).
        # win_rate is the unweighted mean of per-category BT win rates, so that datasets with
        # unequal category sizes (e.g. arena-hard-v2.0: 500 hard_prompt + 250 creative_writing)
        # give equal weight to each category.
        battle_scores_by_cat: dict[str, list[float]] = {}
        battle_offsets_by_cat: dict[str, list[float]] = {}

        for rollouts in tasks:
            for r in rollouts:
                # Self-comparisons (baseline and policy from the same model) are excluded
                # from scoring AND from the failure-rate denominator.
                if r.get("self_comparison"):
                    self_comparison_rollouts += 1
                    continue
                total_rollouts += 1
                raw_games = r.get("games")
                games = raw_games or []

                # A rollout is "failed" if gen_answer produced no output (games is None/empty)
                # or if gen_judgment produced an unparseable verdict for any game.
                if raw_games is None or any((g or {}).get("verdict") is None for g in games):
                    failed_rollouts += 1

                # Count games only for the rollout failure rate denominator.
                total_games += len(games)

                # Accumulate per-battle scores and style features for aggregate metrics.
                # Skip rollouts with [[BB]] or parse failures.
                if len(games) == 2:
                    v1 = (games[0] or {}).get("verdict")
                    v2 = (games[1] or {}).get("verdict")
                    if v1 is not None and v2 is not None:
                        valid_rollout_count += 1
                        if v1 == _VERDICT_LABEL_BOTH_BAD or v2 == _VERDICT_LABEL_BOTH_BAD:
                            BB_rollout_count += 1
                    if v1 and v1 != _VERDICT_LABEL_BOTH_BAD and v2 and v2 != _VERDICT_LABEL_BOTH_BAD:
                        w_scores = _weighted_scores_as_a(v1, weight) + _weighted_scores_as_b(v2, weight)
                        cat_key = r.get("category") or "default"
                        battle_scores_by_cat.setdefault(cat_key, []).extend(w_scores)

                        # Style offset: pre-compute using per-category constants.
                        policy_text = r.get("policy_answer") or ""
                        baseline_text = r.get("baseline_answer") or ""
                        constants = (
                            self._get_style_constants(r.get("category")) if (policy_text and baseline_text) else None
                        )
                        if constants is not None:
                            feat = _compute_raw_style_feature(policy_text, baseline_text)
                            norm_mean, norm_std, coefs = constants
                            offset = float((feat - norm_mean) / norm_std @ coefs)
                        else:
                            offset = 0.0
                        battle_offsets_by_cat.setdefault(cat_key, []).extend([offset] * len(w_scores))

        if self_comparison_rollouts > 0:
            logger.warning(
                "%d self-comparison rollout(s) excluded from scoring (baseline and policy from the same model).",
                self_comparison_rollouts,
            )

        if total_rollouts == 0:
            return {}

        failure_rate = failed_rollouts / total_rollouts
        if failure_rate > self.config.max_rollout_failure_rate:
            raise ValueError(
                f"Too many failed rollouts: {failed_rollouts}/{total_rollouts} "
                f"({failure_rate * 100:.1f}%) exceeds max_rollout_failure_rate="
                f"{self.config.max_rollout_failure_rate * 100:.1f}%. "
                "Check gen_answer and gen_judgment API errors."
            )

        metrics: dict[str, Any] = {
            "rollout_failure_rate": failed_rollouts / total_rollouts,
            "BB_rollout_count": BB_rollout_count,
        }

        if not battle_scores_by_cat:
            return metrics

        cat_scores_arr = {cat: np.array(s, dtype=np.float64) for cat, s in battle_scores_by_cat.items()}
        style_control = self.config.style_control
        metrics["style_control"] = style_control

        if style_control:
            # ── style-controlled win rate (per-category BT, unweighted mean across categories) ──
            cat_offsets_arr = {cat: np.array(o, dtype=np.float64) for cat, o in battle_offsets_by_cat.items()}
            pt_est, ci_lower, ci_upper = _bootstrap_per_category(
                cat_scores_arr, cat_offsets=cat_offsets_arr, n_rounds=100
            )
        else:
            # ── bootstrap mean (per-category, unweighted mean) ──
            pt_est, ci_lower, ci_upper = _bootstrap_per_category(cat_scores_arr, cat_offsets=None, n_rounds=100)

        metrics["win_rate"] = pt_est
        metrics["win_rate_ci_lower"] = ci_lower
        metrics["win_rate_ci_upper"] = ci_upper

        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        key: dict[str, Any] = {}
        for name in (
            "mean/reward",
            "win_rate",
            "win_rate_ci_lower",
            "win_rate_ci_upper",
            "style_control",
            "mean/input_tokens",
            "mean/output_tokens",
            "rollout_failure_rate",
            "BB_rollout_count",
        ):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        return key


if __name__ == "__main__":
    ArenaResourcesServer.run_webserver()
