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
"""Arena Hard v2 pairwise LLM-judge resources server.

Migrated from nemo_skills/inference/eval/arena_judge.py +
nemo_skills/evaluation/metrics/arena_metrics.py to preserve the upstream
arena-hard-auto judging protocol:

1. Candidate answer is judged against a per-task ``baseline_answer`` with
   category-specific system+user prompts (``hard_prompt`` → ``arena.yaml``,
   ``creative_writing`` → ``arena_creative.yaml``).
2. Two judge calls per rollout to control for positional bias: one with
   (A=candidate, B=baseline), one swapped (A=baseline, B=candidate).
3. Verdict regex ``\\[\\[([AB<>=]+)\\]\\]`` yields one of
   ``A>>B / A>B / A=B / B>A / B>>A`` per call.

Aggregate Arena-Elo metrics (MLE logistic regression + bootstrap CI) are
NOT computed here — they require pooling battles across all tasks and are
implemented as a post-hoc script in the migration recipe. This server
emits per-rollout binary `wins` / `strict_wins` / `ties` / `losses` scores
and a category breakdown, which is enough for Tier-1 pass@k comparison.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import ConfigDict, Field

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
from nemo_gym.prompt import PromptConfig, load_prompt_config
from nemo_gym.reward_profile import (
    compute_pass_majority_metrics,
    compute_subset_metrics,
    highest_k_metrics,
)
from nemo_gym.server_utils import get_response_json


logger = logging.getLogger(__name__)

# Regex adapted from arena-hard-auto show_result.py — matches the entire
# [[A>>B]]-style label. We take the first match and require set-uniqueness
# across all matches in the output (same as Skills' ArenaMetrics).
_VERDICT_REGEX = re.compile(r"\[\[([AB<>=]+)\]\]")

# Valid verdicts returned by the judge, in Skills' ArenaMetrics ordering
# (best candidate outcome first). Tie is "A=B".
_VALID_VERDICTS: ClassVar = ("A>>B", "A>B", "A=B", "B>A", "B>>A")


class ArenaJudgeConfig(BaseResourcesServerConfig):
    """Arena Hard v2 pairwise-judge server config."""

    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming

    # Category → prompt-file mapping. Paths are resolved relative to the
    # Gym root by ``load_prompt_config``. The defaults match Skills'
    # JUDGE_PIPELINE_ARGS behaviour for arena-hard-v2.
    judge_prompt_paths: Dict[str, str] = Field(
        default_factory=lambda: {
            "hard_prompt": "resources_servers/arena_judge/prompts/arena.yaml",
            "creative_writing": "resources_servers/arena_judge/prompts/arena_creative.yaml",
        },
        description="Map of arena-hard-v2 category → judge prompt YAML path.",
    )

    # Fallback category used when a task doesn't specify one. Matches
    # Skills' default_prompt behaviour in arena_judge.py.
    default_category: str = "hard_prompt"


class ArenaJudgeRunRequest(BaseRunRequest):
    """Run request with per-task fields flowed through from the JSONL row.

    The JSONL rows produced by ``benchmarks/arena_hard_v2/prepare.py``
    carry ``question``, ``baseline_answer``, ``category``, and ``uid`` at
    the top level; pydantic's ``extra="allow"`` lets them land here.
    """

    model_config = ConfigDict(extra="allow")

    question: Optional[str] = None
    baseline_answer: Optional[str] = None
    category: Optional[str] = None
    uid: Optional[str] = None


class ArenaJudgeVerifyRequest(ArenaJudgeRunRequest, BaseVerifyRequest):
    pass


class ArenaJudgeVerifyResponse(BaseVerifyResponse):
    """Verify response carries raw judge outputs + parsed verdicts.

    The raw text fields (``judgement_gen_base`` / ``judgement_base_gen``)
    are what the post-hoc Arena-Elo script reads to rebuild battles
    exactly as Skills does. ``verdict_*`` are the parsed labels for quick
    inspection.
    """

    model_config = ConfigDict(extra="allow")

    judgement_gen_base: Optional[str] = None
    judgement_base_gen: Optional[str] = None
    verdict_gen_base: Optional[str] = None
    verdict_base_gen: Optional[str] = None
    category: Optional[str] = None
    # True if the gen-base judge call produced no parseable verdict.
    invalid_gen_base: bool = False
    invalid_base_gen: bool = False


class ArenaJudgeServer(SimpleResourcesServer):
    """Pairwise LLM-judge server for arena-hard-v2."""

    config: ArenaJudgeConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        # Eagerly load + validate both category prompts at startup so
        # misconfiguration surfaces before any rollouts are dispatched.
        self._prompts: Dict[str, PromptConfig] = {
            category: load_prompt_config(path) for category, path in self.config.judge_prompt_paths.items()
        }
        if self.config.default_category not in self._prompts:
            raise ValueError(
                f"default_category={self.config.default_category!r} is not in "
                f"judge_prompt_paths keys {sorted(self._prompts)}."
            )

    # ------------------------------------------------------------------
    # verify()
    # ------------------------------------------------------------------

    async def verify(self, body: ArenaJudgeVerifyRequest) -> ArenaJudgeVerifyResponse:
        candidate_answer = self._extract_output_text(body.response)
        question = body.question or ""
        baseline_answer = body.baseline_answer or ""
        category = body.category or self.config.default_category
        if category not in self._prompts:
            logger.warning(
                "Unknown category %r; falling back to default %r.",
                category,
                self.config.default_category,
            )
            category = self.config.default_category

        # Two judge calls in parallel — A=candidate/B=baseline (gen-base)
        # and swapped (base-gen). Matches arena_judge.py's asyncio.gather.
        (gen_base_text, gen_base_verdict), (base_gen_text, base_gen_verdict) = await asyncio.gather(
            self._judge_once(category, question, candidate_answer, baseline_answer),
            self._judge_once(category, question, baseline_answer, candidate_answer),
        )

        # Per user decision: binary reward from the gen-base direction.
        # Candidate wins (reward=1.0) if it strictly beats the baseline in
        # the gen-base call; ties and losses both score 0.
        reward = 1.0 if gen_base_verdict in ("A>>B", "A>B") else 0.0

        return ArenaJudgeVerifyResponse(
            **body.model_dump(),
            reward=reward,
            judgement_gen_base=gen_base_text,
            judgement_base_gen=base_gen_text,
            verdict_gen_base=gen_base_verdict,
            verdict_base_gen=base_gen_verdict,
            category=category,
            invalid_gen_base=gen_base_verdict is None,
            invalid_base_gen=base_gen_verdict is None,
        )

    # ------------------------------------------------------------------
    # Judge dispatch
    # ------------------------------------------------------------------

    async def _judge_once(
        self, category: str, question: str, answer_1: str, answer_2: str
    ) -> tuple[str, Optional[str]]:
        """Run a single judge call. Returns (raw_text, parsed_verdict)."""
        prompt = self._prompts[category]
        fill = {"question": question, "answer_1": answer_1, "answer_2": answer_2}

        messages: List[NeMoGymEasyInputMessage] = []
        if prompt.system is not None:
            messages.append(NeMoGymEasyInputMessage(role="system", content=prompt.system.format_map(fill)))
        messages.append(NeMoGymEasyInputMessage(role="user", content=prompt.user.format_map(fill)))

        request_params = self.config.judge_responses_create_params.model_copy(deep=True)
        request_params.input = messages

        try:
            response_obj = await self.server_client.post(
                server_name=self.config.judge_model_server.name,
                url_path="/v1/responses",
                json=request_params,
            )
            judge_response = NeMoGymResponse.model_validate(await get_response_json(response_obj))
        except Exception:
            # Network / parse failures mirror Skills' behaviour: treat as
            # an invalid score (None verdict) rather than crashing verify.
            logger.exception("Judge call failed for category=%s; treating as invalid verdict.", category)
            return "", None

        text = self._extract_output_text(judge_response)
        verdict = self._parse_verdict(text)
        return text, verdict

    # ------------------------------------------------------------------
    # Helpers (match Skills' arena_metrics._get_judge_score semantics)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_verdict(judgment: str) -> Optional[str]:
        """Parse a verdict label from judge text.

        Matches arena_metrics.ArenaMetrics._get_judge_score: returns the
        single unique match of ``\\[\\[([AB<>=]+)\\]\\]``; if zero matches
        or multiple distinct matches are found, returns None. The verdict
        must also be one of the five valid labels — anything else (e.g.
        stray ``[[A=]]``) is treated as invalid.
        """
        if not judgment:
            return None
        matches = [m for m in _VERDICT_REGEX.findall(judgment) if m]
        unique = set(matches)
        if len(unique) != 1:
            return None
        verdict = next(iter(unique)).strip("\n")
        return verdict if verdict in _VALID_VERDICTS else None

    @staticmethod
    def _extract_output_text(response: NeMoGymResponse) -> str:
        """Concatenate all ``output_text`` content from a Response."""
        chunks: List[str] = []
        for output_item in response.output:
            if output_item.type != "message":
                continue
            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue
                chunks.append(content_item.text)
        return "".join(chunks)

    # ------------------------------------------------------------------
    # Aggregate metrics overrides
    # ------------------------------------------------------------------

    @staticmethod
    def _arena_score_fn(r: dict) -> Dict[str, float]:
        """Map a verify response dict to named float scores for pass@k.

        ``wins`` is the binary reward (strict/slight gen-base A>B). We
        additionally expose strict wins, ties, and losses so the per-task
        pass@k table shows the full verdict distribution.
        """
        gen_verdict = r.get("verdict_gen_base")
        base_verdict = r.get("verdict_base_gen")
        is_strict_win = 1.0 if gen_verdict == "A>>B" else 0.0
        is_slight_win = 1.0 if gen_verdict == "A>B" else 0.0
        is_tie = 1.0 if gen_verdict == "A=B" else 0.0
        is_loss = 1.0 if gen_verdict in ("B>A", "B>>A") else 0.0
        # Count "double wins" — candidate beats baseline in BOTH directions
        # after accounting for the swap (base-gen says candidate is B). This
        # is a conservative win signal robust to positional bias.
        double_win = 1.0 if (gen_verdict in ("A>>B", "A>B") and base_verdict in ("B>A", "B>>A")) else 0.0
        return {
            "wins": is_strict_win + is_slight_win,
            "strict_wins": is_strict_win,
            "ties": is_tie,
            "losses": is_loss,
            "double_wins": double_win,
            "invalid_gen_base": 1.0 if r.get("invalid_gen_base", False) else 0.0,
        }

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        metrics = compute_pass_majority_metrics(
            tasks,
            score_fn=self._arena_score_fn,
            # Use the gen-base verdict as the "answer" for majority voting —
            # gives a majority@k over verdict labels per task, same
            # granularity as Skills' ArenaMetrics.agg_mode.
            answer_key="verdict_gen_base",
        )[0]
        # Per-category breakdown (hard_prompt vs creative_writing).
        subset_metrics = compute_subset_metrics(
            tasks,
            field="category",
            score_fn=self._arena_score_fn,
            answer_key="verdict_gen_base",
        )
        metrics.update(subset_metrics)
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key: Dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", exclude_names=["no_answer"]))
        return key


if __name__ == "__main__":
    ArenaJudgeServer.run_webserver()
