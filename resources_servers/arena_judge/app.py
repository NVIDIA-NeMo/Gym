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

Implements the upstream [arena-hard-auto](https://github.com/lmarena/arena-hard-auto)
judging protocol:

1. Candidate answer is judged against a per-task ``baseline_answer`` with
   a category-specific system+user prompt (``hard_prompt`` →
   ``prompts/arena.yaml``, ``creative_writing`` →
   ``prompts/arena_creative.yaml``).
2. Two judge calls per rollout to control for positional bias: one with
   (A=candidate, B=baseline), one swapped (A=baseline, B=candidate).
3. Verdict regex ``\\[\\[([AB<>=]+)\\]\\]`` yields one of
   ``A>>B / A>B / A=B / B>A / B>>A`` per call.

The primary Arena-Elo headline metric (MLE logistic regression + 100-round
bootstrap 95% CI over pooled pairwise battles, per arena-hard-auto's
``show_result.py``) is computed in ``compute_metrics`` on top of the
per-task verdict collation. Per-category breakdowns are emitted for any
``category`` field observed on the input rows.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections import defaultdict
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field
from sklearn.linear_model import LogisticRegression

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
)
from nemo_gym.prompt import PromptConfig, load_prompt_config
from nemo_gym.reward_profile import (
    compute_pass_majority_metrics,
    compute_subset_metrics,
    highest_k_metrics,
)
from nemo_gym.server_utils import get_response_json


logger = logging.getLogger(__name__)

# Regex adapted from arena-hard-auto's show_result.py — matches the entire
# [[A>>B]]-style label. We take the first match and require set-uniqueness
# across all matches in the output.
_VERDICT_REGEX = re.compile(r"\[\[([AB<>=]+)\]\]")

# Valid verdicts returned by the judge, in decreasing candidate-favor
# order. Used both as a whitelist in parsing and as the preference order
# when collapsing multiple rollouts into a best-of-N verdict per task.
_VALID_VERDICTS: ClassVar = ("A>>B", "A>B", "A=B", "B>A", "B>>A")

# Weight for strict wins when assembling pairwise battles (mirrors
# arena-hard-auto's ``WEIGHT=3``: an A>>B verdict contributes 3 battles).
_STRICT_WEIGHT = 3


def sanitize_generation(generation: str) -> str:
    """Drop UTF-8 surrogate halves and null bytes that break OpenAI-compatible judges.

    Multilingual generations occasionally carry lone surrogate halves
    (model decoding artifacts) or NULs that the judge's HTTP layer rejects
    with 400. Mirrors Skills' ``nemo_skills.inference.eval.arena_judge.sanitize_generation``.
    """
    s = json.dumps(generation, ensure_ascii=False)
    s = s.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")
    s = s.replace("\x00", "")
    return json.loads(s)


class ArenaJudgeConfig(BaseResourcesServerConfig):
    """Arena Hard v2 pairwise-judge server config."""

    judge_model_server: ModelServerRef
    # The judge is called via /v1/chat/completions; this is the endpoint
    # most widely supported across OpenAI-compatible providers for a
    # simple text-verdict judge (no tool calls, no reasoning output
    # needed).
    judge_chat_completions_create_params: NeMoGymChatCompletionCreateParamsNonStreaming

    # Category → prompt-file mapping. Paths are resolved relative to the
    # Gym root by ``load_prompt_config``. Rows without a ``category``
    # field fall back to ``default_category``.
    judge_prompt_paths: Dict[str, str] = Field(
        default_factory=lambda: {
            "hard_prompt": "resources_servers/arena_judge/prompts/arena.yaml",
            "creative_writing": "resources_servers/arena_judge/prompts/arena_creative.yaml",
        },
        description="Map of arena-hard-v2 category → judge prompt YAML path.",
    )

    # Fallback category used when a task doesn't specify one.
    default_category: str = "hard_prompt"

    # Arena-Elo bootstrap: number of resamples used to compute the 95%
    # percentile CI on the Elo win-rate. 100 matches arena-hard-auto's
    # ``show_result.py``.
    arena_elo_bootstrap_rounds: int = 100
    arena_elo_bootstrap_seed: int = 42

    # Multilingual benchmarks (m-arena-hard, m-arena-hard-v2) occasionally
    # produce generations with lone UTF-8 surrogate halves or NULs that
    # break the judge's HTTP layer; enable to scrub those before judging.
    sanitize_generations: bool = False


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
    are preserved alongside the parsed labels (``verdict_*``) so the
    Arena-Elo metric can be recomputed from a rollouts.jsonl without
    re-invoking the judge.
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
        # Candidate answer comes from the POLICY model's response object.
        # The policy server is expected to produce a Responses API shape
        # (``output[].content[].text``); verify() fails soft to "" if not.
        candidate_answer = self._extract_response_output_text(body.response)
        question = body.question or ""
        baseline_answer = body.baseline_answer or ""
        if self.config.sanitize_generations:
            candidate_answer = sanitize_generation(candidate_answer)
            baseline_answer = sanitize_generation(baseline_answer)
        category = body.category or self.config.default_category
        if category not in self._prompts:
            logger.warning(
                "Unknown category %r; falling back to default %r.",
                category,
                self.config.default_category,
            )
            category = self.config.default_category

        # Two judge calls in parallel — A=candidate/B=baseline (gen-base)
        # and swapped (base-gen).
        (gen_base_text, gen_base_verdict), (base_gen_text, base_gen_verdict) = await asyncio.gather(
            self._judge_once(category, question, candidate_answer, baseline_answer),
            self._judge_once(category, question, baseline_answer, candidate_answer),
        )

        # Per-rollout binary reward from the gen-base direction.
        # Candidate wins (reward=1.0) if it strictly beats the baseline in
        # the gen-base call; ties and losses both score 0.
        reward = 1.0 if gen_base_verdict in ("A>>B", "A>B") else 0.0

        # ``body.model_dump()`` already carries ``category`` (declared
        # field). Drop it before spreading so the RESOLVED category
        # (post-fallback) isn't a duplicate kwarg to the response constructor.
        body_dict = body.model_dump()
        body_dict.pop("category", None)
        return ArenaJudgeVerifyResponse(
            **body_dict,
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
        """Run a single judge call via /v1/chat/completions.

        Returns (raw_text, parsed_verdict). Network or parse failures
        return ("", None), treated as "invalid score" by the aggregate.
        """
        prompt = self._prompts[category]
        fill = {"question": question, "answer_1": answer_1, "answer_2": answer_2}

        messages: List[Dict[str, str]] = []
        if prompt.system is not None:
            messages.append({"role": "system", "content": prompt.system.format_map(fill)})
        messages.append({"role": "user", "content": prompt.user.format_map(fill)})

        request_params = self.config.judge_chat_completions_create_params.model_copy(deep=True)
        request_params.messages = messages

        try:
            response_obj = await self.server_client.post(
                server_name=self.config.judge_model_server.name,
                url_path="/v1/chat/completions",
                json=request_params,
            )
            judge_response = NeMoGymChatCompletion.model_validate(await get_response_json(response_obj))
        except Exception:
            logger.exception("Judge call failed for category=%s; treating as invalid verdict.", category)
            return "", None

        text = self._extract_chat_completion_text(judge_response)
        verdict = self._parse_verdict(text)
        return text, verdict

    # ------------------------------------------------------------------
    # Verdict / response helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_verdict(judgment: str) -> Optional[str]:
        """Parse a verdict label from judge text.

        Returns the single unique match of ``\\[\\[([AB<>=]+)\\]\\]`` if
        all matches in the text agree; returns None on zero matches,
        multiple distinct matches, or a match that isn't one of the five
        valid labels.
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
    def _extract_response_output_text(response: Any) -> str:
        """Concatenate all ``output_text`` content from a
        ``NeMoGymResponse`` (the POLICY model's output)."""
        chunks: List[str] = []
        for output_item in response.output:
            if output_item.type != "message":
                continue
            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue
                chunks.append(content_item.text)
        return "".join(chunks)

    @staticmethod
    def _extract_chat_completion_text(completion: NeMoGymChatCompletion) -> str:
        """Extract assistant text from a ChatCompletion (the JUDGE's output)."""
        if not completion.choices:
            return ""
        # Only first choice is meaningful at n=1 (default).
        content = completion.choices[0].message.content
        return content or ""

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
        # "Double wins" — candidate beats baseline in BOTH directions
        # after the swap (base-gen says candidate is B). Conservative
        # win signal robust to positional bias.
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
            # gives a majority@k over verdict labels per task.
            answer_key="verdict_gen_base",
        )[0]
        subset_metrics = compute_subset_metrics(
            tasks,
            subset_key="category",
            score_fn=self._arena_score_fn,
            answer_key="verdict_gen_base",
        )
        metrics.update(subset_metrics)

        # Arena-Elo (MLE + 100-round bootstrap 95% CI) — the headline
        # metric for arena-hard-v2. Overall + per-category.
        metrics.update(self._compute_arena_elo_metrics(tasks))
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key: Dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        # Arena-Elo is the headline for this benchmark.
        for name in ("arena_elo/score", "arena_elo/ci_lower", "arena_elo/ci_upper", "arena_elo/invalid_scores"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", exclude_names=["no_answer"]))
        return key

    # ------------------------------------------------------------------
    # Arena-Elo (MLE logistic regression + 100-round bootstrap 95% CI)
    # ------------------------------------------------------------------
    #
    # Port of arena-hard-auto's ``show_result.py`` scoring. Given the
    # pairwise verdicts for each task, the candidate's score is its
    # predicted win-rate against the baseline, derived from a
    # two-player Elo rating fit over the implied battles.

    def _compute_arena_elo_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compute overall + per-category Arena-Elo metrics for the run."""
        # For each task, collapse all its rollouts into a single
        # (gen-base best, base-gen best) verdict pair — mirrors how
        # arena-hard-auto's repeat runs are aggregated before scoring.
        paired_scores: List[List[Optional[str]]] = []
        categories: List[Optional[str]] = []
        for rollouts in tasks:
            if not rollouts:
                continue
            gen_verdicts = [r.get("verdict_gen_base") for r in rollouts]
            base_verdicts = [r.get("verdict_base_gen") for r in rollouts]
            paired_scores.append(
                [
                    self._best_of_rollouts(gen_verdicts, reverse=False),
                    self._best_of_rollouts(base_verdicts, reverse=True),
                ]
            )
            categories.append(rollouts[0].get("category"))

        out: Dict[str, Any] = {}
        overall = self._aggregate_arena_elo(paired_scores)
        for key, val in overall.items():
            out[f"arena_elo/{key}"] = val

        # Per-category breakdown, same schema.
        buckets: Dict[str, List[List[Optional[str]]]] = defaultdict(list)
        for score, cat in zip(paired_scores, categories, strict=True):
            if cat is not None:
                buckets[cat].append(score)
        for cat, cat_scores in buckets.items():
            agg = self._aggregate_arena_elo(cat_scores)
            for key, val in agg.items():
                out[f"arena_elo/{cat}/{key}"] = val

        return out

    @staticmethod
    def _best_of_rollouts(verdicts: List[Optional[str]], reverse: bool) -> Optional[str]:
        """Collapse multiple rollouts' verdicts for a direction into a
        single "best" verdict, iterating the valid label set in the
        candidate-favoring order (or its reverse for the swapped call).

        Returns None only when no rollout produced a parseable verdict.
        """
        order = _VALID_VERDICTS if not reverse else tuple(reversed(_VALID_VERDICTS))
        for candidate in order:
            if any(v == candidate for v in verdicts):
                return candidate
        return None

    def _aggregate_arena_elo(self, scores: List[List[Optional[str]]]) -> Dict[str, Any]:
        """Return ``{score, ci_lower, ci_upper, invalid_scores, n}``
        for a list of ``[gen_best, base_best]`` verdict pairs."""
        if not scores:
            return {
                "score": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "invalid_scores": 0,
                "n": 0,
            }

        battles, num_invalid = self._get_battles_from_judgment(scores)
        if battles.empty:
            return {
                "score": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "invalid_scores": int(num_invalid),
                "n": len(scores),
            }

        # Logistic regression requires at least two classes of outcomes.
        # If every battle has the same winner (candidate sweeps or gets
        # swept), the headline is a degenerate 0% or 100% with no CI.
        # Detect by looking at the deduplicated winners (ties count as
        # a half-win for each side — see the Y row-duplication in
        # ``_compute_mle_elo``).
        raw_winners = set(battles["winner"])
        has_model_a = "model_a" in raw_winners
        has_model_b = "model_b" in raw_winners
        has_tie = any(w in raw_winners for w in ("tie", "tie (bothbad)"))
        if not ((has_model_a and has_model_b) or has_tie):
            degenerate_score = 100.0 if has_model_a and not has_model_b else 0.0
            return {
                "score": degenerate_score,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "invalid_scores": int(num_invalid),
                "n": len(scores),
            }

        online = self._compute_mle_elo(battles)
        boot = self._bootstrap(battles)

        stats = pd.DataFrame()
        stats["results"] = None
        stats["results"] = stats["results"].astype("object")
        for i, model in enumerate(online.index):
            stats.at[i, "model"] = model
            stats.at[i, "score"] = online[model]
            stats.at[i, "lower"] = np.percentile(boot[model], 2.5)
            stats.at[i, "upper"] = np.percentile(boot[model], 97.5)
            stats.at[i, "results"] = boot[model].tolist()
        stats.sort_values(by="model", inplace=True)
        stats["score"] = self._get_win_rate_column(stats, "score").tolist()
        stats["lower"] = self._get_win_rate_column(stats, "lower").tolist()
        stats["upper"] = self._get_win_rate_column(stats, "upper").tolist()

        cand = stats[stats["model"] == "candidate"]
        return {
            "score": float(cand["score"].iloc[0]),
            "ci_lower": float(round((cand["lower"] - cand["score"]).iloc[0], 2)),
            "ci_upper": float(round((cand["upper"] - cand["score"]).iloc[0], 2)),
            "invalid_scores": int(num_invalid),
            "n": len(scores),
        }

    @staticmethod
    def _get_battles_from_judgment(
        scores: List[List[Optional[str]]],
    ) -> tuple[pd.DataFrame, int]:
        """Expand pairwise verdicts into a DataFrame of
        (model_a, model_b, winner) battles.

        Mirrors arena-hard-auto's ``get_battles_from_judgment``:
        strict wins contribute ``_STRICT_WEIGHT=3`` battles,
        slight wins contribute 1, ties produce a `"tie"` row, and
        invalid / missing verdicts are counted and skipped.
        """
        battles = pd.DataFrame()
        num_invalid = 0

        weight_by_verdict: Dict[str, int] = {
            "A=B": 1,
            "A>B": 1,
            "A>>B": _STRICT_WEIGHT,
            "B>A": 1,
            "B>>A": _STRICT_WEIGHT,
        }

        for score in scores:
            assert len(score) == 2

            # Game 1: A=candidate, B=baseline.
            cur = score[0]
            weight = weight_by_verdict.get(cur, 0)
            if weight == 0:
                num_invalid += 1
            else:
                if cur == "A=B":
                    winner = "tie"
                elif cur in ("A>B", "A>>B"):
                    winner = "model_a"
                else:
                    winner = "model_b"
                battles = pd.concat(
                    [
                        battles,
                        pd.DataFrame([{"model_a": "candidate", "model_b": "baseline", "winner": winner}] * weight),
                    ]
                )

            # Game 2: A=baseline, B=candidate (winner semantics swapped).
            cur = score[1]
            weight = weight_by_verdict.get(cur, 0)
            if weight == 0:
                num_invalid += 1
            else:
                if cur == "A=B":
                    winner = "tie"
                elif cur in ("A>B", "A>>B"):
                    winner = "model_b"
                else:
                    winner = "model_a"
                battles = pd.concat(
                    [
                        battles,
                        pd.DataFrame([{"model_a": "candidate", "model_b": "baseline", "winner": winner}] * weight),
                    ]
                )

        return battles, num_invalid

    @staticmethod
    def _compute_mle_elo(
        df: pd.DataFrame,
        SCALE: int = 400,
        BASE: int = 10,
        INIT_RATING: int = 1000,
    ) -> pd.Series:
        """Fit a two-player Elo rating via logistic regression on the
        battle DataFrame, matching arena-hard-auto's
        ``compute_mle_elo``.
        """
        models = pd.concat([df["model_a"], df["model_b"]]).unique()
        models = pd.Series(np.arange(len(models)), index=models)

        # Duplicate battles to encode ties (half-win on each side).
        df = pd.concat([df, df], ignore_index=True)
        p = len(models.index)
        n = df.shape[0]

        X = np.zeros([n, p])
        X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
        X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

        Y = np.zeros(n)
        Y[df["winner"] == "model_a"] = 1.0

        tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
        tie_idx[len(tie_idx) // 2 :] = False
        Y[tie_idx] = 1.0

        lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-8)
        lr.fit(X, Y)

        elo_scores = SCALE * lr.coef_[0] + INIT_RATING
        if "baseline" in models.index:
            elo_scores += 1000 - elo_scores[models["baseline"]]
        return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)

    def _bootstrap(self, battles: pd.DataFrame) -> pd.DataFrame:
        """Resample the battle table ``arena_elo_bootstrap_rounds`` times
        and refit Elo each time, producing a distribution per model.

        On very small battle tables, a resample can accidentally contain
        only one winner class — the logistic fit would raise. We skip
        those draws rather than poison the distribution.
        """
        np.random.seed(self.config.arena_elo_bootstrap_seed)
        rows = []
        for _ in range(self.config.arena_elo_bootstrap_rounds):
            try:
                rows.append(self._compute_mle_elo(battles.sample(frac=1.0, replace=True)))
            except ValueError:
                continue
        df = pd.DataFrame(rows)
        return df[df.median().sort_values(ascending=False).index]

    @staticmethod
    def _predict_win_rate(
        elo_ratings: Dict[str, float],
        SCALE: int = 400,
        BASE: int = 10,
    ) -> pd.DataFrame:
        names = sorted(elo_ratings.keys())
        wins: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 0.0))
        for a in names:
            for b in names:
                ea = 1 / (1 + BASE ** ((elo_ratings[b] - elo_ratings[a]) / SCALE))
                wins[a][b] = ea
                wins[b][a] = 1 - ea
        data = {a: [wins[a][b] if a != b else np.nan for b in names] for a in names}
        df = pd.DataFrame(data, index=names)
        df.index.name = "model_a"
        df.columns.name = "model_b"
        return df.T

    @classmethod
    def _get_win_rate_column(cls, df: pd.DataFrame, column: str) -> pd.Series:
        """Convert a model × Elo column into a win-rate vs baseline (%)."""
        to_dict = df[["model", column]].set_index("model").to_dict()[column]
        win_rate_table = cls._predict_win_rate(to_dict)
        return win_rate_table["baseline"].fillna(0.5).apply(lambda x: round(x * 100, 2))


if __name__ == "__main__":
    ArenaJudgeServer.run_webserver()
