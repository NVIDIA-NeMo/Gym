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
"""BEAM resources server — long-term-memory probing questions scored by an LLM judge.

Ports the evaluation of BEAM ("Beyond a Million Tokens", arXiv:2510.27246,
https://github.com/mohammadtavakoli78/BEAM) to a NeMo Gym resources server. The
conversation history is laid into ``responses_create_params.input`` at
dataset-prep time (``benchmarks/beam/prepare.py``), so ``verify()`` only needs
the probing question's rubric and the model's answer.

Scoring follows upstream ``src/evaluation/compute_metrics.py`` and
``report_results.py``:

* Every ability except event ordering: each rubric item is judged independently
  with ``unified_llm_judge_base_prompt`` (score 0.0 / 0.5 / 1.0; 0 / 1 with the
  default binary judge), and the row reward is the mean over rubric items
  (``llm_judge_score``).
* ``event_ordering``: the answer is split on newlines, each line is aligned to an
  unused reference event via a YES/NO LLM equivalence call (first match wins),
  and the reward is the normalised Kendall tau-b, ``(tau_b + 1) / 2``, over the
  union of reference and aligned items (``tau_norm``). Precision / recall / F1
  and the rubric ``llm_judge_score`` are reported alongside, as upstream does.

Rewards are therefore continuous in [0, 1], exactly as upstream reports them.

Defaults differ from upstream in one place: the judge is binary.

* ``judge_scale: binary`` (default) judges each rubric item pass / fail;
  ``partial`` restores upstream's 0 / 0.5 / 1 scale.
* ``question_scoring: mean`` (default, as upstream) averages rubric items;
  ``all_pass`` makes a question score 1.0 only when every rubric item scores
  1.0; for event ordering the matched events must also appear in reference order.

Upstream reproduction is ``judge_scale: partial`` + ``question_scoring: mean``.

Every row carries both ``score_mean`` and ``score_all_pass``, and
``compute_metrics`` reports ``mean/overall`` and ``all_pass/overall`` alongside
the configured headline ``overall``.

Deliberate, minimal deviations from upstream:

* ``<think>`` blocks are stripped from the answer before judging.
* An empty answer short-circuits to 0.0 without calling the judge
  (``empty_response``).
* Blank answer lines in event ordering are treated as unmatched without a judge
  call (an empty snippet cannot describe the same event as a reference item).
  The line still takes part in the tau computation, as upstream.
* Within one answer line, the unused reference items are checked in reference
  order in small concurrent batches, stopping at the batch that holds the first
  YES; the first YES in reference order wins, which is identical to upstream's
  sequential scan for a deterministic judge.
* A NaN tau-b (degenerate ranking, e.g. an empty alignment) is mapped to 0.0;
  upstream propagates NaN into its averages.
* A judge reply without a valid ``score`` (unparseable, non-finite, or outside
  the scale) counts as 0.0 for that rubric item (``n_judge_parse_failed``);
  upstream raises and loses the whole conversation.
* A judge call that fails (the judge model server has already retried) raises
  ``nemo_gym.verification.judge.JudgeError``: ``judge_failsafe`` masks the row and routes it to
  the failures sidecar, so it is neither scored as a policy zero nor counted in
  the metrics, and ``gym eval reverify --judge-failed-only`` can retry it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import statistics
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Awaitable, ClassVar, Dict, List, Literal, Optional, Tuple, get_args

from pydantic import ConfigDict, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.reward_profile import compute_subset_metrics
from nemo_gym.verification import JudgeError, call_judge
from resources_servers.beam.prompts import (
    BINARY_LLM_JUDGE_BASE_PROMPT,
    EQUIVALENCE_SYSTEM_PROMPT,
    EQUIVALENCE_USER_TEMPLATE,
    UNIFIED_LLM_JUDGE_BASE_PROMPT,
)
from resources_servers.beam.task_data import TaskData


LOG = logging.getLogger(__name__)

# The ten BEAM memory abilities; TaskData's ``ability`` Literal is the single source of truth.
ABILITIES: Tuple[str, ...] = get_args(TaskData.model_fields["ability"].annotation)
EVENT_ORDERING = "event_ordering"

# Valid judge scores per scale; anything else is a parse failure.
_ALLOWED_SCORES = {"binary": frozenset({0.0, 1.0}), "partial": frozenset({0.0, 0.5, 1.0})}
# Reference events checked concurrently per batch when aligning one answer line.
_EQUIVALENCE_BATCH_SIZE = 4

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_SCORE_RE = re.compile(r'"score"\s*:\s*\[?\s*"?([0-9]*\.?[0-9]+)')
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"(\{.*?\}|\[.*?\])", re.DOTALL)


def strip_think(text: str) -> str:
    """Drop reasoning blocks, including a truncated (unclosed) ``<think>``."""
    if not text:
        return ""
    if "<think>" in text and "</think>" not in text:
        return text.split("<think>", 1)[0].strip()
    if "</think>" not in text:
        return text
    cleaned = _THINK_RE.sub("", text)
    if cleaned == text:
        cleaned = text.split("</think>", 1)[-1]
    return cleaned.strip()


def response_text(response: Optional[NeMoGymResponse]) -> str:
    """Concatenate the assistant message text of a response (reasoning items excluded)."""
    if response is None:
        return ""
    parts: List[str] = []
    for item in response.output or []:
        if getattr(item, "type", None) != "message":
            continue
        for c in getattr(item, "content", None) or []:
            text = getattr(c, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def parse_judge_score(raw: str, allowed: frozenset[float] = _ALLOWED_SCORES["partial"]) -> Optional[float]:
    """Extract ``score`` from a judge reply, mirroring upstream ``parse_json_response``.

    Returns None when no score can be recovered or the score is not one of ``allowed``
    (so an out-of-range or NaN score cannot leave the [0, 1] reward range).
    """

    def valid(value: float) -> Optional[float]:
        return value if math.isfinite(value) and value in allowed else None

    text = (raw or "").strip()
    fenced = _JSON_FENCE_RE.search(text) if text.startswith("```") else None
    candidates = [fenced.group(1).strip()] if fenced else []
    candidates.append(text)
    obj_match = _JSON_OBJ_RE.search(text)
    if obj_match:
        candidates.append(obj_match.group(1))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "score" in parsed:
            try:
                return valid(float(parsed["score"]))
            except (TypeError, ValueError):
                continue
    score_match = _SCORE_RE.search(text)
    return valid(float(score_match.group(1))) if score_match else None


def build_rubric_prompt(rubric_item: str, llm_response: str, *, binary: bool = False) -> str:
    """Render the judge prompt (same replacement order as upstream); ``binary`` drops the 0.5 tier."""
    template = BINARY_LLM_JUDGE_BASE_PROMPT if binary else UNIFIED_LLM_JUDGE_BASE_PROMPT
    return template.replace("<rubric_item>", rubric_item).replace("<llm_response>", llm_response)


def kendall_tau_b(x: List[int], y: List[int]) -> float:
    """Kendall tau-b (same as ``scipy.stats.kendalltau(variant="b")``); NaN when either ranking is constant."""
    concordant = discordant = 0
    n = len(x)
    pairs = n * (n - 1) // 2
    ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            ties_x += dx == 0
            ties_y += dy == 0
            if dx and dy:
                if (dx > 0) == (dy > 0):
                    concordant += 1
                else:
                    discordant += 1
    denom = math.sqrt((pairs - ties_x) * (pairs - ties_y))
    return (concordant - discordant) / denom if denom else math.nan


def event_ordering_score(reference: List[str], system_canon: List[str]) -> Dict[str, float]:
    """Port of upstream ``event_ordering_score`` after alignment.

    ``system_canon`` is the answer's line list where every aligned line has been
    replaced by the matching reference string.
    """
    tp = len(set(reference) & set(system_canon))
    fp = len([x for x in system_canon if x not in reference])
    fn = len([x for x in reference if x not in system_canon])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    union = list(dict.fromkeys(reference + system_canon))
    tie_rank = len(union) + 1

    def to_rank(seq: List[str]) -> List[int]:
        r = {item: i + 1 for i, item in enumerate(seq)}
        return [r.get(u, tie_rank) for u in union]

    tau_b = kendall_tau_b(to_rank(reference), to_rank(system_canon))
    tau_norm = 0.0 if math.isnan(tau_b) else (tau_b + 1) / 2
    return dict(precision=precision, recall=recall, f1=f1, tau_norm=tau_norm, final_score=tau_norm * f1)


def _is_reward(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def events_in_reference_order(reference: List[str], system_canon: List[str]) -> bool:
    """True when every reference event was aligned and the aligned events keep the reference order."""
    return [x for x in system_canon if x in reference] == reference


async def _gather_or_cancel(coros: List[Awaitable[Any]]) -> List[Any]:
    """Run ``coros`` concurrently; the first ``JudgeError`` cancels the rest and is re-raised as is."""
    try:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(c) for c in coros]
    except* JudgeError as group:
        raise group.exceptions[0] from None
    return [t.result() for t in tasks]


class BEAMResourcesServerConfig(BaseResourcesServerConfig):
    """Config for the BEAM resources server (LLM-as-judge scoring)."""

    # verify() depends only on the request, this config and the judge.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    name: str = "beam"
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    judge_endpoint_max_concurrency: Optional[int] = 64
    # "binary" (default): pass (1) / fail (0) per rubric item. "partial": upstream 0 / 0.5 / 1.
    judge_scale: Literal["partial", "binary"] = "binary"
    # Which per-question score becomes the reward. Both are always reported.
    # "mean": average over rubric items (event ordering: tau_norm), as upstream.
    # "all_pass": 1.0 only if every rubric item scores 1.0 (event ordering: and in reference order).
    question_scoring: Literal["mean", "all_pass"] = "mean"


class BEAMRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: TaskData


class BEAMVerifyRequest(BEAMRunRequest, BaseVerifyRequest):
    pass


class BEAMVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    ability: str = ""
    chat_size: str = ""
    conversation_id: str = ""
    llm_judge_score: float = 0.0
    score_mean: float = 0.0
    score_all_pass: float = 0.0
    rubric_scores: List[Optional[float]] = []
    judge_raw: List[str] = []
    event_ordering: Optional[Dict[str, float]] = None
    n_judge_parse_failed: int = 0
    empty_response: bool = False
    generation: str = ""


class BEAMResourcesServer(SimpleResourcesServer):
    config: BEAMResourcesServerConfig

    _semaphore: Any = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        mc = self.config.judge_endpoint_max_concurrency
        self._semaphore = asyncio.Semaphore(mc) if mc is not None else nullcontext()

    async def _call_judge(self, messages: List[Dict[str, str]]) -> str:
        """One judge call; failures (after the judge model server's own retries) raise ``JudgeError``."""
        params = self.config.judge_responses_create_params.model_dump()
        params["input"] = messages
        async with self._semaphore:
            response = await call_judge(
                self.server_client,
                server_name=self.config.judge_model_server.name,
                url_path="/v1/responses",
                json=params,
                response_model=NeMoGymResponse,
            )
        return strip_think(response_text(response)).strip()

    async def _judge_rubric(self, rubric: List[str], llm_response: str) -> Tuple[List[Optional[float]], List[str]]:
        """Score every rubric item; an invalid reply yields None for that item."""
        binary = self.config.judge_scale == "binary"
        raws = await _gather_or_cancel(
            [
                self._call_judge([{"role": "user", "content": build_rubric_prompt(item, llm_response, binary=binary)}])
                for item in rubric
            ]
        )
        allowed = _ALLOWED_SCORES[self.config.judge_scale]
        return [parse_judge_score(r, allowed) for r in raws], list(raws)

    async def _equivalent(self, first: str, second: str) -> bool:
        """Upstream ``llm_equivalence``: True when the judge reply contains "yes"."""
        reply = await self._call_judge(
            [
                {"role": "system", "content": EQUIVALENCE_SYSTEM_PROMPT},
                {"role": "user", "content": EQUIVALENCE_USER_TEMPLATE.format(first=first, second=second)},
            ]
        )
        return "yes" in reply.lower()

    async def _first_equivalent(self, reference: List[str], candidates: List[int], line: str) -> Optional[int]:
        """First candidate (in reference order) the judge calls equivalent to ``line``; batches stop early."""
        for start in range(0, len(candidates), _EQUIVALENCE_BATCH_SIZE):
            batch = candidates[start : start + _EQUIVALENCE_BATCH_SIZE]
            verdicts = await _gather_or_cancel([self._equivalent(reference[i], line) for i in batch])
            matched = next((i for i, ok in zip(batch, verdicts) if ok), None)
            if matched is not None:
                return matched
        return None

    async def _align_with_llm(self, reference: List[str], system: List[str]) -> List[str]:
        """Upstream ``align_with_llm``: map each system line to its first unused equivalent reference."""
        used: set[int] = set()
        system_out: List[str] = []
        for line in system:
            candidates = [i for i in range(len(reference)) if i not in used]
            matched = await self._first_equivalent(reference, candidates, line) if line.strip() else None
            if matched is None:
                system_out.append(line)
            else:
                system_out.append(reference[matched])
                used.add(matched)
        return system_out

    async def verify(self, body: BEAMVerifyRequest) -> BEAMVerifyResponse:
        meta = body.verifier_metadata
        rubric = [r for r in meta.rubric if r.strip()]
        answer = strip_think(response_text(body.response)).strip()

        result: Dict[str, Any] = dict(
            ability=meta.ability,
            chat_size=meta.chat_size or "",
            conversation_id=meta.conversation_id or "",
            empty_response=not answer,
            generation=answer[:2000],
        )
        reward = 0.0
        if answer and rubric:
            # A failing judge call raises JudgeError out of verify(); judge_failsafe handles the row.
            scores, raws = await self._judge_rubric(rubric, answer)
            result["rubric_scores"] = scores
            result["judge_raw"] = raws
            result["n_judge_parse_failed"] = sum(s is None for s in scores)
            result["llm_judge_score"] = sum(s or 0.0 for s in scores) / len(scores)
            result["score_mean"] = result["llm_judge_score"]
            result["score_all_pass"] = float(all(s == 1.0 for s in scores))
            if meta.ability == EVENT_ORDERING:
                system_canon = await self._align_with_llm(rubric, answer.split("\n"))
                result["event_ordering"] = event_ordering_score(rubric, system_canon)
                result["score_mean"] = result["event_ordering"]["tau_norm"]
                if not events_in_reference_order(rubric, system_canon):
                    result["score_all_pass"] = 0.0
            reward = result["score_" + self.config.question_scoring]

        data = body.model_dump()
        for key in list(result) + ["reward"]:
            data.pop(key, None)
        return BEAMVerifyResponse(**data, **result, reward=reward)

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Per-ability and per-chat-size means, plus the 10-ability macro average (upstream headline).

        Rows whose judge failed are masked by ``judge_failsafe`` and removed by the framework
        before this runs, so every row here was scored.
        """
        scored = [r for task_rollouts in tasks for r in task_rollouts if _is_reward(r.get("reward"))]
        metrics: Dict[str, Any] = {}
        if not scored:
            return metrics

        metrics["n_judge_parse_failed"] = sum(int(r.get("n_judge_parse_failed") or 0) for r in scored)
        metrics["n_empty_response"] = sum(bool(r.get("empty_response")) for r in scored)
        metrics["count"] = len(scored)

        def add_group(prefix: str, group: List[Dict[str, Any]], field: str, *, per_ability: bool) -> None:
            by_ability: Dict[str, List[float]] = defaultdict(list)
            for r in group:
                by_ability[str(r.get("ability") or "unknown")].append(float(r.get(field) or 0.0))
            if per_ability:
                for ability, vals in sorted(by_ability.items()):
                    metrics[f"{prefix}{ability}/score"] = sum(vals) / len(vals)
                    metrics[f"{prefix}{ability}/count"] = len(vals)
            known = [sum(v) / len(v) for a, v in by_ability.items() if a in ABILITIES]
            if known:
                metrics[f"{prefix}overall"] = sum(known) / len(known)

        by_size: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in scored:
            by_size[str(r.get("chat_size") or "unknown")].append(r)
        # Headline: the configured reward. Both question scorings are reported alongside.
        add_group("", scored, "reward", per_ability=True)
        for size, group in sorted(by_size.items()):
            add_group(f"chat_size/{size}/", group, "reward", per_ability=True)
        for scoring in ("mean", "all_pass"):
            add_group(f"{scoring}/", scored, f"score_{scoring}", per_ability=True)
            for size, group in sorted(by_size.items()):
                add_group(f"{scoring}/chat_size/{size}/", group, f"score_{scoring}", per_ability=False)
        metrics["mean_reward"] = sum(float(r["reward"]) for r in scored) / len(scored)
        metrics.update(self._pass_at_1_avg_of_k(tasks))
        return metrics

    @staticmethod
    def _pass_at_1_avg_of_k(tasks: List[List[Dict[str, Any]]]) -> Dict[str, float]:
        """``pass@1[avg-of-k]`` and ``pass@k`` for ``reward`` and ``all_pass``, macro-averaged over abilities.

        The per-ability math is the shared ``compute_subset_metrics`` / ``compute_pass_majority_metrics``
        (each question scores the mean of its first k rollouts, or pass@k; percentages); the ten
        abilities are then averaged as BEAM's ``overall`` is, and rescaled to [0, 1]. Rollouts are
        ordered by ``_ng_rollout_index`` first. The spread across runs is computed here instead: run
        ``i`` is rollout index ``i`` of every question that has it, so a missing rollout (a masked
        judge failure or a policy timeout) cannot shift the other runs.
        """
        ordered = [
            sorted(
                (r for r in rollouts if _is_reward(r.get("reward"))),
                key=lambda r: int(r.get("_ng_rollout_index") or 0),
            )
            for rollouts in tasks
        ]
        ordered = [rollouts for rollouts in ordered if rollouts]
        if not ordered:
            return {}
        per_ability = compute_subset_metrics(
            ordered,
            subset_key="ability",
            score_fn=lambda r: {"reward": float(r["reward"]), "all_pass": float(r.get("score_all_pass") or 0.0)},
        )

        def macro(metric: str, name: str) -> Optional[float]:
            values = [per_ability[f"{a}/{metric}/{name}"] for a in ABILITIES if f"{a}/{metric}/{name}" in per_ability]
            return sum(values) / len(values) / 100.0 if values else None

        metrics: Dict[str, float] = {}
        max_k = max(len(rollouts) for rollouts in ordered)
        for k in range(1, max_k + 1):
            for prefix, name in (("", "reward"), ("all_pass/", "all_pass")):
                for metric in (f"pass@1[avg-of-{k}]", f"pass@{k}"):
                    value = macro(metric, name)
                    if value is not None:
                        metrics[f"{metric}/{prefix}overall"] = value

        def overall(per_task: List[Tuple[str, float]]) -> Optional[float]:
            by_ability: Dict[str, List[float]] = defaultdict(list)
            for ability, value in per_task:
                by_ability[ability].append(value)
            known = [sum(v) / len(v) for a, v in by_ability.items() if a in ABILITIES]
            return sum(known) / len(known) if known else None

        by_index = [{int(r.get("_ng_rollout_index") or 0): r for r in rollouts} for rollouts in ordered]
        n_runs = max(max(runs) for runs in by_index) + 1
        runs = [
            overall([(str(r[i].get("ability")), float(r[i]["reward"])) for r in by_index if i in r])
            for i in range(n_runs)
        ]
        runs = [value for value in runs if value is not None]
        if len(runs) > 1:
            std_dev = statistics.stdev(runs)
            metrics[f"pass@1[avg-of-{max_k}]/overall/std_dev_across_runs"] = std_dev
            metrics[f"pass@1[avg-of-{max_k}]/overall/std_err_across_runs"] = std_dev / math.sqrt(len(runs))
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        ks = [int(m.group(1)) for k in agent_metrics if (m := re.match(r"pass@1\[avg-of-(\d+)\]/", k))]
        top = (f"pass@1[avg-of-{max(ks)}]/", f"pass@{max(ks)}/") if ks else None
        return {
            k: v
            for k, v in agent_metrics.items()
            if k in ("overall", "mean/overall", "all_pass/overall", "n_empty_response")
            or (k.startswith("chat_size/") and k.endswith("/overall"))
            or (top is not None and k.startswith(top))
        }


if __name__ == "__main__":
    BEAMResourcesServer.run_webserver()
