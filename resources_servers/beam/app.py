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
  1.0 (event ordering included, instead of tau_norm).

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
* Within one answer line, equivalence calls against the unused reference items
  run concurrently; the first YES in reference order wins, which is identical to
  upstream's sequential scan for a deterministic judge.
* A NaN tau-b (degenerate ranking, e.g. an empty alignment) is mapped to 0.0;
  upstream propagates NaN into its averages.
* A judge reply without a parseable ``score`` counts as 0.0 for that rubric item
  (``n_judge_parse_failed``); upstream raises and loses the whole conversation.
* Judge calls retry with bounded exponential backoff on 429 / 5xx. A row whose
  judge calls still fail is marked ``judge_call_failed`` and excluded from the
  headline metrics (reported through ``n_judge_call_failed``).
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
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import ConfigDict, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.beam.prompts import (
    BINARY_LLM_JUDGE_BASE_PROMPT,
    EQUIVALENCE_SYSTEM_PROMPT,
    EQUIVALENCE_USER_TEMPLATE,
    UNIFIED_LLM_JUDGE_BASE_PROMPT,
)


LOG = logging.getLogger(__name__)

ABILITIES = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)
EVENT_ORDERING = "event_ordering"

ERROR_JUDGE_CALL_FAILED = "judge_call_failed"
ERROR_BAD_METADATA = "bad_metadata"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_SCORE_RE = re.compile(r'"score"\s*:\s*\[?\s*"?([0-9]*\.?[0-9]+)')
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"(\{.*?\}|\[.*?\])", re.DOTALL)

# Upper bound on a single backoff sleep, so a long-lived 429 storm still makes progress.
_RETRY_MAX_DELAY_S = 30.0


class JudgeCallError(RuntimeError):
    """The judge endpoint failed after all retries."""


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


def parse_judge_score(raw: str) -> Optional[float]:
    """Extract ``score`` from a judge reply, mirroring upstream ``parse_json_response``.

    Returns None when no score can be recovered.
    """
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
                return float(parsed["score"])
            except (TypeError, ValueError):
                continue
    score_match = _SCORE_RE.search(text)
    return float(score_match.group(1)) if score_match else None


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


class BEAMResourcesServerConfig(BaseResourcesServerConfig):
    """Config for the BEAM resources server (LLM-as-judge scoring)."""

    name: str = "beam"
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    judge_endpoint_max_concurrency: Optional[int] = 64
    judge_max_retries: int = 5
    judge_retry_base_delay: float = 1.0
    # "binary" (default): pass (1) / fail (0) per rubric item. "partial": upstream 0 / 0.5 / 1.
    judge_scale: Literal["partial", "binary"] = "binary"
    # Which per-question score becomes the reward. Both are always reported.
    # "mean": average over rubric items (event ordering: tau_norm), as upstream.
    # "all_pass": 1.0 only if every rubric item scores 1.0 (event ordering included).
    question_scoring: Literal["mean", "all_pass"] = "mean"


class BEAMRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: Optional[Dict[str, Any]] = None


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
    judge_error: Optional[str] = None
    judge_error_detail: str = ""
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
        """Call the judge, retrying 429 / 5xx with bounded exponential backoff."""
        params = self.config.judge_responses_create_params.model_dump()
        params["input"] = messages
        max_retries = max(0, self.config.judge_max_retries)
        attempt = 0
        while True:
            async with self._semaphore:
                resp = await self.server_client.post(
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/responses",
                    json=params,
                )
                status = getattr(resp, "status", None)
                retryable = isinstance(status, int) and (status == 429 or 500 <= status < 600)
                if attempt >= max_retries or not retryable:
                    try:
                        await raise_for_status(resp)
                        result = await get_response_json(resp)
                    except Exception as exc:  # noqa: BLE001
                        raise JudgeCallError(f"{type(exc).__name__}: {exc}") from exc
                    return strip_think(response_text(NeMoGymResponse(**result))).strip()
            delay = min(self.config.judge_retry_base_delay * (2**attempt), _RETRY_MAX_DELAY_S)
            LOG.warning("beam: judge returned %s, retrying in %.1fs (%d/%d)", status, delay, attempt + 1, max_retries)
            await asyncio.sleep(delay)
            attempt += 1

    async def _judge_rubric(self, rubric: List[str], llm_response: str) -> Tuple[List[Optional[float]], List[str]]:
        """Score every rubric item; an unparseable reply yields None for that item."""
        binary = self.config.judge_scale == "binary"
        raws = await asyncio.gather(
            *(
                self._call_judge([{"role": "user", "content": build_rubric_prompt(item, llm_response, binary=binary)}])
                for item in rubric
            )
        )
        return [parse_judge_score(r) for r in raws], list(raws)

    async def _equivalent(self, first: str, second: str) -> bool:
        """Upstream ``llm_equivalence``: True when the judge reply contains "yes"."""
        reply = await self._call_judge(
            [
                {"role": "system", "content": EQUIVALENCE_SYSTEM_PROMPT},
                {"role": "user", "content": EQUIVALENCE_USER_TEMPLATE.format(first=first, second=second)},
            ]
        )
        return "yes" in reply.lower()

    async def _align_with_llm(self, reference: List[str], system: List[str]) -> List[str]:
        """Upstream ``align_with_llm``: map each system line to its first unused equivalent reference."""
        used: set[int] = set()
        system_out: List[str] = []
        for line in system:
            candidates = [i for i in range(len(reference)) if i not in used]
            matched: Optional[int] = None
            if line.strip() and candidates:
                verdicts = await asyncio.gather(*(self._equivalent(reference[i], line) for i in candidates))
                matched = next((i for i, ok in zip(candidates, verdicts) if ok), None)
            if matched is None:
                system_out.append(line)
            else:
                system_out.append(reference[matched])
                used.add(matched)
        return system_out

    async def verify(self, body: BEAMVerifyRequest) -> BEAMVerifyResponse:
        meta = body.verifier_metadata if isinstance(body.verifier_metadata, dict) else {}
        ability = str(meta.get("ability") or "")
        rubric = [str(r) for r in (meta.get("rubric") or []) if str(r).strip()]
        answer = strip_think(response_text(body.response)).strip()

        result: Dict[str, Any] = dict(
            ability=ability,
            chat_size=str(meta.get("chat_size") or ""),
            conversation_id=str(meta.get("conversation_id") or ""),
            empty_response=not answer,
            generation=answer[:2000],
        )
        reward = 0.0
        if ability not in ABILITIES or not rubric:
            LOG.warning("beam: unusable verifier_metadata (ability=%r, %d rubric items)", ability, len(rubric))
            result["judge_error"] = ERROR_BAD_METADATA
        elif answer:
            try:
                scores, raws = await self._judge_rubric(rubric, answer)
                result["rubric_scores"] = scores
                result["judge_raw"] = raws
                result["n_judge_parse_failed"] = sum(s is None for s in scores)
                result["llm_judge_score"] = sum(s or 0.0 for s in scores) / len(scores)
                result["score_mean"] = result["llm_judge_score"]
                result["score_all_pass"] = float(all(s == 1.0 for s in scores))
                if ability == EVENT_ORDERING:
                    system_canon = await self._align_with_llm(rubric, answer.split("\n"))
                    result["event_ordering"] = event_ordering_score(rubric, system_canon)
                    result["score_mean"] = result["event_ordering"]["tau_norm"]
                reward = result["score_" + self.config.question_scoring]
            except JudgeCallError as exc:
                LOG.warning("beam: judge call failed: %s", exc)
                result["judge_error"] = ERROR_JUDGE_CALL_FAILED
                result["judge_error_detail"] = str(exc)
                reward = 0.0

        data = body.model_dump()
        for key in list(result) + ["reward"]:
            data.pop(key, None)
        return BEAMVerifyResponse(**data, **result, reward=reward)

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Per-ability and per-chat-size means, plus the 10-ability macro average (upstream headline)."""
        rows = [r for task_rollouts in tasks for r in task_rollouts if _is_reward(r.get("reward"))]
        metrics: Dict[str, Any] = {}
        if not rows:
            return metrics

        metrics["n_judge_call_failed"] = sum(r.get("judge_error") == ERROR_JUDGE_CALL_FAILED for r in rows)
        metrics["n_bad_metadata"] = sum(r.get("judge_error") == ERROR_BAD_METADATA for r in rows)
        metrics["n_judge_parse_failed"] = sum(int(r.get("n_judge_parse_failed") or 0) for r in rows)
        scored = [r for r in rows if r.get("judge_error") != ERROR_JUDGE_CALL_FAILED]
        metrics["n_empty_response"] = sum(bool(r.get("empty_response")) for r in scored)
        metrics["count"] = len(scored)
        if not scored:
            return metrics

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
        """``pass@1[avg-of-k]``: each question scores the mean of its first k rollouts; ``pass@k``:
        the best of them. The ten abilities are then macro-averaged (as ``overall``). Run ``i``
        (rollout index ``i`` of every question) gives the spread across runs at the largest k."""

        def macro(per_task: List[Tuple[str, float]]) -> Optional[float]:
            by_ability: Dict[str, List[float]] = defaultdict(list)
            for ability, value in per_task:
                by_ability[ability].append(value)
            known = [sum(v) / len(v) for a, v in by_ability.items() if a in ABILITIES]
            return sum(known) / len(known) if known else None

        groups = []
        for rollouts in tasks:
            kept = [
                r for r in rollouts if _is_reward(r.get("reward")) and r.get("judge_error") != ERROR_JUDGE_CALL_FAILED
            ]
            if kept:
                groups.append(sorted(kept, key=lambda r: int(r.get("_ng_rollout_index") or 0)))
        metrics: Dict[str, float] = {}
        if not groups:
            return metrics
        max_k = max(len(g) for g in groups)
        for k in range(1, max_k + 1):
            for prefix, field in (("", "reward"), ("all_pass/", "score_all_pass")):
                value = macro(
                    [
                        (str(g[0].get("ability")), sum(float(r.get(field) or 0.0) for r in g[:k]) / len(g[:k]))
                        for g in groups
                    ]
                )
                if value is not None:
                    metrics[f"pass@1[avg-of-{k}]/{prefix}overall"] = value
                # pass@k: best of the first k rollouts (max-of-k for continuous scores).
                best = macro(
                    [(str(g[0].get("ability")), max(float(r.get(field) or 0.0) for r in g[:k])) for g in groups]
                )
                if best is not None:
                    metrics[f"pass@{k}/{prefix}overall"] = best
        runs = [
            macro([(str(g[i].get("ability")), float(g[i]["reward"])) for g in groups if len(g) > i])
            for i in range(max_k)
        ]
        runs = [r for r in runs if r is not None]
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
            if k in ("overall", "mean/overall", "all_pass/overall", "n_judge_call_failed", "n_empty_response")
            or (k.startswith("chat_size/") and k.endswith("/overall"))
            or (top is not None and k.startswith(top))
        }


if __name__ == "__main__":
    BEAMResourcesServer.run_webserver()
