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
"""RoleMRC resources server — role-play machine-reading-comprehension scoring.

Two scoring modes, selected by ``config.mode``, ported from the nemo-evaluator
BYOB benchmarks ``rolemrc`` / ``rolemrc_judge``:

* ``reference`` — single-turn reference scoring against a gold reply:
  ROUGE / BLEU / METEOR / BERTScore. ROUGE-L is the per-sample reward; the
  other metrics ride along on the verify response. A per-row ``dimension``
  (derived from the RoleMRC ``task`` suffix) lets ``compute_metrics`` break
  results down by RoleMRC's evaluation taxonomy.
* ``judge`` — LLM-as-judge across five aspects (knowledge_range,
  style_compliance, nested_instruction, multi_turn_instruction,
  instruction_priority). Each row triggers one judge call per relevant aspect
  (per its ``task`` field, see ``_EVALUATION_CONFIG``); the per-row reward is
  the mean 0/1 aspect score. Judge calls go to ``config.judge_model_server``.

Build the dataset with ``prepare_rolemrc.py`` (downloads ``Junrulu/RoleMRC``).
Upstream eval: ``RoleMRC/evaluation/{evaluation,llm_judge}.py``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from contextlib import nullcontext
from functools import lru_cache
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI
from pydantic import ConfigDict, PrivateAttr

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


LOG = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_SCORE_RE = re.compile(r"-?\d+(?:\.\d+)?")


# ── Dimension taxonomy (shared with prepare_rolemrc.py) ──────────────────

_DIMENSION_BY_SUFFIX: Tuple[Tuple[str, str], ...] = (
    ("-2ndrefused", "multi_turn"),
    ("-2ndanswer", "multi_turn"),
    ("-special-content", "nested_instruction"),
    ("-special-format", "nested_instruction"),
    ("-refused", "instruction_priority"),
)


def _task_dimension(task: str) -> str:
    for suffix, dimension in _DIMENSION_BY_SUFFIX:
        if task.endswith(suffix):
            return dimension
    return "on_scene_dialogue"


# ── Text helpers ─────────────────────────────────────────────────────────


def _strip_think(text: str) -> str:
    if not text or "</think>" not in text:
        return text or ""
    cleaned = _THINK_RE.sub("", text)
    if cleaned == text:
        cleaned = text.split("</think>", 1)[-1]
    return cleaned.strip()


def _coerce_text(content: Any) -> str:
    """Flatten Responses-API message content (str or list of parts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for c in content:
            if isinstance(c, str):
                parts.append(c)
                continue
            t = c.get("text") if isinstance(c, dict) else getattr(c, "text", None)
            if isinstance(t, str):
                parts.append(t)
        return "".join(parts)
    return "" if content is None else str(content)


def _input_messages(params: NeMoGymResponseCreateParamsNonStreaming) -> List[Dict[str, str]]:
    """Normalize ``responses_create_params.input`` to ``[{role, content}]``."""
    raw = params.input
    if isinstance(raw, str):
        return [{"role": "user", "content": raw}]
    out: List[Dict[str, str]] = []
    for item in raw or []:
        if isinstance(item, dict):
            role = item.get("role", "user")
            content = item.get("content", "")
        else:
            role = getattr(item, "role", "user")
            content = getattr(item, "content", "")
        out.append({"role": str(role).lower(), "content": _coerce_text(content)})
    return out


def _response_text(response: NeMoGymResponse) -> str:
    """Best-effort extraction of the assistant text from a NeMoGymResponse."""
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text:
        return text
    # Fallback: walk output messages.
    parts: List[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        parts.append(_coerce_text(getattr(item, "content", "")))
    return "".join(parts)


def _safe_call(label: str, fn: Callable, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- one bad sample shouldn't kill the run
        LOG.warning("RoleMRC: %s failed: %s", label, exc)
        return None


# ── Reference metrics (lazy heavy imports, mirroring upstream) ───────────


@lru_cache(maxsize=1)
def _rouge_scorer():
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)


@lru_cache(maxsize=1)
def _bert_scorer():
    from bert_score import BERTScorer

    return BERTScorer(lang="en", rescale_with_baseline=False)


@lru_cache(maxsize=1)
def _ensure_nltk_data() -> None:
    """Pre-resolve the NLTK corpora METEOR needs (pure local lookup)."""
    import nltk

    resources = (
        ("wordnet", "corpora"),
        ("omw-1.4", "corpora"),
        ("punkt", "tokenizers"),
        ("punkt_tab", "tokenizers"),
    )
    for pkg, kind in resources:
        try:
            nltk.data.find(f"{kind}/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def _compute_rouge(response: str, reference: str) -> Dict[str, float]:
    scores = _safe_call("rouge", _rouge_scorer().score, reference, response)
    if scores is None:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
    return {k: v.fmeasure for k, v in scores.items()}


def _compute_bleu(response: str, reference: str) -> float:
    """Sentence-level BLEU via sacrebleu, normalized 0-100 -> 0-1."""
    if not response.strip() or not reference.strip():
        return 0.0
    from sacrebleu import sentence_bleu

    out = _safe_call("bleu", sentence_bleu, response, [reference])
    return float(out.score) / 100.0 if out is not None else 0.0


def _compute_meteor(response: str, reference: str) -> float:
    _ensure_nltk_data()
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import meteor_score

    out = _safe_call(
        "meteor",
        meteor_score,
        [word_tokenize(reference)],
        word_tokenize(response),
    )
    return float(out) if out is not None else 0.0


def _compute_bertscore(response: str, reference: str) -> Dict[str, float]:
    bs = _safe_call("bertscore-load", _bert_scorer)
    if bs is None:
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
    out = _safe_call("bertscore", bs.score, [response], [reference])
    if out is None:
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
    p, r, f1 = out
    return {
        "bertscore_precision": float(p[0]),
        "bertscore_recall": float(r[0]),
        "bertscore_f1": float(f1[0]),
    }


# ── LLM-as-judge — 5-aspect prompts (verbatim from llm_judge.py) ─────────


_KNOWLEDGE_RANGE = """
# An LLM is playing a specific character and answer a question about the given passages:
{conversations}
# The LLM generates the following response of the question:
"{response}"
# Ideally, the LLM should generate a response according to the character's **knowledge range**.
## 1. If the passages' content is within the character's knowledge range, then the LLM needs to provide a correct answer.
## 2. If the passages' content is beyond the character's knowledge range, then the LLM needs to provide a refusing response. Except the LLM is required to give an answer in any case.
# Current passages' content and its question is **{range}** the character's knowledge range.
# Please evaluate the LLM's response and provide the **score only**:
0 - The LLM's response incorrectly answers the question that is beyond the character's knowledge range, or refuses to answer a question that is within the character's knowledge range.
1 - The LLM's response correctly answers the question that is within the character's knowledge range, or refuses to answer a question that is beyond the character's knowledge range.
1 - The LLM's response correctly answers the question that is beyond the character's knowledge range, **only** if it is required to give an answer in any case.
Score:
"""

_STYLE_COMPLIANCE = """
# An LLM is playing a specific character and answer a question about the given passages:
{conversations}
# The LLM generates the following response of the question:
"{response}"
# Ideally, the LLM should generate a response that is consistent with the **character's speech style**.
# Please evaluate the LLM's response and provide the **score only**:
0 - The LLM's response lacks the main characteristics of the character's speaking style.
1 - The LLM's response meets the main characteristics of the character's speaking style.
Score:
"""

_NESTED_INSTRUCTION = """
# An LLM is playing a specific character and answer a question about the given passages. Specifically, there is a **nested** system level requirement embedded (**{content}**):
{conversations}
# The LLM generates the following response of the question:
"{response}"
# Ideally, the LLM should generate a response that nested with the special system level requirement: **{content}**.
# Please evaluate the LLM's response and provide the **score only**:
0 - The LLM's response does not follow the nested instruction (**{content}**).
1 - The LLM's response **follows** the nested instruction (**{content}**).
Score:
"""

_MULTI_TURN_INSTRUCTION = """
# An LLM is playing a specific character and answer a question about the given passages. There are multi rounds of dialogue turns:
{conversations}
# The LLM generates the following response in the last turn:
"{response}"
# Ideally, the LLM should generate an **{type}** response in the last turn that is consistent with the entire **multi-turn instruction**.
# Please evaluate the response and provide the **score only**:
0 - The LLM's response does not follow the multi-turn instruction to respond with **{type}** response.
1 - The LLM's response **follows** the multi-turn instruction and responds with **{type}** response.
Score:
"""

_INSTRUCTION_PRIORITY = """
# An LLM is playing a specific character and answer a question about the given passages. Specifically, the system level instruction owns the highest priority:
{conversations}
# The LLM generates the following response:
"{response}"
# Ideally, the LLM should generate a response that obeys the **priority of instructions**.
## 1. The system's instruction own the highest priority.
## 2. The user's instruction own the second highest priority.
# Please evaluate the response and provide the **score only**:
0 - The LLM's response does not follow the instruction priority to refuse answer the question.
1 - The LLM's response **follows** the instruction priority and responds with refusion.
Score:
"""


_KNOWLEDGE_RANGE_WITHIN = ("knowledge_range", _KNOWLEDGE_RANGE, {"range": "within"})
_KNOWLEDGE_RANGE_OUTSIDE = ("knowledge_range", _KNOWLEDGE_RANGE, {"range": "outside"})
_STYLE = ("style_compliance", _STYLE_COMPLIANCE, {})
_NESTED = ("nested_instruction", _NESTED_INSTRUCTION, {})
_MULTI_UNANSWERABLE = ("multi_turn_instruction", _MULTI_TURN_INSTRUCTION, {"type": "unanswerable"})
_MULTI_ANSWERABLE = ("multi_turn_instruction", _MULTI_TURN_INSTRUCTION, {"type": "answerable"})
_PRIORITY = ("instruction_priority", _INSTRUCTION_PRIORITY, {})

_EVALUATION_CONFIG: Dict[str, List[Tuple[str, str, Dict[str, str]]]] = {
    "role_related_mrc_answer_with_narration": [_KNOWLEDGE_RANGE_WITHIN, _STYLE],
    "role_related_mrc_answer_no_narration": [_KNOWLEDGE_RANGE_WITHIN],
    "role_unrelated_mrc_refused_with_narration": [_KNOWLEDGE_RANGE_OUTSIDE],
    "role_unrelated_mrc_refused_no_narration": [_KNOWLEDGE_RANGE_OUTSIDE, _STYLE],
    "role_related_mrc_refused_with_narration": [_KNOWLEDGE_RANGE_WITHIN],
    "role_unrelated_mrc_answer_with_narration": [_KNOWLEDGE_RANGE_OUTSIDE],
    "role_related_mrc_refused_no_narration": [_STYLE],
    "role_unrelated_mrc_answer_no_narration": [_STYLE],
    "role_related_mrc_answer_with_narration-special-content": [_NESTED],
    "role_related_mrc_answer_with_narration-special-format": [_NESTED],
    "role_related_mrc_answer_no_narration-special-content": [_NESTED],
    "role_related_mrc_answer_no_narration-special-format": [_NESTED],
    "role_related_mrc_refused_with_narration-2ndrefused": [_MULTI_UNANSWERABLE],
    "role_related_mrc_refused_no_narration-2ndrefused": [_MULTI_UNANSWERABLE],
    "role_unrelated_mrc_refused_with_narration-2ndanswer": [_MULTI_ANSWERABLE],
    "role_unrelated_mrc_refused_no_narration-2ndanswer": [_MULTI_ANSWERABLE],
    "role_related_mrc_answer_with_narration-refused": [_PRIORITY],
    "role_related_mrc_answer_no_narration-refused": [_PRIORITY],
}


_NESTED_INSTRUCTION_LEAD = (
    "You love to ",
    "You will ",
    "You must ",
    "You prefer to ",
    "You would like to ",
    "You are used to ",
    "You should ",
    "You are in the habit of ",
)


def _build_conversation_text(messages: List[Dict[str, str]]) -> str:
    """Port of ``llm_judge.build_conversation``."""
    parts: List[str] = []
    for turn in messages:
        role = turn["role"].lower()
        content = turn["content"]
        if role == "system":
            parts.append(f'System Instruction: "{content}"')
        elif role == "user":
            parts.append(f'User Query: "{content}"')
        elif role == "assistant":
            parts.append(f'LLM Response: "{content}"')
    return "\n".join(parts) + ("\n" if parts else "")


def _extract_nested_content(system_content: str) -> str:
    """Pull the second sentence of the system prompt and strip the lead-in."""
    sentences = system_content.split(". ")
    second = sentences[1] if len(sentences) > 1 else system_content
    for lead in _NESTED_INSTRUCTION_LEAD:
        if second.startswith(lead):
            second = second[len(lead) :]
            break
    return second.rstrip(".")


def _build_judge_prompts(
    task: str,
    conversation_text: str,
    system_content: str,
    response: str,
) -> List[Tuple[str, str]]:
    aspects = _EVALUATION_CONFIG.get(task, [])
    prompts: List[Tuple[str, str]] = []
    for aspect_name, template, fmt in aspects:
        kwargs = {"conversations": conversation_text, "response": response, **fmt}
        if aspect_name == "nested_instruction":
            kwargs["content"] = _extract_nested_content(system_content)
        prompts.append((aspect_name, template.format(**kwargs)))
    return prompts


def _parse_judge_score(text: str) -> int:
    cleaned = (text or "").replace("Score:", "").strip()
    match = _SCORE_RE.search(cleaned)
    if not match:
        return 0
    try:
        return 1 if int(round(float(match.group(0)))) >= 1 else 0
    except (TypeError, ValueError):
        return 0


# ── Server config + request/response shapes ──────────────────────────────


class RoleMRCResourcesServerConfig(BaseResourcesServerConfig):
    """Config for the rolemrc resources server.

    Attributes:
        mode: ``reference`` for ROUGE/BLEU/METEOR/BERTScore scoring (reward =
            ROUGE-L), or ``judge`` for the 5-aspect LLM-as-judge (reward = mean
            0/1 aspect score).
        include_bertscore: Compute BERTScore in ``reference`` mode. Default True
            for parity with upstream; downloads a roberta-large checkpoint on
            first use. Turn off for lightweight RL reward signals.
        judge_model_server / judge_responses_create_params: required in
            ``judge`` mode — the model server graded aspects are sent to.
        judge_endpoint_max_concurrency: bound on concurrent judge HTTP calls.
            None disables limiting.
    """

    name: str = "rolemrc"
    mode: Literal["reference", "judge"] = "reference"
    include_bertscore: bool = True

    judge_model_server: Optional[ModelServerRef] = None
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    judge_endpoint_max_concurrency: Optional[int] = 64


class RoleMRCRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    reference: str = ""
    task: str = ""
    dimension: str = ""


class RoleMRCVerifyRequest(RoleMRCRunRequest, BaseVerifyRequest):
    pass


class RoleMRCVerifyResponse(BaseVerifyResponse):
    # Reference metrics, per-aspect judge scores, etc. ride along here.
    model_config = ConfigDict(extra="allow")

    task: str = ""
    dimension: str = ""
    generation: str = ""


class RoleMRCResourcesServer(SimpleResourcesServer):
    config: RoleMRCResourcesServerConfig

    _judge_semaphore: Any = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.config.mode == "judge":
            if self.config.judge_model_server is None or self.config.judge_responses_create_params is None:
                raise ValueError(
                    "rolemrc judge mode requires `judge_model_server` and `judge_responses_create_params`."
                )
            mc = self.config.judge_endpoint_max_concurrency
            self._judge_semaphore = nullcontext() if mc is None else asyncio.Semaphore(mc)
        else:
            # Pre-load CPU scorers off the request path (one-time startup cost
            # instead of blocking — and racing on — the first verify call).
            _rouge_scorer()
            _ensure_nltk_data()
            if self.config.include_bertscore:
                _bert_scorer()

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    async def verify(self, body: RoleMRCVerifyRequest) -> RoleMRCVerifyResponse:
        if self.config.mode == "judge":
            return await self._verify_judge(body)
        return await self._verify_reference(body)

    # --- reference-metric scoring ----------------------------------------

    async def _verify_reference(self, body: RoleMRCVerifyRequest) -> RoleMRCVerifyResponse:
        response = _strip_think(_response_text(body.response))
        reference = str(body.reference or "")
        task = body.task or ""
        dimension = body.dimension or _task_dimension(task)

        # ROUGE/BLEU/METEOR/BERTScore are CPU-bound (BERTScore is a roberta-large
        # forward pass). Run in a worker thread so verify() doesn't block the
        # event loop and concurrent rollouts can overlap.
        metrics = await asyncio.to_thread(self._score_reference, response, reference)

        rouge_l = float(metrics.get("rougeL", 0.0))
        data = body.model_dump()
        data["dimension"] = dimension
        return RoleMRCVerifyResponse(
            **data,
            reward=rouge_l,
            generation=response[:500],
            **metrics,
        )

    def _score_reference(self, response: str, reference: str) -> Dict[str, float]:
        """Synchronous, CPU-bound reference metrics — called via asyncio.to_thread."""
        metrics: Dict[str, float] = {}
        metrics.update(_compute_rouge(response, reference))
        metrics["bleu"] = _compute_bleu(response, reference)
        metrics["meteor"] = _compute_meteor(response, reference)
        if self.config.include_bertscore:
            metrics.update(_compute_bertscore(response, reference))
        return metrics

    # --- LLM-as-judge scoring --------------------------------------------

    async def _verify_judge(self, body: RoleMRCVerifyRequest) -> RoleMRCVerifyResponse:
        response = _strip_think(_response_text(body.response))
        task = body.task or ""
        dimension = body.dimension or _task_dimension(task)

        messages = _input_messages(body.responses_create_params)
        conversation_text = _build_conversation_text(messages)
        system_content = next((m["content"] for m in messages if m["role"] == "system"), "")

        prompts = _build_judge_prompts(task, conversation_text, system_content, response)
        data = body.model_dump()
        data["dimension"] = dimension

        if not prompts:
            LOG.warning(
                "RoleMRC judge: no evaluation config for task %r — skipping (reward=0). Known tasks: %s",
                task,
                ", ".join(sorted(_EVALUATION_CONFIG)),
            )
            return RoleMRCVerifyResponse(
                **data,
                reward=0.0,
                generation=response[:500],
                judge_skipped=True,
            )

        aspect_scores: Dict[str, int] = {}
        bad: List[str] = []
        errors: List[str] = []
        for aspect_name, prompt in prompts:
            text = await self._call_judge(aspect_name, prompt)
            if text is None:
                errors.append(aspect_name)
                aspect_scores[aspect_name] = 0
                bad.append(aspect_name)
                continue
            LOG.debug("RoleMRC judge[%s] raw response: %r", aspect_name, text[:300])
            score = _parse_judge_score(text)
            aspect_scores[aspect_name] = score
            if not _SCORE_RE.search((text or "").replace("Score:", "")):
                LOG.warning(
                    "RoleMRC judge[%s] response had no parseable score (defaulting to 0): %r",
                    aspect_name,
                    text[:300],
                )
                bad.append(aspect_name)
            else:
                LOG.debug("RoleMRC judge[%s] score=%d", aspect_name, score)

        reward = sum(aspect_scores.values()) / len(aspect_scores)
        per_aspect = {f"aspect_{k}": float(v) for k, v in aspect_scores.items()}
        return RoleMRCVerifyResponse(
            **data,
            reward=reward,
            generation=response[:500],
            aspects=aspect_scores,
            n_aspects=len(aspect_scores),
            bad_aspects=bad,
            judge_errors=errors,
            judge_response=text,
            **per_aspect,
        )

    async def _call_judge(self, aspect_name: str, prompt: str) -> Optional[str]:
        """One judge call for a single aspect; returns text or None on failure."""
        params = self.config.judge_responses_create_params.model_copy(deep=True)
        params.input = [NeMoGymEasyInputMessage(role="user", content=prompt)]
        try:
            async with self._judge_semaphore:
                resp = await self.server_client.post(
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/responses",
                    json=params,
                )
                judge_response = NeMoGymResponse.model_validate(await get_response_json(resp))
        except Exception as exc:  # noqa: BLE001 -- retry-by-aspect is intentional
            LOG.warning("RoleMRC judge[%s] call failed: %s", aspect_name, exc, exc_info=True)
            return None
        text = _strip_think(_response_text(judge_response))
        if not text:
            LOG.warning("RoleMRC judge[%s] returned empty response text", aspect_name)
        return text

    # --- aggregation -----------------------------------------------------

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        rows = [r for task_rollouts in tasks for r in task_rollouts]
        metrics: Dict[str, Any] = {}

        rewards = [r["reward"] for r in rows if isinstance(r.get("reward"), (int, float))]
        if rewards:
            metrics["mean_reward"] = sum(rewards) / len(rewards)
            metrics["count"] = len(rewards)

        by_dim: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            rw = r.get("reward")
            if isinstance(rw, (int, float)):
                by_dim[r.get("dimension") or "unknown"].append(rw)
        for dim, vals in sorted(by_dim.items()):
            metrics[f"dimension/{dim}/mean_reward"] = sum(vals) / len(vals)
            metrics[f"dimension/{dim}/count"] = len(vals)

        by_aspect: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            for k, v in r.items():
                if k.startswith("aspect_") and isinstance(v, (int, float)):
                    by_aspect[k[len("aspect_") :]].append(v)
        for asp, vals in sorted(by_aspect.items()):
            metrics[f"aspect/{asp}/mean"] = sum(vals) / len(vals)
            metrics[f"aspect/{asp}/count"] = len(vals)

        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in ("mean_reward", "mean/reward"):
            if k in agent_metrics:
                out[k] = agent_metrics[k]
        for k, v in agent_metrics.items():
            if k.endswith("/mean_reward") or k.endswith("/mean"):
                out[k] = v
        return out


if __name__ == "__main__":
    RoleMRCResourcesServer.run_webserver()
