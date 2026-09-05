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

Two scoring modes, selected by ``config.mode``:

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

``compute_metrics`` reproduces every number the RoleMRC report quotes, none of
which is a mean of per-row rewards: ``auto/<metric>/mean`` for the reference
metrics, ``aspect/<aspect>/mean`` for the five judge aspects, and the three
judge roll-ups ``judge/avg_simple`` / ``judge/avg_weighted`` /
``judge/avg_simple_no_mt`` (the headline metric — see :func:`_judge_rollups`).

These roll-ups are only produced when the caller asks for them via
``/aggregate_metrics``. A harness that drives this server through ``/verify``
alone gets per-row rewards and has to aggregate them itself; see
``score_rolemrc_report.py`` for rebuilding the roll-ups from per-row output.

Build the dataset with ``prepare_rolemrc.py`` (downloads ``Junrulu/RoleMRC``).
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import threading
from collections import Counter, defaultdict
from contextlib import nullcontext
from functools import lru_cache
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

# Pre-import packages that nltk pulls in during its init so they are already in
# sys.modules before nltk's inisec.py finder is installed. nltk>=3.9 blocks any
# import originating from nltk if the module path falls inside the process CWD —
# which happens in CI where the server venv lives inside the repo root.
import defusedxml.ElementTree  # noqa: F401
import regex  # noqa: F401
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
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)


LOG = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_BLEU_MAX_ORDER = 4

# Reasoning-trace end tags recognized in a policy response.
_JUDGE_REASONING_END_TAGS = ("</think>", "<channel|>")

# Guards every call into NLTK's WordNet reader — see _compute_meteor.
_WORDNET_LOCK = threading.Lock()


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


def _strip_reasoning_for_judge(text: str) -> str:
    """Drop a reasoning trace: split on each end tag, keep the last segment.

    Unlike ``_strip_think``, an unclosed ``<think>`` prefix survives.
    """
    out = text or ""
    for end_tag in _JUDGE_REASONING_END_TAGS:
        if end_tag in out:
            out = out.split(end_tag)[-1].strip()
    return out


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


def _normalize_turns(raw: Any) -> List[Dict[str, str]]:
    """Normalize a conversation (str, or list of dicts/objects) to ``[{role, content}]``."""
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


def _input_messages(params: NeMoGymResponseCreateParamsNonStreaming) -> List[Dict[str, str]]:
    """Normalize ``responses_create_params.input`` to ``[{role, content}]``."""
    return _normalize_turns(params.input)


def _conversation_messages(body: "RoleMRCVerifyRequest") -> List[Dict[str, str]]:
    """The conversation the judge prompt quotes.

    Prefers ``verifier_metadata["conversation"]``: an external runner may not
    deliver ``responses_create_params`` to /verify, but ``verifier_metadata`` is
    forwarded verbatim.
    """
    meta = getattr(body, "verifier_metadata", None)
    turns = meta.get("conversation") if isinstance(meta, dict) else None
    if isinstance(turns, (str, list)) and turns:
        return _normalize_turns(turns)
    return _input_messages(body.responses_create_params)


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


# ── Reference metrics (lazy heavy imports) ───────────────────────────────


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
    """Resolve and eagerly materialize the NLTK corpora METEOR needs.

    ``nltk.data.find`` only locates files; the loaders are lazy and race if
    first touched from several worker threads, so warm them here at startup.
    """
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

    from nltk.corpus import wordnet
    from nltk.tokenize import word_tokenize

    _safe_call("wordnet-warmup", wordnet.ensure_loaded)
    _safe_call("punkt-warmup", word_tokenize, "warm up the punkt tokenizer")


def _compute_rouge(response: str, reference: str) -> Dict[str, float]:
    scores = _safe_call("rouge", _rouge_scorer().score, reference, response)
    if scores is None:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
    return {k: v.fmeasure for k, v in scores.items()}


@lru_cache(maxsize=1)
def _bleu_tokenizer():
    """sacrebleu's 13a tokenizer — the one HuggingFace ``evaluate``'s BLEU vendors."""
    from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a

    return Tokenizer13a()


def _ngram_counts(tokens: List[str], max_order: int) -> "Counter[Tuple[str, ...]]":
    counts: "Counter[Tuple[str, ...]]" = Counter()
    for order in range(1, max_order + 1):
        for i in range(len(tokens) - order + 1):
            counts[tuple(tokens[i : i + order])] += 1
    return counts


def _bleu_score(response: str, reference: str) -> float:
    """Unsmoothed 4-gram BLEU for one pair — see :func:`_compute_bleu`."""
    tokenizer = _bleu_tokenizer()
    hypothesis = tokenizer(response).split()
    ref_tokens = tokenizer(reference).split()
    if not hypothesis or not ref_tokens:
        return 0.0

    matches_by_order = [0] * _BLEU_MAX_ORDER
    overlap = _ngram_counts(hypothesis, _BLEU_MAX_ORDER) & _ngram_counts(ref_tokens, _BLEU_MAX_ORDER)
    for ngram, count in overlap.items():
        matches_by_order[len(ngram) - 1] += count

    precisions: List[float] = []
    for order in range(1, _BLEU_MAX_ORDER + 1):
        possible = len(hypothesis) - order + 1
        precisions.append(matches_by_order[order - 1] / possible if possible > 0 else 0.0)

    # Unsmoothed: a single empty n-gram order zeroes the whole score.
    if min(precisions) <= 0.0:
        return 0.0

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / _BLEU_MAX_ORDER)
    ratio = len(hypothesis) / len(ref_tokens)
    brevity_penalty = 1.0 if ratio > 1.0 else math.exp(1.0 - 1.0 / ratio)
    return geo_mean * brevity_penalty


def _compute_bleu(response: str, reference: str) -> float:
    """BLEU for one (response, reference) pair, 0-1.

    13a tokenization, ``max_order=4``, **unsmoothed**: a response without a
    matching 4-gram scores exactly 0.0, which is why RoleMRC BLEU numbers sit
    around 0.01. ``sacrebleu.sentence_bleu`` is not a substitute — it smooths,
    returns 0-100, and never emits a hard 0 — so the score is computed here
    directly, borrowing only sacrebleu's 13a tokenizer.
    """
    if not response.strip() or not reference.strip():
        return 0.0
    out = _safe_call("bleu", _bleu_score, response, reference)
    return float(out) if out is not None else 0.0


def _compute_meteor(response: str, reference: str) -> float:
    """METEOR for one pair, 0-1.

    Serialized because NLTK's WordNet reader seeks and reads one shared file
    handle per part of speech, so concurrent lookups corrupt each other and
    ``_safe_call`` turns the failures into silent zeros.
    """
    _ensure_nltk_data()
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import meteor_score

    # Tokenization is pure once punkt is loaded, so it stays outside the lock.
    reference_tokens = _safe_call("meteor-tokenize", word_tokenize, reference)
    response_tokens = _safe_call("meteor-tokenize", word_tokenize, response)
    if reference_tokens is None or response_tokens is None:
        return 0.0

    with _WORDNET_LOCK:
        out = _safe_call("meteor", meteor_score, [reference_tokens], response_tokens)
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


# ── LLM-as-judge — 5-aspect prompts ──────────────────────────────────────


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
0 - The LLM's response does not follow the nested instruction (**{content}**).\x20
1 - The LLM's response **follows** the nested instruction (**{content}**).
Score:
"""
# The `\x20` is a significant trailing space, escaped so formatters cannot strip
# it: the rendered prompt must stay byte-identical across runs.

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


# Lead-ins removed from a nested instruction before it is quoted to the judge.
# Each is deliberately written without its trailing space: the space stays in the
# rendered prompt (`** include a joke in your answer**`), and adding it here
# would change every nested-instruction prompt.
_NESTED_INSTRUCTION_LEAD = (
    "You love to",
    "You will",
    "You must",
    "You prefer to",
    "You would like to",
    "You are used to",
    "You should",
    "You are in the habit of",
)


def _build_conversation_text(messages: List[Dict[str, str]]) -> str:
    """Render the conversation as the labelled transcript the judge prompt quotes."""
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
    """The nested requirement quoted to the judge: the second ``". "``-delimited
    sentence of the system prompt, minus any lead-in and one trailing ``.``.

    Lead-ins are removed wherever they occur, not just as a prefix. A system
    prompt with no ``". "`` falls back to the full string.
    """
    sentences = system_content.split(". ")
    content = sentences[1] if len(sentences) > 1 else system_content
    for lead in _NESTED_INSTRUCTION_LEAD:
        content = content.replace(lead, "")
    if content and content[-1] == ".":
        content = content[:-1]
    return content


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


def _parse_judge_score(text: str) -> Tuple[int, bool]:
    """Drop a literal ``Score:`` and parse the remainder as a bare integer.

    Returns ``(score, is_bad)``. The verdict must be a bare integer: prose or a
    float is a bad response and scores 0. The integer is not clamped, so a judge
    that answers ``2`` contributes 2 to the aspect mean.
    """
    cleaned = text or ""
    if "Score:" in cleaned:
        cleaned = cleaned.replace("Score:", "")
    try:
        return int(cleaned), False
    except (TypeError, ValueError):
        return 0, True


# ── Aggregation: the roll-ups the RoleMRC report actually quotes ─────────

# Reference metrics that get a corpus mean in the report.
_AUTO_METRIC_KEYS: Tuple[str, ...] = (
    "rouge1",
    "rouge2",
    "rougeL",
    "rougeLsum",
    "bleu",
    "meteor",
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
)

_MULTI_TURN_ASPECT = "multi_turn_instruction"

_JUDGE_ROLLUP_KEYS: Tuple[str, ...] = (
    "judge/avg_simple_no_mt",
    "judge/avg_simple",
    "judge/avg_weighted",
)


def _judge_rollups(by_aspect: Dict[str, List[float]]) -> Dict[str, Any]:
    """RoleMRC's three published judge aggregates, from per-aspect 0/1 scores.

    The report aggregates per ASPECT, never per row, so none of these can be
    recovered from a mean of per-row rewards:

    * ``avg_simple``       — unweighted mean of the aspect means (report:
      ``AvgSimple``). Each aspect counts once regardless of how many judge
      calls it fired.
    * ``avg_weighted``     — mean over every individual judge call, i.e. the
      aspect means weighted by their call counts (report: ``AvgWeighted``).
    * ``avg_simple_no_mt`` — ``avg_simple`` over the four aspects excluding
      ``multi_turn_instruction`` (report: ``AvgS(noMT)``). **This is RoleMRC's
      headline metric.**

    Caveat on ``avg_weighted``: fitting the aggregates published for 28 model
    runs recovers call counts of (knowledge 601, style 400, nested 159,
    multi-turn 400, priority 84) -- every one within 1 of this dataset's true
    counts except ``instruction_priority``, which is exactly doubled. The raw
    ``roleMRC_test.jsonl`` has 42 rows for the two ``-refused`` tasks that fire
    it (confirmed directly against the file; see ``prepare_rolemrc.py``'s
    ``_EXPECTED_TASK_COUNTS``), so those published runs counted 42 judge calls
    twice. We emit the honest count, which makes our ``avg_weighted`` run
    ~0.7 pp (up to 1.3 pp) above theirs. ``avg_simple`` and ``avg_simple_no_mt``
    are count-independent and therefore identical either way -- including the
    headline metric.
    """
    means = {a: sum(v) / len(v) for a, v in by_aspect.items() if v}
    if not means:
        return {}

    n_calls = sum(len(by_aspect[a]) for a in means)
    rollups: Dict[str, Any] = {
        "judge/n_calls": n_calls,
        "judge/avg_simple": sum(means.values()) / len(means),
        "judge/avg_weighted": sum(m * len(by_aspect[a]) for a, m in means.items()) / n_calls,
    }
    no_mt = [m for a, m in means.items() if a != _MULTI_TURN_ASPECT]
    if no_mt:
        rollups["judge/avg_simple_no_mt"] = sum(no_mt) / len(no_mt)
    return rollups


# ── Server config + request/response shapes ──────────────────────────────


class RoleMRCResourcesServerConfig(BaseResourcesServerConfig):
    """Config for the rolemrc resources server.

    Attributes:
        mode: ``reference`` for ROUGE/BLEU/METEOR/BERTScore scoring (reward =
            ROUGE-L), or ``judge`` for the 5-aspect LLM-as-judge (reward = mean
            0/1 aspect score).
        include_bertscore: Compute BERTScore in ``reference`` mode. On by
            default; downloads a roberta-large checkpoint on first use. Turn off
            for lightweight RL reward signals.
        judge_model_server: required in ``judge`` mode — the model server graded
            aspects are sent to.
        judge_api: which API surface the judge call uses — ``chat_completions``
            (default) or ``responses``. Not cosmetic: the same judge model
            scores measurably differently across the two, so comparable runs
            must all use ``chat_completions``.
        judge_chat_completion_create_params: the request body for
            ``judge_api: chat_completions`` (required in that mode).
        judge_responses_create_params: the request body for
            ``judge_api: responses`` (required in that mode).
        judge_endpoint_max_concurrency: bound on concurrent judge HTTP calls.
            None disables limiting.
    """

    name: str = "rolemrc"
    mode: Literal["reference", "judge"] = "reference"
    include_bertscore: bool = True

    judge_model_server: Optional[ModelServerRef] = None
    judge_api: Literal["chat_completions", "responses"] = "chat_completions"
    judge_chat_completion_create_params: Optional[NeMoGymChatCompletionCreateParamsNonStreaming] = None
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
            params_field = (
                "judge_chat_completion_create_params"
                if self.config.judge_api == "chat_completions"
                else "judge_responses_create_params"
            )
            if self.config.judge_model_server is None or getattr(self.config, params_field) is None:
                raise ValueError(f"rolemrc judge mode requires `judge_model_server` and `{params_field}`.")
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
        response = _strip_reasoning_for_judge(_response_text(body.response))
        task = body.task or ""
        dimension = body.dimension or _task_dimension(task)

        messages = _conversation_messages(body)
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
        text: Optional[str] = None
        for aspect_name, prompt in prompts:
            text, error = await self._call_judge(aspect_name, prompt)
            if error is not None:
                errors.append(f"{aspect_name} ({error})")
                aspect_scores[aspect_name] = 0
                bad.append(aspect_name)
                continue
            LOG.debug("RoleMRC judge[%s] raw response: %r", aspect_name, text[:300])
            score, is_bad = _parse_judge_score(text)
            aspect_scores[aspect_name] = score
            if is_bad:
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
        result = RoleMRCVerifyResponse(
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
        # A judge call error is a judge failure, not a low score from averaging unscored aspects as 0.
        if errors:
            raise JudgeError(f"judge call failed for aspect(s): {'; '.join(errors)}")
        return result

    def _judge_payload(self, prompt: str) -> Dict[str, Any]:
        """The judge request body for one prompt, in the configured dialect.

        Only parameters actually set in the config are sent, and null ones are
        dropped: an endpoint that rejects a parameter (reasoning models reject
        ``temperature`` and ``top_p``) rejects an explicit ``null`` just as hard,
        so ``param: null`` has to mean "omit it" for that to be an escape hatch.
        """
        if self.config.judge_api == "chat_completions":
            params = self.config.judge_chat_completion_create_params.model_copy(deep=True)
            params.messages = [{"role": "user", "content": prompt}]
        else:
            params = self.config.judge_responses_create_params.model_copy(deep=True)
            params.input = [NeMoGymEasyInputMessage(role="user", content=prompt)]
        payload = params.model_dump(exclude_unset=True, mode="json")
        return {k: v for k, v in payload.items() if v is not None}

    async def _call_judge(self, aspect_name: str, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """One judge call for a single aspect.

        Returns ``(text, error)``; exactly one is set. The error string is
        propagated to the caller rather than only logged, so a judge outage says
        *why* in the row's failure record instead of just naming the aspect.
        """
        if self.config.judge_api == "chat_completions":
            url_path, response_model = "/v1/chat/completions", NeMoGymChatCompletion
        else:
            url_path, response_model = "/v1/responses", NeMoGymResponse
        try:
            async with self._judge_semaphore:
                judge_response = await call_judge(
                    self.server_client,
                    server_name=self.config.judge_model_server.name,
                    url_path=url_path,
                    json=self._judge_payload(prompt),
                    response_model=response_model,
                )
        except JudgeError as exc:  # retry-by-aspect is intentional
            LOG.warning("RoleMRC judge[%s] call failed: %s", aspect_name, exc, exc_info=True)
            return None, str(exc)
        if isinstance(judge_response, NeMoGymChatCompletion):
            choices = judge_response.choices or []
            raw = (choices[0].message.content if choices else None) or ""
        else:
            raw = _response_text(judge_response)
        text = _strip_think(raw)
        if not text:
            # A reasoning judge spends `max_output_tokens` on reasoning tokens
            # first, so an exhausted budget yields a well-formed response with
            # no text. That is an infrastructure failure, not a verdict of 0 —
            # report it rather than letting the parser score the empty string.
            LOG.warning("RoleMRC judge[%s] returned empty response text", aspect_name)
            return None, "empty response text (reasoning may have consumed max_output_tokens)"
        return text, None

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

        # Corpus means for every reference metric. `mean_reward` above covers
        # ROUGE-L alone (it is the reward); the report also quotes ROUGE-1/2/Lsum,
        # BLEU, METEOR and BERTScore, so aggregate them here rather than leaving
        # them stranded on the per-row verify responses.
        for key in _AUTO_METRIC_KEYS:
            vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
            if vals:
                metrics[f"auto/{key}/mean"] = sum(vals) / len(vals)
                metrics[f"auto/{key}/count"] = len(vals)

        by_aspect: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            for k, v in r.items():
                if k.startswith("aspect_") and isinstance(v, (int, float)):
                    by_aspect[k[len("aspect_") :]].append(v)
        for asp, vals in sorted(by_aspect.items()):
            metrics[f"aspect/{asp}/mean"] = sum(vals) / len(vals)
            metrics[f"aspect/{asp}/count"] = len(vals)

        metrics.update(_judge_rollups(by_aspect))
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        # Headline first — RoleMRC's published metric is Judge AvgS (no MT).
        for k in _JUDGE_ROLLUP_KEYS + ("mean_reward", "mean/reward"):
            if k in agent_metrics:
                out[k] = agent_metrics[k]
        # Per-dimension rewards, per-aspect judge scores, and the `auto/*`
        # corpus means (all of which end in `/mean` or `/mean_reward`).
        for k, v in agent_metrics.items():
            if k.endswith("/mean_reward") or k.endswith("/mean"):
                out[k] = v
        return out


if __name__ == "__main__":
    RoleMRCResourcesServer.run_webserver()
