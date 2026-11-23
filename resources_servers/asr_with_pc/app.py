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
"""ASR-with-PC resources server: deterministic WER scoring for audio benchmarks.

Generic across the audio-WER benchmark suite (LibriSpeech-PC, asr-leaderboard,
numb3rs, MUSAN, FLEURS, etc.). Dispatches per row on ``task_type`` so the
server can score whichever WER variant a benchmark needs:

  * ``ASR-PC`` (default): full WER + WER_C + WER_PC + PER.
  * ``ASR``: standard WER only (Whisper-normalized, lowercased, no
    punctuation/capitalization).
  * ``Hallucination``: char-rate based hallucination detection (used by
    MUSAN). Requires ``audio_duration`` on the request; threshold
    1500 chars/min.
  * ``ASR_LEADERBOARD``: standard WER against ``expected_answer`` plus
    per-reference WER against each entry in the request's
    ``reference_fields`` (used by Numb3rs for dual ``text_tn`` /
    ``text_itn`` references).
  * ``OpenASR``: Hugging Face Open ASR Leaderboard English protocol. Rows that
    carry ``parent_id`` / ``chunk_index`` are chunks of one session and are
    merged into a single hypothesis at aggregation time.
  * ``Multilingual-ASR``: language-aware WER, or CER with ``use_cer``. Reads
    ``language`` (required), ``use_cer``, ``remove_diacritics``,
    ``normalize_compound``, ``merge_compounds`` and ``cer_whitespace_mode``
    from the request.

The utterance-WER task types (``ASR-PC``, ``ASR``, ``Hallucination`` and ``ASR_LEADERBOARD``) score each row
on its own. ``OpenASR`` and ``Multilingual-ASR`` report additive edit counts
(``*_errors`` / ``*_ref_words``) so ``compute_metrics`` yields corpus-level rates, plus per-profile / subset /
language breakdowns, instead of averaging utterance error rates. ``ASR`` and ``ASR_LEADERBOARD`` rows carry
edit counts too, so they pool the same way.

``task_type`` defaults to the server-level config value but may be
overridden per row via the verify request body's ``task_type`` field.
"""

import asyncio
import re
from functools import partial
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Literal, Optional

import jiwer
import numpy as np
from fastapi import HTTPException
from pydantic import ConfigDict
from whisper_normalizer.english import EnglishTextNormalizer

from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.metrics import compute_aggregate_metrics, compute_pass_majority_metrics, highest_k_metrics
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import VerifierFixture
from resources_servers.asr_with_pc.numb3rs_normalizer import normalize_numb3rs_text
from resources_servers.asr_with_pc.speech_metrics import (
    CerWhitespaceMode,
    _wer_counts,
    base_language,
    cleanup_generation_loops,
    evaluate_multilingual_asr,
    evaluate_open_asr,
    normalize_whitespace,
)
from resources_servers.asr_with_pc.verifier_fixture import create_server


# Hallucination detection: chars/min above this rate signal repetition/hallucination.
HALLUCINATION_CHAR_RATE_THRESHOLD = 1500.0


# ──────────────────────────────────────────────────────────────────────────────
# Text-normalization helpers
# ──────────────────────────────────────────────────────────────────────────────


def split_tokens(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text)


def extract_punctuation(text: str) -> List[str]:
    return [c for c in text if not c.isalnum() and not c.isspace()]


_ENGLISH_NORMALIZER = EnglishTextNormalizer()


def preprocess_asr_text(text: str) -> str:
    """Whisper text normalizer + lowercase + whitespace collapse."""
    text = text.lower()
    text = _ENGLISH_NORMALIZER(text)
    return re.sub(r"\s+", " ", text).strip()


def calculate_per(reference: str, hypothesis: str) -> float:
    """Punctuation Error Rate via DP alignment of punctuation tokens."""
    ref_punct = extract_punctuation(reference)
    hyp_punct = extract_punctuation(hypothesis)

    len_r, len_h = len(ref_punct), len(hyp_punct)
    if len_r == 0 and len_h == 0:
        return 0.0

    dp = np.zeros((len_r + 1, len_h + 1, 4), dtype=int)
    for i in range(1, len_r + 1):
        dp[i, 0][2] = i
    for j in range(1, len_h + 1):
        dp[0, j][3] = j

    for i in range(1, len_r + 1):
        for j in range(1, len_h + 1):
            if ref_punct[i - 1] == hyp_punct[j - 1]:
                dp[i, j] = dp[i - 1, j - 1].copy()
                dp[i, j][0] += 1
            else:
                sub = dp[i - 1, j - 1].copy()
                sub[1] += 1
                delete = dp[i - 1, j].copy()
                delete[2] += 1
                insert = dp[i, j - 1].copy()
                insert[3] += 1
                dp[i, j] = min([sub, delete, insert], key=lambda x: x[1] + x[2] + x[3])

    correct, substitution, deletion, insertion = dp[len_r, len_h]
    total = correct + substitution + deletion + insertion
    return (substitution + deletion + insertion) / total if total > 0 else 0.0


def evaluate_asr_pc(reference: str, hypothesis: str) -> Dict[str, Any]:
    """Compute per-sample WER, WER_C, WER_PC, PER for one (reference, hypothesis) pair.

    Standard WER uses Whisper text normalization + lowercase + punctuation
    strip. WER_C is jiwer over de-punctuated whitespace-normalized text
    (case-sensitive). WER_PC tokenizes punctuation as separate tokens so word
    boundaries and punctuation errors both contribute. PER is the punctuation
    error rate via DP alignment of punctuation tokens.

    Also returns the normalized strings used for each WER variant; the
    resource server keeps those on the verify response so corpus-level
    aggregation can re-run jiwer over the whole corpus.
    """
    ref_pc = normalize_whitespace(reference)
    hyp_pc = normalize_whitespace(hypothesis)
    ref_pc_tok = " ".join(split_tokens(ref_pc))
    hyp_pc_tok = " ".join(split_tokens(hyp_pc))
    wer_pc = jiwer.wer(ref_pc_tok, hyp_pc_tok)

    ref_c = normalize_whitespace(re.sub(r"[^\w\s]", "", reference))
    hyp_c = normalize_whitespace(re.sub(r"[^\w\s]", "", hypothesis))
    wer_c = jiwer.wer(ref_c, hyp_c)

    ref_std = preprocess_asr_text(reference)
    hyp_std = preprocess_asr_text(hypothesis)
    wer_std = jiwer.wer(ref_std, hyp_std)

    return {
        "wer": wer_std,
        "wer_c": wer_c,
        "wer_pc": wer_pc,
        "per": calculate_per(reference, hypothesis),
        "is_correct": wer_pc < 0.5,
        "text": ref_std,
        "pred_text": hyp_std,
        "ref_pc_tok": ref_pc_tok,
        "hyp_pc_tok": hyp_pc_tok,
        "ref_c": ref_c,
        "hyp_c": hyp_c,
    }


def row_normalizer(row: Dict[str, Any]) -> Callable[[str], str]:
    """The text normalizer a row asks for: Whisper's by default, or the category-based Numb3rs one."""
    if row.get("normalization") == "numb3rs":
        return partial(normalize_numb3rs_text, category=row["category"])
    return preprocess_asr_text


def evaluate_asr(
    reference: str, hypothesis: str, normalize: Callable[[str], str] = preprocess_asr_text
) -> Dict[str, Any]:
    """Standard ASR WER (Whisper-normalized unless ``normalize`` is given) — no PC scoring.

    Used by benchmarks that only score standard WER (e.g. asr-leaderboard).
    Empty references are dropped (HF Open ASR Leaderboard convention).
    """
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return {
            "wer": None,
            "is_correct": None,
            "text": "",
            "pred_text": hyp or "",
        }
    if not hyp:
        hyp = "empty"
    # Word-error counts let compute_metrics report corpus WER per profile and subset.
    counts = _wer_counts(ref, hyp)
    return {
        "wer": counts["wer"],
        "is_correct": counts["wer"] < 0.5,
        "text": ref,
        "pred_text": hyp,
        **{key: value for key, value in counts.items() if key != "wer"},
    }


def evaluate_hallucination(reference: str, hypothesis: str, audio_duration: Optional[float]) -> Dict[str, Any]:
    """Detect potential hallucinations via speaking-rate anomaly.

    Normal speech is ~600-900 chars/min; rates above
    ``HALLUCINATION_CHAR_RATE_THRESHOLD`` (1500 chars/min) indicate
    repetition/hallucination. Without ``audio_duration`` the metric is
    undefined, so we return ``is_correct=True`` with an ``error`` flag
    rather than scoring the row.
    """
    if not audio_duration or audio_duration <= 0:
        return {
            "hallucination_rate": 0.0,
            "char_rate": 0.0,
            "is_correct": True,
            "error": "missing_audio_duration",
            "text": reference,
            "pred_text": hypothesis,
        }

    char_rate = (len(hypothesis) / audio_duration) * 60.0
    is_hallucinating = char_rate > HALLUCINATION_CHAR_RATE_THRESHOLD
    return {
        "hallucination_rate": 1.0 if is_hallucinating else 0.0,
        "char_rate": round(char_rate, 2),
        "is_correct": not is_hallucinating,
        "text": reference,
        "pred_text": hypothesis,
    }


def _suffix_for_reference_field(field_name: str) -> str:
    """Strip the leading ``text_`` from a reference field name (``text_tn`` → ``tn``)."""
    return field_name[len("text_") :] if field_name.startswith("text_") else field_name


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────────

_TASK_TYPES = Literal[
    "ASR-PC",
    "ASR",
    "OpenASR",
    "Multilingual-ASR",
    "Hallucination",
    "ASR_LEADERBOARD",
]


class ASRWithPCConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    # Opt-in: collapse runaway repetition in the hypothesis before scoring (the model card applies it to transcripts).
    cleanup_generation_loops: bool = False
    # Default scoring task type; can be overridden per-row via task_type on the
    # request.
    task_type: _TASK_TYPES = "ASR-PC"
    # Rows whose ``benchmark_profile`` is listed here are averaged the way the multilingual Open ASR leaderboard
    # does (corpus WER per dataset, then per language, then over languages) into ``multilingual_macro_wer``.
    multilingual_macro_profiles: List[str] = []


class ASRWithPCVerifyRequest(BaseVerifyRequest):
    # Allow benchmark-specific reference fields (e.g. ``text_tn``,
    # ``text_itn`` for ASR_LEADERBOARD) to ride on the request.
    model_config = ConfigDict(extra="allow")

    expected_answer: str = ""
    sample_id: Optional[str] = None
    split: Optional[str] = None
    task_type: Optional[_TASK_TYPES] = None
    audio_duration: Optional[float] = None
    reference_fields: Optional[List[str]] = None
    # ASR_LEADERBOARD: "numb3rs" scores with the category-based Numb3rs normalizer (needs a ``category``).
    normalization: Optional[Literal["numb3rs"]] = None
    category: Optional[str] = None
    benchmark_profile: Optional[str] = None
    # OpenASR: chunks of one session share ``parent_id`` and are ordered by ``chunk_index``.
    parent_id: Optional[str] = None
    chunk_index: Optional[int] = None
    # Multilingual-ASR. ``None`` selects the default: diacritics kept (as upstream), compounds
    # normalized unless the language is English, compounds not merged.
    language: Optional[str] = None
    use_cer: bool = False
    remove_diacritics: Optional[bool] = None
    normalize_compound: Optional[bool] = None
    merge_compounds: Optional[bool] = None
    cer_whitespace_mode: CerWhitespaceMode = "keep"


# Scoring options of a row. They configure verify() and are not results, so they stay out of the response
# (otherwise the reward profiler would publish ``mean/use_cer``-style metrics for every benchmark).
_SCORING_OPTION_FIELDS = frozenset(
    {"use_cer", "remove_diacritics", "normalize_compound", "merge_compounds", "cer_whitespace_mode"}
)


class ASRWithPCVerifyResponse(BaseVerifyResponse):
    # Allow per-reference ``wer_<suffix>`` / ``is_correct_<suffix>`` fields
    # to ride alongside the canonical schema.
    model_config = ConfigDict(extra="allow")

    text: str = ""
    pred_text: str = ""
    # The model output as extracted from the response, before the optional loop guardrail and any normalization.
    raw_pred_text: str = ""
    # ``None`` marks an empty reference (the row is excluded from corpus rates).
    wer: Optional[float] = 0.0
    wer_c: Optional[float] = 0.0
    wer_pc: Optional[float] = 0.0
    per: Optional[float] = 0.0
    is_correct: Optional[bool] = False
    # Additive edit counts of the edit-count task types (OpenASR, Multilingual-ASR); ``cer_*`` follows the same
    # pattern.
    wer_errors: Optional[int] = None
    wer_ref_words: Optional[int] = None
    wer_substitutions: Optional[int] = None
    wer_insertions: Optional[int] = None
    wer_deletions: Optional[int] = None
    cer: Optional[float] = None
    # Whether ``wer_*`` counts words or characters (``use_cer``); ``None`` for the word-scored task types.
    error_unit: Optional[Literal["word", "char"]] = None
    # Normalized strings retained for corpus-level aggregation in compute_metrics().
    ref_pc_tok: str = ""
    hyp_pc_tok: str = ""
    ref_c: str = ""
    hyp_c: str = ""
    hallucination_rate: Optional[float] = None
    char_rate: Optional[float] = None


# ──────────────────────────────────────────────────────────────────────────────
# Server
# ──────────────────────────────────────────────────────────────────────────────


def _extract_assistant_text(response) -> str:
    """Concatenate all assistant ``output_text`` parts in the Responses-API output."""
    parts: List[str] = []
    for output_item in response.output:
        if output_item.type != "message":
            continue
        for content_item in output_item.content:
            if content_item.type != "output_text":
                continue
            parts.append(content_item.text)
    return "".join(parts)


def _empty_score_record(
    is_correct: bool = False,
    text: str = "",
    pred_text: str = "",
    wer: float = 0.0,
) -> Dict[str, Any]:
    """Score record with PC-variant fields zeroed; used by non-PC task_types."""
    return {
        "wer": wer,
        "wer_c": 0.0,
        "wer_pc": 0.0,
        "per": 0.0,
        "is_correct": is_correct,
        "text": text,
        "pred_text": pred_text,
        "ref_pc_tok": "",
        "hyp_pc_tok": "",
        "ref_c": "",
        "hyp_c": "",
    }


# The PC variants do not apply to the edit-count task types: leaving them unset keeps
# them out of the per-sample means and corpus aggregates.
_PC_NOT_APPLICABLE = {"wer_c": None, "wer_pc": None, "per": None}


def _asr_to_response_scores(asr: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt ``evaluate_asr``'s output to the canonical response score dict.

    ``evaluate_asr`` returns ``wer=None`` for empty references; we fall back
    to ``0.0`` since the response schema is non-nullable, and the aggregator
    already drops empty-text rollouts.
    """
    return _empty_score_record(
        is_correct=bool(asr["is_correct"]),
        text=asr["text"],
        pred_text=asr["pred_text"],
        wer=asr["wer"] or 0.0,
    ) | {key: value for key, value in asr.items() if key.startswith("wer_")}


class ASRWithPCResourcesServer(SimpleResourcesServer):
    config: ASRWithPCConfig

    async def verify(self, body: ASRWithPCVerifyRequest) -> ASRWithPCVerifyResponse:
        raw_hypothesis = _extract_assistant_text(body.response).strip()
        hypothesis = raw_hypothesis
        if self.config.cleanup_generation_loops:
            # The block rule is a backtracking regex: seconds of CPU on long outputs, so keep it off the event loop.
            hypothesis = await asyncio.to_thread(cleanup_generation_loops, raw_hypothesis)
        reference = (body.expected_answer or "").strip()

        # Per-row override beats the server-level default.
        task_type = body.task_type or self.config.task_type

        # Malformed rows are deterministic input errors: answer 422 so they are not retried as infrastructure.
        if body.parent_id is not None and (body.chunk_index is None or body.chunk_index < 0):
            raise HTTPException(422, "parent_id requires a non-negative chunk_index")

        # Extra fields ride alongside the canonical schema via
        # ``ConfigDict(extra="allow")`` and are populated only by the
        # task_types that need them.
        extra_fields: Dict[str, Any] = {}

        if task_type == "ASR-PC":
            scores = evaluate_asr_pc(reference, hypothesis)
        elif task_type == "ASR":
            scores = _asr_to_response_scores(evaluate_asr(reference, hypothesis))
        elif task_type == "OpenASR":
            scores = {**_PC_NOT_APPLICABLE, **evaluate_open_asr(reference, hypothesis)}
        elif task_type == "Multilingual-ASR":
            if not body.language:
                raise HTTPException(422, "Multilingual-ASR rows require a language")
            scores = {
                **_PC_NOT_APPLICABLE,
                **evaluate_multilingual_asr(
                    reference,
                    hypothesis,
                    language=body.language,
                    use_cer=body.use_cer,
                    remove_diacritics=bool(body.remove_diacritics),
                    normalize_compound=(
                        base_language(body.language) != "en"
                        if body.normalize_compound is None
                        else body.normalize_compound
                    ),
                    merge_compounds=bool(body.merge_compounds),
                    cer_whitespace_mode=body.cer_whitespace_mode,
                ),
            }
        elif task_type == "ASR_LEADERBOARD":
            if body.normalization == "numb3rs" and not body.category:
                raise HTTPException(422, "normalization 'numb3rs' requires a category")
            request_fields = body.model_dump()
            normalize = row_normalizer(request_fields)
            scores = _asr_to_response_scores(evaluate_asr(reference, hypothesis, normalize))
            for ref_field in body.reference_fields or []:
                if ref_field not in request_fields:
                    raise HTTPException(
                        422, f"ASR_LEADERBOARD: reference_fields entry {ref_field!r} not found on request"
                    )
                ref_metrics = evaluate_asr(request_fields[ref_field] or "", hypothesis, normalize)
                suffix = _suffix_for_reference_field(ref_field)
                extra_fields[f"wer_{suffix}"] = ref_metrics["wer"]
                extra_fields[f"is_correct_{suffix}"] = ref_metrics["is_correct"]
        else:  # "Hallucination"; the Literal type rejects any other value.
            hall = evaluate_hallucination(reference, hypothesis, body.audio_duration)
            scores = _empty_score_record(
                is_correct=bool(hall["is_correct"]),
                text=hall["text"],
                pred_text=hall["pred_text"],
            )
            extra_fields["hallucination_rate"] = hall["hallucination_rate"]
            extra_fields["char_rate"] = hall["char_rate"]

        # Score fields missing from the evaluator keep the empty-record defaults.
        scores = {**_empty_score_record(), **scores, **extra_fields}
        return ASRWithPCVerifyResponse(
            **{
                **body.model_dump(exclude=_SCORING_OPTION_FIELDS),
                **scores,
                # Per-row reward of a chunk row compares the chunk with the whole session reference, so it is not
                # meaningful for chunked OpenASR rows; their metrics come from the merged session at aggregation.
                "reward": 1.0 if scores["is_correct"] else 0.0,
                "raw_pred_text": raw_hypothesis,
            }
        )

    # ──────────────────────────────────────────────────────────────────────
    # Aggregate metrics
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _score_fn(r: dict) -> Dict[str, float]:
        """Per-rollout scores routed through ``compute_pass_majority_metrics``.

        ``per``, ``hallucination_rate`` and ``no_answer`` are sample-mean
        metrics. Hallucination only contributes when the rollout dict carries
        the field (i.e. it came from a Hallucination task_type).
        """
        pred = (r.get("pred_text") or "").strip()
        scores = {
            "accuracy": float(bool(r.get("is_correct", False))),
            "no_answer": 0.0 if pred else 1.0,
        }
        if r.get("per") is not None:
            scores["per"] = float(r["per"])
        if r.get("hallucination_rate") is not None:
            scores["hallucination_rate"] = float(r["hallucination_rate"])
        return scores

    def _row_task_type(self, row: Dict[str, Any]) -> str:
        return row.get("task_type") or self.config.task_type

    def _chunk_hypothesis(self, row: Dict[str, Any]) -> str:
        text = str(row.get("raw_pred_text") or row.get("pred_text") or "").strip()
        return cleanup_generation_loops(text) if self.config.cleanup_generation_loops else text

    def _merge_open_asr_chunks(self, tasks: List[List[Dict[str, Any]]]) -> tuple[List[List[Dict[str, Any]]], int]:
        """Collapse chunked ``OpenASR`` rows to one parent-session task.

        Earnings22-AA references are session transcripts while inference runs
        on VAD chunks. Concatenating raw hypotheses in ``chunk_index`` order
        before normalization mirrors the leaderboard protocol. Repeats are paired by ``_ng_rollout_index``
        (list position when a row has none). A session attempt that lacks a chunk (a masked rollout, a gap in
        the chunk indices, or a repeated index) is left out and counted, so one lost rollout does not fail the
        whole aggregation. A session whose last chunks are all missing cannot be detected.
        Returns the merged tasks and the number of session attempts left out.
        """
        regular_tasks: List[List[Dict[str, Any]]] = []
        grouped: Dict[tuple[str, str], List[List[Dict[str, Any]]]] = {}
        for rollouts in tasks:
            if not rollouts or rollouts[0].get("parent_id") is None or self._row_task_type(rollouts[0]) != "OpenASR":
                regular_tasks.append(rollouts)
                continue
            parent_ids = {str(rollout.get("parent_id")) for rollout in rollouts}
            chunk_indices = {rollout.get("chunk_index") for rollout in rollouts}
            subsets = {str(rollout.get("subset_for_metrics", "_all_")) for rollout in rollouts}
            if len(parent_ids) != 1 or len(chunk_indices) != 1 or len(subsets) != 1:
                raise ValueError("Open ASR repeats for one chunk must share parent_id, chunk_index, and subset")
            grouped.setdefault((subsets.pop(), parent_ids.pop()), []).append(rollouts)

        incomplete = 0
        for (subset, parent_id), chunk_tasks in sorted(grouped.items()):
            indices = [task[0].get("chunk_index") for task in chunk_tasks]
            attempts: Dict[int, Dict[int, Dict[str, Any]]] = {}
            for position_task in chunk_tasks:
                for position, row in enumerate(position_task):
                    attempt = int(row.get("_ng_rollout_index", position))
                    attempts.setdefault(attempt, {})[row.get("chunk_index")] = row
            if None in indices or sorted(indices) != list(range(len(indices))):
                incomplete += len(attempts)
                continue
            merged_rollouts: List[Dict[str, Any]] = []
            for attempt in sorted(attempts):
                by_chunk = attempts[attempt]
                if len(by_chunk) != len(indices):
                    incomplete += 1
                    continue
                chunks = [by_chunk[index] for index in sorted(indices)]
                references = {str(chunk.get("expected_answer") or "").strip() for chunk in chunks}
                if len(references) != 1:
                    raise ValueError(f"Inconsistent parent references across Open ASR chunks for {parent_id}")
                hypothesis = " ".join(text for text in (self._chunk_hypothesis(chunk) for chunk in chunks) if text)
                merged = dict(chunks[0])
                scores = evaluate_open_asr(references.pop(), hypothesis)
                merged.update(scores)
                merged.update(
                    parent_id=parent_id,
                    sample_id=parent_id,
                    subset_for_metrics=subset,
                    raw_pred_text=hypothesis,
                    reward=1.0 if scores["is_correct"] else 0.0,
                )
                merged_rollouts.append(merged)
            if merged_rollouts:
                regular_tasks.append(merged_rollouts)
        return regular_tasks, incomplete

    @staticmethod
    def _corpus_rate(rows: List[Dict[str, Any]], prefix: str) -> Optional[float]:
        errors = sum(int(row.get(f"{prefix}_errors") or 0) for row in rows)
        references = sum(int(row.get(f"{prefix}_ref_words") or 0) for row in rows)
        return None if references == 0 else 100.0 * errors / references

    @staticmethod
    def _pooled_word_rate(rows: List[Dict[str, Any]]) -> Optional[float]:
        """Corpus WER (percent) over the word-scored rows.

        Rows with edit counts pool them; rows without counts (``ASR-PC``, ``Hallucination``) get counts from
        their normalized ``text`` / ``pred_text``, so a set that mixes task types is pooled as one corpus.
        Character-scored rows (``use_cer``) are left out: their counts are characters, not words. A row whose
        reference normalizes to empty has no counts and is left out too.
        """
        errors = references = 0
        for row in rows:
            if row.get("error_unit") == "char":
                continue
            if row.get("wer_ref_words") is not None:
                errors += int(row.get("wer_errors") or 0)
                references += int(row["wer_ref_words"])
            elif row.get("text"):
                counts = _wer_counts(row["text"], row.get("pred_text") or "")
                errors += counts["wer_errors"]
                references += counts["wer_ref_words"]
        return None if references == 0 else 100.0 * errors / references

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Corpus-level rates (percent) at every k, plus per-profile/subset/language breakdowns.

        ``corpus_wer`` pools word edits over the word-scored rows (see ``_pooled_word_rate``); ``cer`` pools the
        character counts of the rows that report them. ``wer_c`` / ``wer_pc`` / ``per`` / ``hallucination_rate``
        are means over samples, and ``wer_<suffix>`` (one per ``reference_fields`` entry) is corpus-level over
        that reference. ``mean_utterance_wer`` / ``mean_utterance_cer`` average the per-utterance rates of the
        rows that report edit counts (after chunk merging). Breakdowns are keyed by group kind:
        ``wer::profile::<x>``, ``wer::subset::<x>`` and ``wer::language::<x>`` (likewise ``cer::``); a group's
        rate is in the unit its rows are scored in.
        """
        tasks, incomplete_sessions = self._merge_open_asr_chunks(tasks)
        metrics, _, _, max_k = compute_pass_majority_metrics(
            tasks,
            score_fn=self._score_fn,
            answer_key=None,  # ASR has no extracted-answer notion
        )
        if incomplete_sessions:
            metrics["open_asr_incomplete_sessions"] = incomplete_sessions
        if not tasks:
            return metrics
        for k in range(1, max_k + 1):
            selected = [rollout for rollouts in tasks for rollout in rollouts[:k]]
            counted = [row for row in selected if row.get("wer_ref_words") is not None]
            cer_rows = [row for row in selected if row.get("cer_ref_words") is not None]

            corpus_wer = self._pooled_word_rate(selected)
            if corpus_wer is not None:
                metrics[f"corpus_wer@k={k}"] = corpus_wer

            cer_rate = self._corpus_rate(cer_rows, "cer")
            if cer_rate is not None:
                metrics[f"cer@k={k}"] = cer_rate

            for prefix, name, rows in (
                ("wer", "mean_utterance_wer", [row for row in counted if row.get("error_unit") != "char"]),
                ("cer", "mean_utterance_cer", cer_rows),
            ):
                rates = [float(row[prefix]) for row in rows if row.get(prefix) is not None]
                if rates:
                    metrics[f"{name}@k={k}"] = 100.0 * sum(rates) / len(rates)

            for field in ("wer_c", "wer_pc", "per", "hallucination_rate"):
                values = [float(row[field]) for row in selected if row.get(field) is not None]
                if values:
                    metrics[f"{field}@k={k}"] = 100.0 * sum(values) / len(values)

            for kind, group_field in (
                ("profile", "benchmark_profile"),
                ("subset", "subset_for_metrics"),
                ("language", "language"),
            ):
                group_names = sorted({str(row[group_field]) for row in selected if row.get(group_field)})
                for group_name in group_names:
                    group_rows = [row for row in selected if str(row.get(group_field)) == group_name]
                    for prefix in ("wer", "cer"):
                        rate = self._corpus_rate(group_rows, prefix)
                        if rate is not None:
                            metrics[f"{prefix}::{kind}::{group_name}@k={k}"] = rate

            # The multilingual Open ASR leaderboard first averages corpus WER
            # across datasets within each language, then averages the language
            # scores. This intentionally differs from pooling word errors across
            # every row in a language or across the profile. Which rows take part
            # is a server config choice (``multilingual_macro_profiles``).
            macro_profiles = set(self.config.multilingual_macro_profiles)
            multilingual_rows = [row for row in selected if row.get("benchmark_profile") in macro_profiles]
            language_wers = []
            for language in sorted({str(row.get("language")) for row in multilingual_rows if row.get("language")}):
                language_rows = [row for row in multilingual_rows if str(row.get("language")) == language]
                dataset_wers = []
                for subset in sorted(
                    {str(row.get("subset_for_metrics")) for row in language_rows if row.get("subset_for_metrics")}
                ):
                    rate = self._corpus_rate(
                        [row for row in language_rows if str(row.get("subset_for_metrics")) == subset],
                        "wer",
                    )
                    if rate is not None:
                        dataset_wers.append(rate)
                if dataset_wers:
                    language_wer = sum(dataset_wers) / len(dataset_wers)
                    metrics[f"macro_wer::language::{language}@k={k}"] = language_wer
                    language_wers.append(language_wer)
            if language_wers:
                multilingual_macro_wer = sum(language_wers) / len(language_wers)
                metrics[f"multilingual_macro_wer@k={k}"] = multilingual_macro_wer
                if len(multilingual_rows) == len(selected):
                    metrics[f"corpus_wer@k={k}"] = multilingual_macro_wer

            reference_fields = sorted(
                {field for row in selected for field in (row.get("reference_fields") or []) if row.get(field)}
            )
            for field in reference_fields:
                pairs = [
                    (
                        row_normalizer(row)(row[field]),
                        row_normalizer(row)(row.get("pred_text") or "") or "empty",
                    )
                    for row in selected
                    if row.get(field)
                ]
                pairs = [(ref, hyp) for ref, hyp in pairs if ref]
                if pairs:
                    metrics[f"wer_{_suffix_for_reference_field(field)}@k={k}"] = 100.0 * jiwer.wer(
                        [ref for ref, _ in pairs], [hyp for _, hyp in pairs]
                    )

        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Headline metrics: corpus WER (the parity number) + per-rollout pass@k."""
        key: Dict[str, Any] = {}

        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]

        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}"))
        if "open_asr_incomplete_sessions" in agent_metrics:
            key["open_asr_incomplete_sessions"] = agent_metrics["open_asr_incomplete_sessions"]

        # WER aggregates at the highest k: `wer` is corpus-level, `wer_c` /
        # `wer_pc` / `per` are mean-of-per-sample. Exposed under headline names.
        max_k = 0
        for k_str_key in agent_metrics:
            if "@k=" not in k_str_key:
                continue
            try:
                max_k = max(max_k, int(k_str_key.rsplit("@k=", 1)[1]))
            except ValueError:
                continue
        if max_k:
            for src_key, dst_key in (
                (f"corpus_wer@k={max_k}", "wer"),
                (f"wer_c@k={max_k}", "wer_c"),
                (f"wer_pc@k={max_k}", "wer_pc"),
                (f"per@k={max_k}", "per"),
                (f"cer@k={max_k}", "cer"),
                (f"mean_utterance_wer@k={max_k}", "mean_utterance_wer"),
                (f"mean_utterance_cer@k={max_k}", "mean_utterance_cer"),
                (f"multilingual_macro_wer@k={max_k}", "multilingual_macro_wer"),
            ):
                if src_key in agent_metrics:
                    key[dst_key] = agent_metrics[src_key]
            # Per-reference WERs from ASR_LEADERBOARD: expose ``wer_<suffix>``
            # at the highest k; canonical wer_c/wer_pc were already covered.
            for k_str_key in agent_metrics:
                if not k_str_key.endswith(f"@k={max_k}") or not k_str_key.startswith("wer_"):
                    continue
                suffix = k_str_key[len("wer_") : k_str_key.index("@k=")]
                if suffix in ("c", "pc"):
                    continue
                key[f"wer_{suffix}"] = agent_metrics[k_str_key]

            for metric_name, value in agent_metrics.items():
                if "::" not in metric_name or not metric_name.endswith(f"@k={max_k}"):
                    continue
                key[metric_name[: -len(f"@k={max_k}")]] = value

        return key

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """Aggregate in a worker thread: corpus WER over a long-form benchmark keeps the event loop busy for minutes."""
        return await asyncio.to_thread(
            compute_aggregate_metrics,
            body.verify_responses,
            compute_metrics_fn=self.compute_metrics,
            get_key_metrics_fn=self.get_key_metrics,
            compute_repeat_metrics_fn=self.compute_repeat_metrics,
        )


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_server,
    request_model=ASRWithPCVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    ASRWithPCResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = ASRWithPCResourcesServer.run_webserver()  # noqa: F401
