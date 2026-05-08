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
numb3rs, MUSAN, etc.). Dispatches per row on ``task_type`` so the server can
score whichever WER variant a benchmark needs:

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

``task_type`` defaults to the server-level config value but may be
overridden per row via the verify request body's ``task_type`` field.
"""

import re
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics


# Hallucination detection: chars/min above this rate signal repetition/hallucination.
HALLUCINATION_CHAR_RATE_THRESHOLD = 1500.0


# ──────────────────────────────────────────────────────────────────────────────
# Text-normalization helpers
# ──────────────────────────────────────────────────────────────────────────────


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_tokens(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text)


def extract_punctuation(text: str) -> List[str]:
    return [c for c in text if not c.isalnum() and not c.isspace()]


def preprocess_asr_text(text: str) -> str:
    """Whisper text normalizer + lowercase + whitespace collapse."""
    from whisper_normalizer.english import EnglishTextNormalizer

    text = text.lower()
    text = EnglishTextNormalizer()(text)
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
    import jiwer

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


def evaluate_asr(reference: str, hypothesis: str) -> Dict[str, Any]:
    """Standard ASR WER (Whisper-normalized) — no PC scoring.

    Used by benchmarks that only score standard WER (e.g. asr-leaderboard).
    Empty references are dropped (HF Open ASR Leaderboard convention).
    """
    import jiwer

    ref = preprocess_asr_text(reference)
    hyp = preprocess_asr_text(hypothesis)
    if not ref:
        return {
            "wer": None,
            "is_correct": None,
            "text": "",
            "pred_text": hyp or "",
        }
    if not hyp:
        hyp = "empty"
    wer = jiwer.wer(ref, hyp)
    return {
        "wer": wer,
        "is_correct": wer < 0.5,
        "text": ref,
        "pred_text": hyp,
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

_TASK_TYPES = Literal["ASR-PC", "ASR", "Hallucination", "ASR_LEADERBOARD"]


class ASRWithPCConfig(BaseResourcesServerConfig):
    # Default scoring task type; can be overridden per-row via task_type on the
    # request.
    task_type: _TASK_TYPES = "ASR-PC"


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


class ASRWithPCVerifyResponse(BaseVerifyResponse):
    # Allow per-reference ``wer_<suffix>`` / ``is_correct_<suffix>`` fields
    # to ride alongside the canonical schema.
    model_config = ConfigDict(extra="allow")

    text: str = ""
    pred_text: str = ""
    wer: float = 0.0
    wer_c: float = 0.0
    wer_pc: float = 0.0
    per: float = 0.0
    is_correct: bool = False
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
    )


class ASRWithPCResourcesServer(SimpleResourcesServer):
    config: ASRWithPCConfig

    async def verify(self, body: ASRWithPCVerifyRequest) -> ASRWithPCVerifyResponse:
        hypothesis = _extract_assistant_text(body.response).strip()
        reference = (body.expected_answer or "").strip()

        # Per-row override beats the server-level default.
        task_type = body.task_type or self.config.task_type

        # Extra fields ride alongside the canonical schema via
        # ``ConfigDict(extra="allow")`` and are populated only by the
        # task_types that need them.
        extra_fields: Dict[str, Any] = {}

        if task_type == "ASR-PC":
            scores = evaluate_asr_pc(reference, hypothesis)
        elif task_type == "ASR":
            scores = _asr_to_response_scores(evaluate_asr(reference, hypothesis))
        elif task_type == "ASR_LEADERBOARD":
            scores = _asr_to_response_scores(evaluate_asr(reference, hypothesis))
            request_fields = body.model_dump()
            for ref_field in body.reference_fields or []:
                if ref_field not in request_fields:
                    raise ValueError(f"ASR_LEADERBOARD: reference_fields entry {ref_field!r} not found on request")
                ref_metrics = evaluate_asr(request_fields[ref_field] or "", hypothesis)
                suffix = _suffix_for_reference_field(ref_field)
                extra_fields[f"wer_{suffix}"] = ref_metrics["wer"]
                extra_fields[f"is_correct_{suffix}"] = ref_metrics["is_correct"]
        elif task_type == "Hallucination":
            hall = evaluate_hallucination(reference, hypothesis, body.audio_duration)
            scores = _empty_score_record(
                is_correct=bool(hall["is_correct"]),
                text=hall["text"],
                pred_text=hall["pred_text"],
            )
            extra_fields["hallucination_rate"] = hall["hallucination_rate"]
            extra_fields["char_rate"] = hall["char_rate"]
        else:
            raise ValueError(
                f"Unsupported task_type: {task_type!r}. Use one of: ASR-PC, ASR, Hallucination, ASR_LEADERBOARD."
            )

        return ASRWithPCVerifyResponse(
            **body.model_dump(),
            reward=1.0 if scores["is_correct"] else 0.0,
            text=scores["text"],
            pred_text=scores["pred_text"],
            wer=scores["wer"],
            wer_c=scores["wer_c"],
            wer_pc=scores["wer_pc"],
            per=scores["per"],
            is_correct=scores["is_correct"],
            ref_pc_tok=scores["ref_pc_tok"],
            hyp_pc_tok=scores["hyp_pc_tok"],
            ref_c=scores["ref_c"],
            hyp_c=scores["hyp_c"],
            **extra_fields,
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
            "accuracy": float(r.get("is_correct", False)),
            "per": float(r.get("per", 0.0)),
            "no_answer": 0.0 if pred else 1.0,
        }
        if r.get("hallucination_rate") is not None:
            scores["hallucination_rate"] = float(r["hallucination_rate"])
        return scores

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Per-rollout pass@k + WER aggregation.

        The headline standard ``wer`` is **corpus-level** via
        ``jiwer.wer(refs, hyps)`` over the whole eval set; ``wer_c`` /
        ``wer_pc`` / ``per`` are **mean-of-per-sample**. Per-reference
        ``wer_<suffix>`` (ASR_LEADERBOARD) is computed corpus-level over
        the per-row ``text_<suffix>`` references.
        """
        import jiwer

        metrics, _, _, max_k = compute_pass_majority_metrics(
            tasks,
            score_fn=self._score_fn,
            answer_key=None,  # ASR has no extracted-answer notion
        )

        if not tasks:
            return metrics

        # ASR_LEADERBOARD per-reference suffixes (e.g. {"tn", "itn"}).
        ref_suffixes: List[str] = sorted(
            {
                key[len("wer_") :]
                for rollouts in tasks
                for r in rollouts
                for key in r
                if key.startswith("wer_") and key not in ("wer_c", "wer_pc")
            }
        )

        for k in range(1, max_k + 1):
            refs_std: List[str] = []
            hyps_std: List[str] = []
            wer_c_scores: List[float] = []
            wer_pc_scores: List[float] = []
            per_scores: List[float] = []
            ref_corpus: Dict[str, List[str]] = {s: [] for s in ref_suffixes}
            hyp_corpus: Dict[str, List[str]] = {s: [] for s in ref_suffixes}

            for rollouts in tasks:
                for r in rollouts[:k]:
                    refs_std.append(r.get("text", ""))
                    hyps_std.append(r.get("pred_text", ""))
                    if r.get("wer_c") is not None:
                        wer_c_scores.append(float(r["wer_c"]))
                    if r.get("wer_pc") is not None:
                        wer_pc_scores.append(float(r["wer_pc"]))
                    if r.get("per") is not None:
                        per_scores.append(float(r["per"]))
                    for suffix in ref_suffixes:
                        ref_val = r.get(f"text_{suffix}")
                        if not ref_val:
                            continue
                        ref_norm = preprocess_asr_text(ref_val)
                        if not ref_norm:
                            continue
                        hyp_norm = preprocess_asr_text(r.get("pred_text") or "") or "empty"
                        ref_corpus[suffix].append(ref_norm)
                        hyp_corpus[suffix].append(hyp_norm)

            if not refs_std:
                continue

            metrics[f"corpus_wer@k={k}"] = 100.0 * jiwer.wer(refs_std, hyps_std)
            if wer_c_scores:
                metrics[f"wer_c@k={k}"] = 100.0 * sum(wer_c_scores) / len(wer_c_scores)
            if wer_pc_scores:
                metrics[f"wer_pc@k={k}"] = 100.0 * sum(wer_pc_scores) / len(wer_pc_scores)
            if per_scores:
                metrics[f"per@k={k}"] = 100.0 * sum(per_scores) / len(per_scores)
            for suffix in ref_suffixes:
                if ref_corpus[suffix]:
                    metrics[f"wer_{suffix}@k={k}"] = 100.0 * jiwer.wer(ref_corpus[suffix], hyp_corpus[suffix])

        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Headline metrics: corpus WER (the parity number) + per-rollout pass@k."""
        key: Dict[str, Any] = {}

        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]

        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}"))

        # WER aggregates at the highest k: `wer` is corpus-level, `wer_c` /
        # `wer_pc` / `per` are mean-of-per-sample. Exposed under headline names.
        max_k = 0
        for k_str_key in agent_metrics:
            if k_str_key.startswith("corpus_wer@k="):
                max_k = max(max_k, int(k_str_key.split("=")[1]))
        if max_k:
            for src_key, dst_key in (
                (f"corpus_wer@k={max_k}", "wer"),
                (f"wer_c@k={max_k}", "wer_c"),
                (f"wer_pc@k={max_k}", "wer_pc"),
                (f"per@k={max_k}", "per"),
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

        return key


if __name__ == "__main__":
    ASRWithPCResourcesServer.run_webserver()
