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
"""LibriSpeech-PC resources server: deterministic ASR with punctuation+capitalization WER.

The WER computation is a direct port of
``nemo_skills/evaluation/evaluator/audio.py::evaluate_asr_pc`` and the
corpus-level aggregation in ``audio_metrics.py::AudioMetrics.get_metrics``.
Skills is the gold standard; this server reproduces its scoring exactly.
"""

import re
from typing import Any, Dict, List, Optional

import numpy as np

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Text-normalization helpers — direct port from
# nemo_skills/evaluation/evaluator/audio.py.
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

    Mirrors ``evaluate_asr_pc`` in ``nemo_skills/evaluation/evaluator/audio.py``
    with ``normalize_standard_wer=True`` and ``normalization_mode="standard"``
    (the LibriSpeech-PC defaults).

    Also returns the normalized strings used for each WER variant; the resource
    server keeps those on the verify response so corpus-level aggregation can
    re-run jiwer over the whole corpus (matching Skills' AudioMetrics).
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


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────────


class LibriSpeechPCConfig(BaseResourcesServerConfig):
    pass


class LibriSpeechPCVerifyRequest(BaseVerifyRequest):
    expected_answer: str = ""
    sample_id: Optional[str] = None
    split: Optional[str] = None


class LibriSpeechPCVerifyResponse(BaseVerifyResponse):
    text: str = ""
    pred_text: str = ""
    wer: float = 0.0
    wer_c: float = 0.0
    wer_pc: float = 0.0
    per: float = 0.0
    is_correct: bool = False
    # Normalized strings retained for corpus-level aggregation (matches Skills'
    # AudioMetrics, which keeps text/pred_text and re-runs jiwer over the corpus).
    ref_pc_tok: str = ""
    hyp_pc_tok: str = ""
    ref_c: str = ""
    hyp_c: str = ""


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


class LibriSpeechPCResourcesServer(SimpleResourcesServer):
    config: LibriSpeechPCConfig

    async def verify(self, body: LibriSpeechPCVerifyRequest) -> LibriSpeechPCVerifyResponse:
        hypothesis = _extract_assistant_text(body.response).strip()
        reference = (body.expected_answer or "").strip()

        scores = evaluate_asr_pc(reference, hypothesis)
        return LibriSpeechPCVerifyResponse(
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
        )

    # ──────────────────────────────────────────────────────────────────────
    # Aggregate metrics — mirror Skills' AudioMetrics output.
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _score_fn(r: dict) -> Dict[str, float]:
        """Per-rollout scores routed through compute_pass_majority_metrics.

        ``per`` and ``no_answer`` are sample-mean metrics, matching the way
        Skills reports them.
        """
        pred = (r.get("pred_text") or "").strip()
        return {
            "accuracy": float(r.get("is_correct", False)),
            "per": float(r.get("per", 0.0)),
            "no_answer": 0.0 if pred else 1.0,
        }

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Per-rollout pass@k + corpus-level WER aggregation matching Skills.

        Skills' ``AudioMetrics.get_metrics`` aggregates WER values by passing
        the full lists of references and hypotheses to ``jiwer.wer`` (corpus
        level), not by averaging per-sample WERs. We replicate that here by
        re-running jiwer over the rollouts at each k.
        """
        import jiwer

        metrics, _, _, max_k = compute_pass_majority_metrics(
            tasks,
            score_fn=self._score_fn,
            answer_key=None,  # ASR has no extracted-answer notion
        )

        if not tasks:
            return metrics

        for k in range(1, max_k + 1):
            refs_std: List[str] = []
            hyps_std: List[str] = []
            refs_pc: List[str] = []
            hyps_pc: List[str] = []
            refs_c: List[str] = []
            hyps_c: List[str] = []

            for rollouts in tasks:
                for r in rollouts[:k]:
                    refs_std.append(r.get("text", ""))
                    hyps_std.append(r.get("pred_text", ""))
                    refs_pc.append(r.get("ref_pc_tok", ""))
                    hyps_pc.append(r.get("hyp_pc_tok", ""))
                    refs_c.append(r.get("ref_c", ""))
                    hyps_c.append(r.get("hyp_c", ""))

            if not refs_std:
                continue

            metrics[f"corpus_wer@k={k}"] = 100.0 * jiwer.wer(refs_std, hyps_std)
            metrics[f"corpus_wer_c@k={k}"] = 100.0 * jiwer.wer(refs_c, hyps_c)
            metrics[f"corpus_wer_pc@k={k}"] = 100.0 * jiwer.wer(refs_pc, hyps_pc)

        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Headline metrics: corpus WER (the parity number) + per-rollout pass@k."""
        key: Dict[str, Any] = {}

        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]

        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}"))

        # Corpus-level WER at the highest k — the parity-target headline.
        max_k = 0
        for k_str_key in agent_metrics:
            if k_str_key.startswith("corpus_wer@k="):
                max_k = max(max_k, int(k_str_key.split("=")[1]))
        if max_k:
            for name in ("corpus_wer", "corpus_wer_c", "corpus_wer_pc"):
                src = f"{name}@k={max_k}"
                if src in agent_metrics:
                    key[src] = agent_metrics[src]

        return key


if __name__ == "__main__":
    LibriSpeechPCResourcesServer.run_webserver()
