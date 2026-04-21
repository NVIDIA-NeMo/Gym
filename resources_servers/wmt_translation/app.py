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
"""Generic machine-translation verifier for WMT-style benchmarks.

Reproduces NeMo-Skills' `TranslationMetrics` corpus-BLEU + xCOMET-XXL
aggregation in Gym's resource-server shape:

  * `verify()` returns a per-sample sentence-BLEU reward (useful as an RL
    signal) plus the fields `compute_metrics()` needs to recompute the
    authoritative corpus-level numbers.
  * `compute_metrics(tasks)` groups rollouts by
    ``(source_language, target_language, rollout_index)`` and calls
    ``sacrebleu.corpus_bleu`` with the language-specific tokenizer matching
    Skills (``13a`` default, ``ja-mecab``/``ko-mecab``/``zh`` as needed).
  * If ``compute_comet`` is true, the same method also fires a single
    ``@ray.remote(num_gpus=comet_num_gpus)`` task that loads the xCOMET-XXL
    checkpoint once and batch-predicts QE scores for every (src, mt, ref)
    triple. This is the first Gym resource server to pull a heavyweight
    neural eval metric onto a Ray-scheduled GPU — it's a model for how we
    expect to integrate COMET-family or reward-model metrics in future
    benchmarks.

Skills' aggregation emits per-pair BLEU/COMET plus three cross-pair
aggregates (``xx->xx``, ``<src>->xx``, ``xx->{tgt}``); this server does
the same and annotates each with ``std_dev_across_runs`` so runs are
directly comparable to Skills ``metrics.json``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import ray
from fastapi import FastAPI
from sacrebleu import corpus_bleu, sentence_bleu

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


LOG = logging.getLogger(__name__)


# --- Tokenizer selection ------------------------------------------------------
# Mirrors nemo_skills/evaluation/metrics/translation_metrics.py. The
# ``13a`` default is SacreBLEU's default; ``ja-mecab`` / ``ko-mecab`` need
# sacrebleu's [ja]/[ko] extras installed; ``zh`` is built in.
_TOKENIZER_BY_LANG_PREFIX = {
    "ja": "ja-mecab",
    "ko": "ko-mecab",
    "zh": "zh",
}


def _tokenizer_for(target_language: str) -> str:
    return _TOKENIZER_BY_LANG_PREFIX.get(target_language[:2], "13a")


# --- Thinking-preamble handling ---------------------------------------------
# Matches NeMo-Skills' parse_reasoning=True behavior. Reasoning models
# (e.g. Nemotron-3-Nano) emit a pre-reasoning preamble wrapped in
# <think>...</think>. vLLM's reasoning parser strips the opening <think>
# tag but keeps the closing </think>, so the raw response looks like
#   "We need to translate ... </think>\nProlog"
# Skills' parse_reasoning takes the text after the last </think>; if no
# closing tag exists (model never finished reasoning), Skills produces an
# empty string. We must replicate both branches or corpus BLEU is computed
# against the reasoning text, which tanks the score (~3x lower BLEU).


def _strip_reasoning_preamble(text: str) -> str:
    """Remove a pre-answer reasoning preamble, matching Skills' parse_reasoning=True.

    - If ``</think>`` is present, return everything after the *last* occurrence.
    - If absent, return an empty string (model never finished reasoning).

    Both branches match ``nemo_skills`` translation evaluation behavior.
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].lstrip("\n")
    return ""


# --- Request / response shapes ------------------------------------------------


class WmtTranslationResourcesServerConfig(BaseResourcesServerConfig):
    """Config for the wmt_translation resource server.

    Attributes:
        compute_comet: Run xCOMET-XXL in ``compute_metrics``. Default True.
            Turn off for smoke tests or RL training runs where only BLEU
            is needed.
        comet_model: HuggingFace repo for the COMET checkpoint. Resolved via
            ``comet.download_model`` (cached under HF_HOME).
        comet_batch_size: Batch size passed to ``model.predict``.
        comet_num_gpus: GPUs requested for the Ray COMET actor. Default 1
            is enough for xCOMET-XXL at batch size 16; bump for larger
            benchmarks.
    """

    compute_comet: bool = True
    comet_model: str = "Unbabel/XCOMET-XXL"
    comet_batch_size: int = 16
    # num_gpus=0 lets the COMET task share GPUs with vLLM's DP engines. Ray's
    # strict_pack placement group for vLLM DP reserves num_gpus=1 per GPU, so
    # any positive request queues forever. A 0-GPU task skips Ray's GPU
    # accounting entirely; the task still runs on a real GPU via CUDA_VISIBLE_
    # DEVICES=<all> — we just co-tenant with vLLM. Enforced non-CPU via an
    # assert at task entry so a missing GPU fails loud instead of loading
    # xCOMET-XXL on CPU (~hours).
    comet_num_gpus: int = 0
    # When True, strip the reasoning preamble before computing BLEU/COMET, matching
    # NeMo-Skills' parse_reasoning=True. Required for reasoning models that emit
    # <think>...</think> preambles (e.g. Nemotron-3-Nano); otherwise the preamble
    # is scored against the reference and collapses BLEU. Set False for plain
    # instruction-tuned models that do not emit reasoning traces.
    strip_reasoning: bool = True


class WmtTranslationRunRequest(BaseRunRequest):
    # Fields mirror Skills' wmt24pp prepare.py row shape.
    text: str
    translation: str
    source_language: str
    target_language: str
    source_lang_name: Optional[str] = None
    target_lang_name: Optional[str] = None


class WmtTranslationVerifyRequest(WmtTranslationRunRequest, BaseVerifyRequest):
    pass


class WmtTranslationVerifyResponse(WmtTranslationVerifyRequest, BaseVerifyResponse):
    # Extraction of the model's translation (currently == generation text;
    # future extensions may strip reasoning traces).
    generation: str
    # Per-sample sentence-BLEU, useful as a dense RL reward. Corpus-level
    # BLEU lives in compute_metrics() and is the parity target.
    sentence_bleu: float


# --- Ray COMET scoring --------------------------------------------------------


# Build the remote function lazily so importing this module doesn't require
# Ray to already be initialized. ``config.comet_num_gpus`` parameterises the
# GPU allocation at call time.
def _build_comet_remote(num_gpus: float):
    # runtime_env={"py_executable": sys.executable} pins the Ray worker to
    # THIS server's venv, not the Gym-head driver's. Gym's main venv doesn't
    # have unbabel-comet; the wmt_translation server's venv does (via
    # requirements.txt). Without this, the remote task runs in the head venv
    # and fails with ModuleNotFoundError on `comet` (and the head venv also
    # lacks pip, so a lazy subprocess install fails too).
    # Pattern copied from resources_servers/code_gen (lcb_integration).
    import sys

    @ray.remote(num_gpus=num_gpus, runtime_env={"py_executable": sys.executable})
    def _score_comet(triples: List[Tuple[str, str, str]], model_name: str, batch_size: int) -> List[float]:
        import torch
        from comet import download_model, load_from_checkpoint

        # Hard-assert CUDA for the POC path: num_gpus=0 skips Ray's GPU
        # accounting, so we rely on CUDA_VISIBLE_DEVICES exposing a real GPU
        # on the Ray worker's node. Without this assert, the fallback would
        # load xCOMET-XXL (10B params) on CPU and grind for hours.
        assert torch.cuda.is_available(), (
            "wmt_translation COMET task requires a CUDA device. "
            "This path runs co-tenant with vLLM's DP engines via num_gpus=0; "
            "if the Ray worker lands on a CPU-only node, fail loud."
        )

        LOG.info("Loading xCOMET model %s on cuda", model_name)
        ckpt_path = download_model(model_name)
        model = load_from_checkpoint(ckpt_path)
        model.to("cuda").eval()

        data = [{"src": s, "mt": m, "ref": r} for s, m, r in triples]
        LOG.info("Scoring %d (src, mt, ref) triples at batch_size=%d", len(data), batch_size)
        result = model.predict(data, batch_size=batch_size)
        return list(result.scores)

    return _score_comet


# --- Server -------------------------------------------------------------------


class WmtTranslationResourcesServer(SimpleResourcesServer):
    config: WmtTranslationResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    async def verify(self, body: WmtTranslationVerifyRequest) -> WmtTranslationVerifyResponse:
        """Return per-sample sentence-BLEU as the RL reward.

        The authoritative corpus-BLEU (+ optional COMET) lives in
        ``compute_metrics`` and is what parity comparisons to Skills use.
        """
        raw = body.response.output_text or ""
        # Match Skills' parse_reasoning=True: drop the reasoning preamble
        # before scoring so BLEU is computed against the actual translation
        # only. Without this, reasoning models tank BLEU by ~3x.
        if self.config.strip_reasoning:
            raw = _strip_reasoning_preamble(raw)
        generation = raw.strip()
        if not generation:
            return WmtTranslationVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                generation="",
                sentence_bleu=0.0,
            )

        tokenize = _tokenizer_for(body.target_language)
        # sentence_bleu returns a BLEUScore; .score is 0-100.
        sent_score = sentence_bleu(generation, [body.translation], tokenize=tokenize).score
        # Normalize to [0, 1] so the "reward" field stays conventional.
        reward = sent_score / 100.0

        return WmtTranslationVerifyResponse(
            **body.model_dump(),
            reward=reward,
            generation=generation,
            sentence_bleu=sent_score,
        )

    # --- Aggregate metrics ---------------------------------------------------

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compute Skills-equivalent corpus BLEU + (optional) COMET metrics.

        Output keys mirror Skills' ``metrics.json`` for wmt24pp:

          <src>-><tgt>/bleu                 (mean across rollouts)
          <src>-><tgt>/bleu_std_dev_across_runs
          <src>-><tgt>/comet                (mean across rollouts)
          <src>-><tgt>/comet_std_dev_across_runs
          <src>->xx/bleu  xx->xx/bleu  xx-><tgt>/bleu   (aggregations)
          ... same with /comet
        """
        if not tasks:
            return {}

        # 1. Bucket rollouts by (src, tgt) and by rollout index within task.
        #    Skills computes per-seed corpus_bleu, then averages; we replicate
        #    by treating rollout index == seed index. The max k rollouts per
        #    task defines the number of "runs".
        max_k = max(len(rollouts) for rollouts in tasks)

        # per_pair_runs[(src, tgt)][k] = list of (mt, ref) across all tasks
        #                                for rollout index k
        per_pair_runs: Dict[Tuple[str, str], List[List[Tuple[str, str]]]] = defaultdict(
            lambda: [list() for _ in range(max_k)]
        )
        # Flat triples for COMET (src_text, mt, ref) -> per-pair-per-rollout
        # mapping so we can reassemble COMET scores after Ray returns.
        comet_triples: List[Tuple[str, str, str]] = []
        # comet_slots[i] = (pair_key, rollout_index) — aligned with comet_triples
        comet_slots: List[Tuple[Tuple[str, str], int]] = []

        for task_rollouts in tasks:
            for k, rollout in enumerate(task_rollouts):
                src = rollout.get("source_language")
                tgt = rollout.get("target_language")
                if not src or not tgt:
                    continue
                ref = rollout.get("translation") or ""
                mt = rollout.get("generation") or ""
                src_text = rollout.get("text") or ""
                per_pair_runs[(src, tgt)][k].append((mt, ref))
                if self.config.compute_comet:
                    comet_triples.append((src_text, mt, ref))
                    comet_slots.append(((src, tgt), k))

        # 2. Per-(src, tgt) corpus BLEU per rollout.
        bleu_per_pair: Dict[Tuple[str, str], List[float]] = {}
        for (src, tgt), runs in per_pair_runs.items():
            tokenize = _tokenizer_for(tgt)
            per_run = []
            for run in runs:
                if not run:
                    continue
                preds = [mt for mt, _ in run]
                refs = [ref for _, ref in run]
                per_run.append(corpus_bleu(preds, [refs], tokenize=tokenize).score)
            bleu_per_pair[(src, tgt)] = per_run

        # 3. Optional: run COMET on one Ray GPU actor. Single remote call
        #    so the xCOMET-XXL checkpoint is loaded exactly once.
        comet_per_pair: Dict[Tuple[str, str], List[List[float]]] = defaultdict(lambda: [list() for _ in range(max_k)])
        if self.config.compute_comet and comet_triples:
            try:
                remote_fn = _build_comet_remote(self.config.comet_num_gpus)
                LOG.info(
                    "Dispatching %d COMET triples to a Ray GPU actor (%d GPUs, batch=%d, model=%s)",
                    len(comet_triples),
                    self.config.comet_num_gpus,
                    self.config.comet_batch_size,
                    self.config.comet_model,
                )
                scores_future = remote_fn.remote(
                    comet_triples,
                    self.config.comet_model,
                    self.config.comet_batch_size,
                )
                scores: List[float] = ray.get(scores_future)
                for score, (pair_key, k) in zip(scores, comet_slots):
                    comet_per_pair[pair_key][k].append(score)
            except Exception as e:  # don't let COMET failure kill BLEU reporting
                LOG.exception("COMET scoring failed, continuing with BLEU only: %s", e)
                comet_per_pair.clear()

        # Convert COMET per-rollout buckets into mean comet per (pair, rollout).
        # (Skills averages per-sample comet scores per seed, then averages across
        # seeds; we replicate.)
        comet_mean_per_pair: Dict[Tuple[str, str], List[float]] = {}
        for pair_key, per_run in comet_per_pair.items():
            means = []
            for run_scores in per_run:
                if run_scores:
                    means.append(100.0 * sum(run_scores) / len(run_scores))
            comet_mean_per_pair[pair_key] = means

        # 4. Build output dict with Skills-style keys + aggregations.
        metrics: Dict[str, Any] = {}
        all_pairs = sorted(per_pair_runs.keys())

        def _mean_std(values: List[float]) -> Tuple[float, float]:
            if not values:
                return (0.0, 0.0)
            n = len(values)
            mean = sum(values) / n
            if n < 2:
                return (mean, 0.0)
            var = sum((v - mean) ** 2 for v in values) / n  # population std, matches Skills np.std default
            return (mean, var**0.5)

        # Per-pair
        for src, tgt in all_pairs:
            pair_label = f"{src}->{tgt}"
            bleu_runs = bleu_per_pair.get((src, tgt), [])
            m, s = _mean_std(bleu_runs)
            metrics[f"{pair_label}/bleu"] = m
            metrics[f"{pair_label}/bleu_std_dev_across_runs"] = s

            if self.config.compute_comet:
                comet_runs = comet_mean_per_pair.get((src, tgt), [])
                if comet_runs:
                    cm, cs = _mean_std(comet_runs)
                    metrics[f"{pair_label}/comet"] = cm
                    metrics[f"{pair_label}/comet_std_dev_across_runs"] = cs

        # Aggregations: xx->xx, <src>->xx, xx->{tgt}. Skills averages per-run
        # BLEU across the contributing pairs (per-run mean of per-pair BLEU),
        # then averages across runs.
        def _aggregate(pair_filter) -> Dict[str, List[float]]:
            """Return per-run aggregated BLEU/COMET across filtered pairs."""
            filtered_pairs = [p for p in all_pairs if pair_filter(p)]
            if not filtered_pairs:
                return {"bleu": [], "comet": []}
            # Align rollout-index across pairs: take the min number of rollouts
            # present across the pairs so we don't average over missing runs.
            min_runs = min(len(bleu_per_pair.get(p, [])) for p in filtered_pairs)
            bleu_runs = []
            for k in range(min_runs):
                per_pair_k = [bleu_per_pair[p][k] for p in filtered_pairs if k < len(bleu_per_pair[p])]
                if per_pair_k:
                    bleu_runs.append(sum(per_pair_k) / len(per_pair_k))
            comet_runs: List[float] = []
            if self.config.compute_comet:
                comet_min = min(
                    (len(comet_mean_per_pair.get(p, [])) for p in filtered_pairs),
                    default=0,
                )
                for k in range(comet_min):
                    per_pair_k = [
                        comet_mean_per_pair[p][k] for p in filtered_pairs if k < len(comet_mean_per_pair.get(p, []))
                    ]
                    if per_pair_k:
                        comet_runs.append(sum(per_pair_k) / len(per_pair_k))
            return {"bleu": bleu_runs, "comet": comet_runs}

        src_langs = sorted({p[0] for p in all_pairs})
        tgt_langs = sorted({p[1] for p in all_pairs})

        # xx->xx (global)
        agg = _aggregate(lambda p: True)
        m, s = _mean_std(agg["bleu"])
        metrics["xx->xx/bleu"] = m
        metrics["xx->xx/bleu_std_dev_across_runs"] = s
        if agg["comet"]:
            m, s = _mean_std(agg["comet"])
            metrics["xx->xx/comet"] = m
            metrics["xx->xx/comet_std_dev_across_runs"] = s

        # <src>->xx and xx-><tgt>
        for src in src_langs:
            agg = _aggregate(lambda p, _s=src: p[0] == _s)
            m, s = _mean_std(agg["bleu"])
            metrics[f"{src}->xx/bleu"] = m
            metrics[f"{src}->xx/bleu_std_dev_across_runs"] = s
            if agg["comet"]:
                m, s = _mean_std(agg["comet"])
                metrics[f"{src}->xx/comet"] = m
                metrics[f"{src}->xx/comet_std_dev_across_runs"] = s
        for tgt in tgt_langs:
            agg = _aggregate(lambda p, _t=tgt: p[1] == _t)
            m, s = _mean_std(agg["bleu"])
            metrics[f"xx->{tgt}/bleu"] = m
            metrics[f"xx->{tgt}/bleu_std_dev_across_runs"] = s
            if agg["comet"]:
                m, s = _mean_std(agg["comet"])
                metrics[f"xx->{tgt}/comet"] = m
                metrics[f"xx->{tgt}/comet_std_dev_across_runs"] = s

        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Headline metrics: global + per-source aggregations."""
        keys_of_interest = ("xx->xx/bleu", "xx->xx/comet", "en->xx/bleu", "en->xx/comet")
        return {k: agent_metrics[k] for k in keys_of_interest if k in agent_metrics}


if __name__ == "__main__":
    WmtTranslationResourcesServer.run_webserver()
