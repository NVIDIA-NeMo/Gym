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

Two scoring layers:

  * ``verify()`` returns a per-sample sentence-BLEU reward (useful as an RL
    signal) plus a per-rollout xCOMET-XXL ``comet_score``. The COMET score
    is computed by a persistent Ray actor pool (one actor per GPU on the
    extra_gpu node) that keeps the xCOMET-XXL checkpoint resident; each
    verify() call awaits its own future before returning, so every
    rollout in ``rollouts.jsonl`` carries a finite ``comet_score`` whenever
    the model produced a non-empty translation.
  * ``compute_metrics(tasks)`` groups rollouts by
    ``(source_language, target_language, rollout_index)``, computes
    corpus-BLEU with the language-specific sacrebleu tokenizer
    (``13a`` default, ``ja-mecab``/``ko-mecab``/``zh`` as needed), and
    aggregates the per-row COMET scores into per-pair + cross-pair means
    (``xx->xx``, ``<src>->xx``, ``xx->{tgt}``) with ``std_dev_across_runs``.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import ray
from fastapi import FastAPI
from pydantic import PrivateAttr
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
# ``13a`` is sacrebleu's default; ``ja-mecab`` / ``ko-mecab`` need sacrebleu's
# [ja]/[ko] extras installed; ``zh`` is built in.
_TOKENIZER_BY_LANG_PREFIX = {
    "ja": "ja-mecab",
    "ko": "ko-mecab",
    "zh": "zh",
}


def _tokenizer_for(target_language: str) -> str:
    return _TOKENIZER_BY_LANG_PREFIX.get(target_language[:2], "13a")


# --- Thinking-preamble handling ---------------------------------------------
# Reasoning models emit a pre-answer reasoning preamble wrapped in
# <think>...</think>. vLLM's reasoning parser strips the opening <think>
# tag but keeps the closing </think>, so the raw response looks like
#   "We need to translate ... </think>\nProlog"
# We must drop the preamble before scoring or corpus BLEU is computed
# against the reasoning text and collapses (~3x lower BLEU).


def _strip_reasoning_preamble(text: str) -> str:
    """Remove a pre-answer reasoning preamble.

    Three cases:
      1. ``</think>`` present: return everything after the *last* occurrence
         (the actual answer, with the preamble dropped).
      2. ``<think>`` present but no ``</think>``: reasoning started but didn't
         close — the model truncated mid-reasoning. Return empty string so the
         rollout counts as no-answer.
      3. Neither tag present: no inline reasoning preamble (e.g., when the
         endpoint returned reasoning as a structured ``output[i].type="reasoning"``
         block and ``output_text`` already contains only the answer). Return
         the text unchanged.
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].lstrip("\n")
    if "<think>" in text:
        return ""
    return text


# --- Request / response shapes ------------------------------------------------


class WmtTranslationResourcesServerConfig(BaseResourcesServerConfig):
    """Config for the wmt_translation resource server.

    Attributes:
        compute_comet: Run xCOMET-XXL inside ``verify()`` and aggregate in
            ``compute_metrics``. Default True. Turn off for smoke tests or
            RL training runs where only BLEU is needed.
        comet_model: HuggingFace repo for the COMET checkpoint. Resolved via
            ``comet.download_model`` (cached under HF_HOME).
        comet_batch_size: Batch size passed to ``model.predict``.
        comet_num_shards: Number of CometActors to spawn — each loads
            xCOMET-XXL once and serves score requests from the persistent
            actor pool. Each actor requests one ``extra_gpu`` Ray resource,
            so the upper limit is the extra node(s)' GPU count.
        strip_reasoning: When True, drop a ``<think>...</think>`` preamble
            before scoring. Required for reasoning models; safe to leave on
            for instruction-tuned models that don't emit reasoning traces.
    """

    compute_comet: bool = True
    comet_model: str = "Unbabel/XCOMET-XXL"
    comet_batch_size: int = 16
    comet_num_shards: int = 8
    strip_reasoning: bool = True


class WmtTranslationRunRequest(BaseRunRequest):
    text: str
    translation: str
    source_language: str
    target_language: str
    source_lang_name: Optional[str] = None
    target_lang_name: Optional[str] = None


class WmtTranslationVerifyRequest(WmtTranslationRunRequest, BaseVerifyRequest):
    pass


class WmtTranslationVerifyResponse(WmtTranslationVerifyRequest, BaseVerifyResponse):
    # Model's translation, post-strip-reasoning if enabled.
    generation: str
    # Per-sample sentence-BLEU, useful as a dense RL reward.
    sentence_bleu: float
    # Per-rollout xCOMET-XXL score (0–1). None for empty generations or
    # actor-pool failures; aggregate corpus COMET is computed in
    # compute_metrics().
    comet_score: Optional[float] = None


# --- Ray COMET scoring --------------------------------------------------------


def _build_comet_actor_class():
    """Build the persistent CometActor class.

    Each actor is a Ray actor that loads xCOMET-XXL once in ``__init__`` and
    serves score requests from the resident model — no per-call cold load.
    A pool of N actors (one per GPU on the extra_gpu node) is built lazily on
    the first ``verify()`` call. Built lazily so importing this module doesn't
    require Ray to already be initialized.
    """
    import os
    import shutil
    import sys
    from pathlib import Path

    # Cross-node Python setup. The server's venv python may be a symlink into
    # a container-local uv install dir that doesn't exist on remote Ray
    # workers. Stock python on remote workers is ABI-incompatible with the
    # venv's compiled extensions. Fix: mirror the uv-installed python (which
    # ships relocatable python-build-standalone binaries) to a path Ray
    # workers can reach, and hand that to runtime_env as py_executable.
    # One-time copy on first invocation; subsequent calls are a no-op.
    venv_python = Path(sys.executable).resolve()
    if not venv_python.exists():
        raise RuntimeError(
            f"Server-side sys.executable doesn't exist? {venv_python}. "
            "Expected the venv's python to resolve into the local uv install."
        )
    uv_python_root = venv_python.parent.parent

    # Default cache root assumes the canonical container mount at /opt/Gym
    # (cluster deployments). For local dev or non-standard mounts, override
    # via the WMT_TRANSLATION_COMET_PY_CACHE env var to any user-writable
    # path; on multi-node clusters the override must point at a shared
    # filesystem path so cross-node Ray actors find the mirror.
    cache_root = Path(os.environ.get("WMT_TRANSLATION_COMET_PY_CACHE", "/opt/Gym/.cache/comet-python"))
    mirrored_python_root = cache_root / uv_python_root.name
    mirrored_python_bin = mirrored_python_root / "bin" / venv_python.name
    if not mirrored_python_bin.exists():
        LOG.info(
            "Mirroring uv Python install %s -> %s for cross-node Ray tasks",
            uv_python_root,
            mirrored_python_root,
        )
        mirrored_python_root.parent.mkdir(parents=True, exist_ok=True)
        # copytree refuses to overwrite, so use a two-stage atomic rename
        # via a .tmp dir to avoid half-populated caches if interrupted.
        tmp = mirrored_python_root.with_suffix(".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(uv_python_root, tmp, symlinks=True)
        tmp.rename(mirrored_python_root)

    venv_dir = Path(sys.executable).parent.parent
    site_packages = venv_dir / "lib" / "python3.12" / "site-packages"

    env_vars = {
        # Keep CUDA_VISIBLE_DEVICES untouched: when an extra node joins Ray
        # with --num-gpus=0 to hide GPUs from accounting, Ray would zero out
        # CUDA_VISIBLE_DEVICES on the actor. We need physical GPUs visible.
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        # Site-packages (comet, torch, …) is on the shared filesystem; merge
        # with whatever PYTHONPATH the inherited env has.
        "PYTHONPATH": f"{site_packages}:{os.environ.get('PYTHONPATH', '')}",
    }
    # Propagate HF_HOME so actors find the cache populated by the
    # benchmark prepare step. Other HF env vars (HF_HUB_OFFLINE,
    # HF_TOKEN, etc.) are inherited from the parent process — we don't
    # need to override since the prepared cache makes runtime fully
    # offline.
    if os.environ.get("HF_HOME"):
        env_vars["HF_HOME"] = os.environ["HF_HOME"]

    # Schedule on the dedicated COMET node via the custom `extra_gpu` Ray
    # resource. num_gpus=0 because the node hides its GPUs from Ray accounting
    # (advertising them under `extra_gpu` instead); the env_vars flag above
    # preserves physical CUDA_VISIBLE_DEVICES so torch can see them.
    @ray.remote(
        num_gpus=0,
        resources={"extra_gpu": 1},
        runtime_env={"py_executable": str(mirrored_python_bin), "env_vars": env_vars},
    )
    class _CometActor:  # pragma: no cover - needs live Ray cluster + CUDA + unbabel-comet checkpoint
        def __init__(self, gpu_idx: int, model_name: str):
            import torch
            from comet import download_model, load_from_checkpoint

            assert torch.cuda.is_available(), (
                "wmt_translation CometActor requires CUDA. Expected to land on "
                "the extra_gpu node via the custom Ray resource."
            )
            num_devices = torch.cuda.device_count()
            assert num_devices > 0, "No CUDA devices visible to the actor."
            self._gpu_idx = gpu_idx
            # Pin this actor to a specific GPU. Without this every actor
            # defaults to cuda:0 and OOMs (8 × 10B-param xCOMET would need
            # ~320 GB on the first GPU alone).
            self._device = f"cuda:{gpu_idx % num_devices}"
            self._lightning_devices = [gpu_idx % num_devices]

            # Both download_model() and load_from_checkpoint() resolve
            # from the HF cache populated by the benchmark prepare step
            # (see benchmarks/wmt24pp/prepare.py:_prefetch_comet_model).
            # If the cache is missing, this falls back to fetching from
            # HF Hub at startup, subject to HF_HUB_OFFLINE.
            LOG.info("CometActor[%d]: loading %s on %s", gpu_idx, model_name, self._device)
            ckpt_path = download_model(model_name)
            self._model = load_from_checkpoint(ckpt_path)
            self._model.to(self._device).eval()
            LOG.info("CometActor[%d]: ready", gpu_idx)

        def ping(self) -> bool:
            """Cheap readiness probe — server uses this to fail-fast at startup."""
            return True

        def score(self, triples: List[Tuple[str, str, str]], batch_size: int) -> List[float]:
            data = [{"src": s, "mt": m, "ref": r} for s, m, r in triples]
            result = self._model.predict(data, batch_size=batch_size, devices=self._lightning_devices)
            return list(result.scores)

    return _CometActor


# --- Server -------------------------------------------------------------------


class WmtTranslationResourcesServer(SimpleResourcesServer):
    config: WmtTranslationResourcesServerConfig

    # COMET actor pool state — populated lazily on first verify() call so
    # actor creation happens after Ray is fully up and `extra_gpu` is
    # advertised. Pydantic PrivateAttr keeps these out of the config schema.
    _comet_actors: List[Any] = PrivateAttr(default_factory=list)
    _comet_state_lock: Any = PrivateAttr(default=None)
    _comet_actor_idx: int = PrivateAttr(default=0)
    _comet_init_attempted: bool = PrivateAttr(default=False)

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    def _ensure_comet_actors(self) -> None:
        """Initialize the persistent COMET actor pool on first use.

        Lazy on purpose: the resources server may start before the Ray
        cluster has fully stood up (head + workers join asynchronously).
        Deferring actor creation to the first verify() call guarantees the
        cluster is healthy when we ask for actors.
        """
        if self._comet_init_attempted:
            return
        self._comet_init_attempted = True

        if self._comet_state_lock is None:
            self._comet_state_lock = threading.Lock()

        actor_class = _build_comet_actor_class()
        n = max(1, self.config.comet_num_shards)
        actors = [actor_class.remote(gpu_idx=i, model_name=self.config.comet_model) for i in range(n)]
        # Block for actor readiness so init failures surface here instead
        # of stalling verify(). xCOMET-XXL cold-load takes ~60s; large fraction
        # of the budget is consumed by HF 429 retry backoff.
        pings = [a.ping.remote() for a in actors]
        ready, _not_ready = ray.wait(pings, num_returns=n, timeout=300.0)
        # Tolerate partial failure: if some actors exhaust their HF 429 retry
        # budget while others succeed, drop the dead ones and run with the
        # survivors. A reduced pool just scores more slowly.
        ready_actors: List[Any] = []
        for actor, fut in zip(actors, pings):
            if fut not in ready:
                continue
            try:
                ray.get(fut)
                ready_actors.append(actor)
            except Exception:
                LOG.exception("CometActor failed init, dropping from pool")
        if not ready_actors:
            raise RuntimeError(
                f"0/{n} CometActors ready after 300s — check Ray cluster has extra_gpu "
                f"nodes available and HF Hub is reachable."
            )
        self._comet_actors = ready_actors
        if len(ready_actors) < n:
            LOG.warning(
                "COMET pool: %d/%d actors ready (%d failed init); running with reduced pool",
                len(ready_actors),
                n,
                n - len(ready_actors),
            )
        else:
            LOG.info("COMET pool: %d actors ready", n)

    def _dispatch_comet_score(self, src_text: str, generation: str, reference: str) -> Optional[Any]:
        """Fire a per-rollout COMET score on the actor pool, return Ray future.

        Round-robins across the actor pool under a small lock; returns the
        Ray ObjectRef so verify() can ``await`` it directly. Returns
        ``None`` if the pool is unavailable, in which case verify() leaves
        ``comet_score=None`` on that row.
        """
        if not self._comet_actors:
            return None
        with self._comet_state_lock:
            actor = self._comet_actors[self._comet_actor_idx % len(self._comet_actors)]
            self._comet_actor_idx += 1
        try:
            return actor.score.remote([(src_text, generation, reference)], 1)
        except Exception:
            LOG.exception("COMET actor.score.remote dispatch failed")
            return None

    async def verify(self, body: WmtTranslationVerifyRequest) -> WmtTranslationVerifyResponse:
        """Return per-sample sentence-BLEU as the RL reward + per-row COMET.

        Dispatches a per-rollout COMET score on the persistent actor pool
        and awaits the future before returning, so each row in
        rollouts.jsonl carries its own ``comet_score``. Concurrent verify()
        coroutines all dispatch into the same pool; each await yields
        control while its actor processes the triple, so end-to-end
        throughput is bounded by the pool's parallel scoring rate, not by
        the per-row await.
        """
        if self.config.compute_comet:
            self._ensure_comet_actors()

        raw = body.response.output_text or ""
        # Drop the reasoning preamble before scoring so BLEU is computed
        # against the actual translation only.
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

        comet_score: Optional[float] = None
        if self.config.compute_comet:
            comet_future = self._dispatch_comet_score(
                src_text=body.text or "",
                generation=generation,
                reference=body.translation or "",
            )
            if comet_future is not None:
                try:
                    scores = await comet_future
                    if scores:
                        comet_score = float(scores[0])
                except Exception:
                    LOG.exception("COMET await failed for verify(); leaving comet_score=None")

        return WmtTranslationVerifyResponse(
            **body.model_dump(),
            reward=reward,
            generation=generation,
            sentence_bleu=sent_score,
            comet_score=comet_score,
        )

    # --- COMET aggregation ----------------------------------------------------

    def _collect_per_row_comet(
        self,
        tasks: List[List[Dict[str, Any]]],
        max_k: int,
        comet_per_pair: Dict[Tuple[str, str], List[List[float]]],
    ) -> None:
        """Read per-row ``comet_score`` from rollout dicts and bucket by pair/k.

        verify() awaits its COMET future and stores the resolved score on
        each rollout response, so by the time compute_metrics() runs, the
        scores are already in ``tasks``. This method just buckets them.
        """
        for task_rollouts in tasks:
            for k, rollout in enumerate(task_rollouts):
                if k >= max_k:
                    break
                score = rollout.get("comet_score")
                if score is None:
                    continue
                src = rollout.get("source_language")
                tgt = rollout.get("target_language")
                if not src or not tgt:
                    continue
                comet_per_pair[(src, tgt)][k].append(float(score))

    # --- Aggregate metrics ---------------------------------------------------

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compute corpus BLEU + (optional) COMET metrics.

        Output keys:

          <src>-><tgt>/bleu                 (mean across rollouts)
          <src>-><tgt>/bleu_std_dev_across_runs
          <src>-><tgt>/comet                (mean across rollouts)
          <src>-><tgt>/comet_std_dev_across_runs
          <src>->xx/bleu  xx->xx/bleu  xx-><tgt>/bleu   (aggregations)
          ... same with /comet
        """
        if not tasks:
            return {}

        # 1. Bucket rollouts by (src, tgt) × rollout index. Use the MIN
        # rollouts-per-task as the bucket count so every bucket is
        # comparably sized (one fully-covered sample per task).
        rollout_counts = [len(r) for r in tasks]
        max_k = min(rollout_counts) if rollout_counts else 0

        per_pair_runs: Dict[Tuple[str, str], List[List[Tuple[str, str]]]] = defaultdict(
            lambda: [list() for _ in range(max_k)]
        )

        any_comet_rows = False
        for task_rollouts in tasks:
            for k, rollout in enumerate(task_rollouts):
                if k >= max_k:
                    break
                src = rollout.get("source_language")
                tgt = rollout.get("target_language")
                if not src or not tgt:
                    continue
                ref = rollout.get("translation") or ""
                mt = rollout.get("generation") or ""
                per_pair_runs[(src, tgt)][k].append((mt, ref))
                if self.config.compute_comet and rollout.get("comet_score") is not None:
                    any_comet_rows = True

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

        # 3. COMET aggregation: bucket the per-row comet_score values that
        # verify() already populated.
        comet_per_pair: Dict[Tuple[str, str], List[List[float]]] = defaultdict(lambda: [list() for _ in range(max_k)])
        if self.config.compute_comet and any_comet_rows:
            self._collect_per_row_comet(tasks=tasks, max_k=max_k, comet_per_pair=comet_per_pair)

        # Per-rollout-index mean COMET per (pair, k), then averaged across k.
        comet_mean_per_pair: Dict[Tuple[str, str], List[float]] = {}
        for pair_key, per_run in comet_per_pair.items():
            means = []
            for run_scores in per_run:
                if run_scores:
                    means.append(100.0 * sum(run_scores) / len(run_scores))
            comet_mean_per_pair[pair_key] = means

        # 4. Build output dict with per-pair + cross-pair aggregations.
        metrics: Dict[str, Any] = {}
        all_pairs = sorted(per_pair_runs.keys())

        def _mean_std(values: List[float]) -> Tuple[float, float]:
            if not values:
                return (0.0, 0.0)
            n = len(values)
            mean = sum(values) / n
            if n < 2:
                return (mean, 0.0)
            var = sum((v - mean) ** 2 for v in values) / n  # population std
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

        # Aggregations: xx->xx, <src>->xx, xx->{tgt}. For each, average per-run
        # BLEU across the contributing pairs first (per-run mean of per-pair
        # BLEU), then average across runs.
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
