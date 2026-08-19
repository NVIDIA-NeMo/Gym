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
    signal). It does not call xCOMET.
  * ``compute_metrics(tasks)`` groups rollouts by
    ``(source_language, target_language, rollout_index)``, computes
    corpus-BLEU with the language-specific sacrebleu tokenizer
    (``13a`` default, ``ja-mecab``/``ko-mecab``/``zh`` as needed), fills
    missing per-row xCOMET-XXL scores with batched ``predict`` on the
    extra_gpu actor pool (checkpointing ``rollouts.jsonl`` after each
    wave), and aggregates COMET into per-pair + cross-pair means
    (``xx->xx``, ``<src>->xx``, ``xx->{tgt}``) with ``std_dev_across_runs``.
"""

from __future__ import annotations

import logging
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
        compute_comet: Run batched xCOMET-XXL inside ``compute_metrics``.
            Default True. Turn off for smoke tests or RL training runs where
            only BLEU is needed.
        comet_model: HuggingFace repo or local COMET checkpoint path.
        comet_batch_size: Batch size passed to ``model.predict``.
        comet_num_shards: Number of CometActors to spawn — each loads
            xCOMET-XXL once and serves score requests from the persistent
            actor pool. Each actor requests one ``extra_gpu`` Ray resource,
            so the upper limit is the extra node(s)' GPU count.
        comet_use_worker_python: Use the Python environment of the Ray worker
            process instead of mirroring the resources-server Python. Enable
            when the worker node pre-installs the COMET runtime.
        strip_reasoning: When True, drop a ``<think>...</think>`` preamble
            before scoring. Required for reasoning models; safe to leave on
            for instruction-tuned models that don't emit reasoning traces.
    """

    compute_comet: bool = True
    comet_model: str = "Unbabel/XCOMET-XXL"
    comet_batch_size: int = 16
    comet_num_shards: int = 8
    comet_use_worker_python: bool = False
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
    # Per-rollout xCOMET-XXL score (0–1). verify() leaves this unset;
    # compute_metrics() fills it in bulk for non-empty generations.
    comet_score: Optional[float] = None


# --- Ray COMET scoring --------------------------------------------------------


def _build_comet_actor_class(use_worker_python: bool = False):
    """Build the persistent CometActor class.

    Each actor is a Ray actor that loads xCOMET-XXL once in ``__init__`` and
    serves score requests from the resident model — no per-call cold load.
    A pool of N actors (one per GPU on the extra_gpu node) is built lazily on
    the first ``compute_metrics()`` call that has unscored rows. Built lazily
    so importing this module doesn't require Ray to already be initialized.
    """
    import os
    import shutil
    import socket
    import sys
    import uuid
    from pathlib import Path

    env_vars = {
        # Keep CUDA_VISIBLE_DEVICES untouched: when an extra node joins Ray
        # with --num-gpus=0 to hide GPUs from accounting, Ray would zero out
        # CUDA_VISIBLE_DEVICES on the actor. We need physical GPUs visible.
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
    }
    # Propagate HF_HOME so actors find the cache populated by the
    # benchmark prepare step. Other HF env vars (HF_HUB_OFFLINE,
    # HF_TOKEN, etc.) are inherited from the parent process — we don't
    # need to override since the prepared cache makes runtime fully
    # offline.
    if os.environ.get("HF_HOME"):
        env_vars["HF_HOME"] = os.environ["HF_HOME"]

    runtime_env = {"env_vars": env_vars}
    if not use_worker_python:
        # Cross-node Python setup. The server's venv python may be a symlink into
        # a container-local uv install dir that doesn't exist on remote Ray
        # workers. Mirror the relocatable uv Python to a shared path.
        venv_python = Path(sys.executable).resolve()
        if not venv_python.exists():
            raise RuntimeError(
                f"Server-side sys.executable doesn't exist? {venv_python}. "
                "Expected the venv's python to resolve into the local uv install."
            )
        uv_python_root = venv_python.parent.parent
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
            # Stage per-writer; a shared staging path races on rmtree and on the final rename.
            tmp: Path = (
                mirrored_python_root.parent
                / f".{mirrored_python_root.name}.tmp.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex[:8]}"
            )
            try:
                shutil.copytree(uv_python_root, tmp, symlinks=True)
                try:
                    tmp.rename(mirrored_python_root)
                except OSError:
                    # Another builder won the publish; adopt their mirror if it's valid, else re-raise.
                    if not mirrored_python_bin.exists():
                        raise
            finally:
                if tmp.exists():
                    shutil.rmtree(tmp, ignore_errors=True)
        venv_dir = Path(sys.executable).parent.parent
        site_packages = venv_dir / "lib" / "python3.12" / "site-packages"
        env_vars["PYTHONPATH"] = f"{site_packages}:{os.environ.get('PYTHONPATH', '')}"
        runtime_env["py_executable"] = str(mirrored_python_bin)

    # Schedule on the dedicated COMET node via the custom `extra_gpu` Ray
    # resource. num_gpus=0 because the node hides its GPUs from Ray accounting
    # (advertising them under `extra_gpu` instead); the env_vars flag above
    # preserves physical CUDA_VISIBLE_DEVICES so torch can see them.
    @ray.remote(
        num_gpus=0,
        resources={"extra_gpu": 1},
        runtime_env=runtime_env,
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
            ckpt_path = model_name if model_name.startswith("/") else download_model(model_name)
            self._model = load_from_checkpoint(ckpt_path)
            self._model.to(self._device).eval()
            LOG.info("CometActor[%d]: ready", gpu_idx)

        def ping(self) -> bool:
            """Cheap readiness probe — server uses this to fail-fast at startup."""
            return True

        def score(self, triples: List[Tuple[str, str, str]], batch_size: int) -> List[float]:
            import os

            os.chdir("/tmp")
            data = [{"src": s, "mt": m, "ref": r} for s, m, r in triples]
            result = self._model.predict(data, batch_size=batch_size, devices=self._lightning_devices)
            return list(result.scores)

    return _CometActor


# --- Server -------------------------------------------------------------------


class WmtTranslationResourcesServer(SimpleResourcesServer):
    config: WmtTranslationResourcesServerConfig

    # COMET actor pool state — populated lazily during compute_metrics() so
    # actor creation happens after Ray is fully up and `extra_gpu` is
    # advertised. Pydantic PrivateAttr keeps these out of the config schema.
    _comet_actors: List[Any] = PrivateAttr(default_factory=list)
    _comet_init_attempted: bool = PrivateAttr(default=False)

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    def _ensure_comet_actors(self) -> None:
        """Initialize the persistent COMET actor pool on first use.

        Lazy on purpose: the resources server may start before the Ray
        cluster has fully stood up (head + workers join asynchronously).
        Deferring actor creation until aggregate scoring also keeps COMET off
        the rollout-generation path.
        """
        if self._comet_init_attempted:
            return
        self._comet_init_attempted = True

        actor_class = _build_comet_actor_class(use_worker_python=self.config.comet_use_worker_python)
        n = max(1, self.config.comet_num_shards)
        actors = [actor_class.remote(gpu_idx=i, model_name=self.config.comet_model) for i in range(n)]
        # Block for actor readiness so init failures surface here instead
        # of stalling aggregate scoring. xCOMET-XXL cold-load takes ~60s; a large fraction
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

    async def verify(self, body: WmtTranslationVerifyRequest) -> WmtTranslationVerifyResponse:
        """Return per-sample sentence-BLEU and defer COMET to compute_metrics()."""
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

        return WmtTranslationVerifyResponse(
            **body.model_dump(),
            reward=reward,
            generation=generation,
            sentence_bleu=sent_score,
            comet_score=None,
        )

    # --- COMET aggregation ----------------------------------------------------

    def _collect_per_row_comet(
        self,
        tasks: List[List[Dict[str, Any]]],
        max_k: int,
        comet_per_pair: Dict[Tuple[str, str], List[List[float]]],
    ) -> None:
        """Bucket per-row ``comet_score`` values by language pair and rollout."""
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

    def _checkpoint_comet_scores(self, tasks):
        import json
        import os
        from pathlib import Path

        raw = os.environ.get("WMT_COMET_CHECKPOINT_JSONL")
        path = Path(raw) if raw else Path("/results/evaluator_rollouts.jsonl")
        if not path.exists():
            return
        by_key = {}
        for task_rollouts in tasks:
            for rollout in task_rollouts:
                if "_ng_task_index" not in rollout:
                    continue
                score = rollout.get("comet_score")
                if score is None:
                    continue
                key = (rollout["_ng_task_index"], rollout.get("_ng_rollout_index", 0))
                by_key[key] = float(score)
        tmp = path.with_name(path.name + ".comet_tmp")
        with path.open() as inf, tmp.open("w") as out:
            for line in inf:
                row = json.loads(line)
                key = (row.get("_ng_task_index"), row.get("_ng_rollout_index", 0))
                if key in by_key:
                    row["comet_score"] = by_key[key]
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)
        LOG.info("COMET checkpointed %d scores to %s", len(by_key), path)

    def _score_missing_comet(self, tasks):
        if not self.config.compute_comet:
            return
        need = []
        for task_rollouts in tasks:
            for rollout in task_rollouts:
                generation = (rollout.get("generation") or "").strip()
                if generation and rollout.get("comet_score") is None:
                    need.append(rollout)
        if not need:
            return
        self._ensure_comet_actors()
        if not self._comet_actors:
            raise RuntimeError("COMET actor pool empty after _ensure_comet_actors")
        batch_size = max(1, self.config.comet_batch_size)
        n_actors = len(self._comet_actors)
        wave_span = batch_size * n_actors
        n_need = len(need)
        n_waves = (n_need + wave_span - 1) // wave_span
        for wave_i in range(n_waves):
            wave_start = wave_i * wave_span
            wave_end = min(n_need, wave_start + wave_span)
            futures = []
            wave_chunks = []
            actor_i = 0
            chunk_start = wave_start
            while chunk_start < wave_end:
                chunk_end = min(wave_end, chunk_start + batch_size)
                chunk = need[chunk_start:chunk_end]
                triples = [
                    (
                        str(row.get("text") or ""),
                        str(row.get("generation") or ""),
                        str(row.get("translation") or ""),
                    )
                    for row in chunk
                ]
                futures.append(self._comet_actors[actor_i].score.remote(triples, batch_size))
                wave_chunks.append(chunk)
                actor_i += 1
                chunk_start = chunk_end
            results = ray.get(futures)
            if len(results) != len(wave_chunks):
                raise RuntimeError(f"COMET wave returned {len(results)} actor results for {len(wave_chunks)} chunks")
            for chunk, scores in zip(wave_chunks, results):
                if scores is None:
                    raise RuntimeError(f"COMET predict returned None for {len(chunk)} triples")
                if len(scores) != len(chunk):
                    raise RuntimeError(f"COMET predict length mismatch: expected {len(chunk)} got {len(scores)}")
                for rollout, score in zip(chunk, scores):
                    rollout["comet_score"] = float(score)
            self._checkpoint_comet_scores(tasks)
            LOG.info(
                "COMET batched predict wave=%d/%d n=%d remaining=%d configured_batch_size=%d actors=%d",
                wave_i + 1,
                n_waves,
                wave_end - wave_start,
                n_need - wave_end,
                batch_size,
                n_actors,
            )

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

        if self.config.compute_comet:
            self._score_missing_comet(tasks)

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

        # 3. COMET aggregation: bucket the per-row values that
        # _score_missing_comet() or a resumed rollout already populated.
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
