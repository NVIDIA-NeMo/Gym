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
import threading
import uuid
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

    Three cases:
      1. ``</think>`` present: return everything after the *last* occurrence
         (the actual answer, with the preamble dropped).
      2. ``<think>`` present but no ``</think>``: reasoning started but didn't
         close — the model truncated mid-reasoning. Return empty string so the
         rollout counts as no-answer (matches Skills' ``parse_reasoning=True``).
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
        compute_comet: Run xCOMET-XXL in ``compute_metrics``. Default True.
            Turn off for smoke tests or RL training runs where only BLEU
            is needed.
        comet_model: HuggingFace repo for the COMET checkpoint. Resolved via
            ``comet.download_model`` (cached under HF_HOME).
        comet_batch_size: Batch size passed to ``model.predict``.
        comet_num_gpus: Deprecated — scheduling now uses the `extra_gpu`
            custom Ray resource (advertised by nemo_skills' patched
            `get_ray_server_cmd` on DP-excess worker nodes).
        comet_num_shards: Number of parallel Ray tasks to shard COMET
            scoring across. Each task requests `extra_gpu: 1`, so the
            upper limit is the extra node's GPU count (default 8 on the
            A100 draco nodes). Set to 1 for single-GPU scoring.
    """

    compute_comet: bool = True
    comet_model: str = "Unbabel/XCOMET-XXL"
    comet_batch_size: int = 16
    comet_num_shards: int = 8
    # Dedicated-node topology with GPUs hidden from Ray:
    #   * Recipe requests server_nodes = dp_size + 1.
    #   * get_ray_server_cmd starts the extra node with `--num-gpus=0` +
    #     `resources={"extra_gpu": 8}` so vLLM's compiled-DAG / Ray-DP
    #     node scans don't see it (the extra node's presence was causing
    #     a 300s RayChannelTimeoutError in vLLM's compiled DAG even after
    #     placement-group creation was fixed upstream).
    #   * The COMET Ray task asks for `resources={"extra_gpu": 1}` +
    #     `num_gpus=0`, and sets RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
    #     so the physical GPUs stay visible to the task process.
    comet_num_gpus: float = 1.0  # retained for backwards-compat; unused
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
    # Streaming COMET key — when verify() dispatches a COMET score future to
    # the persistent actor pool, this points at the server-side dict entry
    # that compute_metrics() drains via ray.get. Optional so the response is
    # backwards-compatible with rollouts collected before this hook landed.
    comet_id: Optional[str] = None


# --- Ray COMET scoring --------------------------------------------------------


# Build the remote function lazily so importing this module doesn't require
# Ray to already be initialized. ``config.comet_num_gpus`` parameterises the
# GPU allocation at call time.
def _build_comet_remote():
    # runtime_env pins the Ray worker to THIS server's venv (py_executable)
    # and forces HF ONLINE mode for this task (env_vars). Gym's main venv
    # doesn't have unbabel-comet; the wmt_translation server's venv does
    # (via requirements.txt). Without py_executable, the remote task runs
    # in the head venv and fails with ModuleNotFoundError on `comet`.
    # Env-var flip: vLLM needs HF_HUB_OFFLINE=1 at startup to dodge HF 429s
    # on Nemotron model_info, but COMET's xlm-roberta-xxl tokenizer resolver
    # needs online (returns `None` and throws OSError "Not found: None" in
    # offline mode even with cached weights). We also propagate HF_TOKEN /
    # HF_HOME explicitly because runtime_env sandboxes the worker's env.
    # Pattern copied from resources_servers/code_gen (lcb_integration).
    import os
    import shutil
    import sys
    from pathlib import Path

    # Cross-node Python setup. The wmt_translation venv's .venv/bin/python is
    # a symlink into a CONTAINER-LOCAL uv install dir (created on the head
    # when requirements.txt was installed) — the extra node's container
    # never ran uv, so that path doesn't exist there. And the extra node's
    # stock /usr/local/bin/python3.12 is ABI-incompatible with the venv's
    # compiled extensions (torchvision fails with "partially initialized
    # module" on import).
    #
    # Fix: make the uv-installed Python dir itself reachable over Lustre.
    # python-build-standalone binaries (what uv ships) are relocatable, so
    # we mirror the whole install dir under /opt/Gym/.cache/comet-python/
    # (Lustre) and hand that path to Ray as py_executable. One-time copy on
    # first invocation — subsequent calls are a no-op existence check.
    venv_python = Path(sys.executable).resolve()
    if not venv_python.exists():
        raise RuntimeError(
            f"Server-side sys.executable doesn't exist? {venv_python}. "
            "Expected the venv's python to resolve into the local uv install."
        )
    uv_python_root = venv_python.parent.parent  # .../cpython-3.12.12-.../

    # Lustre cache root is overridable via env var for local testing / alternate
    # cluster layouts. Default matches the draco-oci mount layout.
    cache_root = Path(os.environ.get("WMT_TRANSLATION_COMET_PY_CACHE", "/opt/Gym/.cache/comet-python"))
    lustre_python_root = cache_root / uv_python_root.name
    lustre_python_bin = lustre_python_root / "bin" / venv_python.name
    if not lustre_python_bin.exists():
        LOG.info(
            "Mirroring uv Python install %s -> %s for cross-node Ray tasks",
            uv_python_root,
            lustre_python_root,
        )
        lustre_python_root.parent.mkdir(parents=True, exist_ok=True)
        # copytree refuses to overwrite, so use a two-stage atomic rename
        # via a .tmp dir to avoid half-populated caches if we're interrupted.
        tmp = lustre_python_root.with_suffix(".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(uv_python_root, tmp, symlinks=True)
        tmp.rename(lustre_python_root)

    # Resolve the venv site-packages (comet, torch, torchvision — all Lustre
    # files already, not symlinks). Inject via PYTHONPATH so the Lustre-
    # mirrored Python can import them.
    venv_dir = Path(sys.executable).parent.parent  # .venv/bin/python -> .venv
    site_packages = venv_dir / "lib" / "python3.12" / "site-packages"

    env_vars = {
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
        # Keep the task's CUDA_VISIBLE_DEVICES untouched — the extra node's
        # Ray agent already advertises --num-gpus=0, so Ray would zero this
        # out without this flag. We need the physical GPUs visible so COMET
        # can torch.cuda() load.
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        # site-packages for comet, torch, etc. is Lustre-shared; merge it
        # with whatever PYTHONPATH the inherited env has.
        "PYTHONPATH": f"{site_packages}:{os.environ.get('PYTHONPATH', '')}",
    }
    for k in ("HF_HOME", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(k):
            env_vars[k] = os.environ[k]

    # Schedule on the "extra" (DP-unclaimed) node via the custom `extra_gpu`
    # resource advertised by nemo_skills' get_ray_server_cmd on workers with
    # SLURM_PROCID >= dp_size. num_gpus=0 because the extra node hides its
    # GPUs from Ray's accounting (to dodge vLLM's compiled-DAG scan of
    # ray.nodes() that hangs when untracked GPU nodes are present); the
    # env_vars flag above preserves physical CUDA_VISIBLE_DEVICES.
    @ray.remote(
        num_gpus=0,
        resources={"extra_gpu": 1},
        runtime_env={"py_executable": str(lustre_python_bin), "env_vars": env_vars},
    )
    def _score_comet(  # pragma: no cover - needs live Ray cluster + CUDA + unbabel-comet checkpoint
        triples: List[Tuple[str, str, str]], model_name: str, batch_size: int, gpu_idx: int = 0
    ) -> List[float]:
        import torch
        from comet import download_model, load_from_checkpoint

        # Hard-assert CUDA: the task is pinned to the extra Ray node by the
        # `extra_gpu` custom resource, and CUDA_VISIBLE_DEVICES is preserved
        # via RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1. If this fires,
        # the extra node didn't join Ray with --num-gpus=0 + extra_gpu
        # resource (check get_ray_server_cmd wiring).
        assert torch.cuda.is_available(), (
            "wmt_translation COMET task requires a CUDA device. "
            "Expected the Ray worker to land on the extra node via the "
            "extra_gpu custom resource; got no CUDA."
        )
        # Pin this shard to a specific GPU on the extra node. Without this
        # every task defaults to cuda:0 and OOMs (8 × 10B-param xCOMET would
        # need ~320 GB on the first GPU alone).
        num_devices = torch.cuda.device_count()
        assert num_devices > 0, "No CUDA devices visible to the Ray task."
        device = f"cuda:{gpu_idx % num_devices}"

        LOG.info(
            "Loading xCOMET model %s on %s (shard gpu_idx=%d, num_devices=%d)",
            model_name,
            device,
            gpu_idx,
            num_devices,
        )
        ckpt_path = download_model(model_name)
        model = load_from_checkpoint(ckpt_path)
        model.to(device).eval()

        data = [{"src": s, "mt": m, "ref": r} for s, m, r in triples]
        LOG.info("Scoring %d (src, mt, ref) triples at batch_size=%d on %s", len(data), batch_size, device)
        # Constrain lightning's pytorch_lightning `Trainer` (used under
        # `model.predict`) to this task's GPU too. xCOMET's predict()
        # instantiates a Trainer with `devices="auto"` which otherwise
        # auto-detects all 8 visible GPUs and tries to DataParallel across
        # them — fighting the other 7 sibling tasks. `gpus=[idx]` pins it.
        result = model.predict(data, batch_size=batch_size, devices=[gpu_idx % num_devices])
        return list(result.scores)

    return _score_comet


def _build_comet_actor_class():
    """Build the persistent CometActor class used by streaming COMET dispatch.

    Same py_executable / runtime_env / GPU-pinning logic as `_build_comet_remote`
    above, but as a Ray *actor* class so we can keep the xCOMET-XXL model
    resident across the entire run instead of cold-loading per
    ``compute_metrics()`` call. With N>=8 actors on the 8-GPU extra_gpu node,
    every rollout's score request lands on a ready model immediately —
    no idle GPUs during the rollout phase, fail-fast at server startup.
    """
    import os
    import shutil
    import sys
    from pathlib import Path

    venv_python = Path(sys.executable).resolve()
    if not venv_python.exists():
        raise RuntimeError(
            f"Server-side sys.executable doesn't exist? {venv_python}. "
            "Expected the venv's python to resolve into the local uv install."
        )
    uv_python_root = venv_python.parent.parent

    cache_root = Path(os.environ.get("WMT_TRANSLATION_COMET_PY_CACHE", "/opt/Gym/.cache/comet-python"))
    lustre_python_root = cache_root / uv_python_root.name
    lustre_python_bin = lustre_python_root / "bin" / venv_python.name
    if not lustre_python_bin.exists():
        LOG.info(
            "Mirroring uv Python install %s -> %s for cross-node Ray tasks",
            uv_python_root,
            lustre_python_root,
        )
        lustre_python_root.parent.mkdir(parents=True, exist_ok=True)
        tmp = lustre_python_root.with_suffix(".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(uv_python_root, tmp, symlinks=True)
        tmp.rename(lustre_python_root)

    venv_dir = Path(sys.executable).parent.parent
    site_packages = venv_dir / "lib" / "python3.12" / "site-packages"

    env_vars = {
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        "PYTHONPATH": f"{site_packages}:{os.environ.get('PYTHONPATH', '')}",
    }
    for k in ("HF_HOME", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(k):
            env_vars[k] = os.environ[k]

    @ray.remote(
        num_gpus=0,
        resources={"extra_gpu": 1},
        runtime_env={"py_executable": str(lustre_python_bin), "env_vars": env_vars},
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
            self._device = f"cuda:{gpu_idx % num_devices}"
            self._lightning_devices = [gpu_idx % num_devices]

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

    # Streaming COMET state — populated lazily on first verify() call so
    # actor creation happens after Ray is fully up and `extra_gpu` is
    # advertised. Pydantic PrivateAttr keeps these out of the config schema.
    _comet_actors: List[Any] = PrivateAttr(default_factory=list)
    _comet_futures: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _comet_state_lock: Any = PrivateAttr(default=None)
    _comet_actor_idx: int = PrivateAttr(default=0)
    _comet_init_attempted: bool = PrivateAttr(default=False)
    _comet_init_failed: bool = PrivateAttr(default=False)

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    def _ensure_comet_actors(self) -> None:
        """Initialize the persistent COMET actor pool on first use.

        Lazy on purpose: the resources server starts before the vLLM-side
        Ray cluster is fully up (head + workers join asynchronously). Doing
        actor creation at server startup races the Ray cluster bring-up. By
        deferring to the first verify() call, we guarantee the cluster is
        healthy and `extra_gpu` is advertised when we ask for actors.

        On failure, set ``_comet_init_failed`` so subsequent verify() calls
        skip the streaming dispatch (compute_metrics still falls back to the
        legacy batch path with the same graceful BLEU-only behavior).
        """
        if self._comet_init_attempted:
            return
        self._comet_init_attempted = True

        if self._comet_state_lock is None:
            self._comet_state_lock = threading.Lock()

        try:
            actor_class = _build_comet_actor_class()
            n = max(1, self.config.comet_num_shards)
            actors = [actor_class.remote(gpu_idx=i, model_name=self.config.comet_model) for i in range(n)]
            # Block briefly for actor readiness — surfaces "no extra_gpu",
            # "comet checkpoint missing", "cuda init failed" etc as a clear
            # failure here instead of hanging until end-of-batch. Generous
            # timeout because xCOMET-XXL load takes ~60s cold.
            ready, not_ready = ray.wait(
                [a.ping.remote() for a in actors],
                num_returns=n,
                timeout=300.0,
            )
            if not_ready:
                raise RuntimeError(
                    f"{len(not_ready)}/{n} CometActors not ready after 300s — "
                    f"check Ray cluster has extra_gpu nodes available."
                )
            ray.get(ready)  # raises if any actor's ping aborted
            self._comet_actors = actors
            LOG.info("Streaming COMET: %d persistent actors ready on extra_gpu node", n)
        except Exception as e:
            LOG.exception("Streaming COMET init failed; falling back to batch dispatch: %s", e)
            self._comet_init_failed = True
            self._comet_actors = []

    def _dispatch_comet_streaming(self, src_text: str, generation: str, reference: str) -> Optional[str]:
        """Fire a per-rollout COMET score on the actor pool, return future ID.

        Returns ``None`` if streaming COMET is disabled or unavailable; the
        rollout's response will simply not have a ``comet_id`` and
        compute_metrics() will skip it (or fall back to batch path)."""
        if not self._comet_actors:
            return None
        with self._comet_state_lock:
            actor = self._comet_actors[self._comet_actor_idx % len(self._comet_actors)]
            self._comet_actor_idx += 1
        try:
            future = actor.score.remote([(src_text, generation, reference)], 1)
        except Exception:
            LOG.exception("COMET actor.score.remote dispatch failed")
            return None
        comet_id = uuid.uuid4().hex
        with self._comet_state_lock:
            self._comet_futures[comet_id] = future
        return comet_id

    async def verify(self, body: WmtTranslationVerifyRequest) -> WmtTranslationVerifyResponse:
        """Return per-sample sentence-BLEU as the RL reward.

        The authoritative corpus-BLEU (+ optional COMET) lives in
        ``compute_metrics`` and is what parity comparisons to Skills use.

        If streaming COMET is enabled, also fires a per-rollout COMET score
        on the persistent actor pool and stashes the future server-side
        keyed by ``comet_id``. compute_metrics() drains those futures
        instead of doing a single end-of-batch dispatch.
        """
        if self.config.compute_comet:
            self._ensure_comet_actors()

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

        comet_id = self._dispatch_comet_streaming(
            src_text=body.text or "",
            generation=generation,
            reference=body.translation or "",
        )

        return WmtTranslationVerifyResponse(
            **body.model_dump(),
            reward=reward,
            generation=generation,
            sentence_bleu=sent_score,
            comet_id=comet_id,
        )

    # --- COMET drainers (streaming + batch fallback) -------------------------

    def _drain_streaming_comet(
        self,
        tasks: List[List[Dict[str, Any]]],
        max_k: int,
        comet_per_pair: Dict[Tuple[str, str], List[List[float]]],
    ) -> bool:
        """Resolve COMET futures created by verify() and bucket scores by pair/k.

        Returns True if streaming drain handled scoring (success OR the
        graceful "no futures available" case where every rollout's
        comet_id was empty and we should NOT fall back to batch dispatch),
        False if we should fall back to the batch path.
        """
        # Walk tasks, collecting (comet_id, pair_key, k) for rollouts that
        # have a comet_id (i.e., verify() successfully dispatched).
        slot_for_id: Dict[str, Tuple[Tuple[str, str], int]] = {}
        for task_rollouts in tasks:
            for k, rollout in enumerate(task_rollouts):
                if k >= max_k:
                    break
                cid = rollout.get("comet_id")
                if not cid:
                    continue
                src = rollout.get("source_language")
                tgt = rollout.get("target_language")
                if not src or not tgt:
                    continue
                slot_for_id[cid] = ((src, tgt), k)

        if not slot_for_id:
            # No streaming dispatches happened (e.g., actors weren't ready
            # during the rollout phase). Tell caller to use batch fallback.
            return False

        # Look up futures from server state. Some may have been popped
        # already if compute_metrics is called more than once on the same
        # server instance — that's fine, we only score what's still pending.
        pending: List[Tuple[str, Any]] = []
        with self._comet_state_lock:
            for cid in slot_for_id:
                fut = self._comet_futures.pop(cid, None)
                if fut is not None:
                    pending.append((cid, fut))

        if not pending:
            LOG.warning(
                "Streaming COMET: %d rollouts had comet_id but no pending future "
                "(already drained?). Falling back to batch dispatch.",
                len(slot_for_id),
            )
            return False

        LOG.info("Streaming COMET: draining %d futures from persistent actor pool", len(pending))
        try:
            # Each future returns a list of len-1 scores (one triple per
            # actor.score call from verify()). Map back into per-pair buckets.
            cids = [cid for cid, _ in pending]
            futures = [fut for _, fut in pending]
            results = ray.get(futures)  # raises if any actor.score failed
            for cid, scores in zip(cids, results):
                if not scores:
                    continue
                pair_key, k = slot_for_id[cid]
                comet_per_pair[pair_key][k].append(scores[0])
            return True
        except Exception as e:
            LOG.exception("Streaming COMET drain failed; falling back to batch: %s", e)
            comet_per_pair.clear()
            return False

    def _dispatch_batch_comet(
        self,
        comet_triples: List[Tuple[str, str, str]],
        comet_slots: List[Tuple[Tuple[str, str], int]],
        comet_per_pair: Dict[Tuple[str, str], List[List[float]]],
    ) -> None:
        """End-of-batch COMET dispatch — fallback path when streaming is unavailable.

        Same logic as the original implementation: shard triples across
        ``comet_num_shards`` parallel Ray tasks each on one extra_gpu. Cold-
        loads xCOMET-XXL per call (no persistent actors).
        """
        try:
            num_shards = max(1, min(self.config.comet_num_shards, len(comet_triples)))
            shards: List[List[Tuple[str, str, str]]] = [[] for _ in range(num_shards)]
            for i, t in enumerate(comet_triples):
                shards[i * num_shards // len(comet_triples)].append(t)
            shards = [s for s in shards if s]

            remote_fn = _build_comet_remote()
            LOG.info(
                "Batch COMET: dispatching %d triples across %d parallel Ray tasks "
                "on extra_gpu nodes (batch=%d, model=%s)",
                len(comet_triples),
                len(shards),
                self.config.comet_batch_size,
                self.config.comet_model,
            )
            futures = [
                remote_fn.remote(shard, self.config.comet_model, self.config.comet_batch_size, gpu_idx=i)
                for i, shard in enumerate(shards)
            ]
            shard_results: List[List[float]] = ray.get(futures)
            scores: List[float] = [s for chunk in shard_results for s in chunk]
            assert len(scores) == len(comet_slots), (
                f"COMET scores ({len(scores)}) != slots ({len(comet_slots)}) — sharding bug"
            )
            for score, (pair_key, k) in zip(scores, comet_slots):
                comet_per_pair[pair_key][k].append(score)
        except Exception as e:
            LOG.exception("Batch COMET scoring failed, continuing with BLEU only: %s", e)
            comet_per_pair.clear()

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
        #    by treating rollout index == seed index.
        #
        #    Use the MIN rollouts-per-task as the bucket count, not the max.
        #    Gym's rollout collector occasionally retries a failed rollout,
        #    producing a handful of tasks with extra positions (5-12) that
        #    skew corpus BLEU hard: e.g. a size-1 bucket with a single
        #    mismatched pred-ref pair scores 0.0 and drags the cross-run mean
        #    down by 3×. Capping at the min keeps every bucket comparably
        #    sized (one fully-covered sample per task).
        rollout_counts = [len(r) for r in tasks]
        max_k = min(rollout_counts) if rollout_counts else 0

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
                if k >= max_k:
                    break
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

        # 3. COMET scoring. Two paths:
        #    (a) Streaming COMET (preferred): verify() already fired
        #        per-rollout score requests on a persistent actor pool and
        #        stashed ObjectRefs in self._comet_futures keyed by
        #        comet_id (a uuid stored on each verify response). Drain
        #        them here. Wins: ~1-min xCOMET load amortized across the
        #        whole run, all 8 extra_gpu GPUs working DURING the rollout
        #        phase instead of idle-then-burst, fail-fast at server
        #        startup if the actor pool can't initialize.
        #    (b) Batch fallback: if streaming COMET init failed (e.g., Ray
        #        cluster wasn't ready, extra_gpu unavailable), use the
        #        end-of-batch _build_comet_remote() task from before.
        comet_per_pair: Dict[Tuple[str, str], List[List[float]]] = defaultdict(lambda: [list() for _ in range(max_k)])
        if self.config.compute_comet and comet_triples:
            streaming_drained = False
            if self._comet_actors and not self._comet_init_failed:
                streaming_drained = self._drain_streaming_comet(
                    tasks=tasks, max_k=max_k, comet_per_pair=comet_per_pair
                )
            if not streaming_drained:
                self._dispatch_batch_comet(
                    comet_triples=comet_triples,
                    comet_slots=comet_slots,
                    comet_per_pair=comet_per_pair,
                )

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
