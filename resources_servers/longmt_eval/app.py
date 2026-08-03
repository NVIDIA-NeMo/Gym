# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""longmt_eval resource server — document-level MT evaluation with SEGALE.

Scoring layers:
  * verify() strips the reasoning preamble, then dispatches the (source, MT)
    pair to a persistent SegaleActor on the extra_gpu node. The actor runs the
    three-phase SEGALE pipeline (LASER2 embed → vecalign align → COMETKiwi
    score) and returns comet_qe as the RL reward.
  * compute_metrics() groups rollouts by target_language and reports mean
    comet_qe, lang_fidelity, and segment-level statistics per language.

Set compute_segale: false for local smoke tests — verify() returns reward=0.0
without touching the actor pool, so the server starts without a GPU.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

import ray
from fastapi import FastAPI
from pydantic import PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reasoning-tag assertion
# ---------------------------------------------------------------------------
# Reasoning must be parsed/stripped by the inference server (e.g. via
# vLLM's --reasoning-parser flag or an equivalent agent-side step) before the
# response reaches the verifier. We assert that contract here instead of
# silently rescuing malformed generations — a leaked <think>...</think>
# preamble is a configuration bug, not something to paper over.


def _assert_no_reasoning(text: str) -> None:
    assert "<think>" not in text and "</think>" not in text, (
        "longmt_eval received a generation containing <think>/</think> "
        "reasoning tags. Reasoning must be parsed by the inference server "
        "before reaching the verifier."
    )


# ---------------------------------------------------------------------------
# Request / response shapes
# ---------------------------------------------------------------------------


class LongmtEvalConfig(BaseResourcesServerConfig):
    """Config for the longmt_eval resource server.

    Attributes:
        compute_segale: Run the full SEGALE pipeline inside verify(). Set
            False for local smoke tests — verify() returns reward=0.0 without
            touching the actor pool, so the server starts without a GPU.
        comet_model: HuggingFace repo for the COMETKiwi checkpoint. Resolved
            from HF_HOME (pre-populated by the prepare step).
        comet_batch_size: Number of aligned span pairs per COMETKiwi forward
            pass inside each actor. Larger values are faster but use more VRAM;
            8 is safe for wmt22-cometkiwi-da on a 80 GB GPU even for very long
            documents with hundreds of spans.
        comet_num_shards: Number of physical GPUs to use on the extra_gpu
            node(s). Total actors spawned = comet_num_shards * actors_per_gpu.
            Each actor loads LASER2 + COMETKiwi once and handles one document
            at a time, giving document-level parallelism without per-call model
            reloading.
        actors_per_gpu: Number of SegaleActors to co-place on each physical
            GPU. Each actor claims 1/actors_per_gpu of the extra_gpu custom
            resource so Ray packs them onto the same node without over-
            subscribing. actors_per_gpu=4 on a single H100 (comet_num_shards=1)
            mirrors the oracle-test configuration that scored 80 docs in ~6min.
        embed_batch_size: Number of overlap strings per LASER2 encode_sentences()
            call inside each actor. Larger values improve GPU utilisation for
            long documents with many overlaps; 512 is a safe default.
        assert_no_reasoning: When True, assert the incoming generation contains
            no <think>...</think> tags. Reasoning is expected to be parsed by
            the inference server upstream; an assertion failure here surfaces
            misconfiguration instead of silently scoring a leaked preamble.
        use_extra_gpu: When False (default), SEGALE actors claim fractional
            num_gpus so Ray manages CUDA_VISIBLE_DEVICES. Use this when the gym
            runs its own Ray cluster with dedicated GPU nodes (HTTP-separated
            from vLLM). When True, actors claim the custom extra_gpu resource
            (num_gpus=0) for deployments where the gym joins the vLLM Ray
            cluster and a separate node registers extra_gpu resources via
            `ray start --resources='{"extra_gpu": N}'`.
    """

    compute_segale: bool = True
    comet_model: str = "Unbabel/wmt22-cometkiwi-da"
    comet_batch_size: int = 8
    comet_num_shards: int = 4
    actors_per_gpu: int = 4
    embed_batch_size: int = 512
    assert_no_reasoning: bool = True
    use_extra_gpu: bool = False


class LongmtEvalRunRequest(BaseRunRequest):
    # tiktoken-truncated source text written by prepare.py — IS the prompt source.
    text: str
    source_language: str
    target_language: str
    doc_id: str
    target_len: Optional[int] = None  # tiktoken token count the source was truncated to


class LongmtEvalVerifyRequest(LongmtEvalRunRequest, BaseVerifyRequest):
    pass


class LongmtEvalVerifyResponse(LongmtEvalVerifyRequest, BaseVerifyResponse):
    generation: str
    comet_qe: Optional[float] = None
    lang_fidelity: Optional[float] = None
    total_seg: int = 0
    misaligned_seg: int = 0
    spans: Optional[List[Dict]] = None
    segale_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Resource server
# ---------------------------------------------------------------------------


class LongmtEvalServer(SimpleResourcesServer):
    config: LongmtEvalConfig

    _segale_actors: List[Any] = PrivateAttr(default_factory=list)
    _actor_lock: Any = PrivateAttr(default=None)
    _actor_idx: int = PrivateAttr(default=0)
    _actors_init_attempted: bool = PrivateAttr(default=False)

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    def _ensure_actors(self) -> None:
        if self._actors_init_attempted:
            return
        self._actors_init_attempted = True
        self._actor_lock = threading.Lock()

        from segale_actor import _build_segale_actor_class

        actor_cls = _build_segale_actor_class(
            actors_per_gpu=self.config.actors_per_gpu,
            use_extra_gpu=self.config.use_extra_gpu,
        )
        n = max(1, self.config.comet_num_shards * self.config.actors_per_gpu)
        actors = [
            actor_cls.remote(
                gpu_idx=i,
                comet_model=self.config.comet_model,
                comet_batch_size=self.config.comet_batch_size,
                embed_batch_size=self.config.embed_batch_size,
            )
            for i in range(n)
        ]

        pings = [a.ping.remote() for a in actors]
        ready, _ = ray.wait(pings, num_returns=n, timeout=300.0)

        live: List[Any] = []
        for actor, fut in zip(actors, pings):
            if fut not in ready:
                continue
            try:
                ray.get(fut)
                live.append(actor)
            except Exception:
                LOG.exception("SegaleActor failed init; dropping from pool")

        if not live:
            raise RuntimeError(
                f"0/{n} SegaleActors ready after 300s — check that extra_gpu nodes "
                "are available and LASER_HOME / HF_HOME are set."
            )
        if len(live) < n:
            LOG.warning(
                "SegaleActor pool: %d/%d actors ready; running with reduced pool",
                len(live),
                n,
            )
        # Interleave the actor list so round-robin dispatch spreads across
        # physical GPUs before doubling up. In use_extra_gpu=False mode Ray
        # bin-packs the fractional-GPU actors as [GPU0×A, GPU1×A, ...]; without
        # this reorder the first A verify() calls all land on GPU 0 while
        # GPU 1 sits idle. Skip when actors were dropped during ping (shape no
        # longer matches G*A) or in use_extra_gpu=True mode (creation order is
        # already [GPU0, GPU1, GPU0, GPU1, ...] via gpu_idx % device_count).
        G = self.config.comet_num_shards
        A = self.config.actors_per_gpu
        if not self.config.use_extra_gpu and len(live) == G * A:
            live = [live[g * A + s] for s in range(A) for g in range(G)]
        self._segale_actors = live
        LOG.info(
            "SegaleActor pool: %d actors ready (dispatch_order=%s)",
            len(live),
            "interleaved" if not self.config.use_extra_gpu and len(live) == G * A else "creation",
        )

    def _dispatch(self, source_text: str, mt_text: str, target_language: str) -> Optional[Any]:
        if not self._segale_actors:
            return None
        with self._actor_lock:
            actor = self._segale_actors[self._actor_idx % len(self._segale_actors)]
            self._actor_idx += 1
        try:
            return actor.score.remote(source_text, mt_text, target_language)
        except Exception:
            LOG.exception("SegaleActor dispatch failed")
            return None

    async def verify(self, body: LongmtEvalVerifyRequest) -> LongmtEvalVerifyResponse:
        if self.config.compute_segale:
            self._ensure_actors()

        raw = body.response.output_text or ""
        if self.config.assert_no_reasoning:
            _assert_no_reasoning(raw)
        generation = raw.strip()

        base = dict(body.model_dump(), generation=generation)

        if not generation:
            return LongmtEvalVerifyResponse(**base, reward=0.0)

        if not self.config.compute_segale:
            return LongmtEvalVerifyResponse(**base, reward=0.0)

        fut = self._dispatch(body.text, generation, body.target_language)
        if fut is None:
            return LongmtEvalVerifyResponse(**base, reward=0.0, segale_error="actor_unavailable")

        try:
            result: Dict = await fut
        except Exception as exc:
            LOG.exception("SegaleActor.score failed for doc_id=%s", body.doc_id)
            return LongmtEvalVerifyResponse(**base, reward=0.0, segale_error=str(exc))

        if result.get("error"):
            return LongmtEvalVerifyResponse(**base, reward=0.0, segale_error=result["error"])

        comet_qe = result["comet_qe"]
        # comet_qe is the mean COMETKiwi score over all valid (non-sentinel)
        # aligned spans. It is in roughly [0, 1] and serves directly as the RL
        # reward: higher = better translation quality.
        return LongmtEvalVerifyResponse(
            **base,
            reward=float(comet_qe),
            comet_qe=comet_qe,
            lang_fidelity=result.get("lang_fidelity"),
            total_seg=result.get("total_seg", 0),
            misaligned_seg=result.get("misaligned_seg", 0),
            spans=result.get("spans"),
        )

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        by_lang: Dict[str, List] = defaultdict(list)
        for task_rollouts in tasks:
            for rollout in task_rollouts:
                if rollout.get("generation"):
                    by_lang[rollout.get("target_language", "")].append(rollout)

        metrics: Dict = {}
        all_comet: List[float] = []

        for lang, rows in sorted(by_lang.items()):
            comet_vals = [r["comet_qe"] for r in rows if r.get("comet_qe") is not None]
            fidelity_vals = [r["lang_fidelity"] for r in rows if r.get("lang_fidelity") is not None]
            total_seg = sum(r.get("total_seg", 0) for r in rows)
            misaligned = sum(r.get("misaligned_seg", 0) for r in rows)

            lang_metrics = {
                "comet_qe": sum(comet_vals) / len(comet_vals) if comet_vals else None,
                "lang_fidelity": sum(fidelity_vals) / len(fidelity_vals) if fidelity_vals else None,
                "total_seg": total_seg,
                "misaligned_seg": misaligned,
                "misaligned_rate": misaligned / total_seg if total_seg else None,
                "n_docs": len(rows),
            }
            metrics[lang] = lang_metrics
            if comet_vals:
                all_comet.extend(comet_vals)

        if all_comet:
            metrics["overall_comet_qe"] = sum(all_comet) / len(all_comet)

        return metrics

    def get_key_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            lang: v["comet_qe"] for lang, v in metrics.items() if isinstance(v, dict) and v.get("comet_qe") is not None
        }


if __name__ == "__main__":
    LongmtEvalServer.run_webserver()
